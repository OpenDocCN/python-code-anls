# `.\models\whisper\modeling_whisper.py`

```
# 设置 Python 文件的编码格式为 UTF-8
# 版权声明和许可信息
# 此处版权归 OpenAI 和 HuggingFace Inc. 团队所有，保留所有权利

""" PyTorch Whisper model. """
# 导入必要的库和模块
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义模块和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# 导入 Whisper 配置类
from .configuration_whisper import WhisperConfig
# 导入 Whisper 生成混合类
from .generation_whisper import WhisperGenerationMixin

# 检查是否可用 Flash Attention 2.0
if is_flash_attn_2_available():
    # 如果可用，则导入 Flash Attention 相关函数和模块
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 1

# 用于文档的配置示例
_CONFIG_FOR_DOC = "WhisperConfig"
# 用于文档的检查点示例
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"

# Whisper 预训练模型的存档列表
WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # 更多 Whisper 模型请见 https://huggingface.co/models?filter=whisper
]


# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    """从注意力掩码中获取非填充数据"""
    # 计算批次中的序列长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非填充数据的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找到批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """为位置嵌入返回正弦波"""
    # 检查通道数是否为偶数
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    # 计算用于时间缩放的对数时间尺度增量
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    
    # 计算逆时间尺度，通过 torch.exp 函数对每个通道的对数时间尺度增量进行指数运算
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    
    # 创建一个二维张量，其中每一行代表一个时间步，每列代表一个通道的缩放时间
    # 通过乘以逆时间尺度张量，将时间线性缩放到不同的频率
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    
    # 返回一个张量，包含了缩放时间的正弦和余弦值，沿着通道维度连接
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个和 input_ids 形状相同的全零张量 shifted_input_ids
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将 input_ids 的每一行向右移动一位，将结果复制到 shifted_input_ids 中
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将每一行的第一个位置填充为 decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        # 如果 pad_token_id 为 None，则抛出数值错误
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将 shifted_input_ids 中值为 -100 的位置用 pad_token_id 替换
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        # 如果 mask_length 小于 1，则抛出数值错误
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        # 如果 mask_length 大于 sequence_length，则抛出数值错误
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()
    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 计算应该屏蔽的 span 数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保 num_masked_span 不超过 sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保 num_masked_span 不超过 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批次中每个序列的长度
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 创建用于 SpecAugment 的掩码
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算最大允许的 masked span 数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # 计算当前输入长度下的 masked span 数量
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机的掩码索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选取第一个作为 dummy 索引，用于填充向量，确保所有批次维度相同
        if len(spec_aug_mask_idx) == 0:
            # 只有在 input_length 严格小于 sequence_length 时才可能发生这种情况，
            # 此时最后一个标记必须是填充标记，可以用作虚拟掩码 id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 填充掩码索引数组，确保每个批次的维度相同
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将掩码索引扩展为掩码 spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 将起始索引添加偏移量，使索引现在创建一个 span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    # 确保我们不能使用大于 sequence_length - 1 的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        # 将 spec_aug_mask_idxs 中大于 sequence_length - 1 的索引置为 sequence_length - 1
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    
    # 在 spec_aug_mask 上根据 spec_aug_mask_idxs 的索引位置散布值
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    
    # 返回 spec_aug_mask 结果
    return spec_aug_mask
class WhisperPositionalEmbedding(nn.Embedding):
    # 继承自 nn.Embedding 的类 WhisperPositionalEmbedding，用于位置编码的嵌入
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    # 前向传播函数，根据输入的位置 ids 返回对应的嵌入向量
    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            # 如果未提供 position_ids，则根据输入的 input_ids 和历史键值的长度返回相应的嵌入向量
            return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]
        else:
            # 如果提供了 position_ids，则直接返回对应位置的嵌入向量
            return self.weight[position_ids]


class WhisperAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力模块"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[WhisperConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性变换层，用于计算查询、键、值以及输出的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 从 transformers.models.bart.modeling_bart.BartAttention._shape 复制而来，用于调整张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 从 transformers.models.bart.modeling_bart.BartAttention.forward 复制而来，前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 实现注意力机制的前向传播，包括查询、键、值的投影以及输出的投影
        # 注意力掩码、层头掩码等参数用于控制注意力的行为
        # 返回值包括输出张量以及可选的注意力权重
        pass


# 从 Bart->Whisper 改名，并继承自 WhisperAttention，用于实现 Flash Attention 机制
class WhisperFlashAttention2(WhisperAttention):
    """
    Whisper flash attention 模块。此模块继承自 `WhisperAttention`，保持模块权重不变。
    在前向传播中正确调用 Flash Attention 的公共 API，并处理可能包含的填充令牌。
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    # 初始化函数，不同之处在于需调整成正确调用 Flash Attention 的接口及处理填充令牌的逻辑
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[WhisperConfig] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias, is_causal, config)
        # 此处可能需要进行 Flash Attention 特定的初始化
        pass
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, which is default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 设置一个属性来处理 Flash Attention 版本间的差异，当 flash_attn<2.1 时，生成的是左上对齐的因果掩码，而我们需要的是右下对齐的掩码，这在 flash_attn>=2.1 中是默认的行为。

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新塑形张量，将其形状变为 (bsz, seq_len, num_heads, head_dim)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 从隐藏状态开始向前传播，支持可选的键值状态、过去的键值、注意力掩码和层头掩码等参数

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # Flash Attention 前向传播函数，接受查询状态、键状态、值状态、注意力掩码、查询长度以及可选的 dropout 和 softmax_scale 参数
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Determine if causal masking should be applied based on configuration and query length
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Conditionally adjust causal based on a specific condition for Flash Attention in RoCm
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input states using a helper method _upad_input
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Retrieve sequence lengths from the computed values
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Compute attention scores with variable length support using flash_attn_varlen_func
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            # Pad the attention scores back to the original sequence length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Compute attention scores without considering any padding tokens
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the final attention scores
        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 定义一个私有方法，用于处理输入数据，对查询、键和值进行调整，以及相关的注意力掩码
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度信息
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        # 重新形状化键层和值层，按照未填充数据的索引进行索引
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度选择不同的处理分支
        if query_length == kv_seq_len:
            # 如果查询长度等于键值序列长度，则对查询层进行索引操作
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，则处理为标量情况
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个memcpy操作，性能较差。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据查询长度和注意力掩码进行输入数据的解压缩操作
            # 注意，这里的 -query_length: 切片假设左填充。
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回调整后的查询层、键层、值层、查询索引、当前序列长度信息和批次最大序列长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class WhisperSdpaAttention(WhisperAttention):
    # 从 transformers.models.bart.modeling_bart.BartSdpaAttention.forward 复制而来，将 BART->whisper, Bart->Whisper
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 注意力机制的实现，用于计算注意力分数并加权隐藏状态
        pass

# WHISPER_ATTENTION_CLASSES 定义了不同实现的注意力类别映射
WHISPER_ATTENTION_CLASSES = {
    "eager": WhisperAttention,
    "flash_attention_2": WhisperFlashAttention2,
    "sdpa": WhisperSdpaAttention,  # 使用 WhisperSdpaAttention 作为一种注意力实现
}

# 从 transformers.models.mbart.modeling_mbart.MBartEncoderLayer 复制而来，将 MBart->Whisper, MBART->WHISPER
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self_attn 是自注意力层，根据配置选择不同的注意力实现类别
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数的选择
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        # 编码器层的前向传播，包括自注意力、前馈神经网络和层归一化
        pass
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存输入的原始状态，用于残差连接
        residual = hidden_states
        # 对输入的 hidden_states 进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self-attention 模块处理 normalized 后的 hidden_states
        # 返回处理后的 hidden_states、attention 权重和额外的信息
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对处理后的 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到处理后的 hidden_states 上
        hidden_states = residual + hidden_states

        # 再次保存当前的 hidden_states 用于残差连接
        residual = hidden_states
        # 对当前的 hidden_states 进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理第一个全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用第二个全连接层处理 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 对处理后的 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到处理后的 hidden_states 上
        hidden_states = residual + hidden_states

        # 如果 hidden_states 的数据类型是 torch.float16 并且包含无穷大或 NaN 的元素
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对 hidden_states 进行截断处理，避免溢出
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组，包含处理后的 hidden_states
        outputs = (hidden_states,)

        # 如果需要输出 attentions，将 attentions 加入输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出元组
        return outputs
# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制而来，MBart->Whisper, MBART->WHISPER
class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置嵌入维度为配置中的d_model

        # 初始化自注意力层，根据配置选择的注意力机制类别进行设置
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout  # 设置dropout概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数根据配置选择
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout概率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化自注意力层的LayerNorm
        # 初始化编码器注意力层，根据配置选择的注意力机制类别进行设置
        self.encoder_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化编码器注意力层的LayerNorm
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终输出的LayerNorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        计算卷积层的输出长度

        将输入长度减去1，然后整除2，并加上1，计算卷积层的输出长度。
        """
        # 将输入长度减去1，然后整除2，并加上1，得到卷积层的输出长度
        input_lengths = (input_lengths - 1) // 2 + 1

        # 返回计算得到的卷积层输出长度
        return input_lengths
# 定义文档字符串，描述了 `WhisperEncoder` 类的继承和用法说明
WHISPER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 空白的输入文档字符串，待后续补充输入参数的描述
WHISPER_INPUTS_DOCSTRING = r"""
"""

# 定义了用于WhisperEncoder类的输入参数的文档字符串，详细描述了每个参数的类型和作用
WHISPER_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    # `WhisperEncoderLayer` 的编码器层实现。
    
    Args:
        config: WhisperConfig  # 输入参数为 WhisperConfig 类型的配置对象

    def __init__(self, config: WhisperConfig):
        super().__init__(config)  # 调用父类的初始化方法，传入配置对象
        self.dropout = config.dropout  # 设置 dropout 概率
        self.layerdrop = config.encoder_layerdrop  # 设置层丢弃率

        embed_dim = config.d_model  # 获取嵌入维度
        self.num_mel_bins = config.num_mel_bins  # 获取梅尔频谱的数量
        self.padding_idx = config.pad_token_id  # 获取填充标记的索引
        self.max_source_positions = config.max_source_positions  # 获取最大源序列位置
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 计算嵌入缩放因子，根据配置选择是否开启

        # 初始化两个一维卷积层，用于特征提取
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # 初始化位置嵌入层，并设置为不需要梯度计算
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        # 使用 WhisperEncoderLayer 构建编码器层的列表，根据配置中的编码器层数量
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)  # 初始化层归一化层

        self.gradient_checkpointing = False  # 是否启用梯度检查点

        # 初始化权重并应用最终处理
        self.post_init()

    def _freeze_parameters(self):
        # 冻结所有参数，使其不需要梯度计算
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False  # 设置不需要梯度计算标志为 False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1  # 返回输入嵌入层 conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value  # 设置输入嵌入层 conv1 的值为给定的 value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
# 定义一个名为 WhisperDecoder 的类，继承自 WhisperPreTrainedModel 类
class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer 解码器，由 *config.decoder_layers* 层组成。每层是一个 [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig 对象，包含模型的配置信息
    """

    # 主要输入名称为 "input_ids"
    main_input_name = "input_ids"

    # 初始化方法，接收一个 WhisperConfig 类型的参数 config
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层级丢弃概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_target_positions
        # 设置最大源位置
        self.max_source_positions = config.max_source_positions
        # 如果开启了 scale_embedding，则使用 sqrt(config.d_model) 作为嵌入尺度，否则为 1.0
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 嵌入 tokens，使用 nn.Embedding 创建一个嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        # 嵌入位置编码，使用 WhisperPositionalEmbedding 创建一个位置编码嵌入层
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        # 创建解码器层列表，包含 config.decoder_layers 个 WhisperDecoderLayer 层
        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        
        # 根据 config._attn_implementation 决定是否使用 Flash Attention 2.0 注意力机制
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 根据 config._attn_implementation 决定是否使用 SDPA 注意力机制
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 层归一化，使用 nn.LayerNorm 进行归一化处理
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 梯度检查点，默认为 False，是否使用梯度检查点
        self.gradient_checkpointing = False
        
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层对象，返回 self.embed_tokens
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层对象为 value
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数定义，接收多个参数用于解码器的输入和控制
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features


注释：

# 根据输入特征的尺寸在时间轴和/或特征轴上屏蔽提取的特征，根据 SpecAugment 方法
def forward(
    self,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    decoder_head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
    decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
# 添加文档字符串到类定义，描述了 WhisperForConditionalGeneration 类的用途和功能
@add_start_docstrings(
    "The Whisper Model with a language modeling head. Can be used for automatic speech recognition.",
    WHISPER_START_DOCSTRING,
)
class WhisperForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    # 设置基础模型前缀，用于指定模型中与权重共享相关的键
    base_model_prefix = "model"
    # 指定应当共享权重的键名列表
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        # 调用父类的初始化方法，传入 WhisperConfig 对象
        super().__init__(config)
        # 创建 WhisperModel 对象，并将其保存在实例变量 self.model 中
        self.model = WhisperModel(config)
        # 创建线性层，用于输出模型的预测结果，不带偏置项
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 调用额外的初始化方法，用于权重初始化和最终处理
        self.post_init()

    def get_encoder(self):
        # 返回模型中的编码器部分，通过调用 self.model 的 get_encoder 方法实现
        return self.model.get_encoder()

    def get_decoder(self):
        # 返回模型中的解码器部分，通过调用 self.model 的 get_decoder 方法实现
        return self.model.get_decoder()

    def get_output_embeddings(self):
        # 返回输出嵌入层，即预测输出的线性层 self.proj_out
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入层，更新 self.proj_out 的值为 new_embeddings
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        # 返回模型中的输入嵌入层，通过调用 self.model 的 get_input_embeddings 方法实现
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        调用此方法将禁用 Whisper 编码器的梯度计算，使其在训练过程中不会更新参数。
        """
        self.model.encoder._freeze_parameters()

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        覆盖父类中的 forward 方法，实现 Whisper 模型的前向传播。

        Args:
            input_features (Optional[torch.FloatTensor], optional): 输入特征张量。默认为 None。
            attention_mask (Optional[torch.LongTensor], optional): 注意力掩码张量。默认为 None。
            decoder_input_ids (Optional[torch.LongTensor], optional): 解码器输入 ID 张量。默认为 None。
            decoder_attention_mask (Optional[torch.LongTensor], optional): 解码器注意力掩码张量。默认为 None。
            head_mask (Optional[torch.Tensor], optional): 头部掩码张量。默认为 None。
            decoder_head_mask (Optional[torch.Tensor], optional): 解码器头部掩码张量。默认为 None。
            cross_attn_head_mask (Optional[torch.Tensor], optional): 交叉注意力头部掩码张量。默认为 None。
            encoder_outputs (Optional[Tuple[Tuple[torch.FloatTensor]]], optional): 编码器输出元组。默认为 None。
            past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]], optional): 过去的键值元组。默认为 None。
            decoder_inputs_embeds (Optional[Tuple[torch.FloatTensor]], optional): 解码器输入嵌入张量元组。默认为 None。
            decoder_position_ids (Optional[Tuple[torch.LongTensor]], optional): 解码器位置 ID 张量元组。默认为 None。
            labels (Optional[torch.LongTensor], optional): 标签张量。默认为 None。
            use_cache (Optional[bool], optional): 是否使用缓存。默认为 None。
            output_attentions (Optional[bool], optional): 是否输出注意力。默认为 None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态。默认为 None。
            return_dict (Optional[bool], optional): 是否返回字典。默认为 None。

        Returns:
            Seq2SeqLMOutput: 序列到序列的语言模型输出。
        """
        # 实际的前向传播逻辑将在此处实现

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        """
        准备生成过程中的输入，以便在生成文本时使用。

        Args:
            decoder_input_ids: 解码器输入 ID。
            past_key_values: 过去的键值对。
            use_cache: 是否使用缓存。
            encoder_outputs: 编码器输出。
            attention_mask: 注意力掩码。
            decoder_attention_mask: 解码器注意力掩码。
            **kwargs: 其他关键字参数。

        Returns:
            dict: 包含生成过程输入的字典。
        """
        # 实现生成输入准备的逻辑
        ):
            # 初始化变量 decoder_position_ids 为 None
            decoder_position_ids = None
            # 如果存在 decoder_attention_mask，计算每个位置累积和后减一，并确保不小于零
            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

            # 如果存在 past_key_values，则获取其长度
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 某些生成方法可能只传递最后一个输入 ID
                if decoder_input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认行为：保留最后一个 ID
                    remove_prefix_length = decoder_input_ids.shape[1] - 1

                # 仅保留 decoder_input_ids 中的后缀部分
                decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

                # 如果存在 decoder_position_ids 并且其长度大于 decoder_input_ids 的长度，则也截断之
                if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                    decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

            # 返回重构后的信息字典
            return {
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "use_cache": use_cache,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_position_ids": decoder_position_ids,
            }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序的 past_key_values
        reordered_past = ()
        # 遍历 past_key_values 中的每个层的过去状态
        for layer_past in past_key_values:
            # 使用 beam_idx 对每个 past_state 进行重新排序，并将结果添加到 reordered_past 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的 past_key_values
        return reordered_past
class WhisperDecoderWrapper(WhisperPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        # 设置当前模型不是编码器-解码器结构
        config.is_encoder_decoder = False
        # 初始化一个WhisperDecoder对象作为解码器
        self.decoder = WhisperDecoder(config)

    def get_input_embeddings(self):
        # 返回当前模型的解码器的嵌入层
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置当前模型的解码器的嵌入层
        self.decoder.embed_tokens = value

    def forward(self, *args, **kwargs):
        # 前向传播，调用当前模型的解码器进行处理
        return self.decoder(*args, **kwargs)


@add_start_docstrings(
    """
    Whisper decoder with with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    WHISPER_START_DOCSTRING,
)
class WhisperForCausalLM(WhisperPreTrainedModel):
    _tied_weights_keys = ["proj_out.weight"]
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        # 设置当前模型不是编码器-解码器结构
        config.is_encoder_decoder = False
        # 初始化一个WhisperDecoderWrapper对象作为当前模型的主模型
        self.model = WhisperDecoderWrapper(config)

        # 初始化一个线性层，作为模型的输出投影层，将隐藏状态映射到词汇表大小的向量空间
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回当前模型的输出投影层
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        # 设置当前模型的输出投影层
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        # 返回当前模型主模型的解码器的嵌入层
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置当前模型主模型的解码器的嵌入层
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        # 设置当前模型主模型的解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回当前模型主模型的解码器
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播函数，调用当前模型的主模型进行处理
        pass  # 实际操作在self.model.forward中定义

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        # 为生成过程准备输入数据，调用当前模型的主模型方法处理
        pass  # 实际操作在self.model.prepare_inputs_for_generation中定义
        ):
            # 如果过去的键值不为 None，则获取过去键值的第一个元素的第三维度长度作为过去长度
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 某些生成方法可能只传递最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    # 如果输入的 ID 数量大于过去长度，则移除前缀长度为过去长度
                    remove_prefix_length = past_length
                else:
                    # 否则，默认行为：只保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                # 更新输入的 ID，移除前缀部分
                input_ids = input_ids[:, remove_prefix_length:]

            # 返回一个包含各种输出和参数的字典
            return {
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "input_ids": input_ids,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            reordered_past = ()
            # 遍历过去键值中的每一层，并重新排序以匹配 beam_idx 的顺序
            for layer_past in past_key_values:
                reordered_past += (
                    # 对于每个过去状态，根据 beam_idx 在设备上选择相应的索引
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            # 返回重新排序后的过去键值
            return reordered_past
@add_start_docstrings(
    """
    Whisper Encoder Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    """,
    WHISPER_ENCODER_INPUTS_DOCSTRING,
)
class WhisperForAudioClassification(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)  # 初始化Whisper编码器，使用给定的配置
        num_layers = config.num_hidden_layers + 1  # 计算层数，包括transformer层和输入嵌入层
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # 如果使用加权层求和，初始化权重参数
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)  # 初始化线性投影层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)  # 初始化分类器线性层

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理步骤

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training. Only the projection layers and classification head will be updated.
        """
        self.encoder._freeze_parameters()  # 冻结Whisper编码器的参数，使其在训练过程中不更新梯度，只更新投影层和分类头部

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()  # 返回Whisper编码器的输入嵌入层模块

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)  # 设置Whisper编码器的输入嵌入层模块

    @add_start_docstrings_to_model_forward(WHISPER_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```