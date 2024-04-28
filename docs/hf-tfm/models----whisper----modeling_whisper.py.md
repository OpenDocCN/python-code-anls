# `.\transformers\models\whisper\modeling_whisper.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有（c）2022 OpenAI作者和HuggingFace Inc.团队。保留所有权利。
#
# 根据Apache许可2.0版（“许可证”）获得许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不提供任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch Whisper model."""
# 导入所需模块
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
from .configuration_whisper import WhisperConfig
from .generation_whisper import WhisperGenerationMixin

# 检查是否可用flash_attn库的相关函数
if is_flash_attn_2_available():
    # 导入相关函数和模块
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 1

# 用于文档的配置信息
_CONFIG_FOR_DOC = "WhisperConfig"
# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"

# Whisper模型的预训练模型存档列表
WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # 查看所有Whisper模型 https://huggingface.co/models?filter=whisper
]


# 从transformers.models.llama.modeling_llama._get_unpad_data复制
def _get_unpad_data(attention_mask):
    # 计算批次中每个序列的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取非零位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找到批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 对序列长度进行累积求和并填充
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 生成位置嵌入的正弦波
def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    # 如果通道数不能被2整除，则引发错误
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    # 计算对数时间刻度增量，以用于注意力机制中的位置编码
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    # 计算时间刻度的倒数，用于位置编码
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    # 生成长度为 length 的时间序列，并按时间刻度进行缩放
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    # 返回经过正弦和余弦函数处理的时间序列作为位置编码
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
# 从transformers.models.bart.modeling_bart.shift_tokens_right中复制代码
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的id向右移动一个标记。
    """
    # 创建一个与input_ids相同形状的全零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将input_ids中每一行的数据向左移动一位，并将结果赋值给shifted_input_ids，同时保留最后一个位置的数据
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将decoder_start_token_id的值赋值给shifted_input_ids的每一行的第一个位置
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        # 如果pad_token_id未定义，则抛出错误
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 使用pad_token_id替换标签中可能存在的-100值
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices中复制代码
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码范围。用于实现[SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779)。注意，此方法未经过优化，应在训练期间的预处理过程中在CPU上运行，而不是在TPU上运行。

    Args:
        shape: 计算掩码的形状。这应该是大小为2的元组，其中
               第一个元素是批量大小，第二个元素是要跨度的轴的长度。
        mask_prob:  将被掩盖的整个轴的百分比（在0和1之间）。将通过`mask_prob*shape[1]/mask_length`计算长度为`mask_length`的独立生成掩码范围的数量。注意，由于重叠，`mask_prob`是一个上限，实际百分比将更小。
        mask_length: 掩码的大小
        min_masks: 掩码的最小数量
        attention_mask: 一个（右填充的）注意力掩码，可以独立缩短每个批量维度的特征轴。
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        # 如果mask_length小于1，则抛出错误
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        # 如果mask_length大于sequence_length，则抛出错误
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()
    # 定义一个函数，根据输入的长度计算应该被遮罩的span的数量
    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 计算应该被遮罩的span的数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保被遮罩的span数量不超过序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保被遮罩的span数量不超过 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批处理中的被遮罩span的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 创建用于 SpecAugment 的遮罩
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算可以被遮罩的最大span数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    # 如果最大被遮罩span的数量为0，则直接返回空的遮罩
    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # 计算这个输入中被遮罩span的数量
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机的索引以进行遮罩
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个被抽样的索引作为填充向量的虚拟索引，以确保由于概率舍入而使所有批次具有相同的维度
        # 选择第一个样本只是为了对这些向量进行两次填充
        if len(spec_aug_mask_idx) == 0:
            # 只有在 `input_length` 严格小于 `sequence_length` 时才会发生这种情况，
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟的遮罩id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟的遮罩id连接到已有的遮罩id数组中
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 扩展被遮罩的索引以获得遮罩的span
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 为起始索引添加偏移量，使索引现在创建一个span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    # 确保我们不能有超过序列长度的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        # 将大于序列长度减一的索引值设为序列长度减一
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    
    # 将索引散布到掩码上
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    
    # 返回特征增强后的掩码
    return spec_aug_mask
class WhisperPositionalEmbedding(nn.Embedding):
    # WhisperPositionalEmbedding类继承自nn.Embedding类
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    # 前向传播函数，用于计算位置嵌入
    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            # 如果位置ID为空，则返回输入ID对应的位置嵌入
            return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]
        else:
            # 如果位置ID不为空，则返回指定位置ID对应的位置嵌入
            return self.weight[position_ids]


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，用于初始化注意力模块的参数
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
            # 如果头维度乘以头数不等于嵌入维度，则抛出错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性变换函数
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 从Bart模型复制过来的形状函数
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑输入张量的形状以适应多头注意力的计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 从Bart模型复制过来的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,


# 从Bart模型复制过来的WhisperFlashAttention2类，用于实现Whisper模型的闪现注意力机制
class WhisperFlashAttention2(WhisperAttention):
    """
    Whisper flash attention module. This module inherits from `WhisperAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从Llama模型复制过来的初始化函数
    def __init__(self):
```  
    # 初始化方法，接受不定数量的位置参数和关键字参数，调用父类的初始化方法
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        # 注意事项：应该在 Flash Attention for RoCm 升级到 2.1 之后删除这段代码
        # flash_attn<2.1 生成左上角对齐的因果遮罩，而这里需要的是右下角对齐，在 flash_attn>=2.1 版本中已经默认为右下角对齐。这个属性用于处理这种差异。参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 请注意，对于 flash_attn<2.1，使用 q_seqlen != k_seqlen（除了 q_seqlen == 1 的情况外）会产生错误的遮罩（左上角）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    # 重新调整张量形状的方法，接受一个张量、序列长度和批次大小作为参数
    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
    
    # 前向传播方法，接受隐藏状态、键值状态、过去的键值、注意力遮罩、层级头遮罩、输出注意力的布尔值作为参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward 复制过来的
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入的隐藏状态包含至少一个填充标记，则首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                传递给 Flash Attention API 的查询状态输入
            key_states (`torch.Tensor`):
                传递给 Flash Attention API 的键状态输入
            value_states (`torch.Tensor`):
                传递给 Flash Attention API 的值状态输入
            attention_mask (`torch.Tensor`):
                填充蒙版 - 对应于尺寸为 `(batch_size, seq_len)` 的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 前的 QK^T 缩放。默认为 1 / sqrt(head_dim)
        """
        如果不是使用 top-left mask 的 Flash Attention：
            causal = self.is_causal
        否则：
            # 一旦 Flash Attention 用于 RoCm 版本升级到 2.1，可以删除 `query_length != 1` 检查。有关详细信息，请参阅 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        如果 attention_mask 不为 None：
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

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

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        否则：
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        返回 attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 中复制
    # 这个函数接受 query_layer、key_layer、value_layer、attention_mask 和 query_length，并根据 attention_mask 对输入进行拆分和重组，目的是为了减少在注意力机制中的不必要计算
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 从 attention_mask 中获取未填充的数据索引、累积序列长度、以及批量中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取 key_layer 的维度参数，包括批量大小、序列长度、头的数量和每个头的维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    
        # 对 key_layer 进行 reshape，然后根据 indices_k 对第一个轴进行索引，获取未填充的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 对 value_layer 进行同样的处理
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 如果 query_length 和 kv_seq_len 相等，说明无需拆分
        if query_length == kv_seq_len:
            # 对 query_layer 进行 reshape，并根据 indices_k 对第一个轴进行索引
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            # 如果序列长度一致，则 query 和 key 的累积序列长度、最大序列长度是一样的
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果 query_length 为 1，则说明每个批次只有一个查询
        elif query_length == 1:
            # 最大序列长度是 1
            max_seqlen_in_batch_q = 1
            # cu_seqlens_q 用于表示每个批次的累积序列长度，这里通过 torch.arange 生成
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这个地方有一个可能不太好的 memcpy
            # indices_q 表示未填充的序列索引，取除了最后一个元素的部分
            indices_q = cu_seqlens_q[:-1]
            # 对 query_layer 进行压缩，去掉长度为 1 的轴
            query_layer = query_layer.squeeze(1)
        # 如果 query_length 与 kv_seq_len 不相等且不为 1，则需要进行拆分
        else:
            # 对 attention_mask 只保留 query_length 的部分，假设左侧填充
            attention_mask = attention_mask[:, -query_length:]
            # 通过 unpad_input 对 query_layer 进行拆分，返回对应的各个部分
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
    
        # 最后返回修改后的 query_layer、key_layer、value_layer 以及相关的索引和序列长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class WhisperSdpaAttention(WhisperAttention):
    # 从transformers.models.bart.modeling_bart.BartSdpaAttention.forward复制而来，将BART->whisper, Bart->Whisper
    # 定义了前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        key_value_states: Optional[torch.Tensor] = None,  # 键值状态张量，默认为空
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值对，默认为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为空
        layer_head_mask: Optional[torch.Tensor] = None,  # 层头掩码，默认为空
        output_attentions: bool = False,  # 是否输出注意力，默认为False

WHISPER_ATTENTION_CLASSES = {
    "eager": WhisperAttention,  # 指定“eager”关键字对应的WhisperAttention类
    "flash_attention_2": WhisperFlashAttention2,  # 指定“flash_attention_2”关键字对应的WhisperFlashAttention2类
    "sdpa": WhisperSdpaAttention,  # 指定“sdpa”关键字对应的WhisperSdpaAttention类
}


# 从transformers.models.mbart.modeling_mbart.MBartEncoderLayer中复制而来，将MBart->Whisper, MBART->WHISPER
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 将嵌入维度设置为WhisperConfig中的d_model

        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](  # 使用config._attn_implementation配置创建自注意力机制
            embed_dim=self.embed_dim,  # 嵌入维度
            num_heads=config.encoder_attention_heads,  # 编码器注意力头的数量
            dropout=config.attention_dropout,  # 注意力机制的丢弃率
            config=config,  # WhisperConfig实例
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 创建嵌入维度的层归一化层
        self.dropout = config.dropout  # 丢弃率
        self.activation_fn = ACT2FN[config.activation_function]  # 根据激活函数配置选择对应的激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的丢弃率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 创建一个线性层, 输入维度为嵌入维度，输出维度为config.encoder_ffn_dim
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 创建一个线性层, 输入维度为config.encoder_ffn_dim，输出维度为嵌入维度
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 创建嵌入维度的最终层归一化层

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        attention_mask: torch.Tensor,  # 注意力掩码张量
        layer_head_mask: torch.Tensor,  # 层头掩码张量
        output_attentions: bool = False,  # 是否输出注意力，默认为False
    def forward(
        hidden_states: torch.Tensor,  # 输入层的隐藏状态张量，形状为`(batch, seq_len, embed_dim)`
        attention_mask: torch.FloatTensor,  # 注意力掩码张量，大小为`(batch, 1, tgt_len, src_len)`，其中填充元素用非常大的负值表示
        layer_head_mask: torch.FloatTensor,  # 给定层中注意力头的掩码张量，大小为`(encoder_attention_heads,)`
        output_attentions: bool,  # 是否返回所有注意力层的注意力张量。详见返回的张量中的`attentions`。
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
        residual = hidden_states  # 保存隐藏状态的副本，用于残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 应用自注意力层归一化
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )  # 使用自注意力机制处理隐藏状态，计算注意力权重
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 应用丢弃操作
        hidden_states = residual + hidden_states  # 执行残差连接

        residual = hidden_states  # 保存残差连接后的结果
        hidden_states = self.final_layer_norm(hidden_states)  # 应用最终层归一化
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 使用激活函数处理全连接层1
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 应用激活函数的丢弃操作
        hidden_states = self.fc2(hidden_states)  # 应用全连接层2
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 应用丢弃操作
        hidden_states = residual + hidden_states  # 执行残差连接

        # 如果隐藏状态的数据类型为float16且包含无限或NaN值，则对其进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)  # 将处理后的隐藏状态作为输出的一部分

        if output_attentions:  # 如果需要输出注意力张量
            outputs += (attn_weights,)  # 将注意力权重也加入到输出中

        return outputs  # 返回处理后的输出
# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制而来，将MBart改为Whisper，MBART改为WHISPER
class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化自注意力机制
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 初始化Dropout层
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活函数的Dropout层
        self.activation_dropout = config.activation_dropout

        # LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化编码器注意力机制
        self.encoder_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 编码器注意力机制的LayerNorm层
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 第一个线性层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 第二个线性层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 最终的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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
# WhisperPreTrainedModel的基类
class WhisperPreTrainedModel(PreTrainedModel):
    # Whisper模型的配置类
    config_class = WhisperConfig
    # 基础模型的前缀
    base_model_prefix = "model"
    # 主输入名称
    main_input_name = "input_features"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    # 是否支持闪光注意力
    _supports_flash_attn_2 = True
    # 是否支持SDPA
    _supports_sdpa = True

    # 初始化权重
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperEncoder):
            with torch.no_grad():
                # 复制嵌入位置的权重
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))
    # 计算卷积层的输出长度
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        # 对输入长度进行计算，满足卷积操作后的输出长度计算公式
        input_lengths = (input_lengths - 1) // 2 + 1
    
        # 返回计算得到的输出长度
        return input_lengths
# 定义WHISPER_START_DOCSTRING变量，包含模型的基本说明，继承自PreTrainedModel类，提供了一般性方法的详细介绍
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

# 定义WHISPER_INPUTS_DOCSTRING变量，暂时为空，用于后续填写输入说明
WHISPER_INPUTS_DOCSTRING = r"""
"""

# 定义WHISPER_ENCODER_INPUTS_DOCSTRING变量，包含WhisperEncoder类的输入参数说明
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

# 定义WhisperEncoder类，继承自WhisperPreTrainedModel类
class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
        # 初始化 WhiperEncoderLayer 类
        def __init__(self, config: WhisperConfig):
            super().__init__(config)
            self.dropout = config.dropout
            self.layerdrop = config.encoder_layerdrop

            embed_dim = config.d_model
            self.num_mel_bins = config.num_mel_bins
            self.padding_idx = config.pad_token_id
            self.max_source_positions = config.max_source_positions
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            # 创建 1D 卷积层，输入维度为语音特征维度，输出维度为嵌入维度
            self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
            # 创建 1D 卷积层，输入和输出维度均为嵌入维度
            self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

            # 创建位置嵌入层
            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
            # 设置位置嵌入层的参数不可训练
            self.embed_positions.requires_grad_(False)

            # 创建多层的 Transformer 编码器层
            self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
            # 创建 LayerNorm 层，对输入进行归一化
            self.layer_norm = nn.LayerNorm(config.d_model)

            self.gradient_checkpointing = False
            # 初始化权重并进行最终处理
            self.post_init()

        # 冻结模型的参数，使其不可训练
        def _freeze_parameters(self):
            for param in self.parameters():
                param.requires_grad = False
            self._requires_grad = False

        # 获取输入嵌入层
        def get_input_embeddings(self) -> nn.Module:
            return self.conv1

        # 设置输入嵌入层
        def set_input_embeddings(self, value: nn.Module):
            self.conv1 = value

        # 前向传播
        def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # ... 省略部分参数
class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
# 添加输入文档字符串
@add_start_docstrings(
    "The bare Whisper Model outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
class WhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        # 如果配置中设置apply_spec_augment为False，则不进行特征遮蔽，直接返回输入特征
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        # 获取输入特征的维度信息
        batch_size, hidden_size, sequence_length = input_features.size()

        # 如果mask_time_prob大于0且处于训练模式
        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            # 生成需要遮蔽的时间轴坐标并进行遮蔽
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            # 将生成的遮蔽坐标转换为torch.tensor，确保与输入特征在同一设备上
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            # 对时间坐标进行遮蔽
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        # 如果mask_feature_prob大于0且处于训练模式
        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            # 生成需要遮蔽的特征轴坐标并进行遮蔽
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            # 将生成的遮蔽坐标转换为torch.tensor，确保与输入特征在同一设备上
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            # 对特征轴进行遮蔽
            input_features[mask_feature_indices] = 0

        # 返回应用了遮蔽的输入特征
        return input_features

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播方法
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
# 将注释添加到模型类上，并描述其功能
@add_start_docstrings(
    "The Whisper Model with a language modeling head. Can be used for automatic speech recognition.",
    WHISPER_START_DOCSTRING,
)
# 定义 WhisperForConditionalGeneration 类，继承自 WhisperGenerationMixin 和 WhisperPreTrainedModel
class WhisperForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    # 定义模型的前缀
    base_model_prefix = "model"
    # 定义共享权重的键
    _tied_weights_keys = ["proj_out.weight"]

    # 初始化函数，接收 WhisperConfig 类型的配置
    def __init__(self, config: WhisperConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 WhisperModel 对象
        self.model = WhisperModel(config)
        # 创建线性映射层，用于输出
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 返回解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 返回输出的嵌入
    def get_output_embeddings(self):
        return self.proj_out

    # 设置输出的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    # 返回输入的嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    # 冻结编码器的参数，禁用梯度计算以使其参数在训练期间不会更新
    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # 定义输入特征
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
    # 准备生成的输入
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
            # 初始化解码器位置信息为 None
            decoder_position_ids = None
            if decoder_attention_mask is not None:
                # 如果存在注意力掩码，计算解码器位置信息
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

            if past_key_values is not None:
                # 获取过去键值的长度
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法可能只传递最后一个输入ID
                if decoder_input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认保留旧的行为：仅保留最后一个ID
                    remove_prefix_length = decoder_input_ids.shape[1] - 1

                # 更新解码器输入ID，去掉前缀部分
                decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

                if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                    # 如果存在解码器位置信息并且长度大于解码器输入ID长度，则进行相应处理
                    decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

            # 返回包含关键信息的字典
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
            # 重新调整缓存中的值
            reordered_past = ()
            for layer_past in past_key_values:
                # 为每个层次的过去状态重新排序
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            return reordered_past
class WhisperDecoderWrapper(WhisperPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False  # 将配置中的 is_encoder_decoder 属性设为 False
        self.decoder = WhisperDecoder(config)  # 使用配置创建一个 WhisperDecoder 对象

    def get_input_embeddings(self):
        return self.decoder.embed_tokens  # 返回解码器对象的嵌入层

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value  # 设置解码器对象的嵌入层

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)  # 调用解码器对象的前向方法


@add_start_docstrings(
    """
    Whisper decoder with with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    WHISPER_START_DOCSTRING,
)
class WhisperForCausalLM(WhisperPreTrainedModel):
    _tied_weights_keys = ["proj_out.weight"]  # 权重绑定的键列表
    main_input_name = "input_ids"  # 主输入名称为 input_ids

    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False  # 将配置中的 is_encoder_decoder 属性设为 False
        self.model = WhisperDecoderWrapper(config)  # 使用配置创建一个 WhisperDecoderWrapper 对象

        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 创建线性层对象并初始化权重（不使用偏置）

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和应用最终处理的操作

    def get_output_embeddings(self):
        return self.proj_out  # 返回输出嵌入层

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings  # 设置新的输出嵌入层

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()  # 返回解码器对象的嵌入层

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)  # 设置解码器对象的嵌入层

    def set_decoder(self, decoder):
        self.model.decoder = decoder  # 设置解码器

    def get_decoder(self):
        return self.model.decoder  # 返回解码器

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 以上是前向传播方法的参数

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None
        **kwargs,  # 准备用于生成的输入参数
    # 如果已有的 past_key_values 不为空
    if past_key_values is not None:
        # 获取上次生成的序列长度
        past_length = past_key_values[0][0].shape[2]
        
        # 如果当前输入的长度大于之前生成的长度
        if input_ids.shape[1] > past_length:
            # 则保留最后一个输入 ID
            remove_prefix_length = past_length
        else:
            # 否则保留当前输入的所有 ID
            remove_prefix_length = input_ids.shape[1] - 1
        
        # 截取输入序列，移除之前已生成的部分
        input_ids = input_ids[:, remove_prefix_length:]
    
    # 返回一个字典，包含编码器输出、已有的 past_key_values、截取后的输入序列 ID、是否使用缓存标识、注意力掩码
    return {
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "input_ids": input_ids,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
    }
    
    # 一个静态方法，用于重排 past_key_values
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 创建一个空的元组作为重排后的 past_key_values
        reordered_past = ()
        
        # 遍历每一层的 past_key_values
        for layer_past in past_key_values:
            # 对每一层的状态按照 beam_idx 进行重排
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        
        # 返回重排后的 past_key_values
        return reordered_past
# 定义了带有序列分类头部的 Whisper 编码器模型，用于类似 SUPERB Keyword Spotting 等任务
class WhisperForAudioClassification(WhisperPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 Whisper 编码器
        self.encoder = WhisperEncoder(config)
        # 计算层数，包括变压器层和输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置中使用加权的层求和
        if config.use_weighted_layer_sum:
            # 初始化层权重
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 初始化投影层
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 初始化分类器层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结编码器参数，使其在训练过程中不更新梯度
    def freeze_encoder(self):
        self.encoder._freeze_parameters()

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    # 设置输入嵌入层
    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)

    # 定义前向传播函数
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