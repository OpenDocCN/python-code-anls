# `.\models\bart\modeling_bart.py`

```
# 设置文件编码格式为UTF-8

# 版权声明和许可信息

# 导入所需的库和模块
""" PyTorch BART模型."""
import copy  # 导入深拷贝函数
import math  # 导入数学函数
import warnings  # 导入警告模块
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint模块
from torch import nn  # 导入PyTorch的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_attn_mask_utils import (  # 导入注意力掩码的辅助函数
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (  # 导入模型输出相关的类和函数
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的工具函数
from ...utils import (  # 导入工具函数
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .configuration_bart import BartConfig  # 导入BART配置文件

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # 如果支持flash attention，导入相关函数
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # 导入flash attention相关的辅助函数  # noqa

logger = logging.get_logger(__name__)  # 获取logger实例

_CHECKPOINT_FOR_DOC = "facebook/bart-base"  # 用于文档的检查点
_CONFIG_FOR_DOC = "BartConfig"  # 用于文档的配置

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]  # 预期输出形状

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"  # 序列分类任务的检查点
_SEQ_CLASS_EXPECTED_LOSS = 0.0  # 序列分类任务的预期损失
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"  # 序列分类任务的预期输出

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"  # 问答任务的检查点
_QA_EXPECTED_LOSS = 0.59  # 问答任务的预期损失
_QA_EXPECTED_OUTPUT = "' nice puppet'"  # 问答任务的预期输出

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型存档列表
    "facebook/bart-large",
    # 查看所有BART模型，请访问 https://huggingface.co/models?filter=bart
]


# 从transformers.models.llama.modeling_llama._get_unpad_data复制过来的函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)  # 计算批次中每个序列的长度
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()  # 找出非零位置的索引
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # 获取批次中的最大序列长度
    # 对输入的序列长度进行累积求和，并在最前面填充一个零，以形成累计序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # 返回三个值作为元组的形式
    return (
        indices,                # 返回的第一个元素是索引数组
        cu_seqlens,             # 返回的第二个元素是填充后的累计序列长度数组
        max_seqlen_in_batch,    # 返回的第三个元素是批次中最大的序列长度
    )
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个新的张量，与input_ids相同形状
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将input_ids的内容向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将decoder_start_token_id放置在shifted_input_ids的首列
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将shifted_input_ids中可能存在的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        # 调用父类构造函数初始化Embedding层
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        # 根据当前序列长度和历史键值对长度计算位置张量
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        super().__init__()
        # 初始化注意力机制的各种参数
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
        # 缩放因子，用于缩放点积注意力的结果
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性变换层，用于计算Q、K、V向量
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 输出层的线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 调整张量形状，以适应多头注意力计算的需求
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义模型的前向传播方法
    def forward(
        # 隐藏状态：输入的张量，表示模型的隐藏状态
        self,
        # 键值状态：可选的张量，表示用于键值计算的状态
        hidden_states: torch.Tensor,
        # 过去的键值：可选的元组，包含过去计算得到的键值状态
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 注意力掩码：可选的张量，用于掩盖不需要处理的部分
        attention_mask: Optional[torch.Tensor] = None,
        # 层头掩码：可选的张量，用于层间的掩盖操作
        layer_head_mask: Optional[torch.Tensor] = None,
        # 输出注意力：布尔值，表示是否输出注意力权重信息
        output_attentions: bool = False,
    """
    Bart flash attention module. This module inherits from `BartAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        Reshape the input tensor into the desired shape.
        
        Args:
        - tensor (torch.Tensor): The input tensor to reshape.
        - seq_len (int): Length of the sequence.
        - bsz (int): Batch size.
        
        Returns:
        - torch.Tensor: Reshaped tensor of shape (bsz, seq_len, num_heads, head_dim).
        """
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
        """
        Perform the forward pass of the attention module.
        
        Args:
        - hidden_states (torch.Tensor): Input hidden states.
        - key_value_states (Optional[torch.Tensor]): Key and value states if provided separately.
        - past_key_value (Optional[Tuple[torch.Tensor]]): Past key and value tensors.
        - attention_mask (Optional[torch.Tensor]): Mask for attention computation.
        - layer_head_mask (Optional[torch.Tensor]): Mask for heads within a layer.
        - output_attentions (bool): Whether to output attentions.

        Returns:
        - torch.Tensor: Output tensor from the attention module.
        """
        # Forward pass logic goes here
        pass
    
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Perform forward pass specific to flash attention mechanism.
        
        Args:
        - query_states: Query states tensor.
        - key_states: Key states tensor.
        - value_states: Value states tensor.
        - attention_mask: Mask for attention computation.
        - query_length: Length of the query sequence.
        - dropout (float): Dropout rate.
        - softmax_scale: Scaling factor for softmax computation.

        Returns:
        - torch.Tensor: Output tensor after applying flash attention.
        """
        # Implementation details for flash attention forward pass
        pass
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
        # Determine if causal masking is needed based on `_flash_attn_uses_top_left_mask` and `query_length`
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary check for RoCm compatibility; remove when Flash Attention for RoCm is updated
            causal = self.is_causal and query_length != 1

        # Apply attention masking if `attention_mask` is provided
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input sequences based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Compute attention scores for variable-length sequences
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
            # Compute attention scores without masking
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 在内部方法中，根据给定的注意力掩码处理输入数据，返回处理后的查询、键、值、索引等。
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取键层的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        # 根据索引重新排列键层和值层的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度决定如何处理查询层
        if query_length == kv_seq_len:
            # 如果查询长度等于键值序列长度，则重新排列查询层并更新相关变量
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，则直接处理查询层为批次大小，同时更新相关变量
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里存在一次内存拷贝，效率较低。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据查询长度和注意力掩码未填充数据处理查询层
            # 注意力掩码仅保留后面的 -query_length 切片，假设为左填充。
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        
        # 返回处理后的查询层、键层、值层、查询索引、序列长度元组和最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class BartSdpaAttention(BartAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 在 BartSdpaAttention 类中重写 forward 方法，用于执行自注意力机制
        # 参数说明：
        # - hidden_states: 输入的隐藏状态张量
        # - key_value_states: 可选参数，键值状态张量（默认为 None）
        # - past_key_value: 可选参数，过去的键值元组（默认为 None）
        # - attention_mask: 可选参数，注意力掩码张量（默认为 None）
        # - layer_head_mask: 可选参数，层头掩码张量（默认为 None）
        # - output_attentions: 是否输出注意力权重，默认为 False
        pass

BART_ATTENTION_CLASSES = {
    "eager": BartAttention,
    "sdpa": BartSdpaAttention,
    "flash_attention_2": BartFlashAttention2,
}

# 定义 BART 模型中不同注意力机制实现的类映射字典

class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化 BartEncoderLayer 类，根据配置选择注意力机制实现类
        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ):
        # 在 BartEncoderLayer 类中重写 forward 方法，执行编码器层的前向传播
        # 参数说明：
        # - hidden_states: 输入的隐藏状态张量
        # - attention_mask: 注意力掩码张量
        # - layer_head_mask: 层头掩码张量
        # - output_attentions: 是否输出注意力权重，默认为 False
        pass
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
        # 保存输入的隐藏状态作为残差连接的基准
        residual = hidden_states
        # 进行自注意力机制计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对输出的隐藏状态进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接后的结果加回到隐藏状态中
        hidden_states = residual + hidden_states
        # 对加和后的隐藏状态进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 再次保存当前隐藏状态作为残差连接的基准
        residual = hidden_states
        # 应用激活函数并传入第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对输出的隐藏状态进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 传入第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 对输出的隐藏状态进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接后的结果加回到隐藏状态中
        hidden_states = residual + hidden_states
        # 对加和后的隐藏状态进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果隐藏状态的数据类型是 float16 且包含无穷大或 NaN 值，则进行数值的 clamp 处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 准备输出结果，将隐藏状态打包成元组形式
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重也添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出结果
        return outputs
class BartDecoderLayer(nn.Module):
    # BART 解码器层模块，继承自 nn.Module
    def __init__(self, config: BartConfig):
        # 初始化方法，接受一个 BartConfig 类型的参数 config
        super().__init__()
        self.embed_dim = config.d_model
        # 从 config 中获取模型的嵌入维度

        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 初始化自注意力机制，使用 BART_ATTENTION_CLASSES 中对应的实现类

        self.dropout = config.dropout
        # 设置 dropout 概率

        self.activation_fn = ACT2FN[config.activation_function]
        # 选择激活函数

        self.activation_dropout = config.activation_dropout
        # 激活函数的 dropout 概率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # LayerNorm 层，用于自注意力机制的输出

        self.encoder_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 初始化编码器注意力机制，同样使用 BART_ATTENTION_CLASSES 中对应的实现类

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # LayerNorm 层，用于编码器注意力机制的输出

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 第一个全连接层

        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 第二个全连接层

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # 最终的 LayerNorm 层，用于整个层的输出

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
    ):
        # 前向传播方法
        # hidden_states: 输入的隐藏状态张量
        # attention_mask: 注意力掩码，可选
        # encoder_hidden_states: 编码器的隐藏状态张量，可选
        # encoder_attention_mask: 编码器的注意力掩码，可选
        # layer_head_mask: 层级头掩码，可选
        # cross_attn_layer_head_mask: 跨注意力层级头掩码，可选
        # past_key_value: 缓存的键值对，可选
        # output_attentions: 是否输出注意力权重，可选，默认为 False
        # use_cache: 是否使用缓存，可选，默认为 True

        # 下面是具体的前向传播计算过程，不同的操作符和层的作用已在初始化时进行了注释
        # 每一步的输出都需要通过相应的层（如 LayerNorm、Linear、Dropout）进行处理
        pass  # 实际前向传播逻辑在具体使用时实现


class BartClassificationHead(nn.Module):
    # 用于句子级分类任务的头部模块
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        # 初始化方法
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        # 全连接层，将输入维度映射到内部维度

        self.dropout = nn.Dropout(p=pooler_dropout)
        # Dropout 层

        self.out_proj = nn.Linear(inner_dim, num_classes)
        # 最终的全连接层，将内部维度映射到类别数量

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播方法
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        # 具体的前向传播计算过程，包括 Dropout、全连接层、激活函数等
        return hidden_states


class BartPreTrainedModel(PreTrainedModel):
    # BART 预训练模型基类
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    # 一些类属性和标记，指示模型的特性和行为，不涉及具体的计算逻辑
    # 初始化模型的权重，根据模块类型设定不同的初始化方式
    def _init_weights(self, module):
        # 从配置中获取初始化标准差
        std = self.config.init_std
        # 如果是线性层模块
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重数据
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，将其数据初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层模块
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重数据
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，将对应索引的权重数据初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 获取一个用于模型测试的虚拟输入数据字典
    @property
    def dummy_inputs(self):
        # 获取配置中的填充标记 ID
        pad_token = self.config.pad_token_id
        # 构造虚拟的输入 ID 张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构造虚拟输入数据字典，包括注意力遮罩和输入 ID
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 根据填充标记 ID 生成注意力遮罩
            "input_ids": input_ids,  # 将构造的输入 ID 加入输入数据字典
        }
        # 返回构造好的虚拟输入数据字典
        return dummy_inputs
class PretrainedBartModel(BartPreTrainedModel):
    def __init_subclass__(self):
        # 发出警告，提示使用已过时的 `PretrainedBartModel` 类，请改用 `BartPreTrainedModel`
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


class BartPretrainedModel(BartPreTrainedModel):
    def __init_subclass__(self):
        # 发出警告，提示使用已过时的 `PretrainedBartModel` 类，请改用 `BartPreTrainedModel`
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
    ```

    Mask filling example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    >>> TXT = "My friends are <mask> but they eat too many carbs."
    >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    >>> logits = model(input_ids).logits


"""
    # 找到输入序列中第一个遮罩标记的索引位置
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    
    # 使用模型输出的 logits 对遮罩位置的预测结果进行 softmax 处理，得到概率分布
    probs = logits[0, masked_index].softmax(dim=0)
    
    # 获取概率分布中前五个最高概率对应的值和它们的索引
    values, predictions = probs.topk(5)
    
    # 将预测出的索引转换为词汇，并以列表形式返回
    tokenizer.decode(predictions).split()
    ['not', 'good', 'healthy', 'great', 'very']
# 定义 BART 模型的输入文档字符串
BART_INPUTS_DOCSTRING = r"""
"""


class BartEncoder(BartPreTrainedModel):
    """
    BART 编码器，由 *config.encoder_layers* 个自注意力层组成的Transformer编码器。

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): 输出的嵌入表示
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout  # 配置的dropout率
        self.layerdrop = config.encoder_layerdrop  # 编码器层的dropout率

        embed_dim = config.d_model  # 嵌入维度
        self.padding_idx = config.pad_token_id  # 填充token的索引
        self.max_source_positions = config.max_position_embeddings  # 最大源序列位置
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 嵌入缩放因子

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)  # 词嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果提供了预训练的嵌入，则使用它

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )  # 学习的位置编码嵌入

        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])  # 编码器层列表
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"  # 是否使用Flash Attention 2
        self._use_sdpa = config._attn_implementation == "sdpa"  # 是否使用SDPA（Scaled Dot-Product Attention）
        self.layernorm_embedding = nn.LayerNorm(embed_dim)  # 嵌入层的LayerNorm

        self.gradient_checkpointing = False  # 梯度检查点，默认关闭
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 获取输入的嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入的嵌入层

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化方法，接收一个BartConfig对象和一个可选的嵌入词表的参数
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法，传入配置对象config
        super().__init__(config)
        # 设置dropout比例为配置对象中的dropout值
        self.dropout = config.dropout
        # 设置层级dropout比例为配置对象中的decoder_layerdrop值
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引为配置对象中的pad_token_id值
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置为配置对象中的max_position_embeddings值
        self.max_target_positions = config.max_position_embeddings
        # 如果配置对象中设置了scale_embedding为True，则设置embed_scale为d_model的平方根，否则为1.0
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 初始化嵌入词表，使用nn.Embedding类，参数为vocab_size, d_model和padding_idx
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 如果传入了额外的embed_tokens参数，则使用其权重覆盖当前embed_tokens的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 初始化位置嵌入，使用BartLearnedPositionalEmbedding类，参数为max_position_embeddings和d_model
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        
        # 使用BartDecoderLayer类创建decoder层的ModuleList，长度为config.decoder_layers
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        
        # 根据配置中的_attn_implementation值，设置是否使用flash_attention_2方法
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 根据配置中的_attn_implementation值，设置是否使用sdpa方法
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 初始化embedding的LayerNorm，参数为d_model
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点标志为False
        self.gradient_checkpointing = False

        # 调用post_init方法，用于初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法，返回当前的embed_tokens对象
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入的方法，将传入的value赋值给embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    "The BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
# 定义 BART 模型类，继承自 BartPreTrainedModel
class BartModel(BartPreTrainedModel):
    # 被绑定权重的键名列表，用于共享编码和解码器的嵌入权重
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化函数，接收一个 BartConfig 对象作为参数
    def __init__(self, config: BartConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置填充索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的词嵌入层，大小为 vocab_size × config.d_model，使用 padding_idx 进行填充
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器对象，传入配置对象和共享的词嵌入层
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 绑定权重函数
    def _tie_weights(self):
        # 如果配置要求绑定词嵌入权重，则将编码器和解码器的 embed_tokens 与共享的词嵌入层绑定或克隆
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取输入词嵌入函数
    def get_input_embeddings(self):
        return self.shared

    # 设置输入词嵌入函数
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 获取编码器对象函数
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象函数
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 前向传播函数，接收多种输入和掩码，返回预测输出
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化方法，接受一个 BartConfig 对象作为参数
    def __init__(self, config: BartConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 BartModel 对象并赋值给 self.model
        self.model = BartModel(config)
        # 初始化一个形状为 (1, self.model.shared.num_embeddings) 的零张量，作为 final_logits_bias 属性
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层，用于输出 logits，输入维度为 config.d_model，输出维度为 self.model.shared.num_embeddings
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 调用自定义的后初始化方法
        self.post_init()

    # 获取编码器部分的方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器部分的方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 的方法，返回调整后的新 embeddings
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法，获取新的 embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调用私有方法调整 final_logits_bias 属性
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 返回调整后的新 embeddings
        return new_embeddings

    # 调整 final_logits_bias 的私有方法，根据新的 token 数量调整 bias 的大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取当前 final_logits_bias 的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于当前的 token 数量，直接截取对应的部分作为新的 bias
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        # 如果新的 token 数量大于当前的 token 数量，则扩展新的 bias，并将扩展部分填充为零
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册更新后的 final_logits_bias 属性
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings 的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings 的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播方法，接受多个输入参数，详细说明见装饰器内的文档字符串
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Returns either a tuple or `Seq2SeqLMOutput` depending on `return_dict`.

        """
        # Determine whether to use the provided `return_dict` or default from `config`
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If labels are provided, adjust `use_cache` and initialize `decoder_input_ids` if not provided
        if labels is not None:
            if use_cache:
                # Issue a warning about setting `use_cache` to `False` when `labels` are provided
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            # Always set `use_cache` to `False` when `labels` are provided
            use_cache = False
            # If `decoder_input_ids` is not provided, shift `labels` to the right for decoder inputs
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Forward the inputs to the model for computation
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Generate logits from the language model head and add bias
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        # Compute masked language modeling loss if labels are provided
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Return either a tuple or `Seq2SeqLMOutput` based on `return_dict`
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值（past_key_values），则根据其长度调整 decoder_input_ids 的长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回一个字典，包含准备好的生成器输入的各种组件
        return {
            "input_ids": None,  # encoder_outputs 已定义，input_ids 不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此处以避免缓存（可能是为了调试目的）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签向右移动，以准备解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们总是相同的
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,
)
class BartForSequenceClassification(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)  # 初始化一个Bart模型
        self.classification_head = BartClassificationHead(  # 初始化一个Bart分类头部
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化步骤，包括权重初始化等

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the BartForSequenceClassification model.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding tokens.
            decoder_input_ids: Indices of decoder input sequence tokens in the vocabulary.
            decoder_attention_mask: Mask to avoid performing attention on padding tokens for decoder.
            head_mask: Mask to nullify selected heads of the self-attention modules.
            decoder_head_mask: Mask to nullify selected heads of the cross-attention modules.
            cross_attn_head_mask: Mask to nullify selected heads of the cross-attention modules.
            encoder_outputs: Hidden states of the encoder at each layer.
            inputs_embeds: Optional tensor of embeddings to be used instead of input_ids.
            decoder_inputs_embeds: Optional tensor of embeddings to be used instead of decoder_input_ids.
            labels: Labels for computing the sequence classification/regression loss.
            use_cache: Whether or not to use the pre-computed hidden states cache.
            output_attentions: Whether or not to return the attentions tensors.
            output_hidden_states: Whether or not to return the hidden states tensors.
            return_dict: Whether or not to return a dictionary as output.

        Returns:
            Depending on `return_dict`, either a dictionary (`Seq2SeqSequenceClassifierOutput`) or
            a tuple with sequence classifier output and optional hidden states and attentions.
        """
        # Implementation of forward pass, computing sequence classification output

@add_start_docstrings(
    """
    BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BART_START_DOCSTRING,
)
class BartForQuestionAnswering(BartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)  # 初始化一个Bart模型
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 初始化用于QA任务的线性分类器

        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化步骤，包括权重初始化等

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,  # 使用指定的检查点用于问答模型
        output_type=Seq2SeqQuestionAnsweringModelOutput,  # 指定输出类型为Seq2SeqQuestionAnsweringModelOutput
        config_class=_CONFIG_FOR_DOC,  # 使用指定的配置类来配置模型
        expected_loss=_QA_EXPECTED_LOSS,  # 预期的损失值用于模型评估
        expected_output=_QA_EXPECTED_OUTPUT,  # 预期的输出用于模型评估
    )
    def forward(
        self,
        input_ids: torch.Tensor = None,  # 输入的token IDs张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的token IDs张量，可选
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，可选
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器的头部掩码张量，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码张量，可选
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出列表，每个元素是张量，可选
        start_positions: Optional[torch.LongTensor] = None,  # 答案起始位置的张量，可选
        end_positions: Optional[torch.LongTensor] = None,  # 答案结束位置的张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入张量，可选
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入张量，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
class BartDecoderWrapper(BartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        # 调用父类的构造函数，初始化模型配置
        super().__init__(config)
        # 创建一个BartDecoder实例作为这个wrapper的decoder
        self.decoder = BartDecoder(config)

    def forward(self, *args, **kwargs):
        # 将输入参数传递给decoder模型，并返回其输出
        return self.decoder(*args, **kwargs)


@add_start_docstrings(
    """
    BART decoder with with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    BART_START_DOCSTRING,
)
class BartForCausalLM(BartPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深拷贝配置，确保修改不影响原始配置
        config = copy.deepcopy(config)
        # 将配置设置为decoder模式
        config.is_decoder = True
        # 设置为非Encoder-Decoder模型
        config.is_encoder_decoder = False
        # 调用父类的构造函数，初始化模型配置
        super().__init__(config)
        # 使用BartDecoderWrapper创建一个decoder模型
        self.model = BartDecoderWrapper(config)

        # 创建一个线性层作为语言建模头部，权重与输入嵌入层的权重绑定
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型中decoder的嵌入层的嵌入tokens
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型中decoder的嵌入层的嵌入tokens
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回语言建模头部的输出嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言建模头部的输出嵌入层
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置模型中的decoder
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回模型中的decoder
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 正向传播方法，接收多个输入参数，返回语言模型输出
        ...

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 为生成准备输入的方法，返回生成所需的输入
        ...
    ):
        # 如果模型作为编码器-解码器模型的解码器使用，则动态创建解码器注意力掩码
        if attention_mask is None:
            # 如果注意力掩码为空，则创建一个全为1的注意力掩码，形状与输入ID相同
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 如果输入ID的长度大于过去的长度，则移除前缀长度设为过去的长度
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截取输入ID的后部分以保留有效部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含以下内容的字典
        return {
            "input_ids": input_ids,  # 编码器输出已定义，不再需要输入ID
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序缓存中的过去键值对
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的过去状态按beam_idx重新排序，并添加到重新排序过的过去状态中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去状态
        return reordered_past
```