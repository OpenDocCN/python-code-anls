# `.\models\mbart\modeling_mbart.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息，指明代码版权归 Facebook AI Research Team 和 HuggingFace Inc. 团队所有，使用 Apache License, Version 2.0 授权
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则不得使用此文件中的代码
""" PyTorch MBART 模型定义 """
# 导入必要的库和模块
import copy  # 导入深拷贝功能
import math  # 导入数学函数
from typing import List, Optional, Tuple, Union  # 引入类型提示

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数库
import torch.utils.checkpoint  # 导入 PyTorch 的检查点功能
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

# 从本地或者上层模块导入所需的工具函数和类
from ...activations import ACT2FN  # 导入激活函数映射表
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入注意力掩码处理工具函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (  # 导入工具函数和类
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_mbart import MBartConfig  # 从当前模块导入 MBART 配置类

# 如果支持 Flash Attention 2.0，导入相关函数和模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# 模型文档中使用的检查点名称
_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
# 模型文档中使用的配置名称
_CONFIG_FOR_DOC = "MBartConfig"

# 期望的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# MBART 预训练模型的存档列表
MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    # 查看所有 MBART 模型列表：https://huggingface.co/models?filter=mbart
]

# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数，用于获取未填充数据
def _get_unpad_data(attention_mask):
    # 计算每个样本序列的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非填充位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# 将输入的 ID 向右移动一个位置，用于生成输入序列的右移版本，用于 MBart 模型的输入处理
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    将输入的 ID 向右移动一个位置，并包装最后一个非填充标记（即 <LID> 标记）。需要注意的是，与其他类似 Bart 的模型不同，MBart 没有单一的 `decoder_start_token_id`。
    """
    # 复制输入的 token 序列作为输出的初始 token 序列
    prev_output_tokens = input_ids.clone()

    # 如果未定义 pad_token_id，则抛出数值错误异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    
    # 将 labels 中可能存在的值为 -100 的部分替换为 pad_token_id
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    # 找到每个样本中最后一个非 pad_token_id 的位置，形成一个索引张量
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)

    # 根据 index_of_eos 中的索引，获取每个样本中的 decoder 起始 token
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()

    # 将 prev_output_tokens 中每个样本的 token 序列整体左移一位
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    
    # 将 prev_output_tokens 中每个样本的第一个 token 替换为 decoder_start_tokens
    prev_output_tokens[:, 0] = decoder_start_tokens

    # 返回处理后的 prev_output_tokens，即新的输出 token 序列
    return prev_output_tokens
# Copied from transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding with Bart->MBart
class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # MBart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        # Call the constructor of nn.Embedding with adjusted num_embeddings
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        # Extract batch size and sequence length from input_ids tensor
        bsz, seq_len = input_ids.shape[:2]
        # Generate positions tensor starting from past_key_values_length up to past_key_values_length + seq_len
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        # Return the positional embeddings by adding self.offset to positions
        return super().forward(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->MBart
class MBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MBartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # Check if embed_dim is divisible by num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # Scaling factor for dot product attention
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # Linear projections for key, value, query and output
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # Reshape tensor to [batch_size, num_heads, seq_len, head_dim]
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        ):
        # Forward pass through the multi-headed attention mechanism
        # ...
        pass

# Copied from transformers.models.bart.modeling_bart.BartFlashAttention2 with Bart->MBart
class MBartFlashAttention2(MBartAttention):
    """
    Placeholder class for future extension or modification.
    """
    pass
    # MBart flash attention module. This module inherits from `MBartAttention` as the weights of the module stays
    # untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    # flash attention and deal with padding tokens in case the input contains any of them.

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    # 初始化函数，继承自父类，初始化模块。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 控制属性，用于处理 Flash Attention 不同版本之间的差异。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Reshape 操作，将张量重新排列成指定形状。
    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    # Flash Attention 的前向传播函数，处理查询、键、值、注意力掩码等参数。
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
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
        # Determine if causal attention is needed based on current settings and conditions
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary workaround for Flash Attention on RoCm platform; check not needed after version 2.1
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the input sequence
        if attention_mask is not None:
            # Get batch size from query_states tensor
            batch_size = query_states.shape[0]
            
            # Unpad input tensors based on attention_mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Unpacked variables from cu_seq_lens and max_seq_lens
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform Flash Attention with variable length support
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

            # Pad the attention output to match the original input sequence length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Perform regular Flash Attention without padding
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the computed attention output
        return attn_output
    # 定义一个私有方法 `_upad_input`，用于处理注意力机制的输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取非填充数据的索引、当前序列长度及批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取 key_layer 的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据 indices_k 重新索引并重新组织 key_layer 和 value_layer
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据 query_length 的不同情况处理 query_layer
        if query_length == kv_seq_len:
            # 如果 query_length 等于 kv_seq_len，则按 indices_k 重新索引 query_layer
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果 query_length 等于 1，则处理单个 query 的情况
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy，非常糟糕。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，假设存在左填充，根据 query_length 和 attention_mask 进行处理
            # 注意：这里的 -query_length 切片假设存在左填充。
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的 query_layer、key_layer、value_layer，以及相关的索引和长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
MBART_ATTENTION_CLASSES = {
    "eager": MBartAttention,  # 定义一个字典，将字符串映射到对应的注意力机制类
    "flash_attention_2": MBartFlashAttention2,
}

class MBartEncoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 从配置中获取嵌入维度大小

        # 初始化自注意力层，根据配置选择不同的注意力机制类
        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

        # 初始化自注意力层的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 获取配置中的 dropout 概率
        self.dropout = config.dropout

        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.activation_function]

        # 获取配置中的激活函数 dropout 概率
        self.activation_dropout = config.activation_dropout

        # 第一个线性层，将嵌入维度映射到编码器前馈网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)

        # 第二个线性层，将编码器前馈网络维度映射回嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

        # 最终的 LayerNorm 层，对输出进行归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
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
        # 保存输入的隐藏状态作为残差连接的基础
        residual = hidden_states
        # 对隐藏状态进行 LayerNorm 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力机制进行计算，得到新的隐藏状态、注意力权重和注意力概率（此处第三个返回值用下划线 `_` 表示）
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对输出的隐藏状态进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与处理后的隐藏状态相加，实现残差连接
        hidden_states = residual + hidden_states

        # 保存当前状态作为下一步的残差连接基础
        residual = hidden_states
        # 对最终输出的隐藏状态进行 LayerNorm 处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理第一个全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对第一个全连接层的输出进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 经过第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 对第二个全连接层的输出进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与处理后的隐藏状态相加，实现残差连接
        hidden_states = residual + hidden_states

        # 如果隐藏状态的数据类型是 torch.float16，并且存在无穷大或 NaN 的情况，则进行 clamp 操作
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 将最终的隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重也加入到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义 MBartDecoderLayer 类，继承自 nn.Module，用于 MBart 解码器层的实现
class MBartDecoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化自注意力机制，根据配置选择实现类，并设置相关参数
        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 初始化自注意力层规范化器
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化编码器注意力机制，根据配置选择实现类，并设置相关参数
        self.encoder_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 初始化编码器注意力层规范化器
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化第一个线性层（前馈神经网络的第一层）
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)

        # 初始化第二个线性层（前馈神经网络的第二层）
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 初始化最终层规范化器
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，定义了层的数据流向
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
        # 以下为具体的层操作实现
        # 注意力机制和规范化
        ...


# 定义 MBartClassificationHead 类，用于 MBart 模型的分类任务头部
class MBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        # 初始化线性层，用于将输入维度映射到内部维度
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 输出投影层，将内部维度映射到类别数量
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 前向传播函数，定义了头部的数据流向
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 通过线性层进行映射和激活函数
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 最终通过输出投影层得到分类结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# 定义 MBartPreTrainedModel 类，继承自 PreTrainedModel，作为 MBart 模型的基类
class MBartPreTrainedModel(PreTrainedModel):
    config_class = MBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MBartDecoderLayer", "MBartAttention"]
    _supports_flash_attn_2 = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        std = self.config.init_std  # 获取初始化标准差
        if isinstance(module, nn.Linear):  # 如果当前模块是线性层
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化权重为正态分布
            if module.bias is not None:  # 如果存在偏置项
                module.bias.data.zero_()  # 将偏置项初始化为零
        elif isinstance(module, nn.Embedding):  # 如果当前模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化嵌入矩阵的权重为正态分布
            if module.padding_idx is not None:  # 如果指定了填充索引
                module.weight.data[module.padding_idx].zero_()  # 将填充索引位置的权重初始化为零

    @property
    # 获取一个虚拟输入示例的属性方法
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id  # 获取填充标记的 ID
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)  # 创建输入 ID 张量
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 生成注意力掩码，排除填充标记
            "input_ids": input_ids,  # 将输入 ID 放入字典
        }
        return dummy_inputs  # 返回虚拟输入字典
MBART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MBartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MBART_GENERATION_EXAMPLE = r"""
    Translation example:

    ```
    >>> from transformers import AutoTokenizer, MBartForConditionalGeneration

    >>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-en-ro")

    >>> example_english_phrase = "42 is the answer"
    >>> inputs = tokenizer(example_english_phrase, return_tensors="pt")

    >>> # Translate
    >>> generated_ids = model.generate(**inputs, num_beams=4, max_length=5)
    >>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    '42 este răspuns'
    ```

    Mask filling example:

    ```
    >>> from transformers import AutoTokenizer, MBartForConditionalGeneration

    >>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    >>> # de_DE is the language symbol id <LID> for German
    >>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"

    >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="pt")["input_ids"]
    >>> logits = model(input_ids).logits

    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)

    >>> tokenizer.decode(predictions).split()
    ['nett', 'sehr', 'ganz', 'nicht', 'so']
    ```
"""

MBART_INPUTS_DOCSTRING = r"""
"""


class MBartEncoder(MBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MBartEncoderLayer`].

    Args:
        config: MBartConfig
            Model configuration class with all the parameters of the model.
        embed_tokens (nn.Embedding): output embedding
            The output embedding for the model.
    """
    # 初始化函数，接受一个 MBartConfig 对象和一个可选的嵌入词向量对象作为参数
    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 从配置中获取 dropout 和 encoder_layerdrop 的数值
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 从配置中获取嵌入的维度，并设置填充 token 的索引
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        # 设置最大的源序列长度
        self.max_source_positions = config.max_position_embeddings
        # 根据配置选择是否对嵌入进行缩放
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 创建嵌入层，vocab_size 是词汇表的大小，embed_dim 是嵌入的维度，padding_idx 是填充 token 的索引
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 如果提供了外部的嵌入词向量，则使用它来初始化嵌入层的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建学习的位置嵌入对象，max_position_embeddings 是位置的最大数量，embed_dim 是嵌入的维度
        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        
        # 创建一系列 MBartEncoderLayer 层，并存储在 layers 中，数量由 encoder_layers 决定
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        
        # 根据配置决定是否使用 flash_attention_2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        
        # 对嵌入进行 layernorm 处理，embed_dim 是嵌入的维度
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        
        # 对编码器的输出进行 layernorm 处理，config.d_model 是模型的维度
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 是否启用梯度检查点，默认为 False
        self.gradient_checkpointing = False

        # 调用 post_init 方法来初始化权重并进行最终的处理
        self.post_init()

    # 用于向后兼容梯度检查点，如果配置中设置了 gradient_checkpointing，则启用梯度检查点
    def _backward_compatibility_gradient_checkpointing(self):
        # 不删除配置中的梯度检查点属性
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

    # 前向传播函数，接受多个输入参数，包括 input_ids、attention_mask 等
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class MBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MBartDecoderLayer`]

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 从配置中获取dropout率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取层dropout率
        self.padding_idx = config.pad_token_id  # 获取填充token的索引
        self.max_target_positions = config.max_position_embeddings  # 获取目标位置的最大数目
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 计算嵌入尺度

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)  # 初始化嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果提供了预训练的嵌入，使用它们

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )  # 初始化位置编码器

        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建多层解码器层
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"  # 检查是否使用了Flash Attention 2
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 初始化嵌入层的LayerNorm
        self.layer_norm = nn.LayerNorm(config.d_model)  # 初始化层的LayerNorm

        self.gradient_checkpointing = False  # 梯度检查点设为False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入的嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入的嵌入层

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 函数签名和参数说明
        """
        MBartDecoder的前向传播函数

        Args:
            input_ids (torch.LongTensor, optional): 输入的token IDs
            attention_mask (torch.Tensor, optional): 注意力掩码
            encoder_hidden_states (torch.FloatTensor, optional): 编码器的隐藏状态
            encoder_attention_mask (torch.LongTensor, optional): 编码器的注意力掩码
            head_mask (torch.Tensor, optional): 多头注意力的头部掩码
            cross_attn_head_mask (torch.Tensor, optional): 跨注意力头部的掩码
            past_key_values (Tuple[Tuple[torch.FloatTensor]], optional): 缓存的键值对
            inputs_embeds (torch.FloatTensor, optional): 输入的嵌入表示
            use_cache (bool, optional): 是否使用缓存
            output_attentions (bool, optional): 是否输出注意力
            output_hidden_states (bool, optional): 是否输出隐藏状态
            return_dict (bool, optional): 是否返回字典

        Returns:
            根据配置返回不同的输出
        """
        pass  # 此处省略了实际的前向传播逻辑，需要补充完整
    # 返回共享的输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入，并将其分配给编码器和解码器的嵌入
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 如果配置要求，绑定编码器和解码器的词嵌入权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.get_input_embeddings())
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())

    # 前向传播函数，接收多个输入和控制参数，输出Seq2Seq模型的结果
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
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
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 为 MBART 模型添加文档字符串，描述其具有语言建模头部的功能，适用于摘要生成，需要在预训练模型微调后使用。
@add_start_docstrings(
    "The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models.",
    MBART_START_DOCSTRING,
)
class MBartForConditionalGeneration(MBartPreTrainedModel):
    # 基础模型的前缀，用于加载模型时忽略的键
    base_model_prefix = "model"
    # 在加载模型时忽略的缺失键
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 共享权重的键列表
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受一个 MBART 配置对象
    def __init__(self, config: MBartConfig):
        super().__init__(config)
        # 使用给定的配置创建 MBartModel 模型
        self.model = MBartModel(config)
        # 注册一个缓冲区，用于存储最终对数偏置，维度为 (1, num_embeddings)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层 lm_head，用于语言建模，输入维度为 config.d_model，输出维度为 num_embeddings
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取编码器模型
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器模型
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类方法，调整 token embeddings 大小
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整最终对数偏置的大小以匹配新的 token embeddings
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整最终对数偏置的大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册调整后的最终对数偏置
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型前向方法的文档字符串，包括输入的详细说明
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串，指定输出类型为 Seq2SeqLMOutput
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向方法的结尾文档字符串，包括生成示例
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    # 定义模型的前向传播函数，接收多个输入参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入序列的token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 输入序列的注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入token IDs
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器多头注意力机制的掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力机制的多头掩码
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 缓存的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入表示
        labels: Optional[torch.LongTensor] = None,  # 模型的标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Either a Seq2SeqLMOutput containing loss, logits, and other optional outputs, or a tuple of
            torch.FloatTensor containing logits and optional outputs.

        """
        # Determine whether to return results in a dictionary format or not
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle special case where labels are provided
        if labels is not None:
            # Adjust `use_cache` to False when `labels` are provided, with a warning message
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # If `decoder_input_ids` or `decoder_inputs_embeds` are not provided, generate `decoder_input_ids`
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        # Pass inputs to the model for forward computation
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

        # Calculate logits for language modeling head and apply bias
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        # Calculate masked language modeling loss if `labels` are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Depending on `return_dict`, construct and return output tuple or Seq2SeqLMOutput
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return results as Seq2SeqLMOutput with specified outputs
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
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
        # 如果使用了过去的键值对（past_key_values），则根据其长度截断 decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 从 decoder_input_ids 中截取需要的部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回一个包含各种模型输入和设置的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 将此项更改以避免缓存（可能用于调试目的）
        }

    # 从标签（labels）准备解码器输入的静态方法
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    # 重新排序缓存中的 past_key_values，以匹配给定的 beam_idx
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终保持不变
            reordered_past += (
                # 对每个层的过去状态执行索引选择，以匹配给定的 beam_idx
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],  # 添加未修改的剩余部分
            )
        return reordered_past
@add_start_docstrings(
    """
    MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MBART_START_DOCSTRING,
)
# 定义 MBart 序列分类模型，建立在 MBartPreTrainedModel 基础上
class MBartForSequenceClassification(MBartPreTrainedModel):
    # 需要共享权重的键列表，用于 tied weights 功能
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]

    def __init__(self, config: MBartConfig, **kwargs):
        # 调用父类构造函数初始化模型
        super().__init__(config, **kwargs)
        # 初始化 MBart 模型
        self.model = MBartModel(config)
        # 初始化序列分类头部
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.bart.modeling_bart.BartForSequenceClassification.forward 复制而来
    # 前向传播函数定义
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



@add_start_docstrings(
    """
    MBART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MBART_START_DOCSTRING,
)
# 定义 MBart 问答模型，用于类似 SQuAD 的抽取式问答任务
class MBartForQuestionAnswering(MBartPreTrainedModel):
    # 需要共享权重的键列表，用于 tied weights 功能
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]

    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super().__init__(config)

        # 设定分类标签数目为 2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 初始化 MBart 模型
        self.model = MBartModel(config)
        # 初始化问答输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bart.modeling_bart.BartForQuestionAnswering.forward
    # 定义 BartForQuestionAnswering 模型的前向传播方法，接受多个输入参数
    
    def forward(
        self,
        input_ids: torch.Tensor = None,  # 输入的 token IDs 张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，用于指示哪些 token 是需要注意的
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入 token IDs 张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力遮罩张量
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力的掩码
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器的多头注意力掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨层注意力的掩码
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器的输出列表
        start_positions: Optional[torch.LongTensor] = None,  # 答案开始位置的张量
        end_positions: Optional[torch.LongTensor] = None,  # 答案结束位置的张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入向量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
# 从 transformers.models.bart.modeling_bart.BartDecoderWrapper 复制而来，将 Bart 替换为 MBart
class MBartDecoderWrapper(MBartPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与 [`EncoderDecoderModel`] 框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        # 调用父类构造函数，初始化 MBartDecoderWrapper 对象
        super().__init__(config)
        # 创建 MBartDecoder 对象
        self.decoder = MBartDecoder(config)

    def forward(self, *args, **kwargs):
        # 前向传播函数，调用 MBartDecoder 的前向传播方法
        return self.decoder(*args, **kwargs)


# 从 transformers.models.bart.modeling_bart.BartForCausalLM 复制而来，将 Bart 替换为 MBart，facebook/bart-base 替换为 facebook/mbart-large-cc25
class MBartForCausalLM(MBartPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深拷贝配置对象，配置为解码器，不是编码-解码器结构
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类构造函数，初始化 MBartForCausalLM 对象
        super().__init__(config)
        # 创建 MBartDecoderWrapper 对象作为模型的核心部分
        self.model = MBartDecoderWrapper(config)

        # 初始化线性层，用于语言模型的输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层，即 MBartDecoder 的嵌入标记
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入层，即 MBartDecoder 的嵌入标记
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 获取输出嵌入层，即语言模型头部
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层，即语言模型头部
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器，用于动态设置 MBartDecoderWrapper 的解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 获取解码器，即 MBartDecoderWrapper 的解码器
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
        # 前向传播函数，详细参数见函数声明
        pass

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 为生成准备输入的函数，详细参数见函数声明
        pass
    ):
        # 如果模型用作编码器-解码器模型中的解码器，解码器注意力遮罩在需要时动态创建
        if attention_mask is None:
            # 如果注意力遮罩为空，则创建一个与输入形状相同的全1张量作为注意力遮罩
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去关键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 有些生成方法已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 如果当前输入长度大于过去长度，则移除前缀的长度为过去长度
                remove_prefix_length = past_length
            else:
                # 否则，默认行为是只保留最后一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截取输入序列，移除前缀长度
            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，解码器缓存状态为空
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 重新排序过去的关键值，根据 beam_idx 对每一层的 past_state 进行索引选择
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```