# `.\transformers\models\mbart\modeling_mbart.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook AI 研究团队和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch MBART model."""
# 导入所需的库和模块
import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_mbart import MBartConfig

# 检查是否可用 flash_attn 2
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取 logger
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
_CONFIG_FOR_DOC = "MBartConfig"

# 预期输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# MBART 预训练模型存档列表
MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    # 查看所有 MBART 模型：https://huggingface.co/models?filter=mbart
]

# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# 将输入的 token 向右移动一个位置，最后一个非填充 token（<LID> token）会被移到开头
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    # 复制输入的标记作为先前的输出标记
    prev_output_tokens = input_ids.clone()

    # 如果未定义pad_token_id，则引发值错误
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    
    # 将标签中可能存在的-100值替换为pad_token_id
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    # 找到每行中最后一个非pad_token_id的位置
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    
    # 获取解码器的起始标记
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    
    # 将先前的输出标记向右移动一位，并将第一个位置替换为解码器的起始标记
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    # 返回更新后的先前输出标记
    return prev_output_tokens
# 从transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding复制并修改为MBartLearnedPositionalEmbedding
class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # MBart设置了如果指定了padding_idx，则将embedding ids偏移2，并相应调整num_embeddings。其他模型没有这个hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


# 从transformers.models.bart.modeling_bart.BartAttention复制并修改为MBartAttention
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

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 从transformers.models.bart.modeling_bart.BartFlashAttention2复制并修改为MBartFlashAttention2
class MBartFlashAttention2(MBartAttention):
    """
    # MBart flash attention module. This module inherits from `MBartAttention` as the weights of the module stays
    # untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    # flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    # 初始化函数，继承自父类，并调用父类的初始化函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 根据 Flash Attention 版本的不同，设置是否使用顶部左对齐的掩码
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 重塑张量形状的函数
    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    # Flash Attention 的前向传播函数
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入的隐藏状态包含至少一个填充标记，则首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为 `(batch_size, seq_len)` 的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 之前的 QK^T 的缩放。默认为 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦 Flash Attention for RoCm 升级到 2.1，删除 `query_length != 1` 检查。有关详细信息，请参阅 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        if attention_mask is not None:
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
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制而来
    # 用于处理输入数据，根据注意力掩码获取未填充数据的索引、当前序列长度和批次中的最大序列长度
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 重新组织键层数据，根据未填充数据的索引
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 重新组织值层数据，根据未填充数据的索引
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 如果查询长度等于键值序列长度
        if query_length == kv_seq_len:
            # 重新组织查询层数据，根据未填充数据的索引
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为1
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            # 生成一个序列长度为批次大小的张量
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 假设左填充，根据查询长度截取注意力掩码
            attention_mask = attention_mask[:, -query_length:]
            # 处理未填充输入数据
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的查询层、键层、值层、查询索引、当前序列长度元组、最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义了一个字典，将字符串映射到不同的注意力类
MBART_ATTENTION_CLASSES = {
    "eager": MBartAttention,
    "flash_attention_2": MBartFlashAttention2,
}

# 定义了一个 MBartEncoderLayer 类，继承自 nn.Module
class MBartEncoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化 self_attn 层，根据配置选择不同的注意力类
        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
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

    # 定义了 forward 方法，用于前向传播
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
        # 保存输入的 hidden_states 作为残差连接的一部分
        residual = hidden_states
        # 对 hidden_states 进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self-attention 层处理 hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接和处理后的 hidden_states 相加
        hidden_states = residual + hidden_states

        # 保存当前 hidden_states 作为残差连接的一部分
        residual = hidden_states
        # 对 hidden_states 进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用第二个全连接层处理 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接和处理后的 hidden_states 相加
        hidden_states = residual + hidden_states

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对 hidden_states ��行截断处理
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组，包含处理后的 hidden_states
        outputs = (hidden_states,)

        # 如果需要返回 attentions tensors
        if output_attentions:
            # 将注意力权重添加到输出元组中
            outputs += (attn_weights,)

        return outputs
class MBartDecoderLayer(nn.Module):
    # MBart 解码器层的类定义
    def __init__(self, config: MBartConfig):
        # 初始化函数，接受一个 MBartConfig 类型的参数
        super().__init__()
        # 调用父类的初始化函数

        # 设置嵌入维度为模型维度
        self.embed_dim = config.d_model

        # 创建自注意力机制对象
        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 设置 dropout
        self.dropout = config.dropout
        # 设置激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout
        self.activation_dropout = config.activation_dropout

        # 创建自注意力机制层归一化对象
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建编码器注意力机制对象
        self.encoder_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 创建编码器注意力机制层归一化对象
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 创建全连接层2
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 创建最终层归一化对象
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
    ):
        # 前向传播函数，接受多个参数

# 从 transformers.models.bart.modeling_bart.BartClassificationHead 复制并将 Bart 改为 MBart
class MBartClassificationHead(nn.Module):
    # 用于句子级分类任务的头部类定义
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        # 初始化函数，接受输入维度、内部维度、类别数和池化器 dropout 参数
        super().__init__()
        # 调用父类的初始化函数

        # 创建线性层
        self.dense = nn.Linear(input_dim, inner_dim)
        # 创建 dropout 层
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 创建输出投影层
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，接受隐藏状态张量，返回张量

class MBartPreTrainedModel(PreTrainedModel):
    # MBart 预训练模型类定义
    config_class = MBartConfig
    # 配置类为 MBartConfig
    base_model_prefix = "model"
    # 基础模型前缀为 "model"
    supports_gradient_checkpointing = True
    # 支持梯度检查点
    _no_split_modules = ["MBartDecoderLayer", "MBartAttention"]
    # 不分割的模块列表包括 "MBartDecoderLayer" 和 "MBartAttention"
    _supports_flash_attn_2 = True
    # 支持 Flash Attention 2
    # 初始化模型参数的权重
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 返回虚拟输入，用于模型推理
    @property
    def dummy_inputs(self):
        # 获取填充标记
        pad_token = self.config.pad_token_id
        # 创建虚拟输入张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        # 返回虚拟输入字典
        return dummy_inputs
# MBART_START_DOCSTRING 是一个包含模型文档字符串的原始字符串，用于描述 MBart 模型的继承关系和参数说明
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

# MBART_GENERATION_EXAMPLE 是一个包含模型生成示例的原始字符串，展示了如何使用 MBart 模型进行翻译和填充掩码
MBART_GENERATION_EXAMPLE = r"""
    Translation example:

    ```python
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

    ```python
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

# MBART_INPUTS_DOCSTRING 是一个空字符串，用于描述 MBart 模型的输入参数
MBART_INPUTS_DOCSTRING = r"""
"""

# MBartEncoder 类是 MBart 模型的编码器类，继承自 MBartPreTrainedModel 类
class MBartEncoder(MBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MBartEncoderLayer`].

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化函数，接受配置和嵌入标记作为参数
    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化函数
        super().__init__(config)

        # 设置丢弃率和编码器层丢弃率
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 获取嵌入维度
        embed_dim = config.d_model
        # 获取填充索引
        self.padding_idx = config.pad_token_id
        # 获取最大源位置
        self.max_source_positions = config.max_position_embeddings
        # 设置嵌入缩放
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 创建嵌入标记
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 如果传入了嵌入标记，则使用传入的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建位置嵌入
        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 创建编码器层列表
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 检查是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 创建嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        # 创建编码器层的 LayerNorm
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 用于向后兼容梯度检查点
    def _backward_compatibility_gradient_checkpointing(self):
        # 覆盖以不从配置中删除属性
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

    # 前向传播函数
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
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层间 dropout 概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 设置嵌入缩放因子
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 创建嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 如果传入了额外的嵌入层，则使用传入的嵌入层
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建位置嵌入
        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # 创建多层解码器
        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 根据配置决定是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 创建嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        # 创建解码器的 LayerNorm
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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



@add_start_docstrings(
    "The bare MBART Model outputting raw hidden-states without any specific head on top.",
    MBART_START_DOCSTRING,
)
class MBartModel(MBartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MBartConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器
        self.encoder = MBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()
    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        # 将输入嵌入层设置为给定值
        self.shared = value
        # 更新编码器和解码器的嵌入层
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 返回编码器
    def get_encoder(self):
        return self.encoder

    # 返回解码器
    def get_decoder(self):
        return self.decoder

    # 绑定权重
    def _tie_weights(self):
        # 如果配置中指定绑定词嵌入层
        if self.config.tie_word_embeddings:
            # 绑定编码器和解码器的嵌入层权重
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.get_input_embeddings())
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())

    # 前向传播函数
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
# 添加模型文档字符串，描述了 MBART 模型的用途和细节
@add_start_docstrings(
    "The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models.",
    MBART_START_DOCSTRING,
)
# 定义 MBartForConditionalGeneration 类，继承自 MBartPreTrainedModel
class MBartForConditionalGeneration(MBartPreTrainedModel):
    # 指定基础模型前缀
    base_model_prefix = "model"
    # 在加载模型时忽略的键
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 共享权重的键
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受 MBartConfig 类型的参数
    def __init__(self, config: MBartConfig):
        super().__init__(config)
        # 创建 MBartModel 对象
        self.model = MBartModel(config)
        # 注册缓冲区，用于存储最终 logits 的偏置
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建线性层 lm_head，用于语言建模
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整最终 logits 的偏置
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型前向方法的文档字符串
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向方法的结束文档字符串
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        # 输入的 token ID，数据类型为 LongTensor，默认为 None
        input_ids: torch.LongTensor = None,
        # 注意力掩码，数据类型为可选的 Tensor，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器的输入 token ID，数据类型为可选的 LongTensor，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，数据类型为可选的 LongTensor，默认为 None
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，数据类型为可选的 Tensor，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，数据类型为可选的 Tensor，默认为 None
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，数据类型为可选的 Tensor，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，数据类型为可选的元组，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 过去的键值对，数据类型为可选的元组，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 输入的嵌入向量，数据类型为可选的 FloatTensor，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入的嵌入向量，数据类型为可选的 FloatTensor，默认为 None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，数据类型为可选的 LongTensor，默认为 None
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存，数据类型为可选的布尔值，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，数据类型为可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，数据类型为可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，数据类型为可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,
        ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        # 设置返回字典，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                # 如果提供了标签，则将 use_cache 设置为 False，并发出警告
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # 如果未提供解码器输入，将标签右移一个位置作为解码器输入
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        # 使用模型进行前向传播
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
        # 计算语言模型的输出
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # 如果提供了标签，计算掩码语言建模损失
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则返回输出
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 Seq2SeqLMOutput 对象
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
    ):
        # 如果过去的键值对不为空，则根据过去的长度截取解码器输入的 ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能用于调试）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 定义一个带有序列分类/头部的 MBart 模型，例如用于 GLUE 任务
class MBartForSequenceClassification(MBartPreTrainedModel):
    # 定义需要共享权重的键
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]

    def __init__(self, config: MBartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建 MBartModel 实例
        self.model = MBartModel(config)
        # 创建用于分类的头部
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
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



# 定义一个带有问题回答任务的 MBart 模型，用于提取性问题回答任务（如 SQuAD）
class MBartForQuestionAnswering(MBartPreTrainedModel):
    # 定义需要共享权重的键
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置分类数为 2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 创建 MBartModel 实例
        self.model = MBartModel(config)
        # 创建用于问题回答的输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
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
    # 从transformers.models.bart.modeling_bart.BartForQuestionAnswering.forward中复制而来的函数定义
    def forward(
        self,
        input_ids: torch.Tensor = None,  # 输入的token IDs张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的token IDs张量，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码张量，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，默认为None
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器的头部掩码张量，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码张量，默认为None
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出列表，默认为None
        start_positions: Optional[torch.LongTensor] = None,  # 起始位置张量，默认为None
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置张量，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入张量，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入张量，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存的布尔值，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量的布尔值，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典的布尔值，默认为None
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制代码，并将Bart->MBart
class MBartDecoderWrapper(MBartPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与[`EncoderDecoderModel`]框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化MBartDecoder对象
        self.decoder = MBartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


# 从transformers.models.bart.modeling_bart.BartForCausalLM复制代码，并将Bart->MBart, facebook/bart-base->facebook/mbart-large-cc25
class MBartForCausalLM(MBartPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 初始化MBartDecoderWrapper对象
        self.model = MBartDecoderWrapper(config)

        # 初始化线性层，用于LM头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
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
    # 为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
        ):
            # 如果模型作为编码器-解码器模型中的解码器使用，则动态创建解码器注意力掩码
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape)

            if past_key_values:
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法已经只传递最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认为旧行为：仅保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]
            # 第一步，decoder_cached_states 为空
            return {
                "input_ids": input_ids,  # encoder_outputs 已定义。不需要 input_ids
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            reordered_past = ()
            for layer_past in past_key_values:
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            return reordered_past
```