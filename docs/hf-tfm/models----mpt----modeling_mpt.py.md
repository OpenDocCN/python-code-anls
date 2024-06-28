# `.\models\mpt\modeling_mpt.py`

```
# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MPT model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mpt import MptConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mosaicml/mpt-7b"
_CONFIG_FOR_DOC = "MptConfig"

MPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mosaicml/mpt-7b",
    "mosaicml/mpt-7b-storywriter",
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-7b-8k",
    "mosaicml/mpt-7b-8k-instruct",
    "mosaicml/mpt-7b-8k-chat",
    "mosaicml/mpt-30b",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-30b-chat",
    # See all MPT models at https://huggingface.co/models?filter=mpt
]


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8, device=None):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    # 创建一个序列长度为 sequence_length 的张量，元素从 -(sequence_length - 1) 到 -1
    alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32, device=device).view(1, 1, 1, sequence_length)
    
    # 将 num_heads 扩展到最接近的 2 的幂
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    
    # 创建一个长度为 num_heads_power_of_2 的序列，值从 1 到 num_heads_power_of_2
    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.int64, device=device).float()
    
    # 根据 num_heads_power_of_2 调整斜率，使其范围在 [0, alibi_bias_max / 2]
    base = base * (alibi_bias_max / num_heads_power_of_2)
    
    # 计算每个位置的斜率，根据 base 计算 2 的倒数
    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)
    
    # 如果 num_heads_power_of_2 不等于 num_heads，则调整斜率的顺序以匹配 num_heads
    if num_heads_power_of_2 != num_heads:
        slopes = torch.cat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1)[:, :num_heads, ...]
    # 将 alibi 乘以 slopes，假设它们是相同形状的张量，执行逐元素乘法
    alibi = alibi * slopes
    # 压缩张量 alibi 的第一个维度，如果该维度为 1，则去掉该维度
    return alibi.squeeze(0)
# 定义一个多头自注意力模块的类，继承自 nn.Module
class MptAttention(nn.Module):
    """Multi-head self attention.
    Using torch or triton attention implementation enables user to also use additive bias.
    多头自注意力模块，使用 torch 或 triton 实现的注意力机制，允许用户使用附加偏置。
    """

    def __init__(self, config: MptConfig):
        super().__init__()
        # 初始化模块参数
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.n_heads = config.n_heads  # 注意力头的数量
        self.max_seq_length = config.max_seq_len  # 最大序列长度
        self.head_dim = self.hidden_size // self.n_heads  # 每个注意力头的维度
        self.softmax_scale = config.attn_config.softmax_scale  # softmax 缩放因子
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)  # 若未指定，按默认计算

        self.attn_dropout_p = config.attn_config.attn_pdrop  # 注意力 dropout 概率
        # 线性层，用于计算查询、键、值
        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        # 输出投影层，将多头注意力的结果映射回隐藏层大小
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        position_bias: torch.Tensor,  # 位置偏置张量
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值对（可选）
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（可选）
        ):
        ):
            # 获取隐藏状态的批量大小和序列长度
            batch_size, seq_length = hidden_states.shape[:2]

            # 通过权重矩阵Wqkv对隐藏状态进行线性变换，得到混合的查询、键、值状态
            mixed_qkv = self.Wqkv(hidden_states)
            query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
            # 将查询、键、值状态重塑为(batch_size, seq_length, n_heads, head_dim)的形状，并转置以便后续操作
            query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

            # 处理过去的键值对，如果存在，则将当前的键值与过去的连接起来；否则直接使用当前的键值
            if past_key_value is not None:
                if len(past_key_value) != 0:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                past_key_value = (key_states, value_states)
            else:
                past_key_value = (key_states, value_states)

            # 计算注意力分数，通过查询状态与键状态的转置矩阵相乘，并乘以softmax缩放因子
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

            # 如果存在位置偏置，则将注意力分数调整加上位置偏置
            query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]
            if position_bias is not None:
                if len(position_bias.shape) != 3:
                    raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
                key_length = key_states.shape[-2]

                # 根据位置偏置的尺寸调整位置偏置矩阵，并加到注意力分数上
                position_bias_query_index = max(0, position_bias.size(1) - query_length)
                position_bias_key_index = max(0, position_bias.size(2) - key_length)
                position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]
                attention_scores = attention_scores + position_bias

            # 如果存在注意力遮罩，则用一个很小的数填充注意力分数中的遮罩位置
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

            # 对注意力分数应用softmax操作，进行dropout，得到最终的注意力权重
            attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

            # 将注意力权重与值状态相乘，得到上下文状态，并调整维度顺序以便进行最终投影
            context_states = torch.matmul(attn_weights, value_states)
            context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)

            # 通过最终投影得到最终的注意力输出结果，并返回注意力输出、注意力权重和更新后的键值对
            attn_output = self.out_proj(context_states)
            return attn_output, attn_weights, past_key_value
class MptMLP(nn.Module):
    # 定义 MptMLP 类，继承自 nn.Module
    def __init__(self, config: MptConfig):
        # 初始化函数，接收 MptConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化方法

        # 从配置中获取隐藏层大小
        hidden_size = config.hidden_size

        # 定义上投影层，将隐藏状态映射到四倍隐藏大小，不使用偏置
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # 定义激活函数为 GELU，具体参数为"none"
        self.act = nn.GELU(approximate="none")

        # 定义下投影层，将四倍隐藏大小映射回隐藏大小，不使用偏置
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)

        # 从注意力配置中获取隐藏层dropout概率
        self.hidden_dropout = config.attn_config.attn_pdrop

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 前向传播方法
        # 对隐藏状态应用上投影层并使用激活函数
        hidden_states = self.act(self.up_proj(hidden_states))

        # 计算中间输出，通过下投影层
        intermediate_output = self.down_proj(hidden_states)

        # 对中间输出应用dropout，使用预定义的隐藏层dropout概率
        output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)

        # 将dropout后的输出与残差连接
        output = output + residual

        return output


class MptBlock(nn.Module):
    # 定义 MptBlock 类，继承自 nn.Module
    def __init__(self, config: MptConfig):
        # 初始化函数，接收 MptConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化方法

        # 从配置中获取隐藏层大小
        hidden_size = config.hidden_size

        # 定义第一个层归一化层，使用配置中的层归一化epsilon值
        self.norm_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 兼容 Hub 上的权重，将偏置设为 None

        # 设置注意力头数为配置中的值
        self.num_heads = config.n_heads

        # 定义注意力机制层
        self.attn = MptAttention(config)

        # 定义第二个层归一化层，使用配置中的层归一化epsilon值
        self.norm_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 兼容 Hub 上的权重，将偏置设为 None

        # 定义多层感知机
        self.ffn = MptMLP(config)

        # 设置dropout率为配置中的注意力dropout概率
        self.dropout_rate = config.attn_config.attn_pdrop

        # 定义残差注意力dropout层
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 前向传播方法
        # hidden_states: [batch_size, seq_length, hidden_size]

        # 在变换器层的开始进行层归一化
        layernorm_output = self.norm_1(hidden_states)

        # 保存残差连接的初始隐藏状态
        residual = hidden_states

        # 自注意力机制
        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
        )

        # 对注意力输出应用残差注意力dropout，并与初始残差连接
        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        # 再次进行层归一化
        layernorm_output = self.norm_2(hidden_states)

        # 保存残差连接的中间隐藏状态
        residual = hidden_states

        # 应用多层感知机层
        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        # 如果需要缓存，则返回额外的 past_key_value
        if use_cache:
            outputs += (past_key_value,)

        # 如果需要输出注意力权重，则返回额外的 attn_weights
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # 返回输出元组，包含隐藏状态、present、注意力权重


class MptPreTrainedModel(PreTrainedModel):
    # 定义 MptPreTrainedModel 类，继承自 PreTrainedModel
    config_class = MptConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MptBlock"]
    _keys_to_ignore_on_load_missing = [r"lm_head.*."]
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


# 初始化函数，调用父类的初始化方法
def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)



    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 初始化模型权重的函数
def _init_weights(self, module: nn.Module):
    """Initialize the weights."""
    # 如果是线性层，则使用正态分布初始化权重
    if isinstance(module, nn.Linear):
        # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果存在偏置项，则将其初始化为零
        if module.bias is not None:
            module.bias.data.zero_()
    # 如果是嵌入层，则同样使用正态分布初始化权重
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果存在填充索引，则将对应的权重置零
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    # 如果是 LayerNorm 层，则初始化偏置项为零，权重为1
    elif isinstance(module, LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        module.weight.data.fill_(1.0)



    @staticmethod
    def _convert_to_mpt_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Mpt, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


# 将过去的键值对转换为 Mpt 期望的格式的静态方法
@staticmethod
def _convert_to_mpt_cache(
    past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts the cache to the format expected by Mpt, i.e. to tuple(tuple([batch_size * num_heads, ...]))
    """
    # 获取过去键值对的形状信息
    batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
    batch_size_times_num_heads = batch_size * num_heads
    # 将每一层的过去键值对重塑为 Mpt 期望的形状
    return tuple(
        (
            layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
            layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
        )
        for layer_past in past_key_value
    )
# MPT_START_DOCSTRING 是一个长字符串，用来描述这个模型的基本信息和使用说明
MPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MptConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MPT_INPUTS_DOCSTRING 是另一个字符串常量，用来描述模型的输入参数及其用法
MPT_INPUTS_DOCSTRING = r"""
    # 接收输入参数的函数定义，用于处理Transformer模型的输入
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]`
            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
            
            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
            
            Each element of `past_key_values` is a tuple (past_key, past_value):
            - past_key: [batch_size * num_heads, head_dim, kv_length]
            - past_value: [batch_size * num_heads, kv_length, head_dim]
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
            
            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare Mpt Model transformer outputting raw hidden-states without any specific head on top.",
    MPT_START_DOCSTRING,
)
"""
class MptModel(MptPreTrainedModel):
    """
    MPT Model class inheriting from MptPreTrainedModel, initializing the model with given configuration.

    Args:
        config (MptConfig): The configuration class defining model parameters.

    Attributes:
        hidden_size (int): Size of the hidden layers.
        num_heads (int): Number of attention heads.
        wte (nn.Embedding): Word token embeddings.
        blocks (nn.ModuleList): List of transformer blocks.
        norm_f (LayerNorm): Final layer normalization.
        gradient_checkpointing (bool): Flag for gradient checkpointing.

    Methods:
        get_input_embeddings(): Returns the input embeddings.
        build_mpt_alibi_tensor(): Builds alibi tensor for MPT.
        set_input_embeddings(new_embeddings): Sets new input embeddings.
        forward(): Performs forward pass through the model.
    """

    def __init__(self, config: MptConfig):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads

        # Embedding + LN Embedding
        self.wte = nn.Embedding(config.vocab_size, self.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([MptBlock(config) for _ in range(config.n_layers)])

        # Final Layer Norm
        self.norm_f = LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        # backward compatibility with weights on the Hub
        self.norm_f.bias = None

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns:
            nn.Embedding: The input word token embeddings.
        """
        return self.wte

    def build_mpt_alibi_tensor(self, num_heads, sequence_length, alibi_bias_max=8, device=None):
        """
        Builds an alibi tensor for MPT.

        Args:
            num_heads (int): Number of attention heads.
            sequence_length (int): Length of the input sequence.
            alibi_bias_max (int, optional): Maximum bias value for alibi tensor. Defaults to 8.
            device (torch.device, optional): Device to place alibi tensor on. Defaults to None.

        Returns:
            torch.Tensor: Alibi tensor for MPT.
        """
        return build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max, device)

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        """
        Sets new input embeddings.

        Args:
            new_embeddings (torch.Tensor): New input embeddings to be set.
        """
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Performs forward pass through the MPT model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs.
            past_key_values (Tuple[Tuple[torch.Tensor, torch.Tensor], ...], optional): Past key-value states for fast decoding.
            attention_mask (torch.Tensor, optional): Mask to avoid attention on padding tokens.
            inputs_embeds (torch.LongTensor, optional): Optional input embeddings.
            use_cache (bool, optional): Whether to use cached key-value states.
            output_attentions (bool, optional): Whether to output attention weights.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary as output.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Model output including past and cross attentions.
        """
        # Implementation of forward pass is omitted here for brevity
        pass



"""
@add_start_docstrings(
    """
    The MPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MPT_START_DOCSTRING,
)
"""
class MptForCausalLM(MptPreTrainedModel):
    """
    MPT Model for Causal Language Modeling, inheriting from MptPreTrainedModel.

    Args:
        config (MptConfig): The configuration class defining model parameters.

    Attributes:
        transformer (MptModel): The MPT base model transformer.
        lm_head (nn.Linear): Language modeling head.

    Methods:
        get_output_embeddings(): Returns the output embeddings.
        set_output_embeddings(new_embeddings): Sets new output embeddings.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MptConfig):
        super().__init__(config)
        self.transformer = MptModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns:
            nn.Linear: The language modeling head.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        """
        Sets new output embeddings.

        Args:
            new_embeddings (torch.Tensor): New output embeddings to be set.
        """
        self.lm_head = new_embeddings
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        # 如果 past_key_values 不为 None，则仅保留 input_ids 的最后一部分
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 如果传入了 `inputs_embeds`，并且 past_key_values 是 None，则只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，包括 past_key_values、use_cache 和 attention_mask
        model_inputs.update(
            {
                "past_key_values": past_key_values,  # NITS 这里应该是 layer_past 吗？
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Model 的前向传播方法，接受各种输入和参数进行推理和生成。

        Parameters:
        - input_ids (Optional[torch.LongTensor]): 输入的 token IDs.
        - past_key_values (Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]): 用于存储过去的 key 和 value 的元组。
        - attention_mask (Optional[torch.Tensor]): 注意力遮罩，掩盖不需要计算的位置。
        - inputs_embeds (Optional[torch.Tensor]): 如果传入，代表已经嵌入的输入。
        - labels (Optional[torch.Tensor]): 模型的标签，用于计算损失。
        - use_cache (Optional[bool]): 是否使用缓存以加速生成。
        - output_attentions (Optional[bool]): 是否输出注意力权重。
        - output_hidden_states (Optional[bool]): 是否输出隐藏状态。
        - return_dict (Optional[bool]): 是否返回字典格式的输出。

        Returns:
        - 输出字典，包含模型生成的各种输出。
        """
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # Determine whether to return a dictionary of outputs
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through the transformer model
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Extract hidden states from transformer outputs
        hidden_states = transformer_outputs[0]

        # Generate logits from the language model head
        lm_logits = self.lm_head(hidden_states)

        # Initialize loss as None
        loss = None
        # Calculate loss if labels are provided
        if labels is not None:
            # Move labels to the same device as logits for model parallelism
            labels = labels.to(lm_logits.device)
            # Shift logits and labels to align predictions and targets
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the logits and labels to compute loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        # Prepare the output depending on return_dict flag
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # Return structured output using CausalLMOutputWithCrossAttentions class
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        # 创建一个字典，将每个 `layer_past` 的 `device` 映射到对应的 `beam_idx`，确保在每个生成步骤中 `past_key_values` 与正确的 `beam_idx` 匹配
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        # 重新排序 `past`，使得每个 `layer_past` 的数据按照 `device_to_beam_idx` 中的索引顺序重新排列
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        # 返回重新排序后的 `past`，保持与输入 `past` 相同的内存存储结构
        return reordered_past
"""
The MPT Model transformer with a sequence classification head on top (linear layer).

[`MptForSequenceClassification`] uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
@add_start_docstrings(
    """
    MPT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MPT_START_DOCSTRING,
)
class MptForTokenClassification(MptPreTrainedModel):
    def __init__(self, config: MptConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the MPT transformer model with the provided configuration
        self.transformer = MptModel(config)
        
        # Determine the dropout rate for the classifier layer based on the provided configuration
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        
        # Apply dropout regularization to the classifier layer
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Create a linear layer for the classification task with output size as specified in the configuration
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
"""
    # 使用装饰器为 forward 方法添加文档字符串，用于生成代码示例文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 指定文档生成的检查点
        output_type=TokenClassifierOutput,  # 指定输出类型为 TokenClassifierOutput
        config_class=_CONFIG_FOR_DOC,  # 指定配置类用于文档
    )
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，可选的长整型张量
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,  # 过去的键值对，可选的张量元组
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的张量
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入向量，可选的张量
        labels: Optional[torch.Tensor] = None,  # 标签，用于计算序列分类/回归损失的张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选的布尔值
        **deprecated_arguments,  # 其他过时参数
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:  # 返回值可以是损失和输出元组或 TokenClassifierOutput 类型
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 模型进行前向传播
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]  # 获取 Transformer 输出的隐藏状态
        hidden_states = self.dropout(hidden_states)  # 对隐藏状态应用 dropout
        logits = self.classifier(hidden_states)  # 将隐藏状态输入分类器得到 logits

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            # 计算损失，将 logits 和 labels 展平为二维张量
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            # 如果不返回字典，则返回包含 logits 和其他 Transformer 输出的元组
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则创建 TokenClassifierOutput 对象并返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
@add_start_docstrings(
    """
    The MPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MPT_START_DOCSTRING,
)
class MptForQuestionAnswering(MptPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = MptModel(config)  # 使用给定配置初始化 MPT 模型
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # 初始化线性层用于答案起始位置和结束位置的预测

        # Initialize weights and apply final processing
        self.post_init()  # 执行额外的初始化和最终处理步骤

    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 如果 return_dict 为 None，则使用模型配置中的 use_return_dict 设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Transformer 模型进行处理
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Transformer 输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给 QA 输出层获取 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 拆分为起始位置和结束位置的 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除多余的维度，并使得张量连续
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 的维度大于 1，则去除多余的维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的 start/end 位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，并计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失为起始位置损失和结束位置损失的平均值
            total_loss = (start_loss + end_loss) / 2

        # 如果 return_dict 为 False，则返回包含损失和 logits 的元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则返回 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```