# `.\models\persimmon\modeling_persimmon.py`

```
# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch Persimmon model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_persimmon import PersimmonConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PersimmonConfig"


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Persimmon
class PersimmonRotaryEmbedding(nn.Module):
    """
    Rotary positional embedding for Persimmon model.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Initialize the PersimmonRotaryEmbedding module.

        Args:
            dim (int): Dimensionality of the embedding.
            max_position_embeddings (int): Maximum number of positions to embed.
            base (int): Base value for rotational frequencies.
            device (Optional[torch.device]): Device to store the embeddings.
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Calculate inverse frequencies for positional embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Precompute and store cosine and sine values.

        Args:
            seq_len (int): Length of sequence to compute values for.
            device (torch.device): Device to store the cache tensors.
            dtype (torch.dtype): Data type of the cache tensors.
        """
        # Implementation details for precomputing cosine and sine values
        pass
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置当前对象的最大缓存序列长度
        self.max_seq_len_cached = seq_len
        # 创建一个从 0 到 max_seq_len_cached 的整数张量，并根据设备和数据类型初始化
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率矩阵，torch.outer 实现了外积操作
        freqs = torch.outer(t, self.inv_freq)
        # 将频率矩阵按最后一个维度连接起来，形成长度为 2 倍的频率矩阵
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册缓存的余弦值张量，并将其转换为指定的数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 注册缓存的正弦值张量，并将其转换为指定的数据类型
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果传入的序列长度大于当前缓存的最大序列长度，则重新设置缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回缓存的余弦值和正弦值张量，截取前 seq_len 长度，同时转换为 x 的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从transformers.models.falcon.modeling_falcon.FalconLinearScalingRotaryEmbedding复制并将Falcon更改为Persimmon
class PersimmonLinearScalingRotaryEmbedding(PersimmonRotaryEmbedding):
    """PersimmonRotaryEmbedding扩展了线性缩放。鸣谢Reddit用户/u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 设置线性缩放因子
        self.scaling_factor = scaling_factor
        # 调用父类的初始化方法
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 缓存当前序列长度
        self.max_seq_len_cached = seq_len
        # 生成一个序列t，长度为max_seq_len_cached，在给定设备上，并转换为指定数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        # 对序列t进行缩放，得到频率
        t = t / self.scaling_factor

        # 计算频率矩阵，outer操作后的结果是一个形状为(max_seq_len_cached, dim)的张量
        freqs = torch.outer(t, self.inv_freq)
        # 使用不同的排列方式来计算cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将cos值缓存起来
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将sin值缓存起来
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从transformers.models.falcon.modeling_falcon.FalconDynamicNTKScalingRotaryEmbedding复制并将Falcon更改为Persimmon
class PersimmonDynamicNTKScalingRotaryEmbedding(PersimmonRotaryEmbedding):
    """PersimmonRotaryEmbedding扩展了动态NTK缩放。鸣谢Reddit用户/u/bloc97和/u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 设置动态NTK缩放因子
        self.scaling_factor = scaling_factor
        # 调用父类的初始化方法
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 缓存当前序列长度
        self.max_seq_len_cached = seq_len

        # 如果序列长度超过最大位置嵌入长度，则根据公式计算基础
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            # 计算频率的倒数
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            # 将频率的倒数作为缓冲区注册起来
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 生成一个序列t，长度为max_seq_len_cached，在给定设备上，并转换为指定数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率矩阵，outer操作后的结果是一个形状为(max_seq_len_cached, dim)的张量
        freqs = torch.outer(t, self.inv_freq)
        # 使用不同的排列方式来计算cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将cos值缓存起来
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将sin值缓存起来
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从transformers.models.llama.modeling_llama.rotate_half复制
def rotate_half(x):
    """旋转输入张量一半的隐藏维度。"""
    # 将输入张量的前一半切片为x1
    x1 = x[..., : x.shape[-1] // 2]
    # 将输入张量的后一半切片为x2
    x2 = x[..., x.shape[-1] // 2 :]
    # 将x1取反后与x2拼接在一起，并在最后一个维度上进行连接
    return torch.cat((-x2, x1), dim=-1)
# 从 transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb 复制而来的函数，用于应用旋转位置嵌入
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # 根据位置索引从 cos 和 sin 张量中选择对应的部分，并在指定维度上进行 unsqueeze 操作，以便正确广播到 q 和 k 的维度
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 应用旋转位置嵌入到查询张量 q 上，并返回旋转后的查询和键张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 从 transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXMLP 复制而来的类，重命名为 PersimmonMLP
class PersimmonMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将隐藏大小转换为中间大小
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        # 全连接层，将中间大小转换回隐藏大小
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        # 激活函数，根据配置中的隐藏激活函数选择对应的函数
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # 将输入的隐藏状态通过全连接层 dense_h_to_4h 转换为更高维度
        hidden_states = self.dense_h_to_4h(hidden_states)
        # 应用选择的激活函数
        hidden_states = self.act(hidden_states)
        # 将转换后的高维度状态通过全连接层 dense_4h_to_h 转换回原始隐藏维度
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


# 从 transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXMLP 复制而来的类，重命名为 PersimmonAttention
class PersimmonAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化函数，接受一个 PersimmonConfig 类型的配置和一个可选的层索引 layer_idx
    def __init__(self, config: PersimmonConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        
        # 将传入的配置对象 config 存储到实例变量 self.config 中
        self.config = config
        
        # 将传入的层索引 layer_idx 存储到实例变量 self.layer_idx 中
        self.layer_idx = layer_idx
        
        # 如果未传入层索引，则记录警告信息，提示在使用缓存时可能导致前向调用错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        # 将配置中的隐藏层大小存储到实例变量 self.hidden_size 中
        self.hidden_size = config.hidden_size
        
        # 将配置中的注意力头数存储到实例变量 self.num_heads 中
        self.num_heads = config.num_attention_heads
        
        # 计算每个注意力头的维度，并存储到实例变量 self.head_dim 中
        self.head_dim = self.hidden_size // self.num_heads
        
        # 将配置中的最大位置嵌入数量存储到实例变量 self.max_position_embeddings 中
        self.max_position_embeddings = config.max_position_embeddings
        
        # 将配置中的绳索角度存储到实例变量 self.rope_theta 中
        self.rope_theta = config.rope_theta
        
        # 将配置中的部分旋转因子存储到实例变量 self.partial_rotary_factor 中
        self.partial_rotary_factor = config.partial_rotary_factor
        
        # 设置实例变量 self.is_causal 为 True
        self.is_causal = True
        
        # 检查隐藏层大小是否可以被注意力头数整除，若不能，则抛出 ValueError 异常
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 创建一个线性层，用于计算查询、键、值的线性变换
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        
        # 创建一个线性层，用于将多个注意力头的输出进行线性变换和合并
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        
        # 根据配置中的 qk_layernorm 参数决定是否创建 LayerNorm 层，并存储到 self.q_layernorm 和 self.k_layernorm 中
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
        
        # 创建一个 Dropout 层，用于注意力机制中的 dropout 操作
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # 初始化绳索机制
        self._init_rope()
    # 初始化 RoPE（Rotary Positional Encoding）
    def _init_rope(self):
        # 如果配置中没有指定 RoPE 的缩放方式，则使用 PersimmonRotaryEmbedding 类初始化 RoPE
        if self.config.rope_scaling is None:
            self.rotary_emb = PersimmonRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        # 如果配置中指定了 RoPE 的缩放方式，则根据配置的方式选择合适的 RoPE 初始化方法
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            # 如果缩放方式是线性的，则使用 PersimmonLinearScalingRotaryEmbedding 类初始化 RoPE
            if scaling_type == "linear":
                self.rotary_emb = PersimmonLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            # 如果缩放方式是动态的，则使用 PersimmonDynamicNTKScalingRotaryEmbedding 类初始化 RoPE
            elif scaling_type == "dynamic":
                self.rotary_emb = PersimmonDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            # 如果缩放方式不是线性或动态，则抛出 ValueError 异常
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 从 fused_qkv 张量中拆分出 query、key 和 value，并返回拆分后的张量
    # 这里的 fused_qkv 包含了经过融合的查询、键和值张量
    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        # 将 fused_qkv 重塑为 [batch_size, seq_length, num_heads, 3, head_dim] 的形状
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        # 返回拆分后的 query、key 和 value 张量
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
# PersimmonDecoderLayer 类定义，继承自 nn.Module
class PersimmonDecoderLayer(nn.Module):
    # 初始化函数，接受 PersimmonConfig 对象和层索引作为参数
    def __init__(self, config: PersimmonConfig, layer_idx: int):
        # 调用父类初始化方法
        super().__init__()
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size
        # 初始化 self_attn 属性，使用 PersimmonAttention 类进行自注意力计算
        self.self_attn = PersimmonAttention(config=config, layer_idx=layer_idx)
        # 初始化 mlp 属性，使用 PersimmonMLP 类进行多层感知机计算
        self.mlp = PersimmonMLP(config)
        # 初始化 input_layernorm 属性，使用 nn.LayerNorm 进行输入层归一化
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 post_attention_layernorm 属性，使用 nn.LayerNorm 进行自注意力后归一化
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 属性，使用 nn.Dropout 进行隐藏层 dropout
        self.dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播方法定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        # 记录输入的隐藏状态，用于残差连接
        residual = hidden_states

        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # 残差连接
        hidden_states = residual + hidden_states

        # 全连接层归一化
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 多层感知机
        hidden_states = self.mlp(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # 残差连接
        hidden_states = hidden_states + residual

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果使用缓存
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# 定义一个长字符串，描述 Persimmon 模型的文档字符串，包含继承的 `PreTrainedModel` 的通用方法和 PyTorch 的 `torch.nn.Module` 的子类信息。
PERSIMMON_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PersimmonConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加文档字符串到 `PersimmonPreTrainedModel` 类
@add_start_docstrings(
    "The bare Persimmon Model outputting raw hidden-states without any specific head on top.",
    PERSIMMON_START_DOCSTRING,
)
class PersimmonPreTrainedModel(PreTrainedModel):
    # 指定 PersimmonPreTrainedModel 类的配置类
    config_class = PersimmonConfig
    # 模型基础名称前缀
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["PersimmonDecoderLayer"]
    # 跳过设备位置关键字
    _skip_keys_device_placement = "past_key_values"
    # 支持缓存类
    _supports_cache_class = True

    # 初始化权重的函数，根据模块类型设置不同的初始权重
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# 定义一个空的文档字符串，用于 `PersimmonModel` 类
PERSIMMON_INPUTS_DOCSTRING = r"""
"""

# 添加文档字符串到 `PersimmonModel` 类
@add_start_docstrings(
    "The bare Persimmon Model outputting raw hidden-states without any specific head on top.",
    PERSIMMON_START_DOCSTRING,
)
class PersimmonModel(PersimmonPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PersimmonDecoderLayer`]

    Args:
        config: PersimmonConfig
    """

    # 初始化函数，接受 `PersimmonConfig` 类型的参数 `config`
    def __init__(self, config: PersimmonConfig):
        super().__init__(config)
        # 设置填充索引和词汇表大小
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 初始化词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 初始化解码器层的模块列表
        self.layers = nn.ModuleList(
            [PersimmonDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 初始化最终层归一化
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 关闭梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入词嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens
    # 设置模型的输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 在模型的前向传播方法上添加注释，使用了一个特定的装饰器添加了文档字符串
    @add_start_docstrings_to_model_forward(PERSIMMON_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，类型为 LongTensor
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，类型为可选的 Tensor
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，类型为可选的 LongTensor
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，类型为可选的 FloatTensor 列表
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，类型为可选的 FloatTensor
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，类型为可选的布尔值
# 定义 PersimmonForCausalLM 类，继承自 PersimmonPreTrainedModel 类
class PersimmonForCausalLM(PersimmonPreTrainedModel):
    # 定义权重共享的键值列表
    _tied_weights_keys = ["lm_head.weight"]

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ 复制而来，初始化函数
    def __init__(self, config):
        # 调用父类 PersimmonPreTrainedModel 的初始化函数
        super().__init__(config)
        # 创建 PersimmonModel 类的实例并赋值给 self.model
        self.model = PersimmonModel(config)
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 创建一个线性层，将隐藏状态的大小映射到词汇表大小，并且没有偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用额外的初始化函数，用于初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings 复制而来，获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings 复制而来，设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings 复制而来，获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings 复制而来，设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder 复制而来，设置解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder 复制而来，获取解码器
    def get_decoder(self):
        return self.model

    # 应用装饰器并添加文档字符串，标记 forward 方法的输入说明和返回说明
    @add_start_docstrings_to_model_forward(PERSIMMON_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处定义模型的前向传播逻辑，详细说明由装饰器和注释提供
        pass

    # 定义生成输入的函数，从 transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 准备生成模型输入的逻辑，详细说明由调用方提供
        pass
        ):
            # 如果过去的键值对不为空
            if past_key_values is not None:
                # 如果过去的键值对是 Cache 对象
                if isinstance(past_key_values, Cache):
                    # 获取缓存的序列长度
                    cache_length = past_key_values.get_seq_length()
                    # 获取已见的标记长度
                    past_length = past_key_values.seen_tokens
                    # 获取最大缓存长度
                    max_cache_length = past_key_values.get_max_length()
                else:
                    # 否则，假设 past_key_values 是一个列表，取第一个元素的第一个维度的第三个元素作为缓存长度和已见标记长度
                    cache_length = past_length = past_key_values[0][0].shape[2]
                    max_cache_length = None

                # 保留未处理的标记:
                # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，说明一些输入完全作为缓存传递（例如当作 input_embeds 输入时）
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入标记。我们可以基于 past_length 丢弃 input_ids。
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 只包含未处理的标记。

                # 如果我们即将超过最大缓存长度，我们需要裁剪输入的 attention_mask。
                if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
                ):
                    attention_mask = attention_mask[:, -max_cache_length:]

            position_ids = kwargs.get("position_ids", None)
            # 如果 attention_mask 不为空且 position_ids 为空，则动态创建 position_ids 用于批量生成
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                # 如果存在过去的键值对，则仅保留最后 input_ids.shape[1] 个位置标识
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            # 如果传入了 inputs_embeds，则只在第一代步骤中使用它们
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            # 更新 model_inputs 字典
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                }
            )
            # 返回 model_inputs
            return model_inputs

        @staticmethod
    # 定义一个函数 `_reorder_cache`，用于重新排序缓存数据，以便与beam搜索相关联
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组用于存储重新排序后的缓存数据
        reordered_past = ()
        # 遍历每一层的缓存数据
        for layer_past in past_key_values:
            # 对每一层的每个缓存状态进行重新排序，并将结果添加到元组中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存数据元组
        return reordered_past
# 定义了一个带有顺序分类头部的 Persimmon 转换器模型。
# 
# 这个模型 [`PersimmonForSequenceClassification`] 在顶部使用一个线性层来进行序列分类。
# 
# 它使用最后一个令牌来进行分类，类似于其他因果模型（例如 GPT-2）的做法。
# 
# 由于它在最后一个令牌上进行分类，因此需要知道最后一个令牌的位置。如果配置中定义了 `pad_token_id`，则在每一行中找到不是填充令牌的最后一个令牌。如果没有定义 `pad_token_id`，则直接取每个批次中每行的最后一个值。当传递 `inputs_embeds` 而不是 `input_ids` 时，无法猜测填充令牌，因此也采取相同的策略（取每行的最后一个值）。
```