# `.\models\llama\modeling_llama.py`

```py
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LLaMA model."""

# 引入数学库和警告库
import math
import warnings
# 引入类型提示相关的模块
from typing import List, Optional, Tuple, Union

# 引入PyTorch相关的模块
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
# 引入PyTorch的神经网络模块
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引入各种工具函数和模型输出相关的模块
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# 引入LLaMA配置模块
from .configuration_llama import LlamaConfig

# 检查是否可用新的注意力机制库
if is_flash_attn_2_available():
    # 如果可用，引入相关函数
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档使用的配置名称
_CONFIG_FOR_DOC = "LlamaConfig"

# 辅助函数：获取未填充数据
def _get_unpad_data(attention_mask):
    # 计算每个序列在批次中的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找出attention_mask中为1的位置
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 计算批次中最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # 返回结果
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# LLaMA模型的RMS归一化层
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义方差的小量值
        self.variance_epsilon = eps
    # 定义一个前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 将隐藏状态张量转换为 float32 数据类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算隐藏状态张量每个元素的平方，并沿着最后一个维度求均值，保持维度
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 对隐藏状态张量进行归一化处理，使用倒数平方根公式
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回归一化后的隐藏状态张量乘以权重张量
        return self.weight * hidden_states
# 将 LlamaRMSNorm 类添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

# 定义 LlamaRotaryEmbedding 类，继承自 nn.Module
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率的倒数，用于位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将 inv_freq 注册为不可训练的缓冲区
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 缓存最大序列长度
        self.max_seq_len_cached = max_position_embeddings
        # 创建位置编码的张量 t，并根据缩放因子调整
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        # 计算频率矩阵 freqs，并进行拼接以生成位置嵌入 emb
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将 cos 和 sin 值缓存起来，注册为不可训练的缓冲区
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        # 警告：sin_cached 属性将在 4.39 版本中移除，建议使用 RoPE 的 forward 方法代替
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        # 警告：cos_cached 属性将在 4.39 版本中移除，建议使用 RoPE 的 forward 方法代替
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 扩展 inv_freq 和 position_ids 的维度，以便进行矩阵乘法
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # 在关闭自动混合精度的情况下，计算频率并计算 cos 和 sin
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        # 返回计算得到的 cos 和 sin，转换为与 x 相同的数据类型
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding 扩展，添加了线性缩放。感谢 Reddit 用户 /u/kaiokendev 的贡献。"""
    # 定义一个方法 `forward`，接收输入 `x` 和位置标识 `position_ids`
    def forward(self, x, position_ids):
        # 将位置标识转换为浮点数，并应用缩放因子，以调整位置标识的范围
        position_ids = position_ids.float() / self.scaling_factor
        # 调用父类的 `forward` 方法，传入 `x` 和调整后的位置标识 `position_ids`
        cos, sin = super().forward(x, position_ids)
        # 返回计算得到的余弦和正弦值
        return cos, sin
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # 计算序列的长度，找到最大的位置 ID 并加 1
        seq_len = torch.max(position_ids) + 1
        # 如果序列长度超过了最大位置嵌入的设定值
        if seq_len > self.max_position_embeddings:
            # 计算基础值，考虑动态的 NTK 缩放因子
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            # 计算新的频率倒数张量
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            # 将频率倒数张量注册为缓冲区，以便在模型运行中使用，不会被视为模型的参数
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        # 调用父类的 forward 方法计算余弦和正弦部分
        cos, sin = super().forward(x, position_ids)
        # 返回余弦和正弦部分作为输出
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 将输入张量的一半维度旋转180度
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 拼接旋转后的两部分张量
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    # 在指定的维度上对余弦和正弦部分进行展开
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # 对查询张量和键张量应用旋转位置嵌入
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回旋转后的查询张量和键张量作为结果
    return q_embed, k_embed
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存为实例变量
        self.config = config
        # 从配置对象中获取隐藏层大小并保存为实例变量
        self.hidden_size = config.hidden_size
        # 从配置对象中获取中间层大小并保存为实例变量
        self.intermediate_size = config.intermediate_size
        # 创建一个线性变换层，将隐藏层大小映射到中间层大小，没有偏置项
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性变换层，将隐藏层大小映射到中间层大小，没有偏置项
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性变换层，将中间层大小映射回隐藏层大小，没有偏置项
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 根据配置中的激活函数名称从预定义的映射中获取对应的激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    # 前向传播方法，接受输入张量 x 作为参数
    def forward(self, x):
        # 如果预训练类型大于 1
        if self.config.pretraining_tp > 1:
            # 计算每个分片的大小
            slice = self.intermediate_size // self.config.pretraining_tp
            # 将gate_proj权重分片
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            # 将up_proj权重分片
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            # 将down_proj权重分片
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            # 对输入张量 x 执行多个线性变换，然后拼接在一起，形成 gate_proj 的结果
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            # 对输入张量 x 执行多个线性变换，然后拼接在一起，形成 up_proj 的结果
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            # 对 gate_proj 的结果应用激活函数，并与 up_proj 相乘，然后按照 slice 进行分片
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            # 对每个分片应用 down_proj 的线性变换，然后将结果相加
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # 如果预训练类型不大于 1，直接计算 down_proj 的结果
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        # 返回 down_proj 结果作为前向传播的输出
        return down_proj
# 定义一个函数 repeat_kv，用于复制输入张量的内容。这相当于 torch.repeat_interleave(x, dim=1, repeats=n_rep) 的功能。
# 输入参数 hidden_states 是一个四维张量，表示隐藏状态，维度为(batch, num_key_value_heads, seqlen, head_dim)。
# n_rep 是重复复制的次数。
# 函数返回一个张量，将隐藏状态从 (batch, num_key_value_heads, seqlen, head_dim) 转换为 (batch, num_attention_heads, seqlen, head_dim)。

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # 获取输入张量的维度信息
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # 如果 n_rep 等于 1，则直接返回原始的隐藏状态张量
    if n_rep == 1:
        return hidden_states
    
    # 将隐藏状态张量扩展为新的形状，以便复制内容
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # 将扩展后的张量重新整形为所需的形状，即 (batch, num_attention_heads, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # 初始化 LlamaAttention 类的属性
        self.config = config
        self.layer_idx = layer_idx
        
        # 如果未提供 layer_idx，发出警告，因为在使用缓存时可能导致前向调用错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 设置注意力机制的相关参数
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # 检查 hidden_size 是否可以被 num_heads 整除，否则抛出 ValueError
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化线性变换层，用于查询、键、值和输出的投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # 初始化相关的参数和设置
        self._init_rope()
    # 初始化 RoPE（Rotary Positional Embedding）模块
    def _init_rope(self):
        # 检查是否配置了 RoPE 的缩放参数
        if self.config.rope_scaling is None:
            # 如果未配置缩放参数，则使用默认的 LlamaRotaryEmbedding
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 如果配置了缩放参数，则根据类型选择相应的 RoPE 实现
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                # 使用线性缩放的 RoPE 实现
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # 使用动态 NTK 缩放的 RoPE 实现
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 抛出异常，提示未知的 RoPE 缩放类型
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 前向传播函数定义，接受输入的张量和可选的参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 标志变量用于处理 Flash Attention 版本 2.1 以下的兼容性问题，此版本的 flash_attn 生成左上角对齐的因果蒙版，而本模块需要右下角对齐的默认行为。参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
        # 注意，在 flash_attn<2.1 的情况下，如果 q_seqlen != k_seqlen（除了 q_seqlen == 1 的情况），会产生错误的蒙版（左上角）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # 前向传播函数，用于计算注意力机制
        # hidden_states: 输入的隐藏状态张量
        # attention_mask: 可选的注意力蒙版张量，默认为 None
        # position_ids: 可选的位置 ID 张量，默认为 None
        # past_key_value: 可选的缓存键值对，默认为 None
        # output_attentions: 是否输出注意力权重，默认为 False
        # use_cache: 是否使用缓存，默认为 False
        # cache_position: 可选的缓存位置张量，默认为 None
        # **kwargs: 其他关键字参数

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # 内部函数，执行 Flash Attention 的前向传播
        # query_states: 查询状态张量
        # key_states: 键状态张量
        # value_states: 值状态张量
        # attention_mask: 注意力蒙版张量
        # query_length: 查询长度
        # dropout: dropout 概率，默认为 0.0
        # softmax_scale: softmax 缩放参数，可选
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
        # Determine if the attention mechanism should be causal based on configuration and query length
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the input sequence
        if attention_mask is not None:
            # Retrieve batch size from query states tensor
            batch_size = query_states.shape[0]
            # Unpad input states based on attention mask and query length
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract lengths of effective sequences after unpadding
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform variable-length Flash Attention calculation
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

            # Pad the attention output to match original sequence lengths
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Perform standard Flash Attention calculation without padding
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the final attention output
        return attn_output
    # 定义一个方法来处理无需填充的输入数据，根据输入的query_layer、key_layer、value_layer、attention_mask和query_length参数
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 调用_get_unpad_data函数获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取key_layer的形状信息：批次大小、键值对序列长度、键值头的数量和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据索引重新排列key_layer，以便按第一个轴索引重新组织
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据索引重新排列value_layer，以便按第一个轴索引重新组织
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 如果query_length等于kv_seq_len，则按索引重新排列query_layer，并更新相关变量
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果query_length等于1，则设置max_seqlen_in_batch_q为1，cu_seqlens_q为从0到batch_size+1的整数，indices_q为cu_seqlens_q的前一部分
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个memcpy操作，效率很差。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 如果以上条件都不满足，则假设左填充，并截取attention_mask的后-query_length列
            attention_mask = attention_mask[:, -query_length:]
            # 调用unpad_input函数，根据query_layer和截取后的attention_mask获取unpad后的输入数据
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回更新后的query_layer、key_layer、value_layer、indices_q、cu_seqlens_q和max_seqlen_in_batch_q
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass of the LlamaSdpaAttention module.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position ids. Defaults to None.
            past_key_value (Optional[Cache], optional): The past key value cache. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.
            use_cache (bool, optional): Whether to use caching. Defaults to False.
            cache_position (Optional[torch.LongTensor], optional): The position for caching. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor from the attention layer.
        """
        # Forward pass implementation goes here

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Initialize self attention mechanism based on config's specified implementation
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass of the LlamaDecoderLayer module.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position ids. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): The past key value cache. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to False.
            use_cache (Optional[bool], optional): Whether to use caching. Defaults to False.
            cache_position (Optional[torch.LongTensor], optional): The position for caching. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor from the decoder layer.
        """
        # Forward pass implementation goes here
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # 如果kwargs中包含"padding_mask"，则发出警告，该功能将在v4.37版本中移除
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # 保存输入的残差连接
        residual = hidden_states

        # 对输入的hidden_states进行LayerNorm处理
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention部分
        # 调用self_attn方法进行自注意力计算，得到新的hidden_states、自注意力权重self_attn_weights以及新的缓存present_key_value
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # 加上残差连接
        hidden_states = residual + hidden_states

        # Fully Connected部分
        # 保存新的残差连接
        residual = hidden_states

        # 对新的hidden_states进行LayerNorm处理
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 经过MLP层处理
        hidden_states = self.mlp(hidden_states)

        # 加上残差连接
        hidden_states = residual + hidden_states

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要返回注意力权重，则添加到输出结果中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，则添加present_key_value到输出结果中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
    "Document the inputs the LLAMA model accepts (`model_input_ids`, `attention_mask`, etc.) See the superclass "
    "documentation for more details."
    LLAMA_INPUTS_DOCSTRING,
)
    # 创建一个包含字符串的元组，第一个元素是字符串描述模型的功能，第二个元素是模型文档字符串的起始部分
    (
        "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
        LLAMA_START_DOCSTRING,
    )
# 定义 LlamaForCausalLM 类，继承自 LlamaPreTrainedModel 类
class LlamaForCausalLM(LlamaPreTrainedModel):
    # 定义权重共享的键列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 LlamaModel 实例，传入配置参数
        self.model = LlamaModel(config)
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 创建一个线性层 lm_head，用于预测词汇表中的词
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回输入的嵌入层对象
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入的嵌入层对象
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回输出的嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器对象
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器对象
    def get_decoder(self):
        return self.model
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向传播方法的文档字符串，使用指定的输入文档字符串
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
        cache_position: Optional[torch.LongTensor] = None,
    ):
    # 此方法定义了模型的前向传播过程，接受多个可选参数用于生成预测结果或计算损失

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
    # 准备生成过程的输入，接受多个参数用于生成新的模型输入

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 静态方法，用于重新排序缓存中的过去键值，以便与给定的beam索引匹配
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                # 对每个层级的过去状态进行重新排序，使其与beam索引匹配
            )
        return reordered_past
    # 返回重新排序后的过去键值
"""
The Llama Model transformer with a sequence classification head on top (linear layer).

[`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
@add_start_docstrings(
"""
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
""",
LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        # Initialize Llama model with given configuration
        self.transformer = LlamaModel(config)
        # Linear layer for question-answering output (span start and end logits)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Retrieve input embeddings from the Llama model
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        # Set input embeddings for the Llama model
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
"""
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        # 输入序列的 token IDs，类型为长整型张量，可选参数
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力遮罩，类型为单精度浮点张量，可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        # 位置编码 ID，类型为长整型张量，可选参数
        position_ids: Optional[torch.LongTensor] = None,
        # 过去的键值对，类型为浮点张量列表，可选参数
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入的嵌入张量，类型为单精度浮点张量，可选参数
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 起始位置，类型为长整型张量，可选参数
        start_positions: Optional[torch.LongTensor] = None,
        # 结束位置，类型为长整型张量，可选参数
        end_positions: Optional[torch.LongTensor] = None,
        # 是否输出注意力张量，布尔类型，可选参数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，布尔类型，可选参数
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的结果，布尔类型，可选参数
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
        # 确保返回的字典存在，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 模型处理输入，获取输出
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给问答模型输出层，获取开始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU环境中，需要添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # 忽略超出模型输入长度的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不要求返回字典形式的输出，则按元组形式返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 类型的对象，包含损失、开始和结束位置的 logits，以及隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```