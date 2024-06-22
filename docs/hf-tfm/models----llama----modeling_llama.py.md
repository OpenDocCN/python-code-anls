# `.\transformers\models\llama\modeling_llama.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 此代码基于 EleutherAI 的 GPT-NeoX 库和 HuggingFace 公司团队的 GPT-NeoX
# 和此库中使用的 OPT 实现。已经针对与 Meta AI 团队训练的模型相比的轻微架构差异进行了修改。
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以了解特定语言的许可证的管理权限和限制。
""" PyTorch LLaMA 模型。"""
# 导入所需模块
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_fx_available

# 如果支持 Flash Attention 2，导入相关模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 在 FX 图中将 `_prepare_4d_causal_attention_mask` 设置为叶节点函数。
# 这意味着该函数不会被跟踪，并且只会出现在图中作为一个节点。
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档字符串中使用的配置
_CONFIG_FOR_DOC = "LlamaConfig"

# 函数用于获取未填充数据
def _get_unpad_data(attention_mask):
    # 计算批次中的序列长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取非零值的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中的最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度，使用 F.pad 函数在计算结果前补0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # 返回索引、累积序列长度和批中最大的序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
# 警告：调用 `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` 已弃用，并将在 v4.37 中移除。使用 `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask`
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)

# 警告：调用 `transformers.models.llama.modeling_llama._make_causal_mask` 已弃用，并将在 v4.37 中移除。使用 `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask`
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )

# 定义 LlamaRMSNorm 类，继承自 nn.Module
class LlamaRMSNorm(nn.Module):
    # 初始化方法
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # 定义权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义方差的极小值
        self.variance_epsilon = eps
    
    # 前向传播方法
    def forward(self, hidden_states):
        # 获取输入数据的数据类型
        input_dtype = hidden_states.dtype
        # 将输入数据转换为 float32 类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化处理
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回归一化后的隐藏状态
        return self.weight * hidden_states.to(input_dtype)

# 将 LlamaRMSNorm 类添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

# 定义 LlamaRotaryEmbedding 类，继承自 nn.Module
class LlamaRotaryEmbedding(nn.Module):
    # 初始化方法
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 设置维度、最大位置嵌入和基础
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率的倒数，并注册为缓冲
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建余弦和正弦缓存用于 `torch.jit.trace` 的工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存的方法
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但它使用了不同的置换，以获得相同的计算
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册余弦缓存和正弦缓存为缓冲
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        # 定义一个前向传播函数，接受输入 x 和可选的序列长度 seq_len
        # x: [bs, num_attention_heads, seq_len, head_size]
        
        # 如果输入的序列长度大于当前缓存的最大序列长度，重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回余弦缓存和正弦缓存，索引范围为 [0:seq_len]
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 初始化函数，继承自 LlamaRotaryEmbedding，添加了线性缩放功能
        self.scaling_factor = scaling_factor
        调用父类的初始化函数，初始化维度、最大位置嵌入长度、基数和设备
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置余弦和正弦缓存的私有函数
        self.max_seq_len_cached = seq_len
        创建一个序列，用于计算余弦和正弦值，根据缩放因子调整序列
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        计算频率并创建余弦和正弦值
        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但使用不同的排列以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        将余弦缓存注册为缓冲区，并转换为指定的数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        将正弦缓存注册为缓冲区，并转换为指定的数据类型
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 初始化函数，继承自 LlamaRotaryEmbedding，添加了动态 NTK 缩放功能
        self.scaling_factor = scaling_factor
        调用父类的初始化函数，初始化维度、最大位置嵌入长度、基数和设备
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置余弦和正弦缓存的私有函数
        self.max_seq_len_cached = seq_len

        如果序列长度大于最大位置嵌入长度
        base = self.base * (
            (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        ) ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        将频率反转并注册为缓冲区
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        创建一个序列，用于计算余弦和正弦值
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        计算频率并创建余弦和正弦值
        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但使用不同的排列以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        将余弦缓存注册为缓冲区，并转换为指定的数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        将正弦缓存注册为缓冲区，并转换为指定的数据类型
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    将输入的隐藏维度旋转了一半
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    返回将输入的隐藏维度一半旋转的结果
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    将旋转位置嵌入应用到查询和键张量上
    Args:
        q (`torch.Tensor`): The query tensor.  # 接收查询张量
        k (`torch.Tensor`): The key tensor.  # 接收键张量
        cos (`torch.Tensor`): The cosine part of the rotary embedding.  # 接收旋转嵌入的余弦部分
        sin (`torch.Tensor`): The sine part of the rotary embedding.  # 接收旋转嵌入的正弦部分
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.  # 与查询张量和键张量相对应的位置索引，例如，在使用 KV-cache 时可以使用偏移的位置 id
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.  # 'unsqueeze_dim' 参数指定了在哪个维度对 cos[position_ids] 和 sin[position_ids] 进行展开，以便它们可以正确地广播到 q 和 k 的维度。例如，注意到 cos[position_ids] 和 sin[position_ids] 的形状为[batch_size, seq_len, head_dim]。然后，如果 q 和 k 的形状为[batch_size, heads, seq_len, head_dim]，那么设置 unsqueeze_dim=1 使得 cos[position_ids] 和 sin[position_ids] 可以广播到 q 和 k 的形状。类似地，如果 q 和 k 的形状为[batch_size, seq_len, heads, head_dim]，那么设置 unsqueeze_dim=2。

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.  # 包含使用旋转位置嵌入旋转的查询和键张量的元组(torch.Tensor)
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 沿指定维度对 cos[position_ids] 进行展开
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # 沿指定维度对 sin[position_ids] 进行展开
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 计算旋转嵌入后的查询张量
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 计算旋转嵌入后的键张量
    return q_embed, k_embed  # 返回旋转使用旋转位置嵌入后的查询和键张量
class LlamaMLP(nn.Module):
    # 定义 LlamaMLP 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法，接收一个配置参数
        super().__init__()
        # 调用父类的初始化方法
        self.config = config
        # 将配置参数保存在对象中
        self.hidden_size = config.hidden_size
        # 从配置参数中获取隐藏层的大小
        self.intermediate_size = config.intermediate_size
        # 从配置参数中获取中间层的大小
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建线性变换层，用于门控的投影
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建线性变换层，用于向上的投影
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 创建线性变换层，用于向下的投影
        self.act_fn = ACT2FN[config.hidden_act]
        # 从预定义的函数映射中获取激活函数，根据配置参数中的隐藏激活函数类型进行选择

    def forward(self, x):
        # 前向传播方法，接收输入张量 x
        if self.config.pretraining_tp > 1:
            # 如果预训练类型大于1
            slice = self.intermediate_size // self.config.pretraining_tp
            # 计算切片大小
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            # 将门控投影的权重按切片大小分割
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            # 将向上投影的权重按切片大小分割
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            # 将向下投影的权重按切片大小分割

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            # 将输入 x 分别与门控投影的每个切片进行线性变换，并拼接结果
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            # 将输入 x 分别与向上投影的每个切片进行线性变换，并拼接结果

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            # 将门控投影的结果经过激活函数处理后，与向上投影的结果相乘，再按切片大小分割
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            # 将中间状态分别与向下投影的每个切片进行线性变换

            down_proj = sum(down_proj)
            # 对所有向下投影的结果求和
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            # 否则，直接对输入进行门控投影、激活函数处理、向上投影，再相乘，并送入向下投影进行处理

        return down_proj
        # 返回最终的向下投影结果


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 定义 repeat_kv 函数，接收隐藏状态和重复次数，返回重复后的隐藏状态
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # 文档字符串，说明函数的作用
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 获取隐藏状态的形状信息
    if n_rep == 1:
        # 如果重复次数为1，直接返回隐藏状态
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 将隐藏状态的维度扩展，并复制 n_rep 次
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    # 重新组织隐藏状态的形状，使其符合多头注意力的输入格式


class LlamaAttention(nn.Module):
    # 定义 LlamaAttention 类，继承自 nn.Module
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 文档字符串，说明该类实现了《Attention Is All You Need》论文中的多头注意力机制
```  
    # 初始化函数，接受一个 LlamaConfig 对象和一个可选的整数参数 layer_idx
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置对象保存到 self.config 中
        self.config = config
        # 将传入的 layer_idx 参数保存到 self.layer_idx 中
        self.layer_idx = layer_idx
        # 如果 layer_idx 为 None，则发出警告
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 从配置对象中获取注意力机制的相关参数
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # 检查 hidden_size 是否可以被 num_heads 整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            # 若不能整除，则抛出 ValueError 异常
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化 Q、K、V、O 四个投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        # 初始化 RoPE（Relative Positional Encoding）
        self._init_rope()

    # 初始化 RoPE（Relative Positional Encoding）的方法
    def _init_rope(self):
        # 如果配置对象中没有指定 RoPE 缩放参数
        if self.config.rope_scaling is None:
            # 使用 LlamaRotaryEmbedding 初始化 RoPE
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 否则，根据配置中的 RoPE 缩放类型初始化 RoPE
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                # 线性缩放方式
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # 动态缩放方式
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 若未知的 RoPE 缩放类型，则抛出 ValueError 异常
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    # 定义私有方法_shape，用于将输入的张量重塑成指定形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 定义前向传播方法，接收输入的隐藏状态张量和其他可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 设置一个属性来处理 Flash Attention 版本差异，该属性用于判断是否使用顶部左对齐的掩蔽。当 Flash Attention 版本<2.1时，需要使用底部右对齐的掩蔽。参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
        # 注意：在 Flash Attention 版本<2.1时，使用 q_seqlen != k_seqlen（除非 q_seqlen == 1）会产生错误的掩蔽（顶部左）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入隐藏状态至少包含一个填充标记，则首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

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
                注意力的丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 之前的 QK^T 的缩放。默认为 1 / sqrt(head_dim)
        """
        如果不使用 `flash_attn_uses_top_left_mask`:
            causal = self.is_causal
        否则:
            # TODO: 一旦 Flash Attention for RoCm 升级到 2.1，就删除 `query_length != 1` 检查。有关详情，请参阅 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        如果 attention_mask 不为空:
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
        否则:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        返回 attn_output
    # 更新输入数据，根据条件对 query_layer、key_layer、value_layer 进行处理
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度、批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取 key_layer 的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 重新组织 key_layer、value_layer，根据索引进行重新排列
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据不同的 query_length 做不同处理
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 处理 query_length 为 1 的情况
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 处理 query_length 不等于 kv_seq_len 且不等于 1 的情况
            # 进行左填充操作，获取未填充的输入数据
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的结果
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
        # Define a class variable that holds different implementations of LlamaAttention
        LLAMA_ATTENTION_CLASSES = {
            "eager": LlamaAttention,
            "flash_attention_2": LlamaFlashAttention2,
            "sdpa": LlamaSdpaAttention,
        }

class LlamaDecoderLayer(nn.Module):
    # Initialize the LlamaDecoderLayer class
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Set self attention using different implementations of LlamaAttention based on config._attn_implementation
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        # Set multi-layer perceptron
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Forward pass of the LlamaDecoderLayer
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
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
        # 如果传入的kwargs中包含"padding_mask"，则发出警告
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # 保存输入hidden_states，用于后续计算残差
        residual = hidden_states

        # 对输入hidden_states应用输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力机制
        # 调用self_attn方法，返回更新后的hidden_states，self attention权重，当前键-值对
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # 将残差与经过self attention后的hidden_states相加
        hidden_states = residual + hidden_states

        # 全连接层
        # 保存输入hidden_states，用于后续计算残差
        residual = hidden_states
        # 对输入hidden_states应用后attention层归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 对输入hidden_states应用多层感知机（MLP）
        hidden_states = self.mlp(hidden_states)
        # 将残差与经过MLP后的hidden_states相加
        hidden_states = residual + hidden_states

        # 输出
        outputs = (hidden_states,)

        # 如果需要输出attention权重，将self_attn_weights加入到输出中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，将present_key_value加入到输出中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# LLAMA_START_DOCSTRING 是一个包含模型文档字符串的原始字符串常量
LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加开始文档字符串的装饰器，继承于 PreTrainedModel，引用了上述文档字符串
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
# 定义 LlamaPreTrainedModel 类，继承自 PreTrainedModel
class LlamaPreTrainedModel(PreTrainedModel):
    # 模型配置类
    config_class = LlamaConfig
    # 基础模型前缀
    base_model_prefix = "model"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块列表
    _no_split_modules = ["LlamaDecoderLayer"]
    # 跳过键的设备放置
    _skip_keys_device_placement = "past_key_values"
    # 是否支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 是否支持 SDPA
    _supports_sdpa = True
    # 是否支持缓存类
    _supports_cache_class = True

    # 初始化模型权重的函数
    def _init_weights(self, module):
        # 初始化标准差为配置中的 initializer_range
        std = self.config.initializer_range
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置
            if module.bias is not None:
                # 将偏置置零
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引
            if module.padding_idx is not None:
                # 将填充索引对应的权重置零
                module.weight.data[module.padding_idx].zero_()


# LLAMA_INPUTS_DOCSTRING 是一个包含输入文档字符串的原始字符串常量
LLAMA_INPUTS_DOCSTRING = r"""
"""

# 添加开始文档字符串的装饰器，继承于 PreTrainedModel，引用了上述文档字符串
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
# 定义 LlamaModel 类，继承自 LlamaPreTrainedModel
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    # 初始化函数
    def __init__(self, config: LlamaConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 填充索引为配置中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 词汇表大小为配置中的 vocab_size
        self.vocab_size = config.vocab_size

        # 初始化嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 初始化解码器层列表
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 是否使用 SDPA 注意力机制
        self._use_sdpa = config._attn_implementation == "sdpa"
        # 是否使用 Flash Attention 2 注意力机制
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 归一化层
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数，接受多个参数，并返回模型输出结果
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮盖
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 缓存的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入层
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
# 创建一个名为 LlamaForCausalLM 的类，继承自 LlamaPreTrainedModel 类
class LlamaForCausalLM(LlamaPreTrainedModel):
    # 定义 _tied_weights_keys 属性，其值为 ["lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化类实例的方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 LlamaModel 的实例，并赋给 self.model 属性
        self.model = LlamaModel(config)
        # 将 config 对象中的 vocab_size 赋给实例的 vocab_size 属性
        self.vocab_size = config.vocab_size
        # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 config.vocab_size，不使用偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用类实例的 post_init 方法，进行权重初始化和最终处理
        self.post_init()

    # 返回输入嵌入的方法
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回输出嵌入的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器的方法
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器的方法
    def get_decoder(self):
        return self.model

    # 前向传播方法的装饰器，添加模型输入的文档字符串和输出类型信息
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
        # 检查过去的键值是否存在
        if past_key_values is not None:
            # 如果past_key_values是Cache类型的实例
            if isinstance(past_key_values, Cache):
                # 获取cache的长度
                cache_length = past_key_values.get_seq_length()
                # 获取过去处理的token数
                past_length = past_key_values.seen_tokens
                # 获取cache的最大长度
                max_cache_length = past_key_values.get_max_length()
            else:
                # 获取过去处理的token数和cache的长度
                cache_length = past_length = past_key_values[0][0].shape[2]
                # 最大cache长度设为None
                max_cache_length = None

            # 保留未处理的token内容:
            # 1 - 如果attention_mask的长度大于input_ids的长度，则我们处于一种设置下，其中部分输入作为cache的一部分传递(例如当作为输入传递input_embeds时)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果past_length小于input_ids的长度，则input_ids包含所有输入token。我们可以根据past_length丢弃input_ids。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则(past_length >= input_ids.shape[1])，假设input_ids仅包含未处理的token。

            # 如果我们即将超出最大cache长度，我们需要裁剪输入的attention mask。
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 为批次生成动态创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入了`inputs_embeds`，我们只想在第1代步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    # 重新排序缓存中的过去键值，以适应给定的 beam_idx（束索引）
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去状态元组
        reordered_past = ()
        # 遍历每一层的过去键值
        for layer_past in past_key_values:
            # 对每一层的过去状态进行重新排序，并将结果添加到 reordered_past 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去状态元组
        return reordered_past
# 添加文档字符串到类上，描述了LLaMa模型的序列分类头部以及相关操作说明
@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取类别数量
        self.num_labels = config.num_labels
        # 创建LLaMa模型对象
        self.model = LlamaModel(config)
        # 创建分类分数计算线性层
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 前向传播方法，添加了模型输入的文档字符串
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
```