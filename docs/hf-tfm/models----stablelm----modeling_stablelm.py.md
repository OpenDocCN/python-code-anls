# `.\models\stablelm\modeling_stablelm.py`

```
# coding=utf-8
# 版权 2024 EleutherAI 和 HuggingFace Inc. 团队。保留所有权利。
#
# 这段代码基于 EleutherAI 的 GPT-NeoX 库以及该库中的 GPT-NeoX
# 和 OPT 实现进行了修改，以适应与 Meta AI 团队训练模型时的微小架构差异。
#
# 根据 Apache 许可证 2.0 版本许可，您只能在遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch StableLM 模型。"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_stablelm import StableLmConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "StableLmConfig"


# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制而来
def _get_unpad_data(attention_mask):
    # 计算批次中的每个序列的长度总和
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非零位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最长序列的长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度，并在左侧填充一个零
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从 transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding 复制而来，将 Mistral 替换为 StableLm
class StableLmRotaryEmbedding(nn.Module):
    # 初始化方法，设置 Transformer 的位置编码参数和设备相关信息
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类初始化方法
        super().__init__()

        # 设置维度
        self.dim = dim
        # 设置最大位置编码长度，默认为2048
        self.max_position_embeddings = max_position_embeddings
        # 设置基础参数，默认为10000
        self.base = base

        # 计算频率倒数向量，用于位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将频率倒数向量作为缓冲区注册到当前对象中
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使 `torch.jit.trace` 正常工作，在这里构建缓存的余弦和正弦值
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦值的缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 记录缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 创建从0到最大序列长度的张量，设备和类型与位置编码频率倒数向量相匹配
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率乘以位置的张量
        freqs = torch.outer(t, self.inv_freq)
        
        # 按最后一个维度连接余弦和正弦值张量，用于位置编码
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将余弦值作为缓冲区注册到当前对象中，并转换为指定的数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将正弦值作为缓冲区注册到当前对象中，并转换为指定的数据类型
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播方法，生成位置编码的余弦和正弦值
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # 如果指定了新的序列长度，并且超过了当前缓存的最大序列长度
        if seq_len > self.max_seq_len_cached:
            # 更新余弦和正弦值的缓存
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回当前缓存中的余弦和正弦值，截取到指定的序列长度并转换为指定的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从StableLmRotaryEmbedding类复制，并加入线性缩放旋转嵌入的功能，用于稳定语言模型
class StableLmLinearScalingRotaryEmbedding(StableLmRotaryEmbedding):
    """StableLmRotaryEmbedding扩展，带有线性缩放功能。由Reddit用户/u/kaiokendev贡献"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor  # 初始化线性缩放因子
        super().__init__(dim, max_position_embeddings, base, device)  # 调用父类的初始化方法

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor  # 根据缩放因子调整序列

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)  # 缓存余弦值
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)  # 缓存正弦值


# 从StableLmRotaryEmbedding类复制，并加入动态NTK缩放旋转嵌入的功能
class StableLmDynamicNTKScalingRotaryEmbedding(StableLmRotaryEmbedding):
    """StableLmRotaryEmbedding扩展，带有动态NTK缩放功能。由Reddit用户/u/bloc97和/u/emozilla贡献"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor  # 初始化动态缩放因子
        super().__init__(dim, max_position_embeddings, base, device)  # 调用父类的初始化方法

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            # 根据序列长度动态计算基础值
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # 缓存频率的倒数

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)  # 缓存余弦值
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)  # 缓存正弦值


# 从transformers.models.llama.modeling_llama.rotate_half复制
def rotate_half(x):
    """旋转输入的一半隐藏维度。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb复制
# 将旋转位置嵌入应用到查询和键张量上
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`):
            与查询和键张量对应的位置索引。例如，当与KV缓存一起使用时，可以传递偏移的位置ID。
        unsqueeze_dim (`int`, *可选*, 默认为 1):
            指定沿着哪个维度展开 cos[position_ids] 和 sin[position_ids]，以便它们可以正确地广播到 q 和 k 的维度。
            例如，如果 cos[position_ids] 和 sin[position_ids] 的形状是 [batch_size, seq_len, head_dim]，
            当 q 和 k 的形状是 [batch_size, heads, seq_len, head_dim] 时，设置 unsqueeze_dim=1 可以使 cos[position_ids]
            和 sin[position_ids] 能够广播到 q 和 k 的形状。类似地，如果 q 和 k 的形状是 [batch_size, seq_len, heads, head_dim]，
            则设置 unsqueeze_dim=2。
    Returns:
        `tuple(torch.Tensor)`: 旋转使用旋转位置嵌入后的查询和键张量。
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 根据位置索引展开余弦部分
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # 根据位置索引展开正弦部分
    q_embed = (q * cos) + (rotate_half(q) * sin)      # 应用旋转位置嵌入到查询张量
    k_embed = (k * cos) + (rotate_half(k) * sin)      # 应用旋转位置嵌入到键张量
    return q_embed, k_embed


# 从 transformers.models.mistral.modeling_mistral.MistralMLP 复制并修改为 StableLmMLP
class StableLmMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 定义门控投影层
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # 定义上投影层
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # 定义下投影层
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数根据配置选择

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # 前向传播方法，应用门控投影和上投影


# 从 transformers.models.llama.modeling_llama.repeat_kv 复制
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这相当于 torch.repeat_interleave(x, dim=1, repeats=n_rep)。将隐藏状态从 (batch, num_key_value_heads, seqlen, head_dim)
    重复为 (batch, num_attention_heads, seqlen, head_dim)。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states  # 如果重复次数为 1，直接返回隐藏状态
    # 将 hidden_states 的形状从 [batch, num_key_value_heads, n_rep, slen, head_dim]
    # 扩展为 [batch, num_key_value_heads, n_rep, slen, head_dim]，其中:
    # - batch 是批次大小
    # - num_key_value_heads 是键值头的数量
    # - n_rep 是重复次数
    # - slen 是序列长度
    # - head_dim 是头的维度
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 将 hidden_states 重新调整为形状 [batch, num_key_value_heads * n_rep, slen, head_dim]
    # 返回调整后的 hidden_states
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    # 定义一个名为 StableLmAttention 的类，表示稳定的语言模型注意力机制，参考 'Attention Is All You Need' 论文中的多头注意力机制

    def __init__(self, config: StableLmConfig, layer_idx: Optional[int] = None):
        # 初始化函数，接收一个配置对象 config 和一个可选的层索引 layer_idx
        super().__init__()
        self.config = config  # 保存传入的配置对象
        self.layer_idx = layer_idx  # 保存传入的层索引，如果未提供则发出警告

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小
        self.num_heads = config.num_attention_heads  # 从配置中获取注意力头的数量
        self.head_dim = self.hidden_size // self.num_heads  # 计算每个注意力头的维度
        self.num_key_value_heads = config.num_key_value_heads  # 从配置中获取键值头的数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 计算键值头的分组数
        self.max_position_embeddings = config.max_position_embeddings  # 从配置中获取最大位置嵌入数
        self.rope_theta = config.rope_theta  # 从配置中获取绳索θ参数
        self.partial_rotary_factor = config.partial_rotary_factor  # 从配置中获取部分旋转因子
        self.is_causal = True  # 设置是否是因果关系（causal）的注意力机制

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 创建线性层用于查询（Q）、键（K）、值（V）的投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_qkv_bias)
        
        # 创建输出投影层
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 创建注意力机制中的dropout层
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        # 初始化稳定语言模型特有的旋转平面（rope）参数
        self._init_rope()

    # 从 transformers.models.persimmon.modeling_persimmon.PersimmonAttention._init_rope 复制而来，用于初始化稳定语言模型注意力机制中的旋转平面参数
    # 初始化 RoPE（Rotary Positional Embedding）组件的方法
    def _init_rope(self):
        # 如果配置中未指定 RoPE 的缩放设置
        if self.config.rope_scaling is None:
            # 使用稳定的线性 RoPE 嵌入进行初始化
            self.rotary_emb = StableLmRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),  # 计算 RoPE 嵌入的维度
                max_position_embeddings=self.max_position_embeddings,  # 最大位置编码长度
                base=self.rope_theta,  # RoPE 的基础角度
            )
        else:
            # 如果配置中指定了 RoPE 的缩放设置
            scaling_type = self.config.rope_scaling["type"]  # 获取缩放类型
            scaling_factor = self.config.rope_scaling["factor"]  # 获取缩放因子
            # 根据不同的缩放类型选择合适的 RoPE 嵌入进行初始化
            if scaling_type == "linear":
                self.rotary_emb = StableLmLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),  # 计算 RoPE 嵌入的维度
                    max_position_embeddings=self.max_position_embeddings,  # 最大位置编码长度
                    scaling_factor=scaling_factor,  # 线性缩放因子
                    base=self.rope_theta,  # RoPE 的基础角度
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = StableLmDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),  # 计算 RoPE 嵌入的维度
                    max_position_embeddings=self.max_position_embeddings,  # 最大位置编码长度
                    scaling_factor=scaling_factor,  # 动态缩放因子
                    base=self.rope_theta,  # RoPE 的基础角度
                )
            else:
                # 如果未知的 RoPE 缩放类型，抛出异常
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 前向传播方法，用于模型的前向计算
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量（可选）
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID 张量（可选）
        past_key_value: Optional[Cache] = None,  # 缓存的键值对（可选）
        output_attentions: bool = False,  # 是否输出注意力权重（默认为 False）
        use_cache: bool = False,  # 是否使用缓存（默认为 False）

        # 方法的输入参数说明完毕
class StableLmSdpaAttention(StableLmAttention):
    # 继承自 StableLmAttention 的稳定SDPA注意力模块

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # 前向传播函数，接受输入隐藏状态、注意力掩码、位置ID、过去键值、是否输出注意力、是否使用缓存

class StableLmFlashAttention2(StableLmAttention):
    """
    StableLM flash attention module. This module inherits from `StableLmAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    # StableLM闪电注意力模块2，继承自 StableLmAttention，模块的权重保持不变。
    # 唯一需要更改的是前向传播，需要正确调用闪电注意力的公共API，并处理输入中可能包含的填充标记。

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        # 初始化函数，接受任意数量的位置参数和关键字参数

        super().__init__(*args, **kwargs)
        # 调用父类的初始化函数

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        # 如果闪电注意力版本小于2.1，则使用顶部左对齐的因果掩码；大于等于2.1版本，默认使用底部右对齐的掩码。该属性用于处理这种差异。

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # 前向传播函数，接受输入隐藏状态、注意力掩码、位置ID、过去键值、是否输出注意力、是否使用缓存，以及其他关键字参数

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # 闪电注意力的前向传播函数，接受查询状态、键状态、值状态、注意力掩码、查询长度、dropout比率、softmax缩放参数
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
        # Determine if the attention mechanism should be causal based on the configuration
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary workaround for RoCm version 2.1 and above until query_length == 1 issue is resolved
            causal = self.is_causal and query_length != 1

        # Check if there are padding tokens in the sequence to handle
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input sequences based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract lengths for the current batch
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Apply variable-length Flash attention mechanism
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

            # Pad the attention output to match the original input dimensions
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Apply Flash attention mechanism without padding handling
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the final attention output
        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 定义一个私有方法，用于处理输入数据，根据给定的参数进行调整
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取不包含填充的数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头数、头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据索引重新排列键层数据，以去除填充
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据索引重新排列值层数据，以去除填充
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据查询长度调整查询层数据
        if query_length == kv_seq_len:
            # 如果查询长度与键值序列长度相同，直接根据索引重新排列查询层数据
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，进行相应的处理
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个memcpy操作，性能较差。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，假设存在左填充，根据查询长度调整注意力掩码，然后调用unpad_input函数处理查询层数据
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回调整后的查询层、键层、值层数据以及相关的索引和序列长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义一个全局常量，包含了不同注意力机制类与其对应的实现类
ATTENTION_CLASSES = {
    "eager": StableLmAttention,
    "sdpa": StableLmSdpaAttention,
    "flash_attention_2": StableLmFlashAttention2,
}

# 定义了一个稳定语言模型的解码器层类
class StableLmDecoderLayer(nn.Module):
    def __init__(self, config: StableLmConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小
        # 初始化自注意力机制，根据配置选择不同的实现类
        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        self.mlp = StableLmMLP(config)  # 初始化多层感知机
        # 初始化输入层归一化层，使用配置中的层归一化 epsilon 参数
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力后归一化层，同样使用配置中的层归一化 epsilon 参数
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)  # 初始化 dropout 模块

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

        # 保存输入的原始状态作为残差连接的一部分
        residual = hidden_states

        # 输入层的 LayerNormalization
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

        # 添加残差连接
        hidden_states = residual + hidden_states

        # 全连接层的 LayerNormalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP（多层感知机）层
        hidden_states = self.mlp(hidden_states)

        # 应用 dropout
        hidden_states = self.dropout(hidden_states)

        # 添加残差连接
        hidden_states = hidden_states + residual

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# STABLELM_START_DOCSTRING 变量，包含多行字符串，用于描述 StableLmPreTrainedModel 类的说明文档
STABLELM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`StableLmConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# add_start_docstrings 装饰器，添加文档字符串到 StableLmPreTrainedModel 类
@add_start_docstrings(
    "The bare StableLm Model outputting raw hidden-states without any specific head on top.",
    STABLELM_START_DOCSTRING,
)
# StableLmPreTrainedModel 类，继承自 PreTrainedModel 类
class StableLmPreTrainedModel(PreTrainedModel):
    # 指定配置类
    config_class = StableLmConfig
    # 模型基础名称前缀
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["StableLmDecoderLayer"]
    # 跳过设备放置的键
    _skip_keys_device_placement = "past_key_values"
    # 支持闪光注意力 2
    _supports_flash_attn_2 = True
    # 支持缓存类
    _supports_cache_class = True
    # 支持 SDPA（Sparse Dense Parallel Attention）
    _supports_sdpa = True

    # 初始化权重方法，根据模块类型初始化权重
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


# STABLELM_INPUTS_DOCSTRING 变量，目前未赋值，预期用于描述输入参数的文档字符串
STABLELM_INPUTS_DOCSTRING = r"""
"""

# add_start_docstrings 装饰器，添加文档字符串到 StableLmModel 类
@add_start_docstrings(
    "The bare StableLm Model outputting raw hidden-states without any specific head on top.",
    STABLELM_START_DOCSTRING,
)
# StableLmModel 类，继承自 StableLmPreTrainedModel 类
class StableLmModel(StableLmPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`StableLmDecoderLayer`]

    Args:
        config: StableLmConfig
    """

    # 初始化方法，接受一个 StableLmConfig 类型的参数 config
    def __init__(self, config: StableLmConfig):
        super().__init__(config)
        # 设定填充索引
        self.padding_idx = config.pad_token_id
        # 设定词汇表大小
        self.vocab_size = config.vocab_size

        # 创建词嵌入层，用于将词汇索引转换为向量表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 创建多层 Transformer 解码器层的列表
        self.layers = nn.ModuleList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 创建 LayerNorm 层，用于层归一化
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 设置注意力实现类型
        self._attn_implementation = config._attn_implementation
        # 是否支持梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入的嵌入向量表
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入向量表
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 在模型前向传播方法上添加文档字符串，使用给定的文档字符串常量
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，类型为长整型张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选的张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，可选的长整型张量
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，可选的浮点张量列表
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，可选的浮点张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选的布尔值
        ):
# Copied from transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM with PERSIMMON->STABLELM,Persimmon->StableLm
class StableLmForCausalLM(StableLmPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with LLAMA->STABLELM,Llama->StableLm
    def __init__(self, config):
        super().__init__(config)
        # 实例化 StableLmModel，使用给定的配置
        self.model = StableLmModel(config)
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 创建线性层，用于语言模型的输出预测
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        # 返回模型的输入嵌入层
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入层
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        # 返回语言模型的输出嵌入层
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型的输出嵌入层
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        # 设置解码器模型
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        # 返回当前使用的解码器模型
        return self.model

    @add_start_docstrings_to_model_forward(STABLELM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
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
        # 稳定语言模型的前向传播函数，根据给定参数生成预测输出
        pass

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 准备生成模型输入的函数，根据给定参数组织输入
        pass
        ):
        # 如果传入的过去键值不为空
        if past_key_values is not None:
            # 如果过去键值是一个 Cache 对象
            if isinstance(past_key_values, Cache):
                # 获取缓存中序列长度
                cache_length = past_key_values.get_seq_length()
                # 获取已见过的令牌数目
                past_length = past_key_values.seen_tokens
                # 获取缓存中的最大长度
                max_cache_length = past_key_values.get_max_length()
            else:
                # 如果 past_key_values 不是 Cache 对象，则使用默认值
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 保留未处理的令牌：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则说明有部分输入仅作为缓存的一部分传递
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入令牌。我们可以基于 past_length
            #    丢弃 input_ids。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则 (past_length >= input_ids.shape[1])，假设 input_ids 只包含未处理的令牌。

            # 如果即将超出最大缓存长度，需要裁剪输入的 attention_mask。
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        # 如果存在 attention_mask 但不存在 position_ids，则动态创建 position_ids 以用于批次生成
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在过去的键值，将 position_ids 裁剪到与 input_ids 相同的长度
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入了 inputs_embeds，则仅在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，包括 position_ids、past_key_values、use_cache、attention_mask
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        # 返回 model_inputs 作为模型的输入
        return model_inputs
    # 重新排序缓存中的过去键值对，使其适应新的beam索引顺序
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组，用于存储重新排序后的过去键值对
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 对于每个过去状态，根据beam_idx重新排序，选择对应设备上的索引
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
@add_start_docstrings(
    """
    The StableLm transformer with a sequence classification head on top (linear layer).

    [`StableLmForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    STABLELM_START_DOCSTRING,
)
# 定义一个新的类 `StableLmForSequenceClassification`，继承自 `StableLmPreTrainedModel` 类
# 该类包含了一个用于序列分类的线性层在其顶部的 `StableLm` 变换器
class StableLmForSequenceClassification(StableLmPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数，并传入配置参数 `config`
        super().__init__(config)
        # 初始化 `num_labels` 属性，表示分类的类别数量
        self.num_labels = config.num_labels
        # 创建 `StableLmModel` 模型实例，并保存在 `self.model` 属性中
        self.model = StableLmModel(config)
        # 创建一个线性层 `score`，用于将 `hidden_size` 的输出映射到 `num_labels` 的空间
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回模型的输入嵌入层 `embed_tokens`
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置模型的输入嵌入层 `embed_tokens` 的值为 `value`
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(STABLELM_INPUTS_DOCSTRING)
    # 定义模型的前向传播函数 `forward`
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

        # 前向传播函数接受多个输入参数，包括 `input_ids`, `attention_mask`, `position_ids`, `past_key_values`,
        # `inputs_embeds`, `labels`, `use_cache`, `output_attentions`, `output_hidden_states`, `return_dict`
        # 它计算模型的输出并返回结果，根据 `STABLELM_INPUTS_DOCSTRING` 的文档字符串进行注释
        pass
```