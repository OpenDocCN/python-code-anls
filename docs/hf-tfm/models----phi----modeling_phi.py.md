# `.\models\phi\modeling_phi.py`

```py
# coding=utf-8
# 版权 2023 Microsoft 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件将根据“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。

""" PyTorch Phi model. """

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_phi import PhiConfig

# 如果支持 Flash Attention 2，则导入相应函数和模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置名称
_CHECKPOINT_FOR_DOC = "microsoft/phi-1"
_CONFIG_FOR_DOC = "PhiConfig"

# 预训练模型归档列表
PHI_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    # 查看所有 Phi 模型：https://huggingface.co/models?filter=phi
]


# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
# 用于获取去除填充数据的辅助函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从 transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding 复制的类
# 将 Mistral 替换为 Phi，用于实现 PhiRotaryEmbedding 的旋转嵌入类
class PhiRotaryEmbedding(nn.Module):
    # 初始化函数，用于初始化一个位置编码器对象
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类的初始化方法
        super().__init__()

        # 设置对象的维度
        self.dim = dim
        # 设置最大位置编码长度，默认为2048
        self.max_position_embeddings = max_position_embeddings
        # 设置基数，默认为10000
        self.base = base

        # 计算逆频率向量，用于位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将逆频率向量注册为缓冲区，使其可以被PyTorch持久化管理
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建余弦和正弦缓存，以便`torch.jit.trace`方法可以正常工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存的私有方法
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 记录当前缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 创建序列长度张量t，设备为指定设备，数据类型与inv_freq相同
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率张量
        freqs = torch.outer(t, self.inv_freq)
        # 按最后一个维度连接余弦和正弦值，形成位置编码矩阵
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将余弦值注册为缓冲区，并指定数据类型为dtype，使其可以被PyTorch持久化管理
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将正弦值注册为缓冲区，并指定数据类型为dtype，使其可以被PyTorch持久化管理
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播方法，用于位置编码器的前向计算
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果传入的序列长度大于当前缓存的最大序列长度，则重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回当前缓存中的余弦和正弦值，截取到seq_len长度，并将数据类型转换为x的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从transformers.models.falcon.modeling_falcon.FalconLinearScalingRotaryEmbedding复制并将Falcon更改为Phi
class PhiLinearScalingRotaryEmbedding(PhiRotaryEmbedding):
    """PhiRotaryEmbedding扩展了线性缩放。感谢Reddit用户/u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor  # 设置缩放因子
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len  # 设置缓存的最大序列长度
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor  # 缩放t以调整频率

        freqs = torch.outer(t, self.inv_freq)  # 计算频率
        # 与论文不同，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)  # 构造正弦和余弦的缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)  # 注册余弦缓存
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)  # 注册正弦缓存


# 从transformers.models.falcon.modeling_falcon.FalconDynamicNTKScalingRotaryEmbedding复制并将Falcon更改为Phi
class PhiDynamicNTKScalingRotaryEmbedding(PhiRotaryEmbedding):
    """PhiRotaryEmbedding扩展了动态NTK缩放。感谢Reddit用户/u/bloc97和/u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor  # 设置缩放因子
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))  # 计算基础值
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册频率反向

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)  # 计算频率
        # 与论文不同，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)  # 构造正弦和余弦的缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)  # 注册余弦缓存
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)  # 注册正弦缓存


# 从transformers.models.llama.modeling_llama.rotate_half复制
def rotate_half(x):
    """旋转输入张量一半的隐藏维度。"""
    x1 = x[..., : x.shape[-1] // 2]  # 取前一半维度
    x2 = x[..., x.shape[-1] // 2 :]  # 取后一半维度
    return torch.cat((-x2, x1), dim=-1)  # 连接负后半部分和前半部分


# 从transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb复制
# 将给定的旋转位置嵌入应用到查询和键张量上。

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`): 表示与查询和键张量对应的位置索引。
        unsqueeze_dim (`int`, *可选*, 默认为 1):
            'unsqueeze_dim' 参数指定沿其进行展开的维度，以便将 cos[position_ids] 和 sin[position_ids] 广播到 q 和 k 的维度。
            例如，如果 cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]，
            当 q 和 k 的形状为 [batch_size, heads, seq_len, head_dim] 时，设置 unsqueeze_dim=1 使得它们可以正确广播到 q 和 k 的形状。
            同样地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。

    Returns:
        `tuple(torch.Tensor)`: 返回应用了旋转位置嵌入后的查询和键张量。
    """
    # 根据位置索引选择并展开余弦部分
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # 根据位置索引选择并展开正弦部分
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 应用旋转位置嵌入到查询张量上
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 应用旋转位置嵌入到键张量上
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 从 transformers.models.clip.modeling_clip.CLIPMLP 复制，并将 CLIP 替换为 Phi
class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 使用给定的激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一层线性变换
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二层线性变换

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 第一层线性变换
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 第二层线性变换
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 从 transformers.models.llama.modeling_llama.repeat_kv 复制，并将 llama 替换为 phi
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是 torch.repeat_interleave(x, dim=1, repeats=n_rep) 的等效实现。
    将隐藏状态从 (batch, num_key_value_heads, seqlen, head_dim) 扩展为 (batch, num_attention_heads, seqlen, head_dim)。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 在第二个维度上扩展隐藏状态
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 返回调整形状后的隐藏状态张量
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    # 定义 PhiAttention 类，实现多头注意力机制，参考自 'Attention Is All You Need' 论文
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PhiConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # 初始化配置和层索引
        self.config = config
        self.layer_idx = layer_idx
        # 如果未传入 layer_idx，发出警告，因为在使用缓存时可能导致前向调用时的错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 设置注意力机制的丢弃率
        self.attention_dropout = config.attention_dropout
        # 隐藏层大小、注意力头数、每个头的维度
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # 键值头的数量及每组的头数
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 最大位置嵌入数、Rope 参数、部分旋转因子
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        # 是否因果
        self.is_causal = True

        # 检查 hidden_size 是否可以被 num_heads 整除，否则抛出数值错误
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化查询、键、值的线性投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # 初始化密集层
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # 如果配置中要求进行查询和键的 LayerNorm
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            # 初始化查询的 LayerNorm
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            # 初始化键的 LayerNorm
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )

        # 初始化 ROPE（Relative Position Encoding）参数
        self._init_rope()
    def _init_rope(self):
        # 检查配置中是否设置了 RoPE 的缩放参数
        if self.config.rope_scaling is None:
            # 若未设置，则使用 PhiRotaryEmbedding 初始化 RoPE
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 若设置了缩放参数，则根据类型选择不同的 RoPE 初始化方式
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                # 使用 PhiLinearScalingRotaryEmbedding 初始化 RoPE
                self.rotary_emb = PhiLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # 使用 PhiDynamicNTKScalingRotaryEmbedding 初始化 RoPE
                self.rotary_emb = PhiDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 抛出异常，若未知 RoPE 缩放类型
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
# 定义 PhiFlashAttention2 类，继承自 PhiAttention 类。此模块用于实现 Phi flash attention，其中权重与 PhiAttention 相同。
class PhiFlashAttention2(PhiAttention):
    """
    Phi flash attention module. This module inherits from `PhiAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    # 初始化方法，调用父类的初始化方法，并设置一些额外的属性
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 标志变量，用于确定是否使用顶部左对齐的因果掩码，这取决于 flash_attn 的版本是否大于等于 2.1
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward 复制而来
    # 前向传播方法，处理注意力机制的计算
    def _flash_attention_forward(
        self,
        query_states,                # 查询状态的张量
        key_states,                  # 键状态的张量
        value_states,                # 值状态的张量
        attention_mask,              # 注意力掩码，限制注意力计算的范围
        query_length,                # 查询长度
        dropout=0.0,                 # Dropout 比率，默认为 0.0
        softmax_scale=None,          # Softmax 缩放因子，默认为 None
        **kwargs,
    ):
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
        # Determine if causal masking is needed based on `_flash_attn_uses_top_left_mask`
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Check if there are padding tokens in the attention_mask
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input sequences based on attention_mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Separate sequence lengths for queries and keys
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # Maximum sequence lengths in the batch for queries and keys
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Compute attention scores for un-padded inputs
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

            # Pad the attention scores back to the original input sequence length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # If no padding mask, compute attention scores directly
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 定义一个私有方法 `_upad_input`，接受多个输入参数用于处理注意力机制的输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 从注意力掩码中获取未填充数据的索引、当前序列长度和批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取 key_layer 的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 使用索引重排 key_layer，以处理未填充的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 使用索引重排 value_layer，以处理未填充的数据
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据 query_length 的不同情况，处理 query_layer
        if query_length == kv_seq_len:
            # 若 query_length 等于 kv_seq_len，则直接重排 query_layer
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k  # 使用相同的未填充长度信息
            max_seqlen_in_batch_q = max_seqlen_in_batch_k  # 使用相同的最大序列长度信息
            indices_q = indices_k  # 使用相同的索引信息
        elif query_length == 1:
            # 若 query_length 等于 1，则处理成单个长度的序列
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 在设备上创建一个整数张量
            indices_q = cu_seqlens_q[:-1]  # 使用序列长度创建索引
            query_layer = query_layer.squeeze(1)  # 压缩 query_layer 的第一个维度
        else:
            # 否则，根据 query_length 处理未填充的输入数据
            # 注意：此处可能会存在左填充的情况
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的多个数据和元组
        return (
            query_layer,  # 查询层
            key_layer,  # 键层
            value_layer,  # 值层
            indices_q,  # 查询索引
            (cu_seqlens_q, cu_seqlens_k),  # 未填充长度元组
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),  # 批次中最大序列长度元组
        )
class PhiSdpaAttention(PhiAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine if contiguous QKV tensors are required based on Torch version
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    """
    SDPA attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `PhiAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from PhiAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # Placeholder for SDPA-specific forward pass; implementation details are omitted here
        pass

PHI_ATTENTION_CLASSES = {
    "eager": PhiAttention,
    "flash_attention_2": PhiFlashAttention2,
    "sdpa": PhiSdpaAttention,
}


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__()
        # Initialize self-attention mechanism based on configuration
        self.self_attn = PHI_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        # Initialize multi-layer perceptron for the layer
        self.mlp = PhiMLP(config)
        # Layer normalization for input to the layer
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout applied to the residual connection
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        # Placeholder for forward pass through the decoder layer; details are application-specific and omitted here
        pass
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值表示。
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                每个输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.n_positions - 1]`。[什么是位置ID?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多细节，请查看返回的张量中的 `attentions`。
            use_cache (`bool`, *optional*):
                如果设置为 `True`，则返回 `past_key_values` 键值状态，可以用于加速解码（参见 `past_key_values`）。
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                缓存的过去键和值投影状态。
        """

        residual = hidden_states  # 保存输入张量作为残差连接的一部分

        hidden_states = self.input_layernorm(hidden_states)  # 输入张量经过层归一化处理

        # 自注意力机制
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_outputs = self.resid_dropout(attn_outputs)  # 对自注意力输出应用残差dropout

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))  # 经过MLP处理后的前馈隐藏状态，应用残差dropout
        hidden_states = attn_outputs + feed_forward_hidden_states + residual  # 最终的层输出，结合自注意力输出、前馈隐藏状态和残差连接
        outputs = (hidden_states,)  # 输出结果为包含隐藏状态的元组

        if output_attentions:
            outputs += (self_attn_weights,)  # 如果需要返回注意力权重，则将注意力权重添加到输出元组中

        if use_cache:
            outputs += (present_key_value,)  # 如果需要缓存，则将当前的键值状态添加到输出元组中

        return outputs  # 返回最终的输出结果
# 定义文档字符串，描述 PhiPreTrainedModel 类继承自 PreTrainedModel，并指向其通用方法和 PyTorch 模块用法
PHI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PhiConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加装饰器注释，说明 PhiPreTrainedModel 类是一个输出原始隐藏状态的 Phi 模型，无特定的输出层
@add_start_docstrings(
    "The bare Phi Model outputting raw hidden-states without any specific head on top.",
    PHI_START_DOCSTRING,
)
# 定义 PhiPreTrainedModel 类，继承自 PreTrainedModel
class PhiPreTrainedModel(PreTrainedModel):
    config_class = PhiConfig  # 设置配置类为 PhiConfig
    base_model_prefix = "model"  # 基础模型前缀为 "model"
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["PhiDecoderLayer"]  # 不进行模块分割的模块列表，包含 PhiDecoderLayer
    _skip_keys_device_placement = "past_key_values"  # 跳过键设备放置，指定为 "past_key_values"
    _supports_flash_attn_2 = True  # 支持 flash attention 2
    _supports_sdpa = True  # 支持 SDPA（Scaled Dot-Product Attention）
    _supports_cache_class = True  # 支持缓存类

    # 初始化模型权重的方法
    def _init_weights(self, module):
        std = self.config.initializer_range  # 获取初始化范围
        if isinstance(module, nn.Linear):  # 如果是线性层
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化权重为正态分布
            if module.bias is not None:  # 如果存在偏置
                module.bias.data.zero_()  # 初始化偏置为零
        elif isinstance(module, nn.Embedding):  # 如果是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化权重为正态分布
            if module.padding_idx is not None:  # 如果存在填充索引
                module.weight.data[module.padding_idx].zero_()  # 初始化填充索引处权重为零


# 定义输入文档字符串 PHI_INPUTS_DOCSTRING（此处省略了具体内容）
PHI_INPUTS_DOCSTRING = r"""
"""


# 添加装饰器注释，说明 PhiModel 类是一个输出原始隐藏状态的 Phi 模型，无特定的输出层
@add_start_docstrings(
    "The bare Phi Model outputting raw hidden-states without any specific head on top.",
    PHI_START_DOCSTRING,
)
# 定义 PhiModel 类，继承自 PhiPreTrainedModel
class PhiModel(PhiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """
    # 初始化方法，接受一个 PhiConfig 类型的配置对象作为参数
    def __init__(self, config: PhiConfig):
        # 调用父类的初始化方法，传递配置对象作为参数
        super().__init__(config)
        # 设置填充索引为配置对象中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置词汇表大小为配置对象中的 vocab_size
        self.vocab_size = config.vocab_size

        # 创建一个词嵌入层对象，参数为词汇表大小、隐藏层大小和填充索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 创建一个丢弃层对象，丢弃率为配置对象中的 embd_pdrop
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        # 创建多层解码器，每一层使用 PhiDecoderLayer 类初始化，层数由配置对象中的 num_hidden_layers 决定
        self.layers = nn.ModuleList(
            [PhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 创建最终的 LayerNorm 层，大小为配置对象中的 hidden_size，epsilon 参数为配置对象中的 layer_norm_eps
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 根据配置对象中的 _attn_implementation 属性判断是否使用 Flash Attention 2.0
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 根据配置对象中的 _attn_implementation 属性判断是否使用 Self-Dual-Path Attention (SDPA)
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 是否开启梯度检查点，默认为 False
        self.gradient_checkpointing = False
        # 执行后续的初始化和权重设置
        self.post_init()

    # 获取输入词嵌入层对象的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入词嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 根据 PHI_INPUTS_DOCSTRING 给 forward 方法添加文档字符串的装饰器
    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义一个自定义模型类 PhiForCausalLM，继承自 PhiPreTrainedModel 类
class PhiForCausalLM(PhiPreTrainedModel):
    # 定义类变量 _tied_weights_keys，指定需要共享权重的键名列表
    _tied_weights_keys = ["lm_head.weight"]

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ 复制而来，
    # 使用给定的配置初始化对象
    def __init__(self, config):
        # 调用父类 PhiPreTrainedModel 的初始化方法
        super().__init__(config)
        # 使用给定配置初始化 PhiModel 对象，并将其赋给 self.model
        self.model = PhiModel(config)
        # 从配置中获取词汇表大小，并赋值给 self.vocab_size
        self.vocab_size = config.vocab_size
        # 创建一个线性层，用于语言模型的输出，输入维度为 config.hidden_size，输出维度为 config.vocab_size
        # 同时设置 bias=True，表示包含偏置项
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # 调用对象的后初始化方法，用于权重初始化和其他必要的处理
        self.post_init()

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings 复制而来，
    # 返回模型的输入嵌入层对象 self.model.embed_tokens
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings 复制而来，
    # 设置模型的输入嵌入层对象为给定的 value
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings 复制而来，
    # 返回模型的输出嵌入层对象 self.lm_head，用于语言模型的输出
    def get_output_embeddings(self):
        return self.lm_head

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings 复制而来，
    # 设置模型的输出嵌入层对象为新的嵌入层 new_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder 复制而来，
    # 设置模型的解码器部分为给定的 decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # 从 transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder 复制而来，
    # 返回模型的解码器部分 self.model
    def get_decoder(self):
        return self.model

    # 使用装饰器 @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING) 和
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)，
    # 从 transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM.prepare_inputs_for_generation 复制而来，
    # 准备生成过程的输入参数，支持多种输入格式和选项
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
        # 实际的模型前向传播逻辑会在这里定义
        pass

    # 从 transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM.prepare_inputs_for_generation 复制而来，
    # 准备生成过程的输入参数，支持多种输入格式和选项
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 实现输入生成准备的逻辑，具体内容会依赖具体需求和实现
        pass
        ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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
    # 定义一个函数 `_reorder_cache`，用于重新排序缓存数据 `past_key_values`，基于 `beam_idx` 提供的索引
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组 `reordered_past`，用于存储重新排序后的缓存数据
        reordered_past = ()
        # 遍历 `past_key_values` 中的每一层的缓存数据 `layer_past`
        for layer_past in past_key_values:
            # 对于每个 `layer_past` 中的缓存状态 `past_state`，按照 `beam_idx` 提供的索引重新排序并存储
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存数据 `reordered_past`
        return reordered_past
@add_start_docstrings(
    """
    The PhiModel with a sequence classification head on top (linear layer).

    [`PhiForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    PHI_START_DOCSTRING,
)
class PhiForSequenceClassification(PhiPreTrainedModel):
    # PhiForSequenceClassification 类，继承自 PhiPreTrainedModel

    def __init__(self, config):
        super().__init__(config)
        # 调用父类的构造函数初始化模型配置

        self.num_labels = config.num_labels
        # 设置分类标签数目为配置中的 num_labels

        self.model = PhiModel(config)
        # 创建 PhiModel 对象，使用给定的配置参数

        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # 创建一个线性层，用于分类，输入大小为隐藏层大小，输出大小为标签数目，无偏置项

        # Initialize weights and apply final processing
        self.post_init()
        # 执行初始化权重和应用最终处理的方法

    def get_input_embeddings(self):
        return self.model.embed_tokens
        # 返回模型的输入嵌入层对象

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        # 设置模型的输入嵌入层对象为给定的值

    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
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
        # 前向传播方法，接受多种输入参数，包括 input_ids, attention_mask 等



@add_start_docstrings(
    """
    PhiModel with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    PHI_START_DOCSTRING,
)
class PhiForTokenClassification(PhiPreTrainedModel):
    # PhiForTokenClassification 类，继承自 PhiPreTrainedModel
    # 初始化函数，接受一个 PhiConfig 类型的参数 config
    def __init__(self, config: PhiConfig):
        # 调用父类的初始化函数，传入 config 参数
        super().__init__(config)
        # 设置实例变量 num_labels，从 config 参数中获取
        self.num_labels = config.num_labels

        # 使用 config 参数初始化 PhiModel 类的实例，并赋值给 self.model
        self.model = PhiModel(config)

        # 根据 config 参数设置分类器的 dropout 率
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1

        # 使用 nn.Dropout 类初始化 self.dropout，设置 dropout 率为 classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)

        # 使用 nn.Linear 类初始化 self.classifier，设置输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用 self.post_init() 方法，进行权重初始化和最终处理
        # (假设 self.post_init() 方法用于权重初始化和最终处理，具体细节未提供)
        self.post_init()

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
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
        **deprecated_arguments,
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化 return_dict，如果未指定则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型进行前向传播计算
        model_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_states = model_outputs[0]
        # 对隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将处理后的隐藏状态输入分类器，得到预测 logits
        logits = self.classifier(hidden_states)

        # 初始化损失为 None
        loss = None
        # 如果给定了标签，计算损失
        if labels is not None:
            # 将标签移动到对应设备，以支持模型并行计算
            labels = labels.to(logits.device)
            # 获取批次大小和序列长度
            batch_size, seq_length = labels.shape
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        # 如果不需要返回字典形式的结果，则根据情况返回输出
        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的结果，则构造 TokenClassifierOutput 并返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )
```