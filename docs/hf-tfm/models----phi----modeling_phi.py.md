# `.\transformers\models\phi\modeling_phi.py`

```py
# 使用UTF-8编码

# 版权信息

# Apache许可证，版本2.0（“许可证”）下的2023 Microsoft和HuggingFace Inc.团队。保留所有权利。
# 除非符合许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“许可证”分发的软件
# 是基于“按原样”基础分发的，不附带任何形式的保证或条件。
# 有关许可证的特定权限，请参阅许可证。

"""PyTorch Phi模型。"""

# 导入依赖库
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入其他模块
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
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
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

# 导入配置文件
from .configuration_phi import PhiConfig

# 判断是否可以使用“flash_attn”模块
if is_flash_attn_2_available():
    # 导入“flash_attn”模块中的函数和类
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 使用Hugging Face的日志工具
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/phi-1"
_CONFIG_FOR_DOC = "PhiConfig"

# 预训练模型列表
PHI_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    # See all Phi models at https://huggingface.co/models?filter=phi
]


# 从transformers.models.llama.modeling_llama._get_unpad_data复制
def _get_unpad_data(attention_mask):
    # 计算batch中每个样本的序列长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到为1的元素的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找到batch中的最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch, # 返回计算结果
    )


# 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding复制，将Llama替换为Phi
class PhiRotaryEmbedding(nn.Module):
    # 初始化函数，用于初始化 Sinusoidal Positional Embedding 模块
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类的初始化函数
        super().__init__()

        # 设置维度、最大位置编码长度和基数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算频率的倒数，用于计算位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 注册为缓冲，使其在模型保存和加载时保持不变
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建余弦和正弦缓存，以便在序列长度改变时使用
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 记录当前缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 生成序列长度的张量
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 计算频率乘以序列长度得到频率张量
        freqs = torch.outer(t, self.inv_freq)
        # 按最后一个维度连接频率张量和其副本，用于计算余弦和正弦值
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册为缓冲，使其在模型保存和加载时保持不变
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 正向传播函数，用于获取余弦和正弦位置编码
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果序列长度超过当前缓存的最大序列长度，则重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回截取后的余弦和正弦缓存，用于位置编码
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从 transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding 复制过来并将 Llama->Phi
class PhiLinearScalingRotaryEmbedding(PhiRotaryEmbedding):
    """扩展了线性缩放的 PhiRotaryEmbedding。感谢 Reddit 用户 /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        调用父类的构造函数，初始化 PhiRotaryEmbedding
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # 与论文中不同，但使用了不同的排列顺序，以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        创建缓存的余弦值和正弦值张量
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从 transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding 复制过来并将 Llama->Phi
class PhiDynamicNTKScalingRotaryEmbedding(PhiRotaryEmbedding):
    """扩展了动态 NTK 缩放的 PhiRotaryEmbedding。感谢 Reddit 用户 /u/bloc97 和 /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        调用父类的构造函数，初始化 PhiRotaryEmbedding
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            创建缓存的逆频率张量
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文中不同，但使用了不同的排列顺序，以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        创建缓存的余弦值和正弦值张量
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从 transformers.models.llama.modeling_llama.rotate_half 复制过来
def rotate_half(x):
    """旋转输入张量一半的隐藏维度。"""
    将输入张量的一半维度进行旋转
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    返回连接后的张量
    return torch.cat((-x2, x1), dim=-1)


# 从 transformers.models.llama.modeling_llama.apply_rotary_pos_emb 复制过来
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.  # 接收查询张量
        k (`torch.Tensor`): The key tensor.    # 接收键张量
        cos (`torch.Tensor`): The cosine part of the rotary embedding.  # 余弦部分的旋转嵌入
        sin (`torch.Tensor`): The sine part of the rotary embedding.    # 正弦部分的旋转嵌入
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.  # 与查询和键张量对应的令牌的位置索引，用于处理KV缓存时传递偏移的位置id
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.  # 'unsqueeze_dim'参数指定沿着哪个维度对cos[position_ids]和sin[position_ids]进行展开，以使它们可以正确地广播到q和k的维度。例如，注意cos[position_ids]和sin[position_ids]的形状是[batch_size, seq_len, head_dim]。然后，如果q和k的形状是[batch_size, heads, seq_len, head_dim]，那么设置unsqueeze_dim=1将cos[position_ids]和sin[position_ids]广播到q和k的形状。类似地，如果q和k的形状是[batch_size, seq_len, heads, head_dim]，则设置unsqueeze_dim=2。
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.  # 返回由使用旋转位置嵌入旋转的查询和键张量组成的元组。
    """
    # 沿着指定的维度对cos[position_ids]和sin[position_ids]进行展开，以便可以正确地广播到q和k的维度
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 应用旋转位置嵌入到查询张量，并得到旋转后的查询张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 应用旋转位置嵌入到键张量，并得到旋转后的键张量
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回旋转后的查询和键张量组成的元组
    return q_embed, k_embed
# 从transformers.models.clip.modeling_clip.CLIPMLP复制而来，将CLIP替换为Phi
class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 从transformers.models.llama.modeling_llama.repeat_kv复制而来，将llama替换为phi
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这相当于torch.repeat_interleave(x, dim=1, repeats=n_rep)。隐藏状态从(batch, num_key_value_heads, seqlen, head_dim)变为(batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PhiAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力"""
    # 初始化函数，接受 PhiConfig 对象和可选的层索引作为参数
    def __init__(self, config: PhiConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 保存传入的配置对象和层索引
        self.config = config
        self.layer_idx = layer_idx
        # 如果未传入层索引，则发出警告
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 从配置对象中获取参数值
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        # 检查隐藏大小是否可以被头数整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化线性层对象
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # 如果配置中指定了 qk_layernorm，则初始化 LayerNorm 层对象
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )

        # ��始化绳索参数
        self._init_rope()
    # 初始化 RoPE（Rotary Positional Embedding）对象
    def _init_rope(self):
        # 如果配置中未指定 RoPE 缩放参数
        if self.config.rope_scaling is None:
            # 创建 PhiRotaryEmbedding 对象
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 获取 RoPE 缩放类型和缩放因子
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            # 根据不同的缩放类型创建不同的 RoPE 对象
            if scaling_type == "linear":
                self.rotary_emb = PhiLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = PhiDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 抛出异常，未知的 RoPE 缩放类型
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
class PhiFlashAttention2(PhiAttention):
    """
    Phi flash attention module. This module inherits from `PhiAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 设置属性以处理 Flash Attention 版本差异
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
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
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
# 定义一个字典，将字符串映射到对应的注意力类
PHI_ATTENTION_CLASSES = {
    "eager": PhiAttention,
    "flash_attention_2": PhiFlashAttention2,
}

# 定义 PhiDecoderLayer 类，继承自 nn.Module
class PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__()
        # 初始化 self_attn 属性为根据配置选择的注意力类的实例
        self.self_attn = PHI_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        # 初始化 mlp 属性为 PhiMLP 类的实例
        self.mlp = PhiMLP(config)
        # 初始化 input_layernorm 属性为 LayerNorm 类的实例
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 resid_dropout 属性为 Dropout 类的实例
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # 保存输入的 hidden_states 作为残差连接的基准
        residual = hidden_states

        # 对输入的 hidden_states 进行 LayerNorm 处理
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # 调用 self_attn 进行自注意力计算
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 对自注意力输出进行残差连接和 Dropout 处理
        attn_outputs = self.resid_dropout(attn_outputs)

        # Feed Forward
        # 对 hidden_states 进行 MLP 处理
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        # 将自注意力输出、MLP 输出和残差连接起来
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将其加入到输出中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，则将 present_key_value 加入到输出中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# 定义一个字符串常量，包含关于 Phi 模型的文档字符串
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

# 为 Phi 模型添加文档字符串
@add_start_docstrings(
    "The bare Phi Model outputting raw hidden-states without any specific head on top.",
    PHI_START_DOCSTRING,
)
class PhiPreTrainedModel(PreTrainedModel):
    # 定义 Phi 模型的配置类
    config_class = PhiConfig
    # 模型的基础名称前缀
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块
    _no_split_modules = ["PhiDecoderLayer"]
    # 跳过设备放置的键
    _skip_keys_device_placement = "past_key_values"
    # 支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 支持缓存类
    _supports_cache_class = True

    # 初始化模型权重
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

# 定义 Phi 模型的输入文档字符串
PHI_INPUTS_DOCSTRING = r"""
"""

# 为 Phi 模型添加文档字符串
@add_start_docstrings(
    "The bare Phi Model outputting raw hidden-states without any specific head on top.",
    PHI_START_DOCSTRING,
)
class PhiModel(PhiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """

    # 初始化 Phi 模型
    def __init__(self, config: PhiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 创建词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        # 创建多层解码器
        self.layers = nn.ModuleList(
            [PhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入的嵌入向量
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入向量
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 在模型前向传播函数中添加文档字符串注释
    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
class PhiForCausalLM(PhiPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__中复制过来的代码，将Llama替换为Phi，bias=False改为bias=True
    def __init__(self, config):
        super().__init__(config)
        self.model = PhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings中复制过来的代码
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings中复制过来的代码
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings中复制过来的代码
    def get_output_embeddings(self):
        return self.lm_head

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings中复制过来的代码
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder中复制过来的代码
    def set_decoder(self, decoder):
        self.model = decoder

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder中复制过来的代码
    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
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
    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation中复制过来的代码
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 如果过去的键值不为空
        if past_key_values is not None:
            # 如果过去的键值是缓存对象
            if isinstance(past_key_values, Cache):
                # 获取缓存序列长度
                cache_length = past_key_values.get_seq_length()
                # 获取已见标记的数量
                past_length = past_key_values.seen_tokens
                # 获取最大缓存长度
                max_cache_length = past_key_values.get_max_length()
            else:
                # 如果过去的键值不是缓存对象，则获取其维度信息
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 保留未处理的标记：
            # 1 - 如果注意力掩码的长度超过了输入标记的长度，则说明一些输入是作为缓存的一部分传递的
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果过去的长度小于输入标记的长度，则输入标记包含所有输入标记。我们可以根据过去的长度丢弃输入标记。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设输入标记只包含未处理的标记。

            # 如果我们即将超过最大缓存长度，则需要裁剪输入注意力掩码。
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建位置标识
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了`inputs_embeds`，我们只想在第一个生成步骤中使用它们
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
    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache中复制
    # 重新排序缓存中的过去键值对
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键值对元组
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 将每一层的过去状态按照beam_idx重新排序，并转移到相同设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 定义 PhiForSequenceClassification 类，带有一个线性层的序列分类头部
# PhiForSequenceClassification 使用最后一个标记进行分类，与其他因果模型（例如 GPT-2）一样
# 由于它在最后一个标记上进行分类，需要知道最后一个标记的位置。如果配置中定义了 pad_token_id，则在每行中找到不是填充标记的最后一个标记
# 如果未定义 pad_token_id，则简单地取每行批次中的最后一个值。当传递 inputs_embeds 而不是 input_ids 时，无法猜测填充标记，因此执行相同操作（取每行批次中的最后一个值）
class PhiForSequenceClassification(PhiPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PhiModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 前向传播函数
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



# 定义 PhiForTokenClassification 类，带有一个线性层的标记分类头部，例如用于命名实体识别（NER）任务
class PhiForTokenClassification(PhiPreTrainedModel):
    # 初始化 Token 分类器模型
    def __init__(self, config: PhiConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建 Phi 模型对象
        self.model = PhiModel(config)
        
        # 根据配置文件中的参数设置分类器的 dropout
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        # 创建一个 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 方法，用于模型的前向传播
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
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给模型进行处理
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
        # 对隐藏状态进行dropout处理
        hidden_states = self.dropout(hidden_states)
        # 使用分类器对隐藏状态进行分类
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )
```