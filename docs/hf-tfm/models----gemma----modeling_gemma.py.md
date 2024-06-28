# `.\models\gemma\modeling_gemma.py`

```
# coding=utf-8
# 版权 2024 Google Inc. HuggingFace Inc. 团队。保留所有权利。
#
#
# 根据 Apache 许可证版本 2.0 使用本文件（"许可证"）;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言下的权限，请参阅许可证。
""" PyTorch Gemma 模型。"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
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
from .configuration_gemma import GemmaConfig

# 如果支持 flash_attn 2.x 版本，则导入相关函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# 当使用 Torch FX 时，使 `_prepare_4d_causal_attention_mask` 成为 FX 图中的叶子节点。
# 这意味着该函数不会被跟踪，只会作为图中的一个节点出现。
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GemmaConfig"


def _get_unpad_data(attention_mask):
    # 计算每个序列的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找出非零位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找出批次中最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    # 定义一个私有方法 `_norm`，用于对输入张量 x 进行归一化处理
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 定义前向传播方法 `forward`，接收输入张量 x
    def forward(self, x):
        # 使用私有方法 `_norm` 对输入张量 x 进行归一化处理，转换为 float 类型
        output = self._norm(x.float())
        
        # 做了一个特定的乘法操作，修改了输出结果 `output` 的值
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # 参考：https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        
        # 将输出结果 `output` 转换为与输入张量 x 相同的数据类型，并返回
        return output.type_as(x)
# 将 GemmaRMSNorm 类添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(GemmaRMSNorm)

# 定义 GemmaRotaryEmbedding 类，继承自 nn.Module
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        
        # 初始化 GemmaRotaryEmbedding 类的参数
        self.dim = dim  # 维度
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入长度
        self.base = base  # 基础值
        self.register_buffer("inv_freq", None, persistent=False)  # 注册非持久化的缓冲区 inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        
        # 如果 inv_freq 为空，则根据公式计算 inv_freq
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        
        # 扩展 inv_freq 和 position_ids
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # 强制将 freqs 的计算结果转换为 float32，因为 bfloat16 在长上下文中会失去精度
        # 参考 https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # 连接 freqs 的 cos 和 sin
            cos = emb.cos()  # 计算余弦值
            sin = emb.sin()  # 计算正弦值
        
        # 返回 cos 和 sin，转换为输入 x 的数据类型
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# 从 transformers.models.llama.modeling_llama.rotate_half 复制并定义 rotate_half 函数
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # 取输入的前一半维度
    x2 = x[..., x.shape[-1] // 2 :]  # 取输入的后一半维度
    return torch.cat((-x2, x1), dim=-1)  # 将 x2 反转后与 x1 连接并返回


# 从 transformers.models.llama.modeling_llama.apply_rotary_pos_emb 复制并定义 apply_rotary_pos_emb 函数
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=torch.Tensor(), unsqueeze_dim=1):
        """
        Apply rotary position embedding to query and key tensors.
    
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
        # Unsqueezing cos and sin along the specified dimension to enable broadcasting
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        
        # Applying rotary position embedding to q and k tensors
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed
class GemmaMLP(nn.Module):
    # GemmaMLP 类定义，继承自 nn.Module
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小
        self.intermediate_size = config.intermediate_size  # 从配置中获取中间层大小
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建线性变换，用于门控投影，输入维度为隐藏层大小，输出维度为中间层大小，无偏置
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建线性变换，用于上游投影，输入维度为隐藏层大小，输出维度为中间层大小，无偏置
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 创建线性变换，用于下游投影，输入维度为中间层大小，输出维度为隐藏层大小，无偏置
        
        # 如果隐藏层激活函数为 None，则发出警告并设置为 'gelu_pytorch_tanh'
        if config.hidden_activation is None:
            logger.warning_once(
                "Gemma's activation function should be approximate GeLU and not exact GeLU.\n"
                "Changing the activation function to `gelu_pytorch_tanh`."
                f"if you want to use the legacy `{config.hidden_act}`, "
                f"edit the `model.config` to set `hidden_activation={config.hidden_act}` "
                "  instead of `hidden_act`. See https://github.com/huggingface/transformers/pull/29402 for more details."
            )
            hidden_activation = "gelu_pytorch_tanh"
        else:
            hidden_activation = config.hidden_activation
        
        # 根据配置选择激活函数
        self.act_fn = ACT2FN[hidden_activation]

    def forward(self, x):
        # 前向传播方法，使用门控投影和上游投影进行激活函数后的加权，再经过下游投影
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 从 transformers.models.llama.modeling_llama.repeat_kv 复制过来的函数
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 将隐藏状态张量在第三维度上进行复制，使得维度从 (batch, num_key_value_heads, slen, head_dim)
    # 变为 (batch, num_key_value_heads * n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 忽略复制部分
    # 初始化函数，接受配置对象和可选的层索引作为参数
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象和层索引保存到实例变量中
        self.config = config
        self.layer_idx = layer_idx
        
        # 如果未传入层索引，则记录警告信息，建议在使用缓存时传入层索引，以避免前向调用中的错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        # 从配置对象中获取并设置注意力机制的参数
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        # 检查隐藏层大小是否能够被注意力头数整除，如果不能，则引发值错误异常
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 初始化线性变换层，用于将输入向量投影到注意力头的维度上
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # 初始化旋转嵌入层，用于引入轮转注意力机制
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    # 定义函数签名，指定函数的输入参数类型和返回类型
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 获取输入张量的形状信息
        bsz, q_len, _ = hidden_states.size()

        # 将隐藏状态投影到查询、键、值空间
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重新组织张量形状以适应多头注意力机制的计算需求，并进行维度转置
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 获取过去的键值信息（如果存在），并应用旋转位置编码到查询和键状态
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        # 如果存在过去的键值信息，则更新键值状态
        if past_key_value is not None:
            # sin 和 cos 是 RoPE 模型特定的参数；cache_position 是用于静态缓存的参数
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 将键值信息根据 num_key_value_groups 的设置进行重复
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力权重，采用缩放点积注意力计算方法
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 如果存在注意力掩码，则应用到注意力权重上
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 将注意力权重进行 softmax 归一化，并转换为与 query_states 相同的数据类型
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # 检查注意力输出的形状是否符合预期
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 调整注意力输出的维度顺序，并使其连续存储
        attn_output = attn_output.transpose(1, 2).contiguous()

        # 将注意力输出重新组织为最终输出的形状，并应用输出投影层
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # 如果不需要输出注意力权重，则将其置为 None
        if not output_attentions:
            attn_weights = None

        # 返回注意力输出、注意力权重（如果需要）、以及更新后的过去键值信息（如果存在）
        return attn_output, attn_weights, past_key_value
# 从 `transformers.models.llama.modeling_llama.LlamaFlashAttention2` 复制并重命名为 `GemmaFlashAttention2`
class GemmaFlashAttention2(GemmaAttention):
    """
    Gemma flash attention module. This module inherits from `GemmaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        # 调用父类构造函数，传递所有参数
        super().__init__(*args, **kwargs)

        # TODO: Flash Attention 版本升级至 2.1 后应该移除此段代码。
        # flash_attn<2.1 生成左上角对齐的因果蒙版，而这里需要的是默认为 flash_attn>=2.1 的右下角对齐。此属性用于处理这种差异。
        # 参考链接：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 注意，对于 flash_attn<2.1，除了 q_seqlen == 1 的情况外，使用 q_seqlen != k_seqlen 会产生错误的蒙版（左上角）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 忽略复制
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
        """
        执行前向传播，调用 Flash Attention 的公共 API，并处理输入中可能存在的填充标记。
        """
        pass

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
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
        # Determine if causal masking is required based on model configuration and query length
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in GemmaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Apply padding-aware operations if attention_mask is provided
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad inputs based on attention_mask and query_length
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract lengths for query and key sequences after unpadding
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform variable-length Flash Attention computation
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

            # Pad the attention output back to the original sequence length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Perform Flash Attention without padding-aware operations
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the computed attention output
        return attn_output
    # 定义一个方法 `_upad_input`，用于处理注意力机制的输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 从注意力掩码中获取未填充数据的索引、当前序列长度和批次内最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取 key_layer 的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 重新组织 key_layer 和 value_layer，根据索引 indices_k 来索引未填充的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据 query_length 的不同情况处理 query_layer
        if query_length == kv_seq_len:
            # 如果 query_length 等于 kv_seq_len，则直接根据 indices_k 索引未填充的数据
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果 query_length 等于 1，则创建一个长度为 batch_size 的序列长度 cu_seqlens_q，并进行索引处理
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy，这是非常糟糕的。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据 query_layer 和 attention_mask 的未填充数据，获取未填充的输入数据
            # 这里假设 `-q_len:` 切片表示左填充
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的 query_layer, key_layer, value_layer 以及相关的索引和长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Gemma
class GemmaSdpaAttention(GemmaAttention):
    """
    Gemma attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GemmaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Ignore copy

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
        Perform forward pass of GemmaSdpaAttention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[torch.Tensor]): Optional tensor of shape (batch_size, sequence_length) 
                containing attention mask for the input sequence. 1.0 for positions that should be attended to, 
                0.0 for masked positions.
            position_ids (Optional[torch.LongTensor]): Optional tensor of shape (batch_size, sequence_length) 
                containing position indices to help distinguish different positions in the input.
            past_key_value (Optional[Cache]): Optional tuple containing cached key and value tensors used for 
                fast decoding.
            output_attentions (bool): Whether to output attentions weights.
            use_cache (bool): Whether to use past key-value states to speed up decoding.
            cache_position (Optional[torch.LongTensor]): Optional tensor of shape (batch_size,) specifying 
                positions in the cache.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        # Implementation details of attention mechanism adapted for Gemma architecture using SDPA API
        pass

GEMMA_ATTENTION_CLASSES = {
    "eager": GemmaAttention,
    "flash_attention_2": GemmaFlashAttention2,
    "sdpa": GemmaSdpaAttention,
}


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with LLAMA->GEMMA,Llama->Gemma
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        """
        Initialize GemmaDecoderLayer.

        Args:
            config (GemmaConfig): Configuration object containing model-specific settings.
            layer_idx (int): Index of the decoder layer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Initialize self-attention mechanism based on configuration
        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        # Initialize MLP layer
        self.mlp = GemmaMLP(config)

        # Layer normalization for input to the layer
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Layer normalization after attention mechanism
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        Perform forward pass of GemmaDecoderLayer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[torch.Tensor]): Optional tensor of shape (batch_size, sequence_length) 
                containing attention mask for the input sequence. 1.0 for positions that should be attended to, 
                0.0 for masked positions.
            position_ids (Optional[torch.LongTensor]): Optional tensor of shape (batch_size, sequence_length) 
                containing position indices to help distinguish different positions in the input.
            past_key_value (Optional[Tuple[torch.Tensor]]): Optional tuple containing cached key and value tensors 
                used for fast decoding.
            output_attentions (Optional[bool]): Whether to output attentions weights.
            use_cache (Optional[bool]): Whether to use past key-value states to speed up decoding.
            cache_position (Optional[torch.LongTensor]): Optional tensor of shape (batch_size,) specifying 
                positions in the cache.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        # Detailed implementation of forward pass through a GemmaDecoderLayer
        pass
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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # 记录输入的隐藏状态，用于残差连接
        residual = hidden_states

        # 应用输入层的 Layer Normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # 调用 self_attn 方法进行自注意力计算，并返回更新后的隐藏状态、注意力权重和更新的键值对
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

        # 将残差连接加回到更新后的隐藏状态中
        hidden_states = residual + hidden_states

        # Fully Connected
        # 记录当前的隐藏状态，用于残差连接
        residual = hidden_states

        # 应用后注意力层的 Layer Normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 应用 MLP 层
        hidden_states = self.mlp(hidden_states)

        # 将残差连接加回到 MLP 输出的隐藏状态中
        hidden_states = residual + hidden_states

        # 构造输出元组，包含更新后的隐藏状态
        outputs = (hidden_states,)

        # 如果需要返回注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，将更新的键值对添加到输出元组中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# 定义模型文档字符串，描述该模型继承自`PreTrainedModel`，指向其超类文档以获取通用方法信息，
# 并说明它也是一个PyTorch的`torch.nn.Module`子类，应当按照PyTorch文档使用。
GEMMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 应用文档字符串到GemmaPreTrainedModel类，描述其作为一个裸的Gemma模型，输出没有特定顶部头部的原始隐藏状态。
# 包含先前定义的模型文档字符串作为参数详细信息的一部分。
@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
class GemmaPreTrainedModel(PreTrainedModel):
    # GemmaPreTrainedModel类使用GemmaConfig作为其配置类
    config_class = GemmaConfig
    # 指定基础模型前缀为'model'
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 指定需要保持在fp32模块中的参数列表
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]
    # 指定不分割的模块列表
    _no_split_modules = ["GemmaDecoderLayer"]
    # 指定跳过设备放置的键列表
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    # 支持flash_attention_2
    _supports_flash_attn_2 = True
    # 支持sdpa
    _supports_sdpa = True
    # 支持cache类
    _supports_cache_class = True

    # 初始化模型权重的私有方法，根据配置中的initializer_range初始化线性层和嵌入层的权重
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

    # 设置缓存的私有方法，根据特定条件初始化模型的缓存
    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        # 如果使用flash_attention_2且缓存类为StaticCache，则抛出异常
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        # 如果最大缓存长度大于模型的causal_mask形状或设备不匹配，则重新生成causal_mask并注册为模型的缓冲区
        if max_cache_len > self.model.causal_mask.shape[-1] or self.device != self.model.causal_mask.device:
            causal_mask = torch.full((max_cache_len, max_cache_len), fill_value=1, device=self.device)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        # 遍历模型的每一层，为其self-attention层的past_key_value属性设置缓存
        for layer in self.model.layers:
            weights = layer.self_attn.o_proj.weight
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype
            )
    # 重置缓存函数，用于清空模型中每个层的注意力机制的过去键值缓存
    def _reset_cache(self):
        # 遍历模型中的每一层
        for layer in self.model.layers:
            # 将每一层的自注意力机制的过去键值缓存置为None，即清空缓存
            layer.self_attn.past_key_value = None
GEMMA_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
# 从transformers.models.llama.modeling_llama.LlamaModel复制而来，将LLAMA->GEMMA，Llama->Gemma
# GemmaModel类定义，用于Transformer解码器，包含config.num_hidden_layers层，每层是GemmaDecoderLayer
# Args:
#     config: GemmaConfig，Gemma配置对象
class GemmaModel(GemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: GemmaConfig
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        # 设置填充索引为config中的pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置词汇表大小为config中的vocab_size
        self.vocab_size = config.vocab_size

        # 创建词嵌入层，参数为词汇表大小、隐藏大小、填充索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 创建包含config.num_hidden_layers个GemmaDecoderLayer对象的层列表
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 创建GemmaRMSNorm对象，参数为隐藏大小、eps值
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 设置梯度检查点为False
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回词嵌入层对象
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置词嵌入层对象的值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 从transformers.models.llama.modeling_llama.LlamaModel复制而来，将LLAMA->GEMMA，Llama->Gemma
    # GemmaModel前向传播函数，忽略复制
    # TODO: 截至torch==2.2.0，在generate中传递给模型的attention_mask是二维的，即使在使用静态KV缓存时也是动态长度。这是torch.compile的问题，
    #  导致每个解码步骤都重新捕获cudagraphs（例如，`recording cudagraph tree for symint key 13`），速度非常慢。
    #  一个解决方法是@torch.compiler.disable，但这会阻止使用fullgraph=True。详细内容请参见https://github.com/huggingface/transformers/pull/29114
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
        cache_position: Optional[torch.LongTensor] = None,
    ):
        pass


# 从transformers.models.llama.modeling_llama.LlamaForCausalLM复制而来，将LLAMA->GEMMA，Llama->Gemma，llama->gemma
# GemmaForCausalLM类定义，继承自GemmaPreTrainedModel
class GemmaForCausalLM(GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数，接受一个config参数
    def __init__(self, config):
        super().__init__(config)
        # 创建一个GemmaModel对象，传入config参数
        self.model = GemmaModel(config)
        # 设置词汇表大小为config中的vocab_size
        self.vocab_size = config.vocab_size
        # 创建一个线性层，将隐藏大小转换为词汇表大小，没有偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()
    # 返回模型的输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置模型的输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回模型的输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置模型的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器模型
    def set_decoder(self, decoder):
        self.model = decoder

    # 返回当前模型
    def get_decoder(self):
        return self.model

    # 忽略复制，该函数装饰了 forward 方法，添加了模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
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
        cache_position: Optional[torch.LongTensor] = None,
    ):
        pass  # 此处定义了模型的前向传播逻辑，具体实现在其它地方

    # 准备生成的输入，在生成阶段用于处理输入的方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        pass  # 此方法用于生成阶段准备输入，具体实现在其它地方

    # 静态方法：重新排序缓存中的过去键值，用于束搜索生成
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# Gemma 模型用于序列分类任务，在预训练模型基础上增加了顶部的线性层用于分类。
# 这里的 [`GemmaForSequenceClassification`] 类似于其他因果模型（如 GPT-2），使用最后一个标记进行分类。
# 由于它在最后一个标记上执行分类，需要知道最后一个标记的位置。如果配置中定义了 `pad_token_id`，则会找到每行中最后一个非填充标记。如果没有定义 `pad_token_id`，则简单地取批处理中每行的最后一个值。
# 当传递 `inputs_embeds` 而不是 `input_ids` 时，由于无法猜测填充标记，它执行相同的操作（取批处理中每行的最后一个值）。

@add_start_docstrings(
    GEMMA_START_DOCSTRING,
)
# 从 transformers.models.llama.modeling_llama.LlamaForSequenceClassification 复制并将 LLAMA 改为 GEMMA，Llama 改为 Gemma
class GemmaForSequenceClassification(GemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GemmaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    # 前向传播函数，接受多种输入参数用于序列分类任务
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