# `.\models\starcoder2\modeling_starcoder2.py`

```
# 设置文件编码为 UTF-8
# 版权声明和版权信息，说明本代码基于 EleutherAI 的 GPT-NeoX 库，已经进行了修改以适应 Meta AI 团队训练的模型的架构差异
# 依照 Apache License, Version 2.0 授权许可，除非符合许可协议，否则不得使用本文件
# 获取许可协议的副本，请访问 http://www.apache.org/licenses/LICENSE-2.0
#
# 该代码定义了 PyTorch Starcoder2 模型

import inspect  # 导入 inspect 模块用于获取对象信息
import math  # 导入 math 模块提供的数学函数
import warnings  # 导入 warnings 模块用于处理警告
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
import torch.utils.checkpoint  # 导入 PyTorch 中用于实现checkpoint的模块
from torch import nn  # 导入 PyTorch 中的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入 PyTorch 中的损失函数

from ...activations import ACT2FN  # 导入激活函数 ACT2FN
from ...cache_utils import Cache, DynamicCache  # 导入缓存工具类
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa  # 导入处理注意力掩码的函数
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast  # 导入模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型类
from ...utils import (  # 导入工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_starcoder2 import Starcoder2Config  # 导入 Stacoder2 的配置类


if is_flash_attn_2_available():
    # 如果可用 flash attention 2，则导入相关函数和模块
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    # 检查 flash attention 是否支持窗口大小参数
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 文档中的配置信息
_CONFIG_FOR_DOC = "Starcoder2Config"


# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
# 根据 attention_mask 获取非填充数据
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)  # 计算批次中每个序列的长度总和
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()  # 找出非填充位置的索引
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # 获取批次中最长的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))  # 计算序列长度的累积和
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从 transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding 复制的类
# 定义 Starcoder2 的旋转嵌入类
class Starcoder2RotaryEmbedding(nn.Module):
    # 初始化函数，用于初始化一个位置编码器对象
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类的初始化方法
        super().__init__()

        # 设置对象的维度、最大位置嵌入数量和基数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算位置编码中的频率倒数，使用设备上的浮点运算
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        
        # 将频率倒数作为缓冲区注册到对象中，不持久化
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使 `torch.jit.trace` 正常工作，在这里构建余弦和正弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存的函数
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置缓存的最大序列长度
        self.max_seq_len_cached = seq_len

        # 生成一个从0到最大序列长度的张量，使用与频率倒数相同的设备和数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率矩阵
        freqs = torch.outer(t, self.inv_freq)
        
        # 按照论文的描述，拼接余弦和正弦的矩阵，以在位置编码中使用
        emb = torch.cat((freqs, freqs), dim=-1)

        # 将余弦和正弦矩阵注册为对象的缓冲区，使用指定的数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播函数，接受输入张量 x 和可选的序列长度 seq_len
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # 如果传入的序列长度大于当前缓存的最大序列长度，则重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回当前缓存中的余弦和正弦值，截取到指定的序列长度，使用输入张量的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从transformers.models.llama.modeling_llama.rotate_half复制而来
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 取输入张量的后一半维度内容作为x1
    x1 = x[..., : x.shape[-1] // 2]
    # 取输入张量的前一半维度内容作为x2
    x2 = x[..., x.shape[-1] // 2 :]
    # 返回将输入张量的后一半维度内容反向排列、加上前一半维度内容的张量拼接结果
    return torch.cat((-x2, x1), dim=-1)


# 从transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb复制而来
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
    # 根据位置ID从cos和sin中取出对应的值，并在指定维度上进行unsqueeze操作
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 计算经过旋转位置嵌入后的查询和键张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Starcoder2MLP(nn.Module):
    def __init__(self, config: Starcoder2Config):
        super().__init__()
        embed_dim = config.hidden_size
        # 定义线性变换层c_fc，输入维度为embed_dim，输出维度为config.intermediate_size
        self.c_fc = nn.Linear(embed_dim, config.intermediate_size, bias=config.use_bias)
        # 定义线性变换层c_proj，输入维度为config.intermediate_size，输出维度为embed_dim
        self.c_proj = nn.Linear(config.intermediate_size, embed_dim, bias=config.use_bias)
        # 激活函数，根据配置选择不同的激活函数
        self.act = ACT2FN[config.hidden_act]
        # 残差连接中的dropout概率
        self.residual_dropout = config.residual_dropout

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 线性变换c_fc
        hidden_states = self.c_fc(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 线性变换c_proj
        hidden_states = self.c_proj(hidden_states)
        # 应用dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        return hidden_states


# 从transformers.models.llama.modeling_llama.repeat_kv复制而来
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 略
    """
    This function replicates the behavior of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    It transforms hidden states from shape (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim) by repeating along the specified dimension.
    """
    # 获取输入张量的形状参数
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 如果重复次数为1，直接返回原始隐藏状态张量
    if n_rep == 1:
        return hidden_states
    # 在第二个维度上添加一个新维度，并在该维度上扩展以复制隐藏状态
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 将张量重新形状为所需的形状：(batch, num_attention_heads, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
# 定义了一个名为 Starcoder2Attention 的 PyTorch 模型类，用于实现多头注意力机制。
# 该类继承自 nn.Module 类。
class Starcoder2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    # 初始化方法，接受 Starcoder2Config 类型的配置参数和一个可选的层索引参数 layer_idx
    def __init__(self, config: Starcoder2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config  # 存储传入的配置对象
        self.layer_idx = layer_idx  # 存储传入的层索引，可选参数

        # 如果未提供层索引，发出警告
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 初始化模型需要的各种参数和属性
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.use_bias = config.use_bias
        self.is_causal = True  # 固定为 True
        self.attention_dropout = config.attention_dropout
        self.residual_dropout = config.residual_dropout

        # 检查 hidden_size 是否可以被 num_heads 整除，否则抛出 ValueError
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 定义线性变换层，用于生成查询、键、值以及输出
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.use_bias)

        # 初始化旋转嵌入层，用于增强注意力机制的表达能力
        self.rotary_emb = Starcoder2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # 前向传播方法，接受输入的 hidden_states 和一些可选的参数，返回经过注意力机制处理后的输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
# 从 transformers.models.mistral.modeling_mistral.MistralFlashAttention2 复制并修改为 Starcoder2
class Starcoder2FlashAttention2(Starcoder2Attention):
    """
    Starcoder2 flash attention module. This module inherits from `Starcoder2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    """
    # flash attention and deal with padding tokens in case the input contains any of them.
    """
        flash attention and deal with padding tokens in case the input contains any of them.
        """
    
        # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
        # 继承父类构造函数，初始化 FlashAttention2 对象
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
            # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
            # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
            # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
            # 根据 Flash Attention 的版本设置是否使用顶部左对齐的掩码，影响注意力计算中的掩码生成
            self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
        # Ignore copy
        # 执行前向传播
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        # 使用 Flash Attention 进行前向传播
        def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None,
            use_sliding_windows=False,
    # 定义一个方法 _upad_input，接受多个输入参数：query_layer, key_layer, value_layer, attention_mask, query_length
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 从 key_layer 的形状中获取 batch_size, kv_seq_len, num_heads, head_dim 四个变量
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 如果 kv_seq_len 不等于 attention_mask 的最后一个维度大小
        # 则重新创建适当的 padding mask，通过切片在正确的位置进行调整
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 调用 _get_unpad_data 方法获取索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 将 key_layer 重塑为形状为 (batch_size * kv_seq_len, num_heads, head_dim) 的张量，并根据 indices_k 进行索引操作
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        # 将 value_layer 重塑为形状为 (batch_size * kv_seq_len, num_heads, head_dim) 的张量，并根据 indices_k 进行索引操作
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        # 如果 query_length 等于 kv_seq_len
        if query_length == kv_seq_len:
            # 将 query_layer 重塑为形状为 (batch_size * kv_seq_len, num_heads, head_dim) 的张量，并根据 indices_k 进行索引操作
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果 query_length 等于 1
        elif query_length == 1:
            # 设置 max_seqlen_in_batch_q 为 1
            max_seqlen_in_batch_q = 1
            # 创建 cu_seqlens_q 张量，包含从 0 到 batch_size 的整数，数据类型为 torch.int32，存储在 query_layer 的设备上
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy，非常糟糕。
            # 设置 indices_q 为 cu_seqlens_q 的前 n-1 项
            indices_q = cu_seqlens_q[:-1]
            # 压缩 query_layer 的第一个维度
            query_layer = query_layer.squeeze(1)
        else:
            # 使用 attention_mask 的 -query_length: 切片假定左填充，调整 attention_mask
            attention_mask = attention_mask[:, -query_length:]
            # 调用 unpad_input 方法处理 query_layer 和调整后的 attention_mask，返回解压后的输入
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的 query_layer, key_layer, value_layer, indices_q, cu_seqlens, max_seqlen_in_batch
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# Starcoder2SdpaAttention 类定义，继承自 Starcoder2Attention 类
class Starcoder2SdpaAttention(Starcoder2Attention):
    """
    Starcoder2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Starcoder2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # forward 方法重写，定义了 attention 模块的前向传播过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):

# STARCODER2_ATTENTION_CLASSES 是一个字典，用于存储不同 attention 实现的类名及其对应的类对象
STARCODER2_ATTENTION_CLASSES = {
    "eager": Starcoder2Attention,
    "flash_attention_2": Starcoder2FlashAttention2,
    "sdpa": Starcoder2SdpaAttention,  # 将 Starcoder2SdpaAttention 类添加到字典中的 sdpa 键下
}

# Starcoder2DecoderLayer 类定义，继承自 nn.Module
class Starcoder2DecoderLayer(nn.Module):
    # 构造函数，初始化模型参数
    def __init__(self, config: Starcoder2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 初始化 self attention 模块，根据 config._attn_implementation 决定使用哪个具体的 attention 类
        self.self_attn = STARCODER2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # 初始化 MLP 模块
        self.mlp = Starcoder2MLP(config)

        # 初始化输入层归一化层
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        
        # 初始化 attention 后归一化层
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

    # forward 方法重写，定义了解码器层的前向传播过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 如果参数中包含 "padding_mask"，发出警告信息，提醒使用 "attention_mask" 替代
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): 注意力遮罩张量，形状为 `(batch, sequence_length)`，
                其中填充元素用 0 表示。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 更多细节。
            use_cache (`bool`, *optional*):
                如果设置为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码（参见 `past_key_values`）。
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
        """

        # 保存输入张量作为残差连接的基准
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

        # 残差连接和层归一化
        hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # 残差连接和输出
        hidden_states = residual + hidden_states

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要返回注意力权重，则添加到输出中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，则添加当前键值状态到输出中
        if use_cache:
            outputs += (present_key_value,)

        # 返回最终输出
        return outputs
# STARCODER2_START_DOCSTRING 是一个字符串，包含了关于 Starcoder2 模型的文档字符串，描述了模型的继承关系和参数说明
STARCODER2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Starcoder2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# add_start_docstrings 是一个装饰器函数，用来为函数或类添加文档字符串注释
@add_start_docstrings(
    "The bare Starcoder2 Model outputting raw hidden-states without any specific head on top.",  # 描述该类的主要功能
    STARCODER2_START_DOCSTRING,  # 引用上面定义的 STARCODER2_START_DOCSTRING
)
# Starcoder2PreTrainedModel 类，继承自 PreTrainedModel，表示 Starcoder2 模型的基本预训练模型
class Starcoder2PreTrainedModel(PreTrainedModel):
    config_class = Starcoder2Config  # 设置模型的配置类为 Starcoder2Config
    base_model_prefix = "model"  # 设置模型的基础模型前缀为 "model"
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["Starcoder2DecoderLayer"]  # 列出不需要分割的模块名
    _skip_keys_device_placement = "past_key_values"  # 设备放置时跳过的键名
    _supports_flash_attn_2 = True  # 支持 Flash Attention 2
    _supports_sdpa = True  # 支持 Scaled Dot-Product Attention (SDPA)
    _supports_cache_class = True  # 支持缓存类

    # 初始化权重函数，根据模块类型进行权重初始化
    def _init_weights(self, module):
        std = self.config.initializer_range  # 获取配置中的初始化范围
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化线性层的权重
            if module.bias is not None:
                module.bias.data.zero_()  # 初始化线性层的偏置
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化嵌入层的权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 将填充索引处的权重置零

# STARCODER2_INPUTS_DOCSTRING 是一个空字符串，可能用于后续定义输入文档字符串时的引用
STARCODER2_INPUTS_DOCSTRING = r"""
"""

# add_start_docstrings 是一个装饰器函数，用来为函数或类添加文档字符串注释
@add_start_docstrings(
    "The bare Starcoder2 Model outputting raw hidden-states without any specific head on top.",  # 描述该类的主要功能
    STARCODER2_START_DOCSTRING,  # 引用上面定义的 STARCODER2_START_DOCSTRING
)
# Starcoder2Model 类，继承自 Starcoder2PreTrainedModel，表示 Starcoder2 模型的具体实现
class Starcoder2Model(Starcoder2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Starcoder2DecoderLayer`]

    Args:
        config: Starcoder2Config
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config: Starcoder2Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置填充索引为配置对象中的填充标记 ID
        self.padding_idx = config.pad_token_id
        # 设置词汇表大小为配置对象中的词汇表大小
        self.vocab_size = config.vocab_size

        # 创建一个词嵌入层对象，使用配置对象中的参数：词汇表大小、隐藏层大小、填充索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 设置嵌入层的 dropout 率为配置对象中的嵌入层 dropout 率
        self.embedding_dropout = config.embedding_dropout
        # 创建一个由多个解码层组成的层对象列表，每个解码层由配置对象和层索引创建
        self.layers = nn.ModuleList(
            [Starcoder2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 设置注意力机制的实现方式为配置对象中的注意力实现方式
        self._attn_implementation = config._attn_implementation
        # 创建一个 LayerNorm 层，用于归一化隐藏层输出，使用配置对象中的隐藏层大小和归一化 epsilon
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        # 禁用梯度检查点
        self.gradient_checkpointing = False
        # 调用初始化后处理方法，用于初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 重写的 forward 方法，用于模型的前向传播
    @add_start_docstrings_to_model_forward(STARCODER2_INPUTS_DOCSTRING)
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
# 从 transformers.models.mistral.modeling_mistral.MistralForCausalLM 复制代码并进行修改，将 MISTRAL 替换为 STARCODER2，以匹配特定的模型和版本
class Starcoder2ForCausalLM(Starcoder2PreTrainedModel):
    # 定义了共享权重的键名列表，此处只包括 lm_head.weight
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数，接受一个配置对象 config 作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建一个 Starcoder2Model 对象作为模型的基础
        self.model = Starcoder2Model(config)
        # 设置词汇表大小为配置中指定的大小
        self.vocab_size = config.vocab_size
        # 创建一个线性层 lm_head，用于生成词汇表中词的预测
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用后处理函数，用于初始化权重并进行最终处理
        self.post_init()

    # 返回模型的输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置模型的输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回模型的输出嵌入，即 lm_head 线性层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置模型的输出嵌入，即 lm_head 线性层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器部分的模型
    def set_decoder(self, decoder):
        self.model = decoder

    # 返回解码器部分的模型
    def get_decoder(self):
        return self.model

    # 前向传播函数，接受多个输入参数并返回模型的输出结果，带有文档字符串注释和返回值注释
    @add_start_docstrings_to_model_forward(STARCODER2_INPUTS_DOCSTRING)
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
        # 函数主体未提供，由后续代码块定义

    # 准备生成输入的辅助函数，接受多个输入参数，用于生成模型的输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 函数主体未提供，由后续代码块定义
        # 如果 past_key_values 不为 None，则处理它覆盖的令牌
        if past_key_values is not None:
            # 如果 past_key_values 是 Cache 类型的实例
            if isinstance(past_key_values, Cache):
                # 获取缓存的序列长度和已处理的令牌长度
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                # 获取缓存的最大长度
                max_cache_length = past_key_values.get_max_length()
            else:
                # 否则，从 past_key_values 中获取缓存的长度和已处理的令牌长度
                cache_length = past_length = past_key_values[0][0].shape[2]
                # 最大缓存长度设为 None
                max_cache_length = None

            # 保留未处理的令牌：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则说明一些输入仅作为缓存传递（例如当 input_embeds 作为输入时）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入令牌。根据 past_length 可以丢弃 input_ids 的部分令牌。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 只包含未处理的令牌。

            # 如果即将超过最大缓存长度，需要裁剪输入的 attention_mask。
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # 获取 kwargs 中的 position_ids 参数，如果不存在并且 attention_mask 存在，则动态创建 position_ids 以用于批次生成
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果有 past_key_values，则仅保留与 input_ids 相关的 position_ids
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入 inputs_embeds，则仅在第一次生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典中的参数
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        # 返回最终的 model_inputs 字典作为函数的输出
        return model_inputs
    # 定义一个函数 _reorder_cache，用于重新排序缓存中的过去键值
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空的元组 reordered_past 用于存储重新排序后的过去键值
        reordered_past = ()
        # 遍历 past_key_values 中的每一层的过去键值
        for layer_past in past_key_values:
            # 对每一层的 past_state 应用索引选择操作，根据 beam_idx 重新排列
            # 并将重新排序后的结果添加到 reordered_past 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的 past_key_values
        return reordered_past
# 定义了一个用于序列分类任务的 Starcoder2 模型，其顶部有一个线性层作为分类头部。
# 此模型 [`Starcoder2ForSequenceClassification`] 使用最后一个标记进行分类，类似于其他因果模型（如 GPT-2）的做法。

# 当分类任务依赖于最后一个标记时，需要知道最后一个标记的位置。如果配置中定义了 `pad_token_id`，则找到每行中不是填充标记的最后一个标记。
# 如果未定义 `pad_token_id`，则简单地取每个批次中每行的最后一个值。当传递 `inputs_embeds` 而不是 `input_ids` 时，由于无法猜测填充标记，
# 也会采用相同的策略（取每个批次中每行的最后一个值）。

@add_start_docstrings(
    """
    The Starcoder2 Model transformer with a sequence classification head on top (linear layer).

    [`Starcoder2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    STARCODER2_START_DOCSTRING,
)
# 从 transformers.models.llama.modeling_llama.LlamaForSequenceClassification 复制而来，将其中的 Llama 替换为 Starcoder2，LLAMA 替换为 STARCODER2
class Starcoder2ForSequenceClassification(Starcoder2PreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化时指定分类的类别数目
        self.num_labels = config.num_labels
        # 创建 Starcoder2Model 模型实例
        self.model = Starcoder2Model(config)
        # 初始化一个线性层，用于分类，输入大小为 config.hidden_size，输出大小为 num_labels，无偏置项
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的嵌入表示
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入的嵌入表示
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(STARCODER2_INPUTS_DOCSTRING)
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