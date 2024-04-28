# `.\transformers\models\qwen2\modeling_qwen2.py`

```
# 这是一个基于 GPT-NeoX 和 OPT 实现的 Qwen2 模型的 PyTorch 代码
# 该代码已经过修改以适应与 Meta AI 团队训练的模型的小型架构差异
# 在使用该代码前，请注意遵循 Apache License 2.0 协议

# 引入必要的库和模块
import inspect
import math
import warnings
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
from .configuration_qwen2 import Qwen2Config

# 如果 flash_attn 可用，则导入相关函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义文档中使用的常量
_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"

# 定义预训练模型列表
QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen2-7B-beta",
    # 查看所有 Qwen2 模型 https://huggingface.co/models?filter=qwen2
]

# 从 transformers.models.llama.modeling_llama 复制的函数
def _get_unpad_data(attention_mask):
    # 获取批次中每个序列的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取注意力掩码中非填充位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最长序列的长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算每个序列在批次中的起始位置
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# 从 transformers.models.llama.modeling_llama 复制的类
class Qwen2RMSNorm(nn.Module):
    pass
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        # 初始化 Qwen2RMSNorm 类
        super().__init__()
        # 定义权重参数，初始值为1，可训练
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义方差的极小值
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 记录输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 将输入张量转换为 float32 数据类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算张量的方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行 RMS 归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回经过权重调整的隐藏状态
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 初始化 Qwen2RotaryEmbedding 类
        super().__init__()

        # 设置维度、最大位置编码长度和基数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率逆向
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 将频率逆向注册为缓冲张量
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了让 `torch.jit.trace` 能够工作，构建位置编码的余弦和正弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置余弦和正弦缓存
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 按列拼接余弦和正弦缓存
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将余弦缓存和正弦缓存注册为缓冲张量
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果序列长度超过缓存的最大序列长度，则重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回经过位置编码的余弦和正弦值
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 将输入张量的一半维度进行旋转
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 按最后一个维度拼接旋转后的张量
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    Args:
        q (`torch.Tensor`): The query tensor.  # 定义输入参数q为查询张量
        k (`torch.Tensor`): The key tensor.  # 定义输入参数k为关键张量
        cos (`torch.Tensor`): The cosine part of the rotary embedding.  # 定义输入参数cos为旋转嵌入的余弦部分
        sin (`torch.Tensor`): The sine part of the rotary embedding.  # 定义输入参数sin为旋转嵌入的正弦部分
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.  # 位置id表示与查询和关键张量对应的标记的位置索引。例如，可以在处理KV缓存时使用偏移的位置id。
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.  # 参数'unsqueeze_dim'指定了在哪个维度上对cos[position_ids]和sin[position_ids]进行unsqueeze，以便它们能够正确地广播到q和k的维度。
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.  # 返回由使用旋转位置嵌入旋转后的查询和关键张量组成的元组
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 在指定维度上对cos[position_ids]进行unsqueeze
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # 在指定维度上对sin[position_ids]进行unsqueeze
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 计算旋转后的查询张量
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 计算旋转后的关键张量
    return q_embed, k_embed  # 返回旋转后的查询和关键张量
# 从transformers.models.mistral.modeling_mistral.MistralMLP复制并将Mistral更改为Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 从transformers.models.llama.modeling_llama.repeat_kv复制
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这相当于torch.repeat_interleave(x, dim=1, repeats=n_rep)。隐藏状态从(batch, num_key_value_heads, seqlen, head_dim)
    变为(batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    多头注意力来自'Attention Is All You Need'论文。修改为使用滑动窗口注意力：Longformer
    和"Generating Long Sequences with Sparse Transformers"。
    """
    # 初始化函数，包含两个参数：config 和 layer_idx（可选，默认为 None）
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 保存传入的 config 参数
        self.config = config
        # 保存传入的 layer_idx 参数
        self.layer_idx = layer_idx
        # 如果 layer_idx 是 None，则输出警告信息
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
    
        # 保存隐藏层的大小
        self.hidden_size = config.hidden_size
        # 保存注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_heads
        # 保存键值头的数量
        self.num_key_value_heads = config.num_key_value_heads
        # 根据键值头的数量计算每个键值组的数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 保存最大位置嵌入数量
        self.max_position_embeddings = config.max_position_embeddings
        # 保存 rope_theta 参数
        self.rope_theta = config.rope_theta
        # 设置为因果关系（自注意力）
        self.is_causal = True
        # 保存注意力机制的丢弃率
        self.attention_dropout = config.attention_dropout
    
        # 如果隐藏层大小不是注意力头数量的整数倍，抛出数值错误
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 初始化 q 线性投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # 初始化 k 线性投影层
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # 初始化 v 线性投影层
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # 初始化输出线性投影层
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
        # 初始化旋转嵌入
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    # 前向传播函数，包含多个参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 闪光注意力模块，遵循 Qwen2 注意力模块。此模块继承自 `Qwen2Attention`，因为模块的权重保持不变。唯一需要更改的是在前向传递中，
    它需要正确调用闪光注意力的公共 API，并在输入中处理填充标记（如果有）。另外，对于滑动窗口注意力，我们仅对底部的 `config.max_window_layers` 层应用 SWA。
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

        # TODO: 一旦 RoCm 的闪光注意力版本升级到 2.1，应该将其删除。
        # flash_attn
    # 定义一个方法，用于处理输入数据，准备用于注意力机制的查询、键和值
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取批量大小、键值序列长度、注意力头数和头维度
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 在第一次迭代中，我们需要适当地重新创建填充蒙版
        # 通过在正确位置切片蒙版
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 根据索引重新排列键和值的维度
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        # 如果查询长度等于键值序列长度
        if query_length == kv_seq_len:
            # 根据索引重新排列查询的维度
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为 1
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            # 生成一系列长度为 batch_size + 1 的整数，用于未填充数据的处理
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            # 压缩查询维度
            query_layer = query_layer.squeeze(1)
        else:
            # 选择左填充，根据查询长度切片蒙版
            attention_mask = attention_mask[:, -query_length:]
            # 处理未填充数据
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的数据
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 从transformers.models.llama.modeling_llama.LlamaSdpaAttention复制代码，将Llama->Qwen2
class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # 从Qwen2Attention.forward进行修改
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
):
# QWEN2_ATTENTION_CLASSES字典，用于根据不同的配置选择对应的注意力模型
QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}

# Qwen2DecoderLayer类，用于定义Qwen2解码器层
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 如果启用滑动窗口注意力且注意力实现不是"flash_attention_2"，则发出警告
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # 初始化自注意力模块，根据配置选择不同的实现
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # 初始化MLP模块
        self.mlp = Qwen2MLP(config)
        # 初始化输入层归一化模块
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 初始化自注意力后的层归一化模块
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 定义前向传播方法
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
        # 检查是否传入了“padding_mask”，如果有则发出警告，将在4.37版本中移除
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入到该层的张量，形状为`(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): 大小为`(batch, sequence_length)`的注意力掩码，其中填充元素由0表示
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量下的`attentions`
            use_cache (`bool`, *optional*):
                如果设置为`True`，则返回`past_key_values`键值状态可用于加快解码速度（参见`past_key_values`）
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 此处定义了一个长字符串，用于说明该模型的一些基本信息及参数说明

@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    # 继承自PreTrainedModel的Qwen2PreTrainedModel类
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # 初始化权重的函数
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# 定义了一个Qwen2PreTrainedModel类，其中包含了初始化权重的方法

QWEN2_INPUTS_DOCSTRING = r"""
"""

@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        # 初始化Qwen2Model类
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens
    # 设置输入嵌入层的数值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 在模型前向传播方法添加起始文档字符串
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，默认为 None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 上下文的键值对，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入层的数值，默认为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典结构，默认为 None
class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化模型
    def __init__(self, config):
        super().__init__(config)
        # 创建 Qwen2Model 模型实例
        self.model = Qwen2Model(config)
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 创建线性层，用于输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入词嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出词嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出词嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model

    # 定义前向传播方法，添加文档字符串和返回值替换
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
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
    # 准备用于生成的输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        # 如果 past_key_values 不为 None，则进行下面的操作
        if past_key_values is not None:
            # 如果 past_key_values 是 Cache 类型
            if isinstance(past_key_values, Cache):
                # 获取 past_key_values 的序列长度、已处理 token 数和最大缓存长度
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                # 如果 past_key_values 不是 Cache 类型，则获取其第一个元素的维度
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 保留未处理的 token：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，表示有一些输入是作为缓存独立传入的
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入 token。可以根据 past_length 丢弃部分 input_ids
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则 (past_length >= input_ids.shape[1])，假设 input_ids 只有未处理的 token

            # 如果即将超出最大的缓存长度，需要裁剪输入 attention_mask
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        # 如果 attention_mask 存在但 position_ids 不存在
        if attention_mask is not None and position_ids is None:
            # 动态创建 batch 生成所需的 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入了 `inputs_embeds`，并且 past_key_values 不存在，则只在第一代步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 中的键值对
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    # 静态方法
    @staticmethod
    # 重新排序缓存中的过去键值对
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键值对元组
        reordered_past = ()
        # 遍历过去键值对
        for layer_past in past_key_values:
            # 将每一层的过去状态按照给定的beam_idx重新排序，并转移到相同的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对元组
        return reordered_past
# 使用装饰器添加文档字符串，说明了 Qwen2 模型的序列分类头（线性层）的转换器
# Qwen2ForSequenceClassification 使用最后一个令牌进行分类，与其他因果模型（例如 GPT-2）相同
# 由于它在最后一个令牌上进行分类，因此需要知道最后一个令牌的位置。如果在配置中定义了 'pad_token_id'，则在每一行中找到最后一个不是填充令牌的令牌。
# 如果未定义 'pad_token_id'，则仅采用批次中每行的最后一个值。当传递 'inputs_embeds' 而不是 'input_ids' 时，它无法猜测填充令牌，因此采用相同的策略（在批次中每行的最后一个值）。
class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签的数量
        self.num_labels = config.num_labels
        # 初始化 Qwen2 模型
        self.model = Qwen2Model(config)
        # 线性层，用于分类
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 前向传播方法，使用了输入的文档字符串
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
```