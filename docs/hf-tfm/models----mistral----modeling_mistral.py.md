# `.\transformers\models\mistral\modeling_mistral.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 此代码基于 EleutherAI 的 GPT-NeoX 库以及该库中 GPT-NeoX 和 OPT 实现的基础，并对其进行了修改以适应与 Meta AI 团队训练模型时存在的一些架构差异。
# 遵循 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用这个文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是在“原样”基础上分发的，没有任何形式的保证或条件，无论是明示的还是暗示的
# 查看特定语言的许可证以获取有关权限和限制的详细信息

""" PyTorch Mistral model.""" # PyTorch Mistral 模型

# 导入模块
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
from .configuration_mistral import MistralConfig

# 判断是否导入了 flash_attn 模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

logger = logging.get_logger(__name__) # 获取日志记录器

_CONFIG_FOR_DOC = "MistralConfig"

# 以下是函数定义和类定义部分，由于代码过多，无法在此处一一解释，敬请谅解
    # 定义 MistralRMSNorm 类，继承自 nn.Module，功能等同于 T5LayerNorm
    def __init__(self, hidden_size, eps=1e-6):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 定义一个可学习的权重参数 weight，大小为 hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 设置一个很小的常数 eps，用于防止除以零
        self.variance_epsilon = eps
    
    # 定义 forward 方法，实现前向传播
    def forward(self, hidden_states):
        # 保存输入数据的数据类型
        input_dtype = hidden_states.dtype
        # 将输入数据转换为 float32 类型，以便进行后续计算
        hidden_states = hidden_states.to(torch.float32)
        # 计算输入数据在最后一个维度上的方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 使用方差和 epsilon 进行归一化，得到归一化的隐藏状态
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 将归一化的隐藏状态乘以可学习的权重参数 weight，并转回原始的数据类型
        return self.weight * hidden_states.to(input_dtype)
# 将 transformers.models.llama.modeling_llama.LlamaRotaryEmbedding 类复制，并将 Llama 替换为 Mistral
class MistralRotaryEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # 设置属性
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率的倒数，并将其注册为缓冲区
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建缓存以使 `torch.jit.trace` 函数正常工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列顺序以获得相同的计算
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播函数
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# 从 transformers.models.llama.modeling_llama.rotate_half 复制函数
def rotate_half(x):
    """旋转输入的一半隐藏维度."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从 transformers.models.llama.modeling_llama.apply_rotary_pos_emb 复制函数
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """将旋转位置嵌入应用于查询和键张量.
    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`):
            与查询和键张量对应的令牌的位置索引。例如，可以用于在使用 KV 缓存时传递偏移的位置 id。
        unsqueeze_dim (`int`, *可选*, 默认为 1):
            'unsqueeze_dim' 参数指定沿着哪个维度对 cos[position_ids] 和 sin[position_ids] 进行展开，以便能正确广播到 q 和 k 的维度。
            例如，注意到 cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]。那么如果 q 和
            k 的形状为 [batch_size, heads, seq_len, head_dim]，设置 unsqueeze_dim=1 使得 cos[position_ids] 和 sin[position_ids]
            能够广播到 q 和 k 的形状。类似地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。
    Returns:
        包含使用旋转位置嵌入旋转后的查询和键张量的 `tuple(torch.Tensor)`。
    """
    # 沿着指定维度对 cos[position_ids] 进行展开
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # 沿着指定维度对 sin[position_ids] 进行展开
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 用旋转位置嵌入旋转查询张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 用旋转位置嵌入旋转键张量
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回旋转后的查询和键张量
    return q_embed, k_embed
# 定义 MistralMLP 类，继承自 nn.Module
class MistralMLP(nn.Module):
    # 初始化方法，接收配置信息
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置信息
        self.config = config
        # 获取隐藏层大小和中间层大小
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 定义三个全连接层，用于计算门控、上采样和下采样
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 获取激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    # 前向传播方法
    def forward(self, x):
        # 计算门控值，上采样和下采样结果，并相乘得到最终输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 定义 repeat_kv 函数，用于重复张量的某个维度
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是 torch.repeat_interleave(x, dim=1, repeats=n_rep) 的等价操作。
    输入张量 hidden_states 的形状为 (batch, num_key_value_heads, seqlen, head_dim)，
    输出张量的形状为 (batch, num_attention_heads, seqlen, head_dim)。
    """
    # 获取输入张量的形状
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 如果无需重复，直接返回输入张量
    if n_rep == 1:
        return hidden_states
    # 插入一个维度，并扩展到需要的形状
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 将第二个维度合并成一个维度
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 定义 MistralAttention 类，继承自 nn.Module
class MistralAttention(nn.Module):
    """
    多头注意力机制，修改自 "Attention Is All You Need" 论文。
    使用滑动窗口注意力机制：Longformer 和 "Generating Long Sequences with Sparse Transformers"。
    """
    # 初始化 MistralAttentionLayer 类，该类继承自 nn.Module
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 保存传入的 MistralConfig 对象
        self.config = config
        # 保存传入的 layer_idx 值
        self.layer_idx = layer_idx
        # 如果没有传入 layer_idx，则输出警告
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
    
        # 设置一些重要的参数
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
    
        # 检查 hidden_size 是否能被 num_heads 整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
    
        # 初始化 Q/K/V/O 投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
        # 初始化 Rotary Position Embedding
        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    # 定义一个辅助方法，用于调整张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # 在这里实现前向传播的逻辑
        pass
class MistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 当 Flash Attention 升级到 2.1 版本后应移除此部分
        # flash_attn<2.1 生成左上角对齐的因果蒙版，而这里需要的是底部右侧对齐，默认情况下 flash_attn>=2.1 已经实现了这一改进。这个属性用来处理这个不同点。参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 需要注意的是，对于 flash_attn<2.1，使用 q_seqlen != k_seqlen（除非 q_seqlen == 1 的情况）会产生一个错误的蒙版（左上角对齐）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
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
    # 重新组织输入，以适配 transformer 模型的输入要求
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取 batch_size, kv_seq_len, num_heads, head_dim 的值
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 如果 kv_seq_len 不等于 attention_mask 的最后一维长度
        if kv_seq_len != attention_mask.shape[-1]:
            # 则重新创建填充的 mask，将其切片到正确的位置
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 获取 unpad 数据的索引、cu_seqlens、最大在 batch 中的序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 重构 key_layer 和 value_layer
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        # 如果 query_length 等于 kv_seq_len
        if query_length == kv_seq_len:
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
            # 使用 torch.arange 创建 cu_seqlens_q
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这会有一次内存复制
            # 设置 indices_q 为 cu_seqlens_q 切片到最后一个元素
            indices_q = cu_seqlens_q[:-1]
            # 压缩 query_layer 的第一个维度
            query_layer = query_layer.squeeze(1)
        else:
            # 将 attention_mask 切片到 -query_length
            attention_mask = attention_mask[:, -query_length:]
            # 重新组织 query_layer、indices_q、cu_seqlens_q、max_seqlen_in_batch_q
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回结果
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 从transformers.models.llama.modeling_llama.LlamaSdpaAttention复制的MistralSdpaAttention类，将Llama替换为Mistral
class MistralSdpaAttention(MistralAttention):
    """
    使用torch.nn.functional.scaled_dot_product_attention的Mistral注意力模块。此模块继承自`MistralAttention`，因为模块的权重保持不变。唯一的更改是前向传递以适应SDPA API。
    """

    # 适应自MistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "flash_attention_2": MistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}

# MistralDecoderLayer类定义
class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 前向传递函数定义
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
        # 如果参数中包含"padding_mask"，则发出警告
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # 保存输入 hidden_states 的原始值
        residual = hidden_states

        # 输入 hidden_states 经过 input_layernorm 层处理
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # 调用 self_attn 方法进行自注意力机制处理
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 将结果与原始值相加得到新的 hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        # 保存输入 hidden_states 的原始值
        residual = hidden_states
        # 输入 hidden_states 经过 post_attention_layernorm 层处理
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 输入 hidden_states 经过 mlp 处理
        hidden_states = self.mlp(hidden_states)
        # 将结果与原始值相加得到新的 hidden_states
        hidden_states = residual + hidden_states

        # 将处理后的 hidden_states 存入 outputs
        outputs = (hidden_states,)

        # 如果 output_attentions 为真，则将 self_attn_weights 存入 outputs
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果 use_cache 为真，则将 present_key_value 存入 outputs
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# 设置 MISTRAL_START_DOCSTRING 变量，存储 Mistral 模型的文档字符串
MISTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加模型注释，指定输出原始隐藏状态而没有特定的顶部头部
@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralPreTrainedModel(PreTrainedModel):
    # 模型配置类为 MistralConfig
    config_class = MistralConfig
    # 基础模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块名称列表
    _no_split_modules = ["MistralDecoderLayer"]
    # 跳过键的设备放置
    _skip_keys_device_placement = "past_key_values"
    # 支持的闪光注意力 2 版本
    _supports_flash_attn_2 = True
    # 支持的 SDPA（Self-Deattention Pattern Attention）
    _supports_sdpa = True
    # 支持的缓存类
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


# 设置 MISTRAL_INPUTS_DOCSTRING 变量为空字符串
MISTRAL_INPUTS_DOCSTRING = r"""
"""

# 添加模型注释，指定输出原始隐藏状态而没有特定的顶部头部
@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    # 初始化方法
    def __init__(self, config: MistralConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置填充索引和词汇表大小
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 创建嵌入层和模块列表
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        # 输入 token ID 序列
        input_ids: torch.LongTensor = None,
        # 注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # 位置 ID 序列
        position_ids: Optional[torch.LongTensor] = None,
        # 上一时间步的 key-value 缓存
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入嵌入
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出所有隐藏层
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式
        return_dict: Optional[bool] = None,
class MistralForCausalLM(MistralPreTrainedModel):
    # 定义权重共享的关键字
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 实例化一个MistralModel对象
        self.model = MistralModel(config)
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 创建线性层，用于LM头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model

    # 前向传播函数，添加模型输入的文档字符串和替换返回值的文档字符串
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
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
    # 生成推理过程中的输入，用于生成
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
```  
    # 处理 past_key_values 参数，确保只使用未处理的 token
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        # 如果存在 past_key_values，则需要处理一些细节
        if past_key_values is not None:
            # 如果 past_key_values 是 Cache 对象，则获取序列长度和已处理 token 数
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            # 否则直接从 past_key_values 中获取长度
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None
    
            # 保留未处理的 token
            # 1. 如果 attention_mask 长度大于 input_ids，说明部分输入是作为缓存传入的（如传入 input_embeds）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2. 如果 past_length 小于 input_ids 长度，则 input_ids 包含所有输入，可以丢弃 past_length 之前的 token
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
    
            # 如果即将超过最大缓存长度，需要裁剪 attention_mask
            if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
                attention_mask = attention_mask[:, -max_cache_length:]
    
        # 如果没有提供 position_ids，根据 attention_mask 动态生成
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
    
        # 如果提供了 inputs_embeds 且没有 past_key_values，则只使用 inputs_embeds
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
    
        # 更新其他输入参数
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
    
    # 静态方法
    @staticmethod
    # 根据给定的 beam_idx 重排 past_key_values 的顺序
    def _reorder_cache(past_key_values, beam_idx):
        # 创建一个空的元组来存储重排后的 past_key_values
        reordered_past = ()
        # 遍历 past_key_values 中的每一个 layer_past
        for layer_past in past_key_values:
            # 对每个 layer_past 中的每个 past_state 进行重排
            reordered_layer_past = tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            # 将重排后的 layer_past 添加到 reordered_past 中
            reordered_past += (reordered_layer_past,)
        # 返回重排后的 past_key_values
        return reordered_past
# 添加模型的文档字符串，描述了Mistral模型变换器在顶部的序列分类头部（线性层）的作用和特点
# MistralForSequenceClassification使用最后一个标记来进行分类，就像其他类因果模型（例如GPT-2）一样
# 如果在配置中定义了pad_token_id，则会找到每行中不是填充标记的最后一个标记；如果未定义pad_token_id，则简单地获取批次的每行中的最后一个值
# 在使用inputs_embeds而不是input_ids时，无法猜测填充标记，因此它执行相同的操作（获取批次的每行中的最后一个值）
@add_start_docstrings(
    """
    The Mistral Model transformer with a sequence classification head on top (linear layer).

    [`MistralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MISTRAL_START_DOCSTRING,
)
# 从transformers.models.llama.modeling_llama.LlamaForSequenceClassification复制并修改为MistralForSequenceClassification，将Llama更改为Mistral，LLAMA更改为MISTRAL
class MistralForSequenceClassification(MistralPreTrainedModel):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 记录标签数量
        self.num_labels = config.num_labels
        # 创建MistralModel对象
        self.model = MistralModel(config)
        # 创建一个线性层，用于分类
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
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