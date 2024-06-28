# `.\models\mistral\modeling_flax_mistral.py`

```py
# Import necessary modules and functions from libraries
from typing import Optional, Tuple  # 导入类型提示相关的模块和函数

import flax.linen as nn  # 导入 Flax 中的 Linen 模块并重命名为 nn
import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 中的 numpy 模块并重命名为 jnp
import numpy as np  # 导入 numpy 库并重命名为 np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 从 Flax 中导入相关函数
from flax.linen import combine_masks, make_causal_mask  # 从 Flax 的 Linen 模块导入函数
from flax.linen.attention import dot_product_attention_weights  # 从 Flax 的 attention 模块导入函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 从 Flax 的 traverse_util 模块导入函数
from jax import lax  # 导入 JAX 的 lax 模块

# Import specific outputs and utilities from related modules
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPast,
    FlaxCausalLMOutput,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, logging
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_mistral import MistralConfig  # 导入 MistralConfig 配置类

# Get the logger for this module
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# Constants for documentation purposes
_CONFIG_FOR_DOC = "MistralConfig"  # 用于文档的配置示例名称
_REAL_CHECKPOINT_FOR_DOC = "mistralai/Mistral-7B-v0.1"  # 实际检查点的文档示例
_CHECKPOINT_FOR_DOC = "ksmcg/Mistral-tiny"  # 检查点的文档示例

# Start of the model documentation string
MISTRAL_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    # 参数说明：
    # config ([`MistralConfig`]): 模型配置类，包含模型的所有参数。
    #                          使用配置文件初始化时不会加载模型的权重，只加载配置信息。
    #                          查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    #      计算使用的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16` 或 `jax.numpy.bfloat16` 中的一种。
    #      可用于在 GPU 或 TPU 上启用混合精度训练或半精度推断。如果指定，则所有计算将使用给定的 `dtype` 进行。
    #
    #      **注意，这仅指定计算时的数据类型，不影响模型参数的数据类型。**
    #
    #      如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
# 定义了一个文档字符串常量，描述了 `FlaxMistralRMSNorm` 类的输入参数和用法
MISTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            输入序列标记在词汇表中的索引。默认情况下，提供的填充将被忽略。
            可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力操作的掩码。掩码值在 `[0, 1]` 范围内：

            - 对于 **未被掩码** 的标记，值为 1，
            - 对于 **被掩码** 的标记，值为 0。

            可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            如果使用了 `past_key_values`，可以选择仅输入最后的 `decoder_input_ids`（参见 `past_key_values`）。

            如果要更改填充行为，应阅读 [`modeling_opt._prepare_decoder_attention_mask`] 并根据需求进行修改。详见 [该论文中的图表 1](https://arxiv.org/abs/1910.13461) 获取有关默认策略的更多信息。

            - 1 表示头部 **未被掩码**，
            - 0 表示头部 **被掩码**。
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.n_positions - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, 由 `init_cache` 返回或传递先前的 `past_key_values`):
            预计算隐藏状态的字典（键和值在注意力块中）。可用于快速自回归解码。预计算的键和值隐藏状态的形状为 *[batch_size, max_length]*。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""

# 从 `transformers.models.llama.modeling_flax_llama.FlaxLlamaRMSNorm` 复制并修改为 `FlaxMistralRMSNorm`
class FlaxMistralRMSNorm(nn.Module):
    # 类型注解，指定了 `config` 属性的类型为 `MistralConfig`
    config: MistralConfig
    # 默认数据类型为 `jnp.float32`
    dtype: jnp.dtype = jnp.float32
    # 初始化对象的epsilon属性为配置中的rms_norm_eps值
    self.epsilon = self.config.rms_norm_eps

    # 初始化对象的weight属性，使用param方法生成，传入的lambda函数生成一个形状为hidden_size的全1数组
    self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    # 定义对象的调用方法，接收hidden_states作为参数
    def __call__(self, hidden_states):
        # 将hidden_states转换为JAX支持的float32类型的数组variance
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        
        # 对variance中的每个元素求平方
        variance = jnp.power(variance, 2)
        
        # 对variance在最后一个维度上求平均值，并保持维度为1
        variance = variance.mean(-1, keepdims=True)
        
        # 使用JAX的sqrt函数对variance加上epsilon后开方，作为对hidden_states的归一化系数
        # 注意：使用jax.numpy.sqrt代替jax.lax.rsqrt是因为两者的行为不同于torch.rsqrt
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        # 返回归一化后的hidden_states乘以对象的weight属性
        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)
# 从 transformers.models.llama.modeling_flax_llama.FlaxLlamaRotaryEmbedding 复制代码，将 Llama 替换为 Mistral
class FlaxMistralRotaryEmbedding(nn.Module):
    # 使用 MistralConfig 配置信息
    config: MistralConfig
    # 数据类型默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 计算每个注意力头的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 创建正弦和余弦位置编码
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

    def __call__(self, key, query, position_ids):
        # 根据位置编码获取对应的正弦和余弦值
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        # 应用旋转位置编码到键和查询张量
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        # 转换为指定数据类型
        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        # 返回处理后的键和查询张量
        return key, query


# 从 transformers.models.llama.modeling_flax_llama.FlaxLlamaMLP 复制代码，将 Llama 替换为 Mistral
class FlaxMistralMLP(nn.Module):
    # 使用 MistralConfig 配置信息
    config: MistralConfig
    # 数据类型默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 获取嵌入维度和内部维度
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        # 初始化内核，并设置激活函数
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.act = ACT2FN[self.config.hidden_act]

        # 定义门控投影、下游投影和上游投影
        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    def __call__(self, hidden_states):
        # 上游投影处理隐藏状态
        up_proj_states = self.up_proj(hidden_states)
        # 使用激活函数处理门控投影的隐藏状态
        gate_states = self.act(self.gate_proj(hidden_states))

        # 应用门控和上游投影到下游投影的隐藏状态
        hidden_states = self.down_proj(up_proj_states * gate_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 从 transformers.models.llama.modeling_flax_llama.apply_rotary_pos_emb 复制代码
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    # 应用旋转位置编码到张量
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


# 从 transformers.models.llama.modeling_flax_llama.create_sinusoidal_positions 复制代码
def create_sinusoidal_positions(num_pos, dim):
    # 计算逆频率
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    # 创建正弦和余弦位置编码
    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


# 从 transformers.models.llama.modeling_flax_llama.rotate_half 复制代码
def rotate_half(tensor):
    """旋转输入张量的一半隐藏维度。"""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


# 定义 FlaxMistralAttention 类，用于注意力机制，未完整复制
class FlaxMistralAttention(nn.Module):
    # 使用 MistralConfig 配置信息
    config: MistralConfig
    # 数据类型默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        # 从配置中获取参数
        config = self.config
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size
        # 设置注意力头数
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_heads
        # 设置键值头数
        self.num_key_value_heads = config.num_key_value_heads
        # 计算每个键值组的头数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 设置最大位置嵌入数
        self.max_position_embeddings = config.max_position_embeddings
        # 判断是否需要在注意力softmax计算中使用fp32精度
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        # 设置rope_theta
        self.rope_theta = config.rope_theta
        
        # 检查隐藏层大小是否可以被注意力头数整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 初始化查询、键、值和输出的线性投影层
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)
        
        # 创建自回归遮罩
        casual_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        # 根据滑动窗口大小生成自回归遮罩
        self.causal_mask = jnp.triu(casual_mask, k=-config.sliding_window)
        
        # 初始化旋转嵌入
        self.rotary_emb = FlaxMistralRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        # 将隐藏状态分割成多个头
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 合并多个头的隐藏状态
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    # 从transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache复制而来
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否初始化缓存数据
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或者初始化缓存的 key 和 value，若不存在则创建零张量
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或者初始化缓存的索引，若不存在则设置为 0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的 1D 空间切片更新 key 和 value 缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的 key 和 value
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数目
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的自注意力掩码：我们的单个查询位置应仅关注已生成和缓存的 key 位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 组合现有的掩码和给定的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的 key, value 和注意力掩码
        return key, value, attention_mask
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 使用 self.q_proj 对隐藏状态进行投影得到查询状态
        query_states = self.q_proj(hidden_states)
        # 使用 self.k_proj 对隐藏状态进行投影得到键状态
        key_states = self.k_proj(hidden_states)
        # 使用 self.v_proj 对隐藏状态进行投影得到值状态
        value_states = self.v_proj(hidden_states)

        # 将查询状态按照头数进行分割
        query_states = self._split_heads(query_states, self.num_heads)
        # 将键状态按照键值头数进行分割
        key_states = self._split_heads(key_states, self.num_key_value_heads)
        # 将值状态按照键值头数进行分割
        value_states = self._split_heads(value_states, self.num_key_value_heads)

        # 使用 rotary_emb 方法对键状态和查询状态进行旋转嵌入
        key_states, query_states = self.rotary_emb(key_states, query_states, position_ids)

        # 获取查询和键的长度
        query_length, key_length = query_states.shape[1], key_states.shape[1]

        # 根据是否有缓存的键来确定掩码的偏移量和最大解码长度
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # 创建动态切片的因果掩码
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            # 使用预先计算好的因果掩码
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        # 获取批次大小
        batch_size = hidden_states.shape[0]
        # 将因果掩码广播到与注意力掩码相同的形状
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        # 将注意力掩码扩展到与因果掩码相同的形状
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        # 结合注意力掩码和因果掩码
        attention_mask = combine_masks(attention_mask, causal_mask)

        # 如果有缓存的键或者需要初始化缓存，则将键状态、值状态和注意力掩码拼接到缓存中
        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )
        
        # 将键状态在键值组之间重复以支持并行处理
        key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
        # 将值状态在键值组之间重复以支持并行处理
        value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)

        # 创建注意力偏置，根据注意力掩码设置有效和无效区域的偏置值
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # 常规的点积注意力计算
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            deterministic=deterministic,
            dropout_rate=self.config.attention_dropout,
            dtype=attention_dtype,
        )

        # 如果需要在 float32 中执行 softmax，将注意力权重转换为目标 dtype
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        # 使用 einsum 执行注意力加权求和操作
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        # 合并多头的结果
        attn_output = self._merge_heads(attn_output)
        # 对输出应用输出投影
        attn_output = self.o_proj(attn_output)

        # 准备输出，包括注意力权重（如果需要）
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaDecoderLayer with Llama->Mistral
class FlaxMistralDecoderLayer(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化输入层的 Layer Normalization
        self.input_layernorm = FlaxMistralRMSNorm(self.config, dtype=self.dtype)
        # 初始化自注意力机制
        self.self_attn = FlaxMistralAttention(self.config, dtype=self.dtype)
        # 初始化自注意力后的 Layer Normalization
        self.post_attention_layernorm = FlaxMistralRMSNorm(self.config, dtype=self.dtype)
        # 初始化多层感知机 MLP
        self.mlp = FlaxMistralMLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 残差连接
        residual = hidden_states
        # 应用输入层的 Layer Normalization
        hidden_states = self.input_layernorm(hidden_states)
        # 应用自注意力机制
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # 残差连接
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        # 残差连接
        residual = hidden_states
        # 应用自注意力后的 Layer Normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 应用多层感知机 MLP
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states

        return (hidden_states,) + outputs[1:]


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel with GPTNeo->Mistral, GPT_NEO->MISTRAL, transformer->model
class FlaxMistralPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MistralConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: MistralConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建与input_ids形状相同的全1张量作为注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 根据input_ids的形状广播生成位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 拆分随机数生成器rng，生成参数随机数和dropout随机数
        params_rng, dropout_rng = jax.random.split(rng)
        # 存储随机数生成器
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用self.module的初始化方法初始化模型参数，返回未解冻的参数字典
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        # 如果传入了预训练的参数params，则与随机初始化的参数进行合并
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 返回合并后的冻结参数字典
            return freeze(unflatten_dict(params))
        else:
            # 否则返回随机初始化的参数字典
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批处理大小，定义了初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度，定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        # 创建与input_ids形状相同的全1张量作为注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        # 根据input_ids的形状广播生成位置编码张量
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用self.module的初始化方法初始化模型变量，设置init_cache=True以初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回未解冻的缓存字典
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
            # 如果没有显式传入 output_attentions 参数，则使用配置中的设定
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果没有显式传入 output_hidden_states 参数，则使用配置中的设定
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果没有显式传入 return_dict 参数，则使用配置中的设定
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 获取输入张量的批量大小和序列长度
            batch_size, sequence_length = input_ids.shape

            # 如果未传入 position_ids，则根据序列长度和批量大小广播生成位置 ID
            if position_ids is None:
                if past_key_values is not None:
                    raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

                position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

            # 如果未传入 attention_mask，则创建全为 1 的注意力遮罩
            if attention_mask is None:
                attention_mask = jnp.ones((batch_size, sequence_length))

            # 处理任何需要的伪随机数生成器
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng

            inputs = {"params": params or self.params}

            # 如果传入了 past_key_values，则将其作为 cache 输入到模块中，确保 cache 是可变的
            if past_key_values:
                inputs["cache"] = past_key_values
                mutable = ["cache"]
            else:
                mutable = False

            # 调用模块的 apply 方法进行前向传播
            outputs = self.module.apply(
                inputs,
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                jnp.array(position_ids, dtype="i4"),
                not train,
                False,
                output_attentions,
                output_hidden_states,
                return_dict,
                rngs=rngs,
                mutable=mutable,
            )

            # 如果传入了 past_key_values 并且设置了 return_dict，则将更新后的 cache 添加到模型输出中
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            # 如果传入了 past_key_values 但未设置 return_dict，则更新 cache 并将其添加到模型输出中
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

            # 返回模型输出
            return outputs
# 从transformers.models.llama.modeling_flax_llama.FlaxLlamaLayerCollection复制而来，将Llama改为Mistral
class FlaxMistralLayerCollection(nn.Module):
    # MistralConfig的实例变量config，dtype默认为jnp.float32
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 创建self.config.num_hidden_layers个FlaxMistralDecoderLayer对象列表
        self.blocks = [
            FlaxMistralDecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    # 模块调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        # 如果输出attentions，则初始化空元组all_attentions；否则为None
        all_attentions = () if output_attentions else None
        # 如果输出hidden states，则初始化空元组all_hidden_states；否则为None
        all_hidden_states = () if output_hidden_states else None

        # 遍历self.blocks中的每个FlaxMistralDecoderLayer对象
        for block in self.blocks:
            # 如果需要输出hidden states，则将当前hidden_states添加到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 调用block对象进行前向传播，获取layer_outputs
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新hidden_states为block的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出attentions，则将当前层的attentions添加到all_attentions元组中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 输出包含可能为None值的元组outputs，FlaxMistralModule将会过滤掉这些None值
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 返回outputs作为模块的输出结果
        return outputs


# 从transformers.models.llama.modeling_flax_llama.FlaxLlamaModule复制而来，将Llama改为Mistral
class FlaxMistralModule(nn.Module):
    # MistralConfig的实例变量config，dtype默认为jnp.float32
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 设置self.hidden_size为self.config.hidden_size
        self.hidden_size = self.config.hidden_size
        # 使用正态分布初始化embed_tokens的embedding参数
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        # 创建nn.Embed对象embed_tokens，用于token的embedding
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        # 创建FlaxMistralLayerCollection对象self.layers，用于处理层间关系
        self.layers = FlaxMistralLayerCollection(self.config, dtype=self.dtype)
        # 创建FlaxMistralRMSNorm对象self.norm，用于层间正则化
        self.norm = FlaxMistralRMSNorm(self.config, dtype=self.dtype)

    # 模块调用方法
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 返回字典形式的输出结果
        # 输入参数input_ids、attention_mask、position_ids以及其他标志位
    ):
        # 将输入的 token IDs 转换为嵌入表示，数据类型为整数
        input_embeds = self.embed_tokens(input_ids.astype("i4"))

        # 使用 Transformer 层处理输入数据
        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]

        # 对隐藏状态进行规范化处理
        hidden_states = self.norm(hidden_states)

        # 如果需要输出所有隐藏状态，则将当前隐藏状态加入所有隐藏状态列表
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要返回字典形式的输出，则去除所有值为 None 的项并返回元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和注意力值
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 添加文档字符串到 FlaxMistralModel 类，说明其作用是提供裸的 Mistral 模型变换器输出，没有特定的输出头部。
@add_start_docstrings(
    "The bare Mistral Model transformer outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class FlaxMistralModel(FlaxMistralPreTrainedModel):
    # 设置模块类为 FlaxMistralModule
    module_class = FlaxMistralModule


# 向 FlaxMistralModel 类添加调用示例文档字符串，用于样例的调用说明
append_call_sample_docstring(
    FlaxMistralModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPast,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)


# 从 transformers.models.llama.modeling_flax_llama.FlaxLlamaForCausalLMModule 复制代码，并将 Llama 更改为 Mistral
class FlaxMistralForCausalLMModule(nn.Module):
    config: MistralConfig  # 定义配置为 MistralConfig 类型
    dtype: jnp.dtype = jnp.float32  # 设置数据类型为 jnp.float32，默认为 float32

    def setup(self):
        # 使用配置和数据类型创建 FlaxMistralModule 模型
        self.model = FlaxMistralModule(self.config, dtype=self.dtype)
        # 创建 LM 头部，是一个全连接层，用于语言建模任务
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用模型进行前向传播
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]
        # 计算语言建模的 logits
        lm_logits = self.lm_head(hidden_states)

        # 如果不返回字典，则返回一个元组，包含 lm_logits 和其他输出
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回 FlaxCausalLMOutput 对象，包含 logits、隐藏状态和注意力信息
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 添加文档字符串到 FlaxMistralForCausalLM 类，说明其作用是在 Mistral 模型变换器上方增加语言建模头部（线性层）
@add_start_docstrings(
    """
    The Mistral Model transformer with a language modeling head (linear layer) on top.
    """,
    MISTRAL_START_DOCSTRING,
)
# 从 transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM 复制代码，并将 GPTJ 更改为 Mistral
class FlaxMistralForCausalLM(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralForCausalLMModule
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        # 获取输入的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用初始化方法初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)

        # 因为Mistral使用因果遮罩，对超出input_ids.shape[-1]和小于cache_length的位置已经进行了遮罩处理
        # 所以我们可以在这里创建一个静态的注意力遮罩，这对编译效率更高
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 根据给定的注意力遮罩计算位置ID
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 动态更新静态的注意力遮罩，将attention_mask的值复制进去
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有给定注意力遮罩，则使用默认的位置ID
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新生成过程中的输入参数
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新位置ID，将当前位置向后移动一步
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数 `append_call_sample_docstring`，用于向指定类添加示例文档字符串。
# 第一个参数 `FlaxMistralForCausalLM`：目标类，将在其上添加示例文档字符串。
# 第二个参数 `_CHECKPOINT_FOR_DOC`：用作示例文档字符串中的检查点的常量或路径。
# 第三个参数 `FlaxCausalLMOutputWithCrossAttentions`：示例文档字符串中的输出类。
# 第四个参数 `_CONFIG_FOR_DOC`：用作示例文档字符串中的配置的常量或路径。
# 关键字参数 `real_checkpoint=_REAL_CHECKPOINT_FOR_DOC`：用于指定示例文档字符串中真实检查点的常量或路径。
append_call_sample_docstring(
    FlaxMistralForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)
```