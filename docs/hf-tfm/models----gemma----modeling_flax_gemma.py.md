# `.\models\gemma\modeling_flax_gemma.py`

```py
# 导入必要的库和模块
from typing import Optional, Tuple  # 导入类型提示模块

import flax.linen as nn  # 导入Flax的Linen模块，用于定义模型
import jax  # 导入JAX，用于自动求导和数组操作
import jax.numpy as jnp  # 导入JAX的NumPy接口
import numpy as np  # 导入NumPy，用于数组操作
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入Flax的FrozenDict相关模块，用于不可变字典操作
from flax.linen import combine_masks, make_causal_mask  # 导入Flax的Linen模块中的函数
from flax.linen.attention import dot_product_attention_weights  # 导入注意力机制相关函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入Flax的工具函数，用于字典的扁平化和反扁平化
from jax import lax  # 导入JAX的lax模块，用于定义不同的线性代数和控制流操作

# 导入自定义的模块和类
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput  # 导入Flax模型输出相关类
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring  # 导入Flax预训练模型类和辅助函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入辅助工具函数和日志记录模块
from .configuration_gemma import GemmaConfig  # 导入Gemma模型的配置类

# 获取logger对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 文档中使用的配置信息
_CONFIG_FOR_DOC = "GemmaConfig"
_CHECKPOINT_FOR_DOC = "google/gemma-2b"
_REAL_CHECKPOINT_FOR_DOC = "openlm-research/open_llama_3b_v2"

# Gemma模型的开始文档字符串，包含模型的基本信息和JAX的特性说明
GEMMA_START_DOCSTRING = r"""

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
    # Parameters:
    #     config ([`GemmaConfig`]): Model configuration class with all the parameters of the model.
    #         Initializing with a config file does not load the weights associated with the model, only the
    #         configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
    #     dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
    #         The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16`, or
    #         `jax.numpy.bfloat16`.
    #
    #         This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
    #         specified all the computation will be performed with the given `dtype`.
    #
    #         **Note that this only specifies the dtype of the computation and does not influence the dtype of model
    #         parameters.**
    #
    #         If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
    #         [`~FlaxPreTrainedModel.to_bf16`].
"""
定义了一个多行字符串变量 `GEMMA_INPUTS_DOCSTRING`，用于描述以下函数的参数和用法。

"""
def create_sinusoidal_positions(num_pos, dim):
    """
    创建一个正弦位置编码矩阵。

    Args:
        num_pos (int): 序列中位置的总数。
        dim (int): 编码向量的维度。

    Returns:
        numpy.ndarray: 形状为 `(num_pos, dim)` 的正弦位置编码矩阵。
    """
    # 计算逆频率，这里使用了正弦位置编码的标准公式
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    # 计算频率矩阵，使用 numpy 的 einsum 函数进行计算
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    
    # 返回正弦位置编码矩阵
    return freqs
    # 将频率向量 `freqs` 沿着最后一个轴复制一次，然后进行连接
    emb = np.concatenate((freqs, freqs), axis=-1)
    # 对连接后的数组 `emb` 分别计算正弦和余弦值，然后沿着新增的维度合并起来
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    # 返回处理后的部分数据，只保留前 `num_pos` 个位置的结果
    return jnp.array(out[:, :, :num_pos])
# Copied from transformers.models.llama.modeling_flax_llama.rotate_half
# 函数：将输入张量的后一半隐藏维度进行旋转
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


# Copied from transformers.models.llama.modeling_flax_llama.apply_rotary_pos_emb
# 函数：应用旋转位置编码到张量上
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


# 类：FlaxGemmaRMSNorm
# 类变量：配置信息 config 为 GemmaConfig 类型，默认数据类型为 jnp.float32
class FlaxGemmaRMSNorm(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    # 方法：初始化设置
    def setup(self):
        self.epsilon = self.config.rms_norm_eps  # 设置 epsilon 参数
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)  # 初始化权重参数

    # 方法：调用实例时的行为
    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)  # 将隐藏状态转换为 jnp.float32 类型的张量
        variance = jnp.power(variance, 2)  # 计算张量的平方
        variance = variance.mean(-1, keepdims=True)  # 沿着最后一个轴计算张量的均值并保持维度
        # 使用 `jax.numpy.sqrt` 代替 `jax.lax.rsqrt`，因为与 `torch.rsqrt` 不匹配
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)  # 对隐藏状态进行归一化

        return (1 + self.weight) * jnp.asarray(hidden_states, dtype=self.dtype)  # 返回归一化后的结果


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaRotaryEmbedding with Llama->Gemma
# 类：FlaxGemmaRotaryEmbedding
# 类变量：配置信息 config 为 GemmaConfig 类型，默认数据类型为 jnp.float32
class FlaxGemmaRotaryEmbedding(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    # 方法：初始化设置
    def setup(self):
        head_dim = self.config.head_dim  # 从配置中获取头部维度信息
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)  # 创建正弦余弦位置编码

    # 方法：调用实例时的行为
    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]  # 获取指定位置的正弦余弦位置编码
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)  # 将编码拆分为正弦部分和余弦部分

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)  # 对 key 应用旋转位置编码
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)  # 对 query 应用旋转位置编码

        key = jnp.asarray(key, dtype=self.dtype)  # 将 key 转换为指定数据类型
        query = jnp.asarray(query, dtype=self.dtype)  # 将 query 转换为指定数据类型

        return key, query  # 返回应用位置编码后的 key 和 query


# 类：FlaxGemmaAttention
# 类变量：配置信息 config 为 GemmaConfig 类型，默认数据类型为 jnp.float32
#       causal 表示是否是因果注意力，is_cross_attention 表示是否是交叉注意力
class FlaxGemmaAttention(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False
    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size  # 从配置中获取隐藏层大小作为嵌入维度
        self.num_heads = config.num_attention_heads  # 从配置中获取注意力头的数量
        self.head_dim = config.head_dim  # 从配置中获取每个注意力头的维度
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32  # 检查数据类型是否为 jnp.float32，用于注意力 softmax

        self.num_key_value_heads = config.num_key_value_heads  # 从配置中获取键值头的数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 计算键值分组数

        kernel = jax.nn.initializers.normal(self.config.initializer_range)  # 使用正态分布初始化器初始化 kernel

        # 初始化查询投影层，设置输出维度为 num_heads * head_dim
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=kernel
        )

        # 初始化键投影层，设置输出维度为 num_key_value_heads * head_dim
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=kernel,
        )

        # 初始化值投影层，设置输出维度为 num_key_value_heads * head_dim
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=kernel,
        )

        # 初始化输出投影层，设置输出维度为 embed_dim
        self.o_proj = nn.Dense(
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=kernel
        )

        # 创建因果掩码，用于自注意力机制，形状为 (1, max_position_embeddings)，数据类型为布尔型
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")

        # 初始化旋转嵌入，使用 FlaxGemmaRotaryEmbedding 类，传入配置和数据类型
        self.rotary_emb = FlaxGemmaRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        # 将隐藏状态张量按指定的 num_heads 分割成多个头，保留前两个维度不变
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 将多个头的张量合并成一个头，保留前两个维度不变
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads * self.head_dim,))

    @nn.compact
    # 从 transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache 复制而来
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否正在初始化，通过检查是否存在缓存数据来判断
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键值状态，初始化为全零数组，与输入的 key 形状和数据类型相同
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取或创建缓存的值状态，初始化为全零数组，与输入的 value 形状和数据类型相同
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，初始化为整数0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取当前缓存数据的形状，假设为 (*batch_dims, max_length, num_heads, depth_per_head)
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新缓存的键和值，使用新的一维空间切片
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 使用 lax.dynamic_update_slice 函数更新缓存的键和值
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值状态
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 对于缓存的解码器自注意力，生成因果掩码：我们的单个查询位置应仅与已生成和缓存的键位置相对应，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 结合因果掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
    ):
        # 使用 self.q_proj 对隐藏状态进行查询投影
        query = self.q_proj(hidden_states)
        # 使用 self.k_proj 对隐藏状态进行键投影
        key = self.k_proj(hidden_states)
        # 使用 self.v_proj 对隐藏状态进行值投影
        value = self.v_proj(hidden_states)

        # 将查询张量按照头数目拆分
        query = self._split_heads(query, self.num_heads)
        # 将键张量按照键值头数目拆分
        key = self._split_heads(key, self.num_key_value_heads)
        # 将值张量按照键值头数目拆分
        value = self._split_heads(value, self.num_key_value_heads)

        # 对键和查询应用旋转嵌入
        key, query = self.rotary_emb(key, query, position_ids)

        # 获取查询和键的长度
        query_length, key_length = query.shape[1], key.shape[1]

        # 如果存在缓存的键，则根据缓存的键值创建一个因果掩码
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # 使用 lax.dynamic_slice 创建因果掩码
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            # 否则，直接使用预定义的因果掩码
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        # 获取批量大小
        batch_size = hidden_states.shape[0]
        # 广播因果掩码以匹配批量大小
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # 广播注意力掩码以匹配因果掩码的形状
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        # 将注意力掩码与因果掩码结合
        attention_mask = combine_masks(attention_mask, causal_mask)

        # 初始化 dropout_rng
        dropout_rng = None
        # 如果不是确定性运行且配置中指定了注意力 dropout，则创建 dropout_rng
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 在快速自回归解码期间，逐步逐步提供一个位置，并逐步缓存键和值
        if self.has_variable("cache", "cached_key") or init_cache:
            # 将键、值和查询连接到缓存中
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # 将布尔类型的注意力掩码转换为浮点数类型
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # 根据指定的键值组数重复键张量
        key = jnp.repeat(key, repeats=self.num_key_value_groups, axis=2)
        # 根据指定的键值组数重复值张量
        value = jnp.repeat(value, repeats=self.num_key_value_groups, axis=2)

        # 执行常规的点积注意力操作
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        # 如果指定了 attention_softmax_in_fp32，则将注意力权重转换为指定的数据类型
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        # 执行注意力输出计算
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        # 合并多头的输出
        attn_output = self._merge_heads(attn_output)
        # 对注意力输出进行最终的投影
        attn_output = self.o_proj(attn_output)

        # 根据需要返回注意力输出及其权重
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
# Gemma MLP 模型的定义，继承自 nn.Module 类
class FlaxGemmaMLP(nn.Module):
    # 指定配置参数为 GemmaConfig 类型
    config: GemmaConfig
    # 指定数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模型初始化方法
    def setup(self):
        # 获取嵌入维度
        embed_dim = self.config.hidden_size
        # 获取内部维度，如果未指定则设为 4 倍的嵌入维度
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        # 使用正态分布初始化器初始化核矩阵，范围为配置中的 initializer_range
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        
        # 如果隐藏激活函数未指定，发出警告并设置为 'gelu_pytorch_tanh'
        if self.config.hidden_activation is None:
            logger.warning_once(
                "Gemma's activation function should be approximate GeLU and not exact GeLU. "
                "Changing the activation function to `gelu_pytorch_tanh`."
                f"if you want to use the legacy `{self.config.hidden_act}`, "
                f"edit the `model.config` to set `hidden_activation={self.config.hidden_act}` "
                "  instead of `hidden_act`. See https://github.com/huggingface/transformers/pull/29402 for more details."
            )
            hidden_activation = "gelu_pytorch_tanh"
        else:
            # 否则使用配置中指定的隐藏激活函数
            hidden_activation = self.config.hidden_activation
        
        # 根据激活函数名从预定义的 ACT2FN 字典中获取对应的激活函数
        self.act = ACT2FN[hidden_activation]

        # 初始化门控投影层，使用内部维度，不使用偏置，指定数据类型和核初始化器
        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 初始化下投影层，使用嵌入维度，不使用偏置，指定数据类型和核初始化器
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 初始化上投影层，使用内部维度，不使用偏置，指定数据类型和核初始化器
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    # 模型调用方法
    def __call__(self, hidden_states):
        # 上投影操作，将隐藏状态映射到内部维度空间
        up_proj_states = self.up_proj(hidden_states)
        # 门控状态，通过激活函数处理门控投影层的输出
        gate_states = self.act(self.gate_proj(hidden_states))

        # 下投影操作，将上投影状态乘以门控状态，映射到嵌入维度空间
        hidden_states = self.down_proj(up_proj_states * gate_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 从 transformers.models.llama.modeling_flax_llama.FlaxLlamaDecoderLayer 复制而来，将 Llama 改为 Gemma
class FlaxGemmaDecoderLayer(nn.Module):
    # 指定配置参数为 GemmaConfig 类型
    config: GemmaConfig
    # 指定数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模型初始化方法
    def setup(self):
        # 初始化输入层归一化，使用 GemmaRMSNorm 类处理配置和数据类型
        self.input_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        # 初始化自注意力层，使用 GemmaAttention 类处理配置和数据类型
        self.self_attn = FlaxGemmaAttention(self.config, dtype=self.dtype)
        # 初始化注意力后归一化层，使用 GemmaRMSNorm 类处理配置和数据类型
        self.post_attention_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        # 初始化 MLP 层，使用 GemmaMLP 类处理配置和数据类型
        self.mlp = FlaxGemmaMLP(self.config, dtype=self.dtype)

    # 模型调用方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        ):
            residual = hidden_states
            # 应用输入层归一化
            hidden_states = self.input_layernorm(hidden_states)
            # 使用自注意力机制处理隐藏状态
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

            residual = hidden_states
            # 应用自注意力后归一化
            hidden_states = self.post_attention_layernorm(hidden_states)
            # 应用多层感知机（MLP）
            hidden_states = self.mlp(hidden_states)
            # 残差连接
            hidden_states = residual + hidden_states

            return (hidden_states,) + outputs[1:]
# 从transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel复制而来，替换GPTNeo为Gemma，GPT_NEO为GEMMA，transformer为model
class FlaxGemmaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用GemmaConfig作为配置类
    config_class = GemmaConfig
    # 基础模型的前缀为"model"
    base_model_prefix = "model"
    # 模块类未定义
    module_class: nn.Module = None

    def __init__(
        self,
        config: GemmaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的config和dtype初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        # 生成位置ID，广播以匹配输入形状
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 分割随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法生成随机参数
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            # 如果有提供参数，则与随机参数合并
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        """
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小。定义初始化缓存的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用模块的初始化方法生成缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回解冻的缓存变量
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
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
        # 如果指定了输出注意力的选项，则使用指定的值；否则使用默认配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果指定了输出隐藏状态的选项，则使用指定的值；否则使用默认配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了返回字典的选项，则使用指定的值；否则使用默认配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入张量的批量大小和序列长度
        batch_size, sequence_length = input_ids.shape

        # 如果未提供位置编码，则根据序列长度和批量大小创建一个默认的位置编码
        if position_ids is None:
            if past_key_values is not None:
                # 如果传递了过去的键值，但未提供位置编码，则引发值错误
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
            # 使用序列长度创建一个二维数组，用于表示位置编码
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果未提供注意力掩码，则创建一个全为1的注意力掩码数组
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理任何需要的随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 准备输入参数字典，包括模型参数或者自身保存的参数
        inputs = {"params": params or self.params}

        # 如果传递了过去的键值，将它们作为缓存输入，并将"cache"标记为可变
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 调用模块的 apply 方法，执行模型的前向计算
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

        # 如果传递了过去的键值并且要求返回字典，则将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        # 如果传递了过去的键值但不要求返回字典，则将更新后的缓存添加到模型输出的第一个元素中
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回模型的输出
        return outputs
# 从 transformers.models.llama.modeling_flax_llama.FlaxLlamaLayerCollection 复制而来，将 Llama 替换为 Gemma
class FlaxGemmaLayerCollection(nn.Module):
    config: GemmaConfig  # 类型注解，指定 config 为 GemmaConfig 类型
    dtype: jnp.dtype = jnp.float32  # 类型注解，指定 dtype 默认为 jnp.float32

    def setup(self):
        # 初始化 self.blocks 列表，其中每个元素为一个 FlaxGemmaDecoderLayer 实例，根据 config.num_hidden_layers 的值进行循环创建
        self.blocks = [
            FlaxGemmaDecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

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
        # 如果 output_attentions 为 True，则初始化 all_attentions 为空元组，否则为 None
        all_attentions = () if output_attentions else None
        # 如果 output_hidden_states 为 True，则初始化 all_hidden_states 为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历 self.blocks 列表中的每个 block
        for block in self.blocks:
            # 如果 output_hidden_states 为 True，则将当前 hidden_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # 调用 block 的 __call__ 方法，传递参数并接收返回的 layer_outputs
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新 hidden_states 为 layer_outputs 的第一个元素（通常是模型的输出）
            hidden_states = layer_outputs[0]

            # 如果 output_attentions 为 True，则将当前层的注意力矩阵添加到 all_attentions 中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 输出包含可能为 None 值的元组 outputs，`FlaxGemmaModule` 将会过滤掉这些 None 值
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


# 从 transformers.models.llama.modeling_flax_llama.FlaxLlamaModule 复制而来，将 Llama 替换为 Gemma
class FlaxGemmaModule(nn.Module):
    config: GemmaConfig  # 类型注解，指定 config 为 GemmaConfig 类型
    dtype: jnp.dtype = jnp.float32  # 类型注解，指定 dtype 默认为 jnp.float32

    def setup(self):
        # 初始化 hidden_size 为 config.hidden_size
        self.hidden_size = self.config.hidden_size
        # 使用正态分布初始化 embed_tokens，形状为 (config.vocab_size, self.hidden_size)，dtype 为 self.dtype
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        # 初始化 layers 为 FlaxGemmaLayerCollection 实例，传递 config 和 dtype
        self.layers = FlaxGemmaLayerCollection(self.config, dtype=self.dtype)
        # 初始化 norm 为 FlaxGemmaRMSNorm 实例，传递 config 和 dtype
        self.norm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)

    # 忽略复制
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
        # ...
        ):
            # 使用模型的嵌入层处理输入的标识符（转换为整数类型）
            input_embeds = self.embed_tokens(input_ids.astype("i4"))

            # 根据论文中建议的缩放因子对嵌入向量进行缩放
            input_embeds = input_embeds * (self.config.hidden_size**0.5)

            # 将输入嵌入向量传递给模型的多层网络进行处理
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

            # 提取模型处理后的隐藏状态向量
            hidden_states = outputs[0]

            # 对隐藏状态向量进行规范化处理
            hidden_states = self.norm(hidden_states)

            # 如果需要输出所有隐藏状态向量，则构建包含所有隐藏状态的元组
            if output_hidden_states:
                all_hidden_states = outputs[1] + (hidden_states,)
                outputs = (hidden_states, all_hidden_states) + outputs[2:]
            else:
                # 否则，只输出规范化后的隐藏状态向量
                outputs = (hidden_states,) + outputs[1:]

            # 如果不需要返回字典形式的输出，则过滤掉值为None的输出结果
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 返回FlaxBaseModelOutput对象，包含最后的隐藏状态、所有隐藏状态和注意力权重
            return FlaxBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=outputs[1],
                attentions=outputs[-1],
            )
# 定义一个 FlaxGemmaModel 类，继承自 FlaxGemmaPreTrainedModel，用于 Gemma 模型的 transformer 输出原始隐藏状态，没有额外的特定头部。
# 这个类被修改自 transformers.models.llama.modeling_flax_llama.FlaxLlamaModel，其中 Llama 被替换成 Gemma。

@add_start_docstrings(
    "The bare Gemma Model transformer outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
class FlaxGemmaModel(FlaxGemmaPreTrainedModel):
    module_class = FlaxGemmaModule


# 为 FlaxGemmaModel 类添加样例调用文档字符串，用于检查点、配置等文档化信息。
append_call_sample_docstring(
    FlaxGemmaModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)


# 定义一个 FlaxGemmaForCausalLMModule 类，继承自 nn.Module，表示带因果语言建模头部的 Gemma 模型模块。
# 这个类被复制自 transformers.models.llama.modeling_flax_llama.FlaxLlamaForCausalLMModule，其中 Llama 被替换成 Gemma。
class FlaxGemmaForCausalLMModule(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxGemmaModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 忽略复制
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

        hidden_states = outputs[0]
        # 如果配置指定共享词嵌入，则共享的核心来自模型的参数中的嵌入
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        # 如果不返回字典，则返回 logits 和可能的其他输出
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回 FlaxCausalLMOutput 对象，包含 logits、隐藏状态和注意力信息
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings(
    """
    The Gemma Model transformer with a language modeling head (linear layer) on top.
    """,
    GEMMA_START_DOCSTRING,
)
# 定义一个 FlaxGemmaForCausalLM 类，继承自 FlaxGemmaPreTrainedModel，表示带因果语言建模头部的 Gemma 模型。
# 这个类被复制自 transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM，其中 GPTJ 被替换成 Gemma。
class FlaxGemmaForCausalLM(FlaxGemmaPreTrainedModel):
    module_class = FlaxGemmaForCausalLMModule
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        # 获取输入张量的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用模型的方法初始化缓存，返回过去键值对
        past_key_values = self.init_cache(batch_size, max_length)

        # 注意：通常需要在 attention_mask 的 x > input_ids.shape[-1] 和 x < cache_length 的位置上放置 0。
        # 但由于 Gemma 使用因果掩码，这些位置已经被掩盖了。
        # 因此我们可以在这里创建一个静态的 attention_mask，这对编译更有效率。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 计算位置编码，累积求和并减去 1
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用 lax.dynamic_update_slice 更新 extended_attention_mask 的部分区域
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有提供 attention_mask，则使用广播方式创建位置编码
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入字典，包含过去键值对、扩展后的注意力掩码和位置编码
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新输入用于生成的模型参数
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新位置编码为最后一个位置的下一个位置
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数 `append_call_sample_docstring`，添加示例文档字符串到类 `FlaxGemmaForCausalLM` 上
# 使用变量 `_CHECKPOINT_FOR_DOC` 作为实际检查点的示例
# 将类 `FlaxCausalLMOutput` 作为输出配置信息的示例
# 使用变量 `_CONFIG_FOR_DOC` 作为配置信息的示例
# 使用变量 `_REAL_CHECKPOINT_FOR_DOC` 作为真实检查点的示例
append_call_sample_docstring(
    FlaxGemmaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)
```