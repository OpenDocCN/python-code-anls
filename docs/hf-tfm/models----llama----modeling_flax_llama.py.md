# `.\models\llama\modeling_flax_llama.py`

```py
# 引入必要的模块和库
from functools import partial  # 导入 functools 模块中的 partial 函数，用于创建带有部分参数的新函数
from typing import Optional, Tuple  # 导入 typing 模块中的 Optional 和 Tuple 类型，用于类型标注

import flax.linen as nn  # 导入 Flax 的 linen 模块，并用 nn 别名引用
import jax  # 导入 JAX 库，用于自动求导和并行计算
import jax.numpy as jnp  # 导入 JAX 库中的 numpy 模块，并用 jnp 别名引用
import numpy as np  # 导入 numpy 库，并用 np 别名引用
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 中的 FrozenDict 等相关函数
from flax.linen import combine_masks, make_causal_mask  # 导入 Flax 中的相关函数和类
from flax.linen.attention import dot_product_attention_weights  # 导入 Flax 中的 dot_product_attention_weights 函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入 Flax 中的 flatten_dict 和 unflatten_dict 函数
from jax import lax  # 从 JAX 库中导入 lax 模块

# 导入模型相关的输出类和函数
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 导入 LLaMA 模型的配置类
from .configuration_llama import LlamaConfig

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 用于文档的配置、检查点和真实检查点的字符串常量
_CONFIG_FOR_DOC = "LlamaConfig"
_CHECKPOINT_FOR_DOC = "afmck/testing-llama-tiny"
_REAL_CHECKPOINT_FOR_DOC = "openlm-research/open_llama_3b_v2"

# LLaMA 模型的起始文档字符串，包含了模型的继承信息和特性说明
LLAMA_START_DOCSTRING = r"""

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
    # 参数:
    # config ([`LlamaConfig`]): 模型配置类，包含模型的所有参数。
    #     用配置文件初始化不会加载与模型关联的权重，仅加载配置。
    #     可以查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法来加载模型权重。
    # dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
    #     计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16`, 或 `jax.numpy.bfloat16` 中的一种。
    # 
    #     这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推断。如果指定，则所有计算将使用给定的 `dtype` 进行。
    # 
    #     **请注意，这仅指定计算的数据类型，不影响模型参数的数据类型。**
    # 
    #     如果您希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
# 创建正弦位置编码矩阵，用于将位置索引映射为正弦波形式的向量表示
def create_sinusoidal_positions(num_pos, dim):
    # 计算正弦编码的频率逆频率
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    # 计算位置索引乘以频率得到的矩阵，每个维度都是浮点数
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    # 按照最后一个维度将两个频率矩阵连接起来，形成最终的正弦位置编码矩阵
    emb = np.concatenate((freqs, freqs), axis=-1)
    # 将 emb 数组中的每个元素应用正弦函数，然后与对应元素应用余弦函数的结果拼接起来
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    # 从拼接后的数组中取出前 num_pos 列，并转换为 JAX 数组格式返回
    return jnp.array(out[:, :, :num_pos])
# 定义一个函数，用于将输入张量的后一半隐藏维度旋转
def rotate_half(tensor):
    # 将张量按照其最后一个维度的一半进行拼接，实现旋转操作
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


# 定义一个函数，将旋转的位置嵌入应用到输入张量上
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    # 将输入张量乘以余弦位置编码，然后加上经过旋转半隐藏维度的正弦位置编码
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


# 定义一个名为FlaxLlamaRMSNorm的类，继承自nn.Module
class FlaxLlamaRMSNorm(nn.Module):
    # 类的配置信息
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    # 设置方法，在类实例化时调用，用于初始化权重和其他参数
    def setup(self):
        # 设置 epsilon 参数为 RMS 归一化的小数值
        self.epsilon = self.config.rms_norm_eps
        # 初始化权重矩阵，形状为隐藏大小（hidden_size）
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    # 类的调用方法，对隐藏状态进行处理
    def __call__(self, hidden_states):
        # 将隐藏状态转换为 jax 数组，并将数据类型设置为 jnp.float32
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        # 计算方差的平方
        variance = jnp.power(variance, 2)
        # 求取方差的平均值，保持最后一个维度
        variance = variance.mean(-1, keepdims=True)
        # 根据 RMS 归一化公式，将隐藏状态除以标准差加上一个小值 epsilon
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        # 返回加权后的隐藏状态
        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


# 定义一个名为FlaxLlamaRotaryEmbedding的类，继承自nn.Module
class FlaxLlamaRotaryEmbedding(nn.Module):
    # 类的配置信息
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    # 设置方法，在类实例化时调用，用于初始化位置编码
    def setup(self):
        # 计算每个注意力头的维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 创建正弦余弦位置编码
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

    # 类的调用方法，将位置编码应用到键、查询和位置ID上
    def __call__(self, key, query, position_ids):
        # 获取指定位置ID的正弦余弦位置编码
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        # 将正弦余弦位置编码应用到键和查询上
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        # 将键和查询转换为 jax 数组，并将数据类型设置为 self.dtype
        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        # 返回处理后的键和查询
        return key, query


# 定义一个名为FlaxLlamaAttention的类，继承自nn.Module
class FlaxLlamaAttention(nn.Module):
    # 类的配置信息
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    # 设置方法，在类实例化时调用，用于初始化注意力机制的参数
    def setup(self):
        # 从配置中获取参数
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        # 创建偏置注意力层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化查询、键、值和输出投影层
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.o_proj = dense()

        # 创建因果遮罩
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        # 创建旋转嵌入层
        self.rotary_emb = FlaxLlamaRotaryEmbedding(config, dtype=self.dtype)

    # 内部方法，用于将隐藏状态分割为多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
    def _merge_heads(self, hidden_states):
        # 将输入的 hidden_states 重塑成形状为 (batch_size, sequence_length, self.embed_dim) 的张量，并返回
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    # 从 transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache 复制而来
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 如果未初始化，则创建形状和类型与 key 相同的零张量作为 cached_key
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 如果未初始化，则创建形状和类型与 value 相同的零张量作为 cached_value
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 如果未初始化，则创建初始值为 0 的 cache_index
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取当前缓存张量的形状信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的 1 维空间切片更新 key、value 缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新 cache_index 值，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存的因果掩码：我们的单个查询位置只应关注已生成和缓存的键位置，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 将 pad_mask 与 attention_mask 结合
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        ):
            # 使用投影函数计算查询向量
            query = self.q_proj(hidden_states)
            # 使用投影函数计算键向量
            key = self.k_proj(hidden_states)
            # 使用投影函数计算值向量
            value = self.v_proj(hidden_states)

            # 将查询向量分割成多个头
            query = self._split_heads(query)
            # 将键向量分割成多个头
            key = self._split_heads(key)
            # 将值向量分割成多个头
            value = self._split_heads(value)

            # 应用旋转位置编码器到键和查询向量
            key, query = self.rotary_emb(key, query, position_ids)

            # 获取查询向量和键向量的长度
            query_length, key_length = query.shape[1], key.shape[1]

            # 构建因果掩码
            if self.has_variable("cache", "cached_key"):
                # 如果有缓存的键，根据缓存索引和最大解码器长度动态切片因果掩码
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                # 否则，使用静态切片获取因果掩码
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]

            # 获取批次大小
            batch_size = hidden_states.shape[0]
            # 将因果掩码广播到与查询向量和键向量匹配的形状
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            # 广播注意力掩码以匹配因果掩码的形状
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            # 结合注意力掩码和因果掩码
            attention_mask = combine_masks(attention_mask, causal_mask)

            # 初始化 dropout RNG
            dropout_rng = None
            if not deterministic and self.config.attention_dropout > 0.0:
                dropout_rng = self.make_rng("dropout")

            # 在快速自回归解码期间，逐步一次性输入一个位置，逐步缓存键和值。
            if self.has_variable("cache", "cached_key") or init_cache:
                key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

            # 将布尔掩码转换为浮点数掩码
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

            # 标准点积注意力
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

            # 如果需要，将注意力权重转换为指定的数据类型
            if self.attention_softmax_in_fp32:
                attn_weights = attn_weights.astype(self.dtype)

            # 使用注意力权重计算注意力输出
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
            # 合并多头得到的注意力输出
            attn_output = self._merge_heads(attn_output)
            # 应用输出投影层
            attn_output = self.o_proj(attn_output)

            # 准备输出，包括注意力输出和注意力权重（如果需要）
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs
class FlaxLlamaMLP(nn.Module):
    config: LlamaConfig  # 类型注解：指定该类的配置信息来自于LlamaConfig类
    dtype: jnp.dtype = jnp.float32  # 类型注解：指定数据类型为jnp.float32，默认为浮点数类型

    def setup(self):
        embed_dim = self.config.hidden_size  # 从配置中获取隐藏层大小作为嵌入维度
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim
        # 计算内部层维度，如果配置中有中间大小定义则使用，否则使用默认值4倍的嵌入维度

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        # 使用正态分布初始化器初始化核参数，范围由配置的initializer_range定义
        self.act = ACT2FN[self.config.hidden_act]
        # 从ACT2FN字典中获取激活函数，并存储在act属性中，其类型由配置的hidden_act指定

        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 创建具有inner_dim大小的全连接层，不使用偏置，使用上述初始化器初始化权重
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 创建具有embed_dim大小的全连接层，不使用偏置，使用上述初始化器初始化权重
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        # 创建具有inner_dim大小的全连接层，不使用偏置，使用上述初始化器初始化权重

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        # 使用up_proj层处理输入的隐藏状态
        gate_states = self.act(self.gate_proj(hidden_states))
        # 使用激活函数act处理gate_proj层处理后的隐藏状态

        hidden_states = self.down_proj(up_proj_states * gate_states)
        # 使用down_proj层处理up_proj_states与gate_states的乘积，并将结果存储在隐藏状态中
        return hidden_states
        # 返回处理后的隐藏状态作为结果


class FlaxLlamaDecoderLayer(nn.Module):
    config: LlamaConfig  # 类型注解：指定该类的配置信息来自于LlamaConfig类
    dtype: jnp.dtype = jnp.float32  # 类型注解：指定数据类型为jnp.float32，默认为浮点数类型

    def setup(self):
        self.input_layernorm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)
        # 创建一个使用LlamaConfig和指定数据类型的FlaxLlamaRMSNorm实例，存储在input_layernorm属性中
        self.self_attn = FlaxLlamaAttention(self.config, dtype=self.dtype)
        # 创建一个使用LlamaConfig和指定数据类型的FlaxLlamaAttention实例，存储在self_attn属性中
        self.post_attention_layernorm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)
        # 创建一个使用LlamaConfig和指定数据类型的FlaxLlamaRMSNorm实例，存储在post_attention_layernorm属性中
        self.mlp = FlaxLlamaMLP(self.config, dtype=self.dtype)
        # 创建一个使用LlamaConfig和指定数据类型的FlaxLlamaMLP实例，存储在mlp属性中

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
        # 将输入的隐藏状态存储在变量residual中，用于残差连接
        hidden_states = self.input_layernorm(hidden_states)
        # 使用input_layernorm对隐藏状态进行规范化处理

        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # 使用self_attn处理规范化后的隐藏状态，传递额外参数attention_mask、position_ids等，并将结果存储在outputs中

        attn_output = outputs[0]
        # 从outputs中获取注意力机制的输出
        hidden_states = residual + attn_output
        # 将residual与注意力输出相加得到新的隐藏状态

        residual = hidden_states
        # 将新的隐藏状态存储在变量residual中，用于下一步的残差连接
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 使用post_attention_layernorm对新的隐藏状态进行规范化处理
        hidden_states = self.mlp(hidden_states)
        # 使用mlp处理规范化后的隐藏状态，得到最终的输出

        hidden_states = residual + hidden_states
        # 将残差连接的结果与MLP处理后的隐藏状态相加，作为最终的输出

        return (hidden_states,) + outputs[1:]
        # 返回包含最终输出和outputs中其他项的元组


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel with GPTNeo->Llama, GPT_NEO->LLAMA, transformer->model
class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    # 指定配置类为LlamaConfig
    base_model_prefix = "model"
    # 指定基础模型前缀为"model"
    module_class: nn.Module = None
    # 指定模块类为nn.Module，初始值为None
    def __init__(
        self,
        config: LlamaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置和参数初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法，传入配置、模块对象以及其他参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建与 input_ids 相同形状的全 1 张量作为 attention_mask
        attention_mask = jnp.ones_like(input_ids)
        # 根据 input_ids 的维度生成位置编码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 利用输入的随机种子分割出两个随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        # 将随机数生成器存入字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 利用模块的初始化方法初始化参数
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        # 如果提供了额外的参数，则将随机初始化的参数与提供的参数进行合并
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                fast auto-regressive decoding 使用的批大小。定义初始化缓存的批大小。
            max_length (`int`):
                auto-regressive decoding 的最大可能长度。定义初始化缓存的序列长度。
        """
        # 初始化输入变量以检索缓存
        input_ids = jnp.ones((batch_size, max_length))
        # 创建与 input_ids 相同形状的全 1 张量作为 attention_mask
        attention_mask = jnp.ones_like(input_ids)
        # 根据 input_ids 的形状生成位置编码
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 利用模块的初始化方法初始化变量，并指定初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        # 省略了 __call__ 方法的注释，因为该方法通过装饰器 @add_start_docstrings_to_model_forward 添加了文档字符串
        ):
        # 如果没有显式提供输出注意力的设置，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式提供输出隐藏状态的设置，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式提供返回字典的设置，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入张量的批量大小和序列长度
        batch_size, sequence_length = input_ids.shape

        # 如果未提供位置编码，则根据序列长度创建默认位置编码
        if position_ids is None:
            # 如果传入了过去的键值（past_key_values），则需要明确提供位置编码，否则抛出异常
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
            
            # 使用广播操作将序列长度范围内的数组扩展为指定批次大小的位置编码张量
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果未提供注意力遮罩，则创建全1的注意力遮罩张量
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理任何需要的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 准备输入参数字典，包括模型参数或者传入的参数
        inputs = {"params": params or self.params}

        # 如果传入了过去的键值（past_key_values），则将其作为缓存传递给模型，确保缓存是可变的以便后续更新
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模型的正向传播，传递所有必要的输入张量和设置
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

        # 如果传入了过去的键值（past_key_values）并且需要返回字典，则将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        # 如果传入了过去的键值（past_key_values）但不需要返回字典，则将更新后的缓存添加到模型输出元组中
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回模型的输出结果
        return outputs
class FlaxLlamaLayerCollection(nn.Module):
    # LlamaConfig 类型的配置信息
    config: LlamaConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建一系列 FlaxLlamaDecoderLayer 对象并存储在 self.blocks 中
        self.blocks = [
            FlaxLlamaDecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    # 调用实例时执行的方法
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
        # 如果输出注意力矩阵，则初始化空的元组 all_attentions
        all_attentions = () if output_attentions else None
        # 如果输出隐藏状态，则初始化空的元组 all_hidden_states
        all_hidden_states = () if output_hidden_states else None

        # 遍历 self.blocks 中的每个 FlaxLlamaDecoderLayer 对象
        for block in self.blocks:
            # 如果输出隐藏状态，则将当前隐藏状态 hidden_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 调用 block 对象，计算层的输出结果 layer_outputs
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新 hidden_states 为当前层的输出结果中的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力矩阵，则将当前层的注意力矩阵添加到 all_attentions 中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 输出结果包括 hidden_states, all_hidden_states, all_attentions
        # 注意：all_hidden_states 和 all_attentions 可能包含 None 值，由 FlaxLlamaModule 进行过滤处理
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLlamaModule(nn.Module):
    # LlamaConfig 类型的配置信息
    config: LlamaConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 设置隐藏大小为 config 中的隐藏大小
        self.hidden_size = self.config.hidden_size
        # 使用正态分布初始化 embed_tokens 层，存储在 self.embed_tokens 中
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        # 创建 FlaxLlamaLayerCollection 对象并存储在 self.layers 中
        self.layers = FlaxLlamaLayerCollection(self.config, dtype=self.dtype)
        # 创建 FlaxLlamaRMSNorm 对象并存储在 self.norm 中
        self.norm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)

    # 调用实例时执行的方法
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
        # 省略部分代码，未提供完整内容
    # 使用给定的输入 ID 创建输入的嵌入表示，数据类型转换为32位整数
    input_embeds = self.embed_tokens(input_ids.astype("i4"))
    
    # 将输入的嵌入表示传递给模型的层进行处理，并返回处理后的输出结果
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
    
    # 从模型输出中获取隐藏状态，索引为0的元素为模型的最后隐藏状态
    hidden_states = outputs[0]
    
    # 对隐藏状态进行归一化处理
    hidden_states = self.norm(hidden_states)
    
    # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到所有隐藏状态列表中
    if output_hidden_states:
        all_hidden_states = outputs[1] + (hidden_states,)
        outputs = (hidden_states, all_hidden_states) + outputs[2:]
    else:
        outputs = (hidden_states,) + outputs[1:]
    
    # 如果不需要以字典形式返回结果，则返回所有非空的输出值的元组
    if not return_dict:
        return tuple(v for v in outputs if v is not None)
    
    # 如果需要以字典形式返回结果，则使用 FlaxBaseModelOutput 类封装最后的隐藏状态、所有隐藏状态和注意力值
    return FlaxBaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=outputs[1],
        attentions=outputs[-1],
    )
# 添加起始文档字符串和元数据到 FlaxLlamaModel 类，说明它是一个裸 Llama 模型变换器，输出原始隐藏状态，没有特定的顶部头部。
# 使用 LLAMA_START_DOCSTRING 定义的起始文档字符串作为补充信息。
@add_start_docstrings(
    "The bare Llama Model transformer outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class FlaxLlamaModel(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaModule


# 向 FlaxLlamaModel 类添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxLlamaModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)


# 定义 FlaxLlamaForCausalLMModule 类，用于支持因果语言建模任务
class FlaxLlamaForCausalLMModule(nn.Module):
    # 模块配置参数
    config: LlamaConfig
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 使用给定的配置参数和数据类型创建 Llama 模型
        self.model = FlaxLlamaModule(self.config, dtype=self.dtype)
        # 创建语言建模头部，一个全连接层，用于生成词汇表大小的输出
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 定义模块的调用方法
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
        # 调用 Llama 模型来处理输入序列
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

        # 获取模型的隐藏状态
        hidden_states = outputs[0]
        # 使用语言建模头部生成最终的语言建模输出
        lm_logits = self.lm_head(hidden_states)

        # 如果不要求返回字典格式的输出，则返回元组形式的输出
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        # 返回格式化后的因果语言建模输出
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 向 FlaxLlamaForCausalLM 类添加起始文档字符串，说明它是带有语言建模头部的 Llama 模型变换器
@add_start_docstrings(
    """
    The Llama Model transformer with a language modeling head (linear layer) on top.
    """,
    LLAMA_START_DOCSTRING,
)
# 从 transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM 复制到 FlaxLlamaForCausalLM，
# 并将其中的 GPTJ 替换为 Llama
class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaForCausalLMModule
    # 为生成准备输入数据的方法
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存
        batch_size, seq_length = input_ids.shape

        # 使用模型的初始化方法创建缓存
        past_key_values = self.init_cache(batch_size, max_length)

        # 注意：通常需要在 attention_mask 中对超出 input_ids.shape[-1] 和 cache_length 之外的位置置为 0。
        # 但由于 Llama 使用因果注意力机制，这些位置已经被掩码处理。
        # 因此，在这里我们可以创建一个单一的静态 attention_mask，这样更高效地进行编译。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果有传入 attention_mask，则根据它计算 position_ids
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，根据序列长度广播创建 position_ids
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回生成所需的输入数据字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成过程中的输入数据的方法
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新模型关键值缓存和 position_ids
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        
        # 返回更新后的输入数据字典
        return model_kwargs
# 将样本文档字符串附加到指定类中的方法上
append_call_sample_docstring(
    # 目标类：FlaxLlamaForCausalLM，用于添加文档字符串
    FlaxLlamaForCausalLM,
    # 用于文档的检查点对象的名称或引用：_CHECKPOINT_FOR_DOC
    _CHECKPOINT_FOR_DOC,
    # 生成的文档字符串应描述的输出对象类型：FlaxCausalLMOutput
    FlaxCausalLMOutput,
    # 用于文档的配置对象的名称或引用：_CONFIG_FOR_DOC
    _CONFIG_FOR_DOC,
    # 实际使用的检查点对象的名称或引用：_REAL_CHECKPOINT_FOR_DOC
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)
```