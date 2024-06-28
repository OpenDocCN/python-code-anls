# `.\models\bloom\modeling_flax_bloom.py`

```
# 导入所需的模块和库
import math
from functools import partial
from typing import Optional, Tuple

# 导入 Flax 相关模块
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入自定义的模型输出类和工具函数
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutput,
)
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 设置日志记录器
logger = logging.get_logger(__name__)

# 模型的预训练检查点和配置信息用于文档
_CHECKPOINT_FOR_DOC = "bigscience/bloom"
_CONFIG_FOR_DOC = "BloomConfig"

# 模型起始文档字符串，包含对模型的描述和相关链接
BLOOM_START_DOCSTRING = r"""

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
    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""
BLOOM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BloomTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def build_alibi_tensor(attention_mask: jnp.ndarray, num_heads: int, dtype: Optional[jnp.dtype] = jnp.float32):
    """
    Flax implementation of the BLOOM Alibi tensor. BLOOM Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    Link to paper: https://arxiv.org/abs/2108.12409

    Args:
        attention_mask (`jnp.ndarray`):
            Token-wise attention mask, this should be of shape `(batch_size, max_seq_len)`.
        num_heads (`int`):
            Number of attention heads.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            The data type (dtype) of the output tensor.

    Returns: Alibi tensor of shape `(batch_size * num_heads, 1, max_seq_len)`.
    """
    # 获取注意力掩码的形状，batch_size 是批量大小，seq_length 是序列长度
    batch_size, seq_length = attention_mask.shape
    # 计算最接近 num_heads 的 2 的幂次方
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    # 计算基础值，用于调整 softmax 函数的实现
    base = jnp.array(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=jnp.float32)
    # 生成一个包含最接近2的幂的整数的浮点数数组，从1到closest_power_of_2（包括）
    powers = jnp.arange(1, 1 + closest_power_of_2, dtype=jnp.float32)
    # 计算基数 base 的 powers 次幂，得到斜率数组
    slopes = jax.lax.pow(base, powers)

    # 如果 closest_power_of_2 不等于 num_heads
    if closest_power_of_2 != num_heads:
        # 计算额外的基数，用于增加的头部数量
        extra_base = jnp.array(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=jnp.float32)
        # 计算剩余头部的数量
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        # 生成额外的幂，从1到2 * num_remaining_heads，步长为2的浮点数数组
        extra_powers = jnp.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=jnp.float32)
        # 将额外计算得到的幂添加到斜率数组中
        slopes = jnp.cat([slopes, jax.lax.pow(extra_base, extra_powers)], axis=0)

    # 创建一个索引张量，用于生成 Alibi 张量，其形状为 (batch_size, num_heads, query_length, key_length)
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    # 计算 Alibi 张量，将斜率数组乘以 arange_tensor
    alibi = slopes[..., None] * arange_tensor
    # 在第三个维度上扩展 Alibi 张量
    alibi = jnp.expand_dims(alibi, axis=2)
    # 返回 Alibi 张量的 numpy 数组表示，以指定的数据类型
    return jnp.asarray(alibi, dtype)
# 定义一个名为 `FlaxBloomAttention` 的神经网络模块
class FlaxBloomAttention(nn.Module):
    # 类变量：用于存储 BloomConfig 的配置信息
    config: BloomConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数
    def setup(self):
        # 从配置中获取隐藏层大小
        self.hidden_size = self.config.hidden_size
        # 从配置中获取注意力头的数量
        self.num_heads = self.config.n_head
        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_heads
        # 检查隐藏层大小是否能被注意力头的数量整除
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        # 如果隐藏层大小不能被注意力头的数量整除，抛出数值错误异常
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        # 部分函数定义：Dense 层，设置数据类型和权重初始化方式
        dense = partial(
            nn.Dense,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化查询、键、值的 Dense 层
        self.query_key_value = dense(self.hidden_size * 3)
        # 初始化输出 Dense 层
        self.dense = dense(self.hidden_size)
        # 初始化残差 Dropout 层
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    # 将隐藏状态分割成多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_heads, self.head_dim * 3))

    # 合并多个注意力头为一个隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    # 神经网络结构的定义，装饰器标记为 nn.compact
    @nn.compact
    # 从 transformers.models.gptj.modeling_flax_gptj.FlaxGPTJAttention._concatenate_to_cache 复制的方法
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键值，如果不存在则创建全零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值，如果不存在则创建全零数组
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引，如果不存在则创建一个值为0的整数数组
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 解构缓存键的形状以获取批次维度、最大长度、注意力头数和每个头部的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引以反映已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 生成用于缓存解码器自注意力的因果掩码：我们的单个查询位置应仅关注已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 结合前面计算的掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# 定义一个基于 nn.Module 的类 BloomGELU
class BloomGELU(nn.Module):
    # 初始化函数，设置数据类型为 jnp.float32
    def setup(self):
        self.dtype = jnp.float32

    # 对象被调用时执行的函数，实现了 GELU 激活函数
    def __call__(self, x):
        # 计算 GELU 函数的输出：x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
        return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# 定义一个基于 nn.Module 的类 FlaxBloomMLP
class FlaxBloomMLP(nn.Module):
    # 类型定义为 BloomConfig 类
    config: BloomConfig
    # 数据类型设置为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置模型结构
    def setup(self):
        # 从配置中获取隐藏层大小
        hidden_size = self.config.hidden_size

        # 初始化权重的方式为正态分布，标准差为配置中的 initializer_range
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        # 创建全连接层对象，将输入维度为 hidden_size * 4，输出维度为 hidden_size
        self.dense_h_to_4h = nn.Dense(4 * hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        # 创建全连接层对象，将输入维度为 hidden_size，输出维度为 hidden_size
        self.dense_4h_to_h = nn.Dense(hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        # 创建 Dropout 层对象，丢弃率为配置中的 hidden_dropout
        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        # 创建 GELU 激活函数层对象
        self.act = BloomGELU()

    # 对象被调用时执行的函数，实现了多层感知机（MLP）的前向传播逻辑
    def __call__(self, hidden_states, residual, deterministic: bool = True):
        # 输入经过全连接层 dense_h_to_4h 和 GELU 激活函数 act
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)

        # 经过全连接层 dense_4h_to_h 得到中间输出
        intermediate_output = self.dense_4h_to_h(hidden_states)

        # 将中间输出与残差相加
        intermediate_output = intermediate_output + residual
        # 应用 Dropout 操作
        hidden_states = self.hidden_dropout(intermediate_output, deterministic=deterministic)

        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个基于 nn.Module 的类 FlaxBloomBlock
class FlaxBloomBlock(nn.Module):
    # 类型定义为 BloomConfig 类
    config: BloomConfig
    # 数据类型设置为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置模型结构
    def setup(self):
        # 输入层的 LayerNorm 操作，epsilon 为配置中的 layer_norm_epsilon
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 自注意力机制层对象 FlaxBloomAttention
        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype)
        # 后自注意力层的 LayerNorm 操作，epsilon 为配置中的 layer_norm_epsilon
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 多层感知机（MLP）对象 FlaxBloomMLP
        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype)

        # 是否在 LayerNorm 后应用残差连接的标志，从配置中获取
        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        # 隐藏层 Dropout 概率，从配置中获取
        self.hidden_dropout = self.config.hidden_dropout

    # 对象被调用时执行的函数，实现了 Bloom Transformer 中的一个 Block 的前向传播逻辑
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        ):
        # 输入经过输入层的 LayerNorm 操作
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力机制的前向传播
        hidden_states, attention_output = self.self_attention(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            init_cache=init_cache,
        )

        # 是否在自注意力后应用残差连接和 LayerNorm
        if self.apply_residual_connection_post_layernorm:
            hidden_states = hidden_states + alibi
            hidden_states = self.post_attention_layernorm(hidden_states)

        # 经过多层感知机（MLP）的前向传播
        hidden_states = self.mlp(hidden_states, alibi, deterministic=deterministic)

        # 返回处理后的 hidden_states
        return hidden_states
        ):
            # 对输入进行 layer normalization 处理
            layernorm_output = self.input_layernorm(hidden_states)

            # 如果配置要求在保存残差之前进行 layer normalization
            if self.apply_residual_connection_post_layernorm:
                # 将 layer normalization 后的结果作为残差
                residual = layernorm_output
            else:
                # 否则将未处理的隐藏状态作为残差
                residual = hidden_states

            # 进行自注意力机制
            attn_outputs = self.self_attention(
                layernorm_output,
                residual=residual,
                alibi=alibi,
                attention_mask=attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )

            # 获取自注意力机制的输出
            attention_output = attn_outputs[0]

            # 获取额外的输出
            outputs = attn_outputs[1:]

            # 在自注意力输出后进行 layer normalization
            post_layernorm = self.post_attention_layernorm(attention_output)

            # 根据配置设置残差
            if self.apply_residual_connection_post_layernorm:
                # 如果配置要求在后置 layer normalization 后使用残差
                residual = post_layernorm
            else:
                # 否则使用注意力输出作为残差
                residual = attention_output

            # 将 post-layernorm 结果和残差输入到 MLP 中进行处理
            output = self.mlp(post_layernorm, residual, deterministic=deterministic)

            # 将 MLP 的输出与其他输出合并
            outputs = (output,) + outputs

            # 返回所有输出
            return outputs
class FlaxBloomPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 BloomConfig 类作为配置类
    config_class = BloomConfig
    # 基础模型的前缀名称
    base_model_prefix = "transformer"
    # 模块类，初始化为 None
    module_class: nn.Module = None

    def __init__(
        self,
        config: BloomConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用给定的配置和参数初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 创建一个和 input_ids 相同形状的全1张量作为 attention_mask
        attention_mask = jnp.ones_like(input_ids)
        # 拆分随机数生成器为 params_rng 和 dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法生成随机参数
        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)["params"]

        if params is not None:
            # 如果提供了初始参数，则用随机生成的参数填充缺失的键
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 如果没有提供初始参数，则直接返回随机生成的参数
            return random_params

    def init_cache(self, batch_size, max_length):
        """
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 创建一个和 input_ids 相同形状的全1张量作为 attention_mask
        attention_mask = jnp.ones_like(input_ids)

        # 使用模块的初始化方法，设置 init_cache=True 来初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        past_key_values: dict = None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
            # 如果 output_attentions 不为 None，则使用指定的 output_attentions；否则使用配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 output_hidden_states 不为 None，则使用指定的 output_hidden_states；否则使用配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果 return_dict 不为 None，则使用指定的 return_dict；否则使用配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # 获取输入张量的批量大小和序列长度
            batch_size, sequence_length = input_ids.shape

            # 如果 attention_mask 为 None，则创建一个全为 1 的注意力掩码张量
            if attention_mask is None:
                attention_mask = jnp.ones((batch_size, sequence_length))

            # 如果 dropout_rng 不为 None，则将其作为随机数生成器加入到 rngs 字典中
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng

            # 准备输入参数字典，包括模型参数或者当前实例的参数
            inputs = {"params": params or self.params}

            # 如果传入了 past_key_values，则将其作为缓存传递给模型
            if past_key_values:
                inputs["cache"] = past_key_values
                # 设置 mutable 变量以确保缓存可以被修改
                mutable = ["cache"]
            else:
                mutable = False

            # 调用模型的 apply 方法，执行前向推断
            outputs = self.module.apply(
                inputs,
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                not train,
                False,
                output_attentions,
                output_hidden_states,
                return_dict,
                rngs=rngs,
                mutable=mutable,
            )

            # 如果 past_key_values 不为 None 且 return_dict 为 True，则将更新后的缓存添加到模型输出中
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            # 如果 past_key_values 不为 None 且 return_dict 为 False，则将更新后的缓存插入到模型输出的适当位置
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

            # 返回模型的输出
            return outputs
class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化模块，创建包含多个 FlaxBloomBlock 的层列表
        self.layers = [
            FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype)
            for layer_number in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # 根据是否输出注意力和隐藏状态，初始化空元组或者 None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每一层并执行前向传播
        for layer_number in range(self.config.num_hidden_layers):
            if output_hidden_states:
                # 如果输出隐藏状态，将当前隐藏状态添加到 all_hidden_states 中
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播，并更新 hidden_states
            layer_outputs = self.layers[layer_number](
                hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果输出注意力，将当前层的注意力添加到 all_attentions 中
                all_attentions += (layer_outputs[1],)

        # 输出包含可能为 None 的元组，由 FlaxBloomModule 进一步处理
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化模块，设置词嵌入维度和初始化 word embeddings 和 layernorm
        self.embed_dim = self.config.hidden_size

        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )

        self.word_embeddings_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 初始化 transformer 层集合
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype)

        # 初始化最终 layernorm
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 模块的前向传播方法，根据参数执行相应操作并返回结果
        ):
        # 使用输入的词嵌入层，将输入的词索引转换为词嵌入向量
        inputs_embeds = self.word_embeddings(input_ids)
        # 执行词嵌入向量后的层归一化操作
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        # 根据注意力掩码构建alibi（假设），其形状和数据类型与隐藏状态相匹配
        alibi = build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)

        # 将隐藏状态、alibi、注意力掩码以及其他参数传递给self.h函数进行处理
        outputs = self.h(
            hidden_states,
            alibi=alibi,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # 获取self.h函数的输出中的隐藏状态
        hidden_states = outputs[0]
        # 对最终的隐藏状态再进行一次层归一化
        hidden_states = self.ln_f(hidden_states)

        # 如果需要输出所有隐藏状态，则将其存储在all_hidden_states中
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要使用字典形式返回结果，则将输出转换为元组并去除None值
        if not return_dict:
            return tuple(v for v in [outputs[0], outputs[-1]] if v is not None)

        # 使用自定义的输出类生成包含特定属性的输出对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
# 从transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoModel复制而来，将GPTNeo替换为Bloom
class FlaxBloomModel(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomModule


append_call_sample_docstring(FlaxBloomModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 使用给定的配置创建Bloom模块
        self.transformer = FlaxBloomModule(self.config, dtype=self.dtype)
        # 创建语言模型头部，连接到Bloom模块的输出
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用Bloom模块进行前向传播
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            # 如果配置要求共享词嵌入矩阵，则获取共享的权重矩阵并应用于语言模型头部
            shared_kernel = self.transformer.variables["params"]["word_embeddings"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            # 否则直接将隐藏状态传递给语言模型头部
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            # 如果不要求返回字典，则返回元组形式的结果
            return (lm_logits,) + outputs[1:]

        # 否则将结果封装成FlaxCausalLMOutput类型并返回
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
# FlaxBloomForCausalLM的子类，添加了语言建模头部
class FlaxBloomForCausalLM(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomForCausalLMModule
    # 为生成准备输入的函数，接受输入的ID，最大长度和可选的注意力掩码
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 初始化缓存，获取输入的批次大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用self.init_cache方法初始化过去键值
        past_key_values = self.init_cache(batch_size, max_length)

        # 注意：通常需要在attention_mask中将超出input_ids.shape[-1]和cache_length之间的位置设置为0。
        # 但由于Bloom使用因果掩码，这些位置已经被屏蔽。因此，我们可以在这里创建一个静态的attention_mask，
        # 这样更有效地进行编译。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 使用lax.dynamic_update_slice将attention_mask动态更新到extended_attention_mask中的指定位置
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))

        # 返回一个包含过去键值和扩展的注意力掩码的字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
        }

    # 更新用于生成的输入，将模型输出中的过去键值添加到模型参数中
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs
# 调用一个函数来添加一个样例文档字符串
append_call_sample_docstring(FlaxBloomForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)
```