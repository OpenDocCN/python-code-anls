# `.\transformers\models\bloom\modeling_flax_bloom.py`

```py
# 指定编码为UTF-8
# 版权声明，版权归HuggingFace Inc.团队和Bigscience Workshop所有
#
# 根据Apache许可证2.0版（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
"""Flax BLOOM模型。"""

import math
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutput,
)
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bloom import BloomConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/bloom"
_CONFIG_FOR_DOC = "BloomConfig"

# BLOOM模型的文档起始部分
BLOOM_START_DOCSTRING = r"""

    该模型继承自[`FlaxPreTrainedModel`]。检查超类文档以获取库实现的所有通用方法
    （例如下载或保存、调整输入嵌入、修剪头等）。

    该模型还是Flax Linen的子类
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html)。将其用作
    常规Flax模块，并参考Flax文档了解所有与常规使用和行为相关的事项。

    最后，该模型支持固有的JAX功能，例如：

    - [即时（JIT）编译](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [矢量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
            参数：config ([`BloomConfig`])：包含模型所有参数的模型配置类。
            使用配置文件进行初始化不会加载与模型相关联的权重，只会加载配置。
            查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。

        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
            计算的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在 GPU 上）和 `jax.numpy.bfloat16`（在 TPU 上）之一。

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.
            这可以用于在 GPU 或 TPU 上启用混合精度训练或半精度推理。如果指定了，则所有计算将使用给定的 `dtype` 执行。

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
            **注意，这只是指定计算的数据类型，不影响模型参数的数据类型。**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
            如果您希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
```  
"""
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
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = jnp.array(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=jnp.float32)
    # 创建一个包含从1到最接近的2的幂的数组，数据类型为32位浮点数
    powers = jnp.arange(1, 1 + closest_power_of_2, dtype=jnp.float32)
    # 计算基数的各个幂次方
    slopes = jax.lax.pow(base, powers)

    # 如果最接近的2的幂不等于头数，则需要添加额外的斜率
    if closest_power_of_2 != num_heads:
        # 计算额外斜率的基数
        extra_base = jnp.array(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=jnp.float32)
        # 计算额外头数
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        # 创建额外斜率的幂次方数组
        extra_powers = jnp.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=jnp.float32)
        # 将额外斜率的幂次方添加到斜率数组中
        slopes = jnp.cat([slopes, jax.lax.pow(extra_base, extra_powers)], axis=0)

    # 注意：Alibi张量将被添加到将应用于查询的注意力偏置，键的乘积的注意力
    # 因此，Alibi将需要具有形状(batch_size, num_heads, query_length, key_length)
    # => 在这里，我们设置(batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # 以便查询长度维度随后将正确广播。
    # 这与T5的相对位置偏置几乎完全相同：
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    # 创建一个张量，表示累积的注意力掩码，并且将最后一个元素减1，然后乘以原始注意力掩码，得到相应位置
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    # 计算Alibi张量，即斜率乘以相应位置的张量
    alibi = slopes[..., None] * arange_tensor
    # 在第三个维度上添加一个维度，以匹配Alibi张量的形状
    alibi = jnp.expand_dims(alibi, axis=2)
    # 将Alibi张量转换为NumPy数组并返回
    return jnp.asarray(alibi, dtype)
class FlaxBloomAttention(nn.Module):
    # 定义一个名为FlaxBloomAttention的类，继承自nn.Module
    config: BloomConfig
    # 定义一个名为config的属性，类型为BloomConfig
    dtype: jnp.dtype = jnp.float32
    # 定义一个名为dtype的属性，类型为jnp.dtype，默认值为jnp.float32

    def setup(self):
        # 定义一个名为setup的方法，用于初始化模型
        self.hidden_size = self.config.hidden_size
        # 将hidden_size设置为config中的hidden_size属性值
        self.num_heads = self.config.n_head
        # 将num_heads设置为config中的n_head属性值
        self.head_dim = self.hidden_size // self.num_heads
        # 计算每个头的维度
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        # 检查是否需要在softmax中使用fp32

        if self.head_dim * self.num_heads != self.hidden_size:
            # 如果头的维度乘以头的数量不等于隐藏层的大小
            raise ValueError(
                f"`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )
            # 抛出数值错误异常

        dense = partial(
            nn.Dense,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 定义一个名为dense的函数，用于创建Dense层

        self.query_key_value = dense(self.hidden_size * 3)
        # 使用dense函数创建query_key_value属性
        self.dense = dense(self.hidden_size)
        # 使用dense函数创建dense属性
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout)
        # 创建一个Dropout层，用于残差连接

    def _split_heads(self, hidden_states):
        # 定义一个名为_split_heads的方法，用于将隐藏状态分割成多个头
        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_heads, self.head_dim * 3))
        # 返回分割后的隐藏状态

    def _merge_heads(self, hidden_states):
        # 定义一个名为_merge_heads的方法，用于合并多个头
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))
        # 返回合并后的隐藏状态

    @nn.compact
    # 使用nn.compact装饰器

    # Copied from transformers.models.gptj.modeling_flax_gptj.FlaxGPTJAttention._concatenate_to_cache
    # 将投影后的键和值状态与前几步的缓存状态连接起来。此函数是从官方的 Flax 仓库中稍作修改而来。
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # 检测是否通过现有缓存数据进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 初始化缓存的键和值，如果未初始化，则用零值填充。
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 初始化缓存索引，如果未初始化，则将其设为0。
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存的键和值的形状信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键、值缓存，使用新的一维空间切片
            cur_index = cache_index.value
            # 构建切片的索引
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 更新缓存的键和值
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置应仅关注已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    # 通过调用函数将注意力层应用于输入的隐藏状态。
    def __call__(
        self,
        hidden_states,
        residual,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
class BloomGELU(nn.Module):
    # 设置模型的数据类型为 float32
    def setup(self):
        self.dtype = jnp.float32

    # 定义模型的前向传播过程，使用 BloomGELU 函数
    def __call__(self, x):
        # BloomGELU 激活函数
        return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class FlaxBloomMLP(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    # 模型初始化过程
    def setup(self):
        # 从配置中获取隐藏层大小
        hidden_size = self.config.hidden_size

        # 使用正态分布初始化权重
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        # 定义全连接层
        self.dense_h_to_4h = nn.Dense(4 * hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.dense_4h_to_h = nn.Dense(hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        # 使用 BloomGELU 激活函数
        self.act = BloomGELU()

    # 定义模型的前向传播过程
    def __call__(self, hidden_states, residual, deterministic: bool = True):
        # 第一个全连接层和激活函数
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)

        # 第二个全连接层
        intermediate_output = self.dense_4h_to_h(hidden_states)

        # 残差连接
        intermediate_output = intermediate_output + residual

        # 使用 dropout
        hidden_states = self.hidden_dropout(intermediate_output, deterministic=deterministic)

        return hidden_states


class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    # 模型初始化过程
    def setup(self):
        # 输入层归一化
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 自注意力机制
        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype)
        # 注意力后归一化
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # MLP 层
        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype)

        # 是否在后归一化时应用残差连接
        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        # 是否使用隐藏层 dropout
        self.hidden_dropout = self.config.hidden_dropout

    # 定义模型的前向传播过程
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        # 对输入进行 layer normalization 处理
        layernorm_output = self.input_layernorm(hidden_states)

        # 如果配置需要在 layer normalization 之后应用残差连接
        if self.apply_residual_connection_post_layernorm:
            # 使用 layer normalization 的输出作为残差
            residual = layernorm_output
        else:
            # 否则使用原始输入作为残差
            residual = hidden_states

        # 自注意力机制
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

        # 获取其他输出（如果有的话）
        outputs = attn_outputs[1:]

        # 在自注意力输出之后进行 layer normalization
        post_layernorm = self.post_attention_layernorm(attention_output)

        # 根据配置设置残差
        if self.apply_residual_connection_post_layernorm:
            # 如果配置需要在 layer normalization 之后应用残差连接，则使用 layer normalization 后的输出
            residual = post_layernorm
        else:
            # 否则使用自注意力输出作为残差
            residual = attention_output

        # 经过 MLP 网络的处理
        output = self.mlp(post_layernorm, residual, deterministic=deterministic)

        # 将 MLP 处理后的输出加入到输出元组中
        outputs = (output,) + outputs

        # 返回输出元组
        return outputs
class FlaxBloomPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为BloomConfig
    config_class = BloomConfig
    # 指定基础模型的前缀为"transformer"
    base_model_prefix = "transformer"
    # 模块类
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
        # 根据传入的配置和其他参数创建模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        # 切分随机数生成器
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的init方法初始化参数
        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)["params"]

        # 如果已有参数，则将缺失的参数用随机参数填充
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
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        # 使用模块的init方法初始化变量并返回解冻的缓存
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
        # 如果 output_attentions 不为 None，则使用传入的值，否则使用模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 不为 None，则使用传入的值，否则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 不为 None，则使用传入的值，否则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入张量的形状，批次大小和序列长度
        batch_size, sequence_length = input_ids.shape

        # 如果 attention_mask 为 None，则创建全 1 的注意力掩码张量
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理任何 PRNG（伪随机数生成器），如果需要的话
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 准备输入参数，如果 params 不为 None 则使用传入的参数，否则使用模型的参数
        inputs = {"params": params or self.params}

        # 如果传入了 past_key_values，则将其作为缓存传递给模型，确保使用缓存
        # 必须确保将缓存标记为可变，以便可以由 FlaxBloomAttention 模块更改
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模型，获取输出结果
        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            not train,  # 在推断模式下不训练
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # 将更新后的缓存添加到模型输出中
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            # 将更新后的缓存插入到模型输出中的适当位置
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs
# 定义一个名为FlaxBloomBlockCollection的类，继承自nn.Module
class FlaxBloomBlockCollection(nn.Module):
    # 声明类属性config，类型为BloomConfig
    config: BloomConfig
    # 声明类属性dtype，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义setup方法
    def setup(self):
        # 通过列表推导式创建self.layers列表，其中每个元素是一个FlaxBloomBlock对象
        self.layers = [
            FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype)
            for layer_number in range(self.config.num_hidden_layers)
        ]

    # 定义__call__方法
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
        # 如果需要输出注意力矩阵，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None

        # 循环遍历每个隐藏层
        for layer_number in range(self.config.num_hidden_layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的__call__方法，计算当前层的输出
            layer_outputs = self.layers[layer_number](
                hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到all_attentions元组中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 构建输出元组，包括隐藏状态、所有隐藏状态和所有注意力矩阵
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 返回输出元组
        return outputs


# 定义一个名为FlaxBloomModule的类，继承自nn.Module
class FlaxBloomModule(nn.Module):
    # 声明类属性config，类型为BloomConfig
    config: BloomConfig
    # 声明类属性dtype，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义setup方法
    def setup(self):
        # 初始化词嵌入维度为隐藏大小
        self.embed_dim = self.config.hidden_size

        # 创建词嵌入层，使用正态分布初始化参数
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )

        # 创建词嵌入层后的LayerNorm层
        self.word_embeddings_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # 创建transformer层集合
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype)

        # 创建最终的LayerNorm层
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    # 定义__call__方法
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
        # 使用词嵌入层将输入的词索引转换为词嵌入向量
        inputs_embeds = self.word_embeddings(input_ids)
        # 进行嵌入后的层归一化处理
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        # 根据 `attention_mask` 构建 alibi 张量
        alibi = build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)

        # 将隐藏状态、alibi、attention_mask 等传递给 self.h 处理
        outputs = self.h(
            hidden_states,
            alibi=alibi,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # 获取处理后的隐藏状态
        hidden_states = outputs[0]
        # 对最终隐藏状态进行层归一化处理
        hidden_states = self.ln_f(hidden_states)

        # 如果需要输出所有隐藏状态
        if output_hidden_states:
            # 将所有隐藏状态组合成元组
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 返回输出结果的元组形式
            return tuple(v for v in [outputs[0], outputs[-1]] if v is not None)

        # 返回带有过去和交叉注意力的 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 导入所需模块或函数
@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",  # 添加文档字符串，描述此模型的输出
    BLOOM_START_DOCSTRING,  # 引用预定义的 Bloom 模型的文档字符串
)
# 从 transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoModel 复制而来，将 GPTNeo 替换为 Bloom
class FlaxBloomModel(FlaxBloomPreTrainedModel):  # 定义 FlaxBloomModel 类，继承自 FlaxBloomPreTrainedModel
    module_class = FlaxBloomModule  # 设置模型类别为 FlaxBloomModule


# 添加调用示例的文档字符串
append_call_sample_docstring(FlaxBloomModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


# 定义 FlaxBloomForCausalLMModule 类，继承自 nn.Module
class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig  # 模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32

    def setup(self):  # 设置模型结构
        self.transformer = FlaxBloomModule(self.config, dtype=self.dtype)  # 初始化 Bloom 模型
        self.lm_head = nn.Dense(  # 定义语言建模头部
            self.config.vocab_size,  # 输出维度为词汇表大小
            use_bias=False,  # 不使用偏置
            dtype=self.dtype,  # 数据类型
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),  # 使用正态分布初始化
        )

    def __call__(  # 定义对象的调用行为
        self,
        input_ids,  # 输入的标识符
        attention_mask,  # 注意力掩码
        deterministic: bool = True,  # 是否确定性计算
        init_cache: bool = False,  # 是否初始化缓存
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否返回字典格式的输出
    ):
        outputs = self.transformer(  # 对输入进行 Bloom 模型的转换
            input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # 获取隐藏状态输出

        if self.config.tie_word_embeddings:  # 如果词嵌入被绑定
            shared_kernel = self.transformer.variables["params"]["word_embeddings"]["embedding"].T  # 获取共享的核
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)  # 应用语言建模头部
        else:
            lm_logits = self.lm_head(hidden_states)  # 应用语言建模头部

        if not return_dict:  # 如果不返回字典
            return (lm_logits,) + outputs[1:]  # 返回元组形式的结果

        return FlaxCausalLMOutput(  # 返回 FlaxCausalLMOutput 对象
            logits=lm_logits,  # 输出的逻辑回归值
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )


@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,  # 添加文档字符串，描述 Bloom 模型及其语言建模头部
    BLOOM_START_DOCSTRING,  # 引用预定义的 Bloom 模型的文档字符串
)
class FlaxBloomForCausalLM(FlaxBloomPreTrainedModel):  # 定义 FlaxBloomForCausalLM 类，继承自 FlaxBloomPreTrainedModel
    module_class = FlaxBloomForCausalLMModule  # 设置模型类别为 FlaxBloomForCausalLMModule
    # 为生成准备输入数据，包括初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 获取输入数据的批次大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 初始化缓存，获取过去的键值
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常情况下，需要在 attention_mask 中将 x > input_ids.shape[-1] 和 x < cache_length 的位置置为 0。
        # 但由于 Bloom 使用了因果掩码，这些位置已经被掩盖。因此，我们可以在这里创建一个静态的 attention_mask，
        # 这样对编译效率更高
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        # 如果存在外部提供的 attention_mask，则动态更新静态的 attention_mask
        if attention_mask is not None:
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))

        # 返回准备好的输入数据字典，包括过去的键值和扩展的注意力掩码
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
        }

    # 更新生成的输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 将模型输出的过去键值更新到模型参数中
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs
# 将样本函数的文档字符串附加到指定的类上
append_call_sample_docstring(FlaxBloomForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)
```