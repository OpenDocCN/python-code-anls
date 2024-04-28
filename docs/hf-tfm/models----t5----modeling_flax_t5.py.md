# `.\transformers\models\t5\modeling_flax_t5.py`

```
# 设置脚本编码为 UTF-8
# 版权声明，版权归 T5 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用本文件，未经许可不得使用
# 可在以下地址获取许可证详情：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可发布的软件均按“原样”分发，没有任何形式的担保和条件，无论是明示的还是暗示的，
# 请查看许可证获取具体语言关于权限和限制的说明。
# Flax T5 模型

# 导入必要的库和模块
import copy
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_t5 import T5Config

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置说明
_CHECKPOINT_FOR_DOC = "t5-small"
_CONFIG_FOR_DOC = "T5Config"

# 引入 remat 函数
remat = nn_partitioning.remat

# 从 transformers.models.bart.modeling_flax_bart 模块中拷贝的 shift_tokens_right 函数
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    将输入的标记向右移动一个位置。
    """
    # 为向右移动后的输入分配内存空间
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将原始输入从第 2 个位置开始复制到新的分配内存空间中
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 将解码器起始标记放到新分配内存空间的第一个位置
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    
    # 将所有 -100 的位置替换为 pad_token_id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    
    return shifted_input_ids

# FlaxT5LayerNorm 类，用于自定义 LayerNorm 操作
class FlaxT5LayerNorm(nn.Module):
    # 初始化参数包括隐藏层大小，数据类���，默认浮点数，epsilon 值，默认为 1e-6，以及权重初始化函数，默认为 ones
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    eps: float = 1e-6
    weight_init: Callable[..., np.ndarray] = jax.nn.initializers.ones

    # 初始化方法
    def setup(self):
        # 定义模型参数 weight，维度为隐藏层大小
        self.weight = self.param("weight", self.weight_init, (self.hidden_size,))
    # 定义一个方法，接受隐藏状态作为输入
    def __call__(self, hidden_states):
        """
        Construct a layernorm module in the T5 style; No bias and no subtraction of mean.
        构建一个T5风格的layernorm模块；无偏置和无均值减法。
        """
        # layer norm should always be calculated in float32
        # 层归一化应始终以float32计算
        # 计算隐藏状态的方差，并在最后一个维度上取均值，得到一个保持维度的数组
        variance = jnp.power(hidden_states.astype("f4"), 2).mean(axis=-1, keepdims=True)
        # 对隐藏状态进行除法操作，除以方差和self.eps的平方根
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        # 返回加权的隐藏状态
        return self.weight * hidden_states
# 定义一个名为 FlaxT5DenseActDense 的自定义神经网络模块，继承自 nn.Module
class FlaxT5DenseActDense(nn.Module):
    # 设置模块的配置属性为 T5Config 类型
    config: T5Config
    # 定义计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块的初始化方法，在这里设置模块的属性和参数
    def setup(self):
        # 计算输入到 Dense 层的初始化标准差
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        # 计算输出到 Dense 层的初始化标准差
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 定义输入到 Dense 层的权重矩阵 wi
        self.wi = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定权重矩阵的数据类型
        )
        # 定义输出到 Dense 层的权重矩阵 wo
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定权重矩阵的数据类型
        )
        # 定义一个 Dropout 层，用于随机失活神经元
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 根据配置选择激活函数
        self.act = ACT2FN[self.config.dense_act_fn]

    # 定义模块的调用方法，即在前向传播中执行的操作
    def __call__(self, hidden_states, deterministic=True):
        # 将输入 hidden_states 传入 Dense 层 wi
        hidden_states = self.wi(hidden_states)
        # 将 Dense 层的输出通过激活函数 act 处理
        hidden_states = self.act(hidden_states)
        # 对处理后的 hidden_states 进行 Dropout 处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 Dropout 后的 hidden_states 传入 Dense 层 wo
        hidden_states = self.wo(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 FlaxT5DenseGatedActDense 的自定义神经网络模块，继承自 nn.Module
class FlaxT5DenseGatedActDense(nn.Module):
    # 设置模块的配置属性为 T5Config 类型
    config: T5Config
    # 定义计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    # 模块的初始化方法，在这里设置模块的属性和参数
    def setup(self):
        # 计算输入到 Dense 层的初始化标准差
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        # 计算输出到 Dense 层的初始化标准差
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 定义输入到 Dense 层的权重矩阵 wi_0
        self.wi_0 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定权重矩阵的数据类型
        )
        # 定义输入到 Dense 层的权重矩阵 wi_1
        self.wi_1 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定权重矩阵的数据类型
        )
        # 定义输出到 Dense 层的权重矩阵 wo
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 指定权重矩阵的数据类型
        )
        # 定义一个 Dropout 层，用于随机失活神经元
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 根据配置选择激活函数
        self.act = ACT2FN[self.config.dense_act_fn]

    # 定义模块的调用方法，即在前向传播中执行的操作
    def __call__(self, hidden_states, deterministic):
        # 将输入 hidden_states 传入 Dense 层 wi_0，并通过激活函数 act 处理
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将输入 hidden_states 传入 Dense 层 wi_1
        hidden_linear = self.wi_1(hidden_states)
        # 将 gated 激活函数的结果与线性传递的结果相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对处理后的 hidden_states 进行 Dropout 处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 Dropout 后的 hidden_states 传入 Dense 层 wo
        hidden_states = self.wo(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 FlaxT5LayerFF 的自定义神经网络模块，继承自 nn.Module
class FlaxT5LayerFF(nn.Module):
    # 设置模块的配置属性为 T5Config 类型
    config: T5Config
    # 定义计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 设置模型
    def setup(self):
        # 如果配置为门控激活函数，则使用门控激活函数版本的DenseReluDense模块
        if self.config.is_gated_act:
            self.DenseReluDense = FlaxT5DenseGatedActDense(self.config, dtype=self.dtype)
        else:
            # 否则使用普通版本的DenseReluDense模块
            self.DenseReluDense = FlaxT5DenseActDense(self.config, dtype=self.dtype)

        # 初始化LayerNorm模块
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 初始化Dropout模块
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 对输入的hidden_states进行处理
    def __call__(self, hidden_states, deterministic=True):
        # 对hidden_states进行LayerNorm处理
        forwarded_states = self.layer_norm(hidden_states)
        # 传递到DenseReluDense模块进行处理
        forwarded_states = self.DenseReluDense(forwarded_states, deterministic=deterministic)
        # 将处理后的结果与原始的hidden_states相加，并进行Dropout处理
        hidden_states = hidden_states + self.dropout(forwarded_states, deterministic=deterministic)
        # 返回处理后的hidden_states
        return hidden_states
# 定义一个名为FlaxT5Attention的类，该类继承自nn.Module
class FlaxT5Attention(nn.Module):
    # 初始化类的属性：T5Config类型的config，默认值为False的has_relative_attention_bias，默认值为False的causal，jnp.float32类型的dtype
    config: T5Config
    has_relative_attention_bias: bool = False
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 定义setup方法，用于初始化类的属性
    def setup(self):
        # 设置相对注意力机制的参数
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 根据配置参数计算初始化权重的标准差
        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 初始化全连接层，用于计算Query、Key、Value和Output
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        # 如果有相对注意力偏置，初始化Embed层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
                dtype=self.dtype,
            )

    # 定义静态方法
    @staticmethod
        def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
            """
            定义一个函数，将相对位置转换为相对注意力的桶号。
            相对位置定义为 memory_position - query_position，即从关注位置到被关注位置的令牌距离。
            如果 bidirectional=False，则正相对位置是无效的。
            对于小绝对相对位置，我们使用较小的桶，对于较大的绝对相对位置，我们使用较大的桶。
            所有大于或等于 max_distance 的相对位置映射到同一个桶。
            所有小于或等于- max_distance 的相对位置映射到同一个桶。
            这应该能够更优雅地推广到模型已经训练的更长序列。
            """
            relative_buckets = 0
            如果是双向的，将桶的数量减半
            if bidirectional:
                num_buckets //= 2
                relative_buckets += (relative_position > 0) * num_buckets
                使用绝对值函数计算相对位置
                relative_position = jnp.abs(relative_position)
            else:
                将相对位置限制在非正值范围内
                relative_position = -jnp.clip(relative_position, a_max=0)
            # 目前相对位置的范围为 [0, 无穷)

            # 一半的桶用于精确的相对位置增量
            max_exact = num_buckets // 2
            is_small = relative_position < max_exact

            # 其余的桶用于最大距离为 max_distance 时位置上的对数增加
            relative_position_if_large = max_exact + (
                jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
            )
            将相对增量限制在合适的范围内
            relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

            relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

            return relative_buckets.astype("i4")

        def compute_bias(self, query_length, key_length):
            """计算分箱后的相对位置偏差"""
            创建一个表示查询长度的数组
            context_position = jnp.arange(query_length, dtype="i4")[:, None]
            创建一个表示键长度的数组
            memory_position = jnp.arange(key_length, dtype="i4")[None, :]

            计算相对位置
            relative_position = memory_position - context_position
            将相对位置分箱，得到相对位置偏差
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.causal),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            通过相对位置偏差计算相对注意力偏差值
            values = self.relative_attention_bias(relative_position_bucket)
            对数值进行一定的变换
            values = values.transpose((2, 0, 1))[None, :, :, :]
            返回计算的偏差值
            return values

        def _split_heads(self, hidden_states):
            将隐藏状态按照头数和键值投影维度进行重塑
            return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))
    # 将隐藏状态重新组合成特定维度的形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        # 通过缺少现有缓存数据来检测是否正在初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 创建或获取变量，存储键对应的缓存数据
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 创建或获取变量，存储值对应的缓存数据
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 创建或获取变量，存储缓存数据的索引，用于更新缓存
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            # 使用新的1D空间切片更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键、值数据
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            # 更新缓存数据的索引
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions
            # that have already been generated and cached, not the remaining zero elements.
            # 用于缓存的解码器自注意力的因果掩蔽：我们的单个查询位置只应关注已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def _create_position_bias(
        self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
        # 检查缓存是否已填充，并且是否存在“cache”和“cached_key”变量，且未初始化缓存
        cache_is_filled = self.causal and self.has_variable("cache", "cached_key") and (not init_cache)
        # 计算键的长度
        key_length = key_states.shape[1]
        # 如果缓存已填充，则查询长度等于键的长度，否则等于查询状态的长度
        query_length = key_length if cache_is_filled else query_states.shape[1]

        # 如果存在相对注意力偏置，则计算位置偏置
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length, key_length)
        # 如果存在注意力掩码，则创建与注意力掩码相同形状的零矩阵作为位置偏置
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        # 如果不存在相对注意力偏置且不存在注意力掩码，则创建指定形状的零矩阵作为位置偏置
        else:
            position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

        # 如果键和值已经计算，则只取最后一个查询位置的偏置
        if cache_is_filled:
            # 获取缓存中键的最大长度
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # 对位置偏置进行切片操作
            position_bias = jax.lax.dynamic_slice(
                position_bias,
                (0, 0, causal_attention_mask_shift, 0),
                (1, self.n_heads, seq_length, max_decoder_length),
            )
        # 返回位置偏置
        return position_bias

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        use_cache=False,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
# 定义 FlaxT5LayerSelfAttention 类，继承自 nn.Module
class FlaxT5LayerSelfAttention(nn.Module):
    # 配置属性，类型为 T5Config
    config: T5Config
    # 是否包含相对注意力偏差
    has_relative_attention_bias: bool = False
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 类的初始化方法
    def setup(self):
        # 创建 FlaxT5Attention 对象，并赋值给 SelfAttention 属性
        self.SelfAttention = FlaxT5Attention(
            self.config,
            has_relative_attention_bias=self.has_relative_attention_bias,
            causal=self.config.causal,
            dtype=self.dtype,
        )
        # 创建 FlaxT5LayerNorm 对象，并赋值给 layer_norm 属性
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建 nn.Dropout 对象，并赋值给 dropout 属性
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 对象被调用时的执行方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        # 对 hidden_states 进行 Layer Normalization，并赋值给 normed_hidden_states
        normed_hidden_states = self.layer_norm(hidden_states)
        # 对 normed_hidden_states 进行 Self Attention 操作，并赋值给 attention_output
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 计算 hidden_states 与 attention_output[0] 的和，并通过 dropout 操作，赋值给 hidden_states
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 记录输出结果，包括 hidden_states 和 attention_output 中的其它返回值（如果需要的话）
        outputs = (hidden_states,) + attention_output[1:]
        # 返回输出结果
        return outputs

# 定义 FlaxT5LayerCrossAttention 类，继承自 nn.Module
class FlaxT5LayerCrossAttention(nn.Module):
    # 配置属性，类型为 T5Config
    config: T5Config
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 类的初始化方法
    def setup(self):
        # 创建 FlaxT5Attention 对象，并赋值给 EncDecAttention 属性
        self.EncDecAttention = FlaxT5Attention(
            self.config, has_relative_attention_bias=False, causal=False, dtype=self.dtype
        )
        # 创建 FlaxT5LayerNorm 对象，并赋值给 layer_norm 属性
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建 nn.Dropout 对象，并赋值给 dropout 属性
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 对象被调用时的执行方法
    def __call__(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
    ):
        # 对 hidden_states 进行 Layer Normalization，并赋值给 normed_hidden_states
        normed_hidden_states = self.layer_norm(hidden_states)
        # 对 normed_hidden_states 进行交叉 Attention 操作，并赋值给 attention_output
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 计算 hidden_states 与 attention_output[0] 的和，并通过 dropout 操作，赋值给 hidden_states
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 记录输出结果，包括 hidden_states 和 attention_output 中的其它返回值（如果需要的话）
        outputs = (hidden_states,) + attention_output[1:]
        # 返回输出结果
        return outputs

# 定义 FlaxT5Block 类，继承自 nn.Module
class FlaxT5Block(nn.Module):
    # 配置属性，类型为 T5Config
    config: T5Config
    # 是否包含相对注意力偏差
    has_relative_attention_bias: bool = False
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        # 从配置中获取是否启用因果注意力
        self.causal = self.config.causal
        # 初始化层
        self.layer = (
            # 创建 FlaxT5LayerSelfAttention 实例
            FlaxT5LayerSelfAttention(
                self.config,
                # 检查是否有相对位置编码
                has_relative_attention_bias=self.has_relative_attention_bias,
                # 设置层的名称为字符串 "0"
                name=str(0),
                # 设置数据类型
                dtype=self.dtype,
            ),
        )
        # 初始化 feed_forward_index 变量
        feed_forward_index = 1
        # 如果启用因果注意力
        if self.causal:
            # 将 FlaxT5LayerCrossAttention 实例添加到层列表中
            self.layer += (FlaxT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            # 更新 feed_forward_index
            feed_forward_index += 1

        # 将 FlaxT5LayerFF 实例添加到层列表中
        self.layer += (FlaxT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        return_dict=True,
        deterministic=True,
        init_cache=False,
    ):
        # 使用第一个层进行自注意力计算
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 更新 hidden_states 为自注意力的输出
        hidden_states = self_attention_outputs[0]
        # 保留自注意力输出和相对位置权重
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # 检查是否需要进行交叉注意力
        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            # 使用第二个层进行交叉注意力计算
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            # 更新 hidden_states 为交叉注意力的输出
            hidden_states = cross_attention_outputs[0]

            # 保留交叉注意力输出和相对位置权重
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # 应用 Feed Forward 层
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

        # 组装输出
        outputs = (hidden_states,)

        # 将注意力输出添加到输出元组中
        outputs = outputs + attention_outputs

        # 返回隐藏状态、现有的键值状态、自注意力位置偏差、自注意力权重、交叉注意力位置偏差和交叉注意力权重
        return outputs
# 定义一个名为 FlaxT5LayerCollection 的类，继承自 nn.Module
class FlaxT5LayerCollection(nn.Module):
    # 定义一个名为 config 的属性，类型为 T5Config
    config: T5Config
    # 定义一个名为 has_relative_attention_bias 的布尔属性
    has_relative_attention_bias: bool
    # 定义一个名为 dtype 的属性，默认值为 jnp.float32，表示计算的数据类型

    # 定义 setup 方法
    def setup(self):
        # 创建一个名为 layer 的 FlaxT5Block 对象
        self.layer = FlaxT5Block(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )

    # 定义 __call__ 方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        # 调用 layer 对象的方法，传入相应的参数并返回结果
        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )


# 定义一个名为 FlaxT5BlockCollection 的类，继承自 nn.Module
class FlaxT5BlockCollection(nn.Module):
    # 定义一个名为 config 的属性，类型为 T5Config
    config: T5Config
    # 定义一个名为 dtype 的属性，默认值为 jnp.float32，表示计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 定义一个名为 gradient_checkpointing 的布尔属性，默认值为 False

    # 定义 setup 方法
    def setup(self):
        # 获取 config 属性的 causal 值
        self.causal = self.config.causal
        # 如果 gradient_checkpointing 为真
        if self.gradient_checkpointing:
            # 创建名为 FlaxT5CheckpointLayer 的对象
            FlaxT5CheckpointLayer = remat(FlaxT5LayerCollection, static_argnums=(6, 7, 8))
            # 遍历 config 中的层数，创建对应数量的 FlaxT5CheckpointLayer 对象并存储在 blocks 中
            self.blocks = [
                FlaxT5CheckpointLayer(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]
        else:
            # 遍历 config 中的层��，创建对应数量的 FlaxT5LayerCollection 对象并存储在 blocks 中
            self.blocks = [
                FlaxT5LayerCollection(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]

    # 定义 __call__ 方法
    def __call__(
        self,
        hidden_states=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        # 如果需要生成头部掩码，进行准备工作
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.causal) else None
        position_bias = None
        encoder_decoder_position_bias = None

        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                # 如果需要输出隐藏状态，将当前的隐藏状态加入到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 对当前层进行前向传播
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                output_attentions,
                deterministic,
                init_cache,
            )

            # 更新隐藏状态为当前层的输出的第一个元素，也就是处理后的隐藏状态
            hidden_states = layer_outputs[0]

            # 将当前层的自注意力位置偏置更新为当前层的输出的第二个元素
            position_bias = layer_outputs[1]

            # 如果当前 Transformer 模型同时包含自注意力和交叉注意力，并且是自回归模型，则将交叉注意力的位置偏置更新为当前层的输出的第二个元素
            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            # 如果需要输出注意力机制，将当前层的注意力机制加入到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        # 返回 Transformer 模型的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
class FlaxT5Stack(nn.Module):
    config: T5Config  # 类属性：T5 模型配置
    embed_tokens: nn.Embed  # 类属性：嵌入层，用于将输入标识符转换为嵌入向量
    dtype: jnp.dtype = jnp.float32  # 类属性：计算的数据类型，默认为单精度浮点数
    gradient_checkpointing: bool = False  # 类属性：是否使用梯度检查点

    def setup(self):
        self.causal = self.config.causal  # 初始化属性：是否生成因果关系

        # 初始化属性：T5 模型块集合
        self.block = FlaxT5BlockCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 初始化属性：最终层归一化
        self.final_layer_norm = FlaxT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 初始化属性：Dropout 层
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        input_ids=None,  # 输入标识符
        attention_mask=None,  # 注意力掩码
        encoder_hidden_states=None,  # 编码器隐藏状态
        encoder_attention_mask=None,  # 编码器注意力掩码
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否返回字典形式的输出
        deterministic: bool = True,  # 是否确定性计算
        init_cache: bool = False,  # 是否初始化缓存
    ):
        # 获取输入的嵌入表示
        hidden_states = self.embed_tokens(input_ids)
        # 对嵌入表示进行 Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 调用 T5 模型块集合进行前向传播
        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        # 提取模型输出中的隐藏状态
        hidden_states = outputs[0]

        # 对隐藏状态进行最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 再次对隐藏状态进行 Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 添加最后一层的隐藏状态到所有隐藏状态
        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据返回字典参数选择返回形式
        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]
            return (hidden_states,) + outputs[1:]

        # 返回带有过去和跨注意力的输出
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


T5_ENCODE_INPUTS_DOCSTRING = r"""
    # 输入参数：
    # input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
    #   输入序列标记在词汇表中的索引。T5是一个带有相对位置编码的模型，因此您应该能够在右侧和左侧都能够填充输入。
    #   可以使用[`AutoTokenizer`]获得索引。有关详细信息，请参阅[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。
    #   为了了解有关如何为预训练准备`input_ids`的更多信息，请查看[T5 Training](./t5#training)。
    # attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
    #   避免对填充标记索引执行注意力的掩码。掩码值在`[0，1]`中选择：
    #   - 1表示**未被掩码**的标记，
    #   - 0表示**被掩码**的标记。
    #   [什么是注意力掩码？](../glossary#attention-mask)
    # output_attentions (`bool`, *optional*):
    #   是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的`attentions`。
    # output_hidden_states (`bool`, *optional*):
    #   是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的`hidden_states`。
    # return_dict (`bool`, *optional*):
    #   是否返回[`~utils.ModelOutput`]而不是普通元组。
"""
T5_DECODE_INPUTS_DOCSTRING = r"""
# T5 解码器的输入参数文档字符串

    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            # 解码器输入序列标记在词汇表中的索引

            # 索引可以使用 [`AutoTokenizer`]. 参见 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`] 的详细信息。

            # [什么是解码器输入 ID?](../glossary#decoder-input-ids)

            # 在训练时，需要提供 `decoder_input_ids`。
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            # 元组由 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`) 组成
            # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*）是一个隐藏状态的序列，
            # 在编码器的最后一层的输出。在解码器的跨注意力中使用。
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
            # 用于避免在填充标记索引上执行注意力操作的掩码。选择的掩码值范围为`[0, 1]`：

            # - 1 代表 **未被掩码** 的标记，
            # - 0 代表 **被掩码** 的标记。

            # [什么是注意力掩码?](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *可选*):
            # 默认行为：生成一个忽略 `decoder_input_ids` 中的填充标记的张量。默认还将使用因果掩码。

            # 如果要更改填充行为，您应该根据需要进行修改。有关默认策略的更多信息，请参见[该论文的图1](
            # https://arxiv.org/abs/1910.13461)。
        past_key_values (`Dict[str, np.ndarray]`, *可选*, 通过 `init_cache` 或从以前的 `past_key_values` 传递时返回):
            # 预先计算的隐藏状态字典（在注意力块中的键和值），可用于快速自回归解码。预先计算的键和值隐藏状态的形状为 *[batch_size, max_length]*。
        output_attentions (`bool`, *可选*):
            # 是否返回所有注意力层的注意力张量。请参见返回的张量下的 `attentions`，了解更多详细信息。
        output_hidden_states (`bool`, *可选*):
            # 是否返回所有层的隐藏状态。请参见返回的张量下的 `hidden_states`，了解更多详细信息。
        return_dict (`bool`, *可选*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
"""


T5_INPUTS_DOCSTRING = r"""
# T5 的输入参数文档字符串
"""


class FlaxT5PreTrainedModel(FlaxPreTrainedModel):
    """
    # 一个处理权重初始化的抽象类，并提供预训练模型的下载和加载的简单接口。
    """

    # 配置类
    config_class = T5Config
    # 基础模型前缀
    base_model_prefix = "transformer"
    # 模块类
    module_class: nn.Module = None
    # 定义初始化方法，初始化 Transformer 模型
    def __init__(
        self,
        config: T5Config,  # T5 模型的配置信息
        input_shape: Tuple[int] = (1, 1),  # 输入数据的形状
        seed: int = 0,  # 随机种子
        dtype: jnp.dtype = jnp.float32,  # 数据类型
        _do_init: bool = True,  # 是否进行初始化
        gradient_checkpointing: bool = False,  # 是否使用梯度检查点
        **kwargs,
    ):
        # 创建模型对象
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 开启梯度检查点
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 初始化模型权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")

        # 初始化注意力掩码
        attention_mask = jnp.ones_like(input_ids)
        args = [input_ids, attention_mask]

        # 如果模型类不在 [FlaxT5EncoderModule] 中，则还需初始化解码器的输入张量和注意力掩码
        if self.module_class not in [FlaxT5EncoderModule]:
            decoder_input_ids = jnp.ones_like(input_ids)
            decoder_attention_mask = jnp.ones_like(input_ids)
            args.extend([decoder_input_ids, decoder_attention_mask])

        # 拆分随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用随机参数初始化模型
        random_params = self.module.init(
            rngs,
            *args,
        )["params"]

        # 如果存在预训练参数，则使用预训练参数，并补充缺失的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 实现模型的前向传播
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: jnp.ndarray = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        # 如果输出注意力权重未指定，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果解码器输入 ID 未提供，则引发 ValueError
        if decoder_input_ids is None:
            raise ValueError(
                "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                " here."
            )

        # 准备编码器输入的注意力掩码，如果未提供，则使用全 1
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 准备解码器输入的注意力掩码，如果未提供，则使用全 1
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 如果需要处理任何 PRNG，则添加到 RNG 字典中
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 应用 Transformer 模块，传递所需的参数
        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # 如果是推理模式，则确定性为 True，否则为 False
            deterministic=not train,
            rngs=rngs,
        )
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                fast自回归解码使用的batch大小。定义了初始化缓存时的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs`由(`last_hidden_state`，可选：`hidden_states`，可选：`attentions`)组成。
                `last_hidden_state`的形状为`（batch_size，sequence_length，hidden_size）`，（可选）
                是来自编码器最后一层的隐藏状态序列。在解码器的交叉注意力中使用。
    
        初始化缓存时的输入变量
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
    
        定义内部函数_decoder_forward用来执行解码器的前向计算
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )
    
        使用self.module的init方法初始化变量init_variables，并将decoder_input_ids、decoder_attention_mask、
        encoder_outputs[0]（编码器最后一层的隐藏状态）等作为参数传入，设置init_cache为True，设置method为_decoder_forward。
        这里只需要调用解码器进行初始化缓存，无需进行后续的完整解码操作。
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,
        )
    
        返回解码器缓存
        return unfreeze(init_variables["cache"])
    
    @add_start_docstrings(T5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=T5Config)
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    
    注释完毕。
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 如果未提供 output_attentions 参数，则设置为配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供 output_hidden_states 参数，则设置为配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供 return_dict 参数，则设置为配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果 attention_mask 未提供，则创建一个与 input_ids 相同形状的全 1 数组
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理可能存在的 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, **kwargs)

        # 应用模型的 encoder 部分
        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=T5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
# T5 模型的文档字符串，介绍了 T5 模型的来源和特点
T5_START_DOCSTRING = r"""
    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

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
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
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

# 在 T5 模型上新增的文档信息，说明了 T5 模型的基本结构
@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-stateswithout any specific head on top.",
    T5_START_DOCSTRING,
)
# 定义了 FlaxT5Module 类
class FlaxT5Module(nn.Module):
    # T5 模��的配置
    config: T5Config
    # 计算的数据类型，默认为 jax.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 获取编码器部分
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器部分
    def _get_decoder_module(self):
        return self.decoder
    # 在模型初始化的时候设置共享的嵌入矩阵，该矩阵用于编码器和解码器的输入
    def setup(self):
        # 初始化共享的嵌入矩阵
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        # 复制编码器配置并设置为非因果（非自回归）
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        # 初始化编码器
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # 复制解码器配置并设置为因果（自回归），然后重新设置解码器层数
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        # 初始化解码器
        self.decoder = FlaxT5Stack(
            decoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 模型的调用方法，接受多个输入参数，并返回对应输出
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果需要编码（训练或第一次预测），则进行编码
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 解码
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果不需要返回字典，则将解码器和编码器的输出一起返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 构造 FlaxSeq2SeqModelOutput 类型的输出
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
class FlaxT5Model(FlaxT5PreTrainedModel):
    module_class = FlaxT5Module



# 将 FlaxT5Model 的 module_class 属性设置为 FlaxT5Module 类
class FlaxT5Model(FlaxT5PreTrainedModel):
    module_class = FlaxT5Module


append_call_sample_docstring(FlaxT5Model, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

FLAX_T5_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxT5Model

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
    >>> model = FlaxT5Model.from_pretrained("t5-small")

    >>> input_ids = tokenizer(
    ...     "Studies have been shown that owning a dog is good for you", return_tensors="np"
    ... ).input_ids
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids

    >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    >>> decoder_input_ids = model._shift_right(decoder_input_ids)

    >>> # forward pass
    >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 将 FlaxT5Model 的调用示例和文档字符串追加到指定的 FlaxT5Model 上
append_call_sample_docstring(FlaxT5Model, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义 FlaxT5Model 的文档字符串
FLAX_T5_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxT5Model

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
    >>> model = FlaxT5Model.from_pretrained("t5-small")

    >>> input_ids = tokenizer(
    ...     "Studies have been shown that owning a dog is good for you", return_tensors="np"
    ... ).input_ids
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids

    >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    >>> decoder_input_ids = model._shift_right(decoder_input_ids)

    >>> # forward pass
    >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 用指定的文档字符串覆盖 FlaxT5Model 的调用文档字符串
overwrite_call_docstring(FlaxT5Model, T5_INPUTS_DOCSTRING + FLAX_T5_MODEL_DOCSTRING)

# 在 FlaxT5Model 上追加或替换返回的文档字符串
append_replace_return_docstrings(FlaxT5Model, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class FlaxT5EncoderModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # Encode if needed (training, first prediction pass)
        # 根据需要编码（训练、第一次预测）
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        return encoder_outputs


class FlaxT5EncoderModel(FlaxT5PreTrainedModel):
    # 将FlaxT5EncoderModule赋值给module_class变量
    module_class = FlaxT5EncoderModule

    # 添加文档字符串到模型前向传播方法中
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        # 如果output_attentions未指定，则使用self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states未指定，则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict未指定，则使用self.config.return_dict
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理需要的PRNG
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 将参数，输入id，注意力掩码等传递给module的apply方法，并返回结果
        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )
```  
# 给 FlaxT5ForConditionalGenerationModule 类添加文档字符串，描述其是带有语言建模头部的 T5 模型
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class FlaxT5ForConditionalGenerationModule(nn.Module):
    # 模型配置
    config: T5Config
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 梯度检查点标志，默认为 False
    gradient_checkpointing: bool = False

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 设置函数
    def setup(self):
        # 模型维度
        self.model_dim = self.config.d_model

        # 共享嵌入层，将词汇表映射到模型维度的向量
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

        # 编码器配置
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        # 创建编码器
        self.encoder = FlaxT5Stack(
            encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 解码器配置
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers

        # 创建解码器
        self.decoder = FlaxT5Stack(
            decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 语言建模头部，输出词汇表大小的向量，不使用偏置
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

    # 模型调用函数，执行模型的前向传播
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
```  
        ): 
            # 如果 return_dict 不为 None，则使用传入的 return_dict，否则使用配置文件中的 use_return_dict
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # 对输入进行编码
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
            )

            hidden_states = encoder_outputs[0]

            # 对输出进行解码
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic,
            )

            sequence_output = decoder_outputs[0]

            if self.config.tie_word_embeddings:
                # 在词汇表上进行投影之前重新缩放输出
                # 参考 https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)

            if self.config.tie_word_embeddings:
                # 如果分享词嵌入，则获取共享的嵌入矩阵进行计算
                shared_embedding = self.shared.variables["params"]["embedding"]
                lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
            else:
                lm_logits = self.lm_head(sequence_output)

            if not return_dict:
                # 如果不要求返回字典，则返回 lm_logits 和解码器/编码器的其他输出
                return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

            # 返回包含各种输出的 FlaxSeq2SeqLMOutput 对象
            return FlaxSeq2SeqLMOutput(
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
class FlaxT5ForConditionalGeneration(FlaxT5PreTrainedModel):
    # 设置模型类为 FlaxT5ForConditionalGenerationModule
    module_class = FlaxT5ForConditionalGenerationModule

    # 解码方法，根据输入解码生成输出
    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=T5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # 注意通常会对注意力掩码中超出输入范围和小于缓存长度的位置放入 0，但因为解码器使用因果掩码，这些位置已经被掩码了。
        # 因此我们可以在这里创建一个静态的注意力掩码，更有效地进行编译
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
        }

    # 更新生成的输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs


FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
    >>> model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

    >>> ARTICLE_TO_SUMMARIZE = "summarize: My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")

    >>> # 生成摘要
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""


overwrite_call_docstring(
    # 导入 FlaxT5ForConditionalGeneration 类和 T5 输入文档字符串以及 Flax T5 有条件生成文档字符串
    FlaxT5ForConditionalGeneration, T5_INPUTS_DOCSTRING + FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING
# 调用函数append_replace_return_docstrings，并传入参数FlaxT5ForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
append_replace_return_docstrings(
    FlaxT5ForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
```