# `.\models\t5\modeling_flax_t5.py`

```py
# coding=utf-8
# Copyright 2021 T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Flax T5 model."""


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


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google-t5/t5-small"
_CONFIG_FOR_DOC = "T5Config"

remat = nn_partitioning.remat


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    # 初始化一个与input_ids相同形状的零张量
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将input_ids向右移动一位
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 在首位插入decoder_start_token_id
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    # 将所有-100的位置替换为pad_token_id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class FlaxT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    eps: float = 1e-6
    weight_init: Callable[..., np.ndarray] = jax.nn.initializers.ones

    def setup(self):
        # 创建权重参数
        self.weight = self.param("weight", self.weight_init, (self.hidden_size,))
    def __call__(self, hidden_states):
        """
        Construct a layernorm module in the T5 style; No bias and no subtraction of mean.
        """
        # layer norm should always be calculated in float32
        # 计算隐藏状态的方差，并在最后一个轴上求均值，保持维度不变
        variance = jnp.power(hidden_states.astype("f4"), 2).mean(axis=-1, keepdims=True)
        # 对隐藏状态进行标准化，除以标准差（方差的平方根），加上小的常量 self.eps 避免除以零
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        # 返回加权后的标准化隐藏状态
        return self.weight * hidden_states
# 定义一个名为 FlaxT5DenseActDense 的神经网络模块类
class FlaxT5DenseActDense(nn.Module):
    # 配置属性，指定为 T5Config 类型
    config: T5Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块设置方法，用于初始化模块的各个组件
    def setup(self):
        # 计算初始化权重标准差，根据配置的初始化因子和模型维度
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 创建第一个全连接层 wi，用于 d_ff 到 d_model 的映射
        self.wi = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 创建第二个全连接层 wo，用于 d_model 到 d_ff 的映射
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 创建一个 Dropout 层，用于在训练时随机置零部分输入单元，防止过拟合
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 根据配置选择激活函数，ACT2FN 是一个激活函数映射字典
        self.act = ACT2FN[self.config.dense_act_fn]

    # 定义模块的调用方法，用于执行前向传播
    def __call__(self, hidden_states, deterministic=True):
        # 将输入 hidden_states 经过第一个全连接层 wi
        hidden_states = self.wi(hidden_states)
        # 使用配置中指定的激活函数进行激活
        hidden_states = self.act(hidden_states)
        # 对激活后的结果进行 Dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 经过第二个全连接层 wo，得到最终输出
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 定义一个名为 FlaxT5DenseGatedActDense 的神经网络模块类，继承自 nn.Module
class FlaxT5DenseGatedActDense(nn.Module):
    # 配置属性，指定为 T5Config 类型
    config: T5Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 模块设置方法，用于初始化模块的各个组件
    def setup(self):
        # 计算初始化权重标准差，根据配置的初始化因子和模型维度
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 创建第一个全连接层 wi_0，用于 d_ff 到 d_model 的映射
        self.wi_0 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 创建第二个全连接层 wi_1，用于 d_ff 到 d_model 的映射
        self.wi_1 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 创建第三个全连接层 wo，用于 d_model 到 d_ff 的映射
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 创建一个 Dropout 层，用于在训练时随机置零部分输入单元，防止过拟合
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 根据配置选择激活函数，ACT2FN 是一个激活函数映射字典
        self.act = ACT2FN[self.config.dense_act_fn]

    # 定义模块的调用方法，用于执行前向传播
    def __call__(self, hidden_states, deterministic):
        # 经过第一个全连接层 wi_0，并使用配置中指定的激活函数进行激活
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 经过第二个全连接层 wi_1
        hidden_linear = self.wi_1(hidden_states)
        # gated activation function：将 gelu 激活后的结果与 linear 相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对结果进行 Dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 经过第三个全连接层 wo，得到最终输出
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 定义一个名为 FlaxT5LayerFF 的神经网络模块类，继承自 nn.Module
class FlaxT5LayerFF(nn.Module):
    # 配置属性，指定为 T5Config 类型
    config: T5Config
    # 数据类型，默认为 jnp.float32 用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 在对象初始化时设置网络结构
    def setup(self):
        # 如果配置要求使用带门控激活函数的 DenseReluDense 模块
        if self.config.is_gated_act:
            # 使用带门控激活函数的 DenseReluDense 初始化
            self.DenseReluDense = FlaxT5DenseGatedActDense(self.config, dtype=self.dtype)
        else:
            # 否则使用普通的 DenseActDense 初始化
            self.DenseReluDense = FlaxT5DenseActDense(self.config, dtype=self.dtype)

        # 初始化 LayerNorm 层，设置隐藏层维度和 epsilon 值
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 初始化 Dropout 层，设置丢弃率
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 对象调用方法，对隐藏状态进行处理
    def __call__(self, hidden_states, deterministic=True):
        # 对隐藏状态进行 LayerNorm 处理
        forwarded_states = self.layer_norm(hidden_states)
        # 通过 DenseReluDense 模块进行前向传播处理
        forwarded_states = self.DenseReluDense(forwarded_states, deterministic=deterministic)
        # 使用 Dropout 处理后的前向传播结果，与原始隐藏状态相加
        hidden_states = hidden_states + self.dropout(forwarded_states, deterministic=deterministic)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为 FlaxT5Attention 的神经网络模块，继承自 nn.Module
class FlaxT5Attention(nn.Module):
    # 类属性：配置信息，来自于 T5Config 类
    config: T5Config
    # 是否包含相对注意力偏置的标志，默认为 False
    has_relative_attention_bias: bool = False
    # 是否是因果注意力（causal attention）的标志，默认为 False
    causal: bool = False
    # 计算过程中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 相对注意力的桶数，从配置中获取
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        # 相对注意力的最大距离，从配置中获取
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        # 模型维度，从配置中获取
        self.d_model = self.config.d_model
        # 键值投影的维度，从配置中获取
        self.key_value_proj_dim = self.config.d_kv
        # 注意力头的数量，从配置中获取
        self.n_heads = self.config.num_heads
        # 丢弃率，从配置中获取
        self.dropout = self.config.dropout_rate
        # 内部维度，等于注意力头数量乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 初始化权重的标准差
        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 创建查询（q）、键（k）、值（v）和输出（o）的全连接层
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),  # 正态分布初始化权重
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),  # 正态分布初始化权重
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),  # 正态分布初始化权重
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),  # 正态分布初始化权重
            dtype=self.dtype,
        )

        # 如果有相对注意力偏置，创建相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),  # 正态分布初始化嵌入层权重
                dtype=self.dtype,
            )

    # 静态方法
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0
        # 如果允许双向注意力，则将桶的数量减半，并根据相对位置的正负决定桶的偏移
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            # 如果不允许双向注意力，则将相对位置限制在非正数范围内
            relative_position = -jnp.clip(relative_position, a_max=0)
        # 现在，relative_position 的范围是 [0, inf)

        # 将一半的桶用于精确增量位置
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半的桶用于对数增量位置，直到 max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        # 创建上下文位置矩阵和记忆位置矩阵
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        # 计算相对位置并将其转换为桶索引
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),  # 根据是否因果关系来决定是否双向
            num_buckets=self.relative_attention_num_buckets,  # 桶的数量
            max_distance=self.relative_attention_max_distance,  # 最大距离
        )

        # 计算相对注意力偏置值
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]  # 转置并扩展维度以匹配模型输出格式
        return values

    def _split_heads(self, hidden_states):
        # 将隐藏状态重塑为多头注意力的形状
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))
    # 将输入的隐藏状态重塑为指定维度的形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    # 使用 nn.compact 装饰器定义的方法，用于将单个输入令牌的投影键、值状态与先前步骤中的缓存状态连接起来
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据进行初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 初始化缓存的键和值，如果未初始化则为零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 缓存索引，用于跟踪缓存的位置
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键和值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存中的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引以反映已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置应仅参与已生成和缓存的键位置的自注意力，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回连接后的键、值以及更新后的注意力掩码
        return key, value, attention_mask

    # 创建位置偏置的方法，接受关键状态、查询状态、注意力掩码、初始化缓存、序列长度和因果注意力掩码偏移量作为参数
    def _create_position_bias(
        self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
        ):
            # 检查缓存是否已填充，条件包括因果关系和存在特定缓存键，且不是初始化缓存
            cache_is_filled = self.causal and self.has_variable("cache", "cached_key") and (not init_cache)
            # 计算键的长度
            key_length = key_states.shape[1]
            # 如果缓存已填充，则查询长度等于键的长度，否则等于查询状态的长度
            query_length = key_length if cache_is_filled else query_states.shape[1]

            # 如果模型支持相对注意力偏置，则计算位置偏置
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(query_length, key_length)
            # 否则，如果存在注意力掩码，则创建与其相同形状的零张量
            elif attention_mask is not None:
                position_bias = jnp.zeros_like(attention_mask)
            # 否则，默认创建形状为 (1, self.n_heads, query_length, key_length) 的零张量
            else:
                position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

            # 如果缓存已填充，则仅需取最后一个查询位置的偏置
            if cache_is_filled:
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                position_bias = jax.lax.dynamic_slice(
                    position_bias,
                    (0, 0, causal_attention_mask_shift, 0),
                    (1, self.n_heads, seq_length, max_decoder_length),
                )
            # 返回位置偏置张量
            return position_bias

        # 在调用实例时，接收隐藏状态、注意力掩码、键值状态、位置偏置等多个参数
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
# 定义一个 FlaxT5Block 类，继承自 nn.Module
class FlaxT5Block(nn.Module):
    # T5 模型的配置参数
    config: T5Config
    # 是否具有相对注意力偏置，默认为 False
    has_relative_attention_bias: bool = False
    # 计算中使用的数据类型，默认为 jnp.float32

    # 初始化方法，设置模块的组件
    def setup(self):
        # 创建自注意力层对象 SelfAttention，使用 FlaxT5Attention 类
        self.SelfAttention = FlaxT5Attention(
            self.config,
            has_relative_attention_bias=self.has_relative_attention_bias,
            causal=self.config.causal,
            dtype=self.dtype,
        )
        # 创建层归一化对象 layer_norm，使用 FlaxT5LayerNorm 类
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建 Dropout 层对象 dropout，使用 nn.Dropout 类，设置丢弃率为 config.dropout_rate
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 调用方法，定义模块的前向传播逻辑
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        # 对输入的隐藏状态进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用 SelfAttention 层处理归一化后的隐藏状态
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 将原始隐藏状态与经 Dropout 处理后的注意力输出相加，实现残差连接
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 构建输出元组，包括更新后的隐藏状态和可能的注意力输出
        outputs = (hidden_states,) + attention_output[1:]  # 如果有输出注意力信息，将其添加到输出中
        return outputs
    # 初始化方法，设置模型的相关属性和层次结构
    def setup(self):
        # 获取配置中的causal参数
        self.causal = self.config.causal
        # 初始化self.layer为一个元组，包含FlaxT5LayerSelfAttention层对象
        self.layer = (
            FlaxT5LayerSelfAttention(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                name=str(0),
                dtype=self.dtype,
            ),
        )
        # 初始化feed_forward_index为1
        feed_forward_index = 1
        # 如果causal为True，则添加FlaxT5LayerCrossAttention层对象到self.layer中
        if self.causal:
            self.layer += (FlaxT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            feed_forward_index += 1

        # 添加FlaxT5LayerFF层对象到self.layer中，名称为feed_forward_index
        self.layer += (FlaxT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    # 模型调用方法，执行自注意力和交叉注意力计算，并返回相应的输出
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
        # 执行自注意力层计算
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 更新hidden_states为自注意力输出的第一个元素
        hidden_states = self_attention_outputs[0]
        # 保留自注意力输出和相对位置权重
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # 如果需要执行交叉注意力计算
        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            # 执行交叉注意力层计算
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            # 更新hidden_states为交叉注意力输出的第一个元素
            hidden_states = cross_attention_outputs[0]

            # 将交叉注意力输出和相对位置权重添加到attention_outputs中
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # 应用Feed Forward层计算
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

        # 初始化输出为包含hidden_states的元组
        outputs = (hidden_states,)

        # 将attention_outputs添加到输出元组中
        outputs = outputs + attention_outputs

        # 返回包含hidden-states、present_key_value_states、(self-attention position bias)、
        # (self-attention weights)、(cross-attention position bias)、(cross-attention weights)的元组
        return outputs
class FlaxT5LayerCollection(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化单个 T5 层
        self.layer = FlaxT5Block(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )

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
        # 调用单个 T5 层进行计算
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


class FlaxT5BlockCollection(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        # 根据配置初始化 T5 块集合
        self.causal = self.config.causal
        if self.gradient_checkpointing:
            # 如果启用梯度检查点，则使用 remat 函数包装 FlaxT5LayerCollection
            FlaxT5CheckpointLayer = remat(FlaxT5LayerCollection, static_argnums=(6, 7, 8))
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
            # 否则，创建普通的 FlaxT5LayerCollection 实例列表
            self.blocks = [
                FlaxT5LayerCollection(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]

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
        # 调用 T5 块集合进行前向传播
        # 这里假设 self.blocks 包含了多个 T5 块，每个块处理一部分输入数据
        # 返回的结果取决于具体的 T5 模型结构和参数设置
        pass  # 此处的 pass 语句表示函数没有具体的返回内容，实际使用时需根据具体需求实现
    ):
        # 如果需要输出隐藏状态，初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，初始化一个空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力权重且模型是因果的，初始化一个空元组
        all_cross_attentions = () if (output_attentions and self.causal) else None
        # 初始化位置偏置为 None
        position_bias = None
        # 初始化编码器-解码器位置偏置为 None
        encoder_decoder_position_bias = None

        # 遍历每个 Transformer 模块层
        for i, layer_module in enumerate(self.blocks):
            # 如果需要输出隐藏状态，将当前隐藏状态添加到 all_hidden_states 元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用 Transformer 层进行前向传播
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

            # 更新隐藏状态为当前层的输出隐藏状态
            hidden_states = layer_outputs[0]

            # 更新位置偏置为当前层的自注意力位置偏置
            position_bias = layer_outputs[1]

            # 如果模型是因果的且存在编码器隐藏状态，更新编码器-解码器位置偏置
            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            # 如果需要输出注意力权重，将当前层的注意力权重添加到 all_attentions 元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                # 如果模型是因果的，将当前层的交叉注意力权重添加到 all_cross_attentions 元组中
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        # 返回 Transformer 模型的输出，包括最终的隐藏状态、所有层的隐藏状态、所有层的注意力权重和交叉注意力权重
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
class FlaxT5Stack(nn.Module):
    config: T5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # 计算中使用的数据类型，默认为 jnp.float32
    gradient_checkpointing: bool = False  # 是否启用梯度检查点

    def setup(self):
        self.causal = self.config.causal  # 是否是因果关系模型

        # 初始化 T5 模型的块集合
        self.block = FlaxT5BlockCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 最终的层归一化
        self.final_layer_norm = FlaxT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)  # 用于随机失活的 Dropout 层

    def __call__(
        self,
        input_ids=None,  # 输入的 token IDs
        attention_mask=None,  # 注意力掩码
        encoder_hidden_states=None,  # 编码器隐藏状态
        encoder_attention_mask=None,  # 编码器注意力掩码
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态
        return_dict: bool = True,  # 是否返回字典格式结果
        deterministic: bool = True,  # 是否确定性计算
        init_cache: bool = False,  # 是否初始化缓存
    ):
        hidden_states = self.embed_tokens(input_ids)  # 嵌入 token IDs 得到隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 使用 Dropout 层

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )  # 使用 T5 模型块进行前向传播计算

        hidden_states = outputs[0]  # 取得输出中的隐藏状态

        hidden_states = self.final_layer_norm(hidden_states)  # 最终的层归一化
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 再次应用 Dropout

        # 添加最后一层
        all_hidden_states = None

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]  # 返回不同类型的输出
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )  # 返回带有注意力和交叉注意力的最终模型输出


T5_ENCODE_INPUTS_DOCSTRING = r"""
    # 接收输入参数：
    #   - input_ids (`jnp.ndarray`，形状为 `(batch_size, sequence_length)`)：
    #     表示输入序列标记在词汇表中的索引。T5 是一个带有相对位置嵌入的模型，因此可以在左右两侧对输入进行填充。
    #     可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
    #     若要了解有关预训练中如何准备 `input_ids` 的更多信息，请查看 [T5 Training](./t5#training)。
    #   - attention_mask (`jnp.ndarray`，形状为 `(batch_size, sequence_length)`)，*可选*：
    #     遮盖掩码，用于避免在填充标记索引上执行注意力操作。遮盖值在 `[0, 1]` 之间：
    #     - 1 表示 **未遮盖** 的标记，
    #     - 0 表示 **遮盖** 的标记。
    #     [什么是注意力遮盖？](../glossary#attention-mask)
    #   - output_attentions (`bool`，*可选*)：
    #     是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。
    #   - output_hidden_states (`bool`，*可选*)：
    #     是否返回所有层的隐藏状态。查看返回的张量中的 `hidden_states` 以获取更多细节。
    #   - return_dict (`bool`，*可选*)：
    #     是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
定义一个文档字符串常量，用于描述T5模型输入的参数说明文档。

Args:
    decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
        解码器输入序列标记在词汇表中的索引。

        可以使用[`AutoTokenizer`]获取。详见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。

        [decoder_input_ids是什么？](../glossary#decoder-input-ids)

        在训练时，应提供`decoder_input_ids`。
    encoder_outputs (`tuple(tuple(jnp.ndarray)`):
        元组包含(`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)
        `last_hidden_state`的形状为`(batch_size, sequence_length, hidden_size)`，*可选*是编码器最后一层的隐藏状态序列。
        用于解码器的交叉注意力。
    encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *可选*):
        遮罩，避免对填充标记索引执行注意力计算。遮罩中的值选择在`[0, 1]`：

        - 1表示**未遮罩**的标记，
        - 0表示**已遮罩**的标记。

        [注意力遮罩是什么？](../glossary#attention-mask)
    decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *可选*):
        默认行为：生成一个张量，忽略`decoder_input_ids`中的填充标记。默认情况下也将使用因果遮罩。

        如果要更改填充行为，应根据需要进行修改。有关默认策略的更多信息，请参见[论文中的图1](https://arxiv.org/abs/1910.13461)。
    past_key_values (`Dict[str, np.ndarray]`, *可选*, 由`init_cache`返回或传递先前的`past_key_values`):
        预先计算的隐藏状态字典（注意力块中的键和值）。可用于快速自回归解码。预计算的键和值隐藏状态的形状为*[batch_size, max_length]*。
    output_attentions (`bool`, *可选*):
        是否返回所有注意力层的注意力张量。有关返回张量中`attentions`的更多细节，请参见文档。
    output_hidden_states (`bool`, *可选*):
        是否返回所有层的隐藏状态。有关返回张量中`hidden_states`的更多细节，请参见文档。
    return_dict (`bool`, *可选*):
        是否返回[`~utils.ModelOutput`]而不是普通元组。
"""
    # 初始化方法，用于创建一个新的模型实例
    def __init__(
        self,
        config: T5Config,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 使用给定的配置和参数实例化模块对象
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 调用父类的初始化方法，传入配置、模块对象以及其他参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点功能的方法
    def enable_gradient_checkpointing(self):
        # 更新模块对象，设置梯度检查点为 True
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 初始化模型权重的方法
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量 input_ids，全零张量
        input_ids = jnp.zeros(input_shape, dtype="i4")

        # 创建与 input_ids 形状相同的全一张量 attention_mask
        attention_mask = jnp.ones_like(input_ids)
        args = [input_ids, attention_mask]

        # 如果模块类不是 FlaxT5EncoderModule，则初始化解码器相关输入张量
        if self.module_class not in [FlaxT5EncoderModule]:
            decoder_input_ids = jnp.ones_like(input_ids)
            decoder_attention_mask = jnp.ones_like(input_ids)
            args.extend([decoder_input_ids, decoder_attention_mask])

        # 切分随机数生成器 rng，用于参数和 dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用随机数生成器和输入张量初始化模块参数，返回随机初始化后的参数
        random_params = self.module.init(
            rngs,
            *args,
        )["params"]

        # 如果传入了已有的参数 params，则将缺失的参数填充为随机初始化的参数值
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 返回填充后的参数冻结字典
            return freeze(unflatten_dict(params))
        else:
            # 否则直接返回随机初始化的参数
            return random_params

    # 覆盖父类的 __call__ 方法，定义模型的前向传播
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
        ):
            # 如果 `output_attentions` 参数未指定，则使用配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 `output_hidden_states` 参数未指定，则使用配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果 `return_dict` 参数未指定，则使用配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 如果缺少 `decoder_input_ids` 参数，则抛出数值错误异常
            if decoder_input_ids is None:
                raise ValueError(
                    "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                    " here."
                )

            # 准备编码器输入的注意力掩码
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)

            # 准备解码器输入的注意力掩码
            if decoder_attention_mask is None:
                decoder_attention_mask = jnp.ones_like(decoder_input_ids)

            # 处理可能存在的伪随机数生成器
            rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

            # 调用模型的应用方法，传递参数和输入数据
            return self.module.apply(
                {"params": params or self.params},
                input_ids=jnp.array(input_ids, dtype="i4"),
                attention_mask=jnp.array(attention_mask, dtype="i4"),
                decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
                decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=not train,
                rngs=rngs,
            )
    # 初始化缓存函数，用于自动回归解码的快速初始化
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自动回归解码的批大小。定义了初始化缓存时的批大小。
            max_length (`int`):
                自动回归解码的最大可能长度。定义了初始化缓存时的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，
                *可选*: 是编码器最后一层的隐藏状态序列。在解码器的交叉注意力中使用。

        """
        # 初始化用于检索缓存的输入变量
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            # 获取解码器模块
            decoder_module = module._get_decoder_module()
            # 调用解码器模块进行前向传播
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        # 使用指定方法进行初始化，仅需调用解码器以初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,
        )
        # 返回解冻后的缓存变量
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
        ):
        r"""
        Returns:

        Example:

        ```
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        # 根据参数设置或者默认配置决定是否输出注意力机制
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数设置或者默认配置决定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数设置或者默认配置决定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供注意力掩码，则创建一个全为1的掩码，与输入张量维度相同
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果有需要，处理任何的伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 定义编码器前向函数
        def _encoder_forward(module, input_ids, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, **kwargs)

        # 调用模型的应用方法，传入参数并执行编码器前向计算
        return self.module.apply(
            {"params": params or self.params},  # 使用给定的参数或者默认参数
            input_ids=jnp.array(input_ids, dtype="i4"),  # 将输入张量转换为JAX数组
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 将注意力掩码转换为JAX数组
            output_attentions=output_attentions,  # 是否输出注意力机制
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否以字典格式返回输出
            deterministic=not train,  # 是否确定性计算，即非训练模式
            rngs=rngs,  # 传入的伪随机数生成器
            method=_encoder_forward,  # 调用的方法，即编码器的前向计算函数
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
"""
The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder-decoder transformer pre-trained in a
text-to-text denoising generative setting.

This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a Flax Linen
[flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matters related to general usage and behavior.

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

@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class FlaxT5Module(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        """
        Retrieve the encoder module of the T5 model.
        """
        return self.encoder

    def _get_decoder_module(self):
        """
        Retrieve the decoder module of the T5 model.
        """
        return self.decoder
    # 初始化模型参数，包括共享的嵌入层和编码器、解码器的配置
    def setup(self):
        # 初始化共享的嵌入层，使用给定的词汇大小和模型维度，使用正态分布进行初始化
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        # 复制编码器配置，并设置非因果性（causal=False）
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        # 初始化编码器模型，使用共享的嵌入层和复制后的配置
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # 复制解码器配置，并设置因果性（causal=True），以及解码器层数
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        # 初始化解码器模型，使用共享的嵌入层和复制后的配置
        self.decoder = FlaxT5Stack(
            decoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 模型调用函数，用于执行编码和解码操作
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
        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 编码阶段（训练和第一次预测通道）
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 解码阶段
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],  # 使用编码器的隐藏状态作为输入
            encoder_attention_mask=attention_mask,  # 使用编码器的注意力掩码
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果不返回字典格式的输出，则将编码器和解码器的输出合并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回字典格式的输出，包括编码器和解码器的各种隐藏状态和注意力分布
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
# 使用 FlaxT5PreTrainedModel 作为基类定义 FlaxT5Model 类
class FlaxT5Model(FlaxT5PreTrainedModel):
    # 将 module_class 属性设置为 FlaxT5Module
    module_class = FlaxT5Module

# 调用 append_call_sample_docstring 函数，为 FlaxT5Model 类添加示例和文档字符串
append_call_sample_docstring(FlaxT5Model, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义 FLAX_T5_MODEL_DOCSTRING 变量，包含返回值和示例的文档字符串
FLAX_T5_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```
    >>> from transformers import AutoTokenizer, FlaxT5Model

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    >>> model = FlaxT5Model.from_pretrained("google-t5/t5-small")

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

# 调用 overwrite_call_docstring 函数，为 FlaxT5Model 类替换调用时的文档字符串
overwrite_call_docstring(FlaxT5Model, T5_INPUTS_DOCSTRING + FLAX_T5_MODEL_DOCSTRING)

# 调用 append_replace_return_docstrings 函数，为 FlaxT5Model 类添加返回值文档字符串
append_replace_return_docstrings(FlaxT5Model, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

# 使用 add_start_docstrings 函数为 FlaxT5EncoderModule 类添加类注释和初始文档字符串
@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
# 定义 FlaxT5EncoderModule 类，继承自 nn.Module
class FlaxT5EncoderModule(nn.Module):
    # 包含 T5Config 类型的 config 属性和 jnp.float32 类型的 dtype 属性
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    # 定义 setup 方法，初始化模型的共享参数和编码器
    def setup(self):
        # 创建共享的嵌入层，使用正态分布初始化
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        # 复制配置以用于编码器，并设置相关属性
        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.causal = False
        # 创建 FlaxT5Stack 编码器
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 定义 __call__ 方法，处理模型的正向传播
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 如果需要编码（训练或第一次预测），调用编码器进行处理
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 返回编码器的输出
        return encoder_outputs
    # 将模块类设置为FlaxT5EncoderModule
    module_class = FlaxT5EncoderModule

    # 使用装饰器为__call__方法添加文档字符串，文档字符串来源于T5_ENCODE_INPUTS_DOCSTRING
    @add_start_docstrings_to_model_forward(T5_ENCODE_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,  # 输入的token IDs，作为JAX数组
        attention_mask: Optional[jnp.ndarray] = None,  # 可选的注意力遮罩，如果为None则设为全1数组
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，如果为None则使用self.config.output_attentions
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，如果为None则使用self.config.output_hidden_states
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，如果为None则使用self.config.return_dict
        train: bool = False,  # 是否为训练模式
        params: dict = None,  # 参数字典，默认为None
        dropout_rng: PRNGKey = None,  # dropout的随机数生成器，如果为None则表示不使用dropout
    ):
        # 确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备编码器的输入
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)  # 如果注意力遮罩为None，则全设为1的数组

        # 处理可能存在的任何PRNG（伪随机数生成器）
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用self.module的apply方法，对输入参数进行编码
        return self.module.apply(
            {"params": params or self.params},  # 模型参数字典
            input_ids=jnp.array(input_ids, dtype="i4"),  # token IDs转换为JAX数组，数据类型为32位整数
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 注意力遮罩转换为JAX数组，数据类型为32位整数
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            deterministic=not train,  # 是否确定性运行，即非训练模式
            rngs=rngs,  # PRNG（伪随机数生成器）字典
        )
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class FlaxT5ForConditionalGenerationModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        # 返回编码器模块
        return self.encoder

    def _get_decoder_module(self):
        # 返回解码器模块
        return self.decoder

    def setup(self):
        self.model_dim = self.config.d_model

        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化编码器模型
        self.encoder = FlaxT5Stack(
            encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        # 初始化解码器模型
        self.decoder = FlaxT5Stack(
            decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 初始化语言模型头部
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

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
        # T5 条件生成模型的调用函数

        # 参数说明：
        # input_ids: 输入序列的 token IDs
        # attention_mask: 注意力遮罩，指示哪些位置是 padding 的
        # decoder_input_ids: 解码器的输入 token IDs
        # decoder_attention_mask: 解码器的注意力遮罩
        # encoder_outputs: 编码器的输出
        # output_attentions: 是否输出注意力权重
        # output_hidden_states: 是否输出隐藏状态
        # return_dict: 是否返回字典格式的输出
        # deterministic: 是否使用确定性推断

        # 函数主体根据输入参数执行条件生成任务，输出生成的结果
        ):
            # 如果 return_dict 参数为 None，则根据配置决定是否使用默认值 self.config.use_return_dict
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 编码阶段
        encoder_outputs = self.encoder(
            input_ids=input_ids,                      # 输入的 token IDs
            attention_mask=attention_mask,            # 注意力掩码，指示哪些位置是有效的
            output_attentions=output_attentions,      # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,                  # 是否返回字典格式的输出
            deterministic=deterministic,              # 是否确定性计算
        )

        hidden_states = encoder_outputs[0]             # 获取编码器的隐藏状态

        # 解码阶段
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,               # 解码器的输入 token IDs
            attention_mask=decoder_attention_mask,     # 解码器的注意力掩码
            encoder_hidden_states=hidden_states,       # 编码器的隐藏状态，作为解码器的输入
            encoder_attention_mask=attention_mask,     # 编码器的注意力掩码，用于解码器的注意力机制
            output_attentions=output_attentions,      # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,                  # 是否返回字典格式的输出
            deterministic=deterministic,              # 是否确定性计算
        )

        sequence_output = decoder_outputs[0]           # 获取解码器的输出序列

        if self.config.tie_word_embeddings:
            # 如果配置中指定共享词嵌入，则在投影到词汇表之前进行输出的重新缩放
            # 参考：https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.config.tie_word_embeddings:
            # 如果配置中指定共享词嵌入，则从共享的变量中获取嵌入层参数，并应用于 lm_head
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)  # 否则直接将输出序列传递给 lm_head

        if not return_dict:
            # 如果不需要返回字典格式的输出，则返回一组元组
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        # 返回 FlaxSeq2SeqLMOutput 类型的对象，包含详细的输出信息
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
    module_class = FlaxT5ForConditionalGenerationModule

    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=T5Config)
    # 定义解码方法，用于生成模型输出
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
    ):
        # 准备用于生成的输入数据，返回生成所需的上下文和注意力掩码
        def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            max_length,
            attention_mask: Optional[jax.Array] = None,
            decoder_attention_mask: Optional[jax.Array] = None,
            encoder_outputs=None,
            **kwargs,
        ):
            # 初始化缓存，准备生成所需的过去键值
            batch_size, seq_length = decoder_input_ids.shape
            past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
            
            # 创建扩展的注意力掩码，用于遮蔽输入之外的位置
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

        # 更新用于生成的输入，将模型输出的过去键值更新到输入参数中
        def update_inputs_for_generation(self, model_outputs, model_kwargs):
            model_kwargs["past_key_values"] = model_outputs.past_key_values
            return model_kwargs


FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING = """
    返回：
        生成模型的输出结果。
        
    示例：
    
    ```
    >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    >>> model = FlaxT5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    >>> ARTICLE_TO_SUMMARIZE = "summarize: My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")

    >>> # 生成摘要
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""


overwrite_call_docstring(
    # 导入FlaxT5ForConditionalGeneration类，并合并文档字符串常量T5_INPUTS_DOCSTRING和FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING
    FlaxT5ForConditionalGeneration, T5_INPUTS_DOCSTRING + FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING
# 将文档字符串追加到指定类的方法中，并替换已有的文档字符串（如果存在），然后返回文档字符串修饰后的方法。
append_replace_return_docstrings(
    FlaxT5ForConditionalGeneration,  # 将文档字符串添加到 FlaxT5ForConditionalGeneration 类中的方法
    output_type=FlaxSeq2SeqLMOutput,  # 指定输出类型为 FlaxSeq2SeqLMOutput
    config_class=_CONFIG_FOR_DOC      # 使用 _CONFIG_FOR_DOC 配置类
)
```