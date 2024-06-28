# `.\models\t5\modeling_tf_t5.py`

```
# coding=utf-8
# Copyright 2020 T5 Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" TF 2.0 T5 model."""

from __future__ import annotations  # Ensures compatibility with type annotations in older Python versions

import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice

# Imports specific to the T5 model architecture from Hugging Face libraries
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_t5 import T5Config

# Initialize logger for logging messages from this module
logger = logging.get_logger(__name__)

# List of pre-trained model names available in TF T5 from Hugging Face model hub
_CONFIG_FOR_DOC = "T5Config"

TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google-t5/t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]

####################################################
# TF 2.0 Models are constructed using Keras imperative API by sub-classing
# - keras.layers.Layer for the layers and
# - TFPreTrainedModel for the models (it-self a sub-class of keras.Model)
####################################################


class TFT5LayerNorm(keras.layers.Layer):
    def __init__(self, hidden_size, epsilon=1e-6, **kwargs):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__(**kwargs)
        self.variance_epsilon = epsilon
        self.hidden_size = hidden_size

    def build(self, input_shape):
        """Build shared word embedding layer"""
        # Initialize weight parameter for layer normalization
        self.weight = self.add_weight("weight", shape=(self.hidden_size,), initializer="ones")
        super().build(input_shape)
    # 定义一个方法call，接受hidden_states作为参数
    def call(self, hidden_states):
        # 计算hidden_states张量在最后一个轴上的平均平方值，得到方差
        variance = tf.math.reduce_mean(tf.math.square(hidden_states), axis=-1, keepdims=True)
        # 对hidden_states进行归一化处理，使用方差的倒数加上一个小的常数self.variance_epsilon
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)
        # 返回归一化后的hidden_states乘以权重self.weight的结果
        return self.weight * hidden_states
class TFT5DenseActDense(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化权重的随机正态分布，均值为0，标准差为 config.initializer_factor * (config.d_model**-0.5)
        wi_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model**-0.5)
        )
        # 初始化权重的随机正态分布，均值为0，标准差为 config.initializer_factor * (config.d_ff**-0.5)
        wo_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff**-0.5)
        )
        # 创建名为 wi 的 Dense 层，输出维度为 config.d_ff，不使用偏置，使用 wi_initializer 初始化
        self.wi = keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        # 创建名为 wo 的 Dense 层，输出维度为 config.d_model，不使用偏置，使用 wo_initializer 初始化
        self.wo = keras.layers.Dense(
            config.d_model, use_bias=False, name="wo", kernel_initializer=wo_initializer
        )  # Update init weights as in flax
        # 创建 Dropout 层，使用 config.dropout_rate 的丢弃率
        self.dropout = keras.layers.Dropout(config.dropout_rate)
        # 获取激活函数，根据 config.dense_act_fn 配置
        self.act = get_tf_activation(config.dense_act_fn)
        self.config = config

    def call(self, hidden_states, training=False):
        # 前向传播函数：应用 wi 层到隐藏状态，然后应用激活函数，再进行 dropout，最后应用 wo 层
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.wo(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建层的方法，在此处设置 wi 和 wo 层的输入形状
        if getattr(self, "wi", None) is not None:
            with tf.name_scope(self.wi.name):
                self.wi.build([None, None, self.config.d_model])
        if getattr(self, "wo", None) is not None:
            with tf.name_scope(self.wo.name):
                self.wo.build([None, None, self.config.d_ff])


class TFT5DenseGatedActDense(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化权重的随机正态分布，均值为0，标准差为 config.initializer_factor * (config.d_model**-0.5)
        wi_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model**-0.5)
        )
        # 初始化权重的随机正态分布，均值为0，标准差为 config.initializer_factor * (config.d_ff**-0.5)
        wo_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff**-0.5)
        )
        # 创建名为 wi_0 的 Dense 层，输出维度为 config.d_ff，不使用偏置，使用 wi_initializer 初始化
        self.wi_0 = keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi_0", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        # 创建名为 wi_1 的 Dense 层，输出维度为 config.d_ff，不使用偏置，使用 wi_initializer 初始化
        self.wi_1 = keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi_1", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        # 创建名为 wo 的 Dense 层，输出维度为 config.d_model，不使用偏置，使用 wo_initializer 初始化
        self.wo = keras.layers.Dense(
            config.d_model, use_bias=False, name="wo", kernel_initializer=wo_initializer
        )  # Update init weights as in flax
        # 创建 Dropout 层，使用 config.dropout_rate 的丢弃率
        self.dropout = keras.layers.Dropout(config.dropout_rate)
        # 获取激活函数，根据 config.dense_act_fn 配置
        self.act = get_tf_activation(config.dense_act_fn)
        self.config = config
    # 定义一个方法用于处理模型中的隐藏状态，可以选择是否处于训练模式
    def call(self, hidden_states, training=False):
        # 使用激活函数处理第一个线性层的输出
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 获取第二个线性层的输出
        hidden_linear = self.wi_1(hidden_states)
        # 将第一个线性层和第二个线性层的输出进行逐元素相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对隐藏状态应用dropout操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 应用输出层的线性变换
        hidden_states = self.wo(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

    # 构建方法，用于构建模型的各个层和变量
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在第一个线性层 wi_0，则构建该层并命名作用域
        if getattr(self, "wi_0", None) is not None:
            with tf.name_scope(self.wi_0.name):
                self.wi_0.build([None, None, self.config.d_model])
        # 如果存在第二个线性层 wi_1，则构建该层并命名作用域
        if getattr(self, "wi_1", None) is not None:
            with tf.name_scope(self.wi_1.name):
                self.wi_1.build([None, None, self.config.d_model])
        # 如果存在输出层 wo，则构建该层并命名作用域
        if getattr(self, "wo", None) is not None:
            with tf.name_scope(self.wo.name):
                self.wo.build([None, None, self.config.d_ff])
# 定义了一个自定义层 TFT5LayerFF，继承自 keras 的 Layer 类
class TFT5LayerFF(keras.layers.Layer):
    # 初始化方法，接收配置参数 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据配置是否启用门控激活函数，选择不同的 Dense 层组合
        if config.is_gated_act:
            self.DenseReluDense = TFT5DenseGatedActDense(config, name="DenseReluDense")
        else:
            self.DenseReluDense = TFT5DenseActDense(config, name="DenseReluDense")

        # LayerNormalization 层，用于标准化输入张量
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name="layer_norm")
        # Dropout 层，用于在训练时随机断开一定比例的输入单元，防止过拟合
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    # call 方法，定义了层的正向传播逻辑
    def call(self, hidden_states, training=False):
        # 对输入张量进行标准化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将标准化后的张量传入 DenseReluDense 层，得到输出
        dense_output = self.DenseReluDense(normed_hidden_states, training=training)
        # 将原始输入张量与 Dropout 后的输出相加，作为最终的隐藏状态输出
        hidden_states = hidden_states + self.dropout(dense_output, training=training)
        # 返回最终的隐藏状态输出
        return hidden_states

    # build 方法，用于构建层的参数，确保在第一次调用 call 方法之前被调用
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经定义了 layer_norm 属性，则构建 layer_norm
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)
        # 如果已经定义了 DenseReluDense 属性，则构建 DenseReluDense
        if getattr(self, "DenseReluDense", None) is not None:
            with tf.name_scope(self.DenseReluDense.name):
                self.DenseReluDense.build(None)


# 定义了一个自定义层 TFT5Attention，继承自 keras 的 Layer 类
class TFT5Attention(keras.layers.Layer):
    # 类变量 NEW_ID，用于生成唯一的层标识符
    NEW_ID = itertools.count()
    # 初始化方法，用于初始化一个TFT5Attention对象
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置当前层的唯一标识符，通过从TFT5Attention.NEW_ID中获取
        self.layer_id = next(TFT5Attention.NEW_ID)
        # 判断当前层是否为解码器（根据传入的config配置）
        self.is_decoder = config.is_decoder
        # 是否使用缓存（根据传入的config配置）
        self.use_cache = config.use_cache
        # 是否包含相对注意力偏置（默认为False，可根据参数设置）
        self.has_relative_attention_bias = has_relative_attention_bias
        # 是否输出注意力权重（根据传入的config配置）
        self.output_attentions = config.output_attentions

        # 相对注意力相关的配置参数
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 模型维度
        self.d_model = config.d_model
        # 键值投影维度
        self.key_value_proj_dim = config.d_kv
        # 注意力头的数量
        self.n_heads = config.num_heads
        # 内部维度，等于注意力头数乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用Mesh TensorFlow进行初始化，以避免在softmax之前进行缩放
        # 初始化查询矩阵的权重
        q_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        )
        # 初始化键矩阵的权重
        k_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        # 初始化值矩阵的权重
        v_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        # 初始化输出矩阵的权重
        o_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        # 初始化相对注意力偏置的权重
        self.relative_attention_bias_initializer = keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )

        # 定义查询矩阵的全连接层
        self.q = keras.layers.Dense(
            self.inner_dim, use_bias=False, name="q", kernel_initializer=q_initializer
        )  # Update init weights as in flax
        # 定义键矩阵的全连接层
        self.k = keras.layers.Dense(
            self.inner_dim, use_bias=False, name="k", kernel_initializer=k_initializer
        )  # Update init weights as in flax
        # 定义值矩阵的全连接层
        self.v = keras.layers.Dense(
            self.inner_dim, use_bias=False, name="v", kernel_initializer=v_initializer
        )  # Update init weights as in flax
        # 定义输出矩阵的全连接层
        self.o = keras.layers.Dense(
            self.d_model, use_bias=False, name="o", kernel_initializer=o_initializer
        )  # Update init weights as in flax
        # 定义dropout层，用于随机失活
        self.dropout = keras.layers.Dropout(config.dropout_rate)

        # 初始化被剪枝的注意力头集合
        self.pruned_heads = set()
    # 如果模型已经构建，则直接返回，不进行重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果具有相对注意力偏置，则添加相对注意力偏置权重
        if self.has_relative_attention_bias:
            with tf.name_scope("relative_attention_bias"):
                # 添加一个权重张量用于相对注意力偏置，形状为 [相对注意力桶数, 注意力头数]
                self.relative_attention_bias = self.add_weight(
                    name="embeddings",
                    shape=[self.relative_attention_num_buckets, self.n_heads],
                    initializer=self.relative_attention_bias_initializer,  # 使用给定的初始化器进行初始化
                )
        # 如果存在 q 属性，则构建 q 层，并指定形状为 [None, None, self.d_model]
        if getattr(self, "q", None) is not None:
            with tf.name_scope(self.q.name):
                self.q.build([None, None, self.d_model])
        # 如果存在 k 属性，则构建 k 层，并指定形状为 [None, None, self.d_model]
        if getattr(self, "k", None) is not None:
            with tf.name_scope(self.k.name):
                self.k.build([None, None, self.d_model])
        # 如果存在 v 属性，则构建 v 层，并指定形状为 [None, None, self.d_model]
        if getattr(self, "v", None) is not None:
            with tf.name_scope(self.v.name):
                self.v.build([None, None, self.d_model])
        # 如果存在 o 属性，则构建 o 层，并指定形状为 [None, None, self.inner_dim]
        if getattr(self, "o", None) is not None:
            with tf.name_scope(self.o.name):
                self.o.build([None, None, self.inner_dim])

    # 抛出未实现错误，表明 prune_heads 方法尚未实现
    def prune_heads(self, heads):
        raise NotImplementedError

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

        Args:
            relative_position: an int32 Tensor - the relative positions between memory and query
            bidirectional: a boolean - whether the attention is bidirectional or not
            num_buckets: an integer - number of buckets to map relative positions into
            max_distance: an integer - maximum distance to consider for bucketing

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # Initialize relative_buckets to 0
        relative_buckets = 0
        
        # Adjust num_buckets if bidirectional is True
        if bidirectional:
            num_buckets //= 2
            # Add num_buckets to relative_buckets if relative_position > 0
            relative_buckets += (
                tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            # Take absolute value of relative_position
            relative_position = tf.math.abs(relative_position)
        else:
            # Set relative_position to negative minimum value if it is <= 0
            relative_position = -tf.math.minimum(relative_position, 0)
        
        # Calculate max_exact as half of num_buckets
        max_exact = num_buckets // 2
        
        # Check if relative_position is less than max_exact
        is_small = tf.math.less(relative_position, max_exact)
        
        # Calculate relative_position_if_large using logarithmic scaling
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32))
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        
        # Clamp relative_position_if_large to num_buckets - 1
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        
        # Add relative_position or relative_position_if_large to relative_buckets based on is_small condition
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        
        # Return the computed relative_buckets
        return relative_buckets
    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        # 生成一个形状为 (query_length, 1) 的张量，表示查询位置
        context_position = tf.range(query_length)[:, None]
        # 生成一个形状为 (1, key_length) 的张量，表示记忆位置
        memory_position = tf.range(key_length)[None, :]
        # 计算相对位置矩阵，形状为 (query_length, key_length)，每个元素表示相对位置的差值
        relative_position = memory_position - context_position

        # 将相对位置矩阵映射到预定义数量的桶中，以便后续注意力机制使用
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # 根据桶的索引，从预定义的相对注意力偏置中收集对应的值
        values = tf.gather(
            self.relative_attention_bias, relative_position_bucket
        )  # 形状为 (query_length, key_length, num_heads)

        # 调整维度顺序，以符合注意力机制期望的输入形状
        values = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]), axis=0
        )  # 形状为 (1, num_heads, query_length, key_length)

        # 返回调整后的相对注意力偏置值
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        training=False,
        output_attentions=False,
# 定义了一个名为 TFT5LayerSelfAttention 的自定义层，继承自 keras.layers.Layer
class TFT5LayerSelfAttention(keras.layers.Layer):
    # 初始化方法，接受 config 和其他参数
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        # 创建 TFT5Attention 实例 SelfAttention，用于自注意力机制
        self.SelfAttention = TFT5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="SelfAttention",
        )
        # 创建 layer_norm 层，用于层归一化
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name="layer_norm")
        # 创建 dropout 层，用于随机失活
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    # 定义 call 方法，处理层的正向传播
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        # 对输入的 hidden_states 进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用 SelfAttention 层进行自注意力计算
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 将原始 hidden_states 和经过 dropout 后的 attention_output 相加
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        # 如果需要输出 attentions，则将它们添加到 outputs 中
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

    # 定义 build 方法，用于构建层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建 SelfAttention 层
        if getattr(self, "SelfAttention", None) is not None:
            with tf.name_scope(self.SelfAttention.name):
                self.SelfAttention.build(None)
        # 构建 layer_norm 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)


# 定义了一个名为 TFT5LayerCrossAttention 的自定义层，继承自 keras.layers.Layer
class TFT5LayerCrossAttention(keras.layers.Layer):
    # 初始化方法，接受 config 和其他参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建 TFT5Attention 实例 EncDecAttention，用于编码-解码注意力机制
        self.EncDecAttention = TFT5Attention(
            config,
            has_relative_attention_bias=False,
            name="EncDecAttention",
        )
        # 创建 layer_norm 层，用于层归一化
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name="layer_norm")
        # 创建 dropout 层，用于随机失活
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    # 定义 call 方法，处理层的正向传播
    def call(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        # 对输入的 hidden_states 进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用 EncDecAttention 层进行编码-解码注意力计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            key_value_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 将原始 hidden_states 和经过 dropout 后的 attention_output 相加
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        # 如果需要输出 attentions，则将它们添加到 outputs 中
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs
    ):
        # 对隐藏状态进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用编码-解码注意力机制进行计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 将注意力输出与隐藏状态相加，并使用 dropout 进行处理
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        # 构建输出元组，包括隐藏状态和可能的注意力输出
        outputs = (hidden_states,) + attention_output[1:]  # 如果有输出的话，添加注意力
        # 返回模块的输出
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在编码-解码注意力模块，构建它
        if getattr(self, "EncDecAttention", None) is not None:
            with tf.name_scope(self.EncDecAttention.name):
                self.EncDecAttention.build(None)
        # 如果存在层归一化模块，构建它
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)
# 定义自定义层类 TFT5Block，继承自 keras 的 Layer 类
class TFT5Block(keras.layers.Layer):
    # 初始化方法，接受配置 config 和是否具有相对注意力偏置的参数
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        # 标记是否为解码器
        self.is_decoder = config.is_decoder
        # 初始化层列表
        self.layer = []
        # 添加自注意力层 TFT5LayerSelfAttention 到层列表
        self.layer.append(
            TFT5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                name="layer_._0",  # 设置层的名称
            )
        )
        # 如果是解码器，添加交叉注意力层 TFT5LayerCrossAttention 到层列表
        if self.is_decoder:
            self.layer.append(
                TFT5LayerCrossAttention(
                    config,
                    name="layer_._1",  # 设置层的名称
                )
            )
        # 添加前馈神经网络层 TFT5LayerFF 到层列表
        self.layer.append(TFT5LayerFF(config, name=f"layer_._{len(self.layer)}"))

    # 定义调用方法，传入各种参数，进行层的调用
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        # 省略具体实现细节，用于层的前向传播计算

    # 构建方法，用于构建层，并将其加入到模型中
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 遍历层列表，为每一层设置名称作用域并构建它们
        for layer_module in self.layer:
            if hasattr(layer_module, "name"):
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


####################################################
# TFT5MainLayer 是一个 keras 的自定义层，用于表示完整的 T5 模型主体
# 通常称为 "TFT5MainLayer"
####################################################
@keras_serializable
class TFT5MainLayer(keras.layers.Layer):
    # 配置类为 T5Config
    config_class = T5Config

    # 初始化方法，接受配置 config 和嵌入标记 embed_tokens 的参数
    def __init__(self, config, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)

        # 设置模型的配置和各种输出选项
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

        # 嵌入标记，用于输入的词嵌入
        self.embed_tokens = embed_tokens
        # 标记是否为解码器
        self.is_decoder = config.is_decoder

        # 设置模型配置和隐藏层数量
        self.config = config
        self.num_hidden_layers = config.num_layers

        # 创建 T5 模型的每一个块 TFT5Block，并添加到 block 列表中
        self.block = [
            TFT5Block(config, has_relative_attention_bias=bool(i == 0), name=f"block_._{i}")
            for i in range(config.num_layers)
        ]

        # 最终层归一化，用于最终输出
        self.final_layer_norm = TFT5LayerNorm(
            config.d_model, epsilon=config.layer_norm_epsilon, name="final_layer_norm"
        )
        # dropout 层，用于模型的正则化
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    # 用于剪枝特定头部的方法，当前未在 TF 2.0 模型库中实现
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library for TF 2.0 models
    # 定义一个方法 `call`，用于模型推理过程中的前向传播
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 如果模型已经构建完成，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建完成
        self.built = True
        # 如果存在最终层规范化模块，构建其内部结构
        if getattr(self, "final_layer_norm", None) is not None:
            # 在命名空间 `final_layer_norm` 下构建最终层规范化模块
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build(None)
        # 如果存在块结构，逐层构建每个块
        if getattr(self, "block", None) is not None:
            # 遍历模型的每个块
            for layer in self.block:
                # 在每个层的命名空间下构建该层
                with tf.name_scope(layer.name):
                    layer.build(None)
####################################################
# TFT5PreTrainedModel is a sub-class of keras.Model
# which take care of loading and saving pretrained weights
# and various common utilities.
# Here you just need to specify a few (self-explanatory)
# pointers for your model.
####################################################
class TFT5PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Specifies the configuration class to be used for this model
    config_class = T5Config

    # Prefix used to identify the base model within the saved weights
    base_model_prefix = "transformer"

    # List of keys representing layers that are authorized to be missing or unexpected during model loading
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\Wblock[\W_0]+layer[\W_1]+EncDecAttention\Wrelative_attention_bias"
    ]

    def get_input_embeddings(self):
        # Returns the shared input embeddings for the model
        return self.shared

    def set_input_embeddings(self, value):
        # Sets the shared input embeddings for the model and updates related components
        self.shared = value
        self.encoder.embed_tokens = self.shared
        if hasattr(self, "decoder"):
            self.decoder.embed_tokens = self.shared

    def _shift_right(self, input_ids):
        # Retrieves necessary configuration parameters
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # Asserts that decoder_start_token_id is defined
        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In TF T5 it is usually set to the"
            " pad_token_id. See T5 docs for more information"
        )

        # Constructs start_tokens tensor to prepend to input_ids
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensures dtype compatibility for concatenation
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        # Asserts that pad_token_id is defined
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

        # Replaces -100 values in shifted_input_ids with pad_token_id
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        # Verifies that shifted_input_ids contains only positive values and -100
        assert_gte0 = tf.debugging.assert_greater_equal(
            shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype)
        )

        # Ensures the assertion op is called by wrapping the result in an identity operation
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids


T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    # 文档字符串，描述了模型类的基本信息和用法说明。
    # 该模型继承自TFPreTrainedModel类，可以查看其超类文档以了解库实现的通用方法，
    # 如下载或保存模型、调整输入嵌入、剪枝头等。
    # 该模型也是一个keras.Model子类，可以像普通的TF 2.0 Keras模型一样使用，
    # 并参考TF 2.0文档中有关一般用法和行为的信息。
    
    # 提示（Tip）部分：
    # transformers库中的TensorFlow模型和层接受两种输入格式：
    # 1. 所有输入作为关键字参数（类似于PyTorch模型）；
    # 2. 所有输入作为列表、元组或字典传递给第一个位置参数。
    # 第二种格式得到支持是因为Keras方法更倾向于使用这种格式将输入传递给模型和层。
    # 因此，当使用model.fit()等方法时，只需传递您支持的任何格式的输入和标签即可“正常工作”！
    # 然而，如果您想在Keras方法之外（如在使用Keras Functional API创建自己的层或模型时）使用第二种格式，
    # 您可以使用以下三种可能性将所有输入Tensor聚集到第一个位置参数中。
    
    # 参数部分：
    # config参数接受一个T5Config类的实例，其中包含模型的所有参数。
    # 使用配置文件初始化模型不会加载与模型关联的权重，只加载配置。
    # 可以查看~PreTrainedModel.from_pretrained方法来加载模型的权重。
"""

T5_INPUTS_DOCSTRING = r"""
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        inputs (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。T5 是一个具有相对位置嵌入的模型，因此可以在右侧或左侧填充输入。
            
            可以使用 [`AutoTokenizer`] 获取索引。详细信息请参阅 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`]。
            
            若要了解有关预训练的 `inputs` 准备的更多信息，请查看 [T5 Training](./t5#training)。
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            遮罩，避免在填充的标记索引上执行注意力操作。遮罩的值选择在 `[0, 1]` 之间：

            - 1 表示**未遮罩**的标记，
            - 0 表示**遮罩**的标记。

            [什么是注意力遮罩？](../glossary#attention-mask)
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            可选，可以直接传递嵌入表示而不是 `input_ids`。如果要比模型的内部嵌入查找矩阵更精确地控制如何将 `input_ids` 索引转换为相关联的向量，这很有用。
        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于取消选择自注意模块的特定头部的遮罩。遮罩的值选择在 `[0, 1]` 之间：

            - 1 表示**未遮罩**的头部，
            - 0 表示**遮罩**的头部。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
        training (`bool`, *optional*, 默认为 `False`):
            是否在训练模式下使用模型（一些模块如 dropout 在训练和评估之间有不同行为）。
"""

_HEAD_MASK_WARNING_MSG = """
输入参数 `head_mask` 已分为两个参数 `head_mask` 和 `decoder_head_mask`。目前，`decoder_head_mask` 被设置为复制 `head_mask`，但此功能已被弃用，并将在未来版本中移除。
如果现在不想使用任何 `decoder_head_mask`，请设置 `decoder_head_mask = tf.ones((num_layers, num_heads))`。
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class TFT5Model(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建一个共享的嵌入层，用于模型的输入和输出
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(self.config.initializer_factor),
            name="shared",
        )
        # 添加额外的属性，用于指定层的名称作用域（用于加载/存储权重）
        self.shared.load_weight_prefix = "shared"

        # 复制编码器配置，设置不使用缓存，然后创建编码器层
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name="encoder")

        # 复制解码器配置，设置为解码器，指定解码器层数，然后创建解码器层
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFT5MainLayer(decoder_config, self.shared, name="decoder")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 模型的正向传播函数，支持多种输入和输出配置，详细说明见相关文档
        pass

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 共享的/绑定的权重需要在模型基本命名空间中
        # 在 tf.name_scope 的末尾添加 "/" 将其放置在根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            self.shared.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model_dim = config.d_model
        self.shared = keras.layers.Embedding(
            config.vocab_size,
            config.d_model,
            name="shared",
            embeddings_initializer=get_initializer(self.config.initializer_factor),
        )
        # Additional attribute to specify the expected name scope of the layer (for loading/storing weights)
        self.shared.load_weight_prefix = "shared"  # 设置共享层的权重前缀为 "shared"

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False  # 禁用编码器的缓存
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name="encoder")  # 初始化编码器模型

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFT5MainLayer(decoder_config, self.shared, name="decoder")  # 初始化解码器模型

        if not config.tie_word_embeddings:
            lm_head_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor)
            self.lm_head = keras.layers.Dense(
                config.vocab_size, use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # 初始化语言建模头部的全连接层，用于生成词汇表中的单词

        self.config = config

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.get_input_embeddings()  # 如果词嵌入是共享的，则返回输入嵌入
        else:
            # 在密集层中，核的形状为 (last_dim, units)，对于我们来说是 (dim, num_tokens)
            # value 的形状是 (num_tokens, dim)，因此需要转置
            return tf.transpose(self.lm_head.kernel)  # 返回语言建模头部的权重的转置

    def set_output_embeddings(self, value):
        if self.config.tie_word_embeddings:
            self.set_input_embeddings(value)  # 如果词嵌入是共享的，则设置输入嵌入
        else:
            lm_head_initializer = keras.initializers.RandomNormal(mean=0, stddev=self.config.initializer_factor)
            self.lm_head = keras.layers.Dense(
                shape_list(value)[0], use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # 初始化语言建模头部的全连接层，用于生成词汇表中的单词
            # 在密集层中，核的形状为 (last_dim, units)，对于我们来说是 (dim, num_tokens)
            # value 的形状是 (num_tokens, dim)，因此需要转置
            transposed_value = tf.transpose(value)
            self.lm_head.kernel = transposed_value  # 设置语言建模头部的权重为给定值的转置

    def get_encoder(self):
        return self.encoder  # 返回编码器模型

    def get_decoder(self):
        return self.decoder  # 返回解码器模型

    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)  # 将 T5 输入文档字符串添加到模型前向方法
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 定义输入参数 input_ids，类型为 TFModelInputType 或 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 attention_mask，类型为 np.ndarray 或 tf.Tensor 或 None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 decoder_input_ids，类型为 np.ndarray 或 tf.Tensor 或 None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 decoder_attention_mask，类型为 np.ndarray 或 tf.Tensor 或 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 head_mask，类型为 np.ndarray 或 tf.Tensor 或 None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 decoder_head_mask，类型为 np.ndarray 或 tf.Tensor 或 None
        encoder_outputs: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 encoder_outputs，类型为 np.ndarray 或 tf.Tensor 或 None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 定义输入参数 past_key_values，类型为 Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] 或 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 inputs_embeds，类型为 np.ndarray 或 tf.Tensor 或 None
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 decoder_inputs_embeds，类型为 np.ndarray 或 tf.Tensor 或 None
        labels: np.ndarray | tf.Tensor | None = None,  # 定义输入参数 labels，类型为 np.ndarray 或 tf.Tensor 或 None
        use_cache: Optional[bool] = None,  # 定义输入参数 use_cache，类型为 Optional[bool] 或 None
        output_attentions: Optional[bool] = None,  # 定义输入参数 output_attentions，类型为 Optional[bool] 或 None
        output_hidden_states: Optional[bool] = None,  # 定义输入参数 output_hidden_states，类型为 Optional[bool] 或 None
        return_dict: Optional[bool] = None,  # 定义输入参数 return_dict，类型为 Optional[bool] 或 None
        training: Optional[bool] = False,  # 定义输入参数 training，类型为 Optional[bool] 或 False
    ):
        # 此方法用于执行 Seq2Seq 模型的前向计算，接受多种输入参数，返回 TFSeq2SeqLMOutput 类型的输出结果

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values[1:]) if self.config.use_cache else None  # 根据 config 中的 use_cache 设置是否转换 past_key_values 为张量 pkv
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None  # 根据 config 中的 output_hidden_states 设置是否转换 decoder_hidden_states 为张量 dec_hs
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None  # 根据 config 中的 output_attentions 设置是否转换 decoder_attentions 为张量 dec_attns
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None  # 根据 config 中的 output_attentions 设置是否转换 cross_attentions 为张量 cross_attns
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None  # 根据 config 中的 output_hidden_states 设置是否转换 encoder_hidden_states 为张量 enc_hs
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None  # 根据 config 中的 output_attentions 设置是否转换 encoder_attentions 为张量 enc_attns

        return TFSeq2SeqLMOutput(
            logits=output.logits,  # 输出 logits
            past_key_values=pkv,  # 输出 past_key_values
            decoder_hidden_states=dec_hs,  # 输出 decoder_hidden_states
            decoder_attentions=dec_attns,  # 输出 decoder_attentions
            cross_attentions=cross_attns,  # 输出 cross_attentions
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 输出 encoder_last_hidden_state
            encoder_hidden_states=enc_hs,  # 输出 encoder_hidden_states
            encoder_attentions=enc_attns,  # 输出 encoder_attentions
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入参数 input_ids
        past_key_values=None,  # 输入参数 past_key_values，默认为 None
        attention_mask=None,  # 输入参数 attention_mask，默认为 None
        decoder_attention_mask=None,  # 输入参数 decoder_attention_mask，默认为 None
        head_mask=None,  # 输入参数 head_mask，默认为 None
        decoder_head_mask=None,  # 输入参数 decoder_head_mask，默认为 None
        use_cache=None,  # 输入参数 use_cache，默认为 None
        encoder_outputs=None,  # 输入参数 encoder_outputs，默认为 None
        **kwargs,  # 其他关键字参数，不做具体注释
    ):
        # 根据是否使用过去的键值对 past_key_values，截取 input_ids 的最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": None,  # 需要传递该参数以确保 Keras.layer.__call__ 正常运行
            "decoder_input_ids": input_ids,  # 返回处理后的 decoder_input_ids
            "past_key_values": past_key_values,  # 返回 past_key_values
            "encoder_outputs": encoder_outputs,  # 返回 encoder_outputs
            "attention_mask": attention_mask,  # 返回 attention_mask
            "decoder_attention_mask": decoder_attention_mask,  # 返回 decoder_attention_mask
            "head_mask": head_mask,  # 返回 head_mask
            "decoder_head_mask": decoder_head_mask,  # 返回 decoder_head_mask
            "use_cache": use_cache,  # 返回 use_cache
        }
    # 从标签中生成解码器的输入 ID，通过将标签向右移动一位来实现
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return self._shift_right(labels)

    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型已构建的标志
        self.built = True
        
        # 共享/共用权重预期位于模型基本命名空间中
        # 将"/"添加到 tf.name_scope 的末尾（而不是开头！）将其放置在根命名空间而不是当前命名空间中
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享层
            self.shared.build(None)
        
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果存在解码器，则构建解码器
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
        
        # 如果存在语言模型头部，则构建语言模型头部
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                # 构建语言模型头部，输入形状为 [None, None, self.config.d_model]
                self.lm_head.build([None, None, self.config.d_model])
@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-stateswithout any specific head on top.",
    T5_START_DOCSTRING,
)
class TFT5EncoderModel(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 定义共享的嵌入层，用于输入数据的编码
        self.shared = keras.layers.Embedding(
            config.vocab_size,
            config.d_model,
            name="shared",
            embeddings_initializer=get_initializer(self.config.initializer_factor),
        )
        # 加载权重时用于指定层的名称范围
        self.shared.load_weight_prefix = "shared"

        # 复制配置以用于编码器，并禁用缓存以确保每次调用都是独立的
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        # 初始化 T5 主层作为编码器
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name="encoder")

    def get_encoder(self):
        # 返回编码器对象
        return self.encoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        r"""
        Runs the T5 encoder model on inputs.

        Returns:
            TFBaseModelOutput or Tuple: Depending on `return_dict`, returns either a dictionary or a tuple of model outputs.

        Examples:
        
        ```python
        >>> from transformers import AutoTokenizer, TFT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = TFT5EncoderModel.from_pretrained("google-t5/t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids)
        ```
        """

        # 调用编码器计算输出
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果不需要返回字典，则直接返回编码器输出
        if not return_dict:
            return encoder_outputs

        # 返回 TFBaseModelOutput 类型的对象，封装编码器输出
        return TFBaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 如果模型已经构建完成，则直接返回，不再重复构建
    if self.built:
        return
    
    # 设置模型已构建标志为 True
    self.built = True
    
    # 共享/共用权重预期应位于模型基础命名空间中
    # 在 tf.name_scope 后面添加 "/"（而不是在开头添加！）将其放置在根命名空间而不是当前命名空间。
    with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
        # 构建共享部分模型
        self.shared.build(None)
    
    # 如果存在编码器部分
    if getattr(self, "encoder", None) is not None:
        with tf.name_scope(self.encoder.name):
            # 构建编码器模型
            self.encoder.build(None)
```