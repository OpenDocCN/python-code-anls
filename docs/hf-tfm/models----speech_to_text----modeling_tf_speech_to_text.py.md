# `.\models\speech_to_text\modeling_tf_speech_to_text.py`

```
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow Speech2Text model."""


from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation, glu
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
    TFSharedEmbeddings,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_speech_to_text import Speech2TextConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Speech2TextConfig"
_CHECKPOINT_FOR_DOC = "facebook/s2t-small-librispeech-asr"


TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/s2t-small-librispeech-asr",
    # See all Speech2Text models at https://huggingface.co/models?filter=speech_to_text
]


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart.shift_tokens_right
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将输入的 token ids 向右移动一位，用于解码过程
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建起始 token，填充为 decoder_start_token_id
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将 input_ids 向右移动一位，构成 shifted_input_ids
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 替换 labels 中可能的 -100 值为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # "Verify that `labels` has only positive values and -100"
    # 断言 shifted_input_ids 中的值大于等于 0
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # Make sure the assertion op is called by wrapping the result in an identity no-op
    # 确保断言操作被调用，通过将结果包装在一个恒等 no-op 中
    return tf.identity(shifted_input_ids, name="shifted_input_ids")
    # 使用 TensorFlow 中的控制依赖机制，确保 assert_gte0（大于等于0的断言）被执行
    with tf.control_dependencies([assert_gte0]):
        # 使用 tf.identity 创建 shifted_input_ids 的副本，并确保在执行 assert_gte0 后再进行
        shifted_input_ids = tf.identity(shifted_input_ids)
    
    # 返回经过控制依赖处理后的 shifted_input_ids
    return shifted_input_ids
# Copied from transformers.models.bart.modeling_tf_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 获取 batch size 和目标序列长度
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    # 创建一个全为 LARGE_NEGATIVE 的矩阵作为初始 mask
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个条件向量，长度与 mask 的最后一个维度相同
    mask_cond = tf.range(shape_list(mask)[-1])
    # 根据条件向量设置 mask 中的值，实现上三角为 0，其余为 LARGE_NEGATIVE
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    
    # 如果 past_key_values_length 大于 0，则在 mask 左侧添加相应长度的 0
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取原始 mask 的长度
    src_len = shape_list(mask)[1]
    # 如果未指定 tgt_len，则默认为 src_len
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量值为 1.0
    one_cst = tf.constant(1.0)
    # 将 mask 转换为与目标长度相关的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在 mask 的第二个维度上扩展为 `[bsz, 1, tgt_len, src_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的 mask，将其中的 1.0 改为 LARGE_NEGATIVE
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFConv1dSubsampler(keras.layers.Layer):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = config.num_conv_layers
        self.in_channels = config.input_feat_per_channel * config.input_channels
        self.mid_channels = config.conv_channels
        self.out_channels = config.d_model
        self.kernel_sizes = config.conv_kernel_sizes

        # 创建一系列的 1D 卷积层，每一层使用不同的参数配置
        self.conv_layers = [
            keras.layers.Conv1D(
                filters=self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2,
                kernel_size=k,
                strides=2,
                name=f"conv_layers.{i}",
            )
            for i, k in enumerate(self.kernel_sizes)
        ]
    def call(self, input_features: tf.Tensor) -> tf.Tensor:
        # TF Conv1D assumes Batch x Time x Channels, same as the input
        # 将输入特征转换为 float32 类型的张量
        hidden_states = tf.cast(input_features, tf.float32)
        for i, conv in enumerate(self.conv_layers):
            # equivalent to `padding=k // 2` on PT's `nn.Conv1d`
            # 计算填充长度，使得卷积操作的输出与输入在时间维度上保持一致
            pad_len = self.kernel_sizes[i] // 2
            hidden_shapes = shape_list(hidden_states)
            # 在时间维度两侧进行零填充，以保持卷积操作后的维度一致性
            hidden_states = tf.concat(
                (
                    tf.zeros((hidden_shapes[0], pad_len, hidden_shapes[2])),
                    hidden_states,
                    tf.zeros((hidden_shapes[0], pad_len, hidden_shapes[2])),
                ),
                axis=1,
            )

            # 应用卷积操作
            hidden_states = conv(hidden_states)
            # 在通道维度上应用门控线性单元（GLU）操作
            hidden_states = glu(hidden_states, axis=2)  # GLU over the Channel dimension
        # 返回处理后的隐藏状态张量
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv_layers", None) is not None:
            for i, layer in enumerate(self.conv_layers):
                with tf.name_scope(layer.name):
                    # 根据卷积层的要求构建该层的参数
                    layer.build([None, None, self.in_channels] if i == 0 else [None, None, self.mid_channels // 2])
class TFSpeech2TextSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        # 偏移量，用于生成位置编码
        self.offset = 2
        # 嵌入维度
        self.embedding_dim = embedding_dim
        # 填充索引，指定填充位置的特殊索引
        self.padding_idx = padding_idx
        # 初始化嵌入权重矩阵
        self.embedding_weights = self._get_embedding(num_positions + self.offset, embedding_dim, padding_idx)

    @staticmethod
    def _get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> tf.Tensor:
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        # 计算一半的维度
        half_dim = embedding_dim // 2
        # 计算频率
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(tf.range(num_embeddings, dtype=tf.float32), axis=1) * tf.expand_dims(emb, axis=0)
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), shape=[num_embeddings, -1])
        if embedding_dim % 2 == 1:
            # 如果维度是奇数，补零
            emb = tf.concat([emb, tf.zeros((num_embeddings, 1))], axis=1)
        if padding_idx is not None:
            # 如果有填充索引，处理填充位置
            emb = tf.concat([emb[:padding_idx, :], tf.zeros((1, tf.shape(emb)[1])), emb[padding_idx + 1 :, :]], axis=0)
        return emb

    def call(self, input_ids: tf.Tensor, past_key_values_length: int = 0) -> tf.Tensor:
        bsz, seq_len = shape_list(input_ids)
        # 根据输入的 token ids 创建位置 ids，保留任何填充的 token 的填充状态
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)

        # 生成位置嵌入
        embeddings = self._get_embedding(
            self.padding_idx + 1 + seq_len + self.offset + past_key_values_length, self.embedding_dim, self.padding_idx
        )
        return tf.reshape(tf.gather(embeddings, tf.reshape(position_ids, (-1,)), axis=0), (bsz, seq_len, -1))

    @staticmethod
    def create_position_ids_from_input_ids(
        input_ids: tf.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ) -> tf.Tensor:
        # 从输入的 token ids 创建位置 ids
        # 这里会根据填充索引和历史键值长度处理位置 ids
        pass  # 实际的实现将在代码中完成，这里只是声明函数结构
    def make_positions(x: tf.Tensor) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
    
        Args:
            x: tf.Tensor, input tensor where positions will be computed.
    
        Returns:
            tf.Tensor, tensor with replaced positions.
        """
        # 创建一个掩码，标记输入张量中不是填充符号的位置
        mask = tf.cast(tf.math.not_equal(input_ids, padding_idx), dtype=tf.int32)
        # 计算增量索引，加上过去键值的长度，并乘以掩码以忽略填充符号
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask
        # 返回增量索引，并将数据类型转换为 int64，同时加上填充索引
        return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx
# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制并修改为 Speech2Text
class TFSpeech2TextAttention(keras.layers.Layer):
    """多头注意力机制，基于 'Attention Is All You Need'"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 初始化注意力层的参数
        self.embed_dim = embed_dim  # 注意力层的嵌入维度
        self.num_heads = num_heads  # 注意力头的数量
        self.dropout = keras.layers.Dropout(dropout)  # dropout 层
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器模式

        # 初始化线性映射层
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")  # K 线性映射
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")  # Q 线性映射
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")  # V 线性映射
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")  # 输出线性映射

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # 将输入张量重新形状为 [batch_size, num_heads, seq_len, head_dim]，并转置为 [batch_size, num_heads, seq_len, head_dim]
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 注意力层的前向传播函数
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 构建线性映射层
            if getattr(self, "k_proj", None) is not None:
                with tf.name_scope(self.k_proj.name):
                    self.k_proj.build([None, None, self.embed_dim])
            if getattr(self, "q_proj", None) is not None:
                with tf.name_scope(self.q_proj.name):
                    self.q_proj.build([None, None, self.embed_dim])
            if getattr(self, "v_proj", None) is not None:
                with tf.name_scope(self.v_proj.name):
                    self.v_proj.build([None, None, self.embed_dim])
            if getattr(self, "out_proj", None) is not None:
                with tf.name_scope(self.out_proj.name):
                    self.out_proj.build([None, None, self.embed_dim])

class TFSpeech2TextEncoderLayer(keras.layers.Layer):
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config: Speech2TextConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为模型配置中的 d_model
        self.embed_dim = config.d_model
        # 创建自注意力层对象，使用自定义的注意力头数和丢弃率配置
        self.self_attn = TFSpeech2TextAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建自注意力层的层归一化层
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建丢弃层，使用配置中的丢弃率
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数对象，根据配置的激活函数名
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数丢弃层，使用配置中的激活函数丢弃率
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层 fc1，输出维度为配置中的 encoder_ffn_dim
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层 fc2，输出维度与嵌入维度相同
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终层的层归一化层
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 将配置对象保存到实例中
        self.config = config

    # 调用函数，执行实际的前向计算过程
    def call(
        self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, training: bool = False
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码张量，形状为 `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值表示。
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码张量，形状为 `(encoder_attention_heads,)`
            training (`bool`): 是否处于训练模式
        """
        # 保存残差连接，用于后续加法操作
        residual = hidden_states
        # 执行自注意力层的层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行自注意力计算，并返回计算结果、注意力权重及额外信息
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training,
        )

        # 断言确保自注意力操作未修改查询的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用丢弃操作到自注意力结果
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接到自注意力结果上
        hidden_states = residual + hidden_states

        # 保存残差连接，用于后续加法操作
        residual = hidden_states
        # 执行最终层的层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数到第一个全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数的丢弃操作到第一个全连接层的输出
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 应用第二个全连接层，并输出结果
        hidden_states = self.fc2(hidden_states)
        # 应用丢弃操作到第二个全连接层的输出
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接到第二个全连接层的输出上
        hidden_states = residual + hidden_states

        # 返回最终的隐藏状态和自注意力权重
        return hidden_states, self_attn_weights
    # 构建神经网络层的方法，用于在输入形状已知或未知时构建层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为层设置命名空间
            with tf.name_scope(self.self_attn.name):
                # 调用 self_attn 层的 build 方法
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为层设置命名空间
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 调用 self_attn_layer_norm 层的 build 方法，传入输入形状
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为层设置命名空间
            with tf.name_scope(self.fc1.name):
                # 调用 fc1 层的 build 方法，传入输入形状
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为层设置命名空间
            with tf.name_scope(self.fc2.name):
                # 调用 fc2 层的 build 方法，传入输入形状的编码器维度
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为层设置命名空间
            with tf.name_scope(self.final_layer_norm.name):
                # 调用 final_layer_norm 层的 build 方法，传入输入形状
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFSpeech2TextDecoderLayer(keras.layers.Layer):
    # 定义 TF Speech-to-Text 解码器层的类
    def __init__(self, config: Speech2TextConfig, **kwargs):
        # 初始化函数，接收配置参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化方法

        # 设置嵌入维度为模型配置中的维度
        self.embed_dim = config.d_model

        # 创建自注意力层，用于解码器的自注意力机制
        self.self_attn = TFSpeech2TextAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )

        # dropout 层，用于在激活函数前进行随机失活
        self.dropout = keras.layers.Dropout(config.dropout)

        # 获取激活函数并设置激活函数的随机失活
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # 层归一化，用于自注意力层输出的归一化
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")

        # 创建编码器注意力层，用于解码器与编码器之间的注意力机制
        self.encoder_attn = TFSpeech2TextAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )

        # 编码器注意力层的归一化
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")

        # 第一个全连接层，用于解码器中的前馈神经网络
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")

        # 第二个全连接层，输出维度与嵌入维度相同
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")

        # 最终的层归一化，用于前馈神经网络输出的归一化
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

        # 存储配置参数
        self.config = config

    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training=False,
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不进行重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder_attn 层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder_attn_layer_norm 层
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFSpeech2TextPreTrainedModel(TFPreTrainedModel):
    # 指定配置类
    config_class = Speech2TextConfig
    # 模型前缀用于加载
    base_model_prefix = "model"
    # 主要输入特征名称
    main_input_name = "input_features"
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"encoder.embed_positions.weights"]

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        计算卷积层的输出长度
        """
        # 根据配置中的卷积层数进行迭代计算
        for _ in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    @property
    def input_signature(self):
        # 定义模型输入的签名
        return {
            "input_features": tf.TensorSpec(
                # 输入特征的形状：(None, None, 输入通道数 * 每个通道的特征数)
                (None, None, self.config.input_feat_per_channel * self.config.input_channels),
                tf.float32,
                name="input_features",
            ),
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            "decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),
            "decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask"),
        }


SPEECH_TO_TEXT_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
"""
    # 注意：在使用子类化（subclassing）创建模型和层时，您不需要担心以下任何内容，因为您可以像将输入传递给任何其他Python函数一样进行传递！

    </Tip>

    # 参数:
    # config ([`Speech2TextConfig`]):
    #     包含模型所有参数的模型配置类。使用配置文件初始化时，不会加载与模型关联的权重，只加载配置信息。
    #     可以查看[`~TFPreTrainedModel.from_pretrained`]方法以加载模型权重。
"""


SPEECH_TO_TEXT_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFSpeech2TextEncoder(keras.layers.Layer):
    config_class = Speech2TextConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFSpeech2TextEncoderLayer`].

    Args:
        config: Speech2TextConfig
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # 初始化 dropout 层，使用指定的 dropout 概率
        self.dropout = keras.layers.Dropout(config.dropout)
        # layerdrop 是指定的 encoder_layerdrop 参数
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = tf.math.sqrt(float(embed_dim)) if config.scale_embedding else 1.0

        # 创建 TFConv1dSubsampler 对象，用于卷积操作
        self.conv = TFConv1dSubsampler(config, name="conv")

        # 创建 TFSpeech2TextSinusoidalPositionalEmbedding 对象，用于位置编码
        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_source_positions,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            name="embed_positions",
        )
        
        # 创建多个 TFSpeech2TextEncoderLayer 对象，作为 Transformer 编码器的层
        self.layers = [TFSpeech2TextEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        
        # 创建 LayerNormalization 层，用于归一化每个编码器层的输出
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """
        # 计算卷积层的输出长度
        for _ in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # 如果 attention_mask 的维度大于2，则取最后一个维度
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        # 计算特征提取的输出长度
        subsampled_lengths = self._get_feat_extract_output_lengths(tf.math.reduce_sum(attention_mask, -1))
        bsz = shape_list(attention_mask)[0]
        # 创建注意力掩码，将特定位置标记为1
        indices = tf.concat(
            (
                tf.expand_dims(tf.range(bsz, dtype=attention_mask.dtype), -1),
                tf.expand_dims(subsampled_lengths - 1, -1),
            ),
            axis=-1,
        )
        attention_mask = tf.scatter_nd(indices=indices, updates=tf.ones(bsz), shape=[bsz, feature_vector_length])
        # 反转和累积注意力掩码
        attention_mask = tf.cast(tf.reverse(tf.math.cumsum(tf.reverse(attention_mask, [-1]), -1), [-1]), tf.int64)
        return attention_mask

    @unpack_inputs
    def call(
        self,
        input_features=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    # 定义神经网络模型的 build 方法，用于构建模型的各个层次和参数
    def build(self, input_shape=None):
        # 如果已经构建过模型，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 如果存在卷积层，则构建卷积层
        if getattr(self, "conv", None) is not None:
            # 使用卷积层的名称作为 TensorFlow 的命名空间
            with tf.name_scope(self.conv.name):
                self.conv.build(None)
        
        # 如果存在位置嵌入层，则构建位置嵌入层
        if getattr(self, "embed_positions", None) is not None:
            # 使用位置嵌入层的名称作为 TensorFlow 的命名空间
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        
        # 如果存在层归一化层，则构建层归一化层
        if getattr(self, "layer_norm", None) is not None:
            # 使用层归一化层的名称作为 TensorFlow 的命名空间
            with tf.name_scope(self.layer_norm.name):
                # 构建层归一化层，输入形状为 [None, None, self.config.d_model]
                self.layer_norm.build([None, None, self.config.d_model])
        
        # 如果存在多个层，则分别构建每个层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                # 使用每个层的名称作为 TensorFlow 的命名空间
                with tf.name_scope(layer.name):
                    # 构建当前层，输入形状为 None（即不限制输入形状）
                    layer.build(None)
# 使用 keras_serializable 装饰器使类可序列化
@keras_serializable
class TFSpeech2TextDecoder(keras.layers.Layer):
    # 将 config_class 属性设置为 Speech2TextConfig 类
    config_class = Speech2TextConfig

    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFSpeech2TextDecoderLayer`]

    Args:
        config: Speech2TextConfig
    """

    # 初始化方法，接受一个 config 参数和其他关键字参数
    def __init__(self, config: Speech2TextConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 设置 layerdrop 属性为 config.decoder_layerdrop
        self.layerdrop = config.decoder_layerdrop
        # 设置 padding_idx 属性为 config.pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置 max_target_positions 属性为 config.max_target_positions
        self.max_target_positions = config.max_target_positions
        # 如果 config.scale_embedding 为 True，则设置 embed_scale 为 d_model 的平方根，否则为 1.0
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        # 创建 TFSharedEmbeddings 对象并赋值给 embed_tokens 属性
        self.embed_tokens = TFSharedEmbeddings(config.vocab_size, config.d_model, name="embed_tokens")

        # 创建 TFSpeech2TextSinusoidalPositionalEmbedding 对象并赋值给 embed_positions 属性
        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_target_positions,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx,
            name="embed_positions",
        )

        # 创建包含 config.decoder_layers 个 TFSpeech2TextDecoderLayer 的列表并赋值给 layers 属性
        self.layers = [TFSpeech2TextDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        
        # 创建 LayerNormalization 层并赋值给 layer_norm 属性
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建 Dropout 层并赋值给 dropout 属性
        self.dropout = keras.layers.Dropout(config.dropout)

    # 获取 embed_tokens 属性的方法
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置 embed_tokens 属性的方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 使用 unpack_inputs 装饰器定义 call 方法，接受多个参数用于 Transformer 解码器的前向传播
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 此处实现前向传播逻辑，具体内容需要进一步详细注释，但不在此处进行总结

    # build 方法用于构建层，当被调用时检查是否已经构建，如果已构建则直接返回，否则构建各层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 将 built 属性标记为 True，表示已构建
        self.built = True

        # 如果 embed_tokens 属性存在，则构建其内部结构
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)

        # 如果 embed_positions 属性存在，则构建其内部结构
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)

        # 如果 layer_norm 属性存在，则构建其内部结构
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])

        # 遍历 layers 列表中的每一层，并构建其内部结构
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 使用 keras_serializable 装饰器使类可序列化
@keras_serializable
class TFSpeech2TextMainLayer(keras.layers.Layer):
    # 将 config_class 属性设置为 Speech2TextConfig 类
    config_class = Speech2TextConfig
    # 初始化方法，接受一个配置对象 config 和其他关键字参数
    def __init__(self, config: Speech2TextConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置对象保存在实例变量中
        self.config = config

        # 创建一个 TFSpeech2TextEncoder 对象并保存在实例变量 encoder 中
        self.encoder = TFSpeech2TextEncoder(config, name="encoder")
        # 创建一个 TFSpeech2TextDecoder 对象并保存在实例变量 decoder 中
        self.decoder = TFSpeech2TextDecoder(config, name="decoder")

    # 获取输入嵌入的方法，返回 decoder 的 embed_tokens 属性
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入的方法，接受新的嵌入向量并将其赋值给 decoder 的 embed_tokens 属性
    def set_input_embeddings(self, new_embeddings):
        self.decoder.embed_tokens = new_embeddings

    # 装饰器函数，用于解包输入参数
    @unpack_inputs
    def call(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        # 此处是模型的调用方法，接受多个输入参数，并进行相应的处理

    # build 方法用于构建模型，在第一次调用时执行
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果实例变量中存在 encoder 对象，则在命名空间下构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果实例变量中存在 decoder 对象，则在命名空间下构建 decoder
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 定义一个基于 TFSpeech2TextPreTrainedModel 的具体模型类 TFSpeech2TextModel，用于输出未经特定头部处理的原始隐藏状态
@add_start_docstrings(
    "The bare Speech2Text Model outputting raw hidden-states without any specific head on top.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
class TFSpeech2TextModel(TFSpeech2TextPreTrainedModel):
    
    # 初始化方法，接受一个 Speech2TextConfig 类型的配置对象和其他可选参数
    def __init__(self, config: Speech2TextConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建一个 TFSpeech2TextMainLayer 对象作为模型的主层，使用给定的配置对象和名称
        self.model = TFSpeech2TextMainLayer(config, name="model")

    # 获取模型的编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 获取模型的解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 定义模型的调用方法，接受多个输入参数和一些可选的输出控制标志，返回模型输出的元组或 TFSeq2SeqModelOutput 类型
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_features: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[Tuple, TFSeq2SeqModelOutput]:
        
        # 调用模型的主层对象，传递所有参数和标志，并接收输出结果
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型的输出结果
        return outputs
    # 定义一个方法用于生成模型的输出
    def serving_output(self, output):
        # 如果配置中使用缓存，则获取输出中的过去关键值的第二项，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出隐藏状态，则将输出中的解码器隐藏状态转换为 TensorFlow 张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将输出中的解码器注意力权重转换为 TensorFlow 张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出注意力权重，则将输出中的交叉注意力权重转换为 TensorFlow 张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出隐藏状态，则将输出中的编码器隐藏状态转换为 TensorFlow 张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将输出中的编码器注意力权重转换为 TensorFlow 张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，封装了模型的输出
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型已存在
        if getattr(self, "model", None) is not None:
            # 使用模型的名称空间构建模型，输入形状为 None
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 定义一个基于 TFSpeech2TextPreTrainedModel 和 TFCausalLanguageModelingLoss 的模型类，用于语音到文本转换，并具有语言建模头部
@add_start_docstrings(
    "The Speech2Text Model with a language modeling head. Can be used for summarization.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
class TFSpeech2TextForConditionalGeneration(TFSpeech2TextPreTrainedModel, TFCausalLanguageModelingLoss):
    
    # 初始化方法，接受一个 Speech2TextConfig 对象作为参数
    def __init__(self, config: Speech2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 TFSpeech2TextMainLayer 对象作为模型主体，并命名为 "model"
        self.model = TFSpeech2TextMainLayer(config, name="model")
        # 创建一个 Dense 层作为语言建模头部，输出维度为 config.vocab_size，不使用偏置
        self.lm_head = keras.layers.Dense(self.config.vocab_size, use_bias=False, name="lm_head")
        # 设置是否支持在 XLA 生成中使用的标志为 False
        # TODO (Joao): investigate why Speech2Text has numerical issues in XLA generate
        self.supports_xla_generation = False
        # 将传入的 config 对象保存到实例变量中
        self.config = config

    # 返回模型的编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 返回模型的解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 重新调整 token embeddings 的大小，返回更新后的 embeddings
    def resize_token_embeddings(self, new_num_tokens: int) -> tf.Variable:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    # 返回语言建模头部
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 模型的前向传播方法，接受多种输入参数并返回 TFSeq2SeqLMOutput 类型的输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_features: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    # 定义一个方法用于处理模型输出，并根据配置选择性地返回不同的张量
    def serving_output(self, output):
        # 如果配置要求使用缓存，则获取输出中的过去键值（past_key_values）的第二个元素
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力权重，则将输出的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqLMOutput 对象，包含不同类型的模型输出
        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 准备用于生成的输入参数，根据条件截取 decoder_input_ids
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果存在 past_key_values，则截取 decoder_input_ids 的最后一个元素作为输入
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含用于生成的输入参数
        return {
            "input_features": None,  # 需要传递以使 Keras.layer.__call__ 正常运行
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能用于调试）
        }

    # 构建方法，用于建立模型的组件
    def build(self, input_shape=None):
        # 如果已经建立过，则直接返回
        if self.built:
            return
        # 标记模型已经建立
        self.built = True
        # 如果存在模型对象，则在命名空间下建立模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        # 如果存在 lm_head 对象，则在命名空间下建立 lm_head
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.d_model])

    # 转换 TensorFlow 权重名称到 PyTorch 权重名称的方法
    def tf_to_pt_weight_rename(self, tf_weight):
        # 如果输入的 TensorFlow 权重名称是 "lm_head.weight"，则返回对应的 PyTorch 权重名称
        if tf_weight == "lm_head.weight":
            return tf_weight, "model.decoder.embed_tokens.weight"
        else:
            return (tf_weight,)
```