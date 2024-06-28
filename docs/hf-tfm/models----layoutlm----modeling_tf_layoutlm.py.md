# `.\models\layoutlm\modeling_tf_layoutlm.py`

```
# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
""" TF 2.0 LayoutLM model."""


from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFMaskedLMOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutLMConfig"

TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlm-base-uncased",
    "microsoft/layoutlm-large-uncased",
]


class TFLayoutLMEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 TFLayoutLMEmbeddings 类
        self.config = config  # 保存配置对象
        self.hidden_size = config.hidden_size  # 隐藏层大小从配置中获取
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数从配置中获取
        self.max_2d_position_embeddings = config.max_2d_position_embeddings  # 二维最大位置嵌入数从配置中获取
        self.initializer_range = config.initializer_range  # 初始化范围从配置中获取
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")  # LayerNorm 层，使用配置中的 epsilon
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)  # Dropout 层，使用配置中的 dropout 率
    # 定义神经网络层的构建方法，用于构建模型的输入形状
    def build(self, input_shape=None):
        # 在名为"word_embeddings"的命名空间下创建权重张量，形状为[vocab_size, hidden_size]
        self.weight = self.add_weight(
            name="weight",
            shape=[self.config.vocab_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在名为"token_type_embeddings"的命名空间下创建权重张量，形状为[type_vocab_size, hidden_size]
        self.token_type_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.config.type_vocab_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在名为"position_embeddings"的命名空间下创建权重张量，形状为[max_position_embeddings, hidden_size]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在名为"x_position_embeddings"的命名空间下创建权重张量，形状为[max_2d_position_embeddings, hidden_size]
        self.x_position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_2d_position_embeddings, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在名为"y_position_embeddings"的命名空间下创建权重张量，形状为[max_2d_position_embeddings, hidden_size]
        self.y_position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_2d_position_embeddings, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在名为"h_position_embeddings"的命名空间下创建权重张量，形状为[max_2d_position_embeddings, hidden_size]
        self.h_position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_2d_position_embeddings, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 在名为"w_position_embeddings"的命名空间下创建权重张量，形状为[max_2d_position_embeddings, hidden_size]
        self.w_position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_2d_position_embeddings, self.hidden_size],
            initializer=get_initializer(self.initializer_range),
        )

        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return

        # 标记模型已经构建
        self.built = True

        # 如果存在LayerNorm层，则在其命名空间下构建LayerNorm层，输入形状为[None, None, hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 定义神经网络层的调用方法，处理输入数据的前向传播
    def call(
        self,
        input_ids: tf.Tensor = None,
        bbox: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
        ```
    ) -> tf.Tensor:
        """
        应用基于输入张量的嵌入。

        Returns:
            final_embeddings (`tf.Tensor`): 输出的嵌入张量。
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 从权重矩阵中根据 input_ids 获取对应的嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            # 如果未提供 token_type_ids，则初始化为全零张量，形状与 inputs_embeds 最后一维之前相同
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            # 如果未提供 position_ids，则创建一个范围为 [0, input_shape[-1]) 的张量
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        if position_ids is None:
            # 如果再次出现未提供 position_ids 的情况，则创建一个范围为 [0, input_shape[-1]) 的张量
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        if bbox is None:
            # 如果未提供 bbox，则初始化为全零张量，形状为 input_shape 加上 [4]
            bbox = bbox = tf.fill(input_shape + [4], value=0)
        try:
            # 根据 bbox 的坐标值从位置嵌入矩阵中获取对应的位置嵌入
            left_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 0])
            upper_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 1])
            right_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 2])
            lower_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox`coordinate values should be within 0-1000 range.") from e
        # 根据 bbox 的高度和宽度计算对应的位置嵌入
        h_position_embeddings = tf.gather(self.h_position_embeddings, bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = tf.gather(self.w_position_embeddings, bbox[:, :, 2] - bbox[:, :, 0])

        # 根据 position_ids 获取位置嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 获取 token 类型嵌入
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        
        # 计算最终的嵌入向量，将各部分嵌入相加
        final_embeddings = (
            inputs_embeds
            + position_embeds
            + token_type_embeds
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
        )
        # 应用 LayerNorm 归一化
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->LayoutLM

class TFLayoutLMSelfAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化层参数
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 定义查询、键、值的全连接层
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 定义 Dropout 层
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将形状从 [batch_size, seq_length, all_head_size] 转换为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ):
        # 这里应该实现自注意力机制，包括查询、键、值的计算，Dropout 和注意力矩阵的计算
        # 实现详细逻辑在此处省略，应该包括自注意力、Multi-Head Attention 等

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 检查并构建查询、键、值的全连接层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->LayoutLM
class TFLayoutLMSelfOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义层归一化层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义 dropout 层，用于在训练时随机置零部分隐藏状态
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对隐藏状态进行全连接变换
        hidden_states = self.dense(inputs=hidden_states)
        # 对变换后的隐藏状态进行 dropout 操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将 dropout 后的结果与输入张量相加，并进行层归一化
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经定义了 dense 层，构建其参数
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果已经定义了 LayerNorm 层，构建其参数
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->LayoutLM
class TFLayoutLMAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义 LayoutLM 的自注意力层
        self.self_attention = TFLayoutLMSelfAttention(config, name="self")
        # 定义 LayoutLM 的输出层
        self.dense_output = TFLayoutLMSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 暂未实现裁剪注意力头部的方法
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用自注意力层，获取自注意力层的输出
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        # 将自注意力层的输出作为输入，调用输出层进行处理
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力信息，则将注意力信息添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义一个方法 `build`，用于构建神经网络层的计算图
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标记设置为已构建
        self.built = True
        
        # 如果存在自注意力层 `self_attention`
        if getattr(self, "self_attention", None) is not None:
            # 使用 `self_attention` 的名称作为命名空间，构建自注意力层
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        
        # 如果存在密集输出层 `dense_output`
        if getattr(self, "dense_output", None) is not None:
            # 使用 `dense_output` 的名称作为命名空间，构建密集输出层
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 复制并修改为 LayoutLM
class TFLayoutLMIntermediate(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于中间输出，指定输出单元数和初始化器
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 确定中间激活函数，可以是字符串或函数，根据配置选择合适的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层进行前向传播
        hidden_states = self.dense(inputs=hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            # 构建全连接层，指定输入形状和输出单元数
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 复制并修改为 LayoutLM
class TFLayoutLMOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于输出层，指定输出单元数和初始化器
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，用于归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于随机失活
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层进行前向传播
        hidden_states = self.dense(inputs=hidden_states)
        # 应用 Dropout 进行随机失活
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 应用 LayerNormalization 进行归一化，并加上残差连接
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            # 构建全连接层，指定输入形状和输出单元数
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            # 构建 LayerNormalization 层，指定输入形状和归一化的维度
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertLayer 复制并修改为 LayoutLM
class TFLayoutLMLayer(keras.layers.Layer):
    # 使用 LayoutLMConfig 对象和其他关键字参数初始化函数
    def __init__(self, config: LayoutLMConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 LayoutLMAttention 层，使用给定的配置和名称
        self.attention = TFLayoutLMAttention(config, name="attention")

        # 设置是否作为解码器的标志
        self.is_decoder = config.is_decoder

        # 设置是否添加跨注意力的标志
        self.add_cross_attention = config.add_cross_attention

        # 如果需要添加跨注意力
        if self.add_cross_attention:
            # 如果不是解码器模型，抛出数值错误异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")

            # 创建跨注意力层，使用给定的配置和名称
            self.crossattention = TFLayoutLMAttention(config, name="crossattention")

        # 创建 LayoutLMIntermediate 层，使用给定的配置和名称
        self.intermediate = TFLayoutLMIntermediate(config, name="intermediate")

        # 创建 LayoutLMOutput 层，使用给定的配置和名称
        self.bert_output = TFLayoutLMOutput(config, name="output")

    # 定义调用函数，处理输入的张量和参数，生成输出结果
    def call(
        self,
        hidden_states: tf.Tensor,                          # 输入的隐藏状态张量
        attention_mask: tf.Tensor,                         # 注意力掩码张量
        head_mask: tf.Tensor,                              # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,           # 编码器隐藏状态张量或空
        encoder_attention_mask: tf.Tensor | None,          # 编码器注意力掩码张量或空
        past_key_value: Tuple[tf.Tensor] | None,           # 过去的键值元组或空
        output_attentions: bool,                           # 是否输出注意力张量的标志
        training: bool = False,                            # 是否在训练模式的标志，默认为 False
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果有过去的键/值对，则取前两个，否则为 None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力机制处理隐藏状态
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        # 获取自注意力机制的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 输出中排除最后一个元组，因为它是自注意力缓存
            outputs = self_attention_outputs[1:-1]
            # 当前的键/值对为最后一个元组
            present_key_value = self_attention_outputs[-1]
        else:
            # 输出中排除第一个元素，因为它是隐藏状态处理后的输出
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
        

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 检查是否具有交叉注意力层
            if not hasattr(self, "crossattention"):
                # 如果传入了编码器隐藏状态，则需要通过设置 `config.add_cross_attention=True` 来实例化交叉注意力层
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的缓存键/值对在过去键/值对元组的第3、4个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力机制处理自注意力输出
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            # 获取交叉注意力机制的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力机制的输出添加到输出中，排除最后一个元组（如果有的话）
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的当前键/值对添加到现有的键/值对中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用BERT输出层处理中间层和注意力输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 如果输出注意力权重，则添加到输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值对作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 定义一个方法 `build`，用于构建模型的层次结构。如果已经构建过，则直接返回。
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 `attention` 属性，则构建 `attention` 层，并使用 `tf.name_scope` 包装作用域。
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在 `intermediate` 属性，则构建 `intermediate` 层，并使用 `tf.name_scope` 包装作用域。
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在 `bert_output` 属性，则构建 `bert_output` 层，并使用 `tf.name_scope` 包装作用域。
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果存在 `crossattention` 属性，则构建 `crossattention` 层，并使用 `tf.name_scope` 包装作用域。
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertEncoder复制代码，修改为LayoutLM模型的编码器
class TFLayoutLMEncoder(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建多个LayoutLM层，编号从"layer_._0"到"layer_._(config.num_hidden_layers - 1)"
        self.layer = [TFLayoutLMLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量列表
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量或None
        encoder_attention_mask: tf.Tensor | None,  # 编码器的注意力掩码张量或None
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,  # 先前的键值对或None
        use_cache: Optional[bool],  # 是否使用缓存的标志
        output_attentions: bool,  # 是否输出注意力张量
        output_hidden_states: bool,  # 是否输出隐藏状态张量
        return_dict: bool,  # 是否返回字典格式的输出
        training: bool = False,  # 是否处于训练模式
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则初始化空元组all_hidden_states
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力张量，则初始化空元组all_attentions
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力张量且配置中允许，则初始化空元组all_cross_attentions
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要使用缓存，则初始化空元组next_decoder_cache
        next_decoder_cache = () if use_cache else None
        # 遍历每个LayoutLM层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入all_hidden_states元组
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的先前键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的模块，计算层的输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层输出的第一个张量
            hidden_states = layer_outputs[0]

            # 如果需要使用缓存，则将当前层的最后一个输出加入next_decoder_cache
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力张量，则将当前层的注意力张量加入all_attentions元组
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置中允许添加交叉注意力且有编码器的隐藏状态，则将交叉注意力张量加入all_cross_attentions元组
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典格式的结果，则按顺序返回非None的张量组成的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回字典格式的TFBaseModelOutputWithPastAndCrossAttentions对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义神经网络层的构建方法，接受输入形状作为参数（默认为None）
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位表示已经构建过
        self.built = True
        # 如果存在layer属性（神经网络层），则逐个构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用 TensorFlow 的命名空间（name_scope），设置当前层的名称作用域
                with tf.name_scope(layer.name):
                    # 调用每一层的build方法进行具体的构建
                    layer.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->LayoutLM
class TFLayoutLMPooler(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # Initialize a dense layer for pooling with specified units, initializer, and activation function
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # Pooling operation: take the hidden state corresponding to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # Build the dense layer with the specified input shape and hidden size from config
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform with Bert->LayoutLM
class TFLayoutLMPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # Initialize a dense layer for transformation with specified units and initializer
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # Determine the activation function for transformation based on config
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # Layer normalization for stabilizing learning and handling covariate shift
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # Perform dense transformation followed by activation and layer normalization
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # Build the dense layer with the specified input shape and hidden size from config
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # Build the layer normalization with the specified input shape and hidden size from config
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead with Bert->LayoutLM
class TFLayoutLMLMPredictionHead(keras.layers.Layer):
    # This class definition was not provided in the snippet provided.
    pass
    # 初始化方法，接受配置对象和输入嵌入层作为参数
    def __init__(self, config: LayoutLMConfig, input_embeddings: keras.layers.Layer, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置配置对象和隐藏层大小
        self.config = config
        self.hidden_size = config.hidden_size

        # 使用 TFLayoutLMPredictionHeadTransform 对象对输入进行转换
        self.transform = TFLayoutLMPredictionHeadTransform(config, name="transform")

        # 输入嵌入层，这里的权重与输入嵌入的权重相同，但每个标记都有一个输出偏置
        self.input_embeddings = input_embeddings

    # 构建方法，用于构建模型层次结构
    def build(self, input_shape=None):
        # 添加一个形状为 (self.config.vocab_size,) 的可训练的零初始化偏置
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True

        # 如果存在转换层，构建转换层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出嵌入层对象
    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层的权重和词汇大小
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置字典
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置值
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 前向传播方法，接收隐藏状态并返回预测结果
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用转换层对隐藏状态进行转换
        hidden_states = self.transform(hidden_states=hidden_states)

        # 获取序列长度
        seq_length = shape_list(hidden_states)[1]

        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])

        # 矩阵乘法，将隐藏状态与输入嵌入层的权重相乘
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)

        # 将结果重塑为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])

        # 添加偏置项到结果张量
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回预测结果张量
        return hidden_states
# 从 transformers.models.bert.modeling_tf_bert.TFBertMLMHead 复制并将 Bert 替换为 LayoutLM
class TFLayoutLMMLMHead(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 初始化预测层，用于生成 MLM 的预测分数
        self.predictions = TFLayoutLMLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用预测层，生成预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 构建预测层
                self.predictions.build(None)


@keras_serializable
class TFLayoutLMMainLayer(keras.layers.Layer):
    config_class = LayoutLMConfig

    def __init__(self, config: LayoutLMConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 初始化主层的配置
        self.config = config

        # 初始化 LayoutLM 的嵌入层
        self.embeddings = TFLayoutLMEmbeddings(config, name="embeddings")
        
        # 初始化 LayoutLM 的编码器
        self.encoder = TFLayoutLMEncoder(config, name="encoder")
        
        # 如果需要添加池化层，则初始化 LayoutLM 的池化层
        self.pooler = TFLayoutLMPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回嵌入层
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置输入的嵌入层权重和词汇大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头，具体实现未完成
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        bbox: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义 build 方法，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将 built 标志设置为 True，表示模型已经构建
        self.built = True
        
        # 如果模型具有 embeddings 属性且不为 None，则构建 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            # 使用 embeddings 层的名称作为命名空间
            with tf.name_scope(self.embeddings.name):
                # 调用 embeddings 层的 build 方法构建其内部结构
                self.embeddings.build(None)
        
        # 如果模型具有 encoder 属性且不为 None，则构建 encoder 层
        if getattr(self, "encoder", None) is not None:
            # 使用 encoder 层的名称作为命名空间
            with tf.name_scope(self.encoder.name):
                # 调用 encoder 层的 build 方法构建其内部结构
                self.encoder.build(None)
        
        # 如果模型具有 pooler 属性且不为 None，则构建 pooler 层
        if getattr(self, "pooler", None) is not None:
            # 使用 pooler 层的名称作为命名空间
            with tf.name_scope(self.pooler.name):
                # 调用 pooler 层的 build 方法构建其内部结构
                self.pooler.build(None)
"""
This model class extends TFPreTrainedModel and provides methods for weights initialization, downloading pretrained models,
and handling input signatures.
"""

class TFLayoutLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.

    Attributes:
        config_class: A class attribute indicating the configuration class for this model.
        base_model_prefix: A string attribute representing the prefix used for the base model.
    """

    # Set the configuration class for this model
    config_class = LayoutLMConfig
    # Define the prefix for the base model
    base_model_prefix = "layoutlm"

    @property
    def input_signature(self):
        """
        Override the input_signature property of TFPreTrainedModel.

        Returns:
            dict: Updated signature including 'bbox' as a TensorSpec with shape (None, None, 4) and dtype tf.int32.
        """
        signature = super().input_signature
        # Add 'bbox' input with shape (None, None, 4) and dtype tf.int32
        signature["bbox"] = tf.TensorSpec(shape=(None, None, 4), dtype=tf.int32, name="bbox")
        return signature


LAYOUTLM_START_DOCSTRING = r"""
"""

LAYOUTLM_INPUTS_DOCSTRING = r"""
"""
# 添加起始文档字符串，描述这是一个输出原始隐藏状态的LayoutLM模型转换器，没有特定的顶部头
# 包含LayoutLM的起始文档字符串
class TFLayoutLMModel(TFLayoutLMPreTrainedModel):
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化LayoutLM的主要层，并命名为"layoutlm"
        self.layoutlm = TFLayoutLMMainLayer(config, name="layoutlm")

    # 解包输入
    # 向模型的前向传递添加起始文档字符串，描述输入的格式为"batch_size, sequence_length"
    # 替换返回文档字符串，指定输出类型为TFBaseModelOutputWithPoolingAndCrossAttentions，使用_CONFIG_FOR_DOC配置类
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        bbox: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        r"""
        Returns:
        此方法的返回类型为 TFBaseModelOutputWithPoolingAndCrossAttentions 或 Tuple[tf.Tensor]。

        Examples:
        示例代码演示如何使用该方法：
        
        ```python
        >>> from transformers import AutoTokenizer, TFLayoutLMModel
        >>> import tensorflow as tf

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = tf.convert_to_tensor([token_boxes])

        >>> outputs = model(
        ...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
        ... )

        >>> last_hidden_states = outputs.last_hidden_state
        ```
        执行示例代码，使用模型进行推理并获取最后隐藏状态的输出。

        """
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlm", None) is not None:
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
@add_start_docstrings("""LayoutLM Model with a `language modeling` head on top.""", LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForMaskedLM(TFLayoutLMPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"cls.seq_relationship",
        r"cls.predictions.decoder.weight",
        r"nsp___cls",
    ]

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # Warns about potential issues with bi-directional self-attention if config.is_decoder is True
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFLayoutLMForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # Initializes the main LayoutLM layer with optional pooling and names it "layoutlm"
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name="layoutlm")

        # Initializes the MLM (Masked Language Modeling) head for LayoutLM with input embeddings from layoutlm layer, names it "mlm___cls"
        self.mlm = TFLayoutLMMLMHead(config, input_embeddings=self.layoutlm.embeddings, name="mlm___cls")

    def get_lm_head(self) -> keras.layers.Layer:
        # Returns the predictions layer of the MLM head
        return self.mlm.predictions

    def get_prefix_bias_name(self) -> str:
        # Warns that this method is deprecated and suggests using 'get_bias' instead
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # Returns a concatenated string representing the name path of the MLM head's predictions layer
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        bbox: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 负责接收用于计算掩码语言建模损失的标签，形状为 `(batch_size, sequence_length)` 的张量或数组（可选）
        # 标签的索引应在 `[-100, 0, ..., config.vocab_size]` 范围内（参见 `input_ids` 文档字符串）
        # 索引为 `-100` 的标记会被忽略（掩码），损失仅计算具有 `[0, ..., config.vocab_size]` 范围内标签的标记

        Returns:

        # 返回结果说明部分，用于描述函数返回的内容及其含义

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFLayoutLMForMaskedLM
        >>> import tensorflow as tf

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = TFLayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "[MASK]"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = tf.convert_to_tensor([token_boxes])

        >>> labels = tokenizer("Hello world", return_tensors="tf")["input_ids"]

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=labels,
        ... )

        >>> loss = outputs.loss
        ```

        # 示例用法部分，展示了如何使用该函数进行模型推理和损失计算
        """
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 调用 layoutlm 模型进行前向传播，得到模型输出
        sequence_output = outputs[0]

        # 从模型输出中提取序列输出
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)

        # 如果提供了标签，则计算损失；否则损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果 return_dict 为 False，则组装输出并返回
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFMaskedLMOutput 对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义模型的构建方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型中存在 layoutlm 属性，则构建 layoutlm 模块
        if getattr(self, "layoutlm", None) is not None:
            # 使用 layoutlm 模块的名称作为命名空间
            with tf.name_scope(self.layoutlm.name):
                # 调用 layoutlm 模块的 build 方法，传入 None 作为输入形状
                self.layoutlm.build(None)
        # 如果模型中存在 mlm 属性，则构建 mlm 模块
        if getattr(self, "mlm", None) is not None:
            # 使用 mlm 模块的名称作为命名空间
            with tf.name_scope(self.mlm.name):
                # 调用 mlm 模块的 build 方法，传入 None 作为输入形状
                self.mlm.build(None)
"""
LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.
"""
# 使用 LayoutLM 模型进行序列分类或回归，顶部包含一个线性层（在池化输出之上），例如用于 GLUE 任务。
@add_start_docstrings(
    """
    LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForSequenceClassification(TFLayoutLMPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 以'.'结尾的名称表示在从 PT 模型加载 TF 模型时授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    # 缺失的名称以'.'结尾，表示在从 PT 模型加载 TF 模型时忽略的层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.layoutlm = TFLayoutLMMainLayer(config, name="layoutlm")  # 初始化 LayoutLM 主层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)  # Dropout 层
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )  # 分类器层
        self.config = config  # 配置信息

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 调用模型正向传播，对输入进行解包，并替换返回结果的文档字符串
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        bbox: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 正向传播函数的定义，接收多种输入参数和可选的训练标志
        pass  # 占位符，实际功能在后续实现中完成

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlm", None) is not None:
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)  # 构建 LayoutLM 主层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])  # 构建分类器层
    # 在加载模型时需要忽略的键列表，用于处理意外的键
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",               # 忽略名为"pooler"的键
        r"mlm___cls",            # 忽略名为"mlm___cls"的键
        r"nsp___cls",            # 忽略名为"nsp___cls"的键
        r"cls.predictions",      # 忽略名为"cls.predictions"的键
        r"cls.seq_relationship", # 忽略名为"cls.seq_relationship"的键
    ]
    # 在加载模型时需要忽略的键列表，用于处理缺失的键
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # LayoutLM 模型的初始化方法，继承自父类的初始化方法
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数目
        self.num_labels = config.num_labels

        # 初始化 LayoutLM 主层，包括一个可选的池化层
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name="layoutlm")
        
        # 设置 Dropout 层，根据配置中的隐藏层丢弃率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 分类器层，用于模型输出预测
        self.classifier = keras.layers.Dense(
            units=config.num_labels,  # 分类器单元数等于配置中的标签数目
            kernel_initializer=get_initializer(config.initializer_range),  # 使用配置中的初始化范围初始化权重
            name="classifier",  # 层的名称为"classifier"
        )
        
        # 保存配置对象
        self.config = config

    # 模型调用方法的装饰器，用于解压输入参数并添加文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        bbox: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFLayoutLMForTokenClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = TFLayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = tf.convert_to_tensor([token_boxes])
        >>> token_labels = tf.convert_to_tensor([1, 1, 0, 0])

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=token_labels,
        ... )

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        # 调用模型的前向传播方法，传入各种输入参数
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取模型输出的序列输出（通常是模型最后一层的输出）
        sequence_output = outputs[0]
        # 在训练模式下对序列输出应用 dropout
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将 dropout 后的输出送入分类器，得到最终的 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果提供了标签，则计算损失函数，否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果 return_dict 为 False，则返回输出的元组形式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则以 TFTokenClassifierOutput 对象形式返回结果
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型的方法，用于设置模型结构和参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果模型具有 layoutlm 属性且不为 None，则构建 layoutlm 组件
        if getattr(self, "layoutlm", None) is not None:
            # 在 TensorFlow 中为 layoutlm 组件创建命名空间
            with tf.name_scope(self.layoutlm.name):
                # 调用 layoutlm 组件的 build 方法，传入 None 作为输入形状
                self.layoutlm.build(None)
        
        # 如果模型具有 classifier 属性且不为 None，则构建 classifier 组件
        if getattr(self, "classifier", None) is not None:
            # 在 TensorFlow 中为 classifier 组件创建命名空间
            with tf.name_scope(self.classifier.name):
                # 调用 classifier 组件的 build 方法，传入包含 None、None 和 self.config.hidden_size 的列表作为输入形状
                self.classifier.build([None, None, self.config.hidden_size])
"""
LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
[DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span
start logits` and `span end logits`).
"""
# 使用 LayoutLM 模型，其顶部有一个用于抽取式问答任务的跨度分类头部，例如 [DocVQA](https://rrc.cvc.uab.es/?ch=17)。
# 这个头部是在最终隐藏状态输出之上的线性层，用于计算“跨度起始 logits” 和 “跨度终止 logits”。

class TFLayoutLMForQuestionAnswering(TFLayoutLMPreTrainedModel, TFQuestionAnsweringLoss):
    """
    LayoutLM 用于问答的 TensorFlow 模型，继承自 TFLayoutLMPreTrainedModel 和 TFQuestionAnsweringLoss。
    """
    
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    """
    在从 PyTorch 模型加载 TF 模型时，带有 '.' 的名称表示授权的意外/丢失的层。
    """

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        """
        初始化 LayoutLMForQuestionAnswering 模型。

        Args:
            config (LayoutLMConfig): LayoutLM 模型的配置对象。
            *inputs: 可变数量的输入。
            **kwargs: 其他关键字参数。
        """
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化 LayoutLM 主层
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name="layoutlm")
        
        # 初始化用于问答输出的全连接层
        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        bbox: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        """
        LayoutLM 问答模型的前向传播方法。

        Args:
            input_ids (TFModelInputType, optional): 输入的 token IDs.
            bbox (np.ndarray or tf.Tensor, optional): 边界框信息.
            attention_mask (np.ndarray or tf.Tensor, optional): 注意力掩码.
            token_type_ids (np.ndarray or tf.Tensor, optional): token 类型 IDs.
            position_ids (np.ndarray or tf.Tensor, optional): 位置 IDs.
            head_mask (np.ndarray or tf.Tensor, optional): 头部掩码.
            inputs_embeds (np.ndarray or tf.Tensor, optional): 嵌入的输入.
            output_attentions (bool, optional): 是否输出注意力权重.
            output_hidden_states (bool, optional): 是否输出隐藏状态.
            return_dict (bool, optional): 是否返回字典类型的输出.
            start_positions (np.ndarray or tf.Tensor, optional): 起始位置.
            end_positions (np.ndarray or tf.Tensor, optional): 结束位置.
            training (bool, optional): 是否处于训练模式.

        Returns:
            TFQuestionAnsweringModelOutput: LayoutLM 问答模型的输出对象。
        """
        # 省略了具体的前向传播逻辑，用文档字符串和装饰器指定了输入输出的详细描述
        pass

    def build(self, input_shape=None):
        """
        构建模型。

        Args:
            input_shape: 输入的形状信息，可选。

        Notes:
            如果已经构建过，则直接返回。
            构建 LayoutLM 和 qa_outputs 层。
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlm", None) is not None:
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```