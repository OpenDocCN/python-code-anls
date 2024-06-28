# `.\models\convbert\modeling_tf_convbert.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 ConvBERT model."""


from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFTokenClassificationLoss,
    get_initializer,
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
)
from .configuration_convbert import ConvBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "YituTech/conv-bert-base"
_CONFIG_FOR_DOC = "ConvBertConfig"

TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "YituTech/conv-bert-small",
    # See all ConvBERT models at https://huggingface.co/models?filter=convbert
]


# Copied from transformers.models.albert.modeling_tf_albert.TFAlbertEmbeddings with Albert->ConvBert
class TFConvBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ConvBertConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化 TFConvBertEmbeddings 类
        self.config = config
        # 获取嵌入大小
        self.embedding_size = config.embedding_size
        # 获取最大位置嵌入
        self.max_position_embeddings = config.max_position_embeddings
        # 获取初始化范围
        self.initializer_range = config.initializer_range
        # 使用配置的 epsilon 创建 LayerNorm 层，用于正则化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 在 build 方法中构建模型的嵌入层，用于词嵌入
    def build(self, input_shape=None):
        # 声明一个名为 "word_embeddings" 的命名空间，用于 TensorBoard 可视化
        with tf.name_scope("word_embeddings"):
            # 添加一个权重张量，表示词汇表中每个词的嵌入向量
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 声明一个名为 "token_type_embeddings" 的命名空间，用于 TensorBoard 可视化
        with tf.name_scope("token_type_embeddings"):
            # 添加一个权重张量，表示类型词汇表中每个类型的嵌入向量
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 声明一个名为 "position_embeddings" 的命名空间，用于 TensorBoard 可视化
        with tf.name_scope("position_embeddings"):
            # 添加一个权重张量，表示位置编码的嵌入向量
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True

        # 如果存在 LayerNorm 层，则在其命名空间内构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 根据输入的形状构建 LayerNorm 层，形状为 [None, None, self.config.embedding_size]
                self.LayerNorm.build([None, None, self.config.embedding_size])

    # 从 transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call 复制而来
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        根据输入张量应用嵌入。

        Returns:
            final_embeddings (`tf.Tensor`): 输出的嵌入张量。
        """
        # 如果没有提供 input_ids 和 inputs_embeds 中的任何一个，抛出 ValueError
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果提供了 input_ids，则从权重张量中根据索引获取对应的嵌入向量
        if input_ids is not None:
            # 检查 input_ids 是否在词汇表大小的范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 从权重参数 self.weight 中根据索引 input_ids 获取嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取 inputs_embeds 的形状列表，去掉最后一个维度（用于 batch 维度）
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果 token_type_ids 为 None，则填充为零向量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果 position_ids 为 None，则根据 past_key_values_length 和 input_shape 构建位置编码
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 根据 position_ids 从 self.position_embeddings 中获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 从 self.token_type_embeddings 中获取类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 最终的嵌入向量是 inputs_embeds、position_embeds 和 token_type_embeds 的和
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的嵌入向量应用 LayerNorm 层
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 在训练时，对最终的嵌入向量应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
    # 定义一个名为 TFConvBertSelfAttention 的自定义层，继承自 keras.layers.Layer
    class TFConvBertSelfAttention(keras.layers.Layer):
        # 初始化方法，接受配置 config 和其他关键字参数 kwargs
        def __init__(self, config, **kwargs):
            # 调用父类的初始化方法
            super().__init__(**kwargs)

            # 检查 hidden_size 是否能被 num_attention_heads 整除
            if config.hidden_size % config.num_attention_heads != 0:
                # 若不能整除，抛出 ValueError 异常
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({config.num_attention_heads})"
                )

            # 根据配置计算新的 num_attention_heads
            new_num_attention_heads = int(config.num_attention_heads / config.head_ratio)
            # 如果新的 num_attention_heads 小于 1，则使用默认的 config.num_attention_heads
            if new_num_attention_heads < 1:
                self.head_ratio = config.num_attention_heads
                num_attention_heads = 1
            else:
                num_attention_heads = new_num_attention_heads
                self.head_ratio = config.head_ratio

            # 将计算得到的 num_attention_heads 赋值给实例变量 self.num_attention_heads
            self.num_attention_heads = num_attention_heads
            # 将配置中的 conv_kernel_size 赋值给实例变量 self.conv_kernel_size
            self.conv_kernel_size = config.conv_kernel_size

            # 检查 hidden_size 是否能被 self.num_attention_heads 整除
            if config.hidden_size % self.num_attention_heads != 0:
                # 若不能整除，抛出 ValueError 异常
                raise ValueError("hidden_size should be divisible by num_attention_heads")

            # 计算每个 attention head 的大小
            self.attention_head_size = config.hidden_size // config.num_attention_heads
            # 计算所有 attention heads 总共的大小
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # 创建 Dense 层作为 query、key、value 的线性变换
            self.query = keras.layers.Dense(
                self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
            )
            self.key = keras.layers.Dense(
                self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
            )
            self.value = keras.layers.Dense(
                self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
            )

            # 创建 SeparableConv1D 层作为 key 的卷积注意力层
            self.key_conv_attn_layer = keras.layers.SeparableConv1D(
                self.all_head_size,
                self.conv_kernel_size,
                padding="same",
                activation=None,
                depthwise_initializer=get_initializer(1 / self.conv_kernel_size),
                pointwise_initializer=get_initializer(config.initializer_range),
                name="key_conv_attn_layer",
            )

            # 创建 Dense 层作为卷积核的线性变换层
            self.conv_kernel_layer = keras.layers.Dense(
                self.num_attention_heads * self.conv_kernel_size,
                activation=None,
                name="conv_kernel_layer",
                kernel_initializer=get_initializer(config.initializer_range),
            )

            # 创建 Dense 层作为卷积输出的线性变换层
            self.conv_out_layer = keras.layers.Dense(
                self.all_head_size,
                activation=None,
                name="conv_out_layer",
                kernel_initializer=get_initializer(config.initializer_range),
            )

            # 创建 Dropout 层，用于注意力概率的随机丢弃
            self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
            # 将配置对象保存在实例变量 self.config 中
            self.config = config

        # 定义 transpose_for_scores 方法，用于将输入 x 重塑为注意力分数的形状
        def transpose_for_scores(self, x, batch_size):
            # 将 x 从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
            x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
            # 调换维度顺序，变为 [batch_size, num_attention_heads, seq_length, attention_head_size]
            return tf.transpose(x, perm=[0, 2, 1, 3])
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在查询向量，构建查询向量的层，并指定其形状
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        
        # 如果存在键向量，构建键向量的层，并指定其形状
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        
        # 如果存在值向量，构建值向量的层，并指定其形状
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        
        # 如果存在键卷积注意力层，构建该层，并指定其形状
        if getattr(self, "key_conv_attn_layer", None) is not None:
            with tf.name_scope(self.key_conv_attn_layer.name):
                self.key_conv_attn_layer.build([None, None, self.config.hidden_size])
        
        # 如果存在卷积核层，构建该层，并指定其形状
        if getattr(self, "conv_kernel_layer", None) is not None:
            with tf.name_scope(self.conv_kernel_layer.name):
                self.conv_kernel_layer.build([None, None, self.all_head_size])
        
        # 如果存在卷积输出层，构建该层，并指定其形状
        if getattr(self, "conv_out_layer", None) is not None:
            with tf.name_scope(self.conv_out_layer.name):
                self.conv_out_layer.build([None, None, self.config.hidden_size])
# 定义 TFConvBertSelfOutput 类，继承自 keras.layers.Layer
class TFConvBertSelfOutput(keras.layers.Layer):
    # 初始化函数，接受 config 和 kwargs 参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出维度为 config.hidden_size，初始化方式为 config.initializer_range
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，epsilon 参数为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，dropout 率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 将 config 存储在 self.config 中
        self.config = config

    # 定义 call 方法，用于执行层的前向传播
    def call(self, hidden_states, input_tensor, training=False):
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # Dropout 操作，根据 training 参数决定是否执行
        hidden_states = self.dropout(hidden_states, training=training)
        # LayerNormalization 操作，加上输入张量 input_tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # 返回处理后的 hidden_states
        return hidden_states

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该层已构建
        self.built = True
        # 如果 self.dense 存在，则构建该全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 self.LayerNorm 存在，则构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 定义 TFConvBertAttention 类，继承自 keras.layers.Layer
class TFConvBertAttention(keras.layers.Layer):
    # 初始化函数，接受 config 和 kwargs 参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFConvBertSelfAttention 层
        self.self_attention = TFConvBertSelfAttention(config, name="self")
        # 创建 TFConvBertSelfOutput 层
        self.dense_output = TFConvBertSelfOutput(config, name="output")

    # 未实现的函数，用于裁剪注意力头部
    def prune_heads(self, heads):
        raise NotImplementedError

    # 定义 call 方法，用于执行层的前向传播
    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        # 调用 self_attention 层的 call 方法，计算 self-attention 输出
        self_outputs = self.self_attention(
            input_tensor, attention_mask, head_mask, output_attentions, training=training
        )
        # 调用 dense_output 层的 call 方法，计算最终输出
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        # 如果输出注意力信息，将其添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        # 返回输出结果
        return outputs

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该层已构建
        self.built = True
        # 如果 self.self_attention 存在，则构建 self_attention 层
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果 self.dense_output 存在，则构建 dense_output 层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 定义 GroupedLinearLayer 类，继承自 keras.layers.Layer
class GroupedLinearLayer(keras.layers.Layer):
    # 初始化函数，接受 input_size、output_size、num_groups、kernel_initializer 和 kwargs 参数
    def __init__(self, input_size, output_size, num_groups, kernel_initializer, **kwargs):
        super().__init__(**kwargs)
        # 初始化输入维度、输出维度、分组数、初始化方式
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.kernel_initializer = kernel_initializer
        # 计算每组的输入维度和输出维度
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
    # 在神经网络层的构建过程中被调用，用于初始化权重参数
    def build(self, input_shape=None):
        # 添加权重：kernel，用于存储多组卷积核的参数
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.group_out_dim, self.group_in_dim, self.num_groups],
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # 添加权重：bias，用于存储每个输出通道的偏置参数
        self.bias = self.add_weight(
            "bias", shape=[self.output_size], initializer=self.kernel_initializer, dtype=self.dtype, trainable=True
        )
        # 调用父类的 build 方法，完成神经网络层的构建
        super().build(input_shape)

    # 实现神经网络层的前向传播过程
    def call(self, hidden_states):
        # 获取输入张量的 batch size
        batch_size = shape_list(hidden_states)[0]
        # 将输入张量进行形状变换和转置，以便与卷积核进行批次乘积
        x = tf.transpose(tf.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim]), [1, 0, 2])
        # 执行批次乘积操作，计算卷积结果
        x = tf.matmul(x, tf.transpose(self.kernel, [2, 1, 0]))
        # 对卷积结果进行再次转置，使其恢复到原始张量的形状
        x = tf.transpose(x, [1, 0, 2])
        # 将卷积结果重新整形为最终输出的形状
        x = tf.reshape(x, [batch_size, -1, self.output_size])
        # 添加偏置项到卷积结果中
        x = tf.nn.bias_add(value=x, bias=self.bias)
        # 返回经过偏置处理后的最终输出张量
        return x
class TFConvBertIntermediate(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据配置选择使用单一组或多组线性层
        if config.num_groups == 1:
            self.dense = keras.layers.Dense(
                config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
        else:
            self.dense = GroupedLinearLayer(
                config.hidden_size,
                config.intermediate_size,
                num_groups=config.num_groups,
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )

        # 根据配置获取中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states):
        # 应用线性层
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 dense 属性，则构建对应的 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFConvBertOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 根据配置选择使用单一组或多组线性层
        if config.num_groups == 1:
            self.dense = keras.layers.Dense(
                config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
        else:
            self.dense = GroupedLinearLayer(
                config.intermediate_size,
                config.hidden_size,
                num_groups=config.num_groups,
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )
        
        # LayerNormalization 层
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states, input_tensor, training=False):
        # 应用线性层
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # LayerNormalization 层，添加残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 LayerNorm 属性，则构建对应的 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dense 属性，则构建对应的 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


class TFConvBertLayer(keras.layers.Layer):
    # 初始化方法，接受一个配置对象和可选的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 TFConvBertAttention 对象，使用给定的配置对象，并命名为 "attention"
        self.attention = TFConvBertAttention(config, name="attention")
        # 创建 TFConvBertIntermediate 对象，使用给定的配置对象，并命名为 "intermediate"
        self.intermediate = TFConvBertIntermediate(config, name="intermediate")
        # 创建 TFConvBertOutput 对象，使用给定的配置对象，并命名为 "output"
        self.bert_output = TFConvBertOutput(config, name="output")

    # 调用方法，接受隐藏状态、注意力掩码、头部掩码、是否输出注意力、是否训练等参数
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        # 调用 self.attention 对象的 call 方法，传递隐藏状态和其他参数，获取注意力输出
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, training=training
        )
        # 从 attention_outputs 中获取注意力输出的第一个元素
        attention_output = attention_outputs[0]
        # 使用 attention_output 调用 self.intermediate 对象的 call 方法，获取中间输出
        intermediate_output = self.intermediate(attention_output)
        # 使用 intermediate_output 和 attention_output 调用 self.bert_output 对象的 call 方法，获取层输出
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        # 构建输出元组，包括 layer_output 和可能的额外注意力输出
        outputs = (layer_output,) + attention_outputs[1:]  # 如果输出了注意力，将它们添加到输出中

        return outputs

    # 构建方法，接受输入形状参数（在这里未使用）
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果存在 self.attention 对象，则在其名称作用域下构建它
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在 self.intermediate 对象，则在其名称作用域下构建它
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 self.bert_output 对象，则在其名称作用域下构建它
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
# 定义 TFConvBertEncoder 类，继承自 keras.layers.Layer
class TFConvBertEncoder(keras.layers.Layer):
    # 初始化方法，接受 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 创建 TFConvBertLayer 的列表作为层的属性，每个层的名称包含索引号
        self.layer = [TFConvBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义调用方法，处理输入和各种参数，生成输出
    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None

        # 遍历每一层进行处理
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的 call 方法，生成当前层的输出
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
            )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中，如果需要输出隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回非空元组中的元素
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回 TFBaseModelOutput 类的实例，包含最后的隐藏状态、所有隐藏状态和注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 构建方法，用于构建层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self.layer 属性，则对每一层进行构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用 tf.name_scope 对每一层的名称进行命名空间管理
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义 TFConvBertPredictionHeadTransform 类，继承自 keras.layers.Layer
class TFConvBertPredictionHeadTransform(keras.layers.Layer):
    # 初始化方法，接受 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 创建全连接层 Dense，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 根据配置选择激活函数，并赋值给 transform_act_fn
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        
        # LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 存储配置
        self.config = config

    # 定义调用方法，处理输入的隐藏状态，通过全连接层和归一化层输出变换后的隐藏状态
    def call(self, hidden_states):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 归一化处理隐藏状态
        hidden_states = self.LayerNorm(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
    # 定义模型的构建方法，用于构建模型的各层结构，输入形状为可选参数
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不进行重复构建
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        
        # 如果存在名为dense的属性，并且不为None，则执行以下操作
        if getattr(self, "dense", None) is not None:
            # 使用dense层的名称作为命名空间，构建dense层，输入形状为[None, None, self.config.hidden_size]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果存在名为LayerNorm的属性，并且不为None，则执行以下操作
        if getattr(self, "LayerNorm", None) is not None:
            # 使用LayerNorm层的名称作为命名空间，构建LayerNorm层，输入形状为[None, None, self.config.hidden_size]
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 使用 keras_serializable 装饰器标记这个类，表示它可以被序列化为 Keras 模型
@keras_serializable
# 定义 TFConvBertMainLayer 类，继承自 keras.layers.Layer 类
class TFConvBertMainLayer(keras.layers.Layer):
    # 指定配置类为 ConvBertConfig
    config_class = ConvBertConfig

    # 初始化方法，接受 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 TFConvBertEmbeddings 实例，命名为 "embeddings"
        self.embeddings = TFConvBertEmbeddings(config, name="embeddings")

        # 如果嵌入大小不等于隐藏大小，则创建一个全连接层 embeddings_project
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = keras.layers.Dense(config.hidden_size, name="embeddings_project")

        # 创建 TFConvBertEncoder 实例，命名为 "encoder"
        self.encoder = TFConvBertEncoder(config, name="encoder")

        # 存储传入的配置对象
        self.config = config

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层的方法，设定权重和词汇表大小
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    # 未实现的方法，用于修剪模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 获取扩展的注意力遮罩的方法，根据输入形状和类型生成
    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # 将二维张量注意力遮罩转换为三维，以便进行广播
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # 将注意力遮罩转换为所需的格式，用于在 softmax 前过滤掉不需要的位置
        extended_attention_mask = tf.cast(extended_attention_mask, dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    # 获取头部遮罩的方法，如果存在头部遮罩则抛出未实现异常，否则返回与隐藏层数量相同的空列表
    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return head_mask

    # 使用 unpack_inputs 装饰器标记的 call 方法，处理模型的前向传播
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        ):
            # 如果同时指定了 input_ids 和 inputs_embeds，则抛出数值错误
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            # 如果指定了 input_ids，则获取其形状
            elif input_ids is not None:
                input_shape = shape_list(input_ids)
            # 如果指定了 inputs_embeds，则获取其形状，并去掉最后一个维度
            elif inputs_embeds is not None:
                input_shape = shape_list(inputs_embeds)[:-1]
            else:
                # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出数值错误
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            # 如果未指定 attention_mask，则创建一个全为1的张量，形状与 input_shape 相同
            if attention_mask is None:
                attention_mask = tf.fill(input_shape, 1)

            # 如果未指定 token_type_ids，则创建一个全为0的张量，形状与 input_shape 相同
            if token_type_ids is None:
                token_type_ids = tf.fill(input_shape, 0)

            # 使用 embeddings 方法生成隐藏状态张量
            hidden_states = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
            # 获取扩展后的 attention_mask
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, hidden_states.dtype)
            # 获取头部遮罩
            head_mask = self.get_head_mask(head_mask)

            # 如果模型具有 embeddings_project 属性，则使用它处理隐藏状态
            if hasattr(self, "embeddings_project"):
                hidden_states = self.embeddings_project(hidden_states, training=training)

            # 使用 encoder 处理隐藏状态，返回处理后的结果
            hidden_states = self.encoder(
                hidden_states,
                extended_attention_mask,
                head_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
                training=training,
            )

            # 返回处理后的隐藏状态作为最终输出
            return hidden_states

        # 构建模型结构的方法
        def build(self, input_shape=None):
            # 如果模型已经构建过，则直接返回
            if self.built:
                return
            # 标记模型已经构建
            self.built = True
            # 如果模型具有 embeddings 属性，则构建 embeddings 层
            if getattr(self, "embeddings", None) is not None:
                with tf.name_scope(self.embeddings.name):
                    self.embeddings.build(None)
            # 如果模型具有 encoder 属性，则构建 encoder 层
            if getattr(self, "encoder", None) is not None:
                with tf.name_scope(self.encoder.name):
                    self.encoder.build(None)
            # 如果模型具有 embeddings_project 属性，则构建 embeddings_project 层
            if getattr(self, "embeddings_project", None) is not None:
                with tf.name_scope(self.embeddings_project.name):
                    self.embeddings_project.build([None, None, self.config.embedding_size])
"""
An abstract class representing a ConvBERT model for TensorFlow, inheriting from `TFPreTrainedModel`.
Provides functionality for weights initialization, pretrained model handling, and a simple interface for downloading and loading pretrained models.
"""

# 设定配置类为 ConvBertConfig
config_class = ConvBertConfig

# 基础模型的前缀
base_model_prefix = "convbert"
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 调用父类的初始化方法，传递config及其他位置参数和关键字参数

        # 使用TFConvBertMainLayer类初始化一个名为convbert的成员变量
        self.convbert = TFConvBertMainLayer(config, name="convbert")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用convbert对象的call方法，传递各种输入参数
        outputs = self.convbert(
            input_ids=input_ids,
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
        # 返回convbert的输出结果

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果self.convbert存在，则在相应的命名空间下构建convbert对象
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
# 基于 Keras 的自定义层，用于 ConvBERT 模型中的 Masked Language Modeling 头部
class TFConvBertMaskedLMHead(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 保存模型配置信息
        self.embedding_size = config.embedding_size  # 从配置中获取嵌入向量的大小
        self.input_embeddings = input_embeddings  # 输入嵌入层的权重

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        # 创建偏置项权重，形状为词汇表大小，并初始化为零，可训练

        super().build(input_shape)

    def get_output_embeddings(self):
        return self.input_embeddings  # 返回输入嵌入层的权重

    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value  # 设置输入嵌入层的权重为给定值
        self.input_embeddings.vocab_size = shape_list(value)[0]  # 更新词汇表大小

    def get_bias(self):
        return {"bias": self.bias}  # 返回偏置项权重

    def set_bias(self, value):
        self.bias = value["bias"]  # 设置偏置项权重为给定值
        self.config.vocab_size = shape_list(value["bias"])[0]  # 更新配置中的词汇表大小

    def call(self, hidden_states):
        seq_length = shape_list(tensor=hidden_states)[1]  # 获取隐藏状态的序列长度
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])  # 重塑隐藏状态
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 执行矩阵乘法，将嵌入层权重与隐藏状态相乘（转置后）
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 重塑输出形状以匹配模型输出要求
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        # 添加偏置项到输出隐藏状态

        return hidden_states


# ConvBERT 模型中用于生成预测的自定义 Keras 层
class TFConvBertGeneratorPredictions(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # LayerNormalization 层，使用给定的 epsilon 参数
        self.dense = keras.layers.Dense(config.embedding_size, name="dense")
        # 全连接层，输出大小为配置中的嵌入大小
        self.config = config  # 保存模型配置信息

    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)  # 执行全连接操作
        hidden_states = get_tf_activation("gelu")(hidden_states)  # 使用 GELU 激活函数
        hidden_states = self.LayerNorm(hidden_states)  # 应用 LayerNormalization

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
                # 如果存在 LayerNorm 层，则构建其图层结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
                # 如果存在 dense 层，则构建其图层结构


@add_start_docstrings("""ConvBERT Model with a `language modeling` head on top.""", CONVBERT_START_DOCSTRING)
# 使用装饰器添加文档字符串说明的 ConvBERT 模型，带有语言建模头部
class TFConvBertForMaskedLM(TFConvBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 继承 TFConvBertPreTrainedModel 和 TFMaskedLanguageModelingLoss
    # 初始化方法，接受配置参数、多个输入和关键字参数，调用父类的初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, **kwargs)

        # 将配置参数保存到实例变量中
        self.config = config
        # 创建一个 TFConvBertMainLayer 对象，并命名为 convbert
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 创建一个 TFConvBertGeneratorPredictions 对象，并命名为 generator_predictions
        self.generator_predictions = TFConvBertGeneratorPredictions(config, name="generator_predictions")

        # 检查 hidden_act 是否为字符串类型，如果是，则通过 get_tf_activation 获取对应的激活函数，否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        # 创建一个 TFConvBertMaskedLMHead 对象，依赖于 convbert.embeddings，并命名为 generator_lm_head
        self.generator_lm_head = TFConvBertMaskedLMHead(config, self.convbert.embeddings, name="generator_lm_head")

    # 返回 generator_lm_head 实例
    def get_lm_head(self):
        return self.generator_lm_head

    # 返回由实例名称和 generator_lm_head 名称组成的字符串，用于前缀偏置名称
    def get_prefix_bias_name(self):
        return self.name + "/" + self.generator_lm_head.name

    # 使用装饰器将下列函数声明为模型的前向传播函数，并添加相应的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播函数，接受多个输入参数，并返回 TFMaskedLMOutput 类型的输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
        ) -> Union[Tuple, TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 使用类型注解指定函数返回类型为元组或 TFMaskedLMOutput 类型
        generator_hidden_states = self.convbert(
            input_ids=input_ids,
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
        # 从 convbert 模型返回的隐藏状态中获取生成器的序列输出
        generator_sequence_output = generator_hidden_states[0]
        # 使用生成器预测模型对生成器序列输出进行预测
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        # 使用生成器语言模型头部对预测分数进行进一步处理
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        # 如果提供了标签，计算生成器模型的损失；否则损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果 return_dict 为 False，则按顺序返回损失和生成器的隐藏状态
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构造 TFMaskedLMOutput 对象返回
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果 convbert 模型存在，则构建 convbert 模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果 generator_predictions 模型存在，则构建 generator_predictions 模型
        if getattr(self, "generator_predictions", None) is not None:
            with tf.name_scope(self.generator_predictions.name):
                self.generator_predictions.build(None)
        # 如果 generator_lm_head 模型存在，则构建 generator_lm_head 模型
        if getattr(self, "generator_lm_head", None) is not None:
            with tf.name_scope(self.generator_lm_head.name):
                self.generator_lm_head.build(None)
class TFConvBertClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，输出维度为 config.hidden_size，使用指定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 根据 config 配置选择分类器的 dropout 率，如果未指定，则使用隐藏层 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 Dropout 层，应用于全连接层的输出
        self.dropout = keras.layers.Dropout(classifier_dropout)
        
        # 定义一个全连接层，输出维度为 config.num_labels，使用指定的初始化器初始化权重
        self.out_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

        self.config = config

    def call(self, hidden_states, **kwargs):
        # 获取每个样本的第一个 token 的隐藏状态（通常是 [CLS] 标志）
        x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)  # 对隐藏状态应用 Dropout
        x = self.dense(x)  # 将 Dropout 后的隐藏状态输入全连接层
        x = get_tf_activation(self.config.hidden_act)(x)  # 应用激活函数到全连接层的输出
        x = self.dropout(x)  # 对激活函数的输出再次应用 Dropout
        x = self.out_proj(x)  # 将 Dropout 后的输出输入到输出全连接层

        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果 dense 层已经定义，则根据输入形状构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果 out_proj 层已经定义，则根据输入形状构建 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 ConvBert 模型进行前向传播，获取输出结果
        outputs = self.convbert(
            input_ids,
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
        # 将 ConvBert 的输出 logits 传递给分类器，得到分类器的预测结果
        logits = self.classifier(outputs[0], training=training)
        # 如果提供了标签，计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典形式的输出
        if not return_dict:
            # 组装输出结果，包括 logits 和 ConvBert 的其它输出
            output = (logits,) + outputs[1:]
            # 返回包含损失和输出结果的元组，如果损失为 None 则不包含损失
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的 TFSequenceClassifierOutput
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 ConvBert 模型存在，则构建 ConvBert 模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 ConvBERT 主层，使用给定的配置和名称"convbert"
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        
        # 创建用于序列汇总的 TFSequenceSummary 实例，使用配置中的初始化范围和名称"sequence_summary"
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        
        # 创建用于分类的全连接层 Dense，输出维度为1，使用给定的初始化器范围和名称"classifier"
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # 将配置保存到实例中
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果存在 input_ids，则确定 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，使用 inputs_embeds 确定 num_choices 和 seq_length
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 摊平成形状为 (-1, seq_length) 的张量，如果 input_ids 不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 摊平成形状为 (-1, seq_length) 的张量，如果 attention_mask 不为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将 token_type_ids 摊平成形状为 (-1, seq_length) 的张量，如果 token_type_ids 不为 None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        # 将 position_ids 摊平成形状为 (-1, seq_length) 的张量，如果 position_ids 不为 None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        # 将 inputs_embeds 摊平成形状为 (-1, seq_length, hidden_size) 的张量，如果 inputs_embeds 不为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 调用 convbert 模型进行前向传播
        outputs = self.convbert(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 对输出进行序列摘要
        logits = self.sequence_summary(outputs[0], training=training)
        # 对序列摘要后的结果进行分类
        logits = self.classifier(logits)
        # 将 logits 重新整形为 (-1, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        # 如果存在 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不返回字典格式的输出，则组合输出结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TF 模型多选模型的输出对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经建立，则直接返回
        if self.built:
            return
        # 标记模型已经建立
        self.built = True
        # 如果 convbert 模型存在，则建立 convbert 模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果 sequence_summary 模型存在，则建立 sequence_summary 模型
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果 classifier 模型存在，则建立 classifier 模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
@add_start_docstrings(
    """
    ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForTokenClassification(TFConvBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化分类任务的标签数
        self.num_labels = config.num_labels
        # 创建 ConvBERT 主层，命名为 "convbert"
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 定义分类器的 dropout 层，使用 config 中指定的 dropout 或者默认的隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 定义分类器的全连接层，输出维度为 config 中指定的标签数，使用指定的初始化方法
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    # 将输入解包，并添加模型前向传播的文档注释
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 ConvBERT 模型，传入各种输入参数
        outputs = self.convbert(
            input_ids,
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
        # 从 ConvBERT 模型输出中取得序列输出
        sequence_output = outputs[0]
        # 对序列输出应用 dropout，用于防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 dropout 后的输出送入分类器，得到预测 logits
        logits = self.classifier(sequence_output)
        # 如果有提供标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典，则返回 tuple 类型的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则构建 TFTokenClassifierOutput 对象并返回
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 ConvBERT 模型存在，则构建它
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果分类器存在，则构建它，并指定输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器为 TFConvBertForQuestionAnswering 类添加文档字符串，描述其功能和适用于的任务类型
@add_start_docstrings(
    """
    ConvBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CONVBERT_START_DOCSTRING,  # 引用之前定义的 ConvBERT 的文档字符串常量
)
class TFConvBertForQuestionAnswering(TFConvBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置模型需要的标签数
        self.num_labels = config.num_labels
        # 初始化 ConvBERT 主层，命名为 "convbert"
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        # 创建用于回答问题的输出层，包括初始化和命名
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存模型配置
        self.config = config

    # 使用装饰器为 call 方法添加文档字符串，描述输入参数和模型输出的样例和用途
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 引用用于示例的检查点
        output_type=TFQuestionAnsweringModelOutput,  # 输出类型为 TFQuestionAnsweringModelOutput
        config_class=_CONFIG_FOR_DOC,  # 引用模型的配置类
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: tf.Tensor | None = None,
        end_positions: tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 下面的参数包括了模型可能用到的所有输入和控制参数
    ) -> Union[Tuple, TFQuestionAnsweringModelOutput]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用 ConvBert 模型进行推理，获取模型的输出
        outputs = self.convbert(
            input_ids,
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
        # 从模型输出中提取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给 QA 输出层，得到预测的起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 沿着最后一个维度分割成起始位置和结束位置的 logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除 logits 的最后一个维度中的大小为 1 的维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        # 初始化损失为 None
        loss = None

        # 如果给定了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            # 组装标签字典，用于计算损失
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 调用 Hugging Face 的损失计算函数计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不要求返回字典，则组装输出元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则将输出封装成 TFQuestionAnsweringModelOutput 对象返回
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 ConvBert 模型存在，则构建 ConvBert 模型
        if getattr(self, "convbert", None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        # 如果 QA 输出层存在，则构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```