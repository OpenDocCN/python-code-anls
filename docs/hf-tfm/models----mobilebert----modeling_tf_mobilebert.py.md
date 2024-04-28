# `.\transformers\models\mobilebert\modeling_tf_mobilebert.py`

```
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" TF 2.0 MobileBERT model."""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 从 transformers.activations_tf 模块中导入 get_tf_activation 函数
from ...activations_tf import get_tf_activation
# 从 transformers.modeling_tf_outputs 模块中导入各种输出类型
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
# 从 transformers.modeling_tf_utils 模块中导入各种工具函数和类
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
# 从 transformers.tf_utils 模块中导入一些工具函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 从 transformers.utils 模块中导入各种工具函数和类
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 从 transformers.models.mobilebert.configuration_mobilebert 模块中导入 MobileBertConfig
from .configuration_mobilebert import MobileBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/mobilebert-uncased"
_CONFIG_FOR_DOC = "MobileBertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "vumichien/mobilebert-finetuned-ner"
_TOKEN_CLASS_EXPECTED_OUTPUT = "['I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']"
_TOKEN_CLASS_EXPECTED_LOSS = 0.03

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "vumichien/mobilebert-uncased-squad-v2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 3.98
_QA_TARGET_START_INDEX = 12
_QA_TARGET_END_INDEX = 13

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "vumichien/emo-mobilebert"
_SEQ_CLASS_EXPECTED_OUTPUT = "'others'"
_SEQ_CLASS_EXPECTED_LOSS = "4.72"

TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilebert-uncased",
    # See all MobileBERT models at https://huggingface.co/models?filter=mobilebert
]


# 这个类是从 transformers.models.bert.modeling_tf_bert.TFBertPreTrainingLoss 复制过来的
# 它定义了一个 MobileBERT 预训练任务的损失函数
class TFMobileBertPreTrainingLoss:
    """
    这个类定义了一个 MobileBERT 预训练任务的损失函数。
    它基于 BERT 的预训练方式,包括 Masked Language Modeling 和 Next Sentence Prediction 两种任务。
    这个类负责计算在这两种任务上的损失函数值。
    """
    pass
    # 创建适合BERT-like预训练的损失函数，即组合NSP（下一句预测）和MLM（遮蔽语言模型）的预训练任务。
    # 注意：-100的任何标签都会在损失计算中被忽略（以及对应的logits）。
    """
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        # 创建一个带有逻辑回归输出的稀疏分类交叉熵损失函数
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
    
        # 将负标签剪裁为零以避免NaN和错误-这些位置在后续的遮蔽中仍然会被遮蔽
        # 使用损失函数计算未遮蔽的语言模型损失
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # 确保只有不等于-100的标签才会被用于损失计算
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        # 使用遮蔽后的标签和损失计算得到遮蔽的语言模型损失
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        # 计算遮蔽后的语言模型损失的平均值
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)
    
        # 将负标签剪裁为零以避免NaN和错误-这些位置在后续的遮蔽中仍然会被遮蔽
        # 使用损失函数计算未遮蔽的下一句预测损失
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels["next_sentence_label"]), y_pred=logits[1])
        # 确保只有不等于-100的标签才会被用于损失计算
        ns_loss_mask = tf.cast(labels["next_sentence_label"] != -100, dtype=unmasked_ns_loss.dtype)
        # 使用遮蔽后的标签和损失计算得到遮蔽的下一句预测损失
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask
        # 计算遮蔽后的下一句预测损失的平均值
        reduced_masked_ns_loss = tf.reduce_sum(masked_ns_loss) / tf.reduce_sum(ns_loss_mask)
    
        # 返回减少的遮蔽语言模型损失和遮蔽下一句预测损失的和
        return tf.reshape(reduced_masked_lm_loss + reduced_masked_ns_loss, (1,))
class TFMobileBertIntermediate(tf.keras.layers.Layer):
    # 初始化中间层，接收配置参数和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建全连接层，输出维度为中间层大小
        self.dense = tf.keras.layers.Dense(config.intermediate_size, name="dense")

        # 检查隐藏层激活函数是否为字符串，如果是，则获取对应的 TensorFlow 激活函数；否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 定义调用方法，对输入的隐藏状态进行中间层处理
    def call(self, hidden_states):
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用中间层激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层，用于配置模型的内部参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，输入维度为 [None, None, true_hidden_size]
                self.dense.build([None, None, self.config.true_hidden_size])


class TFLayerNorm(tf.keras.layers.LayerNormalization):
    # 初始化 LayerNorm 层，接收特征大小以及其他参数
    def __init__(self, feat_size, *args, **kwargs):
        self.feat_size = feat_size
        super().__init__(*args, **kwargs)

    # 构建层，用于配置模型的内部参数
    def build(self, input_shape=None):
        super().build([None, None, self.feat_size])


class TFNoNorm(tf.keras.layers.Layer):
    # 初始化不使用标准化的层，接收特征大小和其他关键字参数
    def __init__(self, feat_size, epsilon=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_size = feat_size

    # 构建层，用于配置模型的内部参数
    def build(self, input_shape):
        # 添加偏置参数
        self.bias = self.add_weight("bias", shape=[self.feat_size], initializer="zeros")
        # 添加权重参数
        self.weight = self.add_weight("weight", shape=[self.feat_size], initializer="ones")
        super().build(input_shape)

    # 定义调用方法，对输入进行处理，不使用标准化，只进行偏置和缩放
    def call(self, inputs: tf.Tensor):
        return inputs * self.weight + self.bias


# 定义标准化方法的字典，根据配置类型选择对应的标准化方法
NORM2FN = {"layer_norm": TFLayerNorm, "no_norm": TFNoNorm}


class TFMobileBertEmbeddings(tf.keras.layers.Layer):
    # 构建嵌入层，用于构建来自单词、位置和令牌类型的嵌入
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 是否使用三元组输入
        self.trigram_input = config.trigram_input
        # 嵌入大小
        self.embedding_size = config.embedding_size
        self.config = config
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 最大位置嵌入
        self.max_position_embeddings = config.max_position_embeddings
        # 初始化范围
        self.initializer_range = config.initializer_range
        # 嵌入转换层，将输入转换为隐藏层大小
        self.embedding_transformation = tf.keras.layers.Dense(config.hidden_size, name="embedding_transformation")

        # LayerNorm 层，用于标准化嵌入层输出
        # self.LayerNorm 名称没有使用蛇形命名以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 丢弃层，用于减少过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 嵌入输入大小，如果使用三元组输入，则嵌入大小乘以3，否则等于嵌入大小
        self.embedded_input_size = self.embedding_size * (3 if self.trigram_input else 1)
    # 构建词嵌入层
    def build(self, input_shape=None):
        # 在 TensorFlow 图中创建一个命名作用域，用于包裹词嵌入操作
        with tf.name_scope("word_embeddings"):
            # 添加权重张量用于词嵌入，初始化为指定形状和初始化器的随机值
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 在 TensorFlow 图中创建一个命名作用域，用于包裹标记类型嵌入操作
        with tf.name_scope("token_type_embeddings"):
            # 添加标记类型嵌入的权重张量，初始化为指定形状和初始化器的随机值
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 在 TensorFlow 图中创建一个命名作用域，用于包裹位置嵌入操作
        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入的权重张量，初始化为指定形状和初始化器的随机值
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 如果层已经构建完毕，直接返回
        if self.built:
            return
        # 设置标志位，表示层已构建
        self.built = True
        # 如果存在嵌入变换层，则构建该层
        if getattr(self, "embedding_transformation", None) is not None:
            # 在 TensorFlow 图中创建一个命名作用域，用于包裹嵌入变换层的构建操作
            with tf.name_scope(self.embedding_transformation.name):
                # 构建嵌入变换层，输入形状为 [None, None, self.embedded_input_size]
                self.embedding_transformation.build([None, None, self.embedded_input_size])
        # 如果存在 LayerNorm 层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            # 在 TensorFlow 图中创建一个命名作用域，用于包裹 LayerNorm 层的构建操作
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNorm 层，输入形状为 None
                self.LayerNorm.build(None)
    def call(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言确保输入不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果输入的是 token IDs，则从 embedding 表中获取对应的 embeddings
        if input_ids is not None:
            # 检查 token IDs 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 从 embedding 表中根据 token IDs 索引获取 embeddings
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入 embeddings 的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果 token_type_ids 为空，则将其填充为 0
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果采用三元输入模式
        if self.trigram_input:
            # 从论文 MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices 中引用
            #
            # BERT 模型中的 embedding 表占据了模型大小的相当大一部分。为了压缩 embedding 层，我们将 embedding 维度
            # 缩减为 128 在 MobileBERT 中，然后，我们在原始 token embedding 上应用了一个核大小为 3 的 1D 卷积，
            # 以产生一个 512 维的输出。
            inputs_embeds = tf.concat(
                [
                    tf.pad(inputs_embeds[:, 1:], ((0, 0), (0, 1), (0, 0))),
                    inputs_embeds,
                    tf.pad(inputs_embeds[:, :-1], ((0, 0), (1, 0), (0, 0))),
                ],
                axis=2,
            )

        # 如果采用三元输入模式或者 embedding_size 不等于 hidden_size
        if self.trigram_input or self.embedding_size != self.hidden_size:
            # 对输入 embeddings 进行变换
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # 如果 position_ids 为空，则创建默认的 position_ids
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 从 position_embeddings 表中获取 position embeddings
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 从 token_type_embeddings 表中获取 token type embeddings
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 计算最终的 embeddings
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 应用 Layer Normalization
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的 embeddings
        return final_embeddings
class TFMobileBertSelfAttention(tf.keras.layers.Layer):
    # 创建 TFMobileBertSelfAttention 类，继承自 tf.keras.layers.Layer
    def __init__(self, config, **kwargs):
        # 初始化函数，接受配置参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        if config.hidden_size % config.num_attention_heads != 0:
            # 检查隐藏层大小是否为注意力头数的倍数，否则引发数值错误
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        # 设置注意力头数
        self.output_attentions = config.output_attentions
        # 设置是否输出注意力结果
        assert config.hidden_size % config.num_attention_heads == 0
        # 断言隐藏层大小与注意力头数之比为0
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        # 计算注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算总的头大小

        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        # query 层
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        # key 层
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # value 层

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        # dropout 层
        self.config = config
        # 设置 config 参数

    def transpose_for_scores(self, x, batch_size):
        # 为注意力重新排列张量的形状，从 [batch_size, seq_length, all_head_size] 转换成 [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        # 返回转置后的张量

    def call(
        self, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=False
        # call 方法，接受 query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions 和 training 参数
    # 获取注意力掩码的批处理大小
    ):
        # 通过 self.query 方法对查询张量进行处理
        mixed_query_layer = self.query(query_tensor)
        # 通过 self.key 方法对键张量进行处理
        mixed_key_layer = self.key(key_tensor)
        # 通过 self.value 方法对值张量进行处理
        mixed_value_layer = self.value(value_tensor)
        # 调整维度以便用于计算注意力得分
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算“查询”和“键”的点积，得到原始的注意力分数
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        # 计算缩放因子
        dk = tf.cast(shape_list(key_layer)[-1], dtype=attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # 对注意力掩码进行类型转换并加到注意力得分中
            attention_mask = tf.cast(attention_mask, dtype=attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        # 将注意力得分归一化为概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 通过 dropout 方法对注意力概率进行处理
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果存在头掩码，将其应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，即注意力概率和值的乘积
        context_layer = tf.matmul(attention_probs, value_layer)

        # 调整上下文张量的维度
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        # 返回输出，包括上下文张量和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    # 构建方法，用于构建自注意力层
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 检查是否存在查询、键、值方法，如果存在则构建它们的形状
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.true_hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.true_hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build(
                    [
                        None,
                        None,
                        self.config.true_hidden_size
                        if self.config.use_bottleneck_attention
                        else self.config.hidden_size,
                    ]
                )
# 定义 TFMobileBertSelfOutput 类，用于 MobileBERT 模型的自注意力输出部分
class TFMobileBertSelfOutput(tf.keras.layers.Layer):
    # 初始化方法，接受配置信息和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 是否使用瓶颈层的标志
        self.use_bottleneck = config.use_bottleneck
        # 创建一个全连接层，用于将隐藏状态映射到指定大小的输出空间
        self.dense = tf.keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNorm 层，用于对输入进行归一化处理
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 如果不使用瓶颈层，则创建一个 Dropout 层
        if not self.use_bottleneck:
            self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    # 调用方法，实现自注意力输出层的前向传播逻辑
    def call(self, hidden_states, residual_tensor, training=False):
        # 将隐藏状态经过全连接层映射到指定大小的输出空间
        hidden_states = self.dense(hidden_states)
        # 如果不使用瓶颈层，则对输出进行 Dropout 处理
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
        # 将输出与残差张量相加后通过 LayerNorm 层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        # 返回处理后的输出
        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        # 构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


# 定义 TFMobileBertAttention 类，用于 MobileBERT 模型的注意力层
class TFMobileBertAttention(tf.keras.layers.Layer):
    # 初始化方法，接受配置信息和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建自注意力层对象
        self.self = TFMobileBertSelfAttention(config, name="self")
        # 创建自注意力输出层对象
        self.mobilebert_output = TFMobileBertSelfOutput(config, name="output")

    # 剪枝头部的方法，暂未实现
    def prune_heads(self, heads):
        # 抛出未实现异常
        raise NotImplementedError

    # 调用方法，实现注意力层的前向传播逻辑
    def call(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        layer_input,
        attention_mask,
        head_mask,
        output_attentions,
        training=False,
    ):
        # 调用自注意力层进行计算
        self_outputs = self.self(
            query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=training
        )
        # 将自注意力层的输出作为输入，经过自注意力输出层进行计算
        attention_output = self.mobilebert_output(self_outputs[0], layer_input, training=training)
        # 将输出打包成元组，如果需要输出注意力权重，则添加到元组中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回计算结果
        return outputs

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 构建自注意力层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 构建自注意力输出层
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)


# 定义 TFOutputBottleneck 类
class TFOutputBottleneck(tf.keras.layers.Layer):
``` 
    # 初始化函数，接受config和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层，节点数为config.hidden_size，命名为"dense"
        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        # 创建一个LayerNorm层，节点数为config.hidden_size，epsilon为config.layer_norm_eps，命名为"LayerNorm"
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 创建一个dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 将config保存在对象的config属性中
        self.config = config

    # 对象的调用函数，接受hidden_states、residual_tensor和training参数
    def call(self, hidden_states, residual_tensor, training=False):
        # 经过全连接层处理
        layer_outputs = self.dense(hidden_states)
        # 经过dropout处理
        layer_outputs = self.dropout(layer_outputs, training=training)
        # 经过LayerNorm处理并与residual_tensor相加，返回处理后的结果
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs

    # 构建函数，接受input_shape参数
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 设置已构建标志为True
        self.built = True
        # 如果存在dense属性，则构建dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        # 如果存在LayerNorm属性，则构建LayerNorm层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)
# 创建一个TFMobileBertOutput类，继承自tf.keras.layers.Layer类
class TFMobileBertOutput(tf.keras.layers.Layer):
    # 初始化函数，接收config和**kwargs参数
    def __init__(self, config, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 根据配置参数设置是否使用瓶颈层
        self.use_bottleneck = config.use_bottleneck
        # 创建一个全连接层，输出维度为config.true_hidden_size，使用config.initializer_range来初始化权重矩阵，名称为"dense"
        self.dense = tf.keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNorm层，输入维度为config.true_hidden_size，使用config.layer_norm_eps作为epsilon，名称为"LayerNorm"
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 如果不使用瓶颈层，在这里创建一个dropout层，使用config.hidden_dropout_prob作为dropout率
        if not self.use_bottleneck:
            self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 否则，在这里创建一个TFOutputBottleneck瓶颈层，接收config参数，名称为"bottleneck"
        else:
            self.bottleneck = TFOutputBottleneck(config, name="bottleneck")
        # 保存config参数
        self.config = config

    # 前向传播函数，接收hidden_states, residual_tensor_1, residual_tensor_2和training参数
    def call(self, hidden_states, residual_tensor_1, residual_tensor_2, training=False):
        # 首先通过全连接层对hidden_states进行变换
        hidden_states = self.dense(hidden_states)
        # 如果不使用瓶颈层，在这里对hidden_states进行dropout操作
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
            # 将dropout后的hidden_states与residual_tensor_1相加，并经过LayerNorm层
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
        # 否则，将hidden_states与residual_tensor_1相加，并通过LayerNorm层
        else:
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
            # 再将输出与residual_tensor_2一起输入到瓶颈层中
            hidden_states = self.bottleneck(hidden_states, residual_tensor_2)
        # 返回hidden_states
        return hidden_states

    # 构建函数，用于构建层中的子层
    def build(self, input_shape=None):
        # 如果已经构建了层，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在dense层，则调用该层的build函数来构建该层的子层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在LayerNorm层，则调用该层的build函数来构建该层的子层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)
        # 如果存在瓶颈层，则调用该层的build函数来 构建该层的子层
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)


class TFBottleneckLayer(tf.keras.layers.Layer):
    # 初始化函数，接收config和**kwargs参数
    def __init__(self, config, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.intra_bottleneck_size，名称为"dense"
        self.dense = tf.keras.layers.Dense(config.intra_bottleneck_size, name="dense")
        # 创建一个LayerNorm层，输入维度为config.intra_bottleneck_size，使用config.layer_norm_eps作为epsilon，名称为"LayerNorm"
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 保存config参数
        self.config = config

    # 前向传播函数，接收inputs作为输入
    def call(self, inputs):
        # 输入经过全连接层变换
        hidden_states = self.dense(inputs)
        # 经过LayerNorm层
        hidden_states = self.LayerNorm(hidden_states)
        # 返回hidden_states
        return hidden_states

    # 构建函数，用于构建层中的子层
    def build(self, input_shape=None):
        # 如果已经构建了层，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在dense层，则调用该层的build函数来构建该层的子层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在LayerNorm层，则调用该层的build函数来构建该层的子层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFBottleneck(tf.keras.layers.Layer):
    ...


注释：
    # 初始化方法，用于创建一个新的实例对象。它接受一个config参数和可选的关键字参数kwargs。
    # 调用父类的初始化方法，用于初始化继承的属性。
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 从config中获取是否使用共享的瓶颈层和是否使用瓶颈注意力的标志，并将其保存到当前实例对象中
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        # 创建一个TFBottleneckLayer实例对象，用于输入瓶颈层
        self.bottleneck_input = TFBottleneckLayer(config, name="input")
        # 如果使用共享的瓶颈层，则创建一个用于注意力的瓶颈层
        if self.key_query_shared_bottleneck:
            self.attention = TFBottleneckLayer(config, name="attention")

    # 调用方法，用于执行实际的层逻辑，传入hidden_states作为输入
    def call(self, hidden_states):
        # 该方法可以返回三种不同的值组合。这些不同的值利用了瓶颈层，这些瓶颈层是用于将隐藏状态投影到低维向量的线性层，
        # 从而降低内存使用量。这些线性层具有在训练过程中学习的权重。
        #
        # 如果config.use_bottleneck_attention为True，它将返回瓶颈层的结果四次，分别用于键、查询、值和“层输入”，
        # 以供注意力层使用。这个瓶颈层用于投影隐藏状态。最后一个“层输入”将在计算注意力分数后用作注意力自我输出中的残差张量。
        #
        # 如果不是config.use_bottleneck_attention且config.key_query_shared_bottleneck为True，这将返回四个值，
        # 其中三个已通过瓶颈层传递：通过相同瓶颈层传递的查询和键，以及要在注意力自我输出中应用的残差层，通过另一个瓶颈层。
        #
        # 最后，在最后一种情况下，查询、键和值的值是没有经过瓶颈层的隐藏状态，而残余层将是通过瓶颈层传递的此值。
        
        # 对隐藏状态进行瓶颈化处理
        bottlenecked_hidden_states = self.bottleneck_input(hidden_states)
        # 如果使用瓶颈注意力，则返回四次瓶颈化处理后的隐藏状态
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        # 如果不使用瓶颈注意力但使用共享的瓶颈层，则返回四个值：共享注意力输入、共享注意力输入、隐藏状态和瓶颈化处理后的隐藏状态
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        # 如果既不使用瓶颈注意力也不使用共享的瓶颈层，则返回四个值：隐藏状态、隐藏状态、隐藏状态和瓶颈化处理后的隐藏状态
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为True
        self.built = True
        # 如果存在bottleneck_input属性，则构建瓶颈输入层
        if getattr(self, "bottleneck_input", None) is not None:
            with tf.name_scope(self.bottleneck_input.name):
                self.bottleneck_input.build(None)
        # 如果存在attention属性，则构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
# TFFFNOutput 是一个 Keras 层，负责处理前馈神经网络的输出
class TFFFNOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 创建一个全连接层，用于计算隐藏状态的输出
        self.dense = tf.keras.layers.Dense(config.true_hidden_size, name="dense")
        # 创建一个层归一化层，用于对输出进行归一化处理
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 保存配置信息
        self.config = config

    # 定义前向传播过程
    def call(self, hidden_states, residual_tensor):
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 将处理后的隐藏状态和残差相加，然后进行层归一化
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        return hidden_states

    # 定义模型构建过程
    def build(self, input_shape=None):
        # 如果模型已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 构建层归一化层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)

# TFFFNLayer 是一个 Keras 层，负责处理前馈神经网络的计算
class TFFFNLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 创建中间层和输出层
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        self.mobilebert_output = TFFFNOutput(config, name="output")

    # 定义前向传播过程
    def call(self, hidden_states):
        # 计算中间层的输出
        intermediate_output = self.intermediate(hidden_states)
        # 计算输出层的输出
        layer_outputs = self.mobilebert_output(intermediate_output, hidden_states)
        return layer_outputs

    # 定义模型构建过程
    def build(self, input_shape=None):
        # 如果模型已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 构建输出层
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)

# TFMobileBertLayer 是一个 Keras 层，负责处理 MobileBERT 模型的一个层
class TFMobileBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 保存配置信息
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks
        # 创建注意力层、中间层和输出层
        self.attention = TFMobileBertAttention(config, name="attention")
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        self.mobilebert_output = TFMobileBertOutput(config, name="output")

        # 如果使用瓶颈层，则创建一个瓶颈层
        if self.use_bottleneck:
            self.bottleneck = TFBottleneck(config, name="bottleneck")
        # 如果需要多个前馈神经网络，则创建相应数量的 TFFFNLayer
        if config.num_feedforward_networks > 1:
            self.ffn = [TFFFNLayer(config, name=f"ffn.{i}") for i in range(config.num_feedforward_networks - 1)]
    # 定义一个函数，接收隐藏状态、注意力掩码、头遮罩、输出注意力和训练标志作为参数
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        # 如果使用瓶颈结构
        if self.use_bottleneck:
            # 使用瓶颈结构处理隐藏状态
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            # 否则复制隐藏状态四份作为查询、键、值、层输入
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        # 使用注意力机制处理输入并获取注意力输出
        attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions,
            training=training,
        )

        # 从注意力输出中提取注意力值
        attention_output = attention_outputs[0]
        s = (attention_output,)

        # 如果存在多个前馈网络
        if self.num_feedforward_networks != 1:
            # 遍历前馈网络
            for i, ffn_module in enumerate(self.ffn):
                # 使用前馈网络处理注意力输出
                attention_output = ffn_module(attention_output)
                # 将处理后的输出添加到元组中
                s += (attention_output,)

        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用移动BERT输出层处理注意力输出、隐藏状态
        layer_output = self.mobilebert_output(intermediate_output, attention_output, hidden_states, training=training)

        # 构建输出元组，包括处理后的层输出、注意力值，以及其他相关的值
        outputs = (
            (layer_output,)
            + attention_outputs[1:]
            + (
                tf.constant(0),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )  # 如果有输出注意力，则添加到输出中

        return outputs

    # 构建网络
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 检查并构建注意力、中间层、移动BERT输出层、瓶颈层和前馈网络
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        if getattr(self, "ffn", None) is not None:
            for layer in self.ffn:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFMobileBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 设置是否输出注意力权重
        self.output_attentions = config.output_attentions
        # 设置是否输出所有隐藏层的状态
        self.output_hidden_states = config.output_hidden_states
        # 创建多层 MobileBERT 层
        self.layer = [TFMobileBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        # 初始化存储所有隐藏状态的变量
        all_hidden_states = () if output_hidden_states else None
        # 初始化存储所有注意力权重的变量
        all_attentions = () if output_attentions else None
        # 遍历每一层 MobileBERT 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出所有隐藏状态，则保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的处理函数
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则保存当前层的注意力权重
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回指定的值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 如果返回字典，则创建 TFBaseModelOutput 对象返回
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过了，则直接返回
        if getattr(self, "layer", None) is not None:
            # 遍历每一层 MobileBERT 层，并构建之
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFMobileBertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 设置是否激活分类器
        self.do_activate = config.classifier_activation
        # 如果需要激活，则创建一个全连接层
        if self.do_activate:
            self.dense = tf.keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                activation="tanh",
                name="dense",
            )
        self.config = config

    def call(self, hidden_states):
        # 取第一个标记对应的隐藏状态作为汇聚结果
        first_token_tensor = hidden_states[:, 0]
        # 如果不需要激活，则直接返回第一个标记对应的隐藏状态
        if not self.do_activate:
            return first_token_tensor
        else:
            # 如果需要激活，则通过全连接层进行激活
            pooled_output = self.dense(first_token_tensor)
            return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过了，则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])
class TFMobileBertPredictionHeadTransform(tf.keras.layers.Layer):
    # 定义 TFMobileBertPredictionHeadTransform 类，继承自 tf.keras.layers.Layer
    def __init__(self, config, **kwargs):
        # 初始化方法，接受 config 参数和可选的关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化方法
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个全连接层，设置隐藏单元数量和初始化方式
        if isinstance(config.hidden_act, str):
            # 检查 config.hidden_act 是否为字符串
            self.transform_act_fn = get_tf_activation(config.hidden_act)
            # 如果是字符串，则调用 get_tf_activation 函数，将结果保存到 transform_act_fn 中
        else:
            self.transform_act_fn = config.hidden_act
            # 否则直接将 config.hidden_act 保存到 transform_act_fn 中
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 LayerNorm 层
        self.config = config
        # 保存 config 参数

    def call(self, hidden_states):
        # 定义 call 方法，接受 hidden_states 参数
        hidden_states = self.dense(hidden_states)
        # 应用全连接层到 hidden_states 上
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用激活函数到 hidden_states 上
        hidden_states = self.LayerNorm(hidden_states)
        # 应用 LayerNorm 到 hidden_states 上
        return hidden_states
        # 返回处理后的 hidden_states

    def build(self, input_shape=None):
        # 定义 build 方法，接受 input_shape 参数，默认为 None
        if self.built:
            return
            # 如果已经构建了，则直接返回
        self.built = True
        # 设置 built 属性为 True
        if getattr(self, "dense", None) is not None:
            # 检查是否存在 dense 属性
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
                # 使用 name_scope 设置命名空间，构建 dense 层
        if getattr(self, "LayerNorm", None) is not None:
            # 检查是否存在 LayerNorm 属性
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)
                # 使用 name_scope 设置命名空间，构建 LayerNorm 层


class TFMobileBertLMPredictionHead(tf.keras.layers.Layer):
    # 定义 TFMobileBertLMPredictionHead 类，继承自 tf.keras.layers.Layer
    def __init__(self, config, **kwargs):
        # 初始化方法，接受 config 参数和可选的关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化方法
        self.transform = TFMobileBertPredictionHeadTransform(config, name="transform")
        # 创建 TFMobileBertPredictionHeadTransform 实例
        self.config = config
        # 保存 config 参数

    def build(self, input_shape=None):
        # 定义 build 方法，接受 input_shape 参数，默认为 None
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        # 添加名为 bias 的权重，形状为 (config.vocab_size,)，初始值为 "zeros"，可训练，名称为 "bias"
        self.dense = self.add_weight(
            shape=(self.config.hidden_size - self.config.embedding_size, self.config.vocab_size),
            initializer="zeros",
            trainable=True,
            name="dense/weight",
        )
        # 添加名为 dense 的权重，形状为 (config.hidden_size - config.embedding_size, config.vocab_size)，初始值为 "zeros"，可训练，名称为 "dense/weight"
        self.decoder = self.add_weight(
            shape=(self.config.vocab_size, self.config.embedding_size),
            initializer="zeros",
            trainable=True,
            name="decoder/weight",
        )
        # 添加名为 decoder 的权重，形状为 (config.vocab_size, config.embedding_size)，初始值为 "zeros"，可训练，名称为 "decoder/weight"

        if self.built:
            return
            # 如果已经构建了，则直接返回
        self.built = True
        # 设置 built 属性为 True
        if getattr(self, "transform", None) is not None:
            # 检查是否存在 transform 属性
            with tf.name_scope(self.transform.name):
                self.transform.build(None)
                # 使用 name_scope 设置命名空间，构建 transform 层

    def get_output_embeddings(self):
        # 定义 get_output_embeddings 方法
        return self
        # 返回当前实例

    def set_output_embeddings(self, value):
        # 定义 set_output_embeddings 方法，接受 value 参数
        self.decoder = value
        # 将 value 赋值给 decoder
        self.config.vocab_size = shape_list(value)[0]
        # 获取 value 的形状并将第一个元素赋值给 config.vocab_size

    def get_bias(self):
        # 定义 get_bias 方法
        return {"bias": self.bias}
        # 返回一个包含 bias 属性的字典

    def set_bias(self, value):
        # 定义 set_bias 方法，接受 value 参数
        self.bias = value["bias"]
        # 将 value 中的 bias 赋值给实例的 bias 属性
        self.config.vocab_size = shape_list(value["bias"])[0]
        # 获取 value 中 bias 的形状并将第一个元素赋值给 config.vocab_size
    # 定义一个方法，接受隐藏状态作为参数
    def call(self, hidden_states):
        # 对隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        # 计算隐藏状态和解码器的转置以及密集层的矩阵乘积
        hidden_states = tf.matmul(hidden_states, tf.concat([tf.transpose(self.decoder), self.dense], axis=0))
        # 添加偏置
        hidden_states = hidden_states + self.bias
        # 返回处理后的隐藏状态
        return hidden_states
class TFMobileBertMLMHead(tf.keras.layers.Layer):
    # 创建自定义的 TensorFlow 移动 Bert MLM 头部
    def __init__(self, config, **kwargs):
        # 初始化函数
        super().__init__(**kwargs)
        # 创建预测对象
        self.predictions = TFMobileBertLMPredictionHead(config, name="predictions")

    def call(self, sequence_output):
        # 调用函数，对序列输出进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    def build(self, input_shape=None):
        # 构建函数，如果已经构建则直接返回，否则构建预测对象
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


@keras_serializable
class TFMobileBertMainLayer(tf.keras.layers.Layer):
    # 自定义的 TF 移动 Bert 主层
    config_class = MobileBertConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 初始化函数
        super().__init__(**kwargs)
        # 初始化属性
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        # 创建嵌入层对象
        self.embeddings = TFMobileBertEmbeddings(config, name="embeddings")
        # 创建编码器对象
        self.encoder = TFMobileBertEncoder(config, name="encoder")
        # 如果需要添加池化层，则创建池化层对象
        self.pooler = TFMobileBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        # 获取输入嵌入层对象
        return self.embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层对象
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

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
    # 构建函数
    def build(self, input_shape=None):
        # 如果已经构建则直接返回
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    # TF 移动 Bert 预训练模型的抽象类
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    base_model_prefix = "mobilebert"


@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFMobileBertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 损失值，用于模型训练时的优化
    loss: tf.Tensor | None = None
    # 语言模型头部的预测得分，即 SoftMax 之前的每个词汇标记的得分
    prediction_logits: tf.Tensor = None
    # 下一个序列预测（分类）头部的预测得分，即 SoftMax 之前的 True/False 继续的得分
    seq_relationship_logits: tf.Tensor = None
    # 模型每一层输出的隐藏状态，包括初始嵌入输出
    hidden_states: Tuple[tf.Tensor] | None = None
    # 每一层的注意力权重，用于计算自注意力头中的加权平均值
    attentions: Tuple[tf.Tensor] | None = None


MOBILEBERT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
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
    # 在使用模型时，可以采取不同的参数格式：
    # - 一个仅包含 `input_ids` 的 Tensor：`model(input_ids)`
    # - 一个长度可变的列表，其中包含按照文档字符串中给定顺序的一个或多个输入 Tensor：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 一个字典，其中包含一个或多个与文档字符串中给定输入名称相关联的输入 Tensor：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # 需要注意的是，在使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，不需要担心这些细节，可以像对其他 Python 函数一样传递输入！
    
    # 参数：
    # config ([`MobileBertConfig`]): 包含模型所有参数的模型配置类。
    # 使用配置文件初始化模型时不会加载与模型相关的权重，只加载配置。可以查看[`~PreTrainedModel.from_pretrained`]方法来加载模型权重。
"""

MOBILEBERT_INPUTS_DOCSTRING = r"""
"""

# 添加 MobileBert Model 类的定义，继承自 TFMobileBertPreTrainedModel
# 该模型输出原始隐藏状态，没有特定的顶部头部
# 通过装饰器添加了注释
class TFMobileBertModel(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数
        super().__init__(config, *inputs, **kwargs)
        # 创建 MobileBert 主层
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")

    # 定义模型调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        # 调用 MobileBert 主层，返回输出
        outputs = self.mobilebert(
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

        return outputs

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 MobileBert 实例
        if getattr(self, "mobilebert", None) is not None:
            # 在 MobileBert 主层下建立模型
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)

# 添加 MobileBert For PreTraining 类的定义，继承自 TFMobileBertPreTrainedModel 和 TFMobileBertPreTrainingLoss
# 在预训练期间，该模型在顶部有两个头部：一个是“masked language modeling”头部，另一个是“next sentence prediction (classification)”头部
# 通过装饰器添加了注释
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel, TFMobileBertPreTrainingLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数
        super().__init__(config, *inputs, **kwargs)
        # 创建 MobileBert 主层
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        # 创建 MLM 头部
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")
        # 创建 NSP 头部
        self.seq_relationship = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    # 获取语言建模头部
    def get_lm_head(self):
        return self.predictions.predictions
    # 发出警告，提示该方法已经废弃，建议使用 `get_bias` 替代
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回包含特定字符串的名称，由当前对象的属性名组成
        return self.name + "/" + self.predictions.name + "/" + self.predictions.predictions.name

    # 标记输入参数解包，为模型 forward 方法添加起始字符串注释，替换模型 forward 方法返回结果的字符串注释
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        # 定义模型 forward 方法的输入参数
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 指定返回值的类型为 Union[Tuple, TFMobileBertForPreTrainingOutput]
        ) -> Union[Tuple, TFMobileBertForPreTrainingOutput]:
            # 文档字符串，解释函数返回值和示例代码的作用
            r"""
            Return:
    
            Examples:
    
            ```python
            >>> import tensorflow as tf
            >>> from transformers import AutoTokenizer, TFMobileBertForPreTraining
    
            >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
            >>> model = TFMobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # 批处理大小为1
            >>> outputs = model(input_ids)  # 调用模型
            >>> prediction_scores, seq_relationship_scores = outputs[:2]  # 提取前两个输出值
            ```"""
            # 调用 MobileBert 模型，传入输入参数
            outputs = self.mobilebert(
                input_ids,  # 输入的标记 ID
                attention_mask=attention_mask,  # 注意力掩码
                token_type_ids=token_type_ids,  # 标记类型 ID
                position_ids=position_ids,  # 位置 ID
                head_mask=head_mask,  # 头部掩码
                inputs_embeds=inputs_embeds,  # 输入嵌入
                output_attentions=output_attentions,  # 是否输出注意力
                output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
                return_dict=return_dict,  # 是否返回字典
                training=training,  # 是否在训练模式
            )
    
            # 提取 MobileBert 模型的序列输出和池化输出
            sequence_output, pooled_output = outputs[:2]
            # 使用序列输出计算预测得分
            prediction_scores = self.predictions(sequence_output)
            # 使用池化输出计算序列关系得分
            seq_relationship_score = self.seq_relationship(pooled_output)
    
            # 初始化总损失为 None
            total_loss = None
            # 如果提供了标签和下一个句子标签，则计算总损失
            if labels is not None and next_sentence_label is not None:
                d_labels = {"labels": labels}  # 创建一个包含标签的字典
                d_labels["next_sentence_label"] = next_sentence_label  # 添加下一个句子标签
                # 使用提供的标签和 logits 计算总损失
                total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))
    
            # 如果不返回字典形式，则返回元组形式的输出
            if not return_dict:
                output = (prediction_scores, seq_relationship_score) + outputs[2:]  # 合并预测得分、关系得分和其他输出
                # 如果有总损失，则将其添加到元组中返回
                return ((total_loss,) + output) if total_loss is not None else output
    
            # 如果返回字典形式，则创建一个 TFMobileBertForPreTrainingOutput 对象
            return TFMobileBertForPreTrainingOutput(
                loss=total_loss,  # 总损失
                prediction_logits=prediction_scores,  # 预测得分
                seq_relationship_logits=seq_relationship_score,  # 序列关系得分
                hidden_states=outputs.hidden_states,  # 隐藏状态
                attentions=outputs.attentions,  # 注意力
            )
    
        # 构建方法，主要用于初始化模型中的组件
        def build(self, input_shape=None):
            # 如果模型已构建，直接返回
            if self.built:
                return
            # 设置已构建标志为 True
            self.built = True
            # 如果 mobilebert 属性存在，则调用其构建方法
            if getattr(self, "mobilebert", None) is not None:
                with tf.name_scope(self.mobilebert.name):  # 为构建阶段设置命名空间
                    self.mobilebert.build(None)  # 构建 MobileBert
            # 如果 predictions 属性存在，则调用其构建方法
            if getattr(self, "predictions", None) is not None:
                with tf.name_scope(self.predictions.name):  # 为构建阶段设置命名空间
                    self.predictions.build(None)  # 构建预测层
            # 如果 seq_relationship 属性存在，则调用其构建方法
            if getattr(self, "seq_relationship", None) is not None:
                with tf.name_scope(self.seq_relationship.name):  # 为构建阶段设置命名空间
                    self.seq_relationship.build(None)  # 构建序列关系层
    
        # 重命名 TensorFlow 权重到 PyTorch 权重的方法
        def tf_to_pt_weight_rename(self, tf_weight):
            # 如果 TensorFlow 权重名是 "cls.predictions.decoder.weight"
            if tf_weight == "cls.predictions.decoder.weight":
                # 将其映射到 PyTorch 中对应的权重名
                return tf_weight, "mobilebert.embeddings.word_embeddings.weight"
            else:
                # 否则保持原名
                return (tf_weight,)
# 用于为TFMobileBertForMaskedLM类添加文档字符串，指定为一个带有"language modeling"头的MobileBert模型
@add_start_docstrings("""MobileBert Model with a `language modeling` head on top.""", MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMaskedLM(TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 当从PT模型加载TF模型时，带有‘.’的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"seq_relationship___cls",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化MobileBert模型的主层，不添加池化层，指定名称为"mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        # 初始化MobileBert模型的MLM头，指定名称为"predictions___cls"
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")

    # 获取语言模型头
    def get_lm_head(self):
        return self.predictions.predictions

    # 获取前缀偏置名称
    def get_prefix_bias_name(self):
        # 发出警告，此方法被弃用，建议使用`get_bias`替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    # 定义call方法，实现模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.57,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
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
    # 定义一个 Union 类型提示，可以返回 Tuple 或 TFMaskedLMOutput
        ) -> Union[Tuple, TFMaskedLMOutput]:
            r"""
            labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels
            """
            # 通过 mobilebert 模型处理输入数据，获得序列输出
            outputs = self.mobilebert(
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
            sequence_output = outputs[0]
            # 使用 predictions 层计算预测分数
            prediction_scores = self.predictions(sequence_output, training=training)
    
            # 如果提供了 labels，则计算 masked language modeling loss
            loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
    
            # 根据 return_dict 参数返回适当的输出
            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
    
            return TFMaskedLMOutput(
                loss=loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    
        # 构建模型
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 构建 mobilebert 模型
            if getattr(self, "mobilebert", None) is not None:
                with tf.name_scope(self.mobilebert.name):
                    self.mobilebert.build(None)
            # 构建 predictions 层
            if getattr(self, "predictions", None) is not None:
                with tf.name_scope(self.predictions.name):
                    self.predictions.build(None)
    
        # 定义权重转换函数
        def tf_to_pt_weight_rename(self, tf_weight):
            if tf_weight == "cls.predictions.decoder.weight":
                return tf_weight, "mobilebert.embeddings.word_embeddings.weight"
            else:
                return (tf_weight,)
class TFMobileBertOnlyNSPHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于进行序列关系预测，输出维度为2
        self.seq_relationship = tf.keras.layers.Dense(2, name="seq_relationship")
        self.config = config

    def call(self, pooled_output):
        # 通过全连接层计算序列关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

    def build(self, input_shape=None):
        # 如果已经构建完成，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在序列关系全连接层，构建对应的神经网络层
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build([None, None, self.config.hidden_size])

@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top.""",
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):
    # 在从 PyTorch 模型加载到 TensorFlow 模型时，忽略指定的层名称
    _keys_to_ignore_on_load_unexpected = [r"predictions___cls", r"cls.predictions"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 MobileBert 主层，并命名为 "mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        # 创建仅包含序列关系预测的头部，命名为 "seq_relationship___cls"
        self.cls = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFNextSentencePredictorOutput]:
        r"""
        Return: 返回值类型的说明

        Examples: 示例用法

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        ```"""
        获取 MobileBERT 模型的输出
        outputs = self.mobilebert(
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
        # 从 MobileBERT 的输出中获取池化后的输出
        pooled_output = outputs[1]
        # 使用分类层对池化后的输出进行分类，得到下一句预测的得分
        seq_relationship_scores = self.cls(pooled_output)

        # 计算下一句预测的损失
        next_sentence_loss = (
            None
            if next_sentence_label is None
            else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
        )

        # 如果不返回字典形式的结果，则构造输出元组
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 返回 TFNextSentencePredictorOutput 类型的结果
        return TFNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建了模型，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为 True
        self.built = True
        # 如果存在 MobileBERT 模型，则构建 MobileBERT
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在分类层，则构建分类层
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
    @add_start_docstrings(
        """
        MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
        pooled output) e.g. for GLUE tasks.
        """,
        MOBILEBERT_START_DOCSTRING,
    )
    class TFMobileBertForSequenceClassification(TFMobileBertPreTrainedModel, TFSequenceClassificationLoss):
        # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
        _keys_to_ignore_on_load_unexpected = [
            r"predictions___cls",
            r"seq_relationship___cls",
            r"cls.predictions",
            r"cls.seq_relationship",
        ]
        _keys_to_ignore_on_load_missing = [r"dropout"]

        def __init__(self, config, *inputs, **kwargs):
            # 初始化 TFMobileBertForSequenceClassification 类
            super().__init__(config, *inputs, **kwargs)
            # 获得标签数
            self.num_labels = config.num_labels

            # 创建 MobileBert 主层对象
            self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
            # 判断是否有分类器的 dropout 参数，如果没有则用隐藏层的 dropout 参数
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = tf.keras.layers.Dropout(classifier_dropout)
            # 创建分类器
            self.classifier = tf.keras.layers.Dense(
                config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
            )
            self.config = config

        @unpack_inputs
        @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
            output_type=TFSequenceClassifierOutput,
            config_class=_CONFIG_FOR_DOC,
            expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
            expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
        )
        def call(
            self,
            input_ids: TFModelInputType | None = None,
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
    # 根据输入的标签计算序列分类/回归损失
    def call(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        # 通过 mobilebert 模型获取输出，包括隐藏状态和注意力权重
        outputs = self.mobilebert(
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
        # 获取池化输出
        pooled_output = outputs[1]
        
        # 对池化输出进行 dropout 操作
        pooled_output = self.dropout(pooled_output, training=training)
        # 将池化输出传入分类器得到分类结果
        logits = self.classifier(pooled_output)
    
        # 如果传入了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
        # 如果不需要返回 dict，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 构建 mobilebert 子模型
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 构建分类器子模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 定义用于问答任务的 MobileBert 模型
@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForQuestionAnswering(TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss):
    # 定义在加载预训练模型时忽略的意外/缺失层名称
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建 MobileBert 主体层
        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        # 创建用于计算 span 起始和结束位置的全连接层
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
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
    # 定义函数，接受输入并返回包含start和end位置的元组或TFQuestionAnsweringModelOutput类型的对象
    def predict(
        input_ids: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        start_positions: Optional[tf.Tensor] = None,
        end_positions: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        training: bool = False,
    ) -> Union[Tuple, TFQuestionAnsweringModelOutput]:
        # 获取输出结果，调用mobilebert模型进行推理
        outputs = self.mobilebert(
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
        sequence_output = outputs[0]

        # 获取logits并进行分割得到start和end的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 初始化loss值为None
        loss = None
        # 如果有start_positions和end_positions，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果return_dict为False，则返回元组类型的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回TFQuestionAnsweringModelOutput类型的对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果mobilebert模型存在，则构建mobilebert模型
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果qa_outputs模型存在，则构建qa_outputs模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
# 这是一个带有多项选择分类头的 MobileBert 模型
# 它可以用于 RocStories/SWAG 等任务
@add_start_docstrings(
    """
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):
    # 这些是在从 PyTorch 模型加载 TensorFlow 模型时忽略的意外/缺失层的名称
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 MobileBert 的主要层
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        # 添加一个 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 添加一个用于分类的全连接层
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果输入了 input_ids，则获取其第二维的大小为 num_choices
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            # 获取输入的序列长度
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，获取 inputs_embeds 的第二维的大小为 num_choices
            num_choices = shape_list(inputs_embeds)[1]
            # 获取输入的序列长度
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入的 input_ids 展平为二维张量，如果没有输入 input_ids，则置为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将输入的 attention_mask 展平为二维张量，如果没有输入 attention_mask，则置为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将输入的 token_type_ids 展平为二维张量，如果没有输入 token_type_ids，则置为 None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        # 将输入的 position_ids 展平为二维张量，如果没有输入 position_ids，则置为 None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        # 将输入的 inputs_embeds 展平为三维张量，如果没有输入 inputs_embeds，则置为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 使用 MobileBERT 模型处理输入，并返回输出结果
        outputs = self.mobilebert(
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
        # 获取池化后的输出
        pooled_output = outputs[1]
        # 对池化后的输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output, training=training)
        # 使用分类器对池化后的输出进行分类，得到 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重新调整形状为二维张量
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果提供了 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不要求返回字典形式的输出
        if not return_dict:
            # 构建输出元组
            output = (reshaped_logits,) + outputs[2:]
            # 如果存在损失，则将损失添加到输出元组中
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的输出
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在 MobileBERT 模型，则构建 MobileBERT
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器添加模型的文档字符串，介绍带有标记分类头的 MobileBert 模型，例如用于命名实体识别（NER）任务
# 继承 TFMobileBertPreTrainedModel 和 TFTokenClassificationLoss 类
class TFMobileBertForTokenClassification(TFMobileBertPreTrainedModel, TFTokenClassificationLoss):
    # 加载 TF 模型时要忽略的键列表，表示预期未授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    # 加载 TF 模型时要忽略的键列表，表示预期缺失的层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化方法，接受配置和其他参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建 MobileBert 主层对象，不添加池化层，命名为"mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        # 设置分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建丢弃层对象
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 创建全连接层，用于标记分类
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config

    # 使用装饰器 unpack_inputs 对输入进行解包，使得输入可以直接传入函数中
    # 使用装饰器 add_start_docstrings_to_model_forward 添加模型前向传播的文档字符串，介绍输入格式
    # 使用装饰器 add_code_sample_docstrings 添加模型示例的文档字符串，介绍检查点、输出类型、配置类和预期输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
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
    ) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 MobileBERT 模型来获取输出结果
        outputs = self.mobilebert(
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
        # 获取模型输出的序列结果
        sequence_output = outputs[0]

        # 对序列结果进行 dropout 处理
        sequence_output = self.dropout(sequence_output, training=training)
        
        # 通过分类器获取 logits
        logits = self.classifier(sequence_output)

        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不使用返回字典形式，则构建返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典形式，则返回 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 MobileBERT 模型，则构建 MobileBERT 模型
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
```