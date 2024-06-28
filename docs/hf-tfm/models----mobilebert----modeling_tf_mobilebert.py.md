# `.\models\mobilebert\modeling_tf_mobilebert.py`

```py
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

# Importing specific modules and classes from other files in the package
from ...activations_tf import get_tf_activation
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
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilebert import MobileBertConfig

# Setting up logging for this module
logger = logging.get_logger(__name__)

# Documentation constants for different tasks/models

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

# List of pretrained model archives for MobileBERT
TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilebert-uncased",
    # See all MobileBERT models at https://huggingface.co/models?filter=mobilebert
]

# Definition of a custom loss class for MobileBERT pretraining tasks
class TFMobileBertPreTrainingLoss:
    """
    Placeholder class definition for the MobileBERT pre-training loss.
    This class is likely intended to be implemented later.
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """

    # 定义一个计算损失函数，适用于类似BERT的预训练任务，即结合NSP（Next Sentence Prediction）和MLM（Masked Language Modeling）
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        # 使用稀疏分类交叉熵损失函数，适用于逻辑回归（logits），保留每个样本的独立损失
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)

        # 将负标签截断为零，以避免NaN和错误，这些位置稍后会被掩盖
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # 确保仅计算不等于-100的标签的损失
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)

        # 再次将负标签截断为零，避免NaN和错误，这些位置稍后会被掩盖
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels["next_sentence_label"]), y_pred=logits[1])
        ns_loss_mask = tf.cast(labels["next_sentence_label"] != -100, dtype=unmasked_ns_loss.dtype)
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        reduced_masked_ns_loss = tf.reduce_sum(masked_ns_loss) / tf.reduce_sum(ns_loss_mask)

        # 返回损失的张量形状
        return tf.reshape(reduced_masked_lm_loss + reduced_masked_ns_loss, (1,))
class TFMobileBertIntermediate(keras.layers.Layer):
    # 初始化中间层，包括一个全连接层和激活函数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建全连接层，使用配置中的中间层大小，命名为"dense"
        self.dense = keras.layers.Dense(config.intermediate_size, name="dense")

        # 根据配置选择激活函数，如果是字符串则通过辅助函数获取对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 定义调用方法，对输入的隐藏状态执行全连接层和激活函数操作
    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层，确保只构建一次，并设置全连接层的输入形状
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 设置全连接层的输入形状，其中 None 表示批量大小可变
                self.dense.build([None, None, self.config.true_hidden_size])


class TFLayerNorm(keras.layers.LayerNormalization):
    # 初始化 LayerNormalization 层，指定特征大小
    def __init__(self, feat_size, *args, **kwargs):
        self.feat_size = feat_size
        super().__init__(*args, **kwargs)

    # 构建层，设置输入形状为 [None, None, feat_size]
    def build(self, input_shape=None):
        super().build([None, None, self.feat_size])


class TFNoNorm(keras.layers.Layer):
    # 初始化不进行归一化的层，指定特征大小和其他参数
    def __init__(self, feat_size, epsilon=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_size = feat_size

    # 构建层，设置偏置和权重参数的形状，并调用父类的 build 方法
    def build(self, input_shape):
        self.bias = self.add_weight("bias", shape=[self.feat_size], initializer="zeros")
        self.weight = self.add_weight("weight", shape=[self.feat_size], initializer="ones")
        super().build(input_shape)

    # 定义调用方法，对输入执行加权和加偏操作
    def call(self, inputs: tf.Tensor):
        return inputs * self.weight + self.bias


# 定义一个字典，将字符串类型的归一化方式映射到对应的类
NORM2FN = {"layer_norm": TFLayerNorm, "no_norm": TFNoNorm}


class TFMobileBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化嵌入层，包括词、位置和类型嵌入的构建
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 从配置中获取三元输入标志、嵌入大小等信息
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range

        # 创建嵌入转换层，将输入转换为隐藏大小的表示，命名为"embedding_transformation"
        self.embedding_transformation = keras.layers.Dense(config.hidden_size, name="embedding_transformation")

        # 创建归一化层，根据配置中的归一化类型选择对应的类，设置 epsilon 和名称
        # 这里保持不改变 TensorFlow 模型变量名称，以便能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )

        # 创建 dropout 层，根据配置中的隐藏层 dropout 概率设置丢弃率
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

        # 计算嵌入输入大小，考虑是否使用三元输入
        self.embedded_input_size = self.embedding_size * (3 if self.trigram_input else 1)
    # 定义 build 方法，用于构建模型的各个部分
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建权重变量
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 在 "token_type_embeddings" 命名空间下创建 token 类型的嵌入权重变量
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 在 "position_embeddings" 命名空间下创建位置编码的嵌入权重变量
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 如果模型已经构建过，直接返回
        if self.built:
            return
        
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在 embedding_transformation 属性，构建对应的变换层
        if getattr(self, "embedding_transformation", None) is not None:
            with tf.name_scope(self.embedding_transformation.name):
                # 使用 build 方法构建 embedding_transformation 层
                self.embedding_transformation.build([None, None, self.embedded_input_size])
        
        # 如果存在 LayerNorm 属性，构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 使用 build 方法构建 LayerNorm 层
                self.LayerNorm.build(None)
    def call(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言确保 input_ids 或 inputs_embeds 至少有一个不为 None
        assert not (input_ids is None and inputs_embeds is None)

        # 如果传入了 input_ids，则根据 input_ids 从权重矩阵中获取对应的嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状，去掉最后一维（用于嵌入维度）
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 token_type_ids，则创建一个与输入嵌入张量形状相同的张量，并填充为 0
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果设定了 trigram_input 标志
        if self.trigram_input:
            # 根据 MobileBERT 论文中的描述，对输入嵌入张量进行 trigram 输入处理
            inputs_embeds = tf.concat(
                [
                    tf.pad(inputs_embeds[:, 1:], ((0, 0), (0, 1), (0, 0))),
                    inputs_embeds,
                    tf.pad(inputs_embeds[:, :-1], ((0, 0), (1, 0), (0, 0))),
                ],
                axis=2,
            )

        # 如果设定了 trigram_input 标志或者 embedding_size 不等于 hidden_size
        if self.trigram_input or self.embedding_size != self.hidden_size:
            # 对输入嵌入张量进行额外的嵌入转换处理
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # 如果未提供 position_ids，则创建一个一维张量，包含从 0 到输入张量最后维度长度的范围值
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 根据 position_ids 获取位置嵌入张量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 获取 token 类型嵌入张量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 最终的嵌入张量由输入嵌入张量、位置嵌入张量和 token 类型嵌入张量相加而得
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 应用 LayerNorm 层进行标准化处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 根据训练状态应用 dropout 层
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入张量
        return final_embeddings
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 设置注意力头数和是否输出注意力权重的配置
        self.num_attention_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        # 确保隐藏层大小能被注意力头数整除
        assert config.hidden_size % config.num_attention_heads == 0
        # 计算每个注意力头的大小和所有注意力头的总大小
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值矩阵的全连接层
        self.query = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 初始化 dropout 层，并设置注意力概率
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        # 将输入张量 x 从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=False
    ):
        # 实现自注意力机制的前向传播
        ):
            # 获取 batch_size
            batch_size = shape_list(attention_mask)[0]
            # 计算 query 的混合层
            mixed_query_layer = self.query(query_tensor)
            # 计算 key 的混合层
            mixed_key_layer = self.key(key_tensor)
            # 计算 value 的混合层
            mixed_value_layer = self.value(value_tensor)
            # 调整混合后的 query 层为得分计算做准备
            query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
            # 调整混合后的 key 层为得分计算做准备
            key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
            # 调整混合后的 value 层为得分计算做准备
            value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

            # 计算 "query" 和 "key" 之间的点积，得到原始的注意力分数
            attention_scores = tf.matmul(
                query_layer, key_layer, transpose_b=True
            )  # (batch size, num_heads, seq_len_q, seq_len_k)
            # 缩放注意力分数
            dk = tf.cast(shape_list(key_layer)[-1], dtype=attention_scores.dtype)
            attention_scores = attention_scores / tf.math.sqrt(dk)

            # 如果有注意力掩码，应用它（在 TFMobileBertModel call() 函数中预先计算）
            if attention_mask is not None:
                attention_mask = tf.cast(attention_mask, dtype=attention_scores.dtype)
                attention_scores = attention_scores + attention_mask

            # 将注意力分数归一化为概率
            attention_probs = stable_softmax(attention_scores, axis=-1)

            # 对注意力概率进行 dropout
            attention_probs = self.dropout(attention_probs, training=training)

            # 如果有头部掩码，应用头部掩码
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # 计算上下文向量
            context_layer = tf.matmul(attention_probs, value_layer)

            # 转置和重塑上下文向量
            context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
            context_layer = tf.reshape(
                context_layer, (batch_size, -1, self.all_head_size)
            )  # (batch_size, seq_len_q, all_head_size)

            # 返回输出结果，根据是否需要返回注意力概率
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            return outputs

        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 如果已经定义了 query 层，建立它
            if getattr(self, "query", None) is not None:
                with tf.name_scope(self.query.name):
                    self.query.build([None, None, self.config.true_hidden_size])
            # 如果已经定义了 key 层，建立它
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.true_hidden_size])
            # 如果已经定义了 value 层，建立它
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
# 定义 TFMobileBertSelfOutput 类，继承自 keras.layers.Layer
class TFMobileBertSelfOutput(keras.layers.Layer):
    
    # 初始化方法，接收 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据 config 设置是否使用瓶颈层
        self.use_bottleneck = config.use_bottleneck
        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 根据 config 设置归一化层，例如 LayerNorm
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 如果不使用瓶颈层，则创建一个 dropout 层，用于训练时随机丢弃部分神经元
        if not self.use_bottleneck:
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 保存 config 对象
        self.config = config

    # 定义调用方法，用于前向传播计算
    def call(self, hidden_states, residual_tensor, training=False):
        # 使用全连接层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 如果不使用瓶颈层，则对变换后的隐藏状态进行 dropout 处理
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
        # 将变换后的隐藏状态与残差张量相加，并通过归一化层处理
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        return hidden_states

    # 构建方法，用于构建层次结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        # 如果存在归一化层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


# 定义 TFMobileBertAttention 类，继承自 keras.layers.Layer
class TFMobileBertAttention(keras.layers.Layer):
    
    # 初始化方法，接收 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建自注意力层对象
        self.self = TFMobileBertSelfAttention(config, name="self")
        # 创建 TFMobileBertSelfOutput 层对象，用于处理自注意力层的输出
        self.mobilebert_output = TFMobileBertSelfOutput(config, name="output")

    # 头部剪枝方法，抛出未实现错误
    def prune_heads(self, heads):
        raise NotImplementedError

    # 定义调用方法，用于前向传播计算
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
        # 使用自注意力层处理输入张量
        self_outputs = self.self(
            query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=training
        )
        # 使用 TFMobileBertSelfOutput 层处理自注意力层的输出和层输入张量
        attention_output = self.mobilebert_output(self_outputs[0], layer_input, training=training)
        # 构造输出元组，包含注意力输出和可能的额外输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要额外的注意力输出，则添加
        return outputs

    # 构建方法，用于构建层次结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在自注意力层，则构建该层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果存在 TFMobileBertSelfOutput 层，则构建该层
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)


# 定义 TFOutputBottleneck 类，继承自 keras.layers.Layer
class TFOutputBottleneck(keras.layers.Layer):
    # 初始化方法，用于创建对象时初始化各个成员变量和层对象
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层对象，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(config.hidden_size, name="dense")
        # 创建一个归一化层对象，根据配置选择不同的归一化类型
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        # 创建一个 Dropout 层对象，用于在训练时进行随机失活
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 存储配置对象，以便后续使用
        self.config = config

    # 调用方法，用于实际执行神经网络的前向计算过程
    def call(self, hidden_states, residual_tensor, training=False):
        # 线性变换层，将隐藏状态映射到新的空间
        layer_outputs = self.dense(hidden_states)
        # 在训练时对输出进行 Dropout 处理，防止过拟合
        layer_outputs = self.dropout(layer_outputs, training=training)
        # 应用归一化层，处理残差连接和变换后的输出
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        # 返回处理后的输出
        return layer_outputs

    # 构建方法，用于构建网络层的内部结构
    def build(self, input_shape=None):
        # 如果已经构建过网络层，直接返回
        if self.built:
            return
        # 标记当前网络层已构建
        self.built = True
        # 如果存在 dense 层对象，则根据配置构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        # 如果存在 LayerNorm 层对象，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)
class TFMobileBertOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck  # 根据配置决定是否使用瓶颈层
        self.dense = keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )  # 创建全连接层，用于转换输入的隐藏状态维度
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )  # 根据配置选择合适的归一化层
        if not self.use_bottleneck:
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)  # 如果不使用瓶颈层，则创建Dropout层
        else:
            self.bottleneck = TFOutputBottleneck(config, name="bottleneck")  # 如果使用瓶颈层，则创建瓶颈层对象
        self.config = config  # 保存配置信息

    def call(self, hidden_states, residual_tensor_1, residual_tensor_2, training=False):
        hidden_states = self.dense(hidden_states)  # 经过全连接层转换隐藏状态
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)  # 如果不使用瓶颈层，则应用Dropout
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)  # 对输入和残差进行归一化和残差连接
        else:
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)  # 对输入和残差进行归一化和残差连接
            hidden_states = self.bottleneck(hidden_states, residual_tensor_2)  # 经过瓶颈层处理残差
        return hidden_states  # 返回处理后的隐藏状态

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])  # 构建全连接层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)  # 构建归一化层
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)  # 构建瓶颈层


class TFBottleneckLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.intra_bottleneck_size, name="dense")  # 创建瓶颈层的全连接层
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )  # 根据配置选择合适的归一化层
        self.config = config  # 保存配置信息

    def call(self, inputs):
        hidden_states = self.dense(inputs)  # 经过全连接层转换输入
        hidden_states = self.LayerNorm(hidden_states)  # 对转换后的数据进行归一化
        return hidden_states  # 返回处理后的数据

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])  # 构建全连接层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)  # 构建归一化层


class TFBottleneck(keras.layers.Layer):
    # 这里是 TFBottleneck 类的定义，暂时没有额外的代码需要注释
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        # 使用传入的配置信息初始化共享瓶颈层和注意力机制的使用标志
        self.bottleneck_input = TFBottleneckLayer(config, name="input")
        # 如果设置了共享瓶颈层，初始化注意力机制的瓶颈层
        if self.key_query_shared_bottleneck:
            self.attention = TFBottleneckLayer(config, name="attention")

    def call(self, hidden_states):
        # 这个方法可以返回三种不同的元组值。这些不同的值利用了瓶颈层，这些线性层用于将隐藏状态投影到一个低维向量，
        # 从而减少内存使用。这些线性层的权重在训练期间学习。
        #
        # 如果 `config.use_bottleneck_attention` 为真，则会四次返回瓶颈层的结果，
        # 分别用于键、查询、值和“层输入”，供注意力层使用。
        # 这个瓶颈层用于投影隐藏层。这个“层输入”将在计算完注意力分数后，作为注意力自输出中的残差张量使用。
        #
        # 如果不使用 `config.use_bottleneck_attention` 且使用了 `config.key_query_shared_bottleneck`，
        # 则会返回四个值，其中三个经过了瓶颈层处理：查询和键通过同一个瓶颈层，而在注意力自输出中，通过另一个瓶颈层处理残差层。
        #
        # 最后一种情况，查询、键和值的值为未经瓶颈处理的隐藏状态，而残差层则经过了瓶颈处理。

        bottlenecked_hidden_states = self.bottleneck_input(hidden_states)
        # 根据配置决定返回哪些值的元组
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在瓶颈输入层，构建该层
        if getattr(self, "bottleneck_input", None) is not None:
            with tf.name_scope(self.bottleneck_input.name):
                self.bottleneck_input.build(None)
        # 如果存在注意力瓶颈层，构建该层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
# 定义一个 Keras 自定义层 TFMobileBertLayer，继承自 keras.layers.Layer 类
class TFMobileBertLayer(keras.layers.Layer):
    # 初始化方法，接受 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据 config 配置决定是否使用瓶颈结构
        self.use_bottleneck = config.use_bottleneck
        # 存储 feedforward 网络的数量
        self.num_feedforward_networks = config.num_feedforward_networks
        # 创建 TFMobileBertAttention 层，命名为 "attention"
        self.attention = TFMobileBertAttention(config, name="attention")
        # 创建 TFMobileBertIntermediate 层，命名为 "intermediate"
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        # 创建 TFMobileBertOutput 层，命名为 "output"
        self.mobilebert_output = TFMobileBertOutput(config, name="output")

        # 如果使用瓶颈结构，创建 TFBottleneck 层，命名为 "bottleneck"
        if self.use_bottleneck:
            self.bottleneck = TFBottleneck(config, name="bottleneck")
        
        # 如果 feedforward 网络数量大于1，创建多个 TFFFNLayer 层
        if config.num_feedforward_networks > 1:
            # 使用列表推导创建多个 TFFFNLayer 实例，命名为 "ffn.{i}"
            self.ffn = [TFFFNLayer(config, name=f"ffn.{i}") for i in range(config.num_feedforward_networks - 1)]

    # call 方法定义了层的前向传播逻辑
    def call(self, hidden_states):
        # 调用注意力层处理隐藏状态
        attention_output = self.attention(hidden_states)
        # 调用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 调用 MobileBERT 输出层处理中间层输出和原始隐藏状态
        mobilebert_output = self.mobilebert_output(intermediate_output, hidden_states)
        
        # 如果使用瓶颈结构，将输出传入瓶颈层
        if self.use_bottleneck:
            mobilebert_output = self.bottleneck(mobilebert_output)
        
        # 对于每个 feedforward 网络，依次调用处理
        if self.num_feedforward_networks > 1:
            for ffn_layer in self.ffn:
                mobilebert_output = ffn_layer(mobilebert_output)
        
        # 返回处理后的输出
        return mobilebert_output

    # build 方法用于构建层，包括初始化权重等操作
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果存在 intermediate 层，则构建该层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在 MobileBERT 输出层，则构建该层
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)
        
        # 如果使用瓶颈结构，构建瓶颈层
        if self.use_bottleneck and getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        
        # 如果有多个 feedforward 网络，依次构建每个网络层
        if self.num_feedforward_networks > 1:
            for ffn_layer in self.ffn:
                with tf.name_scope(ffn_layer.name):
                    ffn_layer.build(None)
    # 定义一个方法，用于处理网络的前向传播，接受隐藏状态、注意力掩码、头掩码、是否输出注意力权重以及训练标志
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        # 如果使用瓶颈层，调用瓶颈层方法生成查询、键、值张量以及层输入
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            # 否则复制隐藏状态作为查询、键、值张量，同时层输入也设为隐藏状态
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        # 调用注意力层进行注意力计算，传入查询、键、值张量、层输入、注意力掩码、头掩码、是否输出注意力权重以及训练标志
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

        # 从注意力输出中获取注意力张量
        attention_output = attention_outputs[0]
        s = (attention_output,)

        # 如果存在多个前馈网络，则依次对注意力输出进行处理
        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        # 经过中间层处理注意力输出得到中间输出
        intermediate_output = self.intermediate(attention_output)
        # 经过MobileBERT输出层处理中间输出、注意力输出以及隐藏状态，得到层输出
        layer_output = self.mobilebert_output(intermediate_output, attention_output, hidden_states, training=training)

        # 构造最终输出，包括层输出、注意力输出的其它部分以及可能的注意力张量
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
        )  # 如果需要输出注意力权重，则添加进输出中

        # 返回构造好的输出
        return outputs

    # 构建网络层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果注意力层存在，则逐一构建它们
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果中间层存在，则逐一构建它们
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果MobileBERT输出层存在，则逐一构建它们
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)
        # 如果瓶颈层存在，则逐一构建它们
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        # 如果前馈网络存在，则逐一构建它们
        if getattr(self, "ffn", None) is not None:
            for layer in self.ffn:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFMobileBertEncoder(keras.layers.Layer):
    # TFMobileBertEncoder 类定义，继承自 keras 的 Layer 类
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化输出参数的标志
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # 创建多个 TFMobileBertLayer 层组成的列表
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
        # 初始化存储所有隐藏状态和注意力的元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # 遍历所有层并调用它们的 call 方法
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的 call 方法，计算输出
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 决定返回值的形式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 返回 TFBaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和注意力
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFMobileBertPooler(keras.layers.Layer):
    # TFMobileBertPooler 类定义，继承自 keras 的 Layer 类
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据配置决定是否激活分类器的激活函数
        self.do_activate = config.classifier_activation
        if self.do_activate:
            # 如果激活，创建一个全连接层，使用 tanh 激活函数
            self.dense = keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                activation="tanh",
                name="dense",
            )
        self.config = config

    def call(self, hidden_states):
        # 通过获取第一个 token 对应的隐藏状态来实现模型的 "汇聚"
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            # 如果不需要激活，直接返回第一个 token 的隐藏状态
            return first_token_tensor
        else:
            # 否则，通过全连接层处理第一个 token 的隐藏状态
            pooled_output = self.dense(first_token_tensor)
            return pooled_output

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
class TFMobileBertPredictionHeadTransform(keras.layers.Layer):
    # TFMobileBert 模型的预测头变换层，用于处理隐藏状态
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 定义一个全连接层，输出维度为 config.hidden_size，使用指定的初始化方法
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 根据配置选择激活函数，或者直接使用给定的激活函数对象
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        # 创建 LayerNorm 层，用于归一化隐藏状态向量
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    # 定义调用函数，实现层的前向传播
    def call(self, hidden_states):
        # 全连接层处理隐藏状态向量
        hidden_states = self.dense(hidden_states)
        # 应用激活函数变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    # 构建层的方法，用于创建层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则构建 dense 层的权重
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层的权重
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFMobileBertLMPredictionHead(keras.layers.Layer):
    # TFMobileBert 模型的语言模型预测头层
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建预测头变换层对象
        self.transform = TFMobileBertPredictionHeadTransform(config, name="transform")
        self.config = config

    # 构建方法，用于创建层的权重
    def build(self, input_shape=None):
        # 创建偏置项权重，形状为 (config.vocab_size,)
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        # 创建全连接层的权重，形状为 (config.hidden_size - config.embedding_size, config.vocab_size)
        self.dense = self.add_weight(
            shape=(self.config.hidden_size - self.config.embedding_size, self.config.vocab_size),
            initializer="zeros",
            trainable=True,
            name="dense/weight",
        )
        # 创建解码器权重，形状为 (config.vocab_size, config.embedding_size)
        self.decoder = self.add_weight(
            shape=(self.config.vocab_size, self.config.embedding_size),
            initializer="zeros",
            trainable=True,
            name="decoder/weight",
        )

        if self.built:
            return
        self.built = True
        # 如果存在 transform 层，则构建 transform 层的权重
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出的嵌入向量
    def get_output_embeddings(self):
        return self

    # 设置输出的嵌入向量
    def set_output_embeddings(self, value):
        self.decoder = value
        self.config.vocab_size = shape_list(value)[0]

    # 获取偏置项
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏置项
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 定义一个方法，用于处理传入的隐藏状态数据
    def call(self, hidden_states):
        # 调用transform方法，对隐藏状态进行转换处理
        hidden_states = self.transform(hidden_states)
        # 使用矩阵乘法将转换后的隐藏状态与decoder和dense张量的连接进行乘法运算
        hidden_states = tf.matmul(hidden_states, tf.concat([tf.transpose(self.decoder), self.dense], axis=0))
        # 将偏置项加到乘法结果上
        hidden_states = hidden_states + self.bias
        # 返回处理后的隐藏状态数据
        return hidden_states
class TFMobileBertMLMHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化 MLM 预测头部，使用 MobileBertLMPredictionHead 类
        self.predictions = TFMobileBertLMPredictionHead(config, name="predictions")

    def call(self, sequence_output):
        # 调用 predictions 对象进行序列输出的预测评分
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 构建 predictions 对象，传入 None 的输入形状
                self.predictions.build(None)


@keras_serializable
class TFMobileBertMainLayer(keras.layers.Layer):
    config_class = MobileBertConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        # 初始化 MobileBertMainLayer，配置各种属性
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 初始化 MobileBertEmbeddings、MobileBertEncoder 和可选的 MobileBertPooler 层
        self.embeddings = TFMobileBertEmbeddings(config, name="embeddings")
        self.encoder = TFMobileBertEncoder(config, name="encoder")
        self.pooler = TFMobileBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        # 返回嵌入层对象
        return self.embeddings

    def set_input_embeddings(self, value):
        # 设置嵌入层的权重和词汇大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型中的注意力头部，heads_to_prune 参数为要剪枝的头部字典
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
    ):
        # 执行 MobileBertMainLayer 的前向传播，支持参数解包和可选的返回字典模式
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                # 构建 embeddings 对象，传入 None 的输入形状
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                # 构建 encoder 对象，传入 None 的输入形状
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                # 构建 pooler 对象，传入 None 的输入形状
                self.pooler.build(None)


class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    base_model_prefix = "mobilebert"


@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    # TFMobileBert 预训练模型的输出数据结构
    ...
    # 定义一个类似于 Type 注释的多行字符串，描述了 `TFMobileBertForPreTraining` 的输出类型信息
    Output type of [`TFMobileBertForPreTraining`].
    
    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            预测语言建模头部的预测分数（在 SoftMax 之前的每个词汇标记的分数）。
        seq_relationship_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            下一个序列预测（分类）头部的预测分数（在 SoftMax 之前的 True/False 连续性的分数）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `tf.Tensor` 的输出（一个用于嵌入的输出 + 每个层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
    
            模型在每个层输出的隐藏状态以及初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含每个层的 `tf.Tensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    
            注意力 softmax 后的注意力权重，用于在自注意力头部中计算加权平均值。
    
    """
    
    # 定义可选的损失张量
    loss: tf.Tensor | None = None
    # 定义预测语言建模头部的预测分数张量
    prediction_logits: tf.Tensor = None
    # 定义下一个序列预测头部的预测分数张量
    seq_relationship_logits: tf.Tensor = None
    # 定义隐藏状态的元组张量，可选返回，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义注意力权重的元组张量，可选返回，当 `output_attentions=True` 或 `config.output_attentions=True` 时返回
    attentions: Tuple[tf.Tensor] | None = None
"""
    This model inherits from `TFPreTrainedModel`. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a `keras.Model` subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0
    documentation for all matters related to general usage and behavior.

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

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config (`MobileBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the `PreTrainedModel.from_pretrained` method to load the model weights.
"""

"""
    The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits the documentation from `MOBILEBERT_START_DOCSTRING`, which provides detailed information about
    its usage with TensorFlow 2.0, input formats, and integration with Keras.

    Parameters:
        *inputs: Variable length input arguments to allow flexible input formats as described in the `MOBILEBERT_START_DOCSTRING`.
        **kwargs: Additional keyword arguments passed to the superclass constructor.
"""

@add_start_docstrings(
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertModel(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # Instantiate the core MobileBERT main layer with the provided configuration
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")

    @unpack_inputs
    # 将模型的前向方法文档化，添加关于输入参数的说明，使用装饰器实现
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，包括模型的检查点、输出类型、配置类等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
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
        # 调用 MobileBERT 模型的前向方法，传递各种输入参数
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
        # 返回模型前向传播的输出结果
        return outputs

    # 构建模型的方法，用于初始化模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 设置模型已构建的标志为 True
        self.built = True
        # 如果存在 MobileBERT 模型，则在命名空间下构建该模型
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
"""
MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
`next sentence prediction (classification)` head.
"""
# 定义 TFMobileBertForPreTraining 类，继承自 TFMobileBertPreTrainedModel 和 TFMobileBertPreTrainingLoss
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel, TFMobileBertPreTrainingLoss):

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFMobileBertMainLayer 实例，并命名为 'mobilebert'
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        # 创建 TFMobileBertMLMHead 实例，并命名为 'predictions'
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")
        # 创建 TFMobileBertOnlyNSPHead 实例，并命名为 'seq_relationship'
        self.seq_relationship = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    def get_lm_head(self):
        # 返回 predictions 的预测结果
        return self.predictions.predictions

    def get_prefix_bias_name(self):
        # 发出警告，指示 get_prefix_bias_name 方法已过时，建议使用 'get_bias' 方法代替
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回包含预测名称路径的字符串
        return self.name + "/" + self.predictions.name + "/" + self.predictions.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFMobileBertForPreTrainingOutput]:
        r"""
        返回类型注释，此函数返回一个元组或者 TFMobileBertForPreTrainingOutput 对象。

        示例：

        ```
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")
        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]
        ```

        执行模型的前向传播，生成预测分数和序列关系分数。

        Parameters:
        - input_ids (tf.Tensor): 输入的 token IDs
        - attention_mask (Optional[tf.Tensor]): 注意力掩码
        - token_type_ids (Optional[tf.Tensor]): token 类型 IDs
        - position_ids (Optional[tf.Tensor]): 位置 IDs
        - head_mask (Optional[tf.Tensor]): 头部掩码
        - inputs_embeds (Optional[tf.Tensor]): 输入嵌入
        - output_attentions (Optional[bool]): 是否输出注意力
        - output_hidden_states (Optional[bool]): 是否输出隐藏状态
        - return_dict (Optional[bool]): 是否以字典形式返回结果
        - training (Optional[bool]): 是否处于训练模式

        Returns:
        - 如果 return_dict=False，则返回一个元组 (total_loss, prediction_scores, seq_relationship_scores, hidden_states, attentions) 或者 (prediction_scores, seq_relationship_scores, hidden_states, attentions)。
        - 如果 return_dict=True，则返回一个 TFMobileBertForPreTrainingOutput 对象，包含 loss, prediction_logits, seq_relationship_logits, hidden_states, attentions 字段。

        Raises:
        - 无异常抛出。

        """
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

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            d_labels = {"labels": labels}
            d_labels["next_sentence_label"] = next_sentence_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TFMobileBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build(None)

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == "cls.predictions.decoder.weight":
            return tf_weight, "mobilebert.embeddings.word_embeddings.weight"
        else:
            return (tf_weight,)
# 使用装饰器为类添加文档字符串，描述该类是带有顶部语言建模头的 MobileBert 模型
@add_start_docstrings("""MobileBert Model with a `language modeling` head on top.""", MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMaskedLM(TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在从 PT 模型加载 TF 模型时，忽略特定名称的层
    # 包含'.'的名称表示在加载时是授权的意外/丢失层
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"seq_relationship___cls",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 MobileBert 主层，不添加池化层，命名为"mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        # 初始化 MobileBert 的语言建模头，命名为"predictions___cls"
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")

    # 返回语言建模头的预测部分
    def get_lm_head(self):
        return self.predictions.predictions

    # 返回前缀偏置名称，该方法已弃用，将来将使用`get_bias`替代
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    # 使用装饰器为前向传播方法添加文档字符串，描述输入参数和预期输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.57,
    )
    # 前向传播方法，接收多个输入参数，返回模型输出或损失
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
    ) -> Union[Tuple, TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels
        """
        # 使用 `->` 表示函数的返回类型注解，这里返回的是一个元组或者 TFMaskedLMOutput 对象
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
        # 获取模型输出的序列输出，通常是模型的第一个输出
        sequence_output = outputs[0]
        # 将序列输出传入预测模块，生成预测得分
        prediction_scores = self.predictions(sequence_output, training=training)

        # 如果没有提供标签，则损失设为 None；否则计算预测损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果 return_dict 为 False，则返回的输出包括预测得分和可能的额外输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFMaskedLMOutput 对象，包括损失、预测得分、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在 mobilebert 模型，则构建其子模块
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在 predictions 模型，则构建其子模块
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)

    def tf_to_pt_weight_rename(self, tf_weight):
        # 将特定的 TensorFlow 权重名称映射为 PyTorch 权重名称
        if tf_weight == "cls.predictions.decoder.weight":
            return tf_weight, "mobilebert.embeddings.word_embeddings.weight"
        else:
            return (tf_weight,)
# MobileBert 只有下一句预测（NSP）头部的层定义
class TFMobileBertOnlyNSPHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层用于下一句预测，输出维度为2，命名为"seq_relationship"
        self.seq_relationship = keras.layers.Dense(2, name="seq_relationship")
        # 保存配置信息
        self.config = config

    def call(self, pooled_output):
        # 通过全连接层计算序列关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回；否则构建全连接层，输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """MobileBert 模型，顶部带有`下一句预测（分类）`头部。""",
    MOBILEBERT_START_DOCSTRING,
)
# TFMobileBertForNextSentencePrediction 继承自 TFMobileBertPreTrainedModel 和 TFNextSentencePredictionLoss
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):
    # 当从 PT 模型加载 TF 模型时，命名中带有'.'的层表示可接受的未预期/缺失层
    _keys_to_ignore_on_load_unexpected = [r"predictions___cls", r"cls.predictions"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建 MobileBert 主层，命名为"mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        # 创建仅有下一句预测头部的层，命名为"seq_relationship___cls"
        self.cls = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    # 解压输入参数
    # 添加模型前向传播的文档字符串
    # 替换返回文档字符串，输出类型为 TFNextSentencePredictorOutput，配置类为 _CONFIG_FOR_DOC
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
        返回模型的输出结果或损失值。

        Examples:

        ```
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        ```"""

        # 调用 MobileBERT 模型来进行预测
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

        # 提取池化后的输出
        pooled_output = outputs[1]

        # 将池化输出传入分类层，得到下一个句子关系的分数
        seq_relationship_scores = self.cls(pooled_output)

        # 计算下一个句子关系的损失值
        next_sentence_loss = (
            None
            if next_sentence_label is None
            else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
        )

        # 如果不要求返回字典，则组装输出
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 返回 TFNextSentencePredictorOutput 对象，包含损失值、分数、隐藏状态和注意力权重
        return TFNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True

        # 如果模型已经构建，则直接返回
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)

        # 如果分类层已经存在，则构建分类层
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
# 使用装饰器为类添加文档字符串，描述了 MobileBert 模型的用途，特别是在顶部增加了一个线性层用于序列分类或回归任务，例如 GLUE 任务
@add_start_docstrings(
    """
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForSequenceClassification(TFMobileBertPreTrainedModel, TFSequenceClassificationLoss):
    # 当从 PyTorch 模型加载到 TF 模型时，忽略的层名列表，包括预期未找到或多余的层
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    # 当从 PyTorch 模型加载到 TF 模型时，忽略的缺失层名列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数初始化模型配置
        super().__init__(config, *inputs, **kwargs)
        # 设定模型输出的类别数目
        self.num_labels = config.num_labels

        # 创建 MobileBert 主层，使用给定的配置，命名为 "mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        # 根据配置设定分类器的 dropout 率，如果未指定，则使用隐藏层 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个 dropout 层，应用于分类器
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 创建一个全连接层作为分类器，设定输出类别数，使用指定范围的初始化器初始化权重
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config

    # 使用装饰器定义模型的前向传播函数，并添加详细的文档字符串描述其输入参数和预期输出
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
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 MobileBERT 模型进行前向传播，获取输出结果
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
        # 从 MobileBERT 输出中获取池化后的表示，用于分类器输入
        pooled_output = outputs[1]

        # 对池化后的表示应用 dropout，用于模型训练时的正则化
        pooled_output = self.dropout(pooled_output, training=training)
        # 将池化后的表示输入分类器，得到预测 logits
        logits = self.classifier(pooled_output)

        # 如果存在标签，则计算损失；否则损失置为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回 dict 格式的结果，则按照 tuple 形式返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 格式的结果，包括损失、预测 logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在 MobileBERT 模型，则构建 MobileBERT
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在分类器，则构建分类器，指定输入形状为 [None, None, 隐藏层大小]
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用特定的文档字符串为类添加描述信息，说明这是一个在移动端BERT模型基础上构建的用于抽取式问答任务（如SQuAD）的模型
@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForQuestionAnswering(TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss):
    # 在从PyTorch模型加载为TensorFlow模型时，以下层的名称中包含'.'，表示这些层可以忽略或出现未预期的情况
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 创建MobileBERT主层，不添加池化层，命名为"mobilebert"
        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        
        # 创建用于问答任务输出的全连接层，输出大小为config.num_labels，使用指定的初始化器初始化权重，命名为"qa_outputs"
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        
        # 保存配置信息
        self.config = config

    # 使用特定的装饰器为call方法添加文档字符串，描述模型前向传播的输入和输出
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
        # 调用 MobileBERT 模型进行推断，获取模型输出
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
        # 获取模型输出的序列输出部分
        sequence_output = outputs[0]

        # 通过输出序列计算问答任务的 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 沿着最后一个维度分割为 start_logits 和 end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除 start_logits 和 end_logits 的最后一个维度，使其形状变为 (batch_size,)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果提供了 start_positions 和 end_positions，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果 return_dict=False，返回扩展的输出元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，返回 TFQuestionAnsweringModelOutput 对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 MobileBERT 模型，则构建 MobileBERT
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在 QA 输出层，则构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
"""
MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.
"""
# 定义 TFMobileBertForMultipleChoice 类，用于在 MobileBert 模型基础上添加多选分类头部的功能
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):

    # 当从 PT 模型加载 TF 模型时，以下名称表示可以忽略的未预期/丢失的层
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    # 当从 PT 模型加载 TF 模型时，以下名称表示可以忽略的缺失的层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFMobileBertMainLayer 实例作为模型的主体部分
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        
        # 创建 Dropout 层，使用配置中指定的隐藏层 dropout 概率
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        
        # 创建分类器 Dense 层，用于多选分类，输出维度为1
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # 存储模型配置
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
    # 定义模型的前向传播方法
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
        # 如果提供了 input_ids，则获取 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，从 inputs_embeds 中获取 num_choices 和 seq_length
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 和相关的张量重新整形为二维张量
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        
        # 调用 MobileBERT 模型进行前向传播
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
        
        # 获取池化后的输出，并应用 dropout
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        
        # 使用分类器得到 logits
        logits = self.classifier(pooled_output)
        
        # 重新整形 logits 为二维张量
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果提供了 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不需要返回字典形式的输出，则返回 reshaped_logits 和其他输出项
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则构建 TFMultipleChoiceModelOutput 对象
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
        
        # 标记模型为已建立状态
        self.built = True
        
        # 如果存在 MobileBERT 模型，则构建其结构
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        
        # 如果存在分类器，则构建其结构，输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.
"""
@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForTokenClassification(TFMobileBertPreTrainedModel, TFTokenClassificationLoss):
    """
    Subclass of TFMobileBertPreTrainedModel and TFTokenClassificationLoss for token classification tasks,
    incorporating MobileBert architecture.
    """
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    # List of keys to ignore when certain layers are missing during model loading
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        """
        Initialize TFMobileBertForTokenClassification model.

        Args:
            config (MobileBertConfig): Configuration object specifying model parameters.
            *inputs: Variable length argument list for additional inputs.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # Initialize MobileBertMainLayer without pooling layer for token classification
        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        
        # Set dropout rate for classifier layer based on config
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        
        # Linear classification layer for token classification
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # Store the configuration object for reference
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
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
        **kwargs,
    ):
        """
        Perform forward pass of TFMobileBertForTokenClassification model.

        Args:
            input_ids (TFModelInputType, optional): Tensor of input token IDs.
            attention_mask (np.ndarray or tf.Tensor, optional): Tensor of attention masks.
            token_type_ids (np.ndarray or tf.Tensor, optional): Tensor of token type IDs.
            position_ids (np.ndarray or tf.Tensor, optional): Tensor of position IDs.
            head_mask (np.ndarray or tf.Tensor, optional): Tensor of head masks.
            inputs_embeds (np.ndarray or tf.Tensor, optional): Tensor of input embeddings.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            labels (np.ndarray or tf.Tensor, optional): Tensor of labels for token classification.
            training (bool, optional): Whether in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TFTokenClassifierOutput or dict: Output of the model.
        """
        # Forward pass through MobileBert model
        return self.mobilebert(
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
            labels=labels,
            **kwargs,
        )
    ) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 MobileBERT 模型进行推断或训练，获取输出结果
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
        # 获取 MobileBERT 模型的序列输出
        sequence_output = outputs[0]

        # 对序列输出应用 dropout，用于防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 dropout 后的输出传入分类器，生成分类 logits
        logits = self.classifier(sequence_output)

        # 如果存在标签，计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回字典，返回分类 logits 和可能的附加输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，直接返回
        if self.built:
            return
        # 设置构建状态为已完成
        self.built = True
        # 如果存在 MobileBERT 模型，构建 MobileBERT
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        # 如果存在分类器模型，构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
```