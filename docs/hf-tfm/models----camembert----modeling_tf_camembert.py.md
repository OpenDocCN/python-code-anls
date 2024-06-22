# `.\transformers\models\camembert\modeling_tf_camembert.py`

```py
# coding=utf-8
# 代码文件的编码声明，使用 UTF-8 编码
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# 版权声明，版权归属于 Google AI Language Team 和 HuggingFace Inc. 团队
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，版权归属于 NVIDIA 公司，保留所有权利
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
# Apache 2.0 开源许可证声明，允许在遵守许可证的前提下使用该代码
# 获得许可证的副本，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何形式的担保或条件，无论是明示的还是暗示的。
# 有关特定语言的权限，请参阅许可证
""" TF 2.0 CamemBERT model."""
# TF 2.0 CamemBERT 模型的声明
from __future__ import annotations
# 导入未来的注释语法以支持类型注释

import math
# 导入数学库
import warnings
# 导入警告库
from typing import Optional, Tuple, Union
# 导入类型提示所需的模块

import numpy as np
# 导入 numpy 库
import tensorflow as tf
# 导入 TensorFlow 库

from ...activations_tf import get_tf_activation
# 导入激活函数相关模块
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
# 导入模型输出相关的模块
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
# 导入 TensorFlow 模型相关的工具函数和损失函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 导入 TensorFlow 相关的工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 导入工具函数和文档注释相关的模块
from .configuration_camembert import CamembertConfig
# 导入 Camembert 模型的配置类

logger = logging.get_logger(__name__)
# 获取 logger 对象

_CHECKPOINT_FOR_DOC = "camembert-base"
# 用于文档的检查点名称
_CONFIG_FOR_DOC = "CamembertConfig"
# 用于文档的配置名称

TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all CamemBERT models at https://huggingface.co/models?filter=camembert
]
# 预训练模型的存档列表

CAMEMBERT_START_DOCSTRING = r"""

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



# Camembert 模型的文档字符串，提供了一些使用提示
    # 第二种格式被支持的原因是，Keras 方法在将输入传递给模型和层时更喜欢这种格式。由于这种支持，当使用诸如 `model.fit()` 这样的方法时，只需将输入和标签以 `model.fit()` 支持的任何格式传递即可！但是，如果您想在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可以用来收集第一个位置参数中的所有输入张量：

    # - 仅具有 `input_ids` 的单个张量，没有其他内容：`model(input_ids)`
    # - 长度不定的列表，其中包含一个或多个输入张量，按照文档字符串中给定的顺序：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 一个字典，其中包含一个或多个与文档字符串中给定的输入名称相关联的输入张量：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    # 请注意，当使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，您无需担心这些，因为您可以像将输入传递给任何其他 Python 函数一样传递输入！

    # 参数：
    #     config ([`CamembertConfig`]): 包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

CAMEMBERT_INPUTS_DOCSTRING = r"""
"""

# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaEmbeddings 复制过来的类
class TFCamembertEmbeddings(tf.keras.layers.Layer):
    """
    与 BertEmbeddings 相同，只是对位置嵌入索引进行了微小调整。
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 1
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 添加权重矩阵用于词嵌入
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 添加权重矩阵用于标记类型嵌入
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 添加权重矩阵用于位置嵌入
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        用输入的 ID 替换非填充符号的位置数字。位置数字从 padding_idx+1 开始。忽略填充符号。这是从 fairseq 的 `utils.make_positions` 修改而来。

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        training=False,
    ):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 确保输入张量不同时为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果提供了输入的 token IDs，则从权重张量中获取对应的嵌入向量
        if input_ids is not None:
            # 检查输入的 token IDs 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入张量的形状，去掉最后一个维度（即 batch 维度）
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 token 类型 ID，则填充为 0
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果未提供位置 ID，则根据输入的 token IDs 创建位置 ID
        if position_ids is None:
            if input_ids is not None:
                # 从输入的 token IDs 创建位置 ID。任何填充的 token 仍保持填充状态。
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 如果未提供输入的 token IDs，则根据 padding_idx 创建位置 ID
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        # 根据位置 ID 获取位置嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token 类型 ID 获取 token 类型嵌入
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 计算最终的嵌入向量，包括输入嵌入、位置嵌入和 token 类型嵌入
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的嵌入向量进行 Layer Normalization
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终的嵌入向量进行 dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入向量
        return final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制并修改为CamembertPooler
class TFCamembertPooler(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化操作
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过仅仅取第一个标记对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在self.dense，则构建self.dense
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertSelfAttention复制并修改为CamembertSelfAttention
class TFCamembertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 如果hidden_size不能被num_attention_heads整除，则抛出异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建用于self-attention操作的查询、键和值的全连接层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config
    # 将输入张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
    tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

    # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
    return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 模型的调用方法，接受多个输入参数，包括隐藏状态、注意力掩码、头部掩码、编码器隐藏状态、编码器注意力掩码、过去的键值对、是否输出注意力矩阵、是否训练
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
    # 构建方法，用于初始化模型参数，检查是否已经构建，如果已经构建则直接返回，否则初始化模型参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 检查是否存在查询参数，并构建查询参数
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 检查是否存在键参数，并构建键参数
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 检查是否存在值参数，并构建值参数
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 从transformers.models.bert.modeling_tf_bert.TFBertSelfOutput复制代码并将Bert->Camembert
class TFCamembertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，用于对隐藏状态进行归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，用于对隐藏状态进行随机丢弃
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置参数
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对隐藏状态应用全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对隐藏状态进行随机丢弃
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将原始输入与变换后的隐藏状态相加，并进行LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过dense层，则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建dense层
                self.dense.build([None, None, self.config.hidden_size])
        # 如果已经构建过LayerNorm层，则直接返回
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建LayerNorm层
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertAttention复制代码并将Bert->Camembert
class TFCamembertAttention(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建Camembert自注意力层
        self.self_attention = TFCamembertSelfAttention(config, name="self")
        # 创建Camembert自注意力输出层
        self.dense_output = TFCamembertSelfOutput(config, name="output")

    def prune_heads(self, heads):
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
        # 调用自注意力层
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
        # 调用自注意力输出层
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力权重，则将其加入输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 根据输入形状构建模型
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为 True
        self.built = True
        # 如果存在自注意力层，则构建自注意力层
        if getattr(self, "self_attention", None) is not None:
            # 在命名空间下构建自注意力层
            with tf.name_scope(self.self_attention.name):
                # 构建自注意力层，传入 None 作为输入形状
                self.self_attention.build(None)
        # 如果存在密集输出层，则构建密集输出层
        if getattr(self, "dense_output", None) is not None:
            # 在命名空间下构建密集输出层
            with tf.name_scope(self.dense_output.name):
                # 构建密集输出层，传入 None 作为输入形状
                self.dense_output.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制并将Bert->Camembert
class TFCamembertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units为config中的intermediate_size，kernel_initializer为config中的initializer_range，名称为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config中的hidden_act是字符串，则使用get_tf_activation函数获取激活函数，否则直接使用config中的hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 对输入的hidden_states进行处理，先通过全连接层dense，再通过激活函数intermediate_act_fn
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层，设置dense层的输入形状为[None, None, self.config.hidden_size]
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertOutput复制并将Bert->Camembert
class TFCamembertOutput(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units为config中的hidden_size，kernel_initializer为config中的initializer_range，名称为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建LayerNormalization层，epsilon为config中的layer_norm_eps，名称为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建Dropout层，rate为config中的hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 对输入的hidden_states进行处理，先通过全连接层dense，再通过Dropout层，最后通过LayerNormalization层
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建层，设置dense层的输入形状为[None, None, self.config.intermediate_size]，LayerNorm层的输入形状为[None, None, self.config.hidden_size]
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertLayer复制并将Bert->Camembert
class TFCamembertLayer(tf.keras.layers.Layer):
    # 初始化方法，接受一个CamembertConfig对象和其他关键字参数
    def __init__(self, config: CamembertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化注意力层对象，使用CamembertAttention类，设置名称为"attention"
        self.attention = TFCamembertAttention(config, name="attention")
        
        # 判断是否为解码器模型
        self.is_decoder = config.is_decoder
        
        # 判断是否添加跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        
        # 如果添加了跨注意力机制
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出值错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 初始化跨注意力层对象，使用CamembertAttention类，设置名称为"crossattention"
            self.crossattention = TFCamembertAttention(config, name="crossattention")
        
        # 初始化中间层对象，使用CamembertIntermediate类，设置名称为"intermediate"
        self.intermediate = TFCamembertIntermediate(config, name="intermediate")
        
        # 初始化BERT输出层对象，使用CamembertOutput类，设置名称为"output"
        self.bert_output = TFCamembertOutput(config, name="output")

    # 调用方法，接受一系列输入张量和参数
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器隐藏状态张量或None
        encoder_attention_mask: tf.Tensor | None,  # 编码器注意力掩码张量或None
        past_key_value: Tuple[tf.Tensor] | None,  # 先前键值元组或None
        output_attentions: bool,  # 是否输出注意力张量
        training: bool = False,  # 是否处于训练模式，默认为False
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键/值存在，则将自注意力过去的键/值元组放在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
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
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组在过去的键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行交叉注意力计算
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
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到现在的键/值元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果输出注意力，则添加它们

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建方法用于构建模型的层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在注意力层，则构建注意力层
        if getattr(self, "attention", None) is not None:
            # 在 TensorFlow 中使用名称范围管理器，命名注意力层
            with tf.name_scope(self.attention.name):
                # 构建注意力层，input_shape 为 None 表示不指定输入形状
                self.attention.build(None)
        # 如果存在中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            # 在 TensorFlow 中使用名称范围管理器，命名中间层
            with tf.name_scope(self.intermediate.name):
                # 构建中间层，input_shape 为 None 表示不指定输入形状
                self.intermediate.build(None)
        # 如果存在 BERT 输出层，则构建 BERT 输出层
        if getattr(self, "bert_output", None) is not None:
            # 在 TensorFlow 中使用名称范围管理器，命名 BERT 输出层
            with tf.name_scope(self.bert_output.name):
                # 构建 BERT 输出层，input_shape 为 None 表示不指定输入形状
                self.bert_output.build(None)
        # 如果存在交叉注意力层，则构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            # 在 TensorFlow 中使用名称范围管理器，命名交叉注意力层
            with tf.name_scope(self.crossattention.name):
                # 构建交叉注意力层，input_shape 为 None 表示不指定输入形状
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制并修改为 CamembertEncoder 类
class TFCamembertEncoder(tf.keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建 CamembertLayer 实例的列表，数量为 config 中指定的隐藏层数
        self.layer = [TFCamembertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,  # 编码器的隐藏状态张量或 None
        encoder_attention_mask: tf.Tensor | None,  # 编码器的注意力掩码张量或 None
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,  # 过去的键值对元组或 None
        use_cache: Optional[bool],  # 是否使用缓存的标志
        output_attentions: bool,  # 是否输出注意力张量的标志
        output_hidden_states: bool,  # 是否输出隐藏状态的标志
        return_dict: bool,  # 是否返回字典格式的输出
        training: bool = False,  # 是否处于训练模式的标志，默认为 False
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:  # 返回值类型注解
        # 初始化存储所有隐藏状态、注意力张量和交叉注意力张量的元组，若不输出则为 None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 若使用缓存，则初始化下一个解码器缓存的元组，否则为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个 CamembertLayer 实例
        for i, layer_module in enumerate(self.layer):
            # 若输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取过去的键值对，若为 None 则赋值为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的 call 方法，获取当前层的输出
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
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 若使用缓存，则将当前层的缓存添加到下一个解码器缓存元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 若输出注意力张量，则将当前层的注意力张量添加到所有注意力张量元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 若配置中添加交叉注意力且编码器隐藏状态不为空，则将当前层的交叉注意力张量添加到所有交叉注意力张量元组中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 若不返回字典，则返回非空的输出元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回字典格式的输出
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 构建模型的方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 检查是否存在层属性
        if getattr(self, "layer", None) is not None:
            # 遍历每一层
            for layer in self.layer:
                # 使用层的名称创建命名空间
                with tf.name_scope(layer.name):
                    # 构建当前层，input_shape设置为None
                    layer.build(None)
# 使用 keras_serializable 装饰器标记该类为可序列化的 Keras 模型
@keras_serializable
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaMainLayer 复制并修改为 Camembert
class TFCamembertMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 CamembertConfig
    config_class = CamembertConfig

    # 初始化方法，接受配置参数和是否添加池化层的标志
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将配置信息保存到对象属性中
        self.config = config
        self.is_decoder = config.is_decoder
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 创建 CamembertEncoder 对象，命名为 "encoder"
        self.encoder = TFCamembertEncoder(config, name="encoder")
        # 如果指定添加池化层，则创建 TFCamembertPooler 对象，命名为 "pooler"，否则设为 None
        self.pooler = TFCamembertPooler(config, name="pooler") if add_pooling_layer else None
        # 创建 TFCamembertEmbeddings 对象，命名为 "embeddings"，必须放在最后一行以保持权重顺序
        self.embeddings = TFCamembertEmbeddings(config, name="embeddings")

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 复制的方法，获取输入嵌入层
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 复制的方法，设置输入嵌入层
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 复制的方法，剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer 复制的方法，模型调用时的逻辑
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 如果模型已经构建完成，则直接返回，避免重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在编码器（encoder）属性，则构建编码器
        if getattr(self, "encoder", None) is not None:
            # 使用编码器的名称作为命名空间，并构建编码器
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化器（pooler）属性，则构建池化器
        if getattr(self, "pooler", None) is not None:
            # 使用池化器的名称作为命名空间，并构建池化器
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 如果存在嵌入层（embeddings）属性，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 使用嵌入层的名称作为命名空间，并构建嵌入层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
class TFCamembertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为CamembertConfig
    config_class = CamembertConfig
    # 设置基础模型前缀为"roberta"
    base_model_prefix = "roberta"


@add_start_docstrings(
    "The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaModel复制而来，将Roberta->Camembert, ROBERTA->CAMEMBERT
class TFCamembertModel(TFCamembertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化TFCamembertMainLayer作为self.roberta
        self.roberta = TFCamembertMainLayer(config, name="roberta")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
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
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        outputs = self.roberta(
            input_ids=input_ids,  # 输入的标记化输入 ID
            attention_mask=attention_mask,  # 注意力遮罩，标记化输入的 mask
            token_type_ids=token_type_ids,  # 标记化输入的 token 类型 ID
            position_ids=position_ids,  # 标记化输入的位置 ID
            head_mask=head_mask,  # 注意力头的遮罩
            inputs_embeds=inputs_embeds,  # 输入的嵌入
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力遮罩
            past_key_values=past_key_values,  # 用于加速解码的预计算密钥和值隐藏状态
            use_cache=use_cache,  # 是否使用缓存以加速解码
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
            training=training,  # 是否在训练过程中
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:  # 如果已经建立模型，直接返回
            return
        self.built = True  # 将模型标记为已建立
        if getattr(self, "roberta", None) is not None:  # 如果模型存在
            with tf.name_scope(self.roberta.name):  # 使用模型的名称创建命名空间
                self.roberta.build(None)  # 构建模型
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead复制并修改为TFCamembertLMHead
class TFCamembertLMHead(tf.keras.layers.Layer):
    """用于Camembert的遮罩语言建模的头部。"""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个令牌都有一个仅输出的偏置。
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 为每个词汇的输出添加偏置
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # 使用偏置将大小投影回词汇的大小
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings(
    """具有顶部`语言建模`头的CamemBERT模型。""",
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM复制并修改为TFCamembertForMaskedLM，ROBERTA->CAMEMBERT
class TFCamembertForMaskedLM(TFCamembertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 具有'.'的名称表示在从PT模型加载TF模型时，预期的/丢失的层
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    # 初始化方法，接受配置和输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 RoBERTa 主层对象，不添加池化层
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 创建 RoBERTa 语言模型头部对象
        self.lm_head = TFCamembertLMHead(config, self.roberta.embeddings, name="lm_head")

    # 获取语言模型头部对象
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称
    def get_prefix_bias_name(self):
        # 发出警告，该方法已弃用，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 调用方法，接受多种输入参数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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
    def call(self,
             input_ids: tf.Tensor,
             attention_mask: tf.Tensor,
             token_type_ids: Optional[tf.Tensor] = None,
             position_ids: Optional[tf.Tensor] = None,
             head_mask: Optional[tf.Tensor] = None,
             inputs_embeds: Optional[tf.Tensor] = None,
             output_attentions: Optional[bool] = None,
             output_hidden_states: Optional[bool] = None,
             return_dict: Optional[bool] = None,
             training: Optional[bool] = None,
             labels: Optional[tf.Tensor] = None
             ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 RoBERTa 模型进行推理
        outputs = self.roberta(
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

        # 从 RoBERTa 输出中提取序列输出
        sequence_output = outputs[0]
        # 通过 LM 头进行预测
        prediction_scores = self.lm_head(sequence_output)

        # 计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不要求返回字典，则返回元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有 Masked LM 输出的字典
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建网络结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 构建 LM 头
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaClassificationHead复制得到
class TFCamembertClassificationHead(tf.keras.layers.Layer):
    """句子级别分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于句子的分类任务
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 如果设置了classifier_dropout，则使用该值，否则使用hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个Dropout层，用于正则化
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 创建一个全连接层，用于将特征转换为输出标签
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        # 保存配置信息
        self.config = config

    def call(self, features, training=False):
        # 取出特征的第一个token（相当于[CLS]）
        x = features[:, 0, :]
        # 使用Dropout层对特征进行正则化
        x = self.dropout(x, training=training)
        # 将特征传入全连接层进行线性变换
        x = self.dense(x)
        # 再次使用Dropout层进行正则化
        x = self.dropout(x, training=training)
        # 将线性变换后的结果传入全连接层得到最终输出
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层已经构建，则跳过
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建dense层
                self.dense.build([None, None, self.config.hidden_size])
        # 如果out_proj层已经构建，则跳过
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建out_proj层
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    带有顶部序列分类/回归头部的CamemBERT模型变换器（池化输出的线性层）例如用于GLUE任务。
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForSequenceClassification复制得到，将Roberta->Camembert, ROBERTA->CAMEMBERT
class TFCamembertForSequenceClassification(TFCamembertPreTrainedModel, TFSequenceClassificationLoss):
    # 当从PT模型加载TF模型时，带有'.'的名称表示授权的未预期/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 类别数
        self.num_labels = config.num_labels

        # Camembert主层
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 分类头部
        self.classifier = TFCamembertClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 RoBERTa 模型进行推理
        outputs = self.roberta(
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
        # 获取 RoBERTa 模型的输出序列
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output, training=training)

        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典，则返回输出元组
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
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 RoBERTa 模型存在，则构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 为 CamemBERT 模型添加一个顶部的标记分类头部（在隐藏状态输出的顶部添加一个线性层），例如用于命名实体识别（NER）任务
# 这是一个类装饰器，用于添加文档字符串
@add_start_docstrings(
    """
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForTokenClassification 复制并修改为 CamemBERT，ROBERTA->CAMEMBERT
class TFCamembertForTokenClassification(TFCamembertPreTrainedModel, TFTokenClassificationLoss):
    # 在加载 TF 模型时忽略的预期不匹配的层的名称
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    # 在加载 TF 模型时忽略的缺失的层的名称
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设定标签数量
        self.num_labels = config.num_labels

        # 初始化 CamemBERT 主层
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 初始化分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置
        self.config = config

    # 模型调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-large-ner-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 RoBERTa 模型进行前向传播，获取输出
        outputs = self.roberta(
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
        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output, training=training)
        # 将序列输出传入分类器，获取分类器的 logits
        logits = self.classifier(sequence_output)

        # 计算损失，如果 labels 不为 None，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回结果元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型已构建标志
        self.built = True
        # 如果存在 RoBERTa 模型，则构建 RoBERTa
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 导入必要的库和模块
@add_start_docstrings(
    """
    在 CamemBERT 模型的顶部添加了一个用于多选分类的分类头部（一个线性层位于汇总输出之上，然后是 softmax 层），例如用于 RocStories/SWAG 任务。
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 定义 TFCamembertForMultipleChoice 类，继承自 TFCamembertPreTrainedModel 和 TFMultipleChoiceLoss 类
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForMultipleChoice 复制而来，将 Roberta->Camembert，ROBERTA->CAMEMBERT
class TFCamembertForMultipleChoice(TFCamembertPreTrainedModel, TFMultipleChoiceLoss):
    # 在加载 TF 模型时忽略的未预期/缺失的层的键列表，包括一个正则表达式
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    # 在加载 TF 模型时忽略的缺失的层的键列表，包括一个正则表达式
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 实例化 TFCamembertMainLayer 类，命名为 "roberta"
        self.roberta = TFCamembertMainLayer(config, name="roberta")
        # 添加一个 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 添加一个全连接层
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 设置配置属性
        self.config = config

    # 定义模型的前向传播过程
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
    def call(
        self, inputs: Dict[str, Union[tf.Tensor, Any]],  # 定义call方法，接受输入的张量字典
        attention_mask: Optional[tf.Tensor] = None,  # 注意力掩码张量，默认为None
        token_type_ids: Optional[tf.Tensor] = None,  # 标记类型ID张量，默认为None
        position_ids: Optional[tf.Tensor] = None,  # 位置ID张量，默认为None
        head_mask: Optional[tf.Tensor] = None,  # 头掩码张量，默认为None
        inputs_embeds: Optional[tf.Tensor] = None,  # 输入嵌入张量，默认为None
        labels: Optional[tf.Tensor] = None,  # 标签张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        training: bool = False,  # 是否处于训练模式，默认为False
        **kwargs,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:  # 返回值类型注解
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):  # 标签张量的形状和说明
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        if input_ids is not None:  # 如果存在input_ids张量
            num_choices = shape_list(input_ids)[1]  # 计算选择数量
            seq_length = shape_list(input_ids)[2]  # 计算序列长度
        else:  # 否则
            num_choices = shape_list(inputs_embeds)[1]  # 计算选择数量
            seq_length = shape_list(inputs_embeds)[2]  # 计算序列长度

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None  # 将input_ids张量展平
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None  # 将attention_mask张量展平
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None  # 将token_type_ids张量展平
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None  # 将position_ids张量展平
        outputs = self.roberta(  # 使用RoBERTa模型处理输入张量
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]  # 获取池化输出
        pooled_output = self.dropout(pooled_output, training=training)  # 对池化输出进行dropout
        logits = self.classifier(pooled_output)  # 使用分类器预测标签
        reshaped_logits = tf.reshape(logits, (-1, num_choices))  # 调整logits的形状

        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)  # 计算损失，如果没有标签则损失为None

        if not return_dict:  # 如果不返回字典
            output = (reshaped_logits,) + outputs[2:]  # 构建输出元组
            return ((loss,) + output) if loss is not None else output  # 返回带有损失的输出元组或者仅输出元组

        return TFMultipleChoiceModelOutput(  # 返回TFMultipleChoiceModelOutput对象
            loss=loss,  # 损失
            logits=reshaped_logits,  # 调整形状后的logits
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力
        )

    def build(self, input_shape=None):  # 定义build方法，用于构建模型
        if self.built:  # 如果已经构建过
            return  # 直接返回
        self.built = True  # 将标志设置为已构建
        if getattr(self, "roberta", None) is not None:  # 如果存在RoBERTa模型
            with tf.name_scope(self.roberta.name):  # 使用RoBERTa模型的名称范围
                self.roberta.build(None)  # 构建RoBERTa模型
        if getattr(self, "classifier", None) is not None:  # 如果存在分类器
            with tf.name_scope(self.classifier.name):  # 使用分类器的名称范围
                self.classifier.build([None, None, self.config.hidden_size])  # 构建分类器
# 导入必要的模块
@add_start_docstrings(
    """
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 定义了一个CamemBERT模型，用于问答任务，包含一个用于提取性问题回答的跨度分类头（在隐藏状态输出之上的线性层，用于计算`跨度起始对数`和`跨度结束对数`）。
class TFCamembertForQuestionAnswering(TFCamembertPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义了在从PyTorch模型加载到TensorFlow模型时可以忽略的授权的不期望/丢失层的名称列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数
        self.num_labels = config.num_labels

        # 初始化CamemBERT主层
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化用于输出的全连接层
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存模型配置
        self.config = config

    # 前向传播函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
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
        # 使用 Roberta 模型处理输入数据
        outputs = self.roberta(
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
        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 获取问题回答的 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 拆分为起始和结束 logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果有起始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 Roberta 模型，则构建它
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在 qa_outputs，则构建它
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
# 使用指定的文档字符串作为CamemBERT模型的说明，该模型在顶部有一个用于CLM微调的语言建模头部
@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", CAMEMBERT_START_DOCSTRING
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForCausalLM复制过来，将Roberta->Camembert，ROBERTA->CAMEMBERT
class TFCamembertForCausalLM(TFCamembertPreTrainedModel, TFCausalLanguageModelingLoss):
    # 名称中带有'.'的表示加载TF模型时授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    def __init__(self, config: CamembertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果不是解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `TFCamembertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 CamemBERT 主层
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化语言建模头部
        self.lm_head = TFCamembertLMHead(config, input_embeddings=self.roberta.embeddings, name="lm_head")

    # 获取语言建模头部
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名
    def get_prefix_bias_name(self):
        # 发出警告：该方法已弃用，请使用`get_bias`代替
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 从transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel中复制，准备用于生成的输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果模型作为编码器-解码器模型的解码器使用，解码器注意力掩码会即时创建
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果存在过去的键值，则截取解码器输入ID
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 从CamemBERT输入文档字符串中解包输入参数，并添加到模型前向传播的文档字符串中
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
```  
    # 定义一个方法，用于调用模型，接收各种输入参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，可以为 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以为 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，可以为 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，可以为 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以为 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入，可以为 None
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态，可以为 None
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力掩码，可以为 None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值，可选的元组，可以为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可以为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为 None
        return_dict: Optional[bool] = None,  # 是否返回字典，可以为 None
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，可以为 None
        training: Optional[bool] = False,  # 是否处于训练模式，默认为 False
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果有 roberta 属性，构建 roberta 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):  # 使用命名空间
                self.roberta.build(None)  # 构建 roberta 模型
        # 如果有 lm_head 属性，构建 lm_head 模型
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):  # 使用命名空间
                self.lm_head.build(None)  # 构建 lm_head 模型
```