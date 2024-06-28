# `.\models\electra\modeling_tf_electra.py`

```py
# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" TF Electra model."""


from __future__ import annotations

import math  # 导入数学库
import warnings  # 导入警告模块
from dataclasses import dataclass  # 导入 dataclass 用于创建结构化的类
from typing import Optional, Tuple, Union  # 导入类型提示相关库

import numpy as np  # 导入 numpy 库
import tensorflow as tf  # 导入 TensorFlow 库

from ...activations_tf import get_tf_activation  # 导入获取 TensorFlow 激活函数的函数
from ...modeling_tf_outputs import (  # 导入 TensorFlow 模型输出相关类
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 导入 TensorFlow 模型工具函数和类
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
from ...tf_utils import (  # 导入 TensorFlow 工具函数
    check_embeddings_within_bounds,
    shape_list,
    stable_softmax,
)
from ...utils import (  # 导入通用工具函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_electra import ElectraConfig  # 导入 Electra 的配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

_CHECKPOINT_FOR_DOC = "google/electra-small-discriminator"  # 用于文档的预训练模型检查点
_CONFIG_FOR_DOC = "ElectraConfig"  # 用于文档的 Electra 配置


TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型的存档列表
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # 查看所有 ELECTRA 模型：https://huggingface.co/models?filter=electra
]


# 从 transformers.models.bert.modeling_tf_bert.TFBertSelfAttention 复制并修改为 Electra 模型
class TFElectraSelfAttention(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        # 调用父类初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建查询、键、值的全连接层，并初始化
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 设置 dropout 层，用于注意力概率的 dropout
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder
        # 保存配置对象
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将输入张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量进行转置，从 [batch_size, seq_length, num_attention_heads, attention_head_size] 变为 [batch_size, num_attention_heads, seq_length, attention_head_size]
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
        # 在这里实现模型的前向传播
        # （这部分根据代码的未提供，无法详细解释其具体功能）
        pass

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
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
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->Electra
class TFElectraSelfOutput(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于映射输入到隐藏状态大小的输出
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义LayerNormalization层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义Dropout层，用于随机丢弃部分隐藏状态，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # Dropout层应用于全连接层的输出
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # LayerNormalization层应用于处理后的隐藏状态和输入张量的残差连接
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层已定义，则根据输入形状构建dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果LayerNorm层已定义，则根据输入形状构建LayerNorm层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->Electra
class TFElectraAttention(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义Electra自注意力层
        self.self_attention = TFElectraSelfAttention(config, name="self")
        # 定义Electra自输出层
        self.dense_output = TFElectraSelfOutput(config, name="output")

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
        # 调用自注意力层处理输入Tensor
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
        # 调用自输出层处理自注意力层的输出和原始输入Tensor
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力值，将它们与输出合并
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义一个方法 `build`，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self_attention 属性
        if getattr(self, "self_attention", None) is not None:
            # 在命名作用域内，构建 self_attention 层
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果存在 dense_output 属性
        if getattr(self, "dense_output", None) is not None:
            # 在命名作用域内，构建 dense_output 层
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->Electra
class TFElectraIntermediate(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.intermediate_size，权重初始化方式为 config.initializer_range
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置选择激活函数，如果是字符串形式，则通过工具函数获取对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 输入 hidden_states 经过全连接层 dense 处理
        hidden_states = self.dense(inputs=hidden_states)
        # 经过选定的激活函数处理后返回
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则按照给定的形状构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->Electra
class TFElectraOutput(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.hidden_size，权重初始化方式为 config.initializer_range
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，epsilon 为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，dropout rate 为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 输入 hidden_states 经过全连接层 dense 处理
        hidden_states = self.dense(inputs=hidden_states)
        # 根据训练状态应用 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 输入 hidden_states 与 input_tensor 相加后，经过 LayerNormalization 处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则按照给定的形状构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在 LayerNorm 层，则按照给定的形状构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->Electra
class TFElectraLayer(keras.layers.Layer):
    # 这部分未提供完整代码，暂无法添加注释
    # 初始化 ElectraModel 类的实例
    def __init__(self, config: ElectraConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建注意力机制对象，并命名为 'attention'
        self.attention = TFElectraAttention(config, name="attention")
        
        # 检查当前模型是否为解码器
        self.is_decoder = config.is_decoder
        
        # 检查是否需要添加跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        
        # 如果需要添加跨注意力机制且当前模型不是解码器，则抛出值错误异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建跨注意力机制对象，并命名为 'crossattention'
            self.crossattention = TFElectraAttention(config, name="crossattention")
        
        # 创建 Electra 中间层对象，并命名为 'intermediate'
        self.intermediate = TFElectraIntermediate(config, name="intermediate")
        
        # 创建 Electra 输出层对象，并命名为 'output'
        self.bert_output = TFElectraOutput(config, name="output")

    # 定义调用方法，用于前向传播计算
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
        # 定义函数的输入和输出类型，这里返回一个包含 Tensor 元组的 Tuple
        ) -> Tuple[tf.Tensor]:
        # 如果过去的键/值缓存不为空，提取自注意力的缓存键/值元组的前两个位置
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力层处理隐藏状态，生成自注意力输出
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
        # 提取自注意力的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 提取除了自注意力输出之外的所有输出作为结果
            outputs = self_attention_outputs[1:-1]
            # 提取自注意力的当前键/值缓存
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，添加自注意力权重输出到结果中
            outputs = self_attention_outputs[1:]

        # 初始化交叉注意力的当前键/值缓存为 None
        cross_attn_present_key_value = None
        # 如果是解码器且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型没有交叉注意力层，抛出数值错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值缓存不为空，提取交叉注意力的缓存键/值元组的后两个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 调用交叉注意力层处理自注意力输出，生成交叉注意力输出
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
            # 提取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力的输出添加到结果中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的当前键/值缓存添加到当前键/值缓存中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 通过中间层处理注意力输出，生成中间输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 通过 BERT 输出层处理中间输出和注意力输出，生成层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 将注意力输出添加到结果中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后一个输出添加到结果中
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回所有输出作为函数的结果
        return outputs
    # 构建方法，用于建立模型的网络结构
    def build(self, input_shape=None):
        # 如果已经建立过，直接返回，避免重复建立
        if self.built:
            return
        # 设置标记，表示模型已经建立
        self.built = True
        
        # 如果存在注意力模型，建立其网络结构
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层模型，建立其网络结构
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在BERT输出模型，建立其网络结构
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果存在交叉注意力模型，建立其网络结构
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertEncoder with Bert->Electra
class TFElectraEncoder(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建多个 Electra 层组成的列表，每层命名为 "layer_._{i}"
        self.layer = [TFElectraLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 初始化存储所有隐藏状态、注意力等的空元组，如果不需要输出则为 None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要缓存下一层的输出，则初始化为空元组
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有过去的键值对，则获取当前层的过去键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的前向传播方法，获取当前层的输出
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
            hidden_states = layer_outputs[0]  # 更新当前隐藏状态为当前层的输出的第一个元素

            # 如果需要缓存，将当前层的输出最后一个元素添加到下一层的缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力，将当前层的注意力添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置要求添加跨层注意力，并且有编码器隐藏状态，则添加跨层注意力
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回非空的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回带有过去键值对和跨层注意力的 Electra 模型输出对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义一个方法 `build`，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        # 将标志位 `built` 设为 True，表示网络已构建
        self.built = True
        # 如果属性 `layer` 存在
        if getattr(self, "layer", None) is not None:
            # 遍历每个层对象
            for layer in self.layer:
                # 在 TensorFlow 的命名空间中，按层的名称设置命名空间
                with tf.name_scope(layer.name):
                    # 调用每个层对象的 `build` 方法来构建层，参数为 None
                    layer.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->Electra
class TFElectraPooler(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # Initialize a dense layer for pooling with specified hidden size, tanh activation, and initializer
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # Extract the hidden state of the first token for pooling
        first_token_tensor = hidden_states[:, 0]
        # Apply the dense layer to the first token's hidden state for pooling
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        # Check if layer is already built
        if self.built:
            return
        self.built = True
        # Build the dense layer with specified input shape and hidden size from config
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.albert.modeling_tf_albert.TFAlbertEmbeddings with Albert->Electra
class TFElectraEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        # Layer normalization for embeddings with specified epsilon
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout layer for embeddings with specified dropout rate
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        # Build word embeddings with vocab size and embedding size
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # Build token type embeddings with type vocab size and embedding size
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # Build position embeddings with max position and embedding size
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # Check if layer is already built
        if self.built:
            return
        self.built = True
        # Build layer normalization for embeddings with specified input shape and embedding size
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
    # Copied from transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call
    def call(
        self,
        input_ids: tf.Tensor = None,                   # 输入的 token ids 张量
        position_ids: tf.Tensor = None,                # 位置 ids 张量
        token_type_ids: tf.Tensor = None,              # token 类型 ids 张量
        inputs_embeds: tf.Tensor = None,               # 嵌入的输入张量
        past_key_values_length=0,                      # 过去的键值对长度，默认为0
        training: bool = False,                        # 是否处于训练模式的布尔值
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.
    
        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")
    
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
    
        input_shape = shape_list(inputs_embeds)[:-1]   # 获取输入嵌入张量的形状
    
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)   # 如果没有指定 token 类型 ids，默认填充为0
    
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )
            # 如果没有指定位置 ids，生成一个范围为 [past_key_values_length, input_shape[1] + past_key_values_length) 的张量
    
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据位置 ids 从位置嵌入参数中获取位置嵌入张量
    
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 根据 token 类型 ids 从 token 类型嵌入参数中获取 token 类型嵌入张量
    
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 计算最终的嵌入张量，将输入嵌入、位置嵌入和 token 类型嵌入相加
    
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 使用层归一化处理最终的嵌入张量
    
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        # 使用 dropout 进行训练时的正则化处理
    
        return final_embeddings
class TFElectraDiscriminatorPredictions(keras.layers.Layer):
    # Electra 判别器预测层，继承自 Keras 层
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 创建一个全连接层，输出维度为 config.hidden_size，命名为 "dense"
        self.dense = keras.layers.Dense(config.hidden_size, name="dense")
        
        # 创建一个全连接层，输出维度为 1，命名为 "dense_prediction"
        self.dense_prediction = keras.layers.Dense(1, name="dense_prediction")
        
        # 保存配置信息
        self.config = config

    def call(self, discriminator_hidden_states, training=False):
        # 将判别器隐藏状态输入到全连接层中
        hidden_states = self.dense(discriminator_hidden_states)
        
        # 根据配置中的激活函数，对隐藏状态进行激活
        hidden_states = get_tf_activation(self.config.hidden_act)(hidden_states)
        
        # 压缩预测结果的维度，去除最后一个维度
        logits = tf.squeeze(self.dense_prediction(hidden_states), -1)

        return logits

    def build(self, input_shape=None):
        if self.built:
            return
        
        self.built = True
        
        # 如果 dense 层已经存在，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果 dense_prediction 层已经存在，则构建该层
        if getattr(self, "dense_prediction", None) is not None:
            with tf.name_scope(self.dense_prediction.name):
                self.dense_prediction.build([None, None, self.config.hidden_size])


class TFElectraGeneratorPredictions(keras.layers.Layer):
    # Electra 生成器预测层，继承自 Keras 层
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 创建 LayerNormalization 层，epsilon 设置为 config.layer_norm_eps，命名为 "LayerNorm"
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 创建全连接层，输出维度为 config.embedding_size，命名为 "dense"
        self.dense = keras.layers.Dense(config.embedding_size, name="dense")
        
        # 保存配置信息
        self.config = config

    def call(self, generator_hidden_states, training=False):
        # 将生成器隐藏状态输入到全连接层中
        hidden_states = self.dense(generator_hidden_states)
        
        # 使用 GELU 激活函数对隐藏状态进行激活
        hidden_states = get_tf_activation("gelu")(hidden_states)
        
        # 对激活后的隐藏状态进行 LayerNormalization 处理
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        
        self.built = True
        
        # 如果 LayerNorm 层已经存在，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
        
        # 如果 dense 层已经存在，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFElectraPreTrainedModel(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """
    
    # 默认配置类为 ElectraConfig
    config_class = ElectraConfig
    
    # 基础模型前缀为 "electra"
    base_model_prefix = "electra"
    
    # 从 PT 模型加载时忽略的键
    _keys_to_ignore_on_load_unexpected = [r"generator_lm_head.weight"]
    
    # 加载时缺失的键
    _keys_to_ignore_on_load_missing = [r"dropout"]


@keras_serializable
class TFElectraMainLayer(keras.layers.Layer):
    # Electra 主层，继承自 Keras 层
    config_class = ElectraConfig
    # 初始化方法，接受配置和可选参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将配置保存到实例变量中
        self.config = config
        # 根据配置设置是否为解码器的标志
        self.is_decoder = config.is_decoder

        # 创建电力特拉嵌入层对象，并命名为"embeddings"
        self.embeddings = TFElectraEmbeddings(config, name="embeddings")

        # 如果嵌入层的嵌入大小不等于隐藏大小，则创建一个全连接层用于投影
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = keras.layers.Dense(config.hidden_size, name="embeddings_project")

        # 创建电力特拉编码器对象，并命名为"encoder"
        self.encoder = TFElectraEncoder(config, name="encoder")

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        # 返回嵌入层对象
        return self.embeddings

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        # 设置嵌入层的权重
        self.embeddings.weight = value
        # 设置嵌入层的词汇表大小
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型头部的方法，抛出未实现错误
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError
    def get_extended_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length=0):
        # 获取输入的批量大小和序列长度
        batch_size, seq_length = input_shape

        # 如果没有提供注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)

        # 创建一个3D注意力掩码，从一个2D张量掩码中生成
        # 大小为 [batch_size, 1, 1, to_seq_length]
        # 这样可以广播到 [batch_size, num_heads, from_seq_length, to_seq_length]
        # 这个注意力掩码比在OpenAI GPT中使用的三角形遮盖更简单，我们只需要准备广播维度。
        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        # 从 `modeling_tf_t5.py` 复制而来
        # 提供一个维度为 [batch_size, mask_seq_length] 的填充掩码
        # - 如果模型是解码器，除了填充掩码外还应用因果掩码
        # - 如果模型是编码器，使掩码可广播到 [batch_size, num_heads, mask_seq_length, mask_seq_length]
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(
                extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
            )
            # 如果存在过去的键值长度大于0，则修剪注意力掩码
            if past_key_values_length > 0:
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            # 对于编码器，将注意力掩码重塑为 [batch_size, 1, 1, attention_mask_shape[1]]
            extended_attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # 将注意力掩码转换为指定的数据类型
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=dtype)
        one_cst = tf.constant(1.0, dtype=dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
        # 将掩码中的1.0变为0.0，0.0变为-10000.0，以便在softmax之前抑制掉未关注的位置
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        return extended_attention_mask
    # 如果头部遮罩(head_mask)不为None，则抛出未实现的错误，暂不支持头部遮罩
    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            # 如果头部遮罩为None，则创建一个长度为self.config.num_hidden_layers的空遮罩列表
            head_mask = [None] * self.config.num_hidden_layers

        # 返回头部遮罩
        return head_mask

    # 使用装饰器unpack_inputs解包输入参数
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
    ):
        # 在构建模型时，如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在self.embeddings，则构建它
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在self.encoder，则构建它
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在self.embeddings_project，则根据指定的形状构建它
        if getattr(self, "embeddings_project", None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                self.embeddings_project.build([None, None, self.config.embedding_size])
@dataclass
class TFElectraForPreTrainingOutput(ModelOutput):
    """
    [`TFElectraForPreTraining`]的输出类型。

    Args:
        loss (*可选*, 当提供 `labels` 时返回, `tf.Tensor` 形状为 `(1,)`):
            ELECTRA 目标的总损失。
        logits (`tf.Tensor` 形状为 `(batch_size, sequence_length)`):
            头部的预测分数（SoftMax 前每个标记的分数）。
        hidden_states (`tuple(tf.Tensor)`, *可选*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            元组的 `tf.Tensor`（一个用于嵌入输出 + 每个层的输出）形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每层输出的隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *可选*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            元组的 `tf.Tensor`（每个层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头部的加权平均。

    """

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


ELECTRA_START_DOCSTRING = r"""

    此模型继承自 [`TFPreTrainedModel`]。查看超类文档以获取库实现的所有模型的通用方法（如下载或保存、调整输入嵌入、修剪头部等）。

    此模型还是 [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。将其视为常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档，了解有关一般用法和行为的所有内容。

    <Tip>

    `transformers` 中的 TensorFlow 模型和层接受两种输入格式：

    - 将所有输入作为关键字参数（类似于 PyTorch 模型）；
    - 将所有输入作为列表、元组或字典的第一个位置参数。

    支持第二种格式的原因是，当传递输入给模型和层时，Keras 方法更喜欢此格式。由于这种支持，在使用诸如 `model.fit()` 等方法时，您应该能够“只需传递”您的输入和标签 - 只需使用 `model.fit()` 支持的任何格式！但是，如果您想在 Keras 方法如 `fit()` 和 `predict()` 之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可以用于在第一个位置参数中收集所有输入张量：

    - 只有 `input_ids` 的单个张量：`model(input_ids)`
    - 可变长度列表，其中按文档字符串中给出的顺序包含一个或多个输入张量：

"""
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - 当使用模型对象 `model` 时，可以传入一个包含输入张量的字典，键名需与文档字符串中给出的输入名称对应：
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!
    - 当使用子类化创建模型和层时，您无需担心这些细节，可以像传递任何其他 Python 函数的输入一样操作！

    Parameters:
        config ([`ElectraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        - config ([`ElectraConfig`]): 包含模型所有参数的配置类。
          使用配置文件初始化模型时，并不会加载与模型关联的权重，只加载配置信息。
          查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型的权重。
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    # 添加文档字符串前缀，将其应用于下方的函数装饰器
    """
    生成器模型和判别器模型的检查点可以加载到此模型中。

    这是一个裸的 Electra 模型变压器，输出未经任何特定头部处理的原始隐藏状态。与 BERT 模型相似，但如果隐藏大小和嵌入大小不同，则在嵌入层和编码器之间使用额外的线性层。

    ELECTRA_START_DOCSTRING 标识符，指示这是 Electra 模型的文档字符串的起始部分。
    """
    )
    # 结束类定义的括号

class TFElectraModel(TFElectraPreTrainedModel):
    # TFElectraModel 类继承自 TFElectraPreTrainedModel 类

    def __init__(self, config, *inputs, **kwargs):
        # 初始化方法，接受 config 对象和任意其他输入参数

        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFElectraMainLayer 实例并赋值给 self.electra
        self.electra = TFElectraMainLayer(config, name="electra")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
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
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
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
        # 调用 Electra 模型进行前向传播，接受多个输入参数
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 Electra 模型的输出结果
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 self.electra 存在，则在对应的命名空间下构建 Electra 模型
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                # 构建 Electra 模型，传入 None 作为输入形状
                self.electra.build(None)
# 使用装饰器为类添加文档字符串，描述该类的作用和功能
@add_start_docstrings(
    """
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.

    Even though both the discriminator and generator may be loaded into this model, the discriminator is the only model
    of the two to have the correct classification head to be used for this model.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForPreTraining(TFElectraPreTrainedModel):
    
    # 初始化方法，接收配置和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        
        # 创建 Electra 主层，并命名为 "electra"
        self.electra = TFElectraMainLayer(config, name="electra")
        
        # 创建 Electra 鉴别器预测层，并命名为 "discriminator_predictions"
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name="discriminator_predictions")

    # 调用方法，接收多个输入参数，执行模型的前向传播
    @unpack_inputs
    # 使用装饰器添加模型前向传播的文档字符串，描述输入参数的格式和作用
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器替换返回值的文档字符串，指定返回结果的类型为 TFElectraForPreTrainingOutput
    @replace_return_docstrings(output_type=TFElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
        # 最后一个参数没有被完全列出

        # 表示是否返回字典形式的结果
        return_dict: Optional[bool] = None,
        # 是否在训练模式下运行模型
        training: Optional[bool] = False,
        discriminator_hidden_states = self.electra(
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

# 调用 self.electra 模型进行前向传播，传入各种输入参数，获取鉴别器模型的隐藏状态。


        discriminator_sequence_output = discriminator_hidden_states[0]

# 从鉴别器模型的隐藏状态中提取序列输出，即第一个元素。


        logits = self.discriminator_predictions(discriminator_sequence_output)

# 使用 self.discriminator_predictions 模型预测鉴别器输出的 logits（对数概率）。


        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]

# 如果 return_dict 参数为 False，则返回 logits 和鉴别器模型的其他隐藏状态。


        return TFElectraForPreTrainingOutput(
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

# 如果 return_dict 参数为 True，则返回 TFElectraForPreTrainingOutput 对象，包含 logits、隐藏状态和注意力权重。



    def build(self, input_shape=None):
        if self.built:
            return

# 如果模型已经构建过，直接返回，避免重复构建。


        self.built = True

# 将模型标记为已构建状态。


        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)

# 如果 self.electra 存在，使用其名称作为命名空间，在该命名空间下构建 self.electra 模型。


        if getattr(self, "discriminator_predictions", None) is not None:
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)

# 如果 self.discriminator_predictions 存在，使用其名称作为命名空间，在该命名空间下构建 self.discriminator_predictions 模型。
class TFElectraMaskedLMHead(keras.layers.Layer):
    # 定义 Electra 模型的 Masked Language Modeling 头部的层
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.embedding_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        # 添加权重，初始化偏置向量为全零向量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        # 返回输入的嵌入层对象
        return self.input_embeddings

    def set_output_embeddings(self, value):
        # 设置输入的嵌入层的权重和词汇大小
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        # 返回偏置向量字典
        return {"bias": self.bias}

    def set_bias(self, value):
        # 设置偏置向量
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        # 计算 Masked Language Modeling 的输出
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings(
    """
    Electra model with a language modeling head on top.

    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of
    the two to have been trained for the masked language modeling task.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMaskedLM(TFElectraPreTrainedModel, TFMaskedLanguageModelingLoss):
    # Electra 模型加上顶部的语言建模头部
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        # Electra 主层
        self.electra = TFElectraMainLayer(config, name="electra")
        # Electra 生成器预测
        self.generator_predictions = TFElectraGeneratorPredictions(config, name="generator_predictions")

        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        # Electra 的 Masked Language Modeling 头部
        self.generator_lm_head = TFElectraMaskedLMHead(config, self.electra.embeddings, name="generator_lm_head")

    def get_lm_head(self):
        # 返回 Masked Language Modeling 头部
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        # 警告：方法已弃用，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.generator_lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/electra-small-generator",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output="'paris'",
        expected_loss=1.22,
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
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        Define the call function for the Electra generator model.
    
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # Generate hidden states using the Electra model with provided inputs
        generator_hidden_states = self.electra(
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
        # Extract sequence output from generator hidden states
        generator_sequence_output = generator_hidden_states[0]
        # Generate prediction scores using the generator predictions function
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        # Apply language modeling head to the generator prediction scores
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        # Compute loss only if labels are provided using the provided loss computation function
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
    
        # Prepare output based on whether return_dict is False or True
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output
    
        # Return TFMaskedLMOutput with detailed components if return_dict is True
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )
    # 构建模型的方法，用于设置模型结构和参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在名为 "electra" 的子模型，进行其构建
        if getattr(self, "electra", None) is not None:
            # 使用电力转换模型的名字作为命名空间
            with tf.name_scope(self.electra.name):
                # 调用电力转换模型的构建方法，输入形状为 None 表示使用默认形状
                self.electra.build(None)
        
        # 如果存在名为 "generator_predictions" 的子模型，进行其构建
        if getattr(self, "generator_predictions", None) is not None:
            # 使用生成器预测模型的名字作为命名空间
            with tf.name_scope(self.generator_predictions.name):
                # 调用生成器预测模型的构建方法，输入形状为 None 表示使用默认形状
                self.generator_predictions.build(None)
        
        # 如果存在名为 "generator_lm_head" 的子模型，进行其构建
        if getattr(self, "generator_lm_head", None) is not None:
            # 使用生成器语言模型头部的名字作为命名空间
            with tf.name_scope(self.generator_lm_head.name):
                # 调用生成器语言模型头部的构建方法，输入形状为 None 表示使用默认形状
                self.generator_lm_head.build(None)
    """
    ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """

@add_start_docstrings(
    """
    ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForSequenceClassification(TFElectraPreTrainedModel, TFSequenceClassificationLoss):
    """
    ELECTRA模型的转换器，顶部带有序列分类/回归头（在汇聚输出顶部的线性层），例如用于GLUE任务。
    """

    def __init__(self, config, *inputs, **kwargs):
        """
        初始化方法。

        Args:
            config (ElectraConfig): 模型的配置对象，包含模型的超参数。
            *inputs: 可变长度的输入参数。
            **kwargs: 其他关键字参数。
        """
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels  # 设置模型的标签数
        self.electra = TFElectraMainLayer(config, name="electra")  # ELECTRA主层对象
        self.classifier = TFElectraClassificationHead(config, name="classifier")  # 分类头部对象

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'joy'",
        expected_loss=0.06,
    )
    def forward(self, *model_args, **model_kwargs):
        """
        正向传播方法，根据输入计算模型输出。

        Args:
            *model_args: 可变长度的模型输入参数。
            **model_kwargs: 模型输入的关键字参数。

        Returns:
            TFSequenceClassifierOutput: 序列分类器的输出对象。
        """
        pass  # 这里的方法体未提供，仅有注释和装饰器的声明
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 接收输入的文本序列的 ID，可以为空
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，用于指示模型在处理输入时哪些部分需要注意
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 用于区分不同文本序列的 token 类型 ID
        position_ids: np.ndarray | tf.Tensor | None = None,  # 表示输入中每个 token 的位置 ID
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，用于指示模型在自注意力机制中哪些头部需要被屏蔽
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 可选的嵌入输入，可以直接提供输入的嵌入表示
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回结果字典
        labels: np.ndarray | tf.Tensor | None = None,  # 用于计算序列分类/回归损失的标签
        training: Optional[bool] = False,  # 是否处于训练模式
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 Electra 模型进行前向传播
        outputs = self.electra(
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
        # 将 Electra 输出传递给分类器
        logits = self.classifier(outputs[0])
        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 根据 return_dict 参数决定返回结果的格式
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            # 如果 return_dict 为 True，则返回 TFSequenceClassifierOutput 对象
            return TFSequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 Electra 模型，则构建其内部结构
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        # 如果存在分类器模型，则构建其内部结构
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
"""
ELECTRA 模型，顶部带有多选分类头部（在池化输出的基础上是一个线性层和一个 softmax），例如用于 RocStories/SWAG 任务。

继承自 TFElectraPreTrainedModel 和 TFMultipleChoiceLoss。
"""
@add_start_docstrings(
    """
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMultipleChoice(TFElectraPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        """
        初始化方法，设置模型的各个组件。

        Parameters:
        - config: ELECTRA 模型的配置对象。
        - *inputs: 可变长度的输入。
        - **kwargs: 其他关键字参数。
        """
        super().__init__(config, *inputs, **kwargs)

        # ELECTRA 主体层
        self.electra = TFElectraMainLayer(config, name="electra")
        # 序列汇总层
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        # 分类器层
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        """
        调用方法，执行 ELECTRA 模型的前向传播。

        Parameters:
        - input_ids: 输入的 token IDs。
        - attention_mask: 注意力掩码。
        - token_type_ids: token 类型 IDs。
        - position_ids: 位置 IDs。
        - head_mask: 头部掩码。
        - inputs_embeds: 输入的嵌入。
        - output_attentions: 是否输出注意力。
        - output_hidden_states: 是否输出隐藏状态。
        - return_dict: 是否返回字典形式结果。
        - labels: 标签数据。
        - training: 是否处于训练模式。

        Returns:
        ELECTRA 模型的输出对象。
        """
        ...
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果给定了 input_ids，则获取其第二和第三维的大小
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 如果没有 input_ids，则获取 inputs_embeds 的第二和第三维的大小
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入张量展平为二维张量，如果相应输入不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        
        # 调用 Electra 模型进行前向传播，传入展平后的张量及其他参数
        outputs = self.electra(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 对 Electra 模型的输出进行序列汇总
        logits = self.sequence_summary(outputs[0])
        # 将汇总后的序列 logits 输入分类器进行分类预测
        logits = self.classifier(logits)
        # 重新整形 logits 张量为形状为 (-1, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        # 如果提供了 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果 return_dict=False，则按指定格式返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，则返回带有多选模型输出的对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记已构建
        self.built = True
        
        # 如果存在 self.electra 属性，则构建 Electra 模型
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        
        # 如果存在 self.sequence_summary 属性，则构建序列汇总层
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        
        # 如果存在 self.classifier 属性，则构建分类器层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Electra model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForTokenClassification(TFElectraPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # 初始化 Electra 主模型层，命名为 "electra"
        self.electra = TFElectraMainLayer(config, name="electra")

        # 根据配置中的 dropout 概率设置分类器的 dropout 层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)

        # 定义一个全连接层作为分类器，输出维度为类别数目
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

        # 将配置保存在对象中
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-discriminator-finetuned-conll03-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['B-LOC', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC']",
        expected_loss=0.11,
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
        # 调用 ELECTRA 模型进行预测，获取鉴别器的隐藏状态
        discriminator_hidden_states = self.electra(
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
        # 从鉴别器的隐藏状态中取出序列输出
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 对鉴别器的序列输出应用 dropout 操作
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        # 将 dropout 后的输出传递给分类器，得到预测的 logits
        logits = self.classifier(discriminator_sequence_output)
        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 根据 return_dict 的值决定返回的结果格式
        if not return_dict:
            # 如果不要求返回字典，则输出 logits 和其它隐藏状态
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典格式的结果，则返回 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 ELECTRA 模型，建立其内部结构
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        # 如果存在分类器模型，建立其内部结构
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器添加模型文档字符串，描述了 Electra 模型在提取式问答任务（如 SQuAD）中的应用，包括在隐藏状态输出之上的线性层，用于计算“span start logits”和“span end logits”。
@add_start_docstrings(
    """
    Electra Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ELECTRA_START_DOCSTRING,
)
# 定义 TFElectraForQuestionAnswering 类，继承自 TFElectraPreTrainedModel 和 TFQuestionAnsweringLoss
class TFElectraForQuestionAnswering(TFElectraPreTrainedModel, TFQuestionAnsweringLoss):
    
    # 初始化方法，接受配置 config 和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建 Electra 主层对象，命名为 "electra"
        self.electra = TFElectraMainLayer(config, name="electra")
        # 创建输出层，使用 Dense 层，输出大小为 config.num_labels，使用指定的初始化器初始化权重，命名为 "qa_outputs"
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置对象
        self.config = config

    # 使用装饰器来包装 call 方法，添加模型前向传播的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=11,
        qa_target_end_index=12,
        expected_output="'a nice puppet'",
        expected_loss=2.64,
    )
    # 定义模型的前向传播方法，接受多个输入参数和一些控制参数
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
        # 结尾未完，继续下一页 。
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
        discriminator_hidden_states = self.electra(
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
        # 获取鉴别器模型的隐藏状态
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 通过输出序列计算问题回答的逻辑张量
        logits = self.qa_outputs(discriminator_sequence_output)
        # 将逻辑张量沿最后一个维度分割为起始和结束的逻辑张量
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 压缩起始和结束的逻辑张量的最后一个维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        # 初始化损失变量
        loss = None

        # 如果提供了起始和结束的位置信息，则计算损失
        if start_positions is not None and end_positions is not None:
            # 准备标签，用于计算损失
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用标签和预测的逻辑张量计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不要求返回字典，则组装输出
        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + discriminator_hidden_states[1:]
            # 返回损失和输出，如果损失不为None
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 对象，包含损失和其他输出信息
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 electra 层，则构建它
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        # 如果存在 qa_outputs 层，则构建它
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```