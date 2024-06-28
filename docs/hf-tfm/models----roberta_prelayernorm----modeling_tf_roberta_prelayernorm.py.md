# `.\models\roberta_prelayernorm\modeling_tf_roberta_prelayernorm.py`

```
# coding=utf-8
# 版权 2022 年由 Google AI 语言团队和 HuggingFace Inc. 团队所有。
# 版权 (c) 2018 年 NVIDIA 公司。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据“原样”提供软件，
# 没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
""" TF 2.0 RoBERTa-PreLayerNorm 模型。"""


from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
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
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "andreasmadsen/efficient_mlm_m0.40"
_CONFIG_FOR_DOC = "RobertaPreLayerNormConfig"

# 预训练模型存档列表
TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "andreasmadsen/efficient_mlm_m0.15",
    "andreasmadsen/efficient_mlm_m0.20",
    "andreasmadsen/efficient_mlm_m0.30",
    "andreasmadsen/efficient_mlm_m0.40",
    "andreasmadsen/efficient_mlm_m0.50",
    "andreasmadsen/efficient_mlm_m0.60",
    "andreasmadsen/efficient_mlm_m0.70",
    "andreasmadsen/efficient_mlm_m0.80",
    # 查看所有 RoBERTaWithPreLayerNorm 模型，请访问 https://huggingface.co/models?filter=roberta_with_prelayernorm
]


# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaEmbeddings 复制并修改为 TFRobertaPreLayerNormEmbeddings
class TFRobertaPreLayerNormEmbeddings(keras.layers.Layer):
    """
    与 BertEmbeddings 相同，但进行了微小的调整以适应位置嵌入的索引。
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 1  # 设定填充符号的索引为1
        self.config = config  # 保存配置对象
        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小
        self.max_position_embeddings = config.max_position_embeddings  # 从配置中获取最大位置嵌入数
        self.initializer_range = config.initializer_range  # 从配置中获取初始化范围
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")  # 创建LayerNorm层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)  # 创建Dropout层

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),  # 使用指定的初始化器初始化权重
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),  # 使用指定的初始化器初始化类型嵌入
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),  # 使用指定的初始化器初始化位置嵌入
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])  # 构建LayerNorm层

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor  # 输入的张量，表示输入的标识符

        Returns: tf.Tensor  # 返回的张量，表示生成的位置标识符
        """
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)  # 创建一个掩码，标识非填充符号
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask  # 计算递增的位置索引

        return incremental_indices + self.padding_idx  # 返回生成的位置标识符

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
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # 检查输入的 token ids 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 使用权重矩阵按照输入的 token ids 获取对应的嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状，去除最后一个维度
        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            # 如果没有提供 token_type_ids，则填充全为 0 的张量
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            if input_ids is not None:
                # 根据输入的 token ids 创建位置 ids，任何填充的 token 仍保持填充状态
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 如果没有输入 token ids，则生成位置 ids 的张量
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        # 使用位置嵌入矩阵按照位置 ids 获取对应的位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 使用 token_type 嵌入矩阵按照 token_type ids 获取对应的 token_type 嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将输入嵌入、位置嵌入和 token_type 嵌入相加得到最终的嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的嵌入向量进行 LayerNormalization 处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 在训练时对最终的嵌入向量进行 dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入向量
        return final_embeddings
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->RobertaPreLayerNorm
class TFRobertaPreLayerNormPooler(keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义全连接层，用于池化操作，输出维度为 config.hidden_size
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 从隐藏状态中取出第一个 token 的隐藏状态，作为池化输出
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            # 构建全连接层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->RobertaPreLayerNorm
class TFRobertaPreLayerNormSelfAttention(keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

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
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config
    # 将输入张量重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size] 的形状
    tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

    # 将张量的维度顺序从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
    return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 如果层已经构建则直接返回，否则开始构建
    if self.built:
        return

    # 设置层为已构建状态
    self.built = True

    # 如果存在查询权重张量，则根据配置的隐藏大小构建查询权重张量
    if getattr(self, "query", None) is not None:
        with tf.name_scope(self.query.name):
            self.query.build([None, None, self.config.hidden_size])

    # 如果存在键权重张量，则根据配置的隐藏大小构建键权重张量
    if getattr(self, "key", None) is not None:
        with tf.name_scope(self.key.name):
            self.key.build([None, None, self.config.hidden_size])

    # 如果存在值权重张量，则根据配置的隐藏大小构建值权重张量
    if getattr(self, "value", None) is not None:
        with tf.name_scope(self.value.name):
            self.value.build([None, None, self.config.hidden_size])
# 定义一个自定义的 Keras 层，用于 RoBERTa 模型的前层归一化注意力模块的输出
class TFRobertaPreLayerNormAttention(keras.layers.Layer):
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建自注意力层对象，用于处理自注意力机制
        self.self_attention = TFRobertaPreLayerNormSelfAttention(config, name="self")
        
        # 创建输出层对象，用于处理自注意力层的输出
        self.dense_output = TFRobertaPreLayerNormSelfOutput(config, name="output")
        
        # 创建层归一化对象，对输入进行归一化处理
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 保存配置对象，包含模型超参数
        self.config = config

    # 从 transformers 库中的 TFBertAttention 类的 prune_heads 方法复制而来，用于裁剪注意力头
    def prune_heads(self, heads):
        raise NotImplementedError

    # 定义层的前向传播逻辑，输入和输出均为 TensorFlow 张量
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
        # 对输入张量进行层归一化处理
        hidden_states_pre_layer_norm = self.LayerNorm(inputs=input_tensor)
        
        # 将归一化后的张量输入到自注意力层中进行处理
        self_outputs = self.self_attention(
            hidden_states=hidden_states_pre_layer_norm,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        
        # 将自注意力层的输出输入到输出层中进行处理
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        
        # 如果需要输出注意力信息，则将注意力信息添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义神经网络层的构建方法，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 如果存在 self_attention 属性，则构建 self_attention 层
        if getattr(self, "self_attention", None) is not None:
            # 使用 self_attention 层的名称作为命名空间
            with tf.name_scope(self.self_attention.name):
                # 调用 self_attention 层的构建方法
                self.self_attention.build(None)
        # 如果存在 dense_output 属性，则构建 dense_output 层
        if getattr(self, "dense_output", None) is not None:
            # 使用 dense_output 层的名称作为命名空间
            with tf.name_scope(self.dense_output.name):
                # 调用 dense_output 层的构建方法
                self.dense_output.build(None)
        # 如果存在 LayerNorm 属性，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            # 使用 LayerNorm 层的名称作为命名空间
            with tf.name_scope(self.LayerNorm.name):
                # 调用 LayerNorm 层的构建方法，输入形状为 [None, None, self.config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])
class TFRobertaPreLayerNormIntermediate(keras.layers.Layer):
    # 初始化方法，设置层的参数
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # LayerNormalization层，用于对输入进行归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dense层，用于进行线性变换
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置文件设置中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 前向传播方法，定义了层的计算过程
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对输入进行层归一化
        hidden_states = self.LayerNorm(inputs=hidden_states)
        # 线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建方法，用于创建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建LayerNormalization层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 构建Dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFRobertaPreLayerNormOutput(keras.layers.Layer):
    # 初始化方法，设置层的参数
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)

        # Dense层，用于进行线性变换
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # Dropout层，用于随机失活
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 前向传播方法，定义了层的计算过程
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # Dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states

    # 构建方法，用于创建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建Dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


# 以下是从transformers库中复制并修改的代码，用于Roberta模型的预处理层归一化
class TFRobertaPreLayerNormLayer(keras.layers.Layer):
    # 初始化函数，用于创建一个 RoBERTa 预层归一化模型实例
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建注意力层对象，用给定的配置信息初始化，命名为"attention"
        self.attention = TFRobertaPreLayerNormAttention(config, name="attention")
        
        # 检查是否为解码器模型
        self.is_decoder = config.is_decoder
        
        # 检查是否添加了跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        
        # 如果添加了跨注意力机制，但当前模型不是解码器，则抛出值错误异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建跨注意力层对象，用给定的配置信息初始化，命名为"crossattention"
            self.crossattention = TFRobertaPreLayerNormAttention(config, name="crossattention")
        
        # 创建 RoBERTa 预层归一化模型的中间层对象，用给定的配置信息初始化，命名为"intermediate"
        self.intermediate = TFRobertaPreLayerNormIntermediate(config, name="intermediate")
        
        # 创建 RoBERTa 预层归一化模型的输出层对象，用给定的配置信息初始化，命名为"output"
        self.bert_output = TFRobertaPreLayerNormOutput(config, name="output")
        ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention for the decoder layer
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

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention between decoder and encoder layers
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
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建神经网络模型，如果已经构建过了则直接返回，否则执行构建过程
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回，不做任何操作
        if self.built:
            return
        # 将标识位设置为已构建
        self.built = True
        # 如果存在注意力模型，则根据其名称创建命名空间，并执行注意力模型的构建过程
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层模型，则根据其名称创建命名空间，并执行中间层模型的构建过程
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在BERT输出模型，则根据其名称创建命名空间，并执行BERT输出模型的构建过程
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果存在交叉注意力模型，则根据其名称创建命名空间，并执行交叉注意力模型的构建过程
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertEncoder with Bert->RobertaPreLayerNorm
# 定义一个自定义层 TFRobertaPreLayerNormEncoder，继承自 keras.layers.Layer 类
class TFRobertaPreLayerNormEncoder(keras.layers.Layer):
    # 初始化方法，接受一个 config 对象和额外的关键字参数
    def __init__(self, config: RobertaPreLayerNormConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 保存传入的 config 对象
        # 创建一个由 TFRobertaPreLayerNormLayer 实例组成的列表，共 config.num_hidden_layers 个
        self.layer = [TFRobertaPreLayerNormLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义 call 方法，用于执行层的前向传播
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
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，初始化 all_attentions 为空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力且配置允许，则初始化 all_cross_attentions 为空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要使用缓存，则初始化 next_decoder_cache 为空元组
        next_decoder_cache = () if use_cache else None
        
        # 遍历 self.layer 列表中的每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前 hidden_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果 past_key_values 不为 None，则获取当前层的 past_key_value
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的前向传播方法，得到 layer_outputs
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
            # 更新 hidden_states 为当前层输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要使用缓存，则将当前层的输出的最后一个元素加入 next_decoder_cache
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力，则将当前层的第二个元素加入 all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置允许且 encoder_hidden_states 不为 None，则将当前层的第三个元素加入 all_cross_attentions
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态，如果需要输出隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回非空的元组中的元素
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 如果 return_dict 为 True，则返回 TFBaseModelOutputWithPastAndCrossAttentions 对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义 build 方法，用于构建神经网络模型的层结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型包含层属性
        if getattr(self, "layer", None) is not None:
            # 遍历每个层并设置 TensorFlow 的命名作用域
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    # 调用每个层的 build 方法构建层结构，传入 None 作为输入形状
                    layer.build(None)
# 声明一个自定义层，用于TFRoberta预处理层的主要部分，通过装饰器指定其可序列化
@keras_serializable
class TFRobertaPreLayerNormMainLayer(keras.layers.Layer):
    # 将配置类设为RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig

    # 初始化方法，接受配置参数config和是否添加池化层的标志add_pooling_layer
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        # 将config保存到实例属性中
        self.config = config
        # 检查配置中是否为解码器模式
        self.is_decoder = config.is_decoder

        # 初始化其他实例属性，从配置中获取对应的值
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 创建编码器层实例，并命名为"encoder"
        self.encoder = TFRobertaPreLayerNormEncoder(config, name="encoder")
        # 创建LayerNormalization层实例，使用配置中的epsilon值，并命名为"LayerNorm"
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 如果add_pooling_layer为True，则创建池化层实例，命名为"pooler"，否则设置为None
        self.pooler = TFRobertaPreLayerNormPooler(config, name="pooler") if add_pooling_layer else None
        
        # 最后声明嵌入层实例，命名为"embeddings"，必须放在最后声明以保持权重顺序
        self.embeddings = TFRobertaPreLayerNormEmbeddings(config, name="embeddings")

    # 获取输入嵌入层的方法，返回嵌入层实例
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入层的方法，接受一个tf.Variable作为参数，设置权重和词汇大小
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 未实现的方法，用于剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 装饰器函数，用于解包输入参数，并接受多种类型的输入数据，处理预处理层的调用
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
    # 如果已经构建过网络，则直接返回，不进行重复构建
    if self.built:
        return
    # 标记网络为已构建状态
    self.built = True
    
    # 如果存在编码器(encoder)属性，则构建编码器
    if getattr(self, "encoder", None) is not None:
        # 在命名空间下构建编码器
        with tf.name_scope(self.encoder.name):
            self.encoder.build(None)
    
    # 如果存在 LayerNorm 属性，则构建 LayerNorm 层
    if getattr(self, "LayerNorm", None) is not None:
        # 在命名空间下构建 LayerNorm 层，指定输入形状为 [None, None, self.config.hidden_size]
        with tf.name_scope(self.LayerNorm.name):
            self.LayerNorm.build([None, None, self.config.hidden_size])
    
    # 如果存在池化器(pooler)属性，则构建池化器
    if getattr(self, "pooler", None) is not None:
        # 在命名空间下构建池化器
        with tf.name_scope(self.pooler.name):
            self.pooler.build(None)
    
    # 如果存在嵌入(embeddings)属性，则构建嵌入层
    if getattr(self, "embeddings", None) is not None:
        # 在命名空间下构建嵌入层
        with tf.name_scope(self.embeddings.name):
            self.embeddings.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaPreTrainedModel中复制代码，并将Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm进行了替换
class TFRobertaPreLayerNormPreTrainedModel(TFPreTrainedModel):
    """
    处理权重初始化、预训练模型下载和加载的抽象类。
    """

    # 指定配置类为RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig
    # 指定基础模型前缀为"roberta_prelayernorm"
    base_model_prefix = "roberta_prelayernorm"


ROBERTA_PRELAYERNORM_START_DOCSTRING = r"""

    该模型继承自[`TFPreTrainedModel`]。查阅超类文档以获取库实现的通用方法，如下载或保存模型、调整输入嵌入、修剪头等。

    该模型也是一个[keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。可以像常规TF 2.0 Keras模型一样使用，并且请参考TF 2.0文档以获取所有与一般使用和行为相关的事项。

    <Tip>

    `transformers`中的TensorFlow模型和层接受两种输入格式：

    - 将所有输入作为关键字参数传递（类似于PyTorch模型），或者
    - 将所有输入作为列表、元组或字典的第一个位置参数传递。

    支持第二种格式的原因是，Keras方法在将输入传递给模型和层时更喜欢这种格式。因此，在使用`model.fit()`等方法时，只需将输入和标签以任何`model.fit()`支持的格式传递即可！但是，如果要在Keras方法之外（如创建自己的层或使用Keras`Functional`API创建模型时）使用第二种格式，可以使用三种方法来收集第一个位置参数中的所有输入张量：

    - 仅包含`input_ids`的单个张量，没有其他内容：`model(input_ids)`
    - 长度可变的列表，按照文档字符串中给定的顺序包含一个或多个输入张量：`model([input_ids, attention_mask])`或`model([input_ids, attention_mask, token_type_ids])`
    - 一个字典，包含一个或多个与文档字符串中给定输入名称相关联的输入张量：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    注意，当使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，您不需要担心这些问题，因为可以像将输入传递给任何其他Python函数一样传递输入！

    </Tip>

    参数:
        config ([`RobertaPreLayerNormConfig`]): 包含模型所有参数的配置类。使用配置文件初始化不会加载与模型相关的权重，只会加载配置。请查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = r"""
"""
# 使用装饰器添加文档字符串，描述这是 RoBERTa-PreLayerNorm 模型，输出原始隐藏状态而不带特定的顶层头部。
# ROBERTA_PRELAYERNORM_START_DOCSTRING 是一个预定义的文档字符串变量。
@add_start_docstrings(
    "The bare RoBERTa-PreLayerNorm Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaModel 复制而来，将类名中的 ROBERTA 替换为 ROBERTA_PRELAYERNORM，Roberta 替换为 RobertaPreLayerNorm，roberta 替换为 roberta_prelayernorm。
class TFRobertaPreLayerNormModel(TFRobertaPreLayerNormPreTrainedModel):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 RoBERTa-PreLayerNorm 的主要层，使用给定的配置，并命名为 "roberta_prelayernorm"。
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(config, name="roberta_prelayernorm")

    @unpack_inputs
    # 使用装饰器添加模型前向传播的文档字符串，描述输入参数的形状和用途。
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，包括模型的检查点、输出类型、配置类。
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
        # 这里省略了函数体的部分，需要在实际代码中完整添加。
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
        outputs = self.roberta_prelayernorm(
            input_ids=input_ids,  # 输入的token ID序列
            attention_mask=attention_mask,  # 注意力掩码，避免对填充的token进行注意力计算
            token_type_ids=token_type_ids,  # token类型ID，用于BERT模型中的segment embedding
            position_ids=position_ids,  # token位置ID，用于BERT模型中的position embedding
            head_mask=head_mask,  # 头部掩码，用于屏蔽某些注意力头
            inputs_embeds=inputs_embeds,  # 嵌入的输入向量，代替输入的token ID
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态，用于解码器的交叉注意力
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力掩码，用于解码器的交叉注意力
            past_key_values=past_key_values,  # 预先计算的注意力块的键值隐藏状态，用于解码时加速
            use_cache=use_cache,  # 是否使用缓存来加速解码
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否以字典形式返回输出
            training=training,  # 是否在训练模式下
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead复制而来，将Roberta改为RobertaPreLayerNorm
class TFRobertaPreLayerNormLMHead(keras.layers.Layer):
    """用于预层归一化的Roberta LM头部，用于掩码语言建模。"""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        # 创建一个全连接层，大小为config.hidden_size，使用给定范围的初始化器进行初始化
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个层归一化层，epsilon为config.layer_norm_eps
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 获取GELU激活函数
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个标记有一个仅输出的偏置
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 添加一个偏置，形状为(config.vocab_size,)，使用零初始化，可训练，命名为"bias"
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建了，直接返回
        if self.built:
            return
        self.built = True
        # 构建全连接层和层归一化层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        # 设置输出嵌入
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        # 设置偏置
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 应用层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 通过偏置将其投影回词汇表大小
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings(
    """带有`语言建模`头部的RoBERTa-PreLayerNorm模型。""", ROBERTA_PRELAYERNORM_START_DOCSTRING
)
class TFRobertaPreLayerNormForMaskedLM(TFRobertaPreLayerNormPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 当从PT模型加载TF模型时，带有'.'的名称表示授权的意外/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    # 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM.__init__ 复制而来，替换 ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 RobBERTa 预层归一化主层，不添加池化层，命名为 "roberta_prelayernorm"
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 初始化 RobBERTa 预层归一化语言模型头部，使用 roberta_prelayernorm 的嵌入层，命名为 "lm_head"
        self.lm_head = TFRobertaPreLayerNormLMHead(config, self.roberta_prelayernorm.embeddings, name="lm_head")

    # 返回 lm_head 属性，即 RobBERTa 预层归一化语言模型头部
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称的方法，已弃用，发出未来警告，建议使用 `get_bias` 替代
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回对象的名称加上 lm_head 的名称，作为前缀偏置名称
        return self.name + "/" + self.lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.69,
    )
    # 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM.call 复制而来，替换 ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
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
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 定义方法签名，指定输入参数和返回类型注解
        outputs = self.roberta_prelayernorm(
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

        # 获取模型输出的序列表示
        sequence_output = outputs[0]
        # 使用语言模型头部预测得分
        prediction_scores = self.lm_head(sequence_output)

        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不返回字典形式的结果，则按元组方式构造输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 对象，包含损失、预测 logits、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 Roberta 预层规范化，构建其结构
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果存在语言模型头部，构建其结构
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForCausalLM 复制并修改为 TFRobertaPreLayerNormForCausalLM，将 ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class TFRobertaPreLayerNormForCausalLM(TFRobertaPreLayerNormPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在从 PT 模型加载 TF 模型时，忽略以下名称的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    def __init__(self, config: RobertaPreLayerNormConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning(
                "If you want to use `TFRobertaPreLayerNormLMHeadModel` as a standalone, add `is_decoder=True.`"
            )

        # 初始化 RoBERTa 预层归一化主层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 初始化 RoBERTa 预层归一化语言模型头部
        self.lm_head = TFRobertaPreLayerNormLMHead(
            config, input_embeddings=self.roberta_prelayernorm.embeddings, name="lm_head"
        )

    def get_lm_head(self):
        # 返回语言模型头部
        return self.lm_head

    def get_prefix_bias_name(self):
        # 发出警告，此方法已弃用
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回名称前缀和偏置名
        return self.name + "/" + self.lm_head.name

    # 从 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果未提供注意力遮罩，则创建全为1的遮罩
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果使用过去的键值对，则截取 decoder_input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回生成模型输入所需的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 解包输入参数
    @unpack_inputs
    # 添加 ROBERTA_PRELAYERNORM 输入文档字符串
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义类方法 `build`，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在 `roberta_prelayernorm` 属性，则构建其相关层
        if getattr(self, "roberta_prelayernorm", None) is not None:
            # 使用 `roberta_prelayernorm` 的名称作为命名空间
            with tf.name_scope(self.roberta_prelayernorm.name):
                # 调用 `roberta_prelayernorm` 的 build 方法
                self.roberta_prelayernorm.build(None)
        
        # 如果存在 `lm_head` 属性，则构建其相关层
        if getattr(self, "lm_head", None) is not None:
            # 使用 `lm_head` 的名称作为命名空间
            with tf.name_scope(self.lm_head.name):
                # 调用 `lm_head` 的 build 方法
                self.lm_head.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaClassificationHead复制并修改为TFRobertaPreLayerNormClassificationHead
class TFRobertaPreLayerNormClassificationHead(keras.layers.Layer):
    """用于句子级分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，输出大小为config.hidden_size，使用指定的初始化器初始化权重，激活函数为tanh
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 根据配置设定分类器的dropout率，若未设定则使用隐藏层dropout率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个dropout层，用于在训练时随机断开输入单元，以防止过拟合
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 创建一个全连接层，输出大小为config.num_labels，使用指定的初始化器初始化权重
        self.out_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        # 取features中的第一个位置的特征向量，相当于取< s >标记（等效于[CLS]）
        x = features[:, 0, :]
        # 对x应用dropout，根据training参数确定是否在训练时使用
        x = self.dropout(x, training=training)
        # 通过全连接层dense处理x
        x = self.dense(x)
        # 再次应用dropout
        x = self.dropout(x, training=training)
        # 通过全连接层out_proj处理x，得到最终的输出
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在dense层，则构建dense层，指定输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在out_proj层，则构建out_proj层，指定输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm 模型转换器，顶部带有序列分类/回归头部（在汇聚输出之上的线性层），例如用于GLUE任务。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class TFRobertaPreLayerNormForSequenceClassification(
    TFRobertaPreLayerNormPreTrainedModel, TFSequenceClassificationLoss
):
    # 在从PT模型加载TF模型时，带'.'的名称表示授权的意外/缺少的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 设置类别数目
        self.num_labels = config.num_labels

        # 创建RoBERTa-PreLayerNorm的主层，不添加池化层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 创建分类器头部
        self.classifier = TFRobertaPreLayerNormClassificationHead(config, name="classifier")

    # 将@unpack_inputs应用于下面的函数
    # 将ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING的内容添加到模型前向传播的文档字符串中
    # 添加代码示例的文档字符串
    # 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForSequenceClassification.call 复制而来，将 roberta 替换为 roberta_prelayernorm
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
        # 调用 self.roberta_prelayernorm 方法进行前向传播，生成模型输出
        outputs = self.roberta_prelayernorm(
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 使用 self.classifier 对序列输出进行分类预测
        logits = self.classifier(sequence_output, training=training)

        # 如果提供了 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 不为 True，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFSequenceClassifierOutput 类型的对象
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
        # 标记模型已经构建
        self.built = True
        # 如果存在 self.roberta_prelayernorm 属性，则构建它
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果存在 self.classifier 属性，则构建它
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a multiple choice classification head on top (a linear layer on top of the pooled
    output and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMultipleChoice复制而来，将ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
class TFRobertaPreLayerNormForMultipleChoice(TFRobertaPreLayerNormPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 在加载PT模型时，'.'表示授权的不符合预期/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化模型的主要层，命名为'roberta_prelayernorm'
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(config, name="roberta_prelayernorm")
        # 添加dropout层，使用给定的隐藏层dropout概率
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 分类器层，使用给定的初始化范围初始化权重，输出维度为1
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置信息
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接受多种输入参数，具体见ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING
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
        # ...
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果存在 input_ids，则确定 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取选择数量
            seq_length = shape_list(input_ids)[2]   # 获取序列长度
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 否则，从 inputs_embeds 确定选择数量
            seq_length = shape_list(inputs_embeds)[2]   # 从 inputs_embeds 确定序列长度

        # 根据是否为 None，将 input_ids, attention_mask, token_type_ids, position_ids 进行扁平化处理
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None

        # 调用 self.roberta_prelayernorm 方法，传入参数扁平化后的输入数据以及其他可选参数
        outputs = self.roberta_prelayernorm(
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

        # 获取池化后的输出（通常是第二个元素）
        pooled_output = outputs[1]

        # 在训练时应用 dropout
        pooled_output = self.dropout(pooled_output, training=training)

        # 使用分类器对池化输出进行分类预测
        logits = self.classifier(pooled_output)

        # 将 logits 重新调整为原始形状，以便与输入匹配
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果没有提供 labels，则不计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不要求返回字典，则按照 tuple 的形式返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回一个 TFMultipleChoiceModelOutput 对象，包含损失、预测的 logits、隐藏状态和注意力权重
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return

        # 标记模型为已构建
        self.built = True

        # 如果存在 self.roberta_prelayernorm 属性，则构建其内部结构
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)

        # 如果存在 self.classifier 属性，则构建其内部结构
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
    """
    RoBERTa-PreLayerNorm Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """
    # 定义了一个 RoBERTa-PreLayerNorm 模型，用于标记分类任务，例如命名实体识别（NER）
    @add_start_docstrings(
        ROBERTA_PRELAYERNORM_START_DOCSTRING,
    )
    class TFRobertaPreLayerNormForTokenClassification(TFRobertaPreLayerNormPreTrainedModel, TFTokenClassificationLoss):
        # 当从 PyTorch 模型加载到 TF 模型时，以下名称表示可以忽略的意外/缺失层
        _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
        # 当从 PyTorch 模型加载到 TF 模型时，以下名称表示可以忽略的缺失层
        _keys_to_ignore_on_load_missing = [r"dropout"]

        def __init__(self, config, *inputs, **kwargs):
            # 调用父类构造函数初始化模型
            super().__init__(config, *inputs, **kwargs)
            # 设置标签的数量
            self.num_labels = config.num_labels

            # 初始化 RoBERTa-PreLayerNorm 主层，不包括池化层，命名为 "roberta_prelayernorm"
            self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
                config, add_pooling_layer=False, name="roberta_prelayernorm"
            )
            # 根据配置设置分类器的 dropout，如果未设置，则使用隐藏层的 dropout
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            # 添加一个 dropout 层
            self.dropout = keras.layers.Dropout(classifier_dropout)
            # 添加一个全连接层，用于分类，输出维度为标签的数量，初始化方法使用配置中的范围设置
            self.classifier = keras.layers.Dense(
                config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
            )
            # 保存配置对象
            self.config = config

        # 对模型的前向传播函数进行装饰，添加输入参数的说明文档
        @unpack_inputs
        @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFTokenClassifierOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        # 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForTokenClassification.call 复制，并将 robera->roberta_prelayernorm
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
        # 使用 Roberta 的预层归一化层处理输入数据，并返回输出结果
        outputs = self.roberta_prelayernorm(
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
        # 获取模型的序列输出
        sequence_output = outputs[0]

        # 在训练过程中对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output, training=training)
        # 将经过 dropout 处理后的序列输出送入分类器进行分类得到 logits
        logits = self.classifier(sequence_output)

        # 如果提供了标签，计算分类损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回字典格式的输出，则按照非字典格式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则构建 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果定义了 roberta_prelayernorm 层，则构建该层
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果定义了 classifier 层，则构建该层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
RoBERTa-PreLayerNorm Model with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 导入所需的模块和函数
@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm Model with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 定义 RoBERTa-PreLayerNorm 用于问答任务的模型类，继承自 TFRobertaPreLayerNormPreTrainedModel 和 TFQuestionAnsweringLoss
class TFRobertaPreLayerNormForQuestionAnswering(TFRobertaPreLayerNormPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 在从 PyTorch 模型加载到 TF 模型时，指定可以忽略的不匹配的层的名称列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    # 初始化方法，接收配置和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数目
        self.num_labels = config.num_labels

        # 初始化 RoBERTa-PreLayerNorm 主层，不包含池化层
        self.roberta_prelayernorm = TFRobertaPreLayerNormMainLayer(
            config, add_pooling_layer=False, name="roberta_prelayernorm"
        )
        # 初始化问答输出层，包含一个 Dense 层用于输出问题答案的 logits
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置对象
        self.config = config

    # 使用装饰器添加模型前向传播的注释和示例
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 复制自 transformers.models.roberta.modeling_tf_roberta.TFRobertaForQuestionAnswering.call，将 robera 改为 roberta_prelayernorm
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
        # 使用 TFQuestionAnsweringModelOutput 或 (start_logits, end_logits) 的元组作为返回类型注解

        # 将输入传递给 self.roberta_prelayernorm 模型的前向传播，并获取输出
        outputs = self.roberta_prelayernorm(
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
        # 从输出中提取序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给 self.qa_outputs 模型，获取问题回答的 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 沿着最后一个维度分割成 start_logits 和 end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除 start_logits 和 end_logits 的单维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 初始化损失为 None
        loss = None
        # 如果提供了 start_positions 和 end_positions，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 调用 hf_compute_loss 方法计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果 return_dict 为 False，则按非字典方式返回输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则按 TFQuestionAnsweringModelOutput 格式返回输出
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置标志位表示模型已经构建
        self.built = True
        # 如果 self.roberta_prelayernorm 存在，则构建其模型
        if getattr(self, "roberta_prelayernorm", None) is not None:
            with tf.name_scope(self.roberta_prelayernorm.name):
                self.roberta_prelayernorm.build(None)
        # 如果 self.qa_outputs 存在，则构建其模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```