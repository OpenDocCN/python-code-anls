# `.\models\roberta\modeling_tf_roberta.py`

```
# coding=utf-8
# 定义了文件编码格式为UTF-8

# 以下部分代码的版权归 Google AI Language Team 和 HuggingFace Inc. 团队所有，以及 NVIDIA 公司。保留所有权利。
# 根据 Apache License, Version 2.0 许可证使用本文件。您可以在以下网址获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0

# 如果没有符合适用法律的要求或书面同意，本软件是按“原样”提供的，不提供任何明示或暗示的担保或条件。
# 请参阅许可证以获取详细的权限说明和限制。
""" TF 2.0 RoBERTa 模型。"""

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
from .configuration_roberta import RobertaConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "FacebookAI/roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

# 支持的预训练模型列表
TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "FacebookAI/roberta-large-mnli",
    "distilbert/distilroberta-base",
    # 可查看所有 RoBERTa 模型列表：https://huggingface.co/models?filter=roberta
]

class TFRobertaEmbeddings(keras.layers.Layer):
    """
    BertEmbeddings 的变种，用于处理位置编码索引的微小调整。
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 填充索引设定为1
        self.padding_idx = 1
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        
        # LayerNormalization 层，使用配置中的 epsilon 参数初始化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # Dropout 层，使用配置中的 dropout 概率初始化
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
 good answer should meticulously annotate each line of the provided Python code block, adhering to the specified format. This includes adding comments that explain the purpose and functionality of each statement, ensuring clarity and completeness without altering the original code structure or indentation.

Here is a potential answer:


    def build(self, input_shape=None):
        # 开始构建词嵌入层
        with tf.name_scope("word_embeddings"):
            # 添加权重张量用于词嵌入
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 添加令牌类型嵌入层的权重张量
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入层的权重张量
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
                # 如果存在 LayerNorm 层，则构建它
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 创建一个掩码，标记非填充符号的位置
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 计算累积位置索引，从 past_key_values_length 开始
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


This annotation ensures that each method and operation within the provided class is clearly explained, promoting understanding and maintenance of the code.
    ):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)  # 检查输入的 input_ids 和 inputs_embeds 不能同时为空

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 检查 input_ids 是否在有效的词汇范围内
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)  # 根据 input_ids 从权重矩阵中获取对应的嵌入向量

        input_shape = shape_list(inputs_embeds)[:-1]  # 获取输入嵌入向量的形状，去除最后一个维度（通常是嵌入维度）

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)  # 如果 token_type_ids 为空，则用零填充形状与输入嵌入向量相同的张量

        if position_ids is None:
            if input_ids is not None:
                # 根据输入的 token ids 创建位置 ids，保留任何填充的标记
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 如果 input_ids 为空，则创建位置 ids，从 padding_idx 开始，长度为输入形状的最后一个维度加上 padding_idx
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)  # 根据位置 ids 从位置嵌入矩阵中获取位置嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)  # 根据 token_type_ids 获取 token type 嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds  # 计算最终的嵌入向量，包括输入嵌入、位置嵌入和 token type 嵌入
        final_embeddings = self.LayerNorm(inputs=final_embeddings)  # 使用 LayerNorm 对最终的嵌入向量进行归一化
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)  # 使用 dropout 对最终的嵌入向量进行随机失活

        return final_embeddings  # 返回最终的嵌入向量作为输出
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->Roberta
class TFRobertaPooler(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层用于池化操作，输出维度为config.hidden_size，激活函数为tanh
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过取第一个 token 对应的隐藏状态来进行“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，直接返回；否则，根据配置信息构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->Roberta
class TFRobertaSelfAttention(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 计算每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建查询、键、值的全连接层
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

        # 是否为解码器层
        self.is_decoder = config.is_decoder
        self.config = config
    # 将张量重塑从 [batch_size, seq_length, all_head_size] 到 [batch_size, seq_length, num_attention_heads, attention_head_size]
    tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

    # 将张量转置从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
    return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 检查是否已经构建模型，如果已经构建则直接返回
    if self.built:
        return
    self.built = True

    # 如果存在查询（query）张量，则构建查询张量
    if getattr(self, "query", None) is not None:
        with tf.name_scope(self.query.name):
            self.query.build([None, None, self.config.hidden_size])

    # 如果存在键（key）张量，则构建键张量
    if getattr(self, "key", None) is not None:
        with tf.name_scope(self.key.name):
            self.key.build([None, None, self.config.hidden_size])

    # 如果存在值（value）张量，则构建值张量
    if getattr(self, "value", None) is not None:
        with tf.name_scope(self.value.name):
            self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->Roberta
class TFRobertaSelfOutput(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于映射隐藏状态到与配置中指定大小相同的空间
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义一个层归一化层，用于归一化输入数据，设置了配置中的 epsilon 参数
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义一个 dropout 层，用于在训练时随机丢弃部分输入数据，以减少过拟合风险
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的隐藏状态通过全连接层映射到指定大小的空间
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用 dropout，随机丢弃部分输入数据
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将 dropout 后的结果与输入数据进行残差连接，并通过层归一化层处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建全连接层，指定输入形状为 [None, None, hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建层归一化层，指定输入形状为 [None, None, hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->Roberta
class TFRobertaAttention(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 使用 TFRobertaSelfAttention 定义自注意力层
        self.self_attention = TFRobertaSelfAttention(config, name="self")
        # 使用 TFRobertaSelfOutput 定义自注意力输出层
        self.dense_output = TFRobertaSelfOutput(config, name="output")

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
        # 调用自注意力层处理输入张量
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
        # 调用自注意力输出层处理自注意力层的输出，得到注意力输出张量
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力，将注意力张量加入输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义一个方法 `build`，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示已经构建了该层
        self.built = True
        
        # 如果存在 self_attention 属性，进行以下操作
        if getattr(self, "self_attention", None) is not None:
            # 使用 `tf.name_scope` 创建一个作用域，作用域名称为 self_attention 的名称
            with tf.name_scope(self.self_attention.name):
                # 调用 self_attention 对象的 build 方法，传入 input_shape=None
                self.self_attention.build(None)
        
        # 如果存在 dense_output 属性，进行以下操作
        if getattr(self, "dense_output", None) is not None:
            # 使用 `tf.name_scope` 创建一个作用域，作用域名称为 dense_output 的名称
            with tf.name_scope(self.dense_output.name):
                # 调用 dense_output 对象的 build 方法，传入 input_shape=None
                self.dense_output.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制过来，将Bert改为Roberta
class TFRobertaIntermediate(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出大小为config.intermediate_size，使用给定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config.hidden_act是字符串，则使用对应的TensorFlow激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的hidden_states传递给全连接层，得到输出
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数对输出进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了dense层，则按照指定的形状构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertOutput复制过来，将Bert改为Roberta
class TFRobertaOutput(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出大小为config.hidden_size，使用给定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，epsilon为config.layer_norm_eps，用于归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，dropout比率为config.hidden_dropout_prob，用于正则化
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的hidden_states传递给全连接层，得到输出
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时使用dropout进行正则化
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对加和原始输入的hidden_states应用LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了dense层，则按照指定的形状构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果已经构建了LayerNorm层，则按照指定的形状构建它
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertLayer复制过来，将Bert改为Roberta
class TFRobertaLayer(keras.layers.Layer):
    # 初始化函数，用于创建一个 RoBERTa 模型的实例
    def __init__(self, config: RobertaConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 RoBERTa 的自注意力层，并命名为 "attention"
        self.attention = TFRobertaAttention(config, name="attention")
        
        # 检查是否为解码器模型
        self.is_decoder = config.is_decoder
        
        # 检查是否添加了跨注意力
        self.add_cross_attention = config.add_cross_attention
        
        # 如果模型添加了跨注意力但不是解码器，抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建 RoBERTa 的跨注意力层，并命名为 "crossattention"
            self.crossattention = TFRobertaAttention(config, name="crossattention")
        
        # 创建 RoBERTa 的中间层
        self.intermediate = TFRobertaIntermediate(config, name="intermediate")
        
        # 创建 RoBERTa 的输出层
        self.bert_output = TFRobertaOutput(config, name="output")
        ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键/值存在，则从中提取自注意力的过去键/值元组的前两个位置；否则设为None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self.attention方法计算自注意力
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,  # 输入张量
            attention_mask=attention_mask,  # 注意力掩码
            head_mask=head_mask,  # 头部掩码
            encoder_hidden_states=None,  # 编码器隐藏状态（自注意力时为None）
            encoder_attention_mask=None,  # 编码器注意力掩码（自注意力时为None）
            past_key_value=self_attn_past_key_value,  # 过去的键/值，用于缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            training=training,  # 是否处于训练模式
        )
        attention_output = self_attention_outputs[0]  # 注意力输出的第一个元素

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 解码器时，输出除了最后一个元素之外的所有元素
            present_key_value = self_attention_outputs[-1]  # 解码器时，最后一个元素为当前的键/值元组
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
                                                  
        cross_attn_present_key_value = None
        # 如果是解码器且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 如果没有定义交叉注意力层，则抛出错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 从过去的键/值中提取交叉注意力的键/值元组，位置为过去键/值元组的倒数第二个和最后一个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用self.crossattention方法计算交叉注意力
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,  # 输入张量为自注意力输出
                attention_mask=attention_mask,  # 注意力掩码
                head_mask=head_mask,  # 头部掩码
                encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                past_key_value=cross_attn_past_key_value,  # 过去的键/值，用于缓存
                output_attentions=output_attentions,  # 是否输出注意力权重
                training=training,  # 是否处于训练模式
            )
            attention_output = cross_attention_outputs[0]  # 交叉注意力的输出为第一个元素
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力的当前键/值添加到当前键/值元组中的第三和第四个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用self.intermediate方法计算中间输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用self.bert_output方法计算BERT输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output,  # 中间隐藏状态
            input_tensor=attention_output,  # 输入张量为自注意力输出
            training=training  # 是否处于训练模式
        )
        outputs = (layer_output,) + outputs  # 如果输出注意力，则添加到输出元组中

        # 如果是解码器，将注意力的键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建函数用于构造模型，接收输入形状参数
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回，不进行重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在注意力层，根据其名称创建命名作用域并构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层，根据其名称创建命名作用域并构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在BERT输出层，根据其名称创建命名作用域并构建BERT输出层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        
        # 如果存在交叉注意力层，根据其名称创建命名作用域并构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制并改为使用 Roberta
class TFRobertaEncoder(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化 RobeertaEncoder 层的每个子层，命名为 layer_._{i}
        self.layer = [TFRobertaLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        # 如果要输出隐藏状态，则初始化 all_hidden_states 为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果要输出注意力，则初始化 all_attentions 为空元组
        all_attentions = () if output_attentions else None
        # 如果要输出交叉注意力且配置允许，则初始化 all_cross_attentions 为空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果 use_cache 为真，则初始化 next_decoder_cache 为空元组
        next_decoder_cache = () if use_cache else None
        # 遍历每一层的编码器
        for i, layer_module in enumerate(self.layer):
            # 如果要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的过去键值，如果存在的话
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的模块进行前向传播
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
            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果 use_cache 为真，则更新 next_decoder_cache 为当前层的最后一个输出
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果要输出注意力，则将当前层的注意力添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置允许添加交叉注意力且存在编码器隐藏状态，则将当前层的交叉注意力添加到 all_cross_attentions 中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回所有非空的结果元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回 TFBaseModelOutputWithPastAndCrossAttentions 类型的字典结果
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义一个方法用于构建神经网络层次结构，接受输入形状参数，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果存在层列表属性
        if getattr(self, "layer", None) is not None:
            # 遍历每个层并设置 TensorFlow 的命名空间
            for layer in self.layer:
                # 使用每个层的名称作为 TensorFlow 的命名空间
                with tf.name_scope(layer.name):
                    # 调用每个层的 build 方法，传入 None 作为输入形状
                    layer.build(None)
# 使用 keras_serializable 装饰器标记这个类，使其可以被 Keras 序列化
@keras_serializable
class TFRobertaMainLayer(keras.layers.Layer):
    # 将 config_class 属性设置为 RobertaConfig 类，用于配置模型
    config_class = RobertaConfig

    # 初始化函数，接受 config 和其他参数
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 设置 self.is_decoder 标志位，表示是否为解码器
        self.is_decoder = config.is_decoder

        # 初始化其他属性，从 config 中获取相关配置
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 创建 TFRobertaEncoder 对象，并命名为 "encoder"
        self.encoder = TFRobertaEncoder(config, name="encoder")

        # 如果 add_pooling_layer 为 True，则创建 TFRobertaPooler 对象，并命名为 "pooler"
        self.pooler = TFRobertaPooler(config, name="pooler") if add_pooling_layer else None
        
        # 创建 TFRobertaEmbeddings 对象，并命名为 "embeddings"，这必须是最后声明的，以保持权重顺序
        self.embeddings = TFRobertaEmbeddings(config, name="embeddings")

    # 从 transformers 库中复制的方法：获取输入嵌入层对象
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 从 transformers 库中复制的方法：设置输入嵌入层的权重
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 从 transformers 库中复制的方法：剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 使用 unpack_inputs 装饰器，从 transformers 库中复制的方法：调用模型的主要处理逻辑
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
        ):
        # 实际的方法体暂未提供
        pass
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 标记模型为已构建状态
    self.built = True
    
    # 如果存在编码器（encoder），则在其命名空间下构建编码器
    if getattr(self, "encoder", None) is not None:
        with tf.name_scope(self.encoder.name):
            self.encoder.build(None)
    
    # 如果存在池化器（pooler），则在其命名空间下构建池化器
    if getattr(self, "pooler", None) is not None:
        with tf.name_scope(self.pooler.name):
            self.pooler.build(None)
    
    # 如果存在嵌入层（embeddings），则在其命名空间下构建嵌入层
    if getattr(self, "embeddings", None) is not None:
        with tf.name_scope(self.embeddings.name):
            self.embeddings.build(None)
# TFRobertaPreTrainedModel 类，用于处理权重初始化以及预训练模型的下载和加载接口
class TFRobertaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类，指定为 RobertaConfig
    config_class = RobertaConfig
    # 基础模型前缀，指定为 "roberta"
    base_model_prefix = "roberta"


# ROBERTA_START_DOCSTRING 常量，包含以下注释内容
ROBERTA_START_DOCSTRING = r"""

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

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ROBERTA_INPUTS_DOCSTRING 常量，尚未注释，待后续添加相关内容
ROBERTA_INPUTS_DOCSTRING = r"""
"""

# 使用 add_start_docstrings 装饰器，添加注释说明到 TFRobertaModel 类
@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class TFRobertaModel(TFRobertaPreTrainedModel):
    # TFRobertaPreTrainedModel 类的子类，继承其功能和特性
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传递config以及其他输入参数
        super().__init__(config, *inputs, **kwargs)
        # 创建一个名为roberta的TFRobertaMainLayer层，并用config配置它
        self.roberta = TFRobertaMainLayer(config, name="roberta")

    # 装饰器：将输入参数解包并传递给函数
    @unpack_inputs
    # 装饰器：为模型的前向传播函数添加描述性文档
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 装饰器：为模型添加代码示例的描述文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 示例中的检查点说明
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,  # 输出类型的说明
        config_class=_CONFIG_FOR_DOC,  # 示例中的配置类说明
    )
    # 定义模型的前向传播函数，接收多个输入参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 嵌入的输入
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器的隐藏状态
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器的注意力掩码
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典类型结果
        training: Optional[bool] = False,  # 是否处于训练模式
        # 下面没有更多的输入参数了，这里只是列出所有可能的输入参数
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
        # 调用 self.roberta 的前向传播，传入各种参数，包括输入的编码器隐藏状态、注意力掩码等
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 RoBERTa 模型的输出
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果 self.roberta 已经初始化
        if getattr(self, "roberta", None) is not None:
            # 在命名空间下构建 self.roberta 模型
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
class TFRobertaLMHead(keras.layers.Layer):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 存储模型配置信息
        self.hidden_size = config.hidden_size  # 提取隐藏层大小
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )  # 创建全连接层，大小与隐藏层一致
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.act = get_tf_activation("gelu")  # 获取激活函数 GELU

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings  # 存储输入的嵌入层权重作为解码器的权重

    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        # 创建偏置项，形状为词汇表大小，初始化为零，可训练

        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在全连接层，构建全连接层

        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果存在层归一化层，构建层归一化层

    def get_output_embeddings(self):
        return self.decoder
        # 返回嵌入层权重作为输出的解码器权重

    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]
        # 设置解码器的权重为给定值，并更新词汇表大小

    def get_bias(self):
        return {"bias": self.bias}
        # 返回偏置项

    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
        # 设置偏置项，并更新配置中的词汇表大小信息

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 全连接层
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states = self.layer_norm(hidden_states)  # 层归一化

        # project back to size of vocabulary with bias
        seq_length = shape_list(tensor=hidden_states)[1]  # 获取序列长度
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])  # 重塑隐藏状态
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)  # 矩阵乘法
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])  # 重塑隐藏状态
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)  # 添加偏置项

        return hidden_states
        # 返回最终的隐藏状态
    # 初始化方法，接受配置和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 使用TF-Roberta的主层，不添加池化层，命名为"roberta"
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 使用TF-Roberta的LM头部，连接到self.roberta的嵌入，命名为"lm_head"
        self.lm_head = TFRobertaLMHead(config, self.roberta.embeddings, name="lm_head")

    # 返回LM头部模型
    def get_lm_head(self):
        return self.lm_head

    # 返回带有前缀偏差名称的字符串
    def get_prefix_bias_name(self):
        # 发出警告，说明方法get_prefix_bias_name已被弃用
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回字符串，包含对象名称和LM头部名称
        return self.name + "/" + self.lm_head.name

    # 调用方法，用于处理输入和生成输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用RoBERTa模型进行前向传播
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
        # 使用LM头部生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        # 如果没有标签，则损失为None；否则使用标签和预测分数计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不返回字典，则返回包含预测分数和可能的隐藏状态的元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFMaskedLMOutput对象，包含损失、预测分数、隐藏状态和注意力
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义神经网络模型的构建方法，input_shape参数可选
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，不重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 检查是否存在名为"roberta"的属性，并且不为None
        if getattr(self, "roberta", None) is not None:
            # 在TensorFlow中，使用name_scope为模型组织命名空间
            with tf.name_scope(self.roberta.name):
                # 构建self.roberta模型，传入None作为输入形状
                self.roberta.build(None)
        
        # 检查是否存在名为"lm_head"的属性，并且不为None
        if getattr(self, "lm_head", None) is not None:
            # 在TensorFlow中，使用name_scope为模型组织命名空间
            with tf.name_scope(self.lm_head.name):
                # 构建self.lm_head模型，传入None作为输入形状
                self.lm_head.build(None)
    # TFRobertaForCausalLM 类继承自 TFRobertaPreTrainedModel 和 TFCausalLanguageModelingLoss
    class TFRobertaForCausalLM(TFRobertaPreTrainedModel, TFCausalLanguageModelingLoss):
        # 在从 PyTorch 模型加载 TF 模型时，忽略的层的名称列表，包括一些预期之外的/缺失的层
        _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

        def __init__(self, config: RobertaConfig, *inputs, **kwargs):
            # 调用父类的构造函数，并传入配置和其他输入参数
            super().__init__(config, *inputs, **kwargs)

            # 如果配置不是 decoder 类型，发出警告提示
            if not config.is_decoder:
                logger.warning("If you want to use `TFRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

            # 创建 RoBERTa 主体层，不包括 pooling 层，命名为 "roberta"
            self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
            # 创建 RoBERTa LM 头部层，传入 RoBERTa embeddings 作为输入，命名为 "lm_head"
            self.lm_head = TFRobertaLMHead(config, input_embeddings=self.roberta.embeddings, name="lm_head")

        # 返回 LM 头部层对象的方法
        def get_lm_head(self):
            return self.lm_head

        # 获取前缀偏置名称的方法，已经过时，将发出未来警告提示
        def get_prefix_bias_name(self):
            warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
            # 返回拼接的名称，包括实例名称和 LM 头部层名称
            return self.name + "/" + self.lm_head.name

        # 从 transformers 库中复制的方法，准备生成输入，用于生成文本的准备工作
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            # 获取输入的形状信息
            input_shape = input_ids.shape
            # 如果没有提供注意力遮罩，则创建全 1 的注意力遮罩
            if attention_mask is None:
                attention_mask = tf.ones(input_shape)

            # 如果使用了过去的键值对，则截取最后一个输入的 token ID
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            # 返回包含生成所需输入的字典
            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

        # 将输入解包的装饰器，用于 call 方法
        @unpack_inputs
        # 将 ROBERTA_INPUTS_DOCSTRING 格式化应用到模型前向传播的参数说明上
        @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        # 添加代码示例的文档字符串，包括检查点、输出类型、配置类
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFCausalLMOutputWithCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
        # 模型的前向传播方法，接受多个输入参数，输出模型的结果
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
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    # 定义模型构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型中包含名为 "roberta" 的属性
        if getattr(self, "roberta", None) is not None:
            # 使用 "roberta" 属性的名字作为命名空间，构建 "roberta" 子模型
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果模型中包含名为 "lm_head" 的属性
        if getattr(self, "lm_head", None) is not None:
            # 使用 "lm_head" 属性的名字作为命名空间，构建 "lm_head" 子模型
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
class TFRobertaClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 定义一个全连接层，用于分类任务，输入大小为 config.hidden_size
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 设置分类器的 dropout 层，使用 config.classifier_dropout 或者 config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 定义输出投影层，输出大小为 config.num_labels
        self.out_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        # 取输入 features 的第一个 token 的特征作为输入，相当于取 <s> token (对应 [CLS])
        x = features[:, 0, :]
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 dense 层已经存在，则建立 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 out_proj 层已经存在，则建立 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForSequenceClassification(TFRobertaPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化模型，设置分类数量
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 主层，不添加池化层，命名为 "roberta"
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化分类器头部
        self.classifier = TFRobertaClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 使用 RoBERTa 模型处理输入数据
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
        # 提取 RoBERTa 模型的输出中的序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类，得到 logits
        logits = self.classifier(sequence_output, training=training)

        # 如果没有提供 labels，则不计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回扁平化的输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

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
        # 标记模型已经构建
        self.built = True
        # 如果存在 RoBERTa 模型，则构建 RoBERTa
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForMultipleChoice(TFRobertaPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # Initialize the Roberta main layer with the provided configuration
        self.roberta = TFRobertaMainLayer(config, name="roberta")
        # Dropout layer with a dropout rate set according to the configuration
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # Classifier dense layer for multiple choice tasks, with 1 output unit
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # Store the configuration for reference
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        # Function defining the forward pass of the model with multiple choice inputs
        # Details the inputs and expected outputs in the documentation
        # Uses specified configurations for checkpoint, output type, and configuration class
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果存在 `input_ids`，获取其第二和第三维的尺寸
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，使用 `inputs_embeds` 的第二和第三维的尺寸
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入张量展平为二维张量，如果存在的话
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        
        # 使用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
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
        
        # 提取池化后的输出表示
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        
        # 将池化后的输出送入分类器得到 logits
        logits = self.classifier(pooled_output)
        
        # 将 logits 重塑为预期的形状
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果提供了标签 `labels`，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果 `return_dict` 为 False，返回扁平化后的 logits 和可能的额外输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 `return_dict` 为 True，返回 TFMultipleChoiceModelOutput 对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型的方法，设置 RoBERTa 和分类器的结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果存在 RoBERTa 模型，则构建其结构
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        
        # 如果存在分类器，则构建其结构，包括指定隐藏层的大小
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 导入所需模块和类
class TFRobertaForTokenClassification(TFRobertaPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 指定加载时忽略的不符合预期的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    # 指定加载时忽略的缺失层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置分类的标签数量
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 主层
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 获取分类器的 dropout 率，如果未指定，则使用隐藏层的 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 dropout 层
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 定义分类器层
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置信息
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        **kwargs,
    ):
        # 调用 RoBERTa 模型，传递输入参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
            **kwargs,
        )

        # 获取 RoBERTa 的输出 hidden states
        sequence_output = outputs[0]

        # 对输出进行 dropout
        sequence_output = self.dropout(sequence_output, training=training)

        # 经过分类器层得到 logits
        logits = self.classifier(sequence_output)

        # 根据需求返回不同的输出格式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((output,) + outputs[2:]) if output else output
        return TFTokenClassifierOutput(logits=logits, hidden_states=outputs.hidden_states)
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 RoBERTa 模型进行前向传播，获取输出结果
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
        # 从 RoBERTa 输出的元组中获取第一个元素，即模型的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 Dropout 操作，用于防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 Dropout 后的输出输入到分类器中，得到分类器的 logits
        logits = self.classifier(sequence_output)

        # 如果没有提供标签，则损失置为 None；否则使用损失计算函数计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict=False，则按非字典格式返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，则按 TFTokenClassifierOutput 类型返回输出
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型具有 RoBERTa 属性，则构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果模型具有分类器属性，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 引入 RoBERTa 模型，添加了一个用于抽取式问答任务的跨度分类头部，例如 SQuAD 数据集。该头部是在隐藏状态输出之上的线性层，
# 用于计算 `span start logits` 和 `span end logits`。

class TFRobertaForQuestionAnswering(TFRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义了在从 PyTorch 模型加载到 TensorFlow 模型时，可以忽略的不匹配的层名列表。
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 主层，不包含池化层，命名为 "roberta"
        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        
        # 初始化用于问答任务输出的全连接层，输出大小为 config.num_labels
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    # 定义模型的前向传播方法，支持一系列输入参数和注释文档
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
        # 调用 RoBERTa 模型进行前向传播，获取输出结果
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
        # 从 RoBERTa 输出的结果中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入 QA 输出层，得到起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 如果提供了起始位置和结束位置的标签，则计算损失值
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果 return_dict=False，则返回不同的输出形式
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，则返回 TFQuestionAnsweringModelOutput 类的对象
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

        # 如果模型中包含 RoBERTa 层，则构建 RoBERTa 层
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)

        # 如果模型中包含 QA 输出层，则构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```