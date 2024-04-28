# `.\transformers\models\rembert\modeling_tf_rembert.py`

```
# 设置文件的编码格式为 UTF-8
# 版权声明
#
# 此代码版权归 The HuggingFace Team 和 The HuggingFace Inc. 团队所有，保留所有权利。
#
# 在遵守许可证的情况下，您可以使用此文件，许可证获取方式如下：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据 "AS IS" 的基础分发软件，
# 不提供任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言下权限和限制
""" 
TF 2.0 RemBERT 模型
"""


from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入所需的自定义模块
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
from .configuration_rembert import RemBertConfig

# 获取日志的 Logger 实例
logger = logging.get_logger(__name__)

# 用于文档的配置示例名称
_CONFIG_FOR_DOC = "RemBertConfig"

# RemBERT 预训练模型存档列表
TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/rembert",
    # 查看所有 RemBERT 模型：https://huggingface.co/models?filter=rembert
]


# 定义 TFRemBertEmbeddings 类
class TFRemBertEmbeddings(tf.keras.layers.Layer):
    """从单词、位置和令牌类型嵌入构造嵌入。"""

    def __init__(self, config: RemBertConfig, **kwargs):
        # 初始化函数
        super().__init__(**kwargs)

        # 保存配置
        self.config = config
        self.input_embedding_size = config.input_embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        # LayerNormalization 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 建立嵌入层的权重参数
    def build(self, input_shape=None):
        # 使用名称范围创建名为"word_embeddings"的范围
        with tf.name_scope("word_embeddings"):
            # 添加权重参数，表示单词的嵌入
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.input_embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 使用名称范围创建名为"token_type_embeddings"的范围
        with tf.name_scope("token_type_embeddings"):
            # 添加权重参数，表示标记类型的嵌入
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.input_embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 使用名称范围创建名为"position_embeddings"的范围
        with tf.name_scope("position_embeddings"):
            # 添加权重参数，表示位置的嵌入
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.input_embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 LayerNorm，则根据输入形状构建 LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.input_embedding_size])

    # 嵌入层的调用函数
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        # 应用根据输入张量进行嵌入

        # 断言输入张量不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果有输入的单词索引
        if input_ids is not None:
            # 检查嵌入是否在范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 从权重中根据索引获取嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果没有标记类型，则设置为默认值0
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果没有位置索引，则自动生成
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 从位置嵌入中获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 从标记类型嵌入中获取标记类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 融合输入嵌入、位置嵌入和标记类型嵌入
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对融合后的嵌入进行 LayerNorm
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对融合后的嵌入进行 dropout，如果训练则为True
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终嵌入向量
        return final_embeddings
# 定义自定义层类，用于实现多头自注意力机制。该类继承自 Keras 层。
class TFRemBertSelfAttention(tf.keras.layers.Layer):
    # 构造函数，接收 RemBertConfig 对象作为配置，支持额外的关键字参数。
    def __init__(self, config: RemBertConfig, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)

        # 确保 hidden_size 是 num_attention_heads 的整数倍，否则抛出异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算总共的注意力头大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算注意力头大小的平方根
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 定义用于生成查询向量的全连接层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, 
            kernel_initializer=get_initializer(config.initializer_range), 
            name="query"
        )
        # 定义用于生成键向量的全连接层
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, 
            kernel_initializer=get_initializer(config.initializer_range), 
            name="key"
        )
        # 定义用于生成值向量的全连接层
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, 
            kernel_initializer=get_initializer(config.initializer_range), 
            name="value"
        )
        # 定义用于注意力概率的 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 保存配置
        self.config = config

    # 转置张量以适应注意力计算的维度需求
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 转置张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 构建层
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果 query 属性存在，则构建 query 层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果 key 属性存在，则构建 key 层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果 value 属性存在，则构建 value 层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 从 transformers.models.bert.modeling_tf_bert.TFBertSelfOutput 复制代码，并将Bert->RemBert
class TFRemBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个密集网络层，单元数为config.hidden_size，初始化方式为config中的initializer_range
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建LayerNormalization层，epsilon为config中的值，名称为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，rate为config中的hidden_dropout_prob值
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存config对象
        self.config = config

    # 执行层的调用，处理hidden_states
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过密集网络层处理hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 通过Dropout层处理hidden_states
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 通过LayerNormalization层处理hidden_states和input_tensor的和
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建层，根据输入形状构建dense和LayerNorm层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在self.dense的话，在其name scope下根据形状构建dense
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在self.LayerNorm的话，在其name scope下根据形状构建LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertAttention 复制代码，并将Bert->RemBert
class TFRemBertAttention(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建TFRemBertSelfAttention对象，使用给定的config和名称"self"
        self.self_attention = TFRemBertSelfAttention(config, name="self")
        # 创建TFRemBertSelfOutput对象，使用给定的config和名称"output"
        self.dense_output = TFRemBertSelfOutput(config, name="output")

    # 剪枝功能，暂未���现
    def prune_heads(self, heads):
        raise NotImplementedError

    # 调用函数，根据输入参数执行自注意力和输出操作
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
        # 执行self_attention模块操作
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
        # 执行dense_output模块操作
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果输出attentions，将其和past_key_value一起添加到outputs
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 构建神经网络层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建，直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 若存在自注意力机制，构建自注意力层
        if getattr(self, "self_attention", None) is not None:
            # 使用名称作用域构建自注意力层
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 若存在密集输出层，构建密集输出层
        if getattr(self, "dense_output", None) is not None:
            # 使用名称作用域构建密集输出层
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 复制并修改为 RemBert 的中间层
class TFRemBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出大小为 config.intermediate_size，使用指定的初始化器初始化参数，命名为 "dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果 hidden_act 是字符串，则从名称获取激活函数，否则使用配置中指定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建了，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建全连接层，指定输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 复制并修改为 RemBert 的输出层
class TFRemBertOutput(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出大小为 config.hidden_size，使用指定的初始化器初始化参数，命名为 "dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，epsilon 为 config.layer_norm_eps，命名为 "LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 Dropout 层进行正则化
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 LayerNormalization 层进行残差连接，并加上输入张量
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建了，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建全连接层，指定输入形状为 [None, None, self.config.intermediate_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在 LayerNormalization 层，则构建 LayerNormalization 层，指定输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertLayer 复制并修改为 RemBert 的层
class TFRemBertLayer(tf.keras.layers.Layer):
    # 初始化方法，接受一个RemBertConfig的实例和可选参数
    def __init__(self, config: RemBertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建注意力层对象
        self.attention = TFRemBertAttention(config, name="attention")
        # 获取是否为解码器模型的标志
        self.is_decoder = config.is_decoder
        # 获取是否添加交叉注意力的标志
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建交叉注意力层对象
            self.crossattention = TFRemBertAttention(config, name="crossattention")
        # 创建中间层对象
        self.intermediate = TFRemBertIntermediate(config, name="intermediate")
        # 创建BERT输出层对象
        self.bert_output = TFRemBertOutput(config, name="output")

    # 调用方法，接受多个输入参数
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
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果存在过去的键值对，则将解码器自注意力的缓存键/值元组放在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力机制计算
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

        # 如果是解码器，则最后的输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果要输出注意力权重，则添加自注意力

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 跨注意力缓存的键/值元组在过去的键值对元组的位置3,4处
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行跨注意力计算
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
            # 如果要输出注意力权重，则添加跨注意力
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将跨注意力缓存添加到现有的键/值元组中的第3,4个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果输出则添加注意力

        # 如果是解码器，则将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建网络层，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self.attention 属性，构建 attention 层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在 self.intermediate 属性，构建 intermediate 层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 self.bert_output 属性，构建 bert_output 层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果存在 self.crossattention 属性，构建 crossattention 层
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 定义 TFRemBertEncoder 类，用于实现 RemBert 模型的编码器部分
class TFRemBertEncoder(tf.keras.layers.Layer):
    # 初始化方法，接受配置参数 config 和其他关键字参数
    def __init__(self, config: RemBertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存传入的配置参数
        self.config = config

        # 创建一个全连接层，用于将输入嵌入到隐藏空间中
        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(
            units=config.hidden_size,  # 隐藏层单元数为配置参数中的隐藏大小
            kernel_initializer=get_initializer(config.initializer_range),  # 使用配置中的初始化器初始化权重矩阵
            name="embedding_hidden_mapping_in",  # 层的名称
        )
        # 创建多个 RemBertLayer 层，用于组成编码器的层
        self.layer = [TFRemBertLayer(config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]

    # 定义 call 方法，用于定义层的前向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量
        head_mask: tf.Tensor,  # 头部掩码张量
        encoder_hidden_states: tf.Tensor,  # 编码器隐藏状态张量
        encoder_attention_mask: tf.Tensor,  # 编码器注意力掩码张量
        past_key_values: Tuple[Tuple[tf.Tensor]],  # 过去的键值对
        use_cache: bool,  # 是否使用缓存
        output_attentions: bool,  # 是否输出注意力权重
        output_hidden_states: bool,  # 是否输出隐藏状态
        return_dict: bool,  # 是否返回字典格式结果
        training: bool = False,  # 是否处于训练模式
    # 定义 decoder 方法，接收隐藏状态作为输入，返回输出或包含交叉注意力等信息的元组
    def decoder(
        hidden_states: tf.Tensor,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[Tuple[tf.Tensor]] = None,
        encoder_hidden_states: Optional[tf.Tensor] = None,
        encoder_attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 将隐藏状态投影到嵌入空间
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        # 初始化存储所有隐藏状态的变量
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 初始化下一个解码器缓存变量
        next_decoder_cache = () if use_cache else None
        # 遍历所有的解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要保存隐藏状态，添加当前隐藏状态到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取先前保存的键值对（如果有）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 在解码器层中执行操作
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
            # 更新隐藏状态
            hidden_states = layer_outputs[0]

            # 如果使用缓存，将当前层的输出添加到缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要保存注意力分布，将当前层的注意力分布添加到列表中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果需要交叉注意力并且有编码器隐藏状态，则添加交叉注意力分布到列表中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，返回元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回 TFBaseModelOutputWithPastAndCrossAttentions 对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建了模型，直接返回
        if self.built:
            return
        # 将已构建标志置为 True
        self.built = True
        # 构建嵌入层或解码器层
        if getattr(self, "embedding_hidden_mapping_in", None) is not None:
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                self.embedding_hidden_mapping_in.build([None, None, self.config.input_embedding_size])
        if getattr(self, "layer", None) is not None:
            # 遍历所有解码器层并构建
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler中复制代码并将Bert->RemBert
class TFRemBertPooler(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为config.hidden_size，使用config.initializer_range作为初始化器，激活函数为"tanh"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config
    
    # 根据传入的hidden_states进行调用，返回经过池化处理后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 我们通过简单地取对应于第一个标记的隐藏状态来对模型进行"池化"
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output
    
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])

# 创建TFRemBertLMPredictionHead类，继承自tf.keras.layers.Layer
class TFRemBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.initializer_range = config.initializer_range
        self.output_embedding_size = config.output_embedding_size
        # 创建一个全连接层，输出维度为config.output_embedding_size，使用config.initializer_range作为初始化器
        self.dense = tf.keras.layers.Dense(
            config.output_embedding_size, kernel_initializer=get_initializer(self.initializer_range), name="dense"
        )
        # 确定激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        # 创建LayerNormalization层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def build(self, input_shape=None):
        # 创建解码器权重
        self.decoder = self.add_weight(
            name="decoder/weight",
            shape=[self.config.vocab_size, self.output_embedding_size],
            initializer=get_initializer(self.initializer_range),
        )
        # 创建解码器偏置
        self.decoder_bias = self.add_weight(
            shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="decoder/bias"
        )

        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建LayerNormalization层
                self.LayerNorm.build([None, self.config.output_embedding_size])

    # 获取输出嵌入
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self
    
    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.decoder = value
        self.decoder.vocab_size = shape_list(value)[0]
    # 返回解码器偏置的字典
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"decoder_bias": self.decoder_bias}

    # 设置解码器偏置的值
    def set_bias(self, value: tf.Variable):
        # 从给定的值中获取解码器偏置
        self.decoder_bias = value["decoder_bias"]
        # 更新配置中的词汇表大小
        self.config.vocab_size = shape_list(value["decoder_bias"])[0]

    # 实现 Transformer 解码器的前向传播
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将隐藏状态通过全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 获取序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 将隐藏状态重塑成二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.output_embedding_size])
        # 对隐藏状态进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states)
        # 执行解码器权重矩阵与隐藏状态的乘积
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder, transpose_b=True)
        # 将结果重塑成三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加解码器偏置
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.decoder_bias)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_tf_bert.TFBertMLMHead复制代码，并将Bert->RemBert
class TFRemBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: RemBertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFRemBertLMPredictionHead 对象，用于预测下一个词的概率分布
        self.predictions = TFRemBertLMPredictionHead(config, input_embeddings, name="predictions")

    # 根据序列输出计算预测分数
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经存在 predictions，就构建 predictions
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


# keras_serializable 类的结构
@keras_serializable
class TFRemBertMainLayer(tf.keras.layers.Layer):
    config_class = RemBertConfig

    def __init__(self, config: RemBertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 初始化配置
        self.config = config
        self.is_decoder = config.is_decoder

        # 创建 TFRemBertEmbeddings 对象，用于嵌入输入的词
        self.embeddings = TFRemBertEmbeddings(config, name="embeddings")
        # 创建 TFRemBertEncoder 对象，用于编码输入的词
        self.encoder = TFRemBertEncoder(config, name="encoder")
        # 如果 add_pooling_layer 为真，则创建 TFRemBertPooler 对象，用于池化编码后的词
        self.pooler = TFRemBertPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入嵌入
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 解包输入，调用TFBertMainLayer的call函数
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
        ```
    # 此方法用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经被构建则直接返回
        if self.built:
            return
        # 标记模型已被构建
        self.built = True
        # 如果模型中存在 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            # 使用 embeddings 层的名称创建作用域
            with tf.name_scope(self.embeddings.name):
                # 调用 embeddings 层的 build 方法
                self.embeddings.build(None)
        # 如果模型中存在 encoder 层
        if getattr(self, "encoder", None) is not None:
            # 使用 encoder 层的名称创建作用域
            with tf.name_scope(self.encoder.name):
                # 调用 encoder 层的 build 方法
                self.encoder.build(None)
        # 如果模型中存在 pooler 层
        if getattr(self, "pooler", None) is not None:
            # 使用 pooler 层的名称创建作用域
            with tf.name_scope(self.pooler.name):
                # 调用 pooler 层的 build 方法
                self.pooler.build(None)
class TFRemBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 RemBertConfig 类作为配置类
    config_class = RemBertConfig
    # 基础模型的前缀
    base_model_prefix = "rembert"


# RemBERT 模型的起始文档字符串，包含一些通用信息和使用提示
REMBERT_START_DOCSTRING = r"""

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

    Args:
        config ([`RemBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入文档字符串
REMBERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare RemBERT Model transformer outputing raw hidden-states without any specific head on top.",
    REMBERT_START_DOCSTRING,
)
# RemBERT 模型类，继承自 TFRemBertPreTrainedModel 类
class TFRemBertModel(TFRemBertPreTrainedModel):
    # 初始化函数，用于创建RemBertModel对象
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
    
        # 创建TFRemBertMainLayer对象，赋值给self.rembert属性
        self.rembert = TFRemBertMainLayer(config, name="rembert")
    
    # 使用unpack_inputs修饰器，用于将输入参数解包，并进行相应的处理
    # 使用add_start_docstrings_to_model_forward修饰器，为call函数添加模型输入部分的文档字符串
    # 使用add_code_sample_docstrings修饰器，添加一个代码示例的文档字符串
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
        # 略
    def call(self, input_ids: tf.Tensor, attention_mask: tf.Tensor = None, token_type_ids: tf.Tensor = None,
             position_ids: tf.Tensor = None, head_mask: tf.Tensor = None, inputs_embeds: tf.Tensor = None,
             encoder_hidden_states: tf.Tensor = None, encoder_attention_mask: tf.Tensor = None,
             past_key_values: Tuple[Tuple[tf.Tensor]] = None, use_cache: bool = True,
             output_attentions: bool = False, output_hidden_states: bool = False, return_dict: bool = True,
             training: bool = False) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
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
        # 调用 RemBERT 模型
        outputs = self.rembert(
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

        # 返回 RemBERT 模型的输出
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在 RemBERT 模型，则构建 RemBERT 模型
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
# 使用装饰器添加模型的文档字符串，说明这是一个带有顶部`语言建模`头的RemBERT模型
@add_start_docstrings("""RemBERT Model with a `language modeling` head on top.""", REMBERT_START_DOCSTRING)
class TFRemBertForMaskedLM(TFRemBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果配置为decoder，发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFRemBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化RemBERT的主层，不添加池化层
        self.rembert = TFRemBertMainLayer(config, name="rembert", add_pooling_layer=False)
        # 初始化MLM头
        self.mlm = TFRemBertMLMHead(config, input_embeddings=self.rembert.embeddings, name="mlm___cls")

    # 获取语言建模头
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 模型调用函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFMaskedLMOutput,
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
):
        # 调用模型时执行的操作
        # 详细说明了输入参数和输出结果的文档字符串
        # 添加代码示例的文档字符串
        pass  # 函数体未提供，故此处pass
    # 定义方法，用于生成 MaskedLM 模型的输出或者包含损失函数计算结果的元组
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用模型的 rembert 方法，传入参数，生成模型输出
        outputs = self.rembert(
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
        # 获取模型输出的第一个元素，即序列输出
        sequence_output = outputs[0]
        # 使用模型的 mlm 方法对序列输出进行预测得到预测分数
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果存在标签，则调用 hf_compute_loss 方法计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果 return_dict 为 False，则返回包含预测分数的元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFMaskedLMOutput 对象，包含损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 定义方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果模型包含 rembert 属性，则构建 rembert 模型
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果模型包含 mlm 属性，则构建 mlm 模型
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
# 使用装饰器为 TFRemBertForCausalLM 类添加文档字符串，描述其作为 CLM fine-tuning 用的 RemBERT 模型
@add_start_docstrings(
    """RemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", REMBERT_START_DOCSTRING
)
# 定义 TFRemBertForCausalLM 类，继承自 TFRemBertPreTrainedModel 和 TFCausalLanguageModelingLoss
class TFRemBertForCausalLM(TFRemBertPreTrainedModel, TFCausalLanguageModelingLoss):
    # 初始化方法，接受配置对象 config 和其他输入
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 如果不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `TFRemBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 RemBERT 主层和 MLM 头部
        self.rembert = TFRemBertMainLayer(config, name="rembert", add_pooling_layer=False)
        self.mlm = TFRemBertMLMHead(config, input_embeddings=self.rembert.embeddings, name="mlm___cls")

    # 获取语言建模头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 为生成准备输入的方法，处理输入数据
    # 从 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation 复制过来的
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果模型作为编码器-解码器模型中的解码器使用，动态创建解码器的注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果使用了过去的键值，截断 decoder_input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 调用方法，用于模型的正向传播
    @unpack_inputs
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFCausalLMOutputWithCrossAttentions,
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
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 方法实现在其他地方，这里只是声明方法的签名

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 RemBERT 主层，则构建它
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果存在 MLM 头部，则构建它
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)


# 使用装饰器为以下注释添加文档字符串
@add_start_docstrings(
    """
    # RemBERT 模型带有顶部的序列分类/回归头，例如用于 GLUE 任务。
    
    
        def __init__(self, config, add_pooling_layer=True, add_cross_attention=False, topical_token_position="all",**kwargs):
            super().__init__(config, add_pooling_layer=add_pooling_layer, add_cross_attention=add_cross_attention,**kwargs)
            self.num_labels = config.num_labels
    
            self.rembert = RemBertModel(config,
                topical_token_position= topical_token_position)
    
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
    
            self.init_weights()
    
    
    
    注释：
    
    # 初始化函数，接收 config 和一些可选参数
    def __init__(self, config, add_pooling_layer=True, add_cross_attention=False, topical_token_position="all",**kwargs):
        # 调用父类初始化函数，通过传递参数和 kwargs 参数初始化模型
        super().__init__(config, add_pooling_layer=add_pooling_layer, add_cross_attention=add_cross_attention,**kwargs)
        # 配置分类/回归头的标签数量
        self.num_labels = config.num_labels
    
        # 创建 RemBERT 模型对象，通过传递参数 config 和 topical_token_position
        self.rembert = RemBertModel(config,
            topical_token_position= topical_token_position)
    
        # 创建丢弃层对象，通过传递参数 config 中的 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建线性层对象
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
    
        # 初始化模型权重
        self.init_weights()
    
    
    代码：
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        topic_mask=None
    ):
    
    
    
    注释：
    
    # 正向传播函数，接收一些输入参数和一些可选参数
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        topic_mask=None
    ):
    
    
    代码：
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    
    
    注释：
    
    # 如果 return_dict 不为 None，则使用 self.config.use_return_dict, 否则 return_dict 为 None
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    
    代码：
    
    outputs = self.rembert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        topic_mask = topic_mask
    )
    
    
    
    注释：
    
    # 调用 self.rembert，将输入参数传递过去，返回结果对象保存至 outputs 中
    outputs = self.rembert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        topic_mask = topic_mask
    )
    
    
    代码：
    
            sequence_output = outputs[0]
    
            pooled_output = self.pooler(sequence_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
    
    
    
    注释：
    
    # 获取 outputs 对象的第一个元素，保存到 sequence_output 中
    sequence_output = outputs[0]
    
    # 调用 self.pooler，将 sequence_output 作为输入，得到 pooled_output
    pooled_output = self.pooler(sequence_output)
    # 调用 self.dropout，将 pooled_output 作为输入，得到 pooled_output
    pooled_output = self.dropout(pooled_output)
    # 调用 self.classifier，将 pooled_output 作为输入，得到 logits
    logits = self.classifier(pooled_output)
    
    
    代码：
    
            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    
    
    注释：
    
    # 初始化 loss 为 None
    loss = None
    # 如果 labels 不为 None
    if labels is not None:
        # 如果 self.num_labels 为 1
        if self.num_labels == 1:
            # 执行回归任务，创建均方误差对象 loss_fct，计算 logits 和 labels 的损失
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            # 执行分类任务，创建交叉熵损失对象 loss_fct，计算 logits 和 labels 的损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    
    代码：
    
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
    
            return RemBertForSequenceClassificationOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    
    
    
    注释：
    
    # 如果 return_dict 为 False
    if not return_dict:
        # 构建返回的元组 output，包括 logits 和 outputs 对象的第三个元素后面所有元素
        output = (logits,) + outputs[2:]
        # 如果 loss 不为 None，则返回 (loss, output)，否则返回 output
        return ((loss,) + output) if loss is not None else output
    
    # 如果 return_dict 为 True，则将相关结果包装成 RemBertForSequenceClassificationOutput 对象后返回
    return RemBertForSequenceClassificationOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
# 定义一个基于 RemBert 的序列分类模型，继承自 TFRemBertPreTrainedModel 和 TFSequenceClassificationLoss
class TFRemBertForSequenceClassification(TFRemBertPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化函数，接受 RemBertConfig 类型的配置对象以及其他输入参数
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 创建 RemBert 主层
        self.rembert = TFRemBertMainLayer(config, name="rembert")
        # 添加 Dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        # 添加 Dense 层，用于进行分类
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 保存模型的配置对象
        self.config = config

    # 定义模型的前向传播逻辑
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFSequenceClassifierOutput,
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
    # 定义一个方法，用于处理序列分类器的输出和标签，返回序列分类器的输出或者包含损失的元组
    # 输出包含一个TF序列分类器输出（TFSequenceClassifierOutput）对象或者一个包含TF张量的元组
    def call(
        self, 
        input_ids: Union[tf.Tensor, Dict[str, tf.Tensor]], 
        attention_mask: Union[tf.Tensor, Dict[str, tf.Tensor]], 
        token_type_ids: Union[tf.Tensor, Dict[str, tf.Tensor]] = None, 
        position_ids: Union[tf.Tensor, Dict[str, tf.Tensor]] = None, 
        head_mask: Union[tf.Tensor, Dict[str, tf.Tensor]] = None, 
        inputs_embeds: tf.Tensor = None, 
        output_attentions: bool = None, 
        output_hidden_states: bool = None, 
        return_dict: bool = None, 
        training: bool = False, 
        labels: Union[tf.Tensor] = None,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用rembert模型，传入输入的张量和各种参数
        outputs = self.rembert(
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
        # 获取模型输出的池化输出
        pooled_output = outputs[1]
        # 对池化输出进行dropout处理
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 使用分类器处理dropout后的输出，得到logits
        logits = self.classifier(inputs=pooled_output)
        # 如果存在标签，计算损失，否则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不需要返回dict，则返回output的元组，否则返回TF序列分类器输出对象
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 如果rembert模型存在，则构建rembert模型
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 添加起始文档字符串，描述了 RemBERT 模型及其在多项选择分类任务中的应用
@add_start_docstrings(
    """
    RemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    REMBERT_START_DOCSTRING,
)
# 定义 TFRemBertForMultipleChoice 类，继承自 TFRemBertPreTrainedModel 和 TFMultipleChoiceLoss 类
class TFRemBertForMultipleChoice(TFRemBertPreTrainedModel, TFMultipleChoiceLoss):
    # 初始化方法，接收 RemBertConfig 对象和其他输入参数
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 初始化 RemBERT 主层
        self.rembert = TFRemBertMainLayer(config, name="rembert")
        # 初始化 Dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        # 初始化分类器，包括一个全连接层和一个 softmax 激活函数
        self.classifier = tf.keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置参数
        self.config = config

    # 定义模型前向传播方法，接收模型输入并返回模型输出
    @unpack_inputs
    # 添加文档字符串，描述模型的输入格式和功能
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 添加代码示例文档字符串，指定了模型的检查点和输出类型
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
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
        ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        if input_ids is not None:  # 如果输入的 input_ids 不为空
            num_choices = shape_list(input_ids)[1]  # 获取 input_ids 的第二维的大小
            seq_length = shape_list(input_ids)[2]  # 获取 input_ids 的第三维的大小
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 获取 inputs_embeds 的第二维的大小
            seq_length = shape_list(inputs_embeds)[2]  # 获取 inputs_embeds 的第三维的大小

        flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None  # 将 input_ids 展平成二维的，如果 input_ids 不为空
        flat_attention_mask = (  # 将 attention_mask 展平成二维的，如果 attention_mask 不为空
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        flat_token_type_ids = (  # 将 token_type_ids 展平成二维的，如果 token_type_ids 不为空
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        flat_position_ids = (  # 将 position_ids 展平成二维的，如果 position_ids 不为空
            tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        )
        flat_inputs_embeds = (  # 将 inputs_embeds 展平成三维的，如果 inputs_embeds 不为空
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        outputs = self.rembert(  # 调用 self.rembert 方法
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
        pooled_output = outputs[1]  # 获取 outputs 的第二个值
        pooled_output = self.dropout(inputs=pooled_output, training=training)  # 对 pooled_output 进行 dropout 处理
        logits = self.classifier(inputs=pooled_output)  # 使用分类器对 pooled_output 进行输出
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))  # 将 logits 进行形状重塑成二维的
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)  # 计算 loss 值，如果 labels 不为空

        if not return_dict:  # 如果 return_dict 为 False
            output = (reshaped_logits,) + outputs[2:]  # 输出结果包含 reshaped_logits 和 outputs 的第三个值及以后的所有值
            return ((loss,) + output) if loss is not None else output  # 如果 loss 不为空，则返回 loss 和 output 的值，否则返回 output

        return TFMultipleChoiceModelOutput(  # 返回 TFMultipleChoiceModelOutput 类型的对象
            loss=loss,  # 设置 loss 属性
            logits=reshaped_logits,  # 设置 logits 属性
            hidden_states=outputs.hidden_states,  # 设置 hidden_states 属性
            attentions=outputs.attentions,  # 设置 attentions 属性
        )
    # 定义神经网络模型的构建方法，用于构建网络结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型中包含名为"rembert"的属性
        if getattr(self, "rembert", None) is not None:
            # 使用 TensorFlow 的命名空间，建立 rembert 模型的网络结构
            with tf.name_scope(self.rembert.name):
                # 调用 rembert 模型的 build 方法，构建 rembert 模型的网络结构
                self.rembert.build(None)
        # 如果模型中包含名为"classifier"的属性
        if getattr(self, "classifier", None) is not None:
            # 使用 TensorFlow 的命名空间，建立 classifier 模型的网络结构
            with tf.name_scope(self.classifier.name):
                # 调用 classifier 模型的 build 方法，构建 classifier 模型的网络结构
                self.classifier.build([None, None, self.config.hidden_size])
# 定义一个基于 RemBERT 模型的 token 分类模型，用于命名实体识别 (NER) 等任务
@add_start_docstrings(
    """
    RemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    REMBERT_START_DOCSTRING,
)
class TFRemBertForTokenClassification(TFRemBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        # 创建一个 RemBERT 主层对象，不添加池化层
        self.rembert = TFRemBertMainLayer(config, name="rembert", add_pooling_layer=False)
        # 添加一个丢弃层，根据设置的 dropout 率丢弃部分神经元
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 添加一个全连接层作为分类器，输出单元数为标签的数量
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    # 定义前向传播的方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
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
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        定义了一个函数，输入参数有labels、input_ids、attention_mask、token_type_ids、position_ids、head_mask、inputs_embeds、
        output_attentions、output_hidden_states、return_dict和training
        输出类型为Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]
        labels参数是一个可选的Tensor或numpy数组，形状为(batch_size, sequence_length)，表示用于计算标记分类损失的标签。索引应在`[0, ..., config.num_labels - 1]`内。
   
        调用self.rembert方法，传入input_ids、attention_mask、token_type_ids、position_ids、head_mask、inputs_embeds、
        output_attentions、output_hidden_states、return_dict和training作为参数，将结果赋值给outputs
        取出outputs的第一个元素，赋值给sequence_output
        调用self.dropout方法，传入sequence_output和training作为参数，将结果赋值给sequence_output
        调用self.classifier方法，传入sequence_output作为参数，将结果赋值给logits
        如果labels为None，则loss为None，否则调用self.hf_compute_loss方法，传入labels和logits作为参数，将结果赋值给loss

        如果return_dict为False，则将logits与outputs的后续元素组成元组output，并返回output，如果loss不为None，则在返回的元组前面添加loss
        如果return_dict为True，则使用TFTokenClassifierOutput类创建对象，传入loss、logits、outputs的hidden_states和outputs的attentions作为参数，并返回该对象

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        如果self.rembert不为None，则使用tf.name_scope和self.rembert.name对代码进行封装，并调用self.rembert.build(None)方法
        如果self.classifier不为None，则使用tf.name_scope和self.classifier.name对代码进行封装，并调用self.classifier.build([None, None, self.config.hidden_size])方法
@add_start_docstrings(
    """
    RemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    REMBERT_START_DOCSTRING,
)
class TFRemBertForQuestionAnswering(TFRemBertPreTrainedModel, TFQuestionAnsweringLoss):
    # 初始化函数，接受一个 RemBertConfig 对象的参数
    def __init__(self, config: RemBertConfig, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 将配置中的标签数赋值给对象的属性
        self.num_labels = config.num_labels

        # 创建 RemBert 主层对象，不包含池化层
        self.rembert = TFRemBertMainLayer(config, add_pooling_layer=False, name="rembert")
        # 在输出层添加一个全连接层，用于计算起始和结束的标签
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置对象到属性
        self.config = config

    # 装饰器，用于展开输入参数
    @unpack_inputs
    # 添加模型前向传播的文档注释
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档注释
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数
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
    # 定义函数的输入和输出类型注释，指定返回的是TFQuestionAnsweringModelOutput或者Tuple[tf.Tensor]类型
    def call(self, 
        # 起始位置的标签，用于计算token分类损失
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        # 结束位置的标签，用于计算token分类损失
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用rembert模型，传入相应参数
        outputs = self.rembert(
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
        # 获取模型输出的第一个结果序列
        sequence_output = outputs[0]
        # 对模型输出进行QA输出
        logits = self.qa_outputs(inputs=sequence_output)
        # 分割logits，得到start_logits和end_logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 去除维度为1的轴，得到start_logits和end_logits
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化损失为None
        loss = None

        # 如果start_positions和end_positions均不为空
        if start_positions is not None and end_positions is not None:
            # 标签设定为包含start_positions的字典
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 计算损失
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果return_dict为False
        if not return_dict:
            # 组合输出
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFQuestionAnsweringModelOutput类型的结果
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已构建完成，直接返回
        if self.built:
            return
        self.built = True
        # 如果rembert存在
        if getattr(self, "rembert", None) is not None:
            with tf.name_scope(self.rembert.name):
                self.rembert.build(None)
        # 如果qa_outputs存在
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                # 构建qa_outputs
                self.qa_outputs.build([None, None, self.config.hidden_size])
```