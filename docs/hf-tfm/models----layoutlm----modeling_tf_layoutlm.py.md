# `.\models\layoutlm\modeling_tf_layoutlm.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归属于 Microsoft Research Asia LayoutLM 团队作者和 HuggingFace Inc. 团队
#
# 根据 Apache 许可 2.0 版本，除非符合许可，否则不得使用此文件
# 您可以在以下网址获取许可的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 不附带任何明示或暗示的保证或条件。
# 有关许可的特定语言，请参阅许可证
""" TF 2.0 LayoutLM 模型。"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入相关的输出类型和损失函数
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
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "LayoutLMConfig"

# 预训练模型的存档列表
TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlm-base-uncased",
    "microsoft/layoutlm-large-uncased",
]


class TFLayoutLMEmbeddings(tf.keras.layers.Layer):
    """从单词、位置和标记类型嵌入构建嵌入。"""

    def __init__(self, config: LayoutLMConfig, **kwargs):
        # 初始化嵌入层
        super().__init__(**kwargs)
        
        # 从配置中获取参数
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.max_2d_position_embeddings = config.max_2d_position_embeddings
        self.initializer_range = config.initializer_range
        
        # LayerNorm 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 丢弃层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 构建模型，初始化参数
    def build(self, input_shape=None):
        # 嵌入层：词嵌入
        with tf.name_scope("word_embeddings"):
            # 添加词嵌入权重参数
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：标记类型嵌入
        with tf.name_scope("token_type_embeddings"):
            # 添加标记类型嵌入权重参数
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：位置嵌入
        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入权重参数
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：x 轴位置嵌入
        with tf.name_scope("x_position_embeddings"):
            # 添加 x 轴位置嵌入权重参数
            self.x_position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_2d_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：y 轴位置嵌入
        with tf.name_scope("y_position_embeddings"):
            # 添加 y 轴位置嵌入权重参数
            self.y_position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_2d_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：高度位置嵌入
        with tf.name_scope("h_position_embeddings"):
            # 添加高度位置嵌入权重参数
            self.h_position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_2d_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 嵌入层：宽度位置嵌入
        with tf.name_scope("w_position_embeddings"):
            # 添加宽度位置嵌入权重参数
            self.w_position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_2d_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建过模型，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 模型调用函数
    def call(
        self,
        input_ids: tf.Tensor = None,
        bbox: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
        ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)  # 检查输入的 input_ids 和 inputs_embeds 是否都不为 None

        if input_ids is not None:  # 如果 input_ids 不为 None
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 调用函数检查 input_ids 是否在范围内
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)  # 从 weight 中根据 input_ids 获取对应的嵌入向量

        input_shape = shape_list(inputs_embeds)[:-1]  # 获取 inputs_embeds 的形状

        if token_type_ids is None:  # 如果 token_type_ids 为 None
            token_type_ids = tf.fill(dims=input_shape, value=0)  # 创建与 input_shape 形状相同的张量，并填充为 0

        if position_ids is None:  # 如果 position_ids 为 None
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)  # 创建一个范围从 0 到 input_shape[-1] 的张量，并添加一个维度

        if position_ids is None:  # 如果 position_ids 为 None
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)  # 创建一个范围从 0 到 input_shape[-1] 的张量，并添加一个维度

        if bbox is None:  # 如果 bbox 为 None
            bbox = bbox = tf.fill(input_shape + [4], value=0)  # 创建一个形状为 input_shape + [4] 的张量，并填充为 0
        try:
            left_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 0])  # 根据 bbox 中的坐标值获取 x 方向上的位置嵌入向量
            upper_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 1])  # 根据 bbox 中的坐标值获取 y 方向上的位置嵌入向量
            right_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 2])  # 根据 bbox 中的坐标值获取 x 方向上的位置嵌入向量
            lower_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 3])  # 根据 bbox 中的坐标值获取 y 方向上的位置嵌入向量
        except IndexError as e:  # 捕获 IndexError 异常
            raise IndexError("The `bbox`coordinate values should be within 0-1000 range.") from e  # 抛出指定的 IndexError 异常
        h_position_embeddings = tf.gather(self.h_position_embeddings, bbox[:, :, 3] - bbox[:, :, 1])  # 根据 bbox 的高度获取高度位置嵌入向量
        w_position_embeddings = tf.gather(self.w_position_embeddings, bbox[:, :, 2] - bbox[:, :, 0])  # 根据 bbox 的宽度获取宽度位置嵌入向量

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)  # 从 position_embeddings 中获取位置嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)  # 从 token_type_embeddings 中获取 token 类型嵌入向量
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
        )  # 计算最终的嵌入向量
        final_embeddings = self.LayerNorm(inputs=final_embeddings)  # 将最终的嵌入向量进行 LayerNorm 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)  # 使用 dropout 进行训练或者推断时的处理

        return final_embeddings  # 返回最终的嵌入向量
# 从transformers.models.bert.modeling_tf_bert.TFBertSelfAttention复制并修改为LayoutLM
class TFLayoutLMSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 如果隐藏层大小不能整除注意力头数，则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和注意力头大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 初始化查询、键和值的Dense层，用于计算注意力分布
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 初始化丢弃层，用于在计算注意力时丢弃一定比例的值
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config

    # 将张量转换为注意力分数，用于计算自注意力
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 从[batch_size, seq_length, all_head_size]重塑为[batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从[batch_size, seq_length, num_attention_heads, attention_head_size]转置为[batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 调用函数，用于处理自注意力和多头注意力机制
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
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了查询层，则直接返回
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果已经构建了键层，则直接返回
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果已经构建了值层，则直接返回
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# LayoutLM 的自注意力层定义，继承自 tf.keras.layers.Layer 类
class TFLayoutLMSelfOutput(tf.keras.layers.Layer):
    # 初始化函数，接受 LayoutLMConfig 对象作为参数
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建全连接层，用于变换输入特征的维度
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，用于对输入进行归一化处理
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 调用函数，实现自注意力层的前向传播过程
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层进行特征变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练过程中应用 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将变换后的特征与输入进行残差连接，并进行 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建函数，用于构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经存在 dense 层，则构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果已经存在 LayerNorm 层，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# LayoutLM 的注意力层定义，继承自 tf.keras.layers.Layer 类
class TFLayoutLMAttention(tf.keras.layers.Layer):
    # 初始化函数，接受 LayoutLMConfig 对象作为参数
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建自注意力层对象
        self.self_attention = TFLayoutLMSelfAttention(config, name="self")
        # 创建自注意力输出层对象
        self.dense_output = TFLayoutLMSelfOutput(config, name="output")

    # 调用函数，实现注意力层的前向传播过程
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
        # 使用自注意力层进行前向传播
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
        # 使用自注意力输出层进行前向传播
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力信息，则将其加入输出元组中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 构建函数用于构建模型，在输入形状为None的情况下
    def build(self, input_shape=None):
        # 判断模型是否已经构建，如果已经构建则直接返回
        if self.built:
            return
        # 设置built为True，表示模型已经构建
        self.built = True
        # 如果存在self_attention属性，则构建self_attention
        if getattr(self, "self_attention", None) is not None:
            # 使用tf.name_scope为self_attention创建命名空间，并构建self_attention
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果存在dense_output属性，则构建dense_output
        if getattr(self, "dense_output", None) is not None:
            # 使用tf.name_scope为dense_output创建命名空间，并构建dense_output
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制代码，并将Bert->LayoutLM
class TFLayoutLMIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units为intermediate_size，kernel_initializer为config.initializer_range，name为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config.hidden_act是字符串，则使用对应的激活函数，否则使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 对输入的hidden_states进行一次全连接层和激活函数处理
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层，其中包含了对dense的构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertOutput复制代码，并将Bert->LayoutLM
class TFLayoutLMOutput(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units为hidden_size，kernel_initializer为config.initializer_range，name为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，epsilon为config.layer_norm_eps，name为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，rate为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 对输入的hidden_states进行全连接层、Dropout和LayerNormalization处理
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建层，其中包含了对dense和LayerNorm的构建
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


# 从transformers.models.bert.modeling_tf_bert.TFBertLayer复制代码，并将Bert->LayoutLM
class TFLayoutLMLayer(tf.keras.layers.Layer):
    # 初始化 LayoutLM 模型
    def __init__(self, config: LayoutLMConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 LayoutLM 的注意力层对象
        self.attention = TFLayoutLMAttention(config, name="attention")
        # 判断是否为解码器模型
        self.is_decoder = config.is_decoder
        # 判断是否添加跨注意力层
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨注意力层
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出数值异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建跨注意力层对象
            self.crossattention = TFLayoutLMAttention(config, name="crossattention")
        # 创建 LayoutLM 的中间层对象
        self.intermediate = TFLayoutLMIntermediate(config, name="intermediate")
        # 创建 LayoutLM 的输出层对象
        self.bert_output = TFLayoutLMOutput(config, name="output")

    # 调用 LayoutLM 模型
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
        # 定义函数的输入和输出类型，返回的是包含 Tensor 的元组
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的 key/value 不为空，则将解码器自注意力缓存的 key/value 存储在位置 1,2
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
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是解码器自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
           
        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有设置跨注意力层，则抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 跨注意力缓存的 key/values 元组在过去的 key/value 元组的位置 3,4
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
            # 获取跨注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加跨注意力

            # 将跨注意力缓存添加到 present_key_value 的位置 3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对注意力输出进行中间层处理
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用 BERT 输出层处理得到最终输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果输出注意力，则添加

        # 如果是解码器，将注意力的 key/values 作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建模型，如果已经构建好则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果具有注意力机制，则构建注意力机制
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果具有中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果具有BERT输出层，则构建BERT输出层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果具有交叉注意力机制，则构建交叉注意力机制
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertEncoder复制代码，并将Bert->LayoutLM
class TFLayoutLMEncoder(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建包含多个TFLayoutLMLayer的列表，根据num_hidden_layers确定列表长度
        self.layer = [TFLayoutLMLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 对编码器进行调用
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
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组
        all_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力权重并且配置中添加了跨层注意力，则初始化空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果需要使用缓存，则初始化空元组
        next_decoder_cache = () if use_cache else None
        # 遍历所有层模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取过去的键值对，如果过去的键值对不为空，则直接赋值，否则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 对当前层模块进行调用，获取输出
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

            # 如果使用缓存，则将当前层的输出添加到缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置中添加了跨层注意力并且编码器隐藏状态不为空，则将当前层的交叉注意力权重添加到元组中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回所有非空元素的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 如果需要返回字典，则返回TFBaseModelOutputWithPastAndCrossAttentions对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在layer属性
        if getattr(self, "layer", None) is not None:
            # 遍历每个层并为其指定名称作用域，然后构建每个层
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    # 为当前层构建
                    layer.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertPooler 复制并修改为 LayoutLM
class TFLayoutLMPooler(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化模型的隐藏状态，设置单元数为 config.hidden_size
        # 使用给定的初始化范围初始化权重矩阵
        # 激活函数设置为 "tanh"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过获取第一个 token 对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，输入形状为 [None, None, self.config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform 复制并修改为 LayoutLM
class TFLayoutLMPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于预测头转换，设置单元数为 config.hidden_size
        # 使用给定的初始化范围初始化权重矩阵
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据配置设置激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 创建 LayerNormalization 层，使用给定的 epsilon 值
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 应用全连接层到隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，输入形状为 [None, None, self.config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNormalization 层，输入形状为 [None, None, self.config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead 复制并修改为 LayoutLM
class TFLayoutLMLMPredictionHead(tf.keras.layers.Layer):
    # 初始化函数，接受 LayoutLMConfig 配置和输入嵌入层作为参数
    def __init__(self, config: LayoutLMConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)

        # 保存配置信息和隐藏层大小
        self.config = config
        self.hidden_size = config.hidden_size

        # 创建用于预测头部转换的 TFLayoutLMPredictionHeadTransform 层
        self.transform = TFLayoutLMPredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入层相同，但每个令牌有一个输出专用的偏置
        self.input_embeddings = input_embeddings

    # 构建函数，在此处创建并添加偏置权重
    def build(self, input_shape=None):
        # 创建偏置权重，形状为词汇表大小，初始化为零
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在预测头部转换层，则构建该层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层的权重
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        # 更新配置中的词汇表大小
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 前向传播函数
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用预测头部转换层对隐藏状态进行转换
        hidden_states = self.transform(hidden_states=hidden_states)
        # 获取序列长度
        seq_length = shape_list(hidden_states)[1]
        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 计算隐藏状态与输入嵌入层权重的乘积
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重塑为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states
# 从transformers.models.bert.modeling_tf_bert.TFBertMLMHead复制而来，将Bert替换为LayoutLM
class TFLayoutLMMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 创建LayoutLM的MLM头部预测层对象
        self.predictions = TFLayoutLMLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 使用预测层对序列输出进行预测
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
class TFLayoutLMMainLayer(tf.keras.layers.Layer):
    # 配置类为LayoutLMConfig
    config_class = LayoutLMConfig

    def __init__(self, config: LayoutLMConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        # 保存配置
        self.config = config

        # 创建LayoutLM的嵌入层对象
        self.embeddings = TFLayoutLMEmbeddings(config, name="embeddings")
        # 创建LayoutLM的编码器层对象
        self.encoder = TFLayoutLMEncoder(config, name="encoder")
        # 如果添加池化层，则创建LayoutLM的池化层对象
        self.pooler = TFLayoutLMPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        # 获取输入嵌入层
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置输入嵌入层的权重
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
    # 构建函数，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建好了，则直接返回，不进行重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在嵌入层（embeddings），则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 使用嵌入层的名称作为命名空间
            with tf.name_scope(self.embeddings.name):
                # 构建嵌入层
                self.embeddings.build(None)
        # 如果存在编码器（encoder），则构建编码器
        if getattr(self, "encoder", None) is not None:
            # 使用编码器的名称作为命名空间
            with tf.name_scope(self.encoder.name):
                # 构建编码器
                self.encoder.build(None)
        # 如果存在池化层（pooler），则构建池化层
        if getattr(self, "pooler", None) is not None:
            # 使用池化层的名称作为命名空间
            with tf.name_scope(self.pooler.name):
                # 构建池化层
                self.pooler.build(None)
# 定义 TFLayoutLMPreTrainedModel 类，用于处理权重初始化并提供预训练模型的下载和加载的简单接口
class TFLayoutLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 LayoutLMConfig
    config_class = LayoutLMConfig
    # 设置基础模型前缀为 "layoutlm"
    base_model_prefix = "layoutlm"


# 设置 LAYOUTLM_START_DOCSTRING 字符串
LAYOUTLM_START_DOCSTRING = r"""

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
        config ([`LayoutLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 设置 LAYOUTLM_INPUTS_DOCSTRING 字符串
LAYOUTLM_INPUTS_DOCSTRING = r"""
"""

# 装饰器 add_start_docstrings，用于将文档字符串添加到 TFLayoutLMModel 类
@add_start_docstrings(
    "The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLM_START_DOCSTRING,
)
# 定义 TFLayoutLMModel 类，并继承 TFLayoutLMPreTrainedModel 类
class TFLayoutLMModel(TFLayoutLMPreTrainedModel):
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        # 调用父类构造函数，初始化 LayoutLM 模型
        super().__init__(config, *inputs, **kwargs)

        # 初始化 LayoutLM 主层
        self.layoutlm = TFLayoutLMMainLayer(config, name="layoutlm")

    # 用于处理输入数据的方法
    @unpack_inputs
    # 添加输入文档字符串到模型的前向方法中
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换模型前向方法的返回文档字符串
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    # 模型的前向方法，接受各种输入参数，并返回模型输出
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
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        """"
        # 从输入的 input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds等参数中获取模型的输出
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

        # 返回模型的输出
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好了，则直接返回
        if self.built:
            return
        # 标记模型已经构建好
        self.built = True
        # 检查模型的 "layoutlm" 是否存在
        if getattr(self, "layoutlm", None) is not None:
            # 在 "layoutlm" 的名称范围内构建模型
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
@add_start_docstrings("""LayoutLM Model with a `language modeling` head on top.""", LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForMaskedLM(TFLayoutLMPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",  # 忽略加载时不期望或缺失的层
        r"cls.seq_relationship",  # 忽略加载时不期望或缺失的层
        r"cls.predictions.decoder.weight",  # 忽略加载时不期望或缺失的层
        r"nsp___cls",  # 忽略加载时不期望或缺失的层
    ]

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFLayoutLMForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )  # 如果想要使用TFLayoutLMForMaskedLM，请确保`config.is_decoder=False`，以便进行双向自注意力

        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name="layoutlm")  # 创建主要的LayoutLM层
        self.mlm = TFLayoutLMMLMHead(config, input_embeddings=self.layoutlm.embeddings, name="mlm___cls")  # 创建LayoutLM的Masked LM头部

    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions  # 返回Masked LM头部

    def get_prefix_bias_name(self) -> str:
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)  # 提示方法get_prefix_bias_name已过时，请使用`get_bias`
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name  # 返回名称拼接字符串

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token ID
        bbox: np.ndarray | tf.Tensor | None = None,  # 边界框坐标
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型ID
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置ID
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入
        output_attentions: Optional[bool] = None,  # 输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
        labels: np.ndarray | tf.Tensor | None = None,  # 标签
        training: Optional[bool] = False,  # 训练模式
    def forward(
        self,
        input_ids: Optional[Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]] = None,
        bbox: Optional[Union[tf.Tensor..., np.ndarray]] = None,
        attention_mask: Optional[Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]] = None,
        token_type_ids: Optional[Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]] = None,
        position_ids: Optional[Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]] = None,
        head_mask: Optional[Union[tf.Tensor, tf.Tensor]] = None,
        labels: Optional[Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]] = None,
        inputs_embeds: Optional[Union[tf.Tensor, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
    """
    labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
        config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
        loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
    
    Returns:
    
    Examples:
    
    
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
    
    """
    
    # 使用LayoutLM模型进行前向传播
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
    # 获取模型输出中的序列输出
    sequence_output = outputs[0]
    # 使用MaskedLM模型计算预测得分
    prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
    # 如果有标签，计算损失
    loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)
    
    # 根据return_dict的值返回输出
    if not return_dict:
        output = (prediction_scores,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    # 返回具有损失、预测得分、隐藏状态和注意力的TFMaskedLMOutput对象
    return TFMaskedLMOutput(
        loss=loss,
        logits=prediction_scores,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    # 构建模型，如果已经构建过了则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果有 layoutlm 属性，则构建 layoutlm 对象
        if getattr(self, "layoutlm", None) is not None:
            # 在命名空间内构建 layoutlm 对象
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
        # 如果有 mlm 属性，则构建 mlm 对象
        if getattr(self, "mlm", None) is not None:
            # 在命名空间内构建 mlm 对象
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
# 为 TFLayoutLMForSequenceClassification 类添加模型文档字符串，描述该类是对 LayoutLM 模型进行多序列分类/回归处理的变体
# 包含了一个线性层（线性层连接到汇总输出）
@add_start_docstrings(
    """
    LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForSequenceClassification(TFLayoutLMPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化函数，接受布局LM模型的配置并进行设置
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        # 创建布局LM主层
        self.layoutlm = TFLayoutLMMainLayer(config, name="layoutlm")
        # 创建Dropout层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 创建分类器Dense层
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        self.config = config

    # 对模型进行调用的方法，接受多个输入，包括输入ID、边界框、注意力掩码等等
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
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
        
    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        self.built = True
        # 构建布局LM层
        if getattr(self, "layoutlm", None) is not None:
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
        # 构建分类器层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForTokenClassification(TFLayoutLMPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义在加载时要忽略的键列表，这些键与模型结构中的一些层有关
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    # 定义在加载时要忽略的缺失的键列表，这些键通常是模型中的一些层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化函数，用于创建 LayoutLM 模型的实例
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 将标签数量设置为配置文件中的标签数量
        self.num_labels = config.num_labels

        # 创建 LayoutLM 主层，同时添加池化层，命名为 "layoutlm"
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name="layoutlm")
        
        # 创建丢弃层，用于在训练时随机丢弃部分神经元，以减少过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 创建分类器，使用全连接层将 LayoutLM 输出转换为标签数量的输出
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        
        # 将配置文件保存到实例中，方便后续调用
        self.config = config

    # 定义模型的调用方法，用于执行前向传播
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
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
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 应用丢弃层（Dropout）进行训练阶段的输出处理
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 应用分类器得到最终的 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果存在标签数据，计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            # 根据是否返回字典，获取相应的输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不做重复构建
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果模型中存在 layoutlm 属性
        if getattr(self, "layoutlm", None) is not None:
            # 在 TensorFlow 中创建一个命名空间，用于组织相关操作
            with tf.name_scope(self.layoutlm.name):
                # 构建 layoutlm 模型，输入形状为 None，表示未指定
                self.layoutlm.build(None)
        # 如果模型中存在 classifier 属性
        if getattr(self, "classifier", None) is not None:
            # 在 TensorFlow 中创建一个命名空间，用于组织相关操作
            with tf.name_scope(self.classifier.name):
                # 构建 classifier 分类器模型，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 使用给定的文档字符串初始化一个带有跨度分类头的 LayoutLM 模型，用于抽取式问答任务，例如 DocVQA。这个模型包括一个线性层，
# 用于计算 `span start logits` 和 `span end logits`，位于最终隐藏状态输出之上。
@add_start_docstrings(
    """
    LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span
    start logits` and `span end logits`).
    """,
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForQuestionAnswering(TFLayoutLMPreTrainedModel, TFQuestionAnsweringLoss):
    # 在加载 TF 模型时，以下名称表示被授权的预期外/缺失的层
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 初始化 LayoutLM 主层，包括池化层，如果 add_pooling_layer 为真
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name="layoutlm")
        
        # 初始化 QA 输出层，使用全连接层
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )
        # 保存配置
        self.config = config

    # 覆盖模型的 call 方法
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
        # 重建模型
        def build(self, input_shape=None):
            # 如果已经构建过，直接返回
            if self.built:
                return
            # 标记模型已经构建
            self.built = True
            # 如果存在 layoutlm 层，构建 layoutlm 层
            if getattr(self, "layoutlm", None) is not None:
                with tf.name_scope(self.layoutlm.name):
                    self.layoutlm.build(None)
            # 如果存在 qa_outputs 层，构建 qa_outputs 层
            if getattr(self, "qa_outputs", None) is not None:
                with tf.name_scope(self.qa_outputs.name):
                    self.qa_outputs.build([None, None, self.config.hidden_size])
```