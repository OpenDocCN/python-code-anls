# `.\models\lxmert\modeling_tf_lxmert.py`

```
# coding=utf-8
# 定义文件编码为 UTF-8

# 版权声明：以下代码由 Google AI Language Team Authors、HuggingFace Inc. team 和 Lxmert Authors 创作
# 版权所有 (c) 2018, NVIDIA CORPORATION. 保留所有权利。

# 根据 Apache 许可证 2.0 版本授权，除非符合许可证要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 如果根据适用法律要求或书面同意，软件将按“原样”分发，无任何明示或暗示的担保或条件
# 请参阅许可证获取更多详细信息

""" TF 2.0 LXMERT model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 从内部库中导入相关模块和函数
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    shape_list,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_lxmert import LxmertConfig

# 获取全局日志记录器对象
logger = logging.get_logger(__name__)

# 以下是文档示例中使用的模型检查点和配置信息
_CHECKPOINT_FOR_DOC = "unc-nlp/lxmert-base-uncased"
_CONFIG_FOR_DOC = "LxmertConfig"

# 定义 TF LXMERT 预训练模型的存档列表
TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unc-nlp/lxmert-base-uncased",
]

# 定义 TFLxmertModelOutput 数据类，包含语言、视觉和跨模态编码器的最后隐藏状态、汇集输出和注意力概率
@dataclass
class TFLxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder)
    """
    # 定义函数的参数，描述了不同的输出和注意力张量
    Args:
        language_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            语言编码器最后一层的隐藏状态序列。
        vision_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            视觉编码器最后一层的隐藏状态序列。
        pooled_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            序列第一个令牌（CLS令牌）的最后一层隐藏状态，经过线性层和Tanh激活函数进一步处理后的结果。
        language_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            语言编码器每个交叉模态层的输入特征和输出的元组，形状为 `(batch_size, sequence_length, hidden_size)`。
        vision_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            视觉编码器每个交叉模态层的输入特征和输出的元组，形状为 `(batch_size, sequence_length, hidden_size)`。
        language_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力头中注意力softmax后的权重张量元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            用于计算自注意力头中加权平均值。
        vision_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力头中注意力softmax后的权重张量元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            用于计算自注意力头中加权平均值。
        cross_encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力头中注意力softmax后的权重张量元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            用于计算自注意力头中加权平均值。
@dataclass
class TFLxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`LxmertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score (`tf.Tensor` of shape `(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # Optional attributes representing different outputs from the model

    # Total loss combining masked language modeling and next sequence prediction loss
    loss: tf.Tensor | None = None

    # Scores of language modeling head before softmax
    prediction_logits: tf.Tensor | None = None

    # Scores of textual matching objective (True/False continuation) before softmax
    cross_relationship_score: tf.Tensor | None = None

    # Scores of question answering objective
    question_answering_score: tf.Tensor | None = None

    # Hidden states of language model layers and cross-modality layers
    language_hidden_states: tuple[tf.Tensor] | None = None

    # Hidden states of vision model layers and cross-modality layers
    vision_hidden_states: tuple[tf.Tensor] | None = None

    # Attention weights for language model self-attention heads
    language_attentions: tuple[tf.Tensor] | None = None

    # Attention weights for vision model self-attention heads
    vision_attentions: tuple[tf.Tensor] | None = None

    # Attention weights for cross-encoder self-attention heads
    cross_encoder_attentions: tuple[tf.Tensor] | None = None
    # 定义交叉关系得分，初始化为 None
    cross_relationship_score: tf.Tensor | None = None
    # 定义问答得分，初始化为 None
    question_answering_score: tf.Tensor | None = None
    # 定义语言模型的隐藏状态，初始化为 None，是一个包含 Tensor 的元组
    language_hidden_states: Tuple[tf.Tensor] | None = None
    # 定义视觉模型的隐藏状态，初始化为 None，是一个包含 Tensor 的元组
    vision_hidden_states: Tuple[tf.Tensor] | None = None
    # 定义语言模型的注意力分布，初始化为 None，是一个包含 Tensor 的元组
    language_attentions: Tuple[tf.Tensor] | None = None
    # 定义视觉模型的注意力分布，初始化为 None，是一个包含 Tensor 的元组
    vision_attentions: Tuple[tf.Tensor] | None = None
    # 定义交叉编码器的注意力分布，初始化为 None，是一个包含 Tensor 的元组
    cross_encoder_attentions: Tuple[tf.Tensor] | None = None
class TFLxmertVisualFeatureEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # Object feature encoding
        # 创建对象特征编码层，使用 Dense 层进行线性变换
        self.visn_fc = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="visn_fc",
        )
        # 对对象特征编码结果进行 LayerNormalization
        self.visn_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="visn_layer_norm")

        # Box position encoding
        # 创建盒子位置编码层，使用 Dense 层进行线性变换
        self.box_fc = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="box_fc",
        )
        # 对盒子位置编码结果进行 LayerNormalization
        self.box_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="box_layer_norm")

        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.feat_dim = config.visual_feat_dim
        self.pos_dim = config.visual_pos_dim
        self.config = config

    def call(self, visn_input, training=False):
        feats, boxes = visn_input

        # 对对象特征进行线性变换和规范化
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        
        # 对盒子位置进行线性变换和规范化
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        
        # 将对象特征编码和盒子位置编码的结果求平均作为最终输出
        output = (x + y) / 2

        # 对输出结果应用 Dropout
        output = self.dropout(output, training=training)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果已经建立，直接返回；否则，根据输入形状建立各层
        if getattr(self, "visn_fc", None) is not None:
            with tf.name_scope(self.visn_fc.name):
                self.visn_fc.build([None, None, self.feat_dim])
        if getattr(self, "visn_layer_norm", None) is not None:
            with tf.name_scope(self.visn_layer_norm.name):
                self.visn_layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, "box_fc", None) is not None:
            with tf.name_scope(self.box_fc.name):
                self.box_fc.build([None, None, self.pos_dim])
        if getattr(self, "box_layer_norm", None) is not None:
            with tf.name_scope(self.box_layer_norm.name):
                self.box_layer_norm.build([None, None, self.config.hidden_size])


class TFLxmertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 初始化配置信息和参数
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        
        # LayerNormalization 层，用于规范化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 在构建函数中，为词嵌入层添加权重张量
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下，添加名为 "weight" 的权重张量
        self.weight = self.add_weight(
            name="weight",
            shape=[self.config.vocab_size, self.hidden_size],
            initializer=get_initializer(initializer_range=self.initializer_range),
        )

        # 在 "token_type_embeddings" 命名空间下，添加名为 "embeddings" 的权重张量
        self.token_type_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.config.type_vocab_size, self.hidden_size],
            initializer=get_initializer(initializer_range=self.initializer_range),
        )

        # 在 "position_embeddings" 命名空间下，添加名为 "embeddings" 的权重张量
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=get_initializer(initializer_range=self.initializer_range),
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置已构建标志为 True
        self.built = True
        # 如果存在 LayerNorm 层，则在其命名空间下构建
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 在调用函数中，基于输入张量应用嵌入
    def call(self, input_ids=None, token_type_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 确保输入中至少包含 input_ids 或 inputs_embeds
        assert not (input_ids is None and inputs_embeds is None)

        # 如果提供了 input_ids，则根据 input_ids 和权重张量获取嵌入
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入的形状列表，并去掉最后一个维度
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 token_type_ids，则创建与输入形状相同的全零张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 创建位置编码的位置张量，用于获取位置嵌入
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)

        # 根据 token_type_ids 获取 token type 嵌入
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)

        # 将输入嵌入、位置嵌入和 token type 嵌入相加得到最终嵌入
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds

        # 对最终嵌入应用 LayerNorm
        final_embeddings = self.LayerNorm(inputs=final_embeddings)

        # 在训练模式下对最终嵌入应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终嵌入张量
        return final_embeddings
    # 定义一个名为 TFLxmertAttention 的自定义层，继承自 keras 的 Layer 类
    class TFLxmertAttention(keras.layers.Layer):
        # 初始化方法，接受一个 config 对象和其他关键字参数
        def __init__(self, config, **kwargs):
            # 调用父类的初始化方法
            super().__init__(**kwargs)
            # 检查隐藏大小是否能被注意力头数整除
            if config.hidden_size % config.num_attention_heads != 0:
                # 如果不能整除，抛出 ValueError 异常
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({config.num_attention_heads}"
                )

            # 设置注意力头数和注意力头大小
            self.num_attention_heads = config.num_attention_heads
            assert config.hidden_size % config.num_attention_heads == 0
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # 定义 query、key、value 三个全连接层，用于计算注意力分数
            self.query = keras.layers.Dense(
                self.all_head_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="query",
            )
            self.key = keras.layers.Dense(
                self.all_head_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="key",
            )
            self.value = keras.layers.Dense(
                self.all_head_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="value",
            )

            # 定义 dropout 层，用于在注意力计算中进行随机失活
            self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
            # 设置上下文维度为隐藏大小
            self.ctx_dim = config.hidden_size
            # 保存配置信息
            self.config = config

        # 定义方法 transpose_for_scores，用于将输入张量重新形状并转置以计算注意力分数
        def transpose_for_scores(self, x, batch_size):
            # 将输入 x 从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
            x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
            # 转置张量以匹配注意力计算的期望维度顺序 [batch_size, num_attention_heads, seq_length, attention_head_size]
            return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, hidden_states, context, attention_mask, output_attentions, training=False):
        # 获取批量大小
        batch_size = shape_list(hidden_states)[0]
        # 使用 self.query 对隐藏状态进行转换
        mixed_query_layer = self.query(hidden_states)
        # 使用 self.key 对上下文进行转换
        mixed_key_layer = self.key(context)
        # 使用 self.value 对上下文进行转换
        mixed_value_layer = self.value(context)

        # 将转换后的查询向量调整为注意力分数的形状
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将转换后的键向量调整为注意力分数的形状
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将转换后的值向量调整为注意力分数的形状
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算查询向量和键向量的点积，得到原始注意力分数
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        # 计算缩放因子 dk，并将注意力分数进行缩放
        dk = tf.cast(shape_list(key_layer)[-1], dtype=attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # 如果存在注意力遮罩，则应用它（预先为 TFLxmertModel call() 函数中的所有层计算）
            attention_mask = tf.cast(attention_mask, dtype=attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为注意力概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 使用 dropout 进行注意力概率的随机失活
        attention_probs = self.dropout(attention_probs, training=training)
        # 计算上下文向量，加权和值向量
        context_layer = tf.matmul(attention_probs, value_layer)

        # 调整上下文向量的形状，以便输出
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        # 准备模型输出，包括上下文层和注意力概率（如果需要输出注意力）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建查询、键、值的神经网络层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.ctx_dim])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.ctx_dim])
# 定义一个名为 TFLxmertIntermediate 的自定义层，继承自 keras.layers.Layer
class TFLxmertIntermediate(keras.layers.Layer):
    # 初始化函数，接收 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层 dense，输出维度为 config.intermediate_size
        self.dense = keras.layers.Dense(
            config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 根据 config 中的 hidden_act 字段，获取激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 将 config 存储在当前对象的属性中
        self.config = config

    # 定义 call 方法，处理输入 hidden_states
    def call(self, hidden_states):
        # 将 hidden_states 输入到全连接层 dense 中进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的 hidden_states 应用 intermediate_act_fn 激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果层已经构建过，直接返回
        if self.built:
            return
        # 标记当前层已构建
        self.built = True
        # 如果 dense 层存在，则在 tf 的 name_scope 下构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建 dense 层，输入维度为 [None, None, self.config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])


# 定义一个名为 TFLxmertOutput 的自定义层，继承自 keras.layers.Layer
class TFLxmertOutput(keras.layers.Layer):
    # 初始化函数，接收 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层 dense，输出维度为 config.hidden_size
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 创建一个 LayerNormalization 层，epsilon 设置为 config.layer_norm_eps，命名为 LayerNorm
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 将 config 存储在当前对象的属性中
        self.config = config

    # 定义 call 方法，处理输入 hidden_states 和 input_tensor
    def call(self, hidden_states, input_tensor, training=False):
        # 将 hidden_states 输入到全连接层 dense 中进行变换
        hidden_states = self.dense(hidden_states)
        # 如果处于训练阶段，对变换后的 hidden_states 进行 dropout 操作
        hidden_states = self.dropout(hidden_states, training)
        # 将 dropout 后的 hidden_states 与 input_tensor 相加，并进行 LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的 hidden_states
        return hidden_states

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果层已经构建过，直接返回
        if self.built:
            return
        # 标记当前层已构建
        self.built = True
        # 如果 dense 层存在，则在 tf 的 name_scope 下构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建 dense 层，输入维度为 [None, None, self.config.intermediate_size]
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果 LayerNorm 层存在，则在 tf 的 name_scope 下构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNorm 层，输入维度为 [None, None, self.config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 定义一个名为 TFLxmertAttentionOutput 的自定义层，继承自 keras.layers.Layer
class TFLxmertAttentionOutput(keras.layers.Layer):
    # 初始化函数，接收 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层 dense，输出维度为 config.hidden_size
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 创建一个 LayerNormalization 层，epsilon 设置为 config.layer_norm_eps，命名为 LayerNorm
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 将 config 存储在当前对象的属性中
        self.config = config
    # 定义神经网络层的调用方法，接收隐藏状态、输入张量和训练标志作为参数
    def call(self, hidden_states, input_tensor, training=False):
        # 将隐藏状态通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行 dropout 操作，根据训练标志决定是否启用
        hidden_states = self.dropout(hidden_states, training=training)
        # 将 dropout 后的隐藏状态与输入张量相加，并通过 LayerNorm 进行归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过处理后的隐藏状态
        return hidden_states

    # 构建神经网络层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则根据指定的输入形状构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层，则根据指定的输入形状构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 定义一个 TensorFlow Keras 自定义层 TFLxmertSelfAttentionLayer
class TFLxmertSelfAttentionLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 TFLxmertAttention 实例，用于自注意力机制
        self.self = TFLxmertAttention(config, name="self")
        # 创建一个 TFLxmertAttentionOutput 实例，用于处理注意力输出
        self.attention_output = TFLxmertAttentionOutput(config, name="output")

    # 定义层的调用方法
    def call(self, input_tensor, attention_mask, output_attentions, training=False):
        # 执行自注意力机制，键和查询均为输入张量本身
        self_output = self.self(input_tensor, input_tensor, attention_mask, output_attentions)
        if output_attentions:
            # 如果需要输出注意力权重，则从 self_output 中获取注意力权重
            attention_probs = self_output[1]
        # 将自注意力的输出传递给注意力输出层处理
        attention_output = self.attention_output(self_output[0], input_tensor)
        # 根据是否需要输出注意力权重，返回不同的结果元组
        return (attention_output, attention_probs) if output_attentions else (attention_output,)

    # 构建层，用于初始化子层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self 层，建立 self 层的计算图
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果存在 attention_output 层，建立 attention_output 层的计算图
        if getattr(self, "attention_output", None) is not None:
            with tf.name_scope(self.attention_output.name):
                self.attention_output.build(None)


# 定义一个 TensorFlow Keras 自定义层 TFLxmertCrossAttentionLayer
class TFLxmertCrossAttentionLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 TFLxmertAttention 实例，用于跨注意力机制
        self.att = TFLxmertAttention(config, name="att")
        # 创建一个 TFLxmertAttentionOutput 实例，用于处理注意力输出
        self.attention_output = TFLxmertAttentionOutput(config, name="output")

    # 定义层的调用方法
    def call(
        self,
        input_tensor,
        ctx_tensor,
        ctx_att_mask,
        output_attentions=False,
        training=False,
    ):
        # 执行跨注意力机制，处理输入张量和上下文张量
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions, training=training)
        if output_attentions:
            # 如果需要输出注意力权重，则从 output 中获取注意力权重
            attention_probs = output[1]
        # 将跨注意力的输出传递给注意力输出层处理
        attention_output = self.attention_output(output[0], input_tensor, training=training)
        # 根据是否需要输出注意力权重，返回不同的结果元组
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

    # 构建层，用于初始化子层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 att 层，建立 att 层的计算图
        if getattr(self, "att", None) is not None:
            with tf.name_scope(self.att.name):
                self.att.build(None)
        # 如果存在 attention_output 层，建立 attention_output 层的计算图
        if getattr(self, "attention_output", None) is not None:
            with tf.name_scope(self.attention_output.name):
                self.attention_output.build(None)


# 定义一个 TensorFlow Keras 自定义层 TFLxmertLayer
class TFLxmertLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 TFLxmertSelfAttentionLayer 实例，用于自注意力层
        self.attention = TFLxmertSelfAttentionLayer(config, name="attention")
        # 创建一个 TFLxmertIntermediate 实例，用于处理中间层计算
        self.intermediate = TFLxmertIntermediate(config, name="intermediate")
        # 创建一个 TFLxmertOutput 实例，用于输出转换层
        self.transformer_output = TFLxmertOutput(config, name="output")
    # 定义一个方法用于调用 Transformer 模型的前向传播过程
    def call(self, hidden_states, attention_mask, output_attentions, training=False):
        # 调用注意力层，得到注意力输出结果
        attention_outputs = self.attention(hidden_states, attention_mask, output_attentions, training=training)
        # 获取注意力输出中的第一个元素，即注意力输出本身
        attention_output = attention_outputs[0]
        # 将注意力输出传入中间层
        intermediate_output = self.intermediate(attention_output)
        # 将中间层输出和注意力输出传入 Transformer 输出层
        layer_output = self.transformer_output(intermediate_output, attention_output, training=training)
        # 构建输出元组，包括层输出和可能的注意力输出
        outputs = (layer_output,) + attention_outputs[1:]  # 如果有的话，添加注意力信息
        # 返回最终的输出
        return outputs

    # 构建方法用于在第一次调用前初始化模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为 True，表示模型已构建
        self.built = True
        # 如果存在注意力层，则构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 Transformer 输出层，则构建 Transformer 输出层
        if getattr(self, "transformer_output", None) is not None:
            with tf.name_scope(self.transformer_output.name):
                self.transformer_output.build(None)
# 定义一个自定义的 Keras 层，用于 TFLxmert 模型中的一个层
class TFLxmertXLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建视觉注意力层对象，命名为 "visual_attention"
        self.visual_attention = TFLxmertCrossAttentionLayer(config, name="visual_attention")

        # 创建自注意力层对象用于语言输入，命名为 "lang_self_att"
        self.lang_self_att = TFLxmertSelfAttentionLayer(config, name="lang_self_att")
        # 创建自注意力层对象用于视觉输入，命名为 "visn_self_att"
        self.visn_self_att = TFLxmertSelfAttentionLayer(config, name="visn_self_att")

        # 创建中间层和输出层对象（前馈神经网络）
        self.lang_inter = TFLxmertIntermediate(config, name="lang_inter")
        self.lang_output = TFLxmertOutput(config, name="lang_output")
        self.visn_inter = TFLxmertIntermediate(config, name="visn_inter")
        self.visn_output = TFLxmertOutput(config, name="visn_output")

    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visn_input,
        visn_attention_mask,
        output_attentions,
        training=False,
    ):
        # 交叉注意力操作

        # 复制语言输入，避免因为同一输入在两个层间传递导致 Keras 模型保存与加载时出现问题
        lang_attention_lang_input = tf.identity(lang_input)
        visn_attention_lang_input = tf.identity(lang_input)
        lang_attention_visn_input = tf.identity(visn_input)
        visn_attention_visn_input = tf.identity(visn_input)

        # 对语言输入进行视觉注意力计算
        lang_att_output = self.visual_attention(
            lang_attention_lang_input,
            lang_attention_visn_input,
            visn_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 对视觉输入进行视觉注意力计算
        visn_att_output = self.visual_attention(
            visn_attention_visn_input,
            visn_attention_lang_input,
            lang_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        return lang_att_output, visn_att_output

    def self_att(
        self,
        lang_input,
        lang_attention_mask,
        visn_input,
        visn_attention_mask,
        training=False,
    ):
        # 自注意力操作
        output_attentions = False
        # 对语言输入进行语言自注意力计算
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions, training=training)
        # 对视觉输入进行视觉自注意力计算
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask, output_attentions, training=training)
        return lang_att_output[0], visn_att_output[0]

    def output_fc(self, lang_input, visn_input, training=False):
        # 全连接层操作
        # 对语言输入进行中间层计算
        lang_inter_output = self.lang_inter(lang_input)
        # 对视觉输入进行中间层计算
        visn_inter_output = self.visn_inter(visn_input)

        # 计算层的输出
        lang_output = self.lang_output(lang_inter_output, lang_input, training)
        visn_output = self.visn_output(visn_inter_output, visn_input, training)
        return lang_output, visn_output

    def call(
        self,
        lang_feats,
        lang_attention_mask,
        visn_feats,
        visn_attention_mask,
        output_attentions,
        training=False,
    ):
        # 调用函数，定义层的调用方式，处理语言和视觉特征

        # 返回语言和视觉特征的交叉注意力输出
        return self.cross_att(
            lang_feats,
            lang_attention_mask,
            visn_feats,
            visn_attention_mask,
            output_attentions,
            training=training,
        )
        ):
        # 将语言特征和视觉特征输出赋给相应的变量
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        # 调用交叉注意力机制进行特征交互
        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output,
            lang_attention_mask,
            visn_att_output,
            visn_attention_mask,
            output_attentions,
            training=training,
        )

        # 从语言注意力输出中提取注意力概率，排除第一个元素（通常是注意力分数）
        attention_probs = lang_att_output[1:]

        # 调用自注意力机制分别处理语言和视觉特征
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visn_att_output[0],
            visn_attention_mask,
            training=training,
        )

        # 使用全连接层处理语言和视觉特征的输出
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output, training=training)

        # 根据输出注意力开关决定返回结果，包含注意力概率的第一个元素
        return (lang_output, visn_output, attention_probs[0]) if output_attentions else (lang_output, visn_output)

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return

        # 设置模型已构建标志为True
        self.built = True

        # 如果存在视觉注意力模型，构建其网络结构
        if getattr(self, "visual_attention", None) is not None:
            with tf.name_scope(self.visual_attention.name):
                self.visual_attention.build(None)

        # 如果存在语言自注意力模型，构建其网络结构
        if getattr(self, "lang_self_att", None) is not None:
            with tf.name_scope(self.lang_self_att.name):
                self.lang_self_att.build(None)

        # 如果存在视觉自注意力模型，构建其网络结构
        if getattr(self, "visn_self_att", None) is not None:
            with tf.name_scope(self.visn_self_att.name):
                self.visn_self_att.build(None)

        # 如果存在语言交互模型，构建其网络结构
        if getattr(self, "lang_inter", None) is not None:
            with tf.name_scope(self.lang_inter.name):
                self.lang_inter.build(None)

        # 如果存在语言输出模型，构建其网络结构
        if getattr(self, "lang_output", None) is not None:
            with tf.name_scope(self.lang_output.name):
                self.lang_output.build(None)

        # 如果存在视觉交互模型，构建其网络结构
        if getattr(self, "visn_inter", None) is not None:
            with tf.name_scope(self.visn_inter.name):
                self.visn_inter.build(None)

        # 如果存在视觉输出模型，构建其网络结构
        if getattr(self, "visn_output", None) is not None:
            with tf.name_scope(self.visn_output.name):
                self.visn_output.build(None)
# 定义一个自定义的 TensorFlow 层 TFLxmertEncoder，用于实现 LXMERT 模型的编码器功能
class TFLxmertEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 初始化视觉特征编码器，使用 TFLxmertVisualFeatureEncoder 类
        self.visn_fc = TFLxmertVisualFeatureEncoder(config, name="visn_fc")

        # 设置层的数量
        self.num_l_layers = config.l_layers  # 从配置中获取 L 层的数量
        self.num_x_layers = config.x_layers  # 从配置中获取 X 层的数量
        self.num_r_layers = config.r_layers  # 从配置中获取 R 层的数量

        # 初始化各个层
        # 使用 self.layer 而不是 self.l_layer 是为了支持加载 BERT 权重
        self.layer = [TFLxmertLayer(config, name=f"layer_._{i}") for i in range(self.num_l_layers)]
        self.x_layers = [TFLxmertXLayer(config, name=f"x_layers_._{i}") for i in range(self.num_x_layers)]
        self.r_layers = [TFLxmertLayer(config, name=f"r_layers_._{i}") for i in range(self.num_r_layers)]
        self.config = config

    # 定义 call 方法，用于定义层的前向传播逻辑
    def call(
        self,
        lang_feats=None,
        lang_attention_mask=None,
        visual_feats=None,
        visual_pos=None,
        visual_attention_mask=None,
        output_attentions=None,
        training=False,
        **kwargs
    ):
    ):
        # 初始化空的视觉隐藏状态和语言隐藏状态元组
        vision_hidden_states = ()
        language_hidden_states = ()
        # 根据是否需要输出注意力权重，初始化视觉和语言的注意力权重元组
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

        # 对视觉特征进行全连接层处理
        visual_feats = self.visn_fc([visual_feats, visual_pos], training=training)

        # 运行语言层的每个模块
        for layer_module in self.layer:
            # 调用当前语言层模块，更新语言特征和可能的注意力权重
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions, training=training)
            lang_feats = l_outputs[0]
            # 更新语言隐藏状态元组
            language_hidden_states = language_hidden_states + (lang_feats,)
            # 如果需要输出注意力权重，更新语言注意力权重元组
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)

        # 运行关系层的每个模块
        for layer_module in self.r_layers:
            # 调用当前关系层模块，更新视觉特征和可能的注意力权重
            v_outputs = layer_module(
                visual_feats,
                visual_attention_mask,
                output_attentions,
                training=training,
            )
            visual_feats = v_outputs[0]
            # 更新视觉隐藏状态元组
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            # 如果需要输出注意力权重，更新视觉注意力权重元组
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)

        # 运行跨模态层的每个模块
        for layer_module in self.x_layers:
            # 调用当前跨模态层模块，更新语言特征、视觉特征和可能的注意力权重
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions,
                training=training,
            )
            lang_feats, visual_feats = x_outputs[:2]
            # 更新视觉和语言隐藏状态元组
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            # 如果需要输出注意力权重，更新跨模态注意力权重元组
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)

        # 组装视觉编码器的输出：视觉隐藏状态和可能的视觉注意力权重
        visual_encoder_outputs = (
            vision_hidden_states,
            vision_attentions if output_attentions else None,
        )
        # 组装语言编码器的输出：语言隐藏状态和可能的语言注意力权重
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )

        # 返回编码器的输出：视觉编码器输出、语言编码器输出和可能的跨编码器注意力权重
        return (
            visual_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )
    # 定义一个方法用于构建模型，如果模型已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 设置模型已构建标志为 True
        self.built = True
        
        # 如果存在名为 "visn_fc" 的属性并且不为 None，则构建其内部的层
        if getattr(self, "visn_fc", None) is not None:
            with tf.name_scope(self.visn_fc.name):
                self.visn_fc.build(None)
        
        # 如果存在名为 "layer" 的属性并且不为 None，则依次构建每个层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
        
        # 如果存在名为 "x_layers" 的属性并且不为 None，则依次构建每个层
        if getattr(self, "x_layers", None) is not None:
            for layer in self.x_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        
        # 如果存在名为 "r_layers" 的属性并且不为 None，则依次构建每个层
        if getattr(self, "r_layers", None) is not None:
            for layer in self.r_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFLxmertMainLayer(keras.layers.Layer):
    # 定义一个 Keras 可序列化的自定义层，用于处理 LXMERT 主层
    config_class = LxmertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化方法，设置层的配置参数
        self.config = config
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFLxmertEmbeddings(config, name="embeddings")
        self.encoder = TFLxmertEncoder(config, name="encoder")
        self.pooler = TFLxmertPooler(config, name="pooler")
        self.config = config

    def get_input_embeddings(self):
        # 返回 embeddings 层
        return self.embeddings

    def set_input_embeddings(self, value):
        # 设置 embeddings 层的权重和词汇大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        # 未实现的方法，用于裁剪注意力头部
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        visual_feats=None,
        visual_pos=None,
        attention_mask=None,
        visual_attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 模型调用方法，处理输入数据，进行前向传播
        # 使用 unpack_inputs 装饰器来解包输入参数

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果已经构建过，则直接返回
        # 根据需要构建 embeddings、encoder 和 pooler 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFLxmertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LxmertConfig
    base_model_prefix = "lxmert"

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        # 定义用于构建网络的虚拟输入数据
        batch_size = 2
        num_visual_features = 10
        input_ids = tf.constant([[3, 5, 6], [2, 3, 4]], dtype=tf.int32)
        visual_feats = tf.random.uniform((batch_size, num_visual_features, self.config.visual_feat_dim))
        visual_pos = tf.random.uniform((batch_size, num_visual_features, 4))

        return {
            "input_ids": input_ids,
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
        }

    @property


这段代码定义了一个自定义的 Keras 层 `TFLxmertMainLayer` 和一个抽象类 `TFLxmertPreTrainedModel`，分别用于处理 LXMERT 模型的主要层和预训练模型的初始化和虚拟输入数据。
    # 定义输入签名函数，返回一个字典，描述了模型输入的各个特征
    def input_signature(self):
        # 定义输入的张量规格：input_ids是一个二维的整数张量，形状为(None, None)，表示批次中的序列长度可以变化
        return {
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            # attention_mask是一个二维的整数张量，形状为(None, None)，用于指示输入序列的填充部分和真实部分
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            # visual_feats是一个三维的浮点数张量，形状为(None, None, visual_feat_dim)，包含了视觉特征的表示
            "visual_feats": tf.TensorSpec((None, None, self.config.visual_feat_dim), tf.float32, name="visual_feats"),
            # visual_pos是一个三维的浮点数张量，形状为(None, None, 4)，描述了视觉特征的位置信息
            "visual_pos": tf.TensorSpec((None, None, 4), tf.float32, name="visual_pos"),
            # visual_attention_mask是一个二维的整数张量，形状为(None, None)，用于指示视觉输入序列的填充和真实部分
            "visual_attention_mask": tf.TensorSpec((None, None), tf.int32, name="visual_attention_mask"),
            # token_type_ids是一个二维的整数张量，形状为(None, None)，用于多任务学习或特定任务的标识符
            "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }
# LXMERT 模型的文档字符串，描述了模型的提出背景、应用场景和训练数据来源等信息
LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from
    Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It's a vision and language transformer
    model, pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual
    genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss
    for question answering attribute prediction, and object tag prediction.

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
        config ([`LxmertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# LXMERT 模型的输入文档字符串，当前为空，用于指定输入格式和输入参数的解释
LXMERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
# 定义 TFLxmertModel 类，继承自 TFLxmertPreTrainedModel，用于表示 LXMERT 模型的核心变换器输出原始隐藏状态
class TFLxmertModel(TFLxmertPreTrainedModel):
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置、输入参数和关键字参数
        super().__init__(config, *inputs, **kwargs)
        # 创建一个 TFLxmertMainLayer 实例，命名为 "lxmert"
        self.lxmert = TFLxmertMainLayer(config, name="lxmert")

    # 将装饰器 unpack_inputs 应用于 call 方法
    # 向模型前向传播函数添加模型输入的文档字符串
    # 向模型前向传播函数添加代码示例的文档字符串，包括检查点、输出类型和配置类
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        visual_feats: tf.Tensor | None = None,
        visual_pos: tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        visual_attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFLxmertModelOutput]:
        # 调用 self.lxmert 来执行 LXMERT 模型的前向传播
        outputs = self.lxmert(
            input_ids,
            visual_feats,
            visual_pos,
            attention_mask,
            visual_attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
            training,
        )
        # 返回模型输出
        return outputs

    # 构建模型的方法，用于定义模型的结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果 self.lxmert 存在，则在其命名作用域内构建它
        if getattr(self, "lxmert", None) is not None:
            with tf.name_scope(self.lxmert.name):
                self.lxmert.build(None)
# 定义一个自定义层 TFLxmertPooler，继承自 keras 的 Layer 类
class TFLxmertPooler(keras.layers.Layer):
    
    # 初始化方法，接受配置参数 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 Dense 层，用于池化操作，输出维度为 config.hidden_size
        # 使用指定的初始化器初始化权重，激活函数为 tanh
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    # 定义调用方法，接受隐藏状态 hidden_states 作为输入
    def call(self, hidden_states):
        # 池化模型，简单地取第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output

    # 构建方法，用于构建层的结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，则直接返回；否则，构建 Dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers 库中复制的类，用于 Lxmert 模型的预测头转换
# 基于 TFBertPredictionHeadTransform 修改
class TFLxmertPredictionHeadTransform(keras.layers.Layer):
    
    # 初始化方法，接受 LxmertConfig 类型的配置参数 config 和其他关键字参数
    def __init__(self, config: LxmertConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 创建一个 Dense 层，输出维度为 config.hidden_size
        # 使用指定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        
        # 根据配置中的隐藏激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        
        # 创建 LayerNormalization 层，epsilon 参数为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    # 定义调用方法，接受 tf.Tensor 类型的隐藏状态作为输入，返回处理后的隐藏状态
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用 Dense 层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    # 构建方法，用于构建层的结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，则直接返回；否则，构建 Dense 层和 LayerNormalization 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers 库中复制的类，用于 Lxmert 模型的语言模型预测头
# 基于 TFBertLMPredictionHead 修改
class TFLxmertLMPredictionHead(keras.layers.Layer):
    
    # 略
    # 初始化函数，用于创建一个新的 LxmertOutput 类的实例
    def __init__(self, config: LxmertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 保存配置信息和隐藏大小
        self.config = config
        self.hidden_size = config.hidden_size

        # 创建一个 TFLxmertPredictionHeadTransform 的实例，用于变换输出
        self.transform = TFLxmertPredictionHeadTransform(config, name="transform")

        # 将输入的嵌入层保存为类的一个属性
        # 输出权重与输入嵌入层相同，但每个标记有一个仅用于输出的偏置
        self.input_embeddings = input_embeddings

    # 在构建层时被调用，用于初始化层的权重
    def build(self, input_shape=None):
        # 添加一个名为 "bias" 的可训练权重，形状为 (词汇表大小,)
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True

        # 如果存在变换器 transform，则在命名空间下构建它
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 返回输入嵌入层的引用
    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings

    # 设置输入嵌入层的权重和词汇表大小
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 返回偏置的字典形式
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置的值
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 模型调用函数，接受隐藏状态张量作为输入，返回预测的标记概率张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用 transform 对隐藏状态进行变换
        hidden_states = self.transform(hidden_states=hidden_states)
        # 获取隐藏状态的序列长度
        seq_length = shape_list(hidden_states)[1]
        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 执行矩阵乘法，将隐藏状态与输入嵌入层的权重相乘
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重塑为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置到隐藏状态的最后一个维度
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回最终的预测张量
        return hidden_states
# 从 transformers.models.bert.modeling_tf_bert.TFBertMLMHead 复制而来，将 Bert 替换为 Lxmert
class TFLxmertMLMHead(keras.layers.Layer):
    def __init__(self, config: LxmertConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        
        # 创建 TFLxmertLMPredictionHead 对象作为预测头部
        self.predictions = TFLxmertLMPredictionHead(config, input_embeddings, name="predictions")

    # 对输入的序列输出进行预测
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 通过预测头部获取预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    # 构建层，确保仅构建一次
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在预测头部，构建它
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


# Lxmert 的预训练头部，包含语言模型预测和序列关系预测
class TFLxmertPreTrainingHeads(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        
        # 创建 TFLxmertLMPredictionHead 对象作为预测头部
        self.predictions = TFLxmertLMPredictionHead(config, input_embeddings, name="predictions")
        
        # 创建用于序列关系预测的全连接层
        self.seq_relationship = keras.layers.Dense(
            2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="seq_relationship",
        )
        self.config = config

    # 对序列输出和池化输出进行调用，生成预测分数和序列关系分数
    def call(self, sequence_output, pooled_output):
        # 获取语言模型预测分数
        prediction_scores = self.predictions(sequence_output)
        # 获取序列关系预测分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

    # 构建层，确保仅构建一次
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在预测头部，构建它
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        # 如果存在序列关系预测层，构建它
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build([None, None, self.config.hidden_size])


# Lxmert 的视觉回答头部，用于分类问题的预测
class TFLxmertVisualAnswerHead(keras.layers.Layer):
    def __init__(self, config, num_labels, **kwargs):
        super().__init__(**kwargs)
        hid_dim = config.hidden_size
        
        # 创建全连接层，输入维度为隐藏维度的两倍，输出维度为标签数量
        self.dense = keras.layers.Dense(
            hid_dim * 2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="logit_fc_._0",
        )
        self.activation = get_tf_activation("gelu")  # 获取 GELU 激活函数
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="logit_fc_._2")
        
        # 创建输出全连接层，输入为隐藏状态的维度，输出为标签数量
        self.dense_1 = keras.layers.Dense(
            num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="logit_fc_._3",
        )
        self.hid_dim = hid_dim

    # 对隐藏状态进行处理，通过全连接层和激活函数生成预测
    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense_1(hidden_states)

        return hidden_states
    # 定义神经网络层的构建方法，参数input_shape为输入形状，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在dense属性，则构建dense层
        if getattr(self, "dense", None) is not None:
            # 使用dense层的名称作为命名空间
            with tf.name_scope(self.dense.name):
                # 构建dense层，输入形状为[None, None, self.hid_dim]
                self.dense.build([None, None, self.hid_dim])
        # 如果存在layer_norm属性，则构建layer_norm层
        if getattr(self, "layer_norm", None) is not None:
            # 使用layer_norm层的名称作为命名空间
            with tf.name_scope(self.layer_norm.name):
                # 构建layer_norm层，输入形状为[None, self.hid_dim * 2]
                self.layer_norm.build([None, self.hid_dim * 2])
        # 如果存在dense_1属性，则构建dense_1层
        if getattr(self, "dense_1", None) is not None:
            # 使用dense_1层的名称作为命名空间
            with tf.name_scope(self.dense_1.name):
                # 构建dense_1层，输入形状为[None, None, self.hid_dim * 2]
                self.dense_1.build([None, None, self.hid_dim * 2])
# 定义一个自定义的 Keras 层，用于处理 Lxmert 模型的视觉对象预测任务
class TFLxmertVisualObjHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化一个用于预测头部变换的层
        self.transform = TFLxmertPredictionHeadTransform(config, name="transform")

        # 根据配置决定是否使用视觉损失
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        if config.visual_feat_loss:
            visual_losses["feat"] = {"shape": (-1, 2048), "num": config.visual_feat_dim}
        self.visual_losses = visual_losses

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置项
        # 创建一个字典，其中每个键对应于一个类型的视觉损失，并且值是对应的全连接层
        self.decoder_dict = {
            key: keras.layers.Dense(
                self.visual_losses[key]["num"],
                kernel_initializer=get_initializer(config.initializer_range),
                name=f"decoder_dict.{key}",
            )
            for key in self.visual_losses
        }
        self.config = config

    def call(self, hidden_states):
        # 对输入的隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        output = {}
        # 对每种视觉损失类型进行预测
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了 transform 层，则构建它
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)
        # 如果已经构建了 decoder_dict 字典中的层，则分别构建每一层
        if getattr(self, "decoder_dict", None) is not None:
            for layer in self.decoder_dict.values():
                with tf.name_scope(layer.name):
                    # 构建每个全连接层，输入形状为 [None, None, config.hidden_size]
                    layer.build([None, None, self.config.hidden_size])


@add_start_docstrings("""Lxmert Model with a `language modeling` head on top.""", LXMERT_START_DOCSTRING)
class TFLxmertForPreTraining(TFLxmertPreTrainedModel):
    # 这里省略类的具体实现部分，但包含了一个 Lxmert 模型和一个语言建模头部
    pass
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 将配置信息存储在实例中
        self.config = config
        # 设置问题回答标签的数量
        self.num_qa_labels = config.num_qa_labels
        # 可视化损失的正常化器
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # 使用预训练任务的标志
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        # Lxmert 主干网络
        self.lxmert = TFLxmertMainLayer(config, name="lxmert")

        # 预训练头部
        self.cls = TFLxmertPreTrainingHeads(config, self.lxmert.embeddings, name="cls")
        # 如果有物体预测任务，则创建物体预测头部
        if self.task_obj_predict:
            self.obj_predict_head = TFLxmertVisualObjHead(config, name="obj_predict_head")
        # 如果有问题回答任务，则创建问题回答头部
        if self.task_qa:
            self.answer_head = TFLxmertVisualAnswerHead(config, self.num_qa_labels, name="answer_head")

        # 损失函数
        self.loss_fcts = {
            "l2": keras.losses.Huber(delta=1.0, name="huber_loss"),  # L2 损失函数
            "visn_ce": keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 稀疏分类交叉熵损失函数
            "ce": keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 稀疏分类交叉熵损失函数
        }

        # 可视化损失字典
        visual_losses = {}
        # 如果配置中包含物体损失，则添加到可视化损失字典中
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),  # 形状为一维向量
                "num": config.num_object_labels,  # 物体标签数量
                "loss": "visn_ce",  # 使用稀疏分类交叉熵损失
            }
        # 如果配置中包含属性损失，则添加到可视化损失字典中
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),  # 形状为一维向量
                "num": config.num_attr_labels,  # 属性标签数量
                "loss": "visn_ce",  # 使用稀疏分类交叉熵损失
            }
        # 如果配置中包含特征损失，则添加到可视化损失字典中
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),  # 形状为二维向量
                "num": config.visual_feat_dim,  # 特征维度
                "loss": "l2",  # 使用L2损失
            }
        # 将可视化损失字典存储在实例中
        self.visual_losses = visual_losses
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 使用装饰器对 call 方法进行功能增强和文档替换
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        visual_feats: tf.Tensor | None = None,
        visual_pos: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        visual_attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        masked_lm_labels: tf.Tensor | None = None,
        obj_labels: Dict[str, Tuple[tf.Tensor, tf.Tensor]] | None = None,
        matched_label: tf.Tensor | None = None,
        ans: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        # 定义 call 方法的参数，包括输入的张量和布尔值控制标志


这段代码定义了一个 `call` 方法，用于执行模型的前向传播操作。方法中使用了装饰器来增强其功能和修改返回文档。
    # 定义模型构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将构建标志设置为True，表示模型已经构建过
        self.built = True
        # 如果存在名为"lxmert"的属性，使用其名称作为命名空间来构建lxmert模块
        if getattr(self, "lxmert", None) is not None:
            with tf.name_scope(self.lxmert.name):
                self.lxmert.build(None)
        # 如果存在名为"cls"的属性，使用其名称作为命名空间来构建cls模块
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
        # 如果存在名为"obj_predict_head"的属性，使用其名称作为命名空间来构建obj_predict_head模块
        if getattr(self, "obj_predict_head", None) is not None:
            with tf.name_scope(self.obj_predict_head.name):
                self.obj_predict_head.build(None)
        # 如果存在名为"answer_head"的属性，使用其名称作为命名空间来构建answer_head模块
        if getattr(self, "answer_head", None) is not None:
            with tf.name_scope(self.answer_head.name):
                self.answer_head.build(None)
```