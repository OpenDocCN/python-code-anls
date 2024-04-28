# `.\transformers\models\lxmert\modeling_tf_lxmert.py`

```
# 设置编码格式为 UTF-8
# 版权声明，包括谷歌 AI 语言团队、HuggingFace Inc. 团队以及 Lxmert 作者
# 版权声明，包括 NVIDIA 公司，保留所有权利
#
# 基于 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

# 导入必要的库
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# 导入相关的工具函数和类
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点
_CHECKPOINT_FOR_DOC = "unc-nlp/lxmert-base-uncased"
# 用于文档的模型配置
_CONFIG_FOR_DOC = "LxmertConfig"

# LXMERT 的 TensorFlow 预训练模型存档列表
TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unc-nlp/lxmert-base-uncased",
]

# 定义 Lxmert 模型的输出，包括最后的隐藏状态、池化输出和语言、视觉、交叉模态编码器的注意概率
@dataclass
class TFLxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")
    """
    Args:
        language_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder. 语言编码器最后一层的隐藏状态序列。
        vision_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder. 视觉编码器最后一层的隐藏状态序列。
        pooled_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. 第一个标记的最后一层隐藏状态，通过线性层和Tanh激活函数进一步处理。
        language_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
            `(batch_size, sequence_length, hidden_size)`. 语言隐藏状态的元组（输入特征和每一种交互模态层的输出的python元组形式），形状为(batch_size, sequence_length, hidden_size)。
        vision_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
            `(batch_size, sequence_length, hidden_size)`. 视觉隐藏状态的元组（输入特征和每一种交互模态层的输出的python元组形式），形状为(batch_size, sequence_length, hidden_size)。
        language_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. 每一层的注意力的元组形式，形状为(batch_size, num_heads, sequence_length, sequence_length)。注意：在通过注意力softmax操作后的注意力权重，用于计算自注意力头部的加权平均值。
        vision_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. 每一层的注意力的元组形式，形状为(batch_size, num_heads, sequence_length, sequence_length)。注意：在通过注意力softmax操作后的注意力权重，用于计算自注意力头部的加权平均值。
        cross_encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. 每一层的注意力的元组形式，形状为(batch_size, num_heads, sequence_length, sequence_length)。注意：在通过注意力softmax操作后的注意力权重，用于计算自注意力头部的加权平均值。
    """

    language_output: tf.Tensor | None = None  # 设置语言输出的张量，并初始化为None
    vision_output: tf.Tensor | None = None  # 设置视觉输出的张量，并初始化为None
    pooled_output: tf.Tensor | None = None  # 设置池化输出的张量，并初始化为None
    language_hidden_states: Tuple[tf.Tensor] | None = None  # 设置语言隐藏状态的张量元组，并初始化为None
    vision_hidden_states: Tuple[tf.Tensor] | None = None  # 设置视觉隐藏状态的张量元组，并初始化为None
    language_attentions: Tuple[tf.Tensor] | None = None  # 设置语言注意力的张量元组，并初始化为None
    vision_attentions: Tuple[tf.Tensor] | None = None  # 设置视觉注意力的张量元组，并初始化为None
    cross_encoder_attentions: Tuple[tf.Tensor] | None = None  # 设置交叉编码器注意力的张量元组，并初始化为None
# 定义一个数据类 TFLxmertForPreTrainingOutput，该类继承自 ModelOutput
class TFLxmertForPreTrainingOutput(ModelOutput):
    """
   Output type of [`LxmertForPreTraining`].
    该类是[`LxmertForPreTraining`]的输出类型。

    Args:
        loss (*optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
        total loss as the sum of the masked language modeling loss and the next sequence prediction
        (classification) loss.
        当`labels`提供时返回，损失值（`tf.Tensor`类型，形状为`(1,)`）：
        总损失，是masked language modeling loss和next sequence prediction (classification) loss的和。

        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        预测语言模型头部的预测分值（SoftMax之前每个词汇的分数）（`tf.Tensor`类型，形状为`(batch_size, sequence_length, config.vocab_size)`）。

        cross_relationship_score (`tf.Tensor` of shape `(batch_size, 2)`):
        Prediction scores of the textual matching objective (classification) head (scores of True/False
        continuation before SoftMax).
        文本匹配目标（分类）头部的预测分数（True/False continuation的分数）（`tf.Tensor`类型，形状为`(batch_size, 2)`）。

        question_answering_score (`tf.Tensor` of shape `(batch_size, n_qa_answers)`):
        Prediction scores of question answering objective (classification).
        问题回答目标的预测分数（`tf.Tensor`类型，形状为`(batch_size, n_qa_answers)`）。

        language_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
        `(batch_size, sequence_length, hidden_size)`.
        语言隐藏状态（`tuple(tf.Tensor)`类型，当传入`output_hidden_states=True`参数或`config.output_hidden_states=True`时返回）：
        `tf.Tensor`元组，包含输入特征和每个跨模态层的输出，形状为`(batch_size, sequence_length, hidden_size)`。

        vision_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
        `(batch_size, sequence_length, hidden_size)`.
        视觉隐藏状态（`tuple(tf.Tensor)`类型，当传入`output_hidden_states=True`参数或`config.output_hidden_states=True`时返回）：
        `tf.Tensor`元组，包含输入特征和每个跨模态层的输出，形状为`(batch_size, sequence_length, hidden_size)`。

        language_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
        Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        语言注意力权重（`tuple(tf.Tensor)`类型，当传入`output_attentions=True`参数或`config.output_attentions=True`时返回）：
        `tf.Tensor`元组，包含每层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
        注意力softmax之后的注意力权重，用于计算自注意力头部的加权平均值。

        vision_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
        Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        视觉注意力权重（`tuple(tf.Tensor)`类型，当传入`output_attentions=True`参数或`config.output_attentions=True`时返回）：
        `tf.Tensor`元组，包含每层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
        注意力softmax之后的注意力权重，用于计算自注意力头部的加权平均值。

        cross_encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
        Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        跨编码器注意力权重（`tuple(tf.Tensor)`类型，当传入`output_attentions=True`参数或`config.output_attentions=True`时返回）：
        `tf.Tensor`元组，包含每层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
        注意力softmax之后的注意力权重，用于计算自注意力头部的加权平均值。
        
    """

    loss: tf.Tensor | None = None
    prediction_logits: tf.Tensor | None = None
    # 交叉关系得分，可能是 TensorFlow 张量或空值
    cross_relationship_score: tf.Tensor | None = None
    # 问题回答得分，可能是 TensorFlow 张量或空值
    question_answering_score: tf.Tensor | None = None
    # 语言隐藏状态的元组，可能是 TensorFlow 张量或空值
    language_hidden_states: Tuple[tf.Tensor] | None = None
    # 视觉隐藏状态的元组，可能是 TensorFlow 张量或空值
    vision_hidden_states: Tuple[tf.Tensor] | None = None
    # 语言注意力的元组，可能是 TensorFlow 张量或空值
    language_attentions: Tuple[tf.Tensor] | None = None
    # 视觉注意力的元组，可能是 TensorFlow 张量或空值
    vision_attentions: Tuple[tf.Tensor] | None = None
    # 交叉编码器注意力的元组，可能是 TensorFlow 张量或空值
    cross_encoder_attentions: Tuple[tf.Tensor] | None = None
class TFLxmertVisualFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 对象特征编码
        self.visn_fc = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="visn_fc",
        )
        # 对象特征层归一化
        self.visn_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="visn_layer_norm"
        )

        # 盒子位置编码
        self.box_fc = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="box_fc",
        )
        # 盒子位置层归一化
        self.box_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="box_layer_norm")

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.feat_dim = config.visual_feat_dim
        self.pos_dim = config.visual_pos_dim
        self.config = config

    def call(self, visn_input, training=False):
        feats, boxes = visn_input

        # 对象特征编码
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        # 盒子位置编码
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        # 对象特征编码和盒子位置编码的平均值
        output = (x + y) / 2

        output = self.dropout(output, training=training)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建对象特征编码层
        if getattr(self, "visn_fc", None) is not None:
            with tf.name_scope(self.visn_fc.name):
                self.visn_fc.build([None, None, self.feat_dim])
        # 构建对象特征层归一化层
        if getattr(self, "visn_layer_norm", None) is not None:
            with tf.name_scope(self.visn_layer_norm.name):
                self.visn_layer_norm.build([None, None, self.config.hidden_size])
        # 构建盒子位置编码层
        if getattr(self, "box_fc", None) is not None:
            with tf.name_scope(self.box_fc.name):
                self.box_fc.build([None, None, self.pos_dim])
        # 构建盒子位置层归一化层
        if getattr(self, "box_layer_norm", None) is not None:
            with tf.name_scope(self.box_layer_norm.name):
                self.box_layer_norm.build([None, None, self.config.hidden_size])


class TFLxmertEmbeddings(tf.keras.layers.Layer):
    """构建从单词、位置和令牌类型嵌入中的嵌入。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        # 层归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 这是一个 TensorFlow 层的 `build` 方法，用于初始化各种嵌入和层规范化（LayerNorm）
    def build(self, input_shape=None):
        # 创建一个命名作用域，给词嵌入权重加上一个标记
        with tf.name_scope("word_embeddings"):
            # 创建一个权重矩阵用于词嵌入，大小为 [词汇表大小, 隐藏层大小]
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                # 使用指定范围的初始化器初始化权重
                initializer=get_initializer(initializer_range=self.initializer_range),
            )
    
        # 创建一个命名作用域，给 Token 类型嵌入权重加上标记
        with tf.name_scope("token_type_embeddings"):
            # 创建一个权重矩阵用于 Token 类型嵌入，大小为 [类型词汇表大小, 隐藏层大小]
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )
    
        # 创建一个命名作用域，给位置嵌入权重加上标记
        with tf.name_scope("position_embeddings"):
            # 创建一个权重矩阵用于位置嵌入，大小为 [最大位置嵌入数, 隐藏层大小]
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )
    
        # 检查 `self.built` 是否为 True，如果是，则直接返回，表示已经构建过了
        if self.built:
            return
        
        # 将 `self.built` 设置为 True，表示已经构建
        self.built = True
        
        # 如果 `LayerNorm` 属性不为 None，则为其构建形状
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 为 LayerNorm 建立形状，预期的维度 [None, None, 隐藏层大小]
                self.LayerNorm.build([None, None, self.config.hidden_size])
    
    
    # 这是一个 TensorFlow 层的 `call` 方法，处理输入张量，并应用嵌入等操作
    def call(self, input_ids=None, token_type_ids=None, inputs_embeds=None, training=False):
        """
        应用基于输入张量的嵌入。
    
        返回:
            final_embeddings (`tf.Tensor`): 输出嵌入张量。
        """
        # 确保 `input_ids` 和 `inputs_embeds` 至少有一个不为空
        assert not (input_ids is None and inputs_embeds is None)
    
        # 如果 `input_ids` 不为空，则从权重中根据索引获取嵌入
        if input_ids is not None:
            # 检查 `input_ids` 是否在词汇表范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 获取嵌入，根据输入的 ID 索引权重矩阵
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
    
        # 获取输入嵌入的形状，并去掉最后一维
        input_shape = shape_list(inputs_embeds)[:-1]
    
        # 如果 `token_type_ids` 为空，则创建一个全零的形状与输入相同的张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
    
        # 为位置嵌入创建索引，从 0 开始
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        # 从位置嵌入权重中获取嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 从 Token 类型嵌入权重中获取嵌入
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        
        # 将输入嵌入、位置嵌入和 Token 类型嵌入相加
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        
        # 应用层规范化
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        
        # 应用 Dropout，如果 `training` 为 True，则启用 Dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
    
        # 返回最终嵌入
        return final_embeddings
class TFLxmertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(  # 抛出值错误异常，如果隐藏大小不能被注意力头数整除
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads  # 设置注意力头数
        assert config.hidden_size % config.num_attention_heads == 0  # 断言确保隐藏大小能被注意力头数整除
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 计算所有头的总大小

        self.query = tf.keras.layers.Dense(  # 创建查询层
            self.all_head_size,  # 输出大小为所有头的总大小
            kernel_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化权重
            name="query",  # 层的名称
        )
        self.key = tf.keras.layers.Dense(  # 创建键层，与查询层类似
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(  # 创建值层，与查询层类似
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)  # 创建丢弃层，用于注意力概率的丢弃
        self.ctx_dim = config.hidden_size  # 上下文维度设置为隐藏大小
        self.config = config  # 保存配置参数

    def transpose_for_scores(self, x, batch_size):
        # 重新调整形状，从 [batch_size, seq_length, all_head_size] 到 [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 转置张量，使得注意力头维度在中间
    # 传入参数包括隐藏状态、上下文、注意力掩码、是否输出注意力权重、是否在训练模式下
    def call(self, hidden_states, context, attention_mask, output_attentions, training=False):
        # 获取批处理大小
        batch_size = shape_list(hidden_states)[0]
        # 使用查询（Q）权重层对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)
        # 使用键（K）权重层对上下文进行处理
        mixed_key_layer = self.key(context)
        # 使用值（V）权重层对上下文进行处理
        mixed_value_layer = self.value(context)

        # 将查询层转置以便进行注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将键层转置以便进行注意力计算
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将值层转置以便进行注意力计算
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 通过"查询"和"键"的点积得到原始注意力分数
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], dtype=attention_scores.dtype)  # 缩放注意力分数
        attention_scores = attention_scores / tf.math.sqrt(dk)

        # 如果存在注意力掩码
        if attention_mask is not None:
            # 将注意力掩码转换为注意力分数数据类型并加到注意力分数上
            attention_mask = tf.cast(attention_mask, dtype=attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        # 将注意力分数规范化为概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 进行注意力概率的dropout处理
        attention_probs = self.dropout(attention_probs, training=training)
        # 对值层进行加权求和得到上下文向量
        context_layer = tf.matmul(attention_probs, value_layer)

        # 进行形状变换
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        # 如果需要输出注意力权重，则将上下文向量和注意力权重一起返回，否则只返回上下文向量
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    # 构建注意力层
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在查询、键、值权重层，则分别进行构建
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.ctx_dim])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.ctx_dim])
class TFLxmertIntermediate(tf.keras.layers.Layer):
    # 初始化函数，用于设置层的参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于进行线性变换
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size,
            # 使用指定的初始化器初始化权重
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 判断激活函数类型，如果是字符串则获取对应的 TensorFlow 激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 前向传播函数，定义了层的计算逻辑
    def call(self, hidden_states):
        # 全连接层的前向计算
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    # 构建层，主要用于在首次调用层的前向传播函数时创建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFLxmertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建全连接层
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 创建 LayerNormalization 层，用于规范化输入
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states, input_tensor, training=False):
        # 全连接层的前向传播
        hidden_states = self.dense(hidden_states)
        # Dropout 层的应用
        hidden_states = self.dropout(hidden_states, training)
        # 输入与输出的残差连接，再加上 LayerNormalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFLxmertAttentionOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建全连接层
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 创建 LayerNormalization 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.config = config
    # 定义 call 方法，用于基于输入 hidden_states 和 input_tensor 计算输出
    def call(self, hidden_states, input_tensor, training=False):
        # 使用全连接层 dense 对 hidden_states 进行变换
        hidden_states = self.dense(hidden_states)
        # 根据 training 参数决定是否使用 dropout 层对 hidden_states 进行 dropout 操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 利用 LayerNorm 层对 hidden_states 和 input_tensor 进行归一化并相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回计算得到的 hidden_states
        return hidden_states
    
    # 定义 build 方法，用于初始化模型层
    def build(self, input_shape=None):
        # 如果模型已经构建好了，直接返回
        if self.built:
            return
        # 标记模型已经构建完成
        self.built = True
        # 如果存在 dense 层，则对其进行构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层，则对其进行构建
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# TFLxmertSelfAttentionLayer 是一个 Keras 层,实现了基于 Lxmert 模型的自注意力机制
class TFLxmertSelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 TFLxmertAttention 层, 用于执行自注意力计算
        self.self = TFLxmertAttention(config, name="self")
        # 创建一个 TFLxmertAttentionOutput 层, 用于处理自注意力的输出
        self.attention_output = TFLxmertAttentionOutput(config, name="output")

    # 定义前向计算过程
    def call(self, input_tensor, attention_mask, output_attentions, training=False):
        # 使用 input_tensor 作为 key 和 query 执行自注意力计算
        self_output = self.self(input_tensor, input_tensor, attention_mask, output_attentions)
        # 如果需要输出注意力权重, 则获取注意力权重
        if output_attentions:
            attention_probs = self_output[1]
        # 使用自注意力输出和 input_tensor 计算最终的注意力输出
        attention_output = self.attention_output(self_output[0], input_tensor)
        # 根据是否需要输出注意力权重, 返回相应的结果
        return (attention_output, attention_probs) if output_attentions else (attention_output,)

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建内部的 TFLxmertAttention 层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 构建内部的 TFLxmertAttentionOutput 层
        if getattr(self, "attention_output", None) is not None:
            with tf.name_scope(self.attention_output.name):
                self.attention_output.build(None)


# TFLxmertCrossAttentionLayer 是一个 Keras 层, 实现了基于 Lxmert 模型的交叉注意力机制
class TFLxmertCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 TFLxmertAttention 层, 用于执行交叉注意力计算
        self.att = TFLxmertAttention(config, name="att")
        # 创建一个 TFLxmertAttentionOutput 层, 用于处理交叉注意力的输出
        self.attention_output = TFLxmertAttentionOutput(config, name="output")

    # 定义前向计算过程
    def call(
        self,
        input_tensor,
        ctx_tensor,
        ctx_att_mask,
        output_attentions=False,
        training=False,
    ):
        # 使用 input_tensor 作为 query, ctx_tensor 作为 key 和 value 执行交叉注意力计算
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions, training=training)
        # 如果需要输出注意力权重, 则获取注意力权重
        if output_attentions:
            attention_probs = output[1]
        # 使用交叉注意力输出和 input_tensor 计算最终的注意力输出
        attention_output = self.attention_output(output[0], input_tensor, training=training)
        # 根据是否需要输出注意力权重, 返回相应的结果
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建内部的 TFLxmertAttention 层
        if getattr(self, "att", None) is not None:
            with tf.name_scope(self.att.name):
                self.att.build(None)
        # 构建内部的 TFLxmertAttentionOutput 层
        if getattr(self, "attention_output", None) is not None:
            with tf.name_scope(self.attention_output.name):
                self.attention_output.build(None)


# TFLxmertLayer 是一个 Keras 层,包含了 Lxmert 模型中的自注意力层、前馈层和输出层
class TFLxmertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个 TFLxmertSelfAttentionLayer 层, 用于执行自注意力计算
        self.attention = TFLxmertSelfAttentionLayer(config, name="attention")
        # 创建一个 TFLxmertIntermediate 层, 用于执行前馈计算
        self.intermediate = TFLxmertIntermediate(config, name="intermediate")
        # 创建一个 TFLxmertOutput 层, 用于计算最终的输出
        self.transformer_output = TFLxmertOutput(config, name="output")
    # 定义一个方法 call，该方法用于执行模型前向传播
    def call(self, hidden_states, attention_mask, output_attentions, training=False):
        # 执行注意力机制的前向传播，获取注意力输出和其他输出
        attention_outputs = self.attention(hidden_states, attention_mask, output_attentions, training=training)
        # 获取注意力输出
        attention_output = attention_outputs[0]
        # 执行中间层的前向传播
        intermediate_output = self.intermediate(attention_output)
        # 执行最终输出层的前向传播，获取最终的层输出
        layer_output = self.transformer_output(intermediate_output, attention_output, training=training)
        # 将注意力输出与最终输出组合成最终的模型输出
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs
    
    # 定义一个方法 build，该方法用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型已经构建完成
        self.built = True
        # 如果模型包含注意力机制模块，则构建注意力机制模块
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果模型包含中间层模块，则构建中间层模块
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果模型包含最终输出层模块，则构建最终输出层模块
        if getattr(self, "transformer_output", None) is not None:
            with tf.name_scope(self.transformer_output.name):
                self.transformer_output.build(None)
class TFLxmertXLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.visual_attention = TFLxmertCrossAttentionLayer(config, name="visual_attention")

        # Self-attention Layers
        self.lang_self_att = TFLxmertSelfAttentionLayer(config, name="lang_self_att")
        self.visn_self_att = TFLxmertSelfAttentionLayer(config, name="visn_self_att")

        # Intermediate and Output Layers (FFNs)
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
        # Cross Attention

        # Keras saving and loading model *does not work* with the same inputs for two layers.
        lang_attention_lang_input = tf.identity(lang_input)
        visn_attention_lang_input = tf.identity(lang_input)
        lang_attention_visn_input = tf.identity(visn_input)
        visn_attention_visn_input = tf.identity(visn_input)

        # 执行语言输入和视觉输入之间的交叉注意力计算
        lang_att_output = self.visual_attention(
            lang_attention_lang_input,
            lang_attention_visn_input,
            visn_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
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
        # Self Attention
        output_attentions = False
        # 执行语言输入和视觉输入的自注意力计算
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions, training=training)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask, output_attentions, training=training)
        return lang_att_output[0], visn_att_output[0]

    def output_fc(self, lang_input, visn_input, training=False):
        # FC layers
        # 执行全连接层计算
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        # 输出层计算
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
        # 将语言特征和视觉特征作为注意力的输出
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        # 通过跨模态注意力层处理语言和视觉特征
        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output,  # 语言注意力输出
            lang_attention_mask,  # 语言注意力掩码
            visn_att_output,  # 视觉注意力输出
            visn_attention_mask,  # 视觉注意力掩码
            output_attentions,  # 是否输出注意力权重
            training=training,  # 是否处于训练模式
        )
        attention_probs = lang_att_output[1:]  # 注意力权重

        # 通过自注意力层处理语言和视觉特征
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output[0],  # 语言特征
            lang_attention_mask,  # 语言注意力掩码
            visn_att_output[0],  # 视觉特征
            visn_attention_mask,  # 视觉注意力掩码
            training=training,  # 是否处于训练模式
        )

        # 使用全连接层输出语言和视觉特征
        lang_output, visn_output = self.output_fc(
            lang_att_output,  # 语言特征
            visn_att_output,  # 视觉特征
            training=training,  # 是否处于训练模式
        )

        # 如果需要输出注意力权重，则返回语言输出、视觉输出和注意力权重；否则返回语言输出和视觉输出
        return (lang_output, visn_output, attention_probs[0]) if output_attentions else (lang_output, visn_output)

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True

        # 如果存在视觉注意力层，则构建视觉注意力层
        if getattr(self, "visual_attention", None) is not None:
            with tf.name_scope(self.visual_attention.name):
                self.visual_attention.build(None)

        # 如果存在语言自注意力层，则构建语言自注意力层
        if getattr(self, "lang_self_att", None) is not None:
            with tf.name_scope(self.lang_self_att.name):
                self.lang_self_att.build(None)

        # 如果存在视觉自注意力层，则构建视觉自注意力层
        if getattr(self, "visn_self_att", None) is not None:
            with tf.name_scope(self.visn_self_att.name):
                self.visn_self_att.build(None)

        # 如果存在语言交互层，则构建语言交互层
        if getattr(self, "lang_inter", None) is not None:
            with tf.name_scope(self.lang_inter.name):
                self.lang_inter.build(None)

        # 如果存在语言输出层，则构建语言输出层
        if getattr(self, "lang_output", None) is not None:
            with tf.name_scope(self.lang_output.name):
                self.lang_output.build(None)

        # 如果存在视觉交互层，则构建视觉交互层
        if getattr(self, "visn_inter", None) is not None:
            with tf.name_scope(self.visn_inter.name):
                self.visn_inter.build(None)

        # 如果存在视觉输出层，则构建视觉输出层
        if getattr(self, "visn_output", None) is not None:
            with tf.name_scope(self.visn_output.name):
                self.visn_output.build(None)
# 定义 TFLxmertEncoder 类，继承自 tf.keras.layers.Layer
class TFLxmertEncoder(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 TFLxmertVisualFeatureEncoder 对象并命名为 visn_fc
        self.visn_fc = TFLxmertVisualFeatureEncoder(config, name="visn_fc")

        # 设置层的数量
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # 定义层
        # 使用 self.layer 而不是 self.l_layer 是为了支持加载 BERT 权重
        self.layer = [TFLxmertLayer(config, name=f"layer_._{i}") for i in range(self.num_l_layers)]
        self.x_layers = [TFLxmertXLayer(config, name=f"x_layers_._{i}") for i in range(self.num_x_layers)]
        self.r_layers = [TFLxmertLayer(config, name=f"r_layers_._{i}") for i in range(self.num_r_layers)]
        self.config = config

    # 调用方法
    def call(
        self,
        lang_feats=None,
        lang_attention_mask=None,
        visual_feats=None,
        visual_pos=None,
        visual_attention_mask=None,
        output_attentions=None,
        training=False,
        ):  
        # 初始化空元组，用于存储视觉隐状态
        vision_hidden_states = ()
        # 初始化空元组，用于存储语言隐状态
        language_hidden_states = ()
        # 如果输出注意力或者配置中要求输出注意力，则初始化空元组，用于存储视觉注意力
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        # 如果输出注意力或者配置中要求输出注意力，则初始化空元组，用于存储语言注意力
        language_attentions = () if output_attentions or self.config.output_attentions else None
        # 如果输出注意力或者配置中要求输出注意力，则初始化空元组，用于存储交叉编码器注意力
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

        # 将视觉特征通过视觉全连接层进行处理
        visual_feats = self.visn_fc([visual_feats, visual_pos], training=training)

        # 遍历语言层
        for layer_module in self.layer:
            # 对语言特征进行层次模块处理
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions, training=training)
            # 更新语言特征
            lang_feats = l_outputs[0]
            # 将新的语言隐状态添加到语言隐状态元组中
            language_hidden_states = language_hidden_states + (lang_feats,)
            # 如果语言注意力不为空，则将注意力结果添加到语言注意力元组中
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)

        # 遍历关系层
        for layer_module in self.r_layers:
            # 对视觉特征进行关系层次模块处理
            v_outputs = layer_module(
                visual_feats,
                visual_attention_mask,
                output_attentions,
                training=training,
            )
            # 更新视觉特征
            visual_feats = v_outputs[0]
            # 将新的视觉隐状态添加到视觉隐状态元组中
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            # 如果视觉注意力不为空，则将注意力结果添加到视觉注意力元组中
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)

        # 遍历跨模态层
        for layer_module in self.x_layers:
            # 对语言特征和视觉特征进行跨模态层次模块处理
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions,
                training=training,
            )
            # 更新语言特征和视觉特征
            lang_feats, visual_feats = x_outputs[:2]
            # 将新的视觉隐状态添加到视觉隐状态元组中
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            # 将新的语言隐状态添加到语言隐状态元组中
            language_hidden_states = language_hidden_states + (lang_feats,)
            # 如果交叉编码器注意力不为空，则将注意力结果添加到交叉编码器注意力元组中
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)

        # 构建视觉编码器输出元组
        visual_encoder_outputs = (
            vision_hidden_states,
            vision_attentions if output_attentions else None,
        )
        # 构建语言编码器输出元组
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )

        # 返回编码器的输出元组
        return (
            visual_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )
    # 定义一个方法用于搭建神经网络结构，input_shape用于指定输入数据的形状
    def build(self, input_shape=None):
        # 如果已经搭建过网络，则直接返回
        if self.built:
            return
        # 标记为已搭建网络
        self.built = True
        # 如果存在名为"visn_fc"的属性，则对其进行搭建
        if getattr(self, "visn_fc", None) is not None:
            # 使用tf.name_scope为操作指定名称空间
            with tf.name_scope(self.visn_fc.name):
                # 对"visn_fc"进行搭建
                self.visn_fc.build(None)
        # 如果存在名为"layer"的属性，则对其中每个层进行搭建
        if getattr(self, "layer", None) is not None:
            # 遍历每个层
            for layer in self.layer:
                # 使用tf.name_scope为操作指定名称空间
                with tf.name_scope(layer.name):
                    # 对每个层进行搭建
                    layer.build(None)
        # 如果存在名为"x_layers"的属性，则对其中每个层进行搭建
        if getattr(self, "x_layers", None) is not None:
            # 遍历每个层
            for layer in self.x_layers:
                # 使用tf.name_scope为操作指定名称空间
                with tf.name_scope(layer.name):
                    # 对每个层进行搭建
                    layer.build(None)
        # 如果存在名为"r_layers"的属性，则对其中每个层进行搭建
        if getattr(self, "r_layers", None) is not None:
            # 遍历每个层
            for layer in self.r_layers:
                # 使用tf.name_scope为操作指定名称空间
                with tf.name_scope(layer.name):
                    # 对每个层进行搭建
                    layer.build(None)
# 使用 keras_serializable 装饰器，将该类标记为可序列化的 Keras 层
@keras_serializable
# 定义 TFLxmertMainLayer 类，继承自 tf.keras.layers.Layer 类
class TFLxmertMainLayer(tf.keras.layers.Layer):
    # 设置 config_class 类属性为 LxmertConfig 类
    config_class = LxmertConfig

    # 初始化方法，接受 config 和 **kwargs 参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的 config 参数赋值给 self.config 属性
        self.config = config
        # 设置 self.num_l_layers 属性为 config 的 l_layers 属性
        self.num_l_layers = config.l_layers
        # 设置 self.num_x_layers 属性为 config 的 x_layers 属性
        self.num_x_layers = config.x_layers
        # 设置 self.num_r_layers 属性为 config 的 r_layers 属性
        self.num_r_layers = config.r_layers
        # 设置 self.initializer_range 属性为 config 的 initializer_range 属性
        self.initializer_range = config.initializer_range
        # 设置 self.output_attentions 属性为 config 的 output_attentions 属性
        self.output_attentions = config.output_attentions
        # 设置 self.output_hidden_states 属性为 config 的 output_hidden_states 属性
        self.output_hidden_states = config.output_hidden_states
        # 设置 self.return_dict 属性为 config 的 use_return_dict 属性
        self.return_dict = config.use_return_dict
        # 创建 TFLxmertEmbeddings 对象，传入 config 参数，设置为 self.embeddings 属性
        self.embeddings = TFLxmertEmbeddings(config, name="embeddings")
        # 创建 TFLxmertEncoder 对象，传入 config 参数，设置为 self.encoder 属性
        self.encoder = TFLxmertEncoder(config, name="encoder")
        # 创建 TFLxmertPooler 对象，传入 config 参数，设置为 self.pooler 属性
        self.pooler = TFLxmertPooler(config, name="pooler")
        # 将传入的 config 参数再次赋值给 self.config 属性

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        # 返回 self.embeddings 属性
        return self.embeddings

    # 设置输入嵌入的方法，接受 value 参数
    def set_input_embeddings(self, value):
        # 将 value 赋值给 self.embeddings.weight 属性
        self.embeddings.weight = value
        # 设置 self.embeddings.vocab_size 属性为 value 的第一维度大小

    # 剪枝注意力头的方法，接受 heads_to_prune 参数
    def _prune_heads(self, heads_to_prune):
        # 抛出未实现错误
        raise NotImplementedError

    # call 方法，实现了模型的正向传播
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
        # 方法的具体实现在后面的代码块中，这里只是定义了方法的参数

    # build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果 self.embeddings 属性存在
        if getattr(self, "embeddings", None) is not None:
            # 在名为 self.embeddings 的命名空间下构建嵌入层
            with tf.name_scope(self.embeddings.name):
                # 调用 self.embeddings 的 build 方法，传入 None 参数
                self.embeddings.build(None)
        # 如果 self.encoder 属性存在
        if getattr(self, "encoder", None) is not None:
            # 在名为 self.encoder 的命名空间下构建编码器层
            with tf.name_scope(self.encoder.name):
                # 调用 self.encoder 的 build 方法，传入 None 参数
                self.encoder.build(None)
        # 如果 self.pooler 属性存在
        if getattr(self, "pooler", None) is not None:
            # 在名为 self.pooler 的命名空间下构建池化器层
            with tf.name_scope(self.pooler.name):
                # 调用 self.pooler 的 build 方法，传入 None 参数
                self.pooler.build(None)


# 定义 TFLxmertPreTrainedModel 类，继承自 TFPreTrainedModel 类
class TFLxmertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 config_class 类属性为 LxmertConfig 类
    config_class = LxmertConfig
    # 设置 base_model_prefix 类属性为 "lxmert"
    base_model_prefix = "lxmert"

    # dummy_inputs 属性，用于返回构建网络所需的虚拟输入
    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        # 设置 batch_size 为 2
        batch_size = 2
        # 设置 num_visual_features 为 10
        num_visual_features = 10
        # 创建输入序列的张量，形状为 (2, 3)，数据为固定值
        input_ids = tf.constant([[3, 5, 6], [2, 3, 4]], dtype=tf.int32)
        # 创建视觉特征的张量，形状为 (2, 10, visual_feat_dim)，数据为均匀分布的随机值
        visual_feats = tf.random.uniform((batch_size, num_visual_features, self.config.visual_feat_dim))
        # 创建视觉位置编码的张量，形状为 (2, 10, 4)，数据为均匀分布的随机值
        visual_pos = tf.random.uniform((batch_size, num_visual_features, 4))

        # 返回一个字典，包含输入的虚拟数据
        return {
            "input_ids": input_ids,
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
        }
    # 定义输入签名，指定了模型输入的数据类型和形状
    def input_signature(self):
        # 输入文本的词索引序列
        return {
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            # 注意力遮罩，用于指示哪些位置是填充的
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            # 视觉特征的张量规范
            "visual_feats": tf.TensorSpec((None, None, self.config.visual_feat_dim), tf.float32, name="visual_feats"),
            # 视觉位置编码的张量规范
            "visual_pos": tf.TensorSpec((None, None, 4), tf.float32, name="visual_pos"),
            # 视觉注意力遮罩，指示哪些位置是填充的
            "visual_attention_mask": tf.TensorSpec((None, None), tf.int32, name="visual_attention_mask"),
            # 标识文本对应的段落ID
            "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }
# LXMERT 模型的文档字符串，介绍了模型的提出背景、结构和用法
LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from
    Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It's a vision and language transformer
    model, pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual
    genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss
    for question answering attribute prediction, and object tag prediction.

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

    Parameters:
        config ([`LxmertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# LXMERT 模型的输入文档字符串，待补充
LXMERT_INPUTS_DOCSTRING = r"""
"""


# 添加 LXMERT 模型文档字符串作为类的注释
@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
# 定义 TFLxmertModel 类，继承自 TFLxmertPreTrainedModel 类
class TFLxmertModel(TFLxmertPreTrainedModel):
    # 初始化 TFLxmertModel 类
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFLxmertMainLayer 对象，并将其赋值给 self.lxmert 属性
        self.lxmert = TFLxmertMainLayer(config, name="lxmert")
    
    # 解包输入参数的装饰器
    @unpack_inputs
    # 为模型 forward 方法添加文档字符串的装饰器
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    # 为模型 forward 方法添加代码示例的装饰器
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLxmertModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的 forward 方法
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
        # 调用 self.lxmert 的 forward 方法，并将输入参数传递给它
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
        # 返回 lxmert 模型的输出
        return outputs
    
    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将 built 属性设置为 True，表示模型已经构建完成
        self.built = True
        # 如果 self.lxmert 对象存在
        if getattr(self, "lxmert", None) is not None:
            # 在 self.lxmert.name 范围内构建 self.lxmert 对象
            with tf.name_scope(self.lxmert.name):
                self.lxmert.build(None)
# 定义一个自定义层 TFLxmertPooler，继承自 tf.keras.layers.Layer
class TFLxmertPooler(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于对隐藏状态进行处理
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,  # 隐藏单元的数量
            kernel_initializer=get_initializer(config.initializer_range),  # 权重初始化器
            activation="tanh",  # 激活函数为双曲正切
            name="dense",  # 层名称
        )
        # 保存配置信息
        self.config = config

    # 定义层的前向传播逻辑
    def call(self, hidden_states):
        # 选择第一个 token 对应的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态通过全连接层进行处理
        pooled_output = self.dense(first_token_tensor)
        return pooled_output

    # 构建层，主要是设置层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果全连接层已经定义，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform 复制并修改为 Lxmert 的预测头变换层
class TFLxmertPredictionHeadTransform(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config: LxmertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于特征变换
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,  # 隐藏单元的数量
            kernel_initializer=get_initializer(config.initializer_range),  # 权重初始化器
            name="dense",  # 层名称
        )

        # 激活函数处理函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 层标准化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 保存配置信息
        self.config = config

    # 前向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 特征变换
        hidden_states = self.dense(inputs=hidden_states)
        # 激活函数处理
        hidden_states = self.transform_act_fn(hidden_states)
        # 标准化处理
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    # 构建层，主要是设置层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果全连接层已经定义，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果标准化层已经定义，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead 复制并修改为 Lxmert 的语言模型预测头层
class TFLxmertLMPredictionHead(tf.keras.layers.Layer):
    # 初始化方法，接受一个 LxmertConfig 对象、一个输入嵌入层对象以及其他关键字参数
    def __init__(self, config: LxmertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将配置信息和隐藏层大小存储在实例变量中
        self.config = config
        self.hidden_size = config.hidden_size

        # 创建一个 TFLxmertPredictionHeadTransform 实例并命名为 "transform"
        self.transform = TFLxmertPredictionHeadTransform(config, name="transform")

        # 将输入嵌入层对象存储在实例变量中
        self.input_embeddings = input_embeddings

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 添加一个名为 "bias" 的可训练的零初始化权重
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 transform 层，则构建 transform 层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出嵌入层对象的方法
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层对象的方法
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置的方法
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置的方法
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 前向传播方法，接受隐藏状态张量并返回预测的标记概率分布
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过 transform 层处理隐藏状态
        hidden_states = self.transform(hidden_states=hidden_states)
        # 获取序列长度
        seq_length = shape_list(hidden_states)[1]
        # 重新整形隐藏状态张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 计算隐藏状态和输入嵌入层权重之间的矩阵乘积
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重新整形为原始形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states
# 从transformers.models.bert.modeling_tf_bert.TFBertMLMHead复制代码，并将Bert->Lxmert
class TFLxmertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: LxmertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 创建TFLxmertLMPredictionHead对象，并命名为predictions
        self.predictions = TFLxmertLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用predictions对象的call方法，传入sequence_output参数，得到预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 调用predictions对象的build方法，传入None参数
                self.predictions.build(None)


class TFLxmertPreTrainingHeads(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        # 创建TFLxmertLMPredictionHead对象，并命名为predictions
        self.predictions = TFLxmertLMPredictionHead(config, input_embeddings, name="predictions")

        # 创建包含两个输出的全连接层对象，并命名为seq_relationship
        self.seq_relationship = tf.keras.layers.Dense(
            2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="seq_relationship",
        )
        # 设置配置参数为当前对象的config属性
        self.config = config

    def call(self, sequence_output, pooled_output):
        # 通过predictions对象预测sequence_output的分数
        prediction_scores = self.predictions(sequence_output)
        # 通过seq_relationship对象预测pooled_output的关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 调用predictions对象的build方法，传入None参数
                self.predictions.build(None)
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                # 调用seq_relationship对象的build方法，传入[None, None, self.config.hidden_size]参数
                self.seq_relationship.build([None, None, self.config.hidden_size])


class TFLxmertVisualAnswerHead(tf.keras.layers.Layer):
    def __init__(self, config, num_labels, **kwargs):
        super().__init__(**kwargs)
        # 创建全连接层对象，并命名为logit_fc_._0
        hid_dim = config.hidden_size
        self.dense = tf.keras.layers.Dense(
            hid_dim * 2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="logit_fc_._0",
        )
        # 创建指定激活函数的对象
        self.activation = get_tf_activation("gelu")
        # 创建LayerNormalization层对象，并命名为logit_fc_._2
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="logit_fc_._2")
        # 创建全连接层对象，并命名为logit_fc_._3
        self.dense_1 = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="logit_fc_._3",
        )
        # 设置hid_dim属性值为config.hidden_size
        self.hid_dim = hid_dim
    # 定义一个 call 方法，接收隐藏状态作为输入
    def call(self, hidden_states):
        # 将输入的隐藏状态通过一个全连接层进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用激活函数
        hidden_states = self.activation(hidden_states)
        # 对激活后的隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态通过另一个全连接层进行变换
        hidden_states = self.dense_1(hidden_states)
        # 返回变换后的隐藏状态
        return hidden_states
    
    # 定义一个 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好了，就直接返回
        if self.built:
            return
        # 标记模型已经构建好了
        self.built = True
        # 如果 dense 层已经存在，就构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hid_dim])
        # 如果 layer_norm 层已经存在，就构建它
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, self.hid_dim * 2])
        # 如果 dense_1 层已经存在，就构建它
        if getattr(self, "dense_1", None) is not None:
            with tf.name_scope(self.dense_1.name):
                self.dense_1.build([None, None, self.hid_dim * 2])
class TFLxmertVisualObjHead(tf.keras.layers.Layer):
    # TFLxmertVisualObjHead 类的构造函数
    def __init__(self, config, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 创建一个 TFLxmertPredictionHeadTransform 实例
        self.transform = TFLxmertPredictionHeadTransform(config, name="transform")

        # 决定是否使用视觉损失
        visual_losses = {}
        # 如果配置中包含视觉对象损失，则定义对象损失形状和数量
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        # 如果配置中包含视觉属性损失，则定义属性损失形状和数量
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        # 如果配置中包含视觉特征损失，则定义特征损失形状和数量
        if config.visual_feat_loss:
            visual_losses["feat"] = {"shape": (-1, 2048), "num": config.visual_feat_dim}
        # 将视觉损失信息保存在类属性中
        self.visual_losses = visual_losses

        # 输出权重与输入嵌入相同，但是每个令牌都有一个只输出的偏置。
        self.decoder_dict = {
            # 为每个视觉损失创建一个 Dense 层，用于预测对应的输出
            key: tf.keras.layers.Dense(
                self.visual_losses[key]["num"],
                kernel_initializer=get_initializer(config.initializer_range),
                name=f"decoder_dict.{key}",
            )
            for key in self.visual_losses
        }
        # 保存配置信息
        self.config = config

    # TFLxmertVisualObjHead 的前向传播方法
    def call(self, hidden_states):
        # 将隐藏状态转换为预测头部
        hidden_states = self.transform(hidden_states)
        output = {}
        # 对每个视觉损失进行预测
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output

    # 构建 TFLxmertVisualObjHead 层
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在变换层，则构建变换层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)
        # 如果存在解码器字典，则构建解码器字典中的每个层
        if getattr(self, "decoder_dict", None) is not None:
            for layer in self.decoder_dict.values():
                with tf.name_scope(layer.name):
                    layer.build([None, None, self.config.hidden_size])


@add_start_docstrings("""Lxmert Model with a `language modeling` head on top.""", LXMERT_START_DOCSTRING)
# TFLxmertForPreTraining 类的声明，继承自 TFLxmertPreTrainedModel
class TFLxmertForPreTraining(TFLxmertPreTrainedModel):
```  
    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用基类的初始化函数
        super().__init__(config, *inputs, **kwargs)
    
        # 保存配置信息
        self.config = config
        # 设置问答标签的数量
        self.num_qa_labels = config.num_qa_labels
        # 设置视觉损失的标准化因子
        self.visual_loss_normalizer = config.visual_loss_normalizer
    
        # 使用不同的预训练任务
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa
    
        # 创建 LXMERT 主干网络
        self.lxmert = TFLxmertMainLayer(config, name="lxmert")
    
        # 创建预训练的头部网络
        self.cls = TFLxmertPreTrainingHeads(config, self.lxmert.embeddings, name="cls")
        if self.task_obj_predict:
            self.obj_predict_head = TFLxmertVisualObjHead(config, name="obj_predict_head")
        if self.task_qa:
            self.answer_head = TFLxmertVisualAnswerHead(config, self.num_qa_labels, name="answer_head")
    
        # 定义损失函数
        self.loss_fcts = {
            "l2": tf.keras.losses.Huber(delta=1.0, name="huber_loss"),
            "visn_ce": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "ce": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        }
    
        # 定义视觉损失的详细信息
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),
                "num": config.num_object_labels,
                "loss": "visn_ce",
            }
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),
                "num": config.num_attr_labels,
                "loss": "visn_ce",
            }
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        self.visual_losses = visual_losses
    
    # 定义属性
    @property
    # 定义一个生成 dummy 输入数据的方法
    def dummy_inputs(self):
        # 设置 batch size 为 2，视觉特征数量为 10
        batch_size = 2
        num_visual_features = 10
    
        # 创建随机的输入 ID 张量
        input_ids = tf.constant([[3, 5, 6], [2, 3, 4]], dtype=tf.int32)
    
        # 创建随机的视觉特征张量
        visual_feats = tf.random.uniform((batch_size, num_visual_features, self.config.visual_feat_dim))
    
        # 创建随机的视觉位置张量
        visual_pos = tf.random.uniform((batch_size, num_visual_features, 4))
    
        # 如果需要进行目标对象预测任务
        if self.config.task_obj_predict:
            obj_labels = {}
    
        # 如果需要进行视觉属性损失和目标对象预测任务
        if self.config.visual_attr_loss and self.config.task_obj_predict:
            obj_labels["attr"] = (
                tf.ones([batch_size, num_visual_features]),
                tf.ones([batch_size, num_visual_features]),
            )
    
        # 如果需要进行视觉特征损失和目标对象预测任务
        if self.config.visual_feat_loss and self.config.task_obj_predict:
            obj_labels["feat"] = (
                tf.ones([batch_size, num_visual_features, self.config.visual_feat_dim]),
                tf.ones([batch_size, num_visual_features]),
            )
    
        # 如果需要进行视觉对象损失和目标对象预测任务
        if self.config.visual_obj_loss and self.config.task_obj_predict:
            obj_labels["obj"] = (
                tf.ones([batch_size, num_visual_features]),
                tf.ones([batch_size, num_visual_features]),
            )
    
        # 返回一个包含输入 ID、视觉特征和视觉位置的字典
        # 如果需要进行目标对象预测任务，还会包含 obj_labels 字典
        return {
            **{
                "input_ids": input_ids,
                "visual_feats": visual_feats,
                "visual_pos": visual_pos,
            },
            **({"obj_labels": obj_labels} if self.config.task_obj_predict else {}),
        }
    
    # 获取语言模型头部
    def get_lm_head(self):
        return self.cls.predictions
    
    # 获取前缀偏差名称（已弃用，请使用 get_bias 方法）
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.cls.name + "/" + self.cls.predictions.name
    
    # 定义模型的前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
    ):
        # 该方法的具体实现省略
        pass
    # 创建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好，则直接返回
        if self.built:
            return
        # 设置模型状态为已构建
        self.built = True
        # 如果模型中有"lxmert"属性，调用build方法并命名空间为self.lxmert.name
        if getattr(self, "lxmert", None) is not None:
            with tf.name_scope(self.lxmert.name):
                self.lxmert.build(None)
        # 如果模型中有"cls"属性，调用build方法并命名空间为self.cls.name
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
        # 如果模型中有"obj_predict_head"属性，调用build方法并命名空间为self.obj_predict_head.name
        if getattr(self, "obj_predict_head", None) is not None:
            with tf.name_scope(self.obj_predict_head.name):
                self.obj_predict_head.build(None)
        # 如果模型中有"answer_head"属性，调用build方法并命名空间为self.answer_head.name
        if getattr(self, "answer_head", None) is not None:
            with tf.name_scope(self.answer_head.name):
                self.answer_head.build(None)
```