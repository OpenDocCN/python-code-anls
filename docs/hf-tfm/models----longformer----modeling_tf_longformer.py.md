# `.\transformers\models\longformer\modeling_tf_longformer.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 The Allen Institute for AI team 和 The HuggingFace Inc. team 所有
# 基于 Apache License, Version 2.0（“许可证”）发行此文件；除非遵守许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，根据许可证分发的软件是在“原样”基础上分发的，没有任何明示或暗示的担保或条件
# 请参阅许可证以了解特定语言控制的权限和限制
"""Tensorflow Longformer model."""  # Tensorflow Longformer 模型

from __future__ import annotations  # 开启对 Python 3.10 新注解语法的支持

# 引入警告模块
import warnings
# 引入 dataclasses 模块中的 dataclass 装饰器
from dataclasses import dataclass
# 引入类型提示相关的模块
from typing import Optional, Tuple, Union

# 引入 numpy 模块
import numpy as np
# 引入 tensorflow 模块
import tensorflow as tf

# 引入 Hugging Face 提供的一些 TensorFlow 相关的工具函数和类
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
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
# 引入一些工具函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 引入一些实用函数和类
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 引入 Longformer 配置类
from .configuration_longformer import LongformerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 为了文档中的检查点而设置的常量
_CHECKPOINT_FOR_DOC = "allenai/longformer-base-4096"
_CONFIG_FOR_DOC = "LongformerConfig"

# 用于stable_softmax函数的一个常量
LARGE_NEGATIVE = -1e8

# 存储 Longformer 预训练模型的列表
TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # 查看所有 Longformer 模型：https://huggingface.co/models?filter=longformer
]
# 定义 TFLongformerBaseModelOutput 类，继承自 ModelOutput 类
@dataclass
class TFLongformerBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.
    # 定义函数参数，表示模型最后一层的隐藏状态
    last_hidden_state: tf.Tensor = None
    # 定义函数参数，表示模型各层的隐藏状态，可选参数，当设置了output_hidden_states=True或config.output_hidden_states=True时返回
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义函数参数，表示模型各层的注意力权重，可选参数，当设置了output_attentions=True或config.output_attentions=True时返回
    attentions: Tuple[tf.Tensor] | None = None
    # 定义函数参数，表示模型各层的全局注意力权重，可选参数，当设置了output_attentions=True或config.output_attentions=True时返回
    global_attentions: Tuple[tf.Tensor] | None = None
# 使用 @dataclass 装饰器定义一个数据类 TFLongformerBaseModelOutputWithPooling，包含 Longformer 输出和最后隐藏状态池化输出
class TFLongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for Longformer's outputs that also contains a pooling of the last hidden states.

    """

    # 最后隐藏状态的张量
    last_hidden_state: tf.Tensor = None
    # 池化输出的张量
    pooler_output: tf.Tensor = None
    # 隐藏状态的元组
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力的元组
    attentions: Tuple[tf.Tensor] | None = None
    # 全局注意力的元组
    global_attentions: Tuple[tf.Tensor] | None = None


# 使用 @dataclass 装饰器定义一个数据类 TFLongformerMaskedLMOutput，包含掩码语言模型的输出
@dataclass
class TFLongformerMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
        Args:
            loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                Masked language modeling (MLM) loss.
            logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
                `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
                attention_window + 1)`, where `x` is the number of tokens with global attention mask.

                Local attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention heads. Those are the attention weights from every token in the sequence to every token with
                global attention (first `x` values) and to every token in the attention window (remaining `attention_window
                + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
                remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
                token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
                (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
                If the attention window contains a token with global attention, the attention weight at the corresponding
                index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
                attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
                accessed from `global_attentions`.
            global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
                is the number of tokens with global attention mask.

                Global attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention heads. Those are the attention weights from every token with global attention to every token
                in the sequence.
        """

        loss: tf.Tensor | None = None
    # 声明变量logits为TensorFlow张量类型，初始值为None
    logits: tf.Tensor = None
    # 声明变量hidden_states为元组类型，元素为TensorFlow张量，初始值为None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 声明变量attentions为元组类型，元素为TensorFlow张量，初始值为None
    attentions: Tuple[tf.Tensor] | None = None
    # 声明变量global_attentions为元组类型，元素为TensorFlow张量，初始值为None
    global_attentions: Tuple[tf.Tensor] | None = None
# 使用dataclass装饰器定义了一个TFLongformerQuestionAnsweringModelOutput类，用于表示问答Longformer模型的输出结果
class TFLongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering Longformer models.

    """

    # 损失值，类型为tf.Tensor或None
    loss: tf.Tensor | None = None
    # 起始位置的概率值，类型为tf.Tensor
    start_logits: tf.Tensor = None
    # 结束位置的概率值，类型为tf.Tensor
    end_logits: tf.Tensor = None
    # 隐藏状态的元组，类型为Tuple[tf.Tensor]或None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重的元组，类型为Tuple[tf.Tensor]或None
    attentions: Tuple[tf.Tensor] | None = None
    # 全局注意力的元组，类型为Tuple[tf.Tensor]或None
    global_attentions: Tuple[tf.Tensor] | None = None


# 使用dataclass装饰器定义了一个TFLongformerSequenceClassifierOutput类，用于表示序列分类模型的输出结果
class TFLongformerSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        # 分类（或回归，如果config.num_labels==1）的损失。
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        # 分类（或回归，如果config.num_labels==1）得分（SoftMax之前的结果）。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        # 返回参数为True时，会返回输出的隐藏状态。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        # 返回参数为True时，会返回注意力权重。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        # 返回参数为True时，会返回全局注意力权重。
        global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
            is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    # 损失值，数据类型为tf.Tensor或None
    loss: tf.Tensor | None = None
    # 定义变量logits，类型为tf.Tensor，初始值为None
    logits: tf.Tensor = None
    
    # 定义变量hidden_states，类型为Tuple[tf.Tensor]或None，初始值为None
    hidden_states: Tuple[tf.Tensor] | None = None
    
    # 定义变量attentions，类型为Tuple[tf.Tensor]或None，初始值为None
    attentions: Tuple[tf.Tensor] | None = None
    
    # 定义变量global_attentions，类型为Tuple[tf.Tensor]或None，初始值为None
    global_attentions: Tuple[tf.Tensor] | None = None
# 使用 dataclass 装饰器定义 TFLongformerMultipleChoiceModelOutput 类，该类用于存储多选模型的输出结果
class TFLongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.
    Args:
        loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失。
            说明：分类损失值，如果提供了标签，则返回。
        logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* 是输入张量的第二个维度。(参见上面的 *input_ids*).

            分类分数（SoftMax 之前）。
            说明：分类模型的预测分数。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组（一个用于嵌入的输出 + 一个用于每层的输出）。

            模型在每层输出的隐藏状态加上初始嵌入的输出。
            说明：当参数 `output_hidden_states=True` 被传递或者 `config.output_hidden_states=True` 时，返回隐藏状态。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)` 的 `tf.Tensor` 元组（每层一个），其中 `x` 是具有全局注意力掩码的令牌数量。

            自注意力头中的局部注意力权重，用于计算自注意力头中的加权平均值。这些是每个令牌到具有全局注意力（前 `x` 个值）和到注意力窗口中的每个令牌（剩余 `attention_window + 1` 个值）的注意力权重。
            说明：局部注意力权重，用于计算自注意力头中的加权平均值。
        global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, x)` 的 `tf.Tensor` 元组（每层一个），其中 `x` 是具有全局注意力掩码的令牌数量。

            全局注意力头中的全局注意力权重，用于计算自注意力头中的加权平均值。这些是具有全局注意力的每个令牌到序列中的每个令牌的注意力权重。
            说明：全局注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    # logits 存储模型输出的预测结果，是一个 TensorFlow 张量，初始值为 None
    logits: tf.Tensor = None
    # hidden_states 存储模型的隐藏状态，是一个元组，其中的元素为 TensorFlow 张量，初始值为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # attentions 存储模型的注意力权重，是一个元组，其中的元素为 TensorFlow 张量，初始值为 None
    attentions: Tuple[tf.Tensor] | None = None
    # global_attentions 存储模型的全局注意力权重，是一个元组，其中的元素为 TensorFlow 张量，初始值为 None
    global_attentions: Tuple[tf.Tensor] | None = None
# 导入 dataclass 模块，用于创建数据类
from dataclasses import dataclass
# 导入 ModelOutput 类，作为 TFLongformerTokenClassifierOutput 类的基类
from transformers.modeling_outputs import ModelOutput

# 定义 TFLongformerTokenClassifierOutput 类，用于表示标记分类模型的输出结果
@dataclass
class TFLongformerTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            # 分类损失，当提供了`labels`时返回
            Classification loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            # 分类分数（SoftMax之前）
            Classification scores (before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 模型的隐藏状态，每一层的输出以及初始嵌入输出的元组
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 本地注意力权重，用于计算自注意力头中的加权平均值
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 全局注意力权重，用于计算自注意力头中的加权平均值
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
            is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    # attentions 和 global_attentions 分别是 Tuple[tf.Tensor] 或者 None 类型的变量，用来存储注意力的信息
    attentions: Tuple[tf.Tensor] | None = None
    global_attentions: Tuple[tf.Tensor] | None = None
def _compute_global_attention_mask(input_ids_shape, sep_token_indices, before_sep_token=True):
    """
    计算全局注意力掩码，如果before_sep_token为True，则在sep_token_id之前放置注意力，否则放在sep_token_id之后。
    """
    # 断言sep_token_indices的第二个维度为2，即`input_ids`应该有两个维度
    assert shape_list(sep_token_indices)[1] == 2, "`input_ids` should have two dimensions"
    # 将sep_token_indices重塑为(input_ids_shape[0], 3, 2)的形状，然后取出第一个位置的索引作为question_end_index
    question_end_index = tf.reshape(sep_token_indices, (input_ids_shape[0], 3, 2))[:, 0, 1][:, None]
    # 为全局注意力创建布尔注意力掩码
    attention_mask = tf.expand_dims(tf.range(input_ids_shape[1], dtype=tf.int64), axis=0)
    attention_mask = tf.tile(attention_mask, (input_ids_shape[0], 1))
    if before_sep_token is True:
        # 使用tile函数将question_end_index扩展成与attention_mask相同的形状，然后使用cast函数转换成布尔类型
        question_end_index = tf.tile(question_end_index, (1, input_ids_shape[1]))
        attention_mask = tf.cast(attention_mask < question_end_index, dtype=question_end_index.dtype)
    else:
        # 最后一个标记是分隔标记，不应计数，中间是两个分隔标记
        question_end_index = tf.tile(question_end_index + 1, (1, input_ids_shape[1]))
        attention_mask = tf.cast(
            attention_mask > question_end_index,
            dtype=question_end_index.dtype,
        ) * tf.cast(attention_mask < input_ids_shape[-1], dtype=question_end_index.dtype)

    return attention_mask


# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead复制过来，将Roberta->Longformer
class TFLongformerLMHead(tf.keras.layers.Layer):
    """用于掩码语言建模的Longformer头部。"""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 初始化偏置项
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        # 构建dense层和layer_norm层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    def get_output_embeddings(self):
        return self.decoder
    # 设置输出词嵌入
    def set_output_embeddings(self, value):
        # 将解码器的权重设置为给定值
        self.decoder.weight = value
        # 更新解码器的词汇量大小
        self.decoder.vocab_size = shape_list(value)[0]

    # 获取偏置项
    def get_bias(self):
        # 返回包含偏置项的字典
        return {"bias": self.bias}

    # 设置偏置项
    def set_bias(self, value):
        # 将偏置项设置为给定值的偏置项
        self.bias = value["bias"]
        # 更新配置的词汇量大小
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 模型的前向传播
    def call(self, hidden_states):
        # 使用全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 应用层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 投影回词汇大小并加上偏置项
        # 获取序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 重新塑造张量形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 矩阵相乘，将隐藏状态映射到词嵌入空间
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        # 重新塑造张量形状
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置项
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回隐藏状态
        return hidden_states
    class TFLongformerEmbeddings(tf.keras.layers.Layer):
        """
        Same as BertEmbeddings with a tiny tweak for positional embeddings indexing and some extra casting.
        """

        def __init__(self, config, **kwargs):
            super().__init__(**kwargs)

            # 定义默认填充索引
            self.padding_idx = 1
            self.config = config
            self.hidden_size = config.hidden_size
            self.max_position_embeddings = config.max_position_embeddings
            self.initializer_range = config.initializer_range
            # 创建 LayerNormalization 层
            self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
            # 创建 Dropout 层
            self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

        def build(self, input_shape=None):
            with tf.name_scope("word_embeddings"):
                # 添加词嵌入矩阵权重
                self.weight = self.add_weight(
                    name="weight",
                    shape=[self.config.vocab_size, self.hidden_size],
                    # 使用指定初始化器初始化
                    initializer=get_initializer(self.initializer_range),
                )

            with tf.name_scope("token_type_embeddings"):
                # 添加 token type embeddings 权重
                self.token_type_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.config.type_vocab_size, self.hidden_size],
                    # 使用指定初始化器初始化
                    initializer=get_initializer(self.initializer_range),
                )

            with tf.name_scope("position_embeddings"):
                # 添加位置嵌入矩阵权重
                self.position_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.max_position_embeddings, self.hidden_size],
                    # 使用指定初始化器初始化
                    initializer=get_initializer(self.initializer_range),
                )

            # 如果已经构建完毕，则返回
            if self.built:
                return
            self.built = True
            if getattr(self, "LayerNorm", None) is not None:
                with tf.name_scope(self.LayerNorm.name):
                    # 构建 LayerNormalization 层
                    self.LayerNorm.build([None, None, self.config.hidden_size])

        def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
            """
            Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
            symbols are ignored. This is modified from fairseq's `utils.make_positions`.

            Args:
                input_ids: tf.Tensor
            Returns: tf.Tensor
            """
            # 创建掩码以标记非填充符号
            mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
            # 计算位置索引
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
        对输入张量应用基于嵌入的操作。

        返回值:
            final_embeddings (`tf.Tensor`): 输出嵌入张量。
        """
        # 确保输入张量不为空
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # 检查输入张量中的嵌入是否在范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 获取与输入张量对应的权重
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            # 如果标记类型 ID 为空，则将其设置为零
            token_type_ids = tf.cast(tf.fill(dims=input_shape, value=0), tf.int64)

        if position_ids is None:
            if input_ids is not None:
                # 从输入标记 ID 创建位置 ID。任何填充标记仍然保持填充状态。
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1, dtype=tf.int64),
                    axis=0,
                )

        # 获取位置嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 获取标记类型嵌入
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 计算最终嵌入张量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 应用 LayerNorm
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制并修改为LongformerIntermediate
class TFLongformerIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于中间状态的转换，设置输出单元数为config.intermediate_size，权重初始化采用config.initializer_range指定的初始化器，层名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config.hidden_act是字符串类型，则将其转换为相应的激活函数，否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 存储配置信息
        self.config = config

    # 定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入hidden_states经过全连接层转换
        hidden_states = self.dense(inputs=hidden_states)
        # 将转换后的hidden_states经过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层已经存在，构建dense层，设置输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertOutput复制并修改为LongformerOutput
class TFLongformerOutput(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于输出层的转换，设置输出单元数为config.hidden_size，权重初始化采用config.initializer_range指定的初始化器，层名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建LayerNormalization层，用于归一化输出，设置epsilon为config.layer_norm_eps，层名为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建Dropout层，用于随机失活，设置失活率为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 存储配置信息
        self.config = config

    # 定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入hidden_states经过全连接层转换
        hidden_states = self.dense(inputs=hidden_states)
        # 使用Dropout对转换后的hidden_states进行随机失活
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将随机失活后的hidden_states与输入tensor相加，并经过LayerNormalization进行归一化处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层已经存在，构建dense层，设置输入形状为[None, None, self.config.intermediate_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果LayerNorm层已经存在，构建LayerNorm层，设置输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制并修改为LongformerPooler
class TFLongformerPooler(tf.keras.layers.Layer):
    # 初始化方法，接受 LongformerConfig 类型的配置参数
    def __init__(self, config: LongformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，配置包括隐藏单元数、初始化方法和激活函数
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 保存配置参数
        self.config = config

    # 调用方法，接受一个输入张量，返回一个输出张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 池化模型的隐藏状态，只取对应于第一个令牌的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个令牌的隐藏状态经过全连接层处理得到输出
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    # 构建模型，指定输入形状
    def build(self, input_shape=None):
        # 如果已经构建好模型则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则为全连接层指定输入形状
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
# 从transformers.models.bert.modeling_tf_bert.TFBertSelfOutput复制代码，并将Bert更改为Longformer
class TFLongformerSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于接收隐藏状态并输出相同大小的输出
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，用于对输入进行归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，用于在训练过程中随机丢弃部分神经元
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 定义Layer的call方法，用于执行正向传播
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 先通过全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 然后进行Dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 最后通过LayerNormalization层进行归一化并与输入张量相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 在调用build()方法时构建层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在密集层，构建密集层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在LayerNormalization层，构建LayerNormalization层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFLongformerSelfAttention(tf.keras.layers.Layer):
    # 初始化函数，接受配置、层编号和其他关键字参数
    def __init__(self, config, layer_id, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将配置参数保存在实例中
        self.config = config

        # 检查隐藏大小是否可以被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 计算注意力头数、头维度和嵌入维度
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        
        # 创建用于查询、键和值的全连接层
        self.query = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # 为具有全局注意力的标记创建单独的投影层
        self.query_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_global",
        )
        self.key_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_global",
        )
        self.value_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_global",
        )
        
        # 创建用于掩码的丢弃层
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        
        # 保存层编号
        self.layer_id = layer_id
        
        # 检查并设置注意力窗口大小
        attention_window = config.attention_window[self.layer_id]

        # 确保注意力窗口大小为偶数
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        
        # 确保注意力窗口大小为正数
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        # 计算单向注意力窗口大小
        self.one_sided_attn_window_size = attention_window // 2
```  
    # 构建模型层
    def build(self, input_shape=None):
        # 如果模型尚未构建
        if not self.built:
            # 使用 query_global 层的 name_scope 进行构建
            with tf.name_scope("query_global"):
                self.query_global.build((self.config.hidden_size,))
            # 使用 key_global 层的 name_scope 进行构建
            with tf.name_scope("key_global"):
                self.key_global.build((self.config.hidden_size,))
            # 使用 value_global 层的 name_scope 进行构建
            with tf.name_scope("value_global"):
                self.value_global.build((self.config.hidden_size,))
    
        # 如果模型已经构建
        if self.built:
            # 直接返回
            return
        # 标记模型为已构建
        self.built = True
        # 如果存在 query 层
        if getattr(self, "query", None) is not None:
            # 使用 query 层的 name_scope 进行构建
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在 key 层
        if getattr(self, "key", None) is not None:
            # 使用 key 层的 name_scope 进行构建
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在 value 层
        if getattr(self, "value", None) is not None:
            # 使用 value 层的 name_scope 进行构建
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在 query_global 层
        if getattr(self, "query_global", None) is not None:
            # 使用 query_global 层的 name_scope 进行构建
            with tf.name_scope(self.query_global.name):
                self.query_global.build([None, None, self.config.hidden_size])
        # 如果存在 key_global 层
        if getattr(self, "key_global", None) is not None:
            # 使用 key_global 层的 name_scope 进行构建
            with tf.name_scope(self.key_global.name):
                self.key_global.build([None, None, self.config.hidden_size])
        # 如果存在 value_global 层
        if getattr(self, "value_global", None) is not None:
            # 使用 value_global 层的 name_scope 进行构建
            with tf.name_scope(self.value_global.name):
                self.value_global.build([None, None, self.config.hidden_size])
    
    # 模型的前向传播
    def call(
        self,
        inputs,
        training=False,
    ):
        pass
    
    # 遮蔽无效位置
    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # 创建正确的上三角布尔掩码
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )
    
        # 填充到完整矩阵
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )
    
        # 创建下三角掩码
        mask_2d = tf.pad(mask_2d_upper, padding)
    
        # 与上三角掩码组合
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])
    
        # 广播到完整矩阵
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))
    
        # 用于掩蔽的无穷大张量
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)
    
        # 应用掩码
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)
    
        return input_tensor
    # 对于给定的注意力权重和值张量，使用滑动窗口计算注意力的上下文
    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        # 获取值张量的形状信息
        batch_size, seq_len, num_heads, head_dim = shape_list(value)
    
        # 确保序列长度是窗口重叠长度的整数倍
        tf.debugging.assert_equal(
            seq_len % (window_overlap * 2), 0, message="Seq_len has to be multiple of 2 * window_overlap"
        )
        # 确保注意力权重和值张量前3个维度大小一致
        tf.debugging.assert_equal(
            shape_list(attn_probs)[:3],
            shape_list(value)[:3],
            message="value and attn_probs must have same dims (except head_dim)",
        )
        # 确保注意力权重张量的最后一个维度大小是 2 * window_overlap + 1
        tf.debugging.assert_equal(
            shape_list(attn_probs)[3],
            2 * window_overlap + 1,
            message="attn_probs last dim has to be 2 * window_overlap + 1",
        )
    
        # 计算块的数量
        chunks_count = seq_len // window_overlap - 1
    
        # 将批量大小和头数量维度合并为一个维度，然后按窗口重叠大小将序列长度分块
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )
    
        # 将批量大小和头数量维度合并为一个维度
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )
    
        # 用-1填充序列长度，分别在开头和结尾添加 window_overlap 个填充元素
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)
    
        # 按 3 * window_overlap 大小的窗口和 window_overlap 的窗口重叠，将填充后的值张量分块
        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count
        chunked_value = tf.signal.frame(
            tf.reshape(padded_value, (batch_size * num_heads, -1)),
            frame_size,
            frame_hop_size,
        )
        chunked_value = tf.reshape(
            chunked_value,
            (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim),
        )
    
        # 确保分块后的值张量的形状正确
        tf.debugging.assert_equal(
            shape_list(chunked_value),
            [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
            message="Chunked value has the wrong shape",
        )
    
        # 填充并对角化分块后的注意力权重张量
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        # 计算上下文张量
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        # 将上下文张量的维度顺序调整回原来的形状
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )
    
        return context
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """填充行，然后翻转行和列"""
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # 填充值不重要，因为它将被覆盖
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        将每一行向右移动1步，将列转换为对角线。

        示例:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (填充和对角化) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # 总头数 x 块数 x 窗口重叠 x (隐藏维度+窗口重叠+1)。填充值不重要，因为它将被覆盖
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # 总头数 x 块数 x 窗口重叠L+窗口重叠窗口重叠+窗口重叠
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # 总头数 x 块数 x 窗口重叠L+窗口重叠窗口重叠
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # 总头数 x 块数，窗口重叠 x 隐藏维度+窗口重叠
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """将隐藏状态转换为重叠的块。块大小=2w，重叠大小=w"""
        # 获取隐藏状态的形状参数
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        # 计算输出块的数量
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # 定义帧大小和帧步幅（类似卷积）
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # 创建带有重叠的块
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        # 断言确保块的形状是正确的
        tf.debugging.assert_equal(
            shape_list(chunked_hidden_states),
            [batch_size, num_output_chunks, frame_size],
            message=(
                "确保块化应用正确。'块化的隐藏状态应该有输出维度"
                f" {[batch_size, frame_size, num_output_chunks]}, 但得到了 {shape_list(chunked_hidden_states)}."
            ),
        )

        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """计算前向传递过程中全局注意力索引所需的全局注意力索引"""
        # 辅助变量
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)

        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)

        # 全局注意力索引的索引
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)

        # 辅助变量
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )

        # 全局注意力索引中非填充值的位置
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)

        # 全局注意力索引中填充值的位置
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        attn_scores,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        ):
        # 获取批量大小
        batch_size = shape_list(key_vectors)[0]

        # 选择全局键向量
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

        # 创建仅包含全局键向量的张量
        key_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_key_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # 计算来自全局键的注意力概率
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # 转置注意力概率
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

        # 散播掩码
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # 转置注意力概率
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # 连接到注意力分数
        # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        return attn_scores

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    # 计算传入注意力概率的形状列表的批次大小
    def reshape_and_transpose(self, vector, batch_size):
        # 将向量重塑为(batch_size, -1, self.num_heads, self.head_dim)的形状
        reshaped_vector = tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim))
        # 转置维度，将注意力头维度移到第二个位置
        transposed_vector = tf.transpose(reshaped_vector, (0, 2, 1, 3))
        # 再次重塑为(batch_size * self.num_heads, -1, self.head_dim)的形状
        return tf.reshape(transposed_vector, (batch_size * self.num_heads, -1, self.head_dim))
class TFLongformerAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)

        # 初始化 self_attention 和 dense_output 层
        self.self_attention = TFLongformerSelfAttention(config, layer_id, name="self")
        self.dense_output = TFLongformerSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 抛出未实现错误
        raise NotImplementedError

    def call(self, inputs, training=False):
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs

        # 调用 self_attention 层生成 self_outputs
        self_outputs = self.self_attention(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        # 将 self_outputs[0] 经过 dense_output 层得到 attention_output
        attention_output = self.dense_output(self_outputs[0], hidden_states, training=training)
        # 组装输出，包括 attention_output 和可能还有其它信息
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建，直接返回
        if self.built:
            return
        self.built = True
        # 构建 self_attention 层（如果存在）
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 构建 dense_output 层（如果存在）
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


class TFLongformerLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)

        # 初始化 attention、intermediate 和 longformer_output 层
        self.attention = TFLongformerAttention(config, layer_id, name="attention")
        self.intermediate = TFLongformerIntermediate(config, name="intermediate")
        self.longformer_output = TFLongformerOutput(config, name="output")

    def call(self, inputs, training=False):
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs

        # 调用 attention 层生成 attention_outputs
        attention_outputs = self.attention(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        # 从 attention_outputs 中获取 attention_output
        attention_output = attention_outputs[0]
        # 将 attention_output 经过 intermediate 层得到 intermediate_output
        intermediate_output = self.intermediate(attention_output)
        # 将 intermediate_output 和 attention_output 经过 longformer_output 层得到 layer_output
        layer_output = self.longformer_output(intermediate_output, attention_output, training=training)
        # 组装输出，包括 layer_output 和可能还有其它信息
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过了，就直接返回
        if self.built:
            return
        # 将 built 属性设置为 True 表示模型已经构建完毕
        self.built = True
        # 如果存在注意力模块，那么构建注意力模块
        if getattr(self, "attention", None) is not None:
            # 为注意力模块创建一个名字作用域
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层模块，那么构建中间层模块  
        if getattr(self, "intermediate", None) is not None:
            # 为中间层模块创建一个名字作用域
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 Longformer 输出模块，那么构建 Longformer 输出模块
        if getattr(self, "longformer_output", None) is not None:
            # 为 Longformer 输出模块创建一个名字作用域
            with tf.name_scope(self.longformer_output.name):
                self.longformer_output.build(None)
# 定义一个名为 TFLongformerEncoder 的类，继承自 tf.keras.layers.Layer 类
class TFLongformerEncoder(tf.keras.layers.Layer):
    # 初始化函数，接受 config 参数和额外的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置是否输出隐藏状态和注意力分布的标志
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        # 创建一系列 TFLongformerLayer 层对象并存储在 self.layer 列表中
        self.layer = [TFLongformerLayer(config, i, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义 call 方法，接受一系列输入参数
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 如果不需要输出隐藏状态，则将 all_hidden_states 设置为 None；否则设置为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则将 all_attentions 和 all_global_attentions 设置为 None；否则设置为空元组
        all_attentions = all_global_attentions = () if output_attentions else None

        # 遍历 Transformer 层
        for idx, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 如果存在填充，则截断隐藏状态
                hidden_states_to_add = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
                # 将截断后的隐藏状态添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states_to_add,)

            # 将输入传递给当前 Transformer 层
            layer_outputs = layer_module(
                [
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    is_index_masked,
                    is_index_global_attn,
                    is_global_attn,
                ],
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重
            if output_attentions:
                # 将当前层的注意力权重进行转置，并添加到 all_attentions 中
                all_attentions = all_attentions + (tf.transpose(layer_outputs[1], (0, 2, 1, 3)),)

                # 将当前层的全局注意力权重进行转置，并添加到 all_global_attentions 中
                all_global_attentions = all_global_attentions + (tf.transpose(layer_outputs[2], (0, 1, 3, 2)),)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            hidden_states_to_add = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
            all_hidden_states = all_hidden_states + (hidden_states_to_add,)

        # 恢复填充之前的隐藏状态长度
        hidden_states = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
        # 如果需要输出注意力权重，对 all_attentions 进行填充恢复
        if output_attentions:
            all_attentions = (
                tuple([state[:, :, :-padding_len, :] for state in all_attentions])
                if padding_len > 0
                else all_attentions
            )

        # 如果不需要返回字典形式的输出，则按需返回各个输出的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
            )

        # 返回字典形式的输出
        return TFLongformerBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已构建，则直接返回
        if self.built:
            return
        self.built = True
        # 遍历每一层并构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    # 构建当前层
                    layer.build(None)
# 使用 keras_serializable 装饰器将类 TFLongformerMainLayer 标记为可序列化的
@keras_serializable
class TFLongformerMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 LongformerConfig
    config_class = LongformerConfig

    # 初始化函数
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 判断 config.attention_window 的类型，并进行相应的检查和处理
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            # 将配置中的 attention_window 转换为列表形式，每层一个值
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        # 初始化类的属性
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.pad_token_id = config.pad_token_id
        self.attention_window = config.attention_window
        self.embeddings = TFLongformerEmbeddings(config, name="embeddings")
        self.encoder = TFLongformerEncoder(config, name="encoder")
        # 如果指定添加池化层，则创建 TFLongformerPooler 对象，否则设为 None
        self.pooler = TFLongformerPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 实现头部修剪的方法，需要在子类中实现
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 调用方法，解包输入参数，处理输入和调用编码器
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    # 对输入进行填充以匹配窗口大小
    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds,
        pad_token_id,
    ):  
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        # 定义一个辅助函数，用于填充标记和掩码以配合Longformer自注意力的实现

        # padding
        attention_window = (
            self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        )
        # 根据self.attention_window是整数还是最大值来确定attention_window的大小

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        # 断言attention_window应该是一个偶数值

        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        # 如果input_ids不是None，则获取input_ids的形状列表，否则获取inputs_embeds的形状列表
        batch_size, seq_len = input_shape[:2]
        # 获取批处理大小和序列长度

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        # 计算需要填充的长度，使序列长度是attention_window的整数倍

        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])
        # 构建填充的张量

        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)
            # 如果input_ids不是None，则使用pad_token_id对input_ids进行填充

        if position_ids is not None:
            # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
            position_ids = tf.pad(position_ids, paddings, constant_values=pad_token_id)
            # 如果position_ids不是None，则使用pad_token_id对position_ids进行填充

        if inputs_embeds is not None:
            if padding_len > 0:
                input_ids_padding = tf.cast(tf.fill((batch_size, padding_len), self.pad_token_id), tf.int64)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)
        # 如果inputs_embeds不是None且有需要填充，则对inputs_embeds进行填充操作

        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)  # no attention on the padding tokens
        # 对注意力掩码进行填充，填充值为False，表示填充的位置不需要注意力
        token_type_ids = tf.pad(token_type_ids, paddings, constant_values=0)  # pad with token_type_id = 0
        # 对token_type_ids进行填充，填充值为0

        return (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
        )
        # 返回填充之后的结果

    @staticmethod
    def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
            # Longformer自注意力期望注意力掩码值为0（无注意力），1（局部注意力），2（全局注意力）
            # (global_attention_mask + 1) => 1表示局部注意力，2表示全局注意力
            # 最终的 attention_mask => 0表示无注意力，1表示局部注意力，2表示全局注意力
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
            # 如果没有提供attention_mask，则直接使用global_attention_mask作为attention_mask

        return attention_mask
        # 返回合并后的attention_mask
    # 构建函数，用于构建模型的输入形状
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置已经构建过的标志
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 使用命名空间为嵌入层设置名称，并构建嵌入层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码层，则构建编码层
        if getattr(self, "encoder", None) is not None:
            # 使用命名空间为编码层设置名称，并构建编码层
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化层，则构建池化层
        if getattr(self, "pooler", None) is not None:
            # 使用命名空间为池化层设置名称，并构建池化层
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
class TFLongformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LongformerConfig  # 配置类，包含模型的所有参数
    base_model_prefix = "longformer"  # 基本模型前缀为“longformer”

    @property
    def input_signature(self):
        sig = super().input_signature  # 调用父类的 input_signature 方法
        sig["global_attention_mask"] = tf.TensorSpec((None, None), tf.int32, name="global_attention_mask")  # 添加全局注意力掩码参数
        return sig


LONGFORMER_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.
    
    ...  # 长模型的开始文档字符串

LONGFORMER_INPUTS_DOCSTRING = r"""
"""  # 长模型的输入文档字符串
# 使用装饰器添加说明文档，描述输出原始隐藏状态而没有特定头部的Longformer模型
# 模型类继承自TFLongformerPreTrainedModel
class TFLongformerModel(TFLongformerPreTrainedModel):
    """
    这个类从`TFRobertaModel`中复制了代码，并用Longformer自注意力覆盖了标准的自注意力，
    以提供处理长序列的能力，遵循Iz Beltagy、Matthew E. Peters和Arman Cohan在["Longformer: the Long-Document Transformer"](https://arxiv.org/abs/2004.05150)中描述的自注意力方法。
    Longformer的自注意力结合了局部(滑动窗口)和全局注意力，以扩展到长文档，而不会增加内存和计算的O(n^2)。

    这里实现的TFLongformerSelfAttention自注意力模块支持局部和全局注意力的组合，但缺少自回归注意力和扩张注意力的支持。自回归和扩张注意力对自回归语言建模比对下游任务微调更相关。未来的版本将添加自回归注意力的支持，但扩张注意力的支持需要自定义CUDA内核才能实现内存和计算的效率。
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化使用TFLongformerMainLayer实现的longformer模块
        self.longformer = TFLongformerMainLayer(config, name="longformer")

    # 解包输入参数，添加模型前向方法的说明文档
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFLongformerBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 使用longformer模块处理输入参数，返回模型输出
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不做任何操作
        if self.built:
            return
        # 将标志位设置为 True，表示模型已经构建
        self.built = True
        # 如果模型中存在 longformer 属性
        if getattr(self, "longformer", None) is not None:
            # 在 TensorFlow 中创建一个命名作用域，命名为 self.longformer.name
            with tf.name_scope(self.longformer.name):
                # 调用 longformer 对象的 build 方法，传入 None 作为输入形状
                self.longformer.build(None)
# 使用`add_start_docstrings`为模型添加长文档说明，说明这是一个在其顶部有语言模型头的Longformer模型。
class TFLongformerForMaskedLM(TFLongformerPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在加载TF模型时，指定要忽略的意外/缺失层的名称列表，包括'pooler'
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 初始化Longformer主层，不添加池化层，命名为'longformer'
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        # 初始化Longformer语言模型头，使用Longformer的嵌入层，命名为'lm_head'
        self.lm_head = TFLongformerLMHead(config, self.longformer.embeddings, name="lm_head")

    # 返回语言模型头
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称（已弃用），会发出未来警告
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回模型名称与语言模型头名称的组合
        return self.name + "/" + self.lm_head.name

    # 使用装饰器为模型的前向传播方法添加文档字符串，包括输入参数的说明和示例代码
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="allenai/longformer-base-4096",
        output_type=TFLongformerMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.44,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFLongformerMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 调用 Longformer 模型，传入各种输入参数
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从输出中取得序列输出
        sequence_output = outputs[0]
        # 在 LM 头部模型上进行预测
        prediction_scores = self.lm_head(sequence_output, training=training)
        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不是返回字典
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]

            # 如果存在损失，则返回损失和输出
            return ((loss,) + output) if loss is not None else output

        # 返回 TFLongformerMaskedLMOutput 类型对象
        return TFLongformerMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 检查模型是否已经构建
        if self.built:
            return
        self.built = True
        # 如果存在 Longformer 模型，则构建 Longformer
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 如果存在 LM 头部模型，则构建 LM 头部
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 使用 add_start_docstrings 函数向 TFLongformerForQuestionAnswering 类添加文档字符串
# 文档字符串描述了 Longformer 模型及其用途
# 此处使用 LONGFORMER_START_DOCSTRING 变量作为默认文档字符串的一部分
class TFLongformerForQuestionAnswering(TFLongformerPreTrainedModel, TFQuestionAnsweringLoss):
    # 在加载 PT 模型时，忽略掉指定的层
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # 初始化函数，接受配置参数 config
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 将标签数设置为配置参数中的标签数
        self.num_labels = config.num_labels
        # 初始化 Longformer 主层
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        # 初始化用于输出答案的 Dense 层
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )
        # 保存配置参数
        self.config = config

    # 将输入进行展开，并向模型前向函数添加文档字符串
    # 文档字符串描述了 Longformer 模型的输入说明、代码示例及预期输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 Longformer，则构建 Longformer 层
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 如果存在 qa_outputs 层，则构建该层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])


# TFLongformerClassificationHead 类的简单描述
class TFLongformerClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    # 初始化方法，接受一个配置对象和额外的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层，指定隐藏层大小、权重初始化器和激活函数
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 创建一个丢弃层，指定丢弃比率
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，指定输出标签数量和权重初始化器
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        # 保存配置对象
        self.config = config

    # 调用方法，接受隐藏状态和训练标志作为参数
    def call(self, hidden_states, training=False):
        # 取隐藏状态的第一个 token，通常为 <s> token（等同于 [CLS]）
        hidden_states = hidden_states[:, 0, :]
        # 对隐藏状态进行丢弃处理，根据训练标志决定是否启用
        hidden_states = self.dropout(hidden_states, training=training)
        # 将丢弃后的隐藏状态输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层输出进行丢弃处理，根据训练标志决定是否启用
        hidden_states = self.dropout(hidden_states, training=training)
        # 将全连接层的输出输入到输出投影层中
        output = self.out_proj(hidden_states)
        # 返回输出结果
        return output

    # 构建方法，接受输入形状作为参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果存在全连接层，构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，指定输入形状
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在输出投影层，构建输出投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建输出投影层，指定输入形状
                self.out_proj.build([None, None, self.config.hidden_size])
# 定义一个带有顶部序列分类/回归头的Longformer模型转换器（在聚合输出之上的线性层），用于GLUE任务
# 使用Longformer的文档开始注释源
class TFLongformerForSequenceClassification(TFLongformerPreTrainedModel, TFSequenceClassificationLoss):
    # 带有'.'的名称代表了当从PT模型加载TF模型时授权的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # 初始化方法，接受config参数和*inputs和**kwargs参数
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置标签数量等于config中的标签数量
        self.num_labels = config.num_labels

        # 创建一个Longformer主层对象，不添加池化层，名称为"longformer"
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        # 创建一个TFLongformerClassificationHead对象，名称为"classifier"
        self.classifier = TFLongformerClassificationHead(config, name="classifier")

    # 调用方法，使用input_ids等参数进行模型前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 输入参数包括输入ID，注意力掩码，头部掩码，令牌类型ID，位置ID，全局注意力掩码，输入嵌入，输出注意力，隐藏状态等
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 定义函数返回类型，可能是 TFLongformerSequenceClassifierOutput 或包含 Tensor 的元组
    ) -> Union[TFLongformerSequenceClassifierOutput, Tuple[tf.Tensor]]:
    
        # 如果 input_ids 不是 None 且不是 tf.Tensor 类型
        if input_ids is not None and not isinstance(input_ids, tf.Tensor):
            # 将 input_ids 转换为 Tensor 类型，并指定数据类型为 int64
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
        # 如果 input_ids 是 Tensor 类型但不是 int64 数据类型
        elif input_ids is not None:
            # 将 input_ids 转换为 int64 类型
            input_ids = tf.cast(input_ids, tf.int64)
    
        # 如果 attention_mask 不是 None 且不是 tf.Tensor 类型
        if attention_mask is not None and not isinstance(attention_mask, tf.Tensor):
            # 将 attention_mask 转换为 Tensor 类型，并指定数据类型为 int64
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
        # 如果 attention_mask 是 Tensor 类型但不是 int64 数据类型
        elif attention_mask is not None:
            # 将 attention_mask 转换为 int64 类型
            attention_mask = tf.cast(attention_mask, tf.int64)
    
        # 如果 global_attention_mask 不是 None 且不是 tf.Tensor 类型
        if global_attention_mask is not None and not isinstance(global_attention_mask, tf.Tensor):
            # 将 global_attention_mask 转换为 Tensor 类型，并指定数据类型为 int64
            global_attention_mask = tf.convert_to_tensor(global_attention_mask, dtype=tf.int64)
        # 如果 global_attention_mask 是 Tensor 类型但不是 int64 数据类型
        elif global_attention_mask is not None:
            # 将 global_attention_mask 转换为 int64 类型
            global_attention_mask = tf.cast(global_attention_mask, tf.int64)
    
        # 如果 global_attention_mask 为空，且 input_ids 不为空
        if global_attention_mask is None and input_ids is not None:
            # 打印警告信息，表示将初始化 CLS 令牌的全局注意力
            logger.warning_once("Initializing global attention on CLS token...")
            # 初始化全局注意力掩码，设置为与 input_ids 相同大小的零张量
            global_attention_mask = tf.zeros_like(input_ids)
            # 创建一个一维张量，大小为 input_ids 的第一个维度，填充 1
            updates = tf.ones(shape_list(input_ids)[0], dtype=tf.int64)
            # 创建索引，表示要更新的位置
            indices = tf.pad(
                tensor=tf.expand_dims(tf.range(shape_list(input_ids)[0], dtype=tf.int64), axis=1),
                paddings=[[0, 0], [0, 1]],
                constant_values=0,
            )
            # 使用索引和 updates 更新 global_attention_mask 张量
            global_attention_mask = tf.tensor_scatter_nd_update(
                global_attention_mask,
                indices,
                updates,
            )
    
        # 调用 Longformer 模型，传入相关参数，包括输入 ID、注意力掩码、头掩码等
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 提取模型的序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类，生成 logits
        logits = self.classifier(sequence_output)
    
        # 如果 labels 为 None，则 loss 也为 None，否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
        # 如果不需要返回字典
        if not return_dict:
            # 组合 logits 和额外的输出
            output = (logits,) + outputs[2:]
            # 返回损失和输出的组合，如果损失为空，则仅返回输出
            return ((loss,) + output) if loss is not None else output
    
        # 如果需要返回字典，则返回 TFLongformerSequenceClassifierOutput 对象
        return TFLongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经被构建过了，就直接返回
        if self.built:
            return
        # 将 built 标志设置为 True，表示模型已经被构建
        self.built = True
        # 如果模型中有 longformer 部分，在 longformer 的命名作用域下构建 longformer
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 如果模型中有 classifier 部分，在 classifier 的命名作用域下构建 classifier
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForMultipleChoice(TFLongformerPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建一个 Longformer 主层的实例，命名为"longformer"
        self.longformer = TFLongformerMainLayer(config, name="longformer")
        # 创建一个dropout层，使用config中的hidden_dropout_prob参数
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，输出维度为1，使用config中的initializer_range参数初始化权重，命名为"classifier"
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 将config属性指向传入的config对象
        self.config = config

    @property
    def input_signature(self):
        # 定义输入的TensorSpec签名，包括input_ids, attention_mask, global_attention_mask三个参数
        return {
            "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),
            "global_attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="global_attention_mask"),
        }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        ) -> Union[TFLongformerMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        if input_ids is not None:
            # 获取输入中第二维的大小作为选择数量
            num_choices = shape_list(input_ids)[1]
            # 获取输入中第三维的大小作为序列长度
            seq_length = shape_list(input_ids)[2]
        else:
            # 获取输入嵌入的第二维的大小作为选择数量
            num_choices = shape_list(inputs_embeds)[1]
            # 获取输入嵌入的第三维的大小作为序列长度
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入转换为二维张量
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_global_attention_mask = (
            tf.reshape(global_attention_mask, (-1, shape_list(global_attention_mask)[-1]))
            if global_attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 调用 longformer 模型进行计算
        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            global_attention_mask=flat_global_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类
        logits = self.classifier(pooled_output)
        # 重塑分类结果
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果存在标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不要求返回字典，则返回损失和输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，根据输出格式返回相应的结果
        return TFLongformerMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
    # 构建神经网络模型，根据输入形状（如果有的话）
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True  # 标记模型已构建
        # 如果存在长形式模型，则构建长形式模型
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):  # 使用长形式模型的名称定义作用域
                self.longformer.build(None)  # 构建长形式模型
        # 如果存在分类器模型，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):  # 使用分类器模型的名称定义作用域
                self.classifier.build([None, None, self.config.hidden_size])  # 构建分类器模型
# 引入必要的库，包括对应的函数和类
@add_start_docstrings(
    """
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
# 定义 TFLongformerForTokenClassification 类，继承自 TFLongformerPreTrainedModel 和 TFTokenClassificationLoss
class TFLongformerForTokenClassification(TFLongformerPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化函数，接受配置和输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 Longformer 主要层对象
        self.longformer = TFLongformerMainLayer(config=config, add_pooling_layer=False, name="longformer")
        # 创建 Dropout 层对象
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 创建分类器层对象
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 设置配置
        self.config = config

    # 调用函数，定义模型的前向传播过程
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.array, tf.Tensor]] = None,
        training: Optional[bool] = False,
    # 计算 Longformer 模型的token分类输出
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ) -> Union[TFLongformerTokenClassifierOutput, Tuple[tf.Tensor]]:
        # 如果给定了标签, 则返回计算标签分类损失的输出, 否则只返回模型预测的logits
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
    
        # 通过Longformer模型计算输出
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 对序列输出进行dropout
        sequence_output = self.dropout(sequence_output)
        # 通过分类器计算logits
        logits = self.classifier(sequence_output)
        # 如果给定了标签, 则计算分类损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
        # 根据是否需要返回字典, 构建不同形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return TFLongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
    
    # 构建Longformer模型和分类器
    def build(self, input_shape=None):
        # 如果已经构建过, 则直接返回
        if self.built:
            return
        self.built = True
        # 构建Longformer模型
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
```