# `.\models\longformer\modeling_tf_longformer.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入数据类装饰器，用于定义数据类
from dataclasses import dataclass
# 导入类型提示模块，用于指定变量类型
from typing import Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 TensorFlow 库
import tensorflow as tf

# 导入自定义模块中的函数和类
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
    keras,
    keras_serializable,
    unpack_inputs,
)
# 导入 TensorFlow 工具函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 导入通用工具模块中的函数和类
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 导入 Longformer 模型配置类
from .configuration_longformer import LongformerConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# Longformer 模型的预训练模型列表
_CHECKPOINT_FOR_DOC = "allenai/longformer-base-4096"
_CONFIG_FOR_DOC = "LongformerConfig"

# 定义一个大负数常量，用于在 Softmax 操作中抑制无关的信息
LARGE_NEGATIVE = -1e8

# Longformer 模型的预训练模型存档列表
TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # 更多 Longformer 模型详见：https://huggingface.co/models?filter=longformer
]

# 定义一个数据类，用于存储 Longformer 模型的基础输出
@dataclass
class TFLongformerBaseModelOutput(ModelOutput):
    """
    Longformer 模型的基础输出类，可能包含隐藏状态、本地和全局注意力等信息。
    继承自 ModelOutput 类。
    """
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。

        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层的隐藏状态，以及初始嵌入输出。

        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含模型每一层局部注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)`，
            其中 `x` 是全局注意力掩码的标记数。

            在注意力 softmax 后的局部注意力权重，用于计算自注意力头中的加权平均值。这些是从序列中每个标记到具有全局注意力的每个标记的注意力权重（前 `x` 个值），
            以及到注意力窗口内每个标记的注意力权重（剩余的 `attention_window + 1` 个值）。
            注意，前 `x` 个值是指文本中固定位置的标记，而剩余的 `attention_window + 1` 个值是指相对位置的标记：
            标记到自身的注意力权重位于索引 `x + attention_window / 2`，而前（后） `attention_window / 2` 个值是到前（后）标记的注意力权重。
            如果注意力窗口包含具有全局注意力的标记，则相应索引处的注意力权重设为 0；该值应从第一个 `x` 个注意力权重中获取。
            如果标记具有全局注意力，则到 `attentions` 中所有其他标记的注意力权重设为 0；该值应从 `global_attentions` 中获取。

        global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含模型每一层全局注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, x)`，
            其中 `x` 是具有全局注意力掩码的标记数。

            在注意力 softmax 后的全局注意力权重，用于计算自注意力头中的加权平均值。这些是从具有全局注意力的每个标记到序列中每个标记的注意力权重。

    Raises:
        None

    Returns:
        None
# 使用 dataclass 装饰器定义 TFLongformerBaseModelOutputWithPooling 类，用于存储 Longformer 模型的输出，并包含最后隐藏状态的汇总信息。
@dataclass
class TFLongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for Longformer's outputs that also contains a pooling of the last hidden states.
    """

    # 最后隐藏状态的张量，通常是模型的最终输出
    last_hidden_state: tf.Tensor = None
    # 汇总器的输出张量，可能用于整合最后隐藏状态
    pooler_output: tf.Tensor = None
    # 隐藏状态的元组，记录模型中间层的隐藏状态，如果有的话
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 注意力张量的元组，记录模型的注意力权重，如果有的话
    attentions: Tuple[tf.Tensor, ...] | None = None
    # 全局注意力张量的元组，记录模型的全局注意力权重，如果有的话
    global_attentions: Tuple[tf.Tensor, ...] | None = None


# 使用 dataclass 装饰器定义 TFLongformerMaskedLMOutput 类，用于存储掩码语言模型的输出。
@dataclass
class TFLongformerMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    """
    # 定义 loss 变量，类型为 tf.Tensor，形状为 (1,)，当提供 labels 参数时返回，用于掩码语言建模的损失
    loss: tf.Tensor | None = None
    # 定义一个变量 logits，类型为 tf.Tensor，初始值为 None，用于存储模型的输出 logits
    logits: tf.Tensor = None
    
    # 定义一个变量 hidden_states，类型为 Tuple[tf.Tensor, ...] 或者 None，初始值为 None，
    # 用于存储模型的隐藏状态（例如 RNN 或者 Transformer 中的隐藏状态）
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    
    # 定义一个变量 attentions，类型为 Tuple[tf.Tensor, ...] 或者 None，初始值为 None，
    # 用于存储模型中的注意力分数或注意力权重
    attentions: Tuple[tf.Tensor, ...] | None = None
    
    # 定义一个变量 global_attentions，类型为 Tuple[tf.Tensor, ...] 或者 None，初始值为 None，
    # 用于存储模型中的全局注意力分数或全局注意力权重
    global_attentions: Tuple[tf.Tensor, ...] | None = None
# 使用 dataclass 装饰器定义 TFLongformerQuestionAnsweringModelOutput 类，表示 Longformer 问答模型的输出
@dataclass
class TFLongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering Longformer models.
    问题回答 Longformer 模型输出的基类。
    """

    # 损失值，是一个 TensorFlow 张量或者 None（表示没有损失）
    loss: tf.Tensor | None = None

    # 起始位置的预测 logits（对数概率）
    start_logits: tf.Tensor = None

    # 结束位置的预测 logits（对数概率）
    end_logits: tf.Tensor = None

    # 隐藏状态的元组，可能为 None
    hidden_states: Tuple[tf.Tensor, ...] | None = None

    # 注意力分布的元组，可能为 None
    attentions: Tuple[tf.Tensor, ...] | None = None

    # 全局注意力的元组，可能为 None
    global_attentions: Tuple[tf.Tensor, ...] | None = None


# 使用 dataclass 装饰器定义 TFLongformerSequenceClassifierOutput 类，表示 Longformer 序列分类模型的输出
@dataclass
class TFLongformerSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    句子分类模型输出的基类。
    """
    # 定义变量 loss，用来存储分类（或回归，如果 config.num_labels==1）的损失值张量，可选项
    loss: tf.Tensor | None = None
    # logits: tf.Tensor = None
    # 声明一个变量 logits，类型为 tf.Tensor，初始赋值为 None，用于存储模型的输出 logits。
    
    # hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 声明一个变量 hidden_states，类型为 Tuple[tf.Tensor, ...] 或 None，初始赋值为 None。
    # 这个变量用于存储模型中间层的隐藏状态，可能是一个张量元组或者空值。
    
    # attentions: Tuple[tf.Tensor, ...] | None = None
    # 声明一个变量 attentions，类型为 Tuple[tf.Tensor, ...] 或 None，初始赋值为 None。
    # 这个变量用于存储模型中注意力机制的输出，可能是一个张量元组或者空值。
    
    # global_attentions: Tuple[tf.Tensor, ...] | None = None
    # 声明一个变量 global_attentions，类型为 Tuple[tf.Tensor, ...] 或 None，初始赋值为 None。
    # 这个变量用于存储模型中的全局注意力机制的输出，可能是一个张量元组或者空值。
# 使用 dataclass 装饰器声明一个数据类，用于存储输出结果
@dataclass
# TFLongformerMultipleChoiceModelOutput 类继承自 ModelOutput 类，表示多选模型的输出基类
class TFLongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.
    """
        Args:
            loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
                分类损失。
            logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
                *num_choices* 是输入张量的第二维度。(见上述的 *input_ids*)。

                分类得分（SoftMax 之前）。
            hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 被传递或者 `config.output_hidden_states=True` 时返回):
                形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组。

                模型在每一层输出的隐藏状态以及初始嵌入输出。
            attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 被传递或者 `config.output_attentions=True` 时返回):
                形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)` 的 `tf.Tensor` 元组，其中 `x` 是具有全局注意力掩码的标记数。

                注意力 softmax 后的局部注意力权重，用于计算自注意力头中的加权平均值。这些是从序列中每个标记到具有全局注意力的每个标记（前 `x` 个值）和到注意力窗口中每个标记（剩余 `attention_window + 1` 个值）的注意力权重。请注意，前 `x` 个值指的是文本中具有固定位置的标记，但剩余的 `attention_window + 1` 个值指的是具有相对位置的标记：一个标记到自身的注意力权重位于索引 `x + attention_window / 2`，前 `attention_window / 2`（后续）的值是到前（后） `attention_window / 2` 个标记的注意力权重。如果注意力窗口包含具有全局注意力的标记，则相应索引处的注意力权重设为 0；其值应从前 `x` 个注意力权重中访问。如果一个标记具有全局注意力，则 `attentions` 中所有其他标记的注意力权重设为 0，其值应从 `global_attentions` 中访问。
            global_attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 被传递或者 `config.output_attentions=True` 时返回):
                形状为 `(batch_size, num_heads, sequence_length, x)` 的 `tf.Tensor` 元组，其中 `x` 是具有全局注意力掩码的标记数。

                注意力 softmax 后的全局注意力权重，用于计算自注意力头中的加权平均值。这些是每个具有全局注意力的标记到序列中每个标记的注意力权重。
    # 定义一个 TensorFlow 张量 logits，初始化为 None
    logits: tf.Tensor = None
    # 定义一个元组，包含多个 TensorFlow 张量的隐藏状态，初始化为 None 或者空值
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 定义一个元组，包含多个 TensorFlow 张量的注意力权重，初始化为 None 或者空值
    attentions: Tuple[tf.Tensor, ...] | None = None
    # 定义一个元组，包含多个 TensorFlow 张量的全局注意力权重，初始化为 None 或者空值
    global_attentions: Tuple[tf.Tensor, ...] | None = None
@dataclass
class TFLongformerTokenClassifierOutput(ModelOutput):
    """
    定义一个数据类 TFLongformerTokenClassifierOutput，用于存储长形式模型的标记分类输出结果。
    继承自 ModelOutput 类。
    Base class for outputs of token classification models.
    """
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            分类损失。
            Classification loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            分类分数（SoftMax 之前）。
            Classification scores (before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态，包括初始嵌入输出。
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            每层的注意力权重。
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
            全局注意力权重。
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
            is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: tf.Tensor | None = None  # 分类损失，初始值为 None
    logits: tf.Tensor = None  # 分类分数，初始值为 None
    hidden_states: Tuple[tf.Tensor, ...] | None = None  # 模型每一层的隐藏状态，初始值为 None
    # 定义变量 attentions 和 global_attentions，它们分别是 Tensorflow 的张量元组或者 None 值
    attentions: Tuple[tf.Tensor, ...] | None = None
    global_attentions: Tuple[tf.Tensor, ...] | None = None
# 根据输入的形状和分隔符索引，计算全局注意力掩码，如果 `before_sep_token` 为 True，则将注意力放在分隔符之前的所有标记上，否则放在分隔符之后。
def _compute_global_attention_mask(input_ids_shape, sep_token_indices, before_sep_token=True):
    # 确保 `sep_token_indices` 的第二个维度为2，即 `input_ids` 应该有两个维度
    assert shape_list(sep_token_indices)[1] == 2, "`input_ids` should have two dimensions"
    # 从 `sep_token_indices` 中提取问题结束索引，为全局注意力掩码准备形状
    question_end_index = tf.reshape(sep_token_indices, (input_ids_shape[0], 3, 2))[:, 0, 1][:, None]
    # 创建布尔类型的注意力掩码，全局注意力位置为 True
    attention_mask = tf.expand_dims(tf.range(input_ids_shape[1], dtype=tf.int64), axis=0)
    attention_mask = tf.tile(attention_mask, (input_ids_shape[0], 1))
    if before_sep_token is True:
        # 如果 `before_sep_token` 为 True，则将问题结束索引扩展到整个序列长度，并生成相应的注意力掩码
        question_end_index = tf.tile(question_end_index, (1, input_ids_shape[1]))
        attention_mask = tf.cast(attention_mask < question_end_index, dtype=question_end_index.dtype)
    else:
        # 否则，将最后一个标记视为分隔符，不计入全局注意力，同时在中间有两个分隔符标记
        question_end_index = tf.tile(question_end_index + 1, (1, input_ids_shape[1]))
        attention_mask = tf.cast(
            (attention_mask > question_end_index) * (attention_mask < input_ids_shape[-1]),
            dtype=question_end_index.dtype,
        )
    return attention_mask


# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead 复制并修改为 Longformer 的 LM 头部模型
class TFLongformerLMHead(keras.layers.Layer):
    """Longformer Head for masked language modeling."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        # 定义全连接层，用于预测下一个标记
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义层归一化层，用于规范化
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 获取 GELU 激活函数
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个标记有一个输出偏置
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        # 定义输出偏置，形状为 (vocab_size,)
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
        # 获取输出嵌入
        return self.decoder
    # 设置输出的嵌入向量
    def set_output_embeddings(self, value):
        # 更新解码器的权重为给定的值
        self.decoder.weight = value
        # 更新解码器的词汇大小为给定值的第一个维度大小
        self.decoder.vocab_size = shape_list(value)[0]
    
    # 获取偏置项
    def get_bias(self):
        # 返回包含偏置项的字典
        return {"bias": self.bias}
    
    # 设置偏置项
    def set_bias(self, value):
        # 更新对象的偏置项为给定字典中的偏置项值
        self.bias = value["bias"]
        # 更新配置的词汇大小为给定偏置项的第一个维度大小
        self.config.vocab_size = shape_list(value["bias"])[0]
    
    # 模型的调用方法
    def call(self, hidden_states):
        # 全连接层：将隐藏状态映射到更高维度
        hidden_states = self.dense(hidden_states)
        # 激活函数：应用激活函数到全连接层输出
        hidden_states = self.act(hidden_states)
        # 层归一化：对激活函数输出进行层归一化处理
    
        # 投影回词汇大小的向量并加上偏置项
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
    
        # 返回处理后的隐藏状态
        return hidden_states
class TFLongformerEmbeddings(keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing and some extra casting.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 设定填充符索引为1
        self.padding_idx = 1
        # 保存配置参数
        self.config = config
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 最大位置编码数
        self.max_position_embeddings = config.max_position_embeddings
        # 初始化范围
        self.initializer_range = config.initializer_range
        # 层归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 添加词嵌入权重
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 添加token类型嵌入
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入
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
                # 构建层归一化
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 创建掩码，标记非填充符号的位置
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 累积索引，考虑过去的键值长度，乘以掩码
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
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 确保 `input_ids` 和 `inputs_embeds` 至少有一个不是 None
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # 检查 `input_ids` 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 从权重矩阵中根据 `input_ids` 提取对应的嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状，去除最后一个维度（通常是序列长度）
        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            # 如果 `token_type_ids` 为 None，则创建一个与 `inputs_embeds` 形状相同的全零张量作为 token 类型 id
            token_type_ids = tf.cast(tf.fill(dims=input_shape, value=0), tf.int64)

        if position_ids is None:
            if input_ids is not None:
                # 如果 `position_ids` 为 None 并且 `input_ids` 不为 None，则从 `input_ids` 创建位置 id
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 如果 `input_ids` 为 None，则创建一个从 `padding_idx + 1` 开始到 `input_shape[-1] + self.padding_idx` 结束的位置 id
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1, dtype=tf.int64),
                    axis=0,
                )

        # 根据位置 id 从位置嵌入矩阵中提取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token 类型 id 从 token 类型嵌入矩阵中提取 token 类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将输入嵌入、位置嵌入和 token 类型嵌入相加得到最终的嵌入张量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终嵌入张量进行 LayerNorm
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 在训练模式下对最终嵌入张量进行 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入张量
        return final_embeddings
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->Longformer
class TFLongformerIntermediate(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于中间表示，设置单元数和初始化器
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置获取中间激活函数，若为字符串则转换为对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理输入的隐藏状态张量
        hidden_states = self.dense(inputs=hidden_states)
        # 应用中间激活函数到处理后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，设置其构建结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->Longformer
class TFLongformerOutput(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于输出表示，设置单元数和初始化器
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，用于归一化输出表示
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于随机丢弃部分输出表示，以减少过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层处理输入的隐藏状态张量
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练阶段随机丢弃部分输出表示，用于正则化
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对处理后的表示进行 LayerNormalization，并加上输入张量，实现残差连接
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，设置其构建结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在 LayerNormalization 层，设置其构建结构
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->Longformer
class TFLongformerPooler(keras.layers.Layer):
    # 此类尚未实现，预留作为 Longformer 池化层的定义
    # 初始化函数，用于创建一个新的Longformer层实例
    def __init__(self, config: LongformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size,  # 设置全连接层的输出单元数为隐藏大小
            kernel_initializer=get_initializer(config.initializer_range),  # 初始化权重的方式
            activation="tanh",  # 激活函数为tanh
            name="dense",  # 层的名称
        )
        # 保存Longformer配置信息
        self.config = config

    # 调用函数，定义了如何使用层处理输入张量并返回输出张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 简单池化模型，通过取第一个token对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]  # 取第一个token对应的隐藏状态张量
        pooled_output = self.dense(inputs=first_token_tensor)  # 使用全连接层处理第一个token的隐藏状态

        return pooled_output  # 返回池化后的输出张量

    # 构建函数，在第一次调用call方法前构建层，通常用于初始化参数
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，直接返回
            return
        self.built = True  # 设置为已构建状态
        if getattr(self, "dense", None) is not None:  # 如果存在全连接层
            with tf.name_scope(self.dense.name):  # 使用全连接层的名称作为命名空间
                self.dense.build([None, None, self.config.hidden_size])  # 构建全连接层的权重
# 从 transformers.models.bert.modeling_tf_bert.TFBertSelfOutput 复制并修改为 Longformer
class TFLongformerSelfOutput(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于将输入向量映射到隐藏大小的空间
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # LayerNormalization 层，用于对输入进行归一化处理
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层，用于在训练过程中随机失活一部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 存储传入的配置信息
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 线性变换操作，将输入张量映射到隐藏大小的空间
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练过程中，对输出进行 dropout 处理，防止过拟合
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 LayerNormalization 对输入向量进行归一化处理，并与原始输入张量相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果 dense 层已定义，则构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 LayerNorm 层已定义，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFLongformerSelfAttention(keras.layers.Layer):
    # 初始化函数，接受配置、层ID以及其他关键字参数
    def __init__(self, config, layer_id, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将配置信息保存在实例变量中
        self.config = config

        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 设置注意力头数和头维度
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        # 创建查询、键、值的全连接层，用于自注意力机制
        self.query = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # 创建查询、键、值的全连接层，用于全局注意力机制
        self.query_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_global",
        )
        self.key_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_global",
        )
        self.value_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_global",
        )

        # 创建注意力概率的丢弃层
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)

        # 设置当前层的ID
        self.layer_id = layer_id

        # 检查并设置局部注意力窗口大小
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2
    # 定义神经网络层的构建函数，用于构建自定义层对象
    def build(self, input_shape=None):
        # 如果尚未构建过，则开始构建
        if not self.built:
            # 在 "query_global" 命名空间下构建 self.query_global 层
            with tf.name_scope("query_global"):
                self.query_global.build((self.config.hidden_size,))
            # 在 "key_global" 命名空间下构建 self.key_global 层
            with tf.name_scope("key_global"):
                self.key_global.build((self.config.hidden_size,))
            # 在 "value_global" 命名空间下构建 self.value_global 层
            with tf.name_scope("value_global"):
                self.value_global.build((self.config.hidden_size,))

        # 如果已经构建过，则直接返回，不进行重复构建
        if self.built:
            return

        # 标记为已构建
        self.built = True

        # 如果存在 self.query 属性，则构建该属性表示的层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])

        # 如果存在 self.key 属性，则构建该属性表示的层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])

        # 如果存在 self.value 属性，则构建该属性表示的层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])

        # 如果存在 self.query_global 属性，则构建该属性表示的层
        if getattr(self, "query_global", None) is not None:
            with tf.name_scope(self.query_global.name):
                self.query_global.build([None, None, self.config.hidden_size])

        # 如果存在 self.key_global 属性，则构建该属性表示的层
        if getattr(self, "key_global", None) is not None:
            with tf.name_scope(self.key_global.name):
                self.key_global.build([None, None, self.config.hidden_size])

        # 如果存在 self.value_global 属性，则构建该属性表示的层
        if getattr(self, "value_global", None) is not None:
            with tf.name_scope(self.value_global.name):
                self.value_global.build([None, None, self.config.hidden_size])

    # 定义神经网络层的调用函数，用于实现层的前向传播逻辑
    def call(
        self,
        inputs,
        training=False,
    ):
        # 函数体内容省略，需在实际应用中填充具体的前向传播逻辑
        pass

    # 定义静态方法，用于生成用于屏蔽无效位置的张量
    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # 创建正确的上三角布尔掩码
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )

        # 填充成完整的矩阵
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )
        mask_2d = tf.pad(mask_2d_upper, padding)  # 创建下三角掩码

        # 与上三角掩码组合
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

        # 广播到完整的矩阵
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))

        # 用于掩码的负无穷张量
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)

        # 执行掩码操作
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

        return input_tensor
    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """

        # 获取 value 张量的形状信息：batch_size, seq_len, num_heads, head_dim
        batch_size, seq_len, num_heads, head_dim = shape_list(value)

        # 断言确保 seq_len 是 2 * window_overlap 的倍数，用于后续分块处理
        tf.debugging.assert_equal(
            seq_len % (window_overlap * 2), 0, message="Seq_len has to be multiple of 2 * window_overlap"
        )

        # 断言确保 attn_probs 和 value 张量在前三个维度上形状相同（除了 head_dim 维度）
        tf.debugging.assert_equal(
            shape_list(attn_probs)[:3],
            shape_list(value)[:3],
            message="value and attn_probs must have same dims (except head_dim)",
        )

        # 断言确保 attn_probs 的最后一个维度为 2 * window_overlap + 1
        tf.debugging.assert_equal(
            shape_list(attn_probs)[3],
            2 * window_overlap + 1,
            message="attn_probs last dim has to be 2 * window_overlap + 1",
        )

        # 计算分块的数量，每个分块的大小为 window_overlap
        chunks_count = seq_len // window_overlap - 1

        # 将 attn_probs 张量重新排列和分块，以便进行后续的矩阵乘法计算
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )

        # 将 value 张量重新排列，以便进行后续的矩阵乘法计算
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )

        # 在 seq_len 的开头和结尾各填充 window_overlap 大小的值
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # 将 padded_value 张量分块，每块大小为 3 * window_overlap * head_dim
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

        # 断言确保 chunked_value 的形状正确
        tf.debugging.assert_equal(
            shape_list(chunked_value),
            [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
            message="Chunked value has the wrong shape",
        )

        # 调用类内部方法 _pad_and_diagonalize 对 chunked_attn_probs 进行填充和对角化处理
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        # 使用 Einsum 函数进行矩阵乘法计算，得到上下文向量
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)

        # 将 context 张量重新排列，以符合标准的张量形状顺序
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )

        # 返回计算得到的上下文张量
        return context
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """
        Pads the last two dimensions of `hidden_states_padded` tensor and then transposes the last two dimensions.

        Args:
            hidden_states_padded: Input tensor to be padded and transposed.
            paddings: Tensor specifying the padding amounts for each dimension.

        Returns:
            Transposed tensor after padding.
        """
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # padding value is not important because it will be overwritten
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        Shifts every row 1 step right, converting columns into diagonals.

        Example:

        chunked_hidden_states: A 4-dimensional tensor representing chunked hidden states.
        window_overlap: Integer representing the number of rows/columns to shift.

        Returns:
            Tensor with padded and diagonalized dimensions.
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

        return chunked_hidden_states
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # 获取隐藏状态张量的形状信息
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        # 计算输出块的数量，每个块大小为2w，重叠大小为w
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # 定义帧大小和帧步长（类似于卷积）
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        # 将隐藏状态重塑为二维张量以便进行分块操作
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # 使用帧大小和帧步长进行分块操作
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        # 断言确保分块操作的输出形状正确
        tf.debugging.assert_equal(
            shape_list(chunked_hidden_states),
            [batch_size, num_output_chunks, frame_size],
            message=(
                "Make sure chunking is correctly applied. `Chunked hidden states should have output dimension"
                f" {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}."
            ),
        )

        # 将分块后的隐藏状态重新重塑为所需的形状
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # 计算每个样本中全局注意力索引的数量
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)

        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)

        # 提取非零元素的全局注意力索引
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)

        # 计算哪些位置是局部索引中的全局注意力索引
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )

        # 提取局部索引中非零元素的位置
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)

        # 提取局部索引中零元素的位置
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
        # 计算批处理大小
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

        # 使用 Einsum 函数计算从全局键向量得到的注意力概率
        # 形状为 (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # 转置操作，将形状调整为 (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))

        # 创建用于掩码的形状
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )

        # 创建掩码张量并转换为与 attn_probs_from_global_key_trans 相同的数据类型
        mask = tf.ones(mask_shape) * -10000.0
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

        # 使用 scatter_nd_update 函数对 attn_probs_from_global_key_trans 应用掩码
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # 再次转置得到最终形状 (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # 将 attn_probs_from_global_key 与 attn_scores 连接起来
        # 形状为 (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        # 返回最终的注意力分数张量
        return attn_scores

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        ):
        # 计算批处理大小
        batch_size = shape_list(attn_probs)[0]

        # 仅保留全局注意力概率，截取前 max_num_global_attn_indices 个
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

        # 根据非零全局注意力索引，选择全局值向量
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)

        # 创建仅包含全局值向量的张量
        value_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_value_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # 计算仅含全局注意力的注意力输出
        attn_output_only_global = tf.einsum("blhs,bshd->blhd", attn_probs_only_global, value_vectors_only_global)

        # 重新整形注意力概率
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]

        # 计算包含全局和局部注意力的注意力输出
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )

        # 返回整合的注意力输出
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        attn_output,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
        training,
    ):
        # 定义向量重整形和转置函数，用于处理批量数据
        def reshape_and_transpose(self, vector, batch_size):
            return tf.reshape(
                tf.transpose(
                    tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)),
                    (0, 2, 1, 3),
                ),
                (batch_size * self.num_heads, -1, self.head_dim),
            )
class TFLongformerAttention(keras.layers.Layer):
    # TFLongformerAttention 类，继承自 keras.layers.Layer
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        # 初始化函数
        # 创建 self_attention 层，使用 TFLongformerSelfAttention 类
        self.self_attention = TFLongformerSelfAttention(config, layer_id, name="self")
        # 创建 dense_output 层，使用 TFLongformerSelfOutput 类
        self.dense_output = TFLongformerSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 修剪头部的函数，抛出未实现错误
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
        # 调用函数，执行注意力计算
        self_outputs = self.self_attention(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        # 使用 dense_output 层对注意力输出进行处理
        attention_output = self.dense_output(self_outputs[0], hidden_states, training=training)
        # 组装输出元组，包括注意力输出和其他可能的输出
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，则直接返回
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                # 使用 self_attention 层构建
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 使用 dense_output 层构建
                self.dense_output.build(None)


class TFLongformerLayer(keras.layers.Layer):
    # TFLongformerLayer 类，继承自 keras.layers.Layer
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        # 初始化函数
        # 创建 attention 层，使用 TFLongformerAttention 类
        self.attention = TFLongformerAttention(config, layer_id, name="attention")
        # 创建 intermediate 层，使用 TFLongformerIntermediate 类
        self.intermediate = TFLongformerIntermediate(config, name="intermediate")
        # 创建 longformer_output 层，使用 TFLongformerOutput 类
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
        # 调用函数，执行注意力计算
        attention_outputs = self.attention(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )
        # 获取注意力输出
        attention_output = attention_outputs[0]
        # 使用 intermediate 层对注意力输出进行处理
        intermediate_output = self.intermediate(attention_output)
        # 使用 longformer_output 层处理 intermediate 输出
        layer_output = self.longformer_output(intermediate_output, attention_output, training=training)
        # 组装输出元组，包括层输出和可能的注意力输出
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs
    # 定义神经网络层的构建方法，当输入形状为None时，表示可以适应任意输入形状
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将标志位设置为True，表示已经进行了构建
        self.built = True
        
        # 检查是否存在注意力层，并进行相应的构建
        if getattr(self, "attention", None) is not None:
            # 在命名空间下构建注意力层
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 检查是否存在中间层，并进行相应的构建
        if getattr(self, "intermediate", None) is not None:
            # 在命名空间下构建中间层
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 检查是否存在长形式输出层，并进行相应的构建
        if getattr(self, "longformer_output", None) is not None:
            # 在命名空间下构建长形式输出层
            with tf.name_scope(self.longformer_output.name):
                self.longformer_output.build(None)
class TFLongformerEncoder(keras.layers.Layer):
    # 定义 TFLongformerEncoder 类，继承自 keras.layers.Layer
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 调用父类构造函数进行初始化

        self.output_hidden_states = config.output_hidden_states
        # 从 config 参数中获取是否输出隐藏状态的设置

        self.output_attentions = config.output_attentions
        # 从 config 参数中获取是否输出注意力权重的设置

        # 创建 Longformer 层的列表，用于处理不同层的输入
        self.layer = [TFLongformerLayer(config, i, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        all_hidden_states = () if output_hidden_states else None
        all_attentions = all_global_attentions = () if output_attentions else None

        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则根据需要去除填充部分并保存
                hidden_states_to_add = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
                all_hidden_states = all_hidden_states + (hidden_states_to_add,)

            # 调用当前层的前向传播
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
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力权重，调整注意力权重的维度顺序
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (tf.transpose(layer_outputs[1], (0, 2, 1, 3)),)

                # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                all_global_attentions = all_global_attentions + (tf.transpose(layer_outputs[2], (0, 1, 3, 2)),)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            hidden_states_to_add = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
            all_hidden_states = all_hidden_states + (hidden_states_to_add,)

        # 取消填充部分
        # 对隐藏状态进行去除填充处理，以使其长度与输入的 input_ids.size(1) 一致
        hidden_states = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
        if output_attentions:
            # 如果需要输出注意力权重，对所有注意力权重进行去除填充处理
            all_attentions = (
                tuple([state[:, :, :-padding_len, :] for state in all_attentions])
                if padding_len > 0
                else all_attentions
            )

        if not return_dict:
            # 如果不需要返回字典形式的结果，则返回一个元组，包含非空的值
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
            )

        # 返回一个 TFLongformerBaseModelOutput 对象，包含指定的结果
        return TFLongformerBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )

    def build(self, input_shape=None):
        # 构建模型，确保每一层已经建立
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    # 使用当前层的名称作为命名空间，构建层
                    layer.build(None)
# 使用 keras_serializable 装饰器将类标记为可序列化的 Keras 层
@keras_serializable
# 定义 TFLongformerMainLayer 类，继承自 keras.layers.Layer
class TFLongformerMainLayer(keras.layers.Layer):
    # 指定配置类为 LongformerConfig
    config_class = LongformerConfig

    # 初始化方法，接受 config 参数和其他关键字参数
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 如果 attention_window 是整数，则进行如下断言和处理
        if isinstance(config.attention_window, int):
            # 断言 attention_window 必须为偶数
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            # 断言 attention_window 必须为正数
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            # 将 attention_window 扩展为一个列表，每层一个值
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            # 如果 attention_window 是列表，则断言其长度与 num_hidden_layers 相等
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        # 将配置参数赋值给对象属性
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.pad_token_id = config.pad_token_id
        self.attention_window = config.attention_window
        # 创建 TFLongformerEmbeddings 对象，并赋值给 embeddings 属性
        self.embeddings = TFLongformerEmbeddings(config, name="embeddings")
        # 创建 TFLongformerEncoder 对象，并赋值给 encoder 属性
        self.encoder = TFLongformerEncoder(config, name="encoder")
        # 如果 add_pooling_layer 为 True，则创建 TFLongformerPooler 对象，并赋值给 pooler 属性；否则 pooler 属性为 None
        self.pooler = TFLongformerPooler(config, name="pooler") if add_pooling_layer else None

    # 返回 embeddings 属性，用作输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层的方法，将 value 赋值给 embeddings 的权重，并更新 vocab_size 属性
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 抽象方法，用于剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 使用 unpack_inputs 装饰器定义模型调用方法，接受多个输入参数
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
    ):
        # 方法体部分未提供，通常用于执行模型的前向传播计算

    # _pad_to_window_size 方法定义，用于将输入序列的长度填充到指定的窗口大小
    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds,
        pad_token_id,
    ):
        # 方法体部分未提供，通常用于执行填充操作
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        # padding
        attention_window = (
            self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"

        # 获取输入数据的形状
        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        # 计算需要填充的长度，使序列长度能够整除注意力窗口大小
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # 创建填充张量
        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

        # 如果存在 input_ids，则对其进行填充，使用 pad_token_id 进行填充
        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)

        # 如果存在 position_ids，则对其进行填充，使用 pad_token_id 进行填充
        if position_ids is not None:
            # 使用与 modeling_roberta.RobertaEmbeddings 相同的方式，用 pad_token_id 填充
            position_ids = tf.pad(position_ids, paddings, constant_values=pad_token_id)

        # 如果存在 inputs_embeds，则根据 padding_len 对其进行填充
        if inputs_embeds is not None:
            if padding_len > 0:
                # 创建与填充长度相匹配的 input_ids_padding 张量，并利用 embeddings 方法得到 inputs_embeds_padding
                input_ids_padding = tf.cast(tf.fill((batch_size, padding_len), self.pad_token_id), tf.int64)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                # 将填充后的 inputs_embeds 与 inputs_embeds_padding 进行拼接
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

        # 对 attention_mask 进行填
    # 构建模型的方法，用于设置模型的各层和参数
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在嵌入层（embeddings），则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 使用嵌入层的名称作为命名空间
            with tf.name_scope(self.embeddings.name):
                # 调用嵌入层的 build 方法，传入空的输入形状
                self.embeddings.build(None)
        
        # 如果存在编码器（encoder），则构建编码器
        if getattr(self, "encoder", None) is not None:
            # 使用编码器的名称作为命名空间
            with tf.name_scope(self.encoder.name):
                # 调用编码器的 build 方法，传入空的输入形状
                self.encoder.build(None)
        
        # 如果存在池化层（pooler），则构建池化层
        if getattr(self, "pooler", None) is not None:
            # 使用池化层的名称作为命名空间
            with tf.name_scope(self.pooler.name):
                # 调用池化层的 build 方法，传入空的输入形状
                self.pooler.build(None)
    """
    这是一个抽象类，处理权重初始化以及下载和加载预训练模型的简单接口。

    config_class = LongformerConfig
    base_model_prefix = "longformer"

    @property
    def input_signature(self):
        sig = super().input_signature
        sig["global_attention_mask"] = tf.TensorSpec((None, None), tf.int32, name="global_attention_mask")
        return sig
    """
LONGFORMER_START_DOCSTRING = r"""
    这个模型继承自[`TFPreTrainedModel`]。请查看超类文档，了解库实现的所有通用方法（如下载或保存、调整输入嵌入大小、修剪头等）。

    这个模型也是一个 [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。可以像使用常规的 TF 2.0 Keras 模型一样使用它，并参考 TF 2.0 的文档了解有关一般使用和行为的所有内容。

    <Tip>

    `transformers` 中的 TensorFlow 模型和层接受两种输入格式：

    - 将所有输入作为关键字参数（类似于 PyTorch 模型）传递，或者
    - 将所有输入作为列表、元组或字典传递给第一个位置参数。

    支持第二种格式的原因是，Keras 方法在传递输入给模型和层时更喜欢这种格式。由于这种支持，当使用 `model.fit()` 等方法时，只需将输入和标签以 `model.fit()` 支持的任何格式传递即可！然而，如果要在 Keras 方法之外（如在使用 Keras `Functional` API 创建自己的层或模型时）使用第二种格式，有三种可能的方法可以使用来收集第一个位置参数中的所有输入张量：

    - 只有 `input_ids` 的单个张量，没有其他内容：`model(input_ids)`
    - 长度可变的列表，按照文档字符串中给定的顺序包含一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    - 一个字典，其中包含一个或多个输入张量，与文档字符串中给定的输入名称相关联：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    注意，当使用 [子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，不需要担心这些内容，因为可以像将输入传递给任何其他 Python 函数一样传递输入！

    </Tip>

    Parameters:
        config ([`LongformerConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化不会加载与模型相关的权重，仅加载配置。
            查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

LONGFORMER_INPUTS_DOCSTRING = r"""
    """
@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerModel(TFLongformerPreTrainedModel):
    """
    TFLongformerModel类继承自TFLongformerPreTrainedModel，用于输出不带特定头部的原始隐藏状态。

    This class copies code from [`TFRobertaModel`] and overwrites standard self-attention with longformer
    self-attention to provide the ability to process long sequences following the self-attention approach described in
    [Longformer: the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and
    Arman Cohan. Longformer self-attention combines a local (sliding window) and global attention to extend to long
    documents without the O(n^2) increase in memory and compute.

    The self-attention module `TFLongformerSelfAttention` implemented here supports the combination of local and global
    attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated
    attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future
    release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA
    kernel to be memory and compute efficient.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        # 初始化一个TFLongformerMainLayer实例，命名为longformer，用于长文档处理
        self.longformer = TFLongformerMainLayer(config, name="longformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 调用self.longformer的call方法，传递输入参数，获取输出结果
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
    # 定义一个方法 `build`，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在 `longformer` 属性，并且不为 None，则执行以下操作
        if getattr(self, "longformer", None) is not None:
            # 使用 `tf.name_scope` 来命名作用域为 `self.longformer.name`
            with tf.name_scope(self.longformer.name):
                # 调用 `self.longformer` 对象的 `build` 方法，传入 `None` 作为输入形状
                self.longformer.build(None)
@add_start_docstrings(
    """Longformer Model with a `language modeling` head on top.""",
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForMaskedLM(TFLongformerPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在加载 TF 模型时，以下带 '.' 的名称表示从 PT 模型中加载时允许的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Longformer 主层，不添加池化层，命名为 "longformer"
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        # 初始化 Longformer 语言模型头部，连接到 Longformer 的嵌入层，命名为 "lm_head"
        self.lm_head = TFLongformerLMHead(config, self.longformer.embeddings, name="lm_head")

    def get_lm_head(self):
        # 返回语言模型头部对象
        return self.lm_head

    def get_prefix_bias_name(self):
        # 警告：方法 get_prefix_bias_name 已弃用，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回模型名称加上语言模型头部名称的字符串
        return self.name + "/" + self.lm_head.name

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
        **kwargs,
    ):
        """
        使用 Longformer 进行前向传播，支持以下输入参数：
        - input_ids: 输入的模型标识符
        - attention_mask: 注意力遮罩，指定哪些元素需要被处理
        - head_mask: 头部遮罩，用于控制多头注意力层的掩码
        - global_attention_mask: 全局注意力遮罩，控制全局注意力机制
        - token_type_ids: 标记类型标识符，用于区分不同文本段落
        - position_ids: 位置标识符，指定输入序列中每个位置的绝对位置
        - inputs_embeds: 输入嵌入，替代输入模型标识符的嵌入表示
        - output_attentions: 是否输出注意力权重
        - output_hidden_states: 是否输出隐藏状态
        - return_dict: 是否返回结果字典
        - labels: 标签，用于模型训练
        - training: 是否为训练模式

        其中，kwargs 包含其他未显式列出的关键字参数。
        """
        pass  # 实际的前向传播逻辑在这里被省略了
    ) -> Union[TFLongformerMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 调用 Longformer 模型执行前向传播，获取模型输出
        outputs = self.longformer(
            input_ids=input_ids,  # 输入的 token IDs
            attention_mask=attention_mask,  # 注意力掩码，指定哪些 token 是有效的
            head_mask=head_mask,  # 头部掩码，指定哪些头部是有效的
            global_attention_mask=global_attention_mask,  # 全局注意力掩码，指定哪些全局注意力是有效的
            token_type_ids=token_type_ids,  # token 类型 IDs，用于区分不同句子的 token
            position_ids=position_ids,  # token 的位置 IDs
            inputs_embeds=inputs_embeds,  # 输入的嵌入表示
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典形式的输出
            training=training,  # 是否处于训练模式
        )
        sequence_output = outputs[0]  # 获取模型输出的序列表示
        prediction_scores = self.lm_head(sequence_output, training=training)  # 使用 LM 头部进行预测
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)  # 计算损失，如果没有标签则损失为 None

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # 如果不返回字典，构造输出元组
            return ((loss,) + output) if loss is not None else output  # 返回损失和输出元组，如果损失为 None 则只返回输出

        # 返回字典形式的 TFLongformerMaskedLMOutput，包括损失、预测 logits、隐藏状态和注意力
        return TFLongformerMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):  # 使用 Longformer 名称创建命名空间
                self.longformer.build(None)  # 构建 Longformer 模型
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):  # 使用 LM 头部名称创建命名空间
                self.lm_head.build(None)  # 构建 LM 头部模型
"""
Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
TriviaQA (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 引用了 TFLongformerPreTrainedModel 和 TFQuestionAnsweringLoss，构建了一个带有跨度分类头部的 Longformer 模型，用于类似 SQuAD / TriviaQA 的抽取式问答任务。

class TFLongformerForQuestionAnswering(TFLongformerPreTrainedModel, TFQuestionAnsweringLoss):
    # 当从 PT 模型加载 TF 模型时，'.' 表示授权的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        # 创建 Longformer 主层，不添加池化层
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        # 创建用于输出的 Dense 层，输出大小为 config.num_labels
        self.qa_outputs = keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )
        self.config = config

    @unpack_inputs
    # 将文档字符串添加到模型前向方法，描述输入的格式
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串，描述了如何使用模型和预期输出
    @add_code_sample_docstrings(
        checkpoint="allenai/longformer-large-4096-finetuned-triviaqa",
        output_type=TFLongformerQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.96,
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
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        pass  # 这里是模型的前向计算方法，具体内容未提供，需要根据具体实现补充

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经定义了 self.longformer，则构建它
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 如果已经定义了 self.qa_outputs，则构建它
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                # 构建 Dense 层，输出形状为 [None, None, self.config.hidden_size]
                self.qa_outputs.build([None, None, self.config.hidden_size])


class TFLongformerClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    # 初始化函数，用于创建一个新的模型实例
    def __init__(self, config, **kwargs):
        # 调用父类（可能是神经网络层）的初始化方法
        super().__init__(**kwargs)
        
        # 创建一个全连接层，用于处理输入数据
        self.dense = keras.layers.Dense(
            config.hidden_size,  # 设置隐藏层的大小，从配置中获取
            kernel_initializer=get_initializer(config.initializer_range),  # 使用指定范围的初始化器来初始化权重矩阵
            activation="tanh",  # 激活函数为双曲正切函数
            name="dense",  # 层的名称为 dense
        )
        
        # 创建一个 Dropout 层，用于在训练过程中随机失活输入单元
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        
        # 创建一个全连接层，用于最终输出模型的预测结果
        self.out_proj = keras.layers.Dense(
            config.num_labels,  # 输出层的大小，从配置中获取
            kernel_initializer=get_initializer(config.initializer_range),  # 使用指定范围的初始化器来初始化权重矩阵
            name="out_proj"  # 层的名称为 out_proj
        )
        
        # 将配置信息存储到模型中，以便在需要时进行访问
        self.config = config
    
    # 前向传播函数，用于计算模型的输出结果
    def call(self, hidden_states, training=False):
        # 只保留每个样本的第一个隐藏状态，相当于取 <s> 标记（对应 [CLS]）
        hidden_states = hidden_states[:, 0, :]
        
        # 根据训练状态应用 Dropout 层，用于防止过拟合
        hidden_states = self.dropout(hidden_states, training=training)
        
        # 通过全连接层处理隐藏状态，以提取特征
        hidden_states = self.dense(hidden_states)
        
        # 再次应用 Dropout 层，增强模型的泛化能力
        hidden_states = self.dropout(hidden_states, training=training)
        
        # 最终通过输出层得到模型的预测结果
        output = self.out_proj(hidden_states)
        
        # 返回模型的输出结果
        return output
    
    # 模型构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建好了，则直接返回
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果存在全连接层 dense，则构建其层次结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果存在全连接层 out_proj，则构建其层次结构
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])
# 在 TFLongformerForSequenceClassification 类的开始处添加详细的文档字符串，描述其作为 Longformer 模型转换器的用途，
# 以及其在顶部具有一个序列分类/回归头部的功能（即在汇总输出之上的线性层），例如用于 GLUE 任务。
@add_start_docstrings(
    """
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForSequenceClassification(TFLongformerPreTrainedModel, TFSequenceClassificationLoss):
    # 当从 PT 模型加载 TF 模型时，带有 '.' 的名称表示在加载过程中可以忽略的授权的意外/丢失的层。
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置模型可以处理的标签数目
        self.num_labels = config.num_labels

        # 创建 Longformer 主层，不添加池化层，命名为 "longformer"
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        
        # 创建 Longformer 分类头部，命名为 "classifier"
        self.classifier = TFLongformerClassificationHead(config, name="classifier")

    # 使用装饰器为 call 方法添加详细的文档字符串，描述其前向推理的输入和输出，基于 LONGFORMER_INPUTS_DOCSTRING 格式化字符串
    # 添加代码示例的文档字符串，显示如何使用此方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerSequenceClassifierOutput,
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
    # 定义函数签名，指定输入参数和返回类型
    ) -> Union[TFLongformerSequenceClassifierOutput, Tuple[tf.Tensor]]:
        # 如果 input_ids 存在且不是 TensorFlow 张量，则将其转换为 TensorFlow 张量
        if input_ids is not None and not isinstance(input_ids, tf.Tensor):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
        # 否则，如果 input_ids 存在，则将其强制转换为 tf.int64 类型的张量
        elif input_ids is not None:
            input_ids = tf.cast(input_ids, tf.int64)

        # 如果 attention_mask 存在且不是 TensorFlow 张量，则将其转换为 TensorFlow 张量
        if attention_mask is not None and not isinstance(attention_mask, tf.Tensor):
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
        # 否则，如果 attention_mask 存在，则将其强制转换为 tf.int64 类型的张量
        elif attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.int64)

        # 如果 global_attention_mask 存在且不是 TensorFlow 张量，则将其转换为 TensorFlow 张量
        if global_attention_mask is not None and not isinstance(global_attention_mask, tf.Tensor):
            global_attention_mask = tf.convert_to_tensor(global_attention_mask, dtype=tf.int64)
        # 否则，如果 global_attention_mask 存在，则将其强制转换为 tf.int64 类型的张量
        elif global_attention_mask is not None:
            global_attention_mask = tf.cast(global_attention_mask, tf.int64)

        # 如果 global_attention_mask 为空且 input_ids 存在，则发出警告并初始化全局注意力掩码
        if global_attention_mask is None and input_ids is not None:
            logger.warning_once("Initializing global attention on CLS token...")
            # 在 CLS 标记上的全局注意力
            global_attention_mask = tf.zeros_like(input_ids)
            # 创建一个更新张量，其形状为 input_ids 的第一个维度大小，数据类型为 tf.int64
            updates = tf.ones(shape_list(input_ids)[0], dtype=tf.int64)
            # 创建索引张量，用于更新 global_attention_mask
            indices = tf.pad(
                tensor=tf.expand_dims(tf.range(shape_list(input_ids)[0], dtype=tf.int64), axis=1),
                paddings=[[0, 0], [0, 1]],
                constant_values=0,
            )
            # 使用 tf.tensor_scatter_nd_update 函数更新 global_attention_mask
            global_attention_mask = tf.tensor_scatter_nd_update(
                global_attention_mask,
                indices,
                updates,
            )

        # 调用 self.longformer 进行序列分类器的计算，传入多个参数
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
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 将序列输出传入分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 如果 labels 存在，则计算损失，否则设置损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典形式的结果，则组合输出，并根据是否存在损失决定是否包含损失
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFLongformerSequenceClassifierOutput 类型的对象，包含损失、logits、隐藏状态和注意力
        return TFLongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
    # 定义神经网络层的构建方法，当输入形状不为None时，指示该方法已经被调用过一次
    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回，避免重复构建
        if self.built:
            return
        # 将网络层标记为已构建状态
        self.built = True
        # 如果存在长形式网络层（longformer），则构建该网络层
        if getattr(self, "longformer", None) is not None:
            # 在命名空间中构建长形式网络层
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 如果存在分类器网络层（classifier），则构建该网络层
        if getattr(self, "classifier", None) is not None:
            # 在命名空间中构建分类器网络层
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
    """
    定义了一个基于Longformer模型的多选题分类器，通过在汇总输出之上添加一个线性层和softmax来实现，
    例如用于RocStories/SWAG任务。
    """

    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_missing = [r"dropout"]
    """
    在从PT模型加载TF模型时，表示授权的意外/缺失层的名称列表。
    """

    def __init__(self, config, *inputs, **kwargs):
        """
        初始化方法，用于创建模型实例。
        Args:
            config: Longformer模型的配置对象。
            *inputs: 可变长度的输入参数。
            **kwargs: 关键字参数。
        """
        super().__init__(config, *inputs, **kwargs)

        self.longformer = TFLongformerMainLayer(config, name="longformer")
        """
        创建Longformer的主层实例，使用给定的配置和名称。
        """

        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        """
        创建一个Dropout层，用于在训练过程中随机丢弃部分神经元，防止过拟合。
        """

        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        """
        创建一个全连接层作为分类器，输出维度为1，使用给定的初始化器范围初始化权重。
        """

        self.config = config
        """
        保存配置对象供后续使用。
        """

    @property
    def input_signature(self):
        """
        定义模型的输入签名，指定了各输入张量的形状和类型。
        """
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

        # 如果 `input_ids` 不为 None，则获取其第二维度的大小作为 `num_choices`，并获取序列长度 `seq_length`
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，使用 `inputs_embeds` 的第二维度大小作为 `num_choices`，并获取序列长度 `seq_length`
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入张量展平成二维张量，如果相应的输入不为 None
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

        # 调用长形式模型进行处理，传入展平后的输入和其他参数
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
        # 获取汇聚输出（pooled output）
        pooled_output = outputs[1]

        # 对汇聚输出应用 dropout 操作
        pooled_output = self.dropout(pooled_output)
        # 将汇聚输出传入分类器，得到 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重新整形为二维张量，形状为 (-1, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果没有提供 labels，则 loss 为 None；否则使用指定方法计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不要求返回字典，则构造输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则构造 TFLongformerMultipleChoiceModelOutput 对象
        return TFLongformerMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
    # 构建函数，用于构建模型的层次结构，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        
        # 如果存在名为"longformer"的属性且不为None，则构建其对应的层次结构
        if getattr(self, "longformer", None) is not None:
            # 在TensorFlow中创建名为self.longformer.name的命名空间
            with tf.name_scope(self.longformer.name):
                # 构建self.longformer层的结构
                self.longformer.build(None)
        
        # 如果存在名为"classifier"的属性且不为None，则构建其对应的层次结构
        if getattr(self, "classifier", None) is not None:
            # 在TensorFlow中创建名为self.classifier.name的命名空间
            with tf.name_scope(self.classifier.name):
                # 构建self.classifier层的结构，输入维度为[None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器添加文档字符串，描述了这是一个在Longformer模型基础上增加了标记分类头的类，用于命名实体识别（NER）等任务
@add_start_docstrings(
    """
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    LONGFORMER_START_DOCSTRING,  # 引用了LONGFORMER_START_DOCSTRING作为模型的开始文档字符串
)
class TFLongformerForTokenClassification(TFLongformerPreTrainedModel, TFTokenClassificationLoss):
    # 在从PyTorch模型加载到TensorFlow模型时，表示可以忽略的预期意外或缺失层的名称列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 在加载模型时可以忽略的缺失层名称列表
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 从配置中获取标签的数量
        self.longformer = TFLongformerMainLayer(config=config, add_pooling_layer=False, name="longformer")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)  # 根据配置添加一个Dropout层
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )  # 增加一个全连接层作为分类器，输出维度为标签数量，使用指定的初始化器初始化权重
        self.config = config  # 保存配置信息

    # 使用装饰器定义模型的call方法，并添加多个文档字符串，描述了模型的输入和输出等信息
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 指定了文档中的检查点示例
        output_type=TFLongformerTokenClassifierOutput,  # 指定了输出类型
        config_class=_CONFIG_FOR_DOC,  # 指定了配置类示例
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
    ) -> Union[TFLongformerTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 定义函数签名和文档字符串，指定函数返回类型为 TFLongformerTokenClassifierOutput 或包含 tf.Tensor 的元组

        # 调用 Longformer 模型进行前向传播
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的输出传入分类器获取 logits
        logits = self.classifier(sequence_output)
        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict=False，则返回不同的输出形式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict=True，则返回 TFLongformerTokenClassifierOutput 对象
        return TFLongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 Longformer 模型，则构建它
        if getattr(self, "longformer", None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        # 如果存在分类器，则构建它，指定其输入形状
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
```