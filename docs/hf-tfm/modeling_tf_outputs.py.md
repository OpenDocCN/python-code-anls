# `.\modeling_tf_outputs.py`

```py
# 导入警告模块，用于处理警告信息
import warnings
# 导入数据类装饰器，用于定义数据类
from dataclasses import dataclass
# 导入类型提示，用于类型注解
from typing import List, Optional, Tuple

# 导入 TensorFlow 库
import tensorflow as tf

# 从当前目录下的 utils 模块中导入 ModelOutput 类
from .utils import ModelOutput


@dataclass
class TFBaseModelOutput(ModelOutput):
    """
    模型输出的基类，包含可能的隐藏状态和注意力信息。

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或者 `config.output_hidden_states=True`):
            元组，包含每一层的隐藏状态 `tf.Tensor`（一个用于嵌入输出，一个用于每一层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层的隐藏状态，包括初始嵌入层的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 时返回或者 `config.output_attentions=True`):
            元组，包含每一层的注意力权重 `tf.Tensor` 的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头的加权平均值。
    """

    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFBaseModelOutputWithNoAttention(ModelOutput):
    """
    模型输出的基类，包含可能的隐藏状态，但不包含注意力信息。

    Args:
        last_hidden_state (`tf.Tensor` shape `(batch_size, num_channels, height, width)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 时返回或者 `config.output_hidden_states=True`):
            元组，包含每一层的隐藏状态 `tf.Tensor`（一个用于嵌入层的输出，如果模型有嵌入层，一个用于每一层的输出）的形状为 `(batch_size, num_channels, height, width)`。

            模型在每一层的隐藏状态，包括可选的初始嵌入层的输出。
    """

    last_hidden_state: tf.Tensor = None
    # 声明一个可选类型的变量hidden_states，默认为None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
@dataclass
class TFBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.

            This output is usually *not* a good summary of the semantic content of the input, you're often better with
            averaging or pooling the sequence of hidden-states for the whole input sequence.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    
    last_hidden_state: tf.Tensor = None  # 最后一层模型输出的隐藏状态张量
    pooler_output: tf.Tensor = None  # 经过线性层和Tanh激活函数处理后的第一个标记的隐藏状态张量
    hidden_states: Tuple[tf.Tensor] | None = None  # 每层输出的隐藏状态张量的元组，包括初始嵌入层输出
    attentions: Tuple[tf.Tensor] | None = None  # 注意力权重的元组，用于计算自注意力头中的加权平均值


@dataclass
class TFBaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: tf.Tensor = None  # 最后一层模型输出的隐藏状态张量
    pooler_output: tf.Tensor = None  # 在空间维度进行池化操作后的最后一层隐藏状态张量
    hidden_states: Tuple[tf.Tensor] | None = None  # 每层输出的隐藏状态张量的元组，包括可选的初始嵌入层输出
    # 定义变量 `last_hidden_state`，类型为 `tf.Tensor`，初始值为 None
    last_hidden_state: tf.Tensor = None
    # 定义变量 `pooler_output`，类型为 `tf.Tensor`，初始值为 None
    pooler_output: tf.Tensor = None
    # 定义变量 `hidden_states`，类型为 `Optional[Tuple[tf.Tensor, ...]]`，初始值为 None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
@dataclass
class TFBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.

            This output is usually *not* a good summary of the semantic content of the input, you're often better with
            averaging or pooling the sequence of hidden-states for the whole input sequence.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    # 最后一个隐藏状态，形状为(batch_size, sequence_length, hidden_size)，表示模型最后一层的隐藏状态序列
    last_hidden_state: tf.Tensor = None

    # 汇聚输出，形状为(batch_size, hidden_size)，表示经过线性层和Tanh激活函数处理的分类标记的最后一层隐藏状态
    # 在预训练期间，线性层的权重由下一个句子预测（分类）目标进行训练
    pooler_output: tf.Tensor = None

    # 历史关键值，形状为List[tf.Tensor]，长度为config.n_layers，每个张量形状为(2, batch_size, num_heads, sequence_length, embed_size_per_head)
    # 当传递use_cache=True或config.use_cache=True时返回，包含预计算的隐藏状态（注意力块中的键和值），可用于加速序列解码
    past_key_values: List[tf.Tensor] | None = None

    # 隐藏状态，形状为tuple(tf.Tensor)，当传递output_hidden_states=True或config.output_hidden_states=True时返回
    # 包含每一层输出的隐藏状态的元组，以及初始嵌入输出
    hidden_states: Tuple[tf.Tensor] | None = None

    # 注意力权重，形状为tuple(tf.Tensor)，当传递output_attentions=True或config.output_attentions=True时返回
    # 包含每一层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)，用于计算自注意力头部的加权平均值
    attentions: Tuple[tf.Tensor] | None = None

    # 交叉注意力权重，形状为tuple(tf.Tensor)，当传递output_attentions=True或config.output_attentions=True时返回
    # 解码器的交叉注意力层的注意力权重，经过注意力softmax后，用于计算交叉注意力头部的加权平均值
    cross_attentions: Tuple[tf.Tensor] | None = None
    cross_attentions: Tuple[tf.Tensor] | None = None
# 定义一个带有过去键/值的模型输出类，继承自`ModelOutput`
@dataclass
class TFBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义类的属性：最后一个隐藏状态
    last_hidden_state: tf.Tensor = None
    # 定义类的属性：过去键/值的列表，用于加速顺序解码
    past_key_values: List[tf.Tensor] | None = None
    # 定义类的属性：包含每层隐藏状态的元组
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义类的属性：每层注意力权重的元组
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFBaseModelOutputWithCrossAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """
    """
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出隐藏状态序列。
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组的形式，包含每层模型的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组的形式，包含每层注意力权重，用于计算自注意力中加权平均值。
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组的形式，包含解码器跨注意力层的注意力权重，用于计算跨注意力中加权平均值。
    """
    
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    cross_attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Model output class for Transformer-based models that includes past key/values and cross-attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Final layer hidden-states of the model.
            
            If `past_key_values` is used, only the last hidden-state of shape `(batch_size, 1, hidden_size)` is output.
        past_key_values (`List[tf.Tensor]`, *optional*):
            List of tensors, each of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`.

            Pre-computed hidden-states (key and values) for sequential decoding speed-up.
        hidden_states (`Tuple[tf.Tensor]`, *optional*):
            Tuple of tensors, one for embeddings and one for each layer's hidden-states, each of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`Tuple[tf.Tensor]`, *optional*):
            Tuple of tensors, one for each layer, each of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attention weights after softmax, used for self-attention heads.
        cross_attentions (`Tuple[tf.Tensor]`, *optional*):
            Tuple of tensors, one for each layer, each of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attention weights of the decoder's cross-attention layer after softmax.
    """

    last_hidden_state: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    cross_attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFSeq2SeqModelOutput(ModelOutput):
    """
    Model output class for Seq2Seq Transformer-based models.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Final layer hidden-states of the encoder.
        past_key_values (`List[tf.Tensor]`, *optional*):
            List of tensors, each of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`.

            Pre-computed hidden-states (key and values) for decoder's sequential decoding speed-up.
        decoder_hidden_states (`Tuple[tf.Tensor]`, *optional*):
            Tuple of tensors for decoder's hidden-states, each of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer for the decoder.
        decoder_attentions (`Tuple[tf.Tensor]`, *optional*):
            Tuple of tensors for decoder's attentions, each of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attention weights after softmax for the decoder.
    """

    last_hidden_state: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 定义交叉注意力的张量元组或空值，初始值为 None
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 定义编码器最后一个隐藏状态的张量或空值，初始值为 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 定义编码器隐藏状态的张量元组或空值，初始值为 None
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 定义编码器注意力的张量元组或空值，初始值为 None
    encoder_attentions: Tuple[tf.Tensor] | None = None
# 基于 ModelOutput 的数据类，表示因果语言模型（或自回归模型）的输出。
@dataclass
class TFCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 损失值张量，形状为 `(n,)`，当提供 `labels` 时返回，用于语言模型的损失计算（用于预测下一个标记）。
    loss: tf.Tensor | None = None
    # 预测分数张量，形状为 `(batch_size, sequence_length, config.vocab_size)`，在 SoftMax 之前的每个词汇标记的预测分数。
    logits: tf.Tensor = None
    # 隐藏状态元组，包含每层输出的张量（嵌入输出和每个层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力张量元组，包含每层的注意力权重张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """
    # 定义函数参数和返回类型的注释
    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            语言建模损失（用于下一个标记预测）。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（在 SoftMax 之前每个词汇标记的分数）。
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            包含预先计算的隐藏状态（注意力块中的键和值）的列表，可用于加速顺序解码。
            长度为 `config.n_layers` 的 `tf.Tensor` 列表，每个张量的形状为 `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层输出的隐藏状态加上初始嵌入输出的元组。
            包含 `tf.Tensor`（嵌入输出的一个 + 每层输出的一个），形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            self-attention 头部中的加权平均计算所使用的注意力 softmax 后的注意力权重。
            包含每一层的 `tf.Tensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
@dataclass
class TFCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: tf.Tensor | None = None  # Language modeling loss tensor, optional
    logits: tf.Tensor = None  # Prediction scores before SoftMax for each token
    past_key_values: List[tf.Tensor] | None = None  # Pre-computed hidden states for sequential decoding
    hidden_states: Tuple[tf.Tensor] | None = None  # Hidden states of the model at each layer output
    attentions: Tuple[tf.Tensor] | None = None  # Attention weights for self-attention heads
    cross_attentions: Tuple[tf.Tensor] | None = None  # Attention weights for cross-attention heads


@dataclass
class TFMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    """
    # 定义 loss 变量，表示掩码语言建模的损失，形状为 (n,)，当提供 labels 参数时返回
    loss: tf.Tensor | None = None
    # 定义 logits 变量，表示语言建模头部的预测分数，形状为 (batch_size, sequence_length, config.vocab_size)
    logits: tf.Tensor = None
    # 定义 hidden_states 变量，表示模型每层的隐藏状态的元组，形状为 (batch_size, sequence_length, hidden_size)
    # 当 output_hidden_states=True 或 config.output_hidden_states=True 时返回
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义 attentions 变量，表示自注意力头部的注意力权重的元组，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    # 当 output_attentions=True 或 config.output_attentions=True 时返回
    attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    """

    # Optional: Loss tensor representing the model's computed loss
    loss: tf.Tensor | None = None

    # Optional: Logits tensor containing the model's predictions
    logits: tf.Tensor = None

    # Optional: List of past key values for attention mechanisms
    past_key_values: List[tf.Tensor] | None = None

    # Optional: Tuple of tensors for hidden states of the decoder
    decoder_hidden_states: Tuple[tf.Tensor] | None = None

    # Optional: Tuple of tensors for attention weights of the decoder
    decoder_attentions: Tuple[tf.Tensor] | None = None

    # Optional: Tuple of tensors for cross-attention weights
    cross_attentions: Tuple[tf.Tensor] | None = None

    # Optional: Tensor representing the last hidden state of the encoder
    encoder_last_hidden_state: tf.Tensor | None = None

    # Optional: Tuple of tensors for hidden states of the encoder
    encoder_hidden_states: Tuple[tf.Tensor] | None = None

    # Optional: Tuple of tensors for attention weights of the encoder
    encoder_attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFNextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """

    # Optional: Loss tensor representing the next sentence prediction loss
    loss: tf.Tensor | None = None

    # Required: Logits tensor for the next sentence prediction
    logits: tf.Tensor = None

    # Optional: Tuple of tensors for hidden states of the model
    hidden_states: Tuple[tf.Tensor] | None = None

    # Optional: Tuple of tensors for attention weights of the model
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    """
    Args:
        loss (`tf.Tensor` of shape `(batch_size, )`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 `config.num_labels==1`）的损失。
            当提供 `labels` 参数时返回。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 `config.num_labels==1`）的分数（SoftMax 之前的值）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            由 `tf.Tensor` 组成的元组（当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回）。
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每个层输出的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            由 `tf.Tensor` 组成的元组（当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回）。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力权重经过注意力 SoftMax 后的结果，用于计算自注意力头部的加权平均值。
    """

    loss: tf.Tensor | None = None  # 初始化为 None，表示损失值尚未设置
    logits: tf.Tensor = None  # 初始化为 None，表示逻辑分数尚未设置
    hidden_states: Tuple[tf.Tensor] | None = None  # 初始化为 None，表示隐藏状态尚未设置
    attentions: Tuple[tf.Tensor] | None = None  # 初始化为 None，表示注意力权重尚未设置
# 使用 `dataclass` 装饰器声明一个类，表示一个序列到序列句子分类模型的输出。
@dataclass
class TFSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    序列到序列句子分类模型输出的基础类。

    """

    # 表示损失值的张量，可以为 None
    loss: tf.Tensor | None = None
    # 表示逻辑回归输出的张量
    logits: tf.Tensor = None
    # 表示过去键值的列表，可以为 None
    past_key_values: List[tf.Tensor] | None = None
    # 表示解码器隐藏状态的元组，可以为 None
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 表示解码器注意力的元组，可以为 None
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 表示交叉注意力的元组，可以为 None
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 表示编码器最后隐藏状态的张量，可以为 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 表示编码器隐藏状态的元组，可以为 None
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 表示编码器注意力的元组，可以为 None
    encoder_attentions: Tuple[tf.Tensor] | None = None


# 使用 `dataclass` 装饰器声明一个类，表示语义分割模型的输出。
@dataclass
class TFSemanticSegmenterOutput(ModelOutput):
    """
    语义分割模型输出的基础类。

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, 当提供 `labels` 时返回):
            分类（或回归，如果 `config.num_labels==1`）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            每个像素的分类分数。

            <Tip warning={true}>

            返回的 logits 不一定与作为输入传递的 `pixel_values` 的大小相同。这是为了避免进行两次插值并在将 logits 调整回原始图像大小时失去一些质量。
            您应该始终检查 logits 的形状并根据需要进行调整大小。

            </Tip>

        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            `tf.Tensor` 的元组（如果模型具有嵌入层，则为一个用于每层输出的隐藏状态的输出 + 一个用于每个层输出的初始嵌入输出），
            形状为 `(batch_size, patch_size, hidden_size)`。

            模型在每层输出之后的隐藏状态，加上可选的初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            `tf.Tensor` 的元组（每层一个），
            形状为 `(batch_size, num_heads, patch_size, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 表示损失值的张量，可以为 None
    loss: tf.Tensor | None = None
    # 表示逻辑回归输出的张量
    logits: tf.Tensor = None
    # 表示隐藏状态的元组，可以为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 表示注意力的元组，可以为 None
    attentions: Tuple[tf.Tensor] | None = None


# 使用 `dataclass` 装饰器声明一个类，表示不输出注意力分数的语义分割模型的输出。
@dataclass
class TFSemanticSegmenterOutputWithNoAttention(ModelOutput):
    """
    不输出注意力分数的语义分割模型输出的基础类。

    """
    # 定义函数的参数和返回值类型注释
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 `config.num_labels==1`）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            每个像素的分类得分。

            <Tip warning={true}>

            返回的 logits 不一定与输入的 `pixel_values` 具有相同的大小。这是为了避免在用户需要将 logits 调整回原始图像大小时进行两次插值并丢失一些质量。您应始终检查 logits 的形状并根据需要调整大小。

            </Tip>

        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, patch_size, hidden_size)` 的 `tf.Tensor` 元组（当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回）。

            模型在每个层的输出隐藏状态加上可选的初始嵌入输出。

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
# 定义 TFImageClassifierOutput 类，用于表示图像分类模型的输出结果
@dataclass
class TFImageClassifierOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类损失（如果提供 `labels` 参数）。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类得分（如果 `config.num_labels==1` 则是回归分数），未经 SoftMax 处理。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `tf.Tensor`（用于嵌入层的输出，如果模型有嵌入层，+ 每个阶段的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每个阶段输出的隐藏状态（也称为特征图）。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `tf.Tensor`（每个层的注意力权重），形状为 `(batch_size, num_heads, patch_size, sequence_length)`。

            注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


# 定义 TFMultipleChoiceModelOutput 类，用于表示多项选择模型的输出结果
@dataclass
class TFMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (`tf.Tensor` of shape *(batch_size, )*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            `num_choices` 是输入张量的第二维。参见上面的 `input_ids`。

            分类得分（未经 SoftMax 处理）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `tf.Tensor`（用于嵌入层的输出 + 每层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `tf.Tensor`（每个层的注意力权重），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFTokenClassifierOutput(ModelOutput):
    """
    Token 分类模型输出的基类。

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of unmasked labels, returned when `labels` is provided):
            分类损失。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            分类分数（SoftMax 之前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含 `tf.Tensor`（一个用于嵌入输出 + 每层输出的 `tf.Tensor`），形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态，加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含 `tf.Tensor`（每个层的注意力权重），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力权重经过 SoftMax 后的结果，用于计算自注意力头部的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFQuestionAnsweringModelOutput(ModelOutput):
    """
    问答模型输出的基类。

    Args:
        loss (`tf.Tensor` of shape `(batch_size, )`, *optional*, returned when `start_positions` and `end_positions` are provided):
            总的跨度提取损失，为开始和结束位置的交叉熵之和。
        start_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            起始位置的分数（SoftMax 之前）。
        end_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            结束位置的分数（SoftMax 之前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含 `tf.Tensor`（一个用于嵌入输出 + 每层输出的 `tf.Tensor`），形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态，加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含 `tf.Tensor`（每个层的注意力权重），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力权重经过 SoftMax 后的结果，用于计算自注意力头部的加权平均值。
    """

    loss: tf.Tensor | None = None
    # 定义变量 start_logits，用于存储开始位置的预测张量，初始值为 None
    start_logits: tf.Tensor = None
    # 定义变量 end_logits，用于存储结束位置的预测张量，初始值为 None
    end_logits: tf.Tensor = None
    # 定义变量 hidden_states，用于存储隐藏状态的元组张量，初始值为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义变量 attentions，用于存储注意力张量的元组张量，初始值为 None
    attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    # 定义一个 loss 变量，默认为 None，用于存储总的 span 抽取损失
    loss: tf.Tensor | None = None
    # 初始化变量 `start_logits`，用于存储模型预测的起始位置的 logits
    start_logits: tf.Tensor = None
    # 初始化变量 `end_logits`，用于存储模型预测的结束位置的 logits
    end_logits: tf.Tensor = None
    # 初始化变量 `past_key_values`，用于存储模型解码器过去的键值张量列表，初始为 None
    past_key_values: List[tf.Tensor] | None = None
    # 初始化变量 `decoder_hidden_states`，用于存储解码器的隐藏状态的元组，初始为 None
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 初始化变量 `decoder_attentions`，用于存储解码器的注意力张量的元组，初始为 None
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 初始化变量 `encoder_last_hidden_state`，用于存储编码器的最后隐藏状态的张量，初始为 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 初始化变量 `encoder_hidden_states`，用于存储编码器的隐藏状态的元组，初始为 None
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 初始化变量 `encoder_attentions`，用于存储编码器的注意力张量的元组，初始为 None
    encoder_attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFSequenceClassifierOutputWithPast(ModelOutput):
    """
    用于句子分类模型输出的基础类。

    Args:
        loss (`tf.Tensor` of shape `(batch_size, )`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 config.num_labels==1）的损失值。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 config.num_labels==1）的分数（SoftMax 之前）。
        past_key_values (`List[tf.Tensor]`, *optional*, 当传递 `use_cache=True` 或 `config.use_cache=True` 时返回):
            长度为 `config.n_layers` 的 `tf.Tensor` 列表，每个张量的形状为 `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`。

            包含预先计算的隐藏状态（注意力块中的键和值），可用于加速序列解码。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组。

            模型在每一层输出的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFImageClassifierOutputWithNoAttention(ModelOutput):
    """
    用于图像分类模型输出的基础类。

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, 当提供 `labels` 时返回):
            分类（或回归，如果 config.num_labels==1）的损失值。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 config.num_labels==1）的分数（SoftMax 之前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, num_channels, height, width)` 的 `tf.Tensor` 元组。

            模型在每个阶段输出的隐藏状态（也称为特征图）。
    """
    # 定义变量 `loss`，其类型为 `tf.Tensor` 或者 `None`，初始值为 `None`
    loss: tf.Tensor | None = None
    # 定义变量 `logits`，其类型为 `tf.Tensor`，初始值为 `None`
    logits: tf.Tensor = None
    # 定义变量 `hidden_states`，其类型为 `Optional`，包含一个元组，元组中的每个元素为 `tf.Tensor` 对象，初始值为 `None`
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
@dataclass
class TFMaskedImageModelingOutput(ModelOutput):
    """
    Base class for outputs of masked image completion / in-painting models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss.
        reconstruction (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed / completed images.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called
            feature maps) of the model at the output of each stage.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # loss 属性，表示重建损失（如果提供了 bool_masked_pos 参数）
    loss: tf.Tensor | None = None
    # reconstruction 属性，表示重建或完成的图像数据
    reconstruction: tf.Tensor = None
    # hidden_states 属性，表示隐藏状态，如果模型有嵌入层，则为嵌入输出以及每个阶段的输出
    hidden_states: Tuple[tf.Tensor] | None = None
    # attentions 属性，表示注意力权重，用于计算自注意力头中的加权平均值
    attentions: Tuple[tf.Tensor] | None = None

    @property
    def logits(self):
        # 提醒警告：logits 属性在 Transformers 版本 5 中将被移除，请使用 reconstruction 属性获取最终输出
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回 reconstruction 属性作为输出
        return self.reconstruction
```