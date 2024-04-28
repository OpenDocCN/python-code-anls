# `.\transformers\modeling_tf_outputs.py`

```py
# 导入警告模块
import warnings
# 导入数据类模块
from dataclasses import dataclass
# 导入类型提示模块
from typing import List, Optional, Tuple
# 导入 TensorFlow 模块
import tensorflow as tf
# 导入自定义的模型输出工具类
from .utils import ModelOutput

# 定义 TFBaseModelOutput 类，继承自 ModelOutput 类
@dataclass
class TFBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义最后一层模型的隐藏状态
    last_hidden_state: tf.Tensor = None
    # 定义隐藏状态的元组，包含每一层的输出
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义注意力权重的元组，用于计算自注意力头中的加权平均值
    attentions: Tuple[tf.Tensor] | None = None

# 定义 TFBaseModelOutputWithNoAttention 类，继承自 ModelOutput 类
@dataclass
class TFBaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`tf.Tensor` shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    # 定义最后一层模型的隐藏状态
    last_hidden_state: tf.Tensor = None
    # 定义变量 hidden_states，类型为 Optional[Tuple[tf.Tensor, ...]]，初始值为 None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
# 定义一个带有池化的模型输出基类，继承自 ModelOutput 类
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

    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


# 定义一个带有池化但不包含注意力的模型输出基类，继承自 ModelOutput 类
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
    # 定义变量 `last_hidden_state`，类型为 TensorFlow 张量，初始值为 None
    last_hidden_state: tf.Tensor = None
    # 定义变量 `pooler_output`，类型为 TensorFlow 张量，初始值为 None
    pooler_output: tf.Tensor = None
    # 定义变量 `hidden_states`，类型为可选的元组，包含多个 TensorFlow 张量，初始值为 None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
# 使用 dataclass 装饰器定义 TFBaseModelOutputWithPoolingAndCrossAttentions 类，该类继承自 ModelOutput 类
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

    # 定义类属性，表示最后一层隐藏状态的张量
    last_hidden_state: tf.Tensor = None
    # 定义类属性，表示池化后的输出张量
    pooler_output: tf.Tensor = None
    # 定义类属性，表示过去的键值对列表
    past_key_values: List[tf.Tensor] | None = None
    # 定义类属性，表示隐藏状态的元组
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义类属性，表示注意力权重的元组
    attentions: Tuple[tf.Tensor] | None = None
    # 定义一个变量 cross_attentions，类型为 Tuple[tf.Tensor] 或 None，默认为 None
    cross_attentions: Tuple[tf.Tensor] | None = None
# 定义一个带有过去键/值的模型输出的基类，用于可能包含过去键/值以加快顺序解码的模型输出
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
    # 定义最后一个隐藏状态
    last_hidden_state: tf.Tensor = None
    # 定义过去键/值，用于加速顺序解码
    past_key_values: List[tf.Tensor] | None = None
    # 定义隐藏状态，包括每一层的输出以及初始嵌入输出
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义注意力权重，用于计算自注意力头中的加权平均值
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFBaseModelOutputWithCrossAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    # 定义函数的参数和返回值类型及其说明
    
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出隐藏状态序列。
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组形式的隐藏状态序列，每层一个 `tf.Tensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
    
            每层模型的隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组形式的注意力权重序列，每层一个 `tf.Tensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    
            经过注意力 softmax 计算后的注意力权重，用于在自注意力头中计算加权平均值。
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组形式的交叉注意力权重序列，每层一个 `tf.Tensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    
            解码器的交叉注意力层的注意力权重，经过注意力 softmax 计算后，用于计算交叉注意力头的加权平均值。
    """
    
    # 定义函数的参数类型及默认值
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    cross_attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
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
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
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

    last_hidden_state: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    cross_attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    last_hidden_state: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 定义一个变量 cross_attentions，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 定义一个变量 encoder_last_hidden_state，类型为 tf.Tensor 或 None，初始值为 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 定义一个变量 encoder_hidden_states，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 定义一个变量 encoder_attentions，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    encoder_attentions: Tuple[tf.Tensor] | None = None
# 用于定义包含有关因果语言模型（或自回归模型）输出的基类
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

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


# 用于定义包含有关带过去状态的因果语言模型（或自回归模型）输出的基类
@dataclass
class TFCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """
    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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

    # 初始化变量，用于存储不同类型的模型输出
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# 定义一个带有交叉注意力的因果语言模型输出的基类
@dataclass
class TFCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
            语言建模损失（用于下一个标记的预测），形状为`(n,)`的`tf.Tensor`，当提供`labels`时返回。

        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言模型头部的预测得分（SoftMax之前每个词汇标记的得分），形状为`(batch_size, sequence_length, config.vocab_size)`的`tf.Tensor`。

        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            每个层输出的隐藏状态以及初始嵌入输出的元组，形状为`(batch_size, sequence_length, hidden_size)`的`tf.Tensor`组成。

        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值，形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组。

        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
            解码器交叉注意力层的注意力权重，在注意力 softmax 后，用于计算交叉注意力头中的加权平均值，形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组。

        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
            包含预先计算的隐藏状态（注意力块中的键和值）的`tf.Tensor`列表，可以用于加速顺序解码，长度为`config.n_layers`，每个张量形状为`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`。

    """
    # 损失
    loss: tf.Tensor | None = None
    # 预测的标记得分
    logits: tf.Tensor = None
    # 预先计算的键值对
    past_key_values: List[tf.Tensor] | None = None
    # 隐藏状态
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力
    attentions: Tuple[tf.Tensor] | None = None
    # 交叉注意力
    cross_attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    基于掩码的语言模型输出的基类。
    """
    # 定义参数说明
    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
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

    # 初始化变量，默认为 None
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# 定义一个基类，用于存储序列到序列语言模型的输出结果
@dataclass
class TFSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，用于存储模型的损失
    loss: tf.Tensor | None = None
    # 预测的logits值
    logits: tf.Tensor = None
    # 存储过去的键值对
    past_key_values: List[tf.Tensor] | None = None
    # 解码器的隐藏状态
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 解码器的注意力权重
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 交叉注意力权重
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 编码器的最后隐藏状态
    encoder_last_hidden_state: tf.Tensor | None = None
    # 编码器的隐藏状态
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 编码器的注意力权重
    encoder_attentions: Tuple[tf.Tensor] | None = None


# 定义一个基类，用于存储预测两个句子是否连续的模型输出结果
@dataclass
class TFNextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `next_sentence_label` is provided):
            Next sentence prediction loss.
        logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
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

    # 损失值，用于存储模型的损失
    loss: tf.Tensor | None = None
    # 预测的logits值
    logits: tf.Tensor = None
    # 隐藏��态
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重
    attentions: Tuple[tf.Tensor] | None = None


# 定义一个基类，用于存储句子分类模型的输出结果
@dataclass
class TFSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`tf.Tensor` of shape `(batch_size, )`, *optional*, returned when `labels` is provided):
            分类（或当 `config.num_labels==1` 时为回归）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或当 `config.num_labels==1` 时为回归）得分（SoftMax 之前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组
            每一层的输出隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组
            自注意力中使用的注意力权重 softmax 后的结果，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
from dataclasses import dataclass
from typing import List, Tuple
import tensorflow as tf
from transformers.modeling_tf_outputs import ModelOutput

@dataclass
class TFSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    """

    # 损失值，如果存在的话
    loss: tf.Tensor | None = None
    # 输出的逻辑值
    logits: tf.Tensor = None
    # 编码器中每一层的键值对
    past_key_values: List[tf.Tensor] | None = None
    # 解码器的隐藏状态
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 解码器的注意力权重
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 交叉注意力权重
    cross_attentions: Tuple[tf.Tensor] | None = None
    # 编码器最后一个隐藏状态
    encoder_last_hidden_state: tf.Tensor | None = None
    # 编码器的隐藏状态
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 编码器的注意力权重
    encoder_attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFSemanticSegmenterOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 损失值，如果存在的话
    loss: tf.Tensor | None = None
    # 输出的逻辑值
    logits: tf.Tensor = None
    # 每一层的隐藏状态
    hidden_states: Tuple[tf.Tensor] | None = None
    # 每一层的注意力权重
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFSemanticSegmenterOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of semantic segmentation models that do not output attention scores.
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果config.num_labels==1，则是回归）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            每个像素的分类分数。

            <Tip warning={true}>
            返回的logits形状不一定与输入的`pixel_values`相同。这是为了避免在用户需要将logits调整到原始图像大小时进行两次插值并丢失一些质量。您应该始终检查logits的形状并根据需要调整大小。
            </Tip>

        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为`(batch_size, patch_size, hidden_size)`的`tf.Tensor`元组（如果模型有嵌入层，则为嵌入层的输出 + 每个层的输出）。

            每层模型的隐藏状态加上可选的初始嵌入输出。
    """

    loss: tf.Tensor | None = None  # 损失值，默认为None
    logits: tf.Tensor = None  # logits，默认为None
    hidden_states: Tuple[tf.Tensor] | None = None  # 隐藏状态，默认为None
# 使用装饰器 @dataclass 声明 TFImageClassifierOutput 类，用于表示图像分类模型的输出
@dataclass
class TFImageClassifierOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 config.num_labels==1）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 config.num_labels==1）得分（SoftMax 前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组 `tf.Tensor`（其中一个为嵌入层的输出，如果模型有嵌入层，+ 每个阶段的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每个阶段输出的隐藏状态（也称为特征图）。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组 `tf.Tensor`（每个层一个）的形状为 `(batch_size, num_heads, patch_size, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


# 使用装饰器 @dataclass 声明 TFMultipleChoiceModelOutput 类，用于表示多选题模型的输出
@dataclass
class TFMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (`tf.Tensor` of shape *(batch_size, )*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* 是输入张量的第二维度。（参见上面的 *input_ids*）。

            分类得分（SoftMax 前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组 `tf.Tensor`（一个为嵌入层的输出 + 每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组 `tf.Tensor`（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# 导入必要的库
from dataclasses import dataclass
from typing import Tuple
import tensorflow as tf

# 创建一个数据类，用于表示基于 Token 的分类模型的输出
@dataclass
class TFTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of unmasked labels, returned when `labels` is provided) :
            Classification loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
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

    # 分类损失
    loss: tf.Tensor | None = None
    # 分类得分
    logits: tf.Tensor = None
    # 隐藏状态
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重
    attentions: Tuple[tf.Tensor] | None = None

# 创建一个数据类，用于表示基于问题回答模型的输出
@dataclass
class TFQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`tf.Tensor` of shape `(batch_size, )`, *optional*, returned when `start_positions` and `end_positions` are provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
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

    # 范围抽取损失
    loss: tf.Tensor | None = None
    # 起始位置得分
    start_logits: tf.Tensor = None
    # 结束位置得分
    end_logits: tf.Tensor = None
    # 隐藏状态
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重
    attentions: Tuple[tf.Tensor] | None = None
    # 定义变量 start_logits，类型为 tf.Tensor，初始值为 None
    start_logits: tf.Tensor = None
    # 定义变量 end_logits，类型为 tf.Tensor，初始值为 None
    end_logits: tf.Tensor = None
    # 定义变量 hidden_states，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义变量 attentions，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    attentions: Tuple[tf.Tensor] | None = None
# 使用 dataclass 装饰器定义 TFSeq2SeqQuestionAnsweringModelOutput 类，用于表示序列到序列问答模型的输出
class TFSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.
    # 序列到序列问答模型输出的基类
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

    # 定义 loss 变量，类型为 tf.Tensor 或 None，默认为 None
    loss: tf.Tensor | None = None
    # 初始化变量 start_logits，用于存储模型生成的开始位置的概率分布
    start_logits: tf.Tensor = None
    # 初始化变量 end_logits，用于存储模型生成的结束位置的概率分布
    end_logits: tf.Tensor = None
    # 初始化变量 past_key_values，用于存储模型解码器的过去键值状态（历史信息），初始值为 None 或空列表
    past_key_values: List[tf.Tensor] | None = None
    # 初始化变量 decoder_hidden_states，用于存储解码器的隐藏状态，初始值为 None 或空元组
    decoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 初始化变量 decoder_attentions，用于存储解码器的注意力权重，初始值为 None 或空元组
    decoder_attentions: Tuple[tf.Tensor] | None = None
    # 初始化变量 encoder_last_hidden_state，用于存储编码器的最后一个隐藏状态，初始值为 None
    encoder_last_hidden_state: tf.Tensor | None = None
    # 初始化变量 encoder_hidden_states，用于存储编码器的所有隐藏状态，初始值为 None 或空元组
    encoder_hidden_states: Tuple[tf.Tensor] | None = None
    # 初始化变量 encoder_attentions，用于存储编码器的注意力权重，初始值为 None 或空元组
    encoder_attentions: Tuple[tf.Tensor] | None = None
# 带有过去键值的 TF 序列分类器输出的基类，继承自模型输出类
@dataclass
class TFSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`tf.Tensor` of shape `(batch_size, )`, *optional*, returned when `labels` is provided):
            分类（或如果 config.num_labels==1 则为回归）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或如果 config.num_labels==1 则为回归）得分（SoftMax 之前）。
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            长度为 `config.n_layers` 的 `tf.Tensor` 列表，每个张量的形状为 `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`。

            包含预先计算的隐藏状态（注意力块中的键和值），可用于加速顺序解码。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组（一个用于嵌入的输出，每一层的输出一个）。

            每一层模型的隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组（每一层一个）。

            注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None  # 分类（或回归）损失
    logits: tf.Tensor = None  # 分类（或回归）得分（SoftMax 之前）
    past_key_values: List[tf.Tensor] | None = None  # 预先计算的隐藏状态
    hidden_states: Tuple[tf.Tensor] | None = None  # 每一层的隐藏状态
    attentions: Tuple[tf.Tensor] | None = None  # 注意力权重



# 不带注意力的 TF 图像分类器输出的基类，继承自模型输出类
@dataclass
class TFImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或如果 config.num_labels==1 则为回归）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或如果 config.num_labels==1 则为回归）得分（SoftMax 之前）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, num_channels, height, width)` 的 `tf.Tensor` 元组（如果模型有嵌入层，则有一个用于嵌入的输出，每个阶段的输出一个）。

            模型每个阶段的隐藏状态（也称为特征图）。
    """
    # 定义损失值，类型为 TensorFlow 的张量，初始值为 None
    loss: tf.Tensor | None = None
    # 定义对数，类型为 TensorFlow 的张量，初始值为 None
    logits: tf.Tensor = None
    # 定义隐藏状态，类型为可选的元组，元素为 TensorFlow 的张量，初始值为 None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
from dataclasses import dataclass
import tensorflow as tf
from typing import Tuple
import warnings
from transformers.modeling_tf_outputs import ModelOutput

# TFMaskedImageModelingOutput 类，用于表示掩码图像完成/修复模型的输出结果
@dataclass
class TFMaskedImageModelingOutput(ModelOutput):
    """
    Base class for outputs of masked image completion / in-painting models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss. 重建损失
        reconstruction (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed / completed images. 重建/完成的图像
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called
            feature maps) of the model at the output of each stage.
            模型每个阶段的隐藏状态(也称为特征图)的元组
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            在注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均

    """

    # 重建损失，默认为 None
    loss: tf.Tensor | None = None
    # 重建/完成的图像，默认为 None
    reconstruction: tf.Tensor = None
    # 隐藏状态，默认为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力，默认为 None
    attentions: Tuple[tf.Tensor] | None = None

    # logits 属性的 getter 方法
    @property
    def logits(self):
        # 警告：logits 属性已弃用，并将在 Transformers 的版本 5 中移除。请使用 reconstruction 属性来获取最终输出。
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回 reconstruction 属性
        return self.reconstruction
```