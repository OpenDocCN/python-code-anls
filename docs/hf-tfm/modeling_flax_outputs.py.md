# `.\transformers\modeling_flax_outputs.py`

```
# 导入必要的模块和类型
from typing import Dict, Optional, Tuple
# 导入 Flax 框架
import flax
# 导入 JAX 中的 NumPy 模块并简称为 jnp
import jax.numpy as jnp
# 从当前目录下的 utils 模块中导入 ModelOutput 类
from .utils import ModelOutput

# 使用 flax 的装饰器定义一个数据类，表示模型输出，包含潜在的隐藏状态和注意力
@flax.struct.dataclass
class FlaxBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    # 最后一层模型输出的隐藏状态
    last_hidden_state: jnp.ndarray = None
    # 模型输出的隐藏状态
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 注意力权重
    attentions: Optional[Tuple[jnp.ndarray]] = None

# 使用 flax 的装饰器定义一个数据类，表示模型输出，只包含潜在的隐藏状态
@flax.struct.dataclass
class FlaxBaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """
    # 最后一层模型输出的隐藏状态
    last_hidden_state: jnp.ndarray = None
    # 定义一个可选的元组类型变量 hidden_states，初始值为 None
# 定义一个带有池化和无注意力的模型输出基类
@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None


# 定义一个不带注意力的图像分类器输出基类
@flax.struct.dataclass
class FlaxImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None


# 定义一个带有过去信息的模型输出基类，可能包含隐藏状态和注意力
@flax.struct.dataclass
class FlaxBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`Dict[str, jnp.ndarray]`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义函数参数及其类型注释
    last_hidden_state: jnp.ndarray = None
    past_key_values: Optional[Dict[str, jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 导入Flax库中的`struct`模块
@flax.struct.dataclass
# 定义一个带有池化的模型输出基类，继承自`ModelOutput`
class FlaxBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    模型输出的基类，还包含最后隐藏状态的池化。

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层的输出的隐藏状态序列。
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
            序列中第一个标记（分类标记）的最后一层隐藏状态，进一步由线性层和Tanh激活函数处理。在预训练期间，线性层的权重是根据下一句预测（分类）目标进行训练的。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每一层输出的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            在注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 导入Flax库中的`struct`模块
@flax.struct.dataclass
# 定义一个带有池化和交叉注意力的模型输出基类，继承自`ModelOutput`
class FlaxBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    模型输出的基类，还包含最后隐藏状态的池化。

    """
    # 参数：last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
    # 模型最后一层的隐藏状态序列，形状为`(batch_size, sequence_length, hidden_size)`
    last_hidden_state: jnp.ndarray = None
    # 定义一个变量pooler_output，类型为jnp.ndarray，初始值为None，用于存储pooler层的输出
    pooler_output: jnp.ndarray = None
    # 定义一个变量hidden_states，类型为Optional[Tuple[jnp.ndarray]]，初始值为None，用于存储隐藏状态
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义一个变量past_key_values，类型为Optional[Tuple[Tuple[jnp.ndarray]]]，初始值为None，用于存储过去的键值
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义一个变量attentions，类型为Optional[Tuple[jnp.ndarray]]，初始值为None，用于存储注意力权重
    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义一个变量cross_attentions，类型为Optional[Tuple[jnp.ndarray]]，初始值为None，用于存储交叉注意力权重
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
# 使用 @flax.struct.dataclass 装饰器创建一个数据类，用于存储模型输出的基本信息以及过去的关键/值（用于加速顺序解码）
@flax.struct.dataclass
class FlaxBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    # 定义最后一层模型输出的隐藏状态
    last_hidden_state: jnp.ndarray = None
    # 定义过去的关键/值，用于加速顺序解码
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义模型在每一层输出的隐藏状态
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义自注意力头中注意力权重
    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义交叉注意力头中注意力权重
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义一个基础类，用于存储模型编码器的输出，同时包含预先计算的隐藏状态，以加快顺序解码的速度
@flax.struct.dataclass
class FlaxSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    # 最后一个隐藏状态，用于存储模型的最后一个隐藏状态
    last_hidden_state: jnp.ndarray = None
    # 预先计算的键值对，用于加速顺序解码
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 解码器的隐藏状态，存储模型解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 解码器的注意力权重，存储模型解码器的注意力权重
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 交叉注意力权重，存储模型交叉注意力的权重
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 编码器的最后一个隐藏状态，存储模型编码器的最后一个隐藏状态
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    # 编码器的隐藏状态，存储模型编码器的隐藏状态
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 编码器的注意力权重，存储模型编码器的注意力权重
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义一个基础类，用于存储因果语言模型（或自回归模型）的输出
@flax.struct.dataclass
class FlaxCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `jnp.ndarray` tuples of length `config.n_layers`, with each tuple containing the cached key, value
            states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting.
            Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """
    # 定义 logits 变量，用于存储模型输出的 logits，初始化为 None
    logits: jnp.ndarray = None
    # 定义 past_key_values 变量，用于存储模型解码过程中的历史键值对，初始化为 None，可选类型为 Tuple，元素类型为 Tuple[jnp.ndarray]
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义 hidden_states 变量，用于存储模型的隐藏状态，初始化为 None，可选类型为 Tuple，元素类型为 jnp.ndarray
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义 attentions 变量，用于存储模型的注意力权重，初始化为 None，可选类型为 Tuple，元素类型为 jnp.ndarray
    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义 cross_attentions 变量，用于存储模型的交叉注意力权重，初始化为 None，可选类型为 Tuple，元素类型为 jnp.ndarray
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义一个数据类，用于表示遮蔽语言模型的输出
@flax.struct.dataclass
class FlaxMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None

# 将 FlaxCausalLMOutput 定义为 FlaxMaskedLMOutput
FlaxCausalLMOutput = FlaxMaskedLMOutput

# 定义一个数据类，用于表示序列到序列语言模型的输出
@flax.struct.dataclass
class FlaxSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    logits: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None

# 定义一个数据类，用于表示预测两个句子是否连续的模型输出
@flax.struct.dataclass
class FlaxNextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    Args:
        logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义变量logits，类型为jnp.ndarray，默认值为None
    logits: jnp.ndarray = None
    # 定义变量hidden_states，类型为Tuple[jnp.ndarray]，可选参数，当传入output_hidden_states=True或config.output_hidden_states=True时返回
    # Tuple包含jnp.ndarray（一个用于嵌入输出+一个用于每个层的输出），形状为(batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量attentions，类型为Tuple[jnp.ndarray]，可选参数，当传入output_attentions=True或config.output_attentions=True时返回
    # Tuple包含jnp.ndarray（每个层一个），形状为(batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 使用 flax.struct.dataclass 装饰器定义 FlaxSequenceClassifierOutput 类，该类是句子分类模型输出的基类
class FlaxSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义 logits 属性，类型为 jnp.ndarray，默认值为 None，用于存储分类得分（SoftMax 之前）
    logits: jnp.ndarray = None
    # 定义 hidden_states 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储模型在每一层的隐藏状态和初始嵌入输出
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义 attentions 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储注意力权重，用于计算自注意力头中的加权平均值
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 flax.struct.dataclass 装饰器定义 FlaxSeq2SeqSequenceClassifierOutput 类，该类是序列到序列句子分类模型输出的基类
class FlaxSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    """

    # 定义 logits 属性，类型为 jnp.ndarray，默认值为 None，用于存储分类得分
    logits: jnp.ndarray = None
    # 定义 past_key_values 属性，类型为可选的元组，其中元素类型为元组的元组，元组内元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储序列到序列模型的过去键值
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义 decoder_hidden_states 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义 decoder_attentions 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储解码器的注意力权重
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义 cross_attentions 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储交叉注意力权重
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义 encoder_last_hidden_state 属性，类型为可选的 jnp.ndarray，默认值为 None，
    # 用于存储编码器的最后一个隐藏状态
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    # 定义 encoder_hidden_states 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储编码器的隐藏状态
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义 encoder_attentions 属性，类型为可选的元组，其中元素类型为 jnp.ndarray，默认值为 None，
    # 用于存储编码器的注意力权重
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 flax.struct.dataclass 装饰器定义 FlaxMultipleChoiceModelOutput 类，该类是多选模型输出的基类
class FlaxMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.
    Args:
        logits (`jnp.ndarray` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).
            Classification scores (before SoftMax).

        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义变量 logits，表示分类的得分，形状为 (batch_size, num_choices)
    logits: jnp.ndarray = None
    # 定义变量 hidden_states，表示模型每一层的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量 attentions，表示模型每一层的注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义一个数据类，用于表示标记分类模型的输出
@flax.struct.dataclass
class FlaxTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义一个数据类，用于表示问答模型的输出
@flax.struct.dataclass
class FlaxQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: jnp.ndarray = None
    end_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义一个数据类，用于表示序列到序列问答模型的输出
@flax.struct.dataclass
class FlaxSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    """

    start_logits: jnp.ndarray = None
    end_logits: jnp.ndarray = None
    # 初始化变量 past_key_values，用于存储解码器的过去键值对
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 初始化变量 decoder_hidden_states，用于存储解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量 decoder_attentions，用于存储解码器的注意力权重
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量 cross_attentions，用于存储交叉注意力权重
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量 encoder_last_hidden_state，用于存储编码器的最后隐藏状态
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    # 初始化变量 encoder_hidden_states，用于存储编码器的隐藏状态
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量 encoder_attentions，用于存储编码器的注意力权重
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None
```