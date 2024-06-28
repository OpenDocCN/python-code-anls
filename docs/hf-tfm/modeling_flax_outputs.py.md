# `.\modeling_flax_outputs.py`

```py
# 导入必要的模块和类
from typing import Dict, Optional, Tuple  # 导入类型提示相关模块

import flax  # 导入Flax库，用于结构化数据类
import jax.numpy as jnp  # 导入JAX的NumPy接口

from .utils import ModelOutput  # 从当前目录下的utils模块导入ModelOutput类


@flax.struct.dataclass
class FlaxBaseModelOutput(ModelOutput):
    """
    模型输出的基础类，包含可能的隐藏状态和注意力机制。

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列输出。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, 当 `output_hidden_states=True` 被传递或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `jnp.ndarray` 元组。

            模型每一层的隐藏状态加上初始嵌入输出。
        attentions (`tuple(jnp.ndarray)`, *optional*, 当 `output_attentions=True` 被传递或 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `jnp.ndarray` 元组。

            注意力机制softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithNoAttention(ModelOutput):
    """
    模型输出的基础类，包含可能的隐藏状态，但不包含注意力机制。

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            模型最后一层的隐藏状态序列输出。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, 当 `output_hidden_states=True` 被传递或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, num_channels, height, width)` 的 `jnp.ndarray` 元组。

            模型每一层的隐藏状态加上可选的初始嵌入输出。
    """

    last_hidden_state: jnp.ndarray = None
    # 定义一个可选的变量 hidden_states，类型为包含 jnp.ndarray 的元组，初始值为 None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
# 使用 @flax.struct.dataclass 装饰器声明一个数据类，表示带有池化和无注意力机制的模型输出
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

    # 声明类的属性及其类型注解
    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None


# 使用 @flax.struct.dataclass 装饰器声明一个数据类，表示不带注意力机制的图像分类模型输出
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

    # 声明类的属性及其类型注解
    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None


# 使用 @flax.struct.dataclass 装饰器声明一个数据类，表示带有过去状态的模型输出
@flax.struct.dataclass
class FlaxBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """
    # `last_hidden_state` 是模型最后一层的隐藏状态序列，形状为 `(batch_size, sequence_length, hidden_size)`
    # 这里使用了 JAX 数组 `jnp.ndarray`，表示 JAX 程序的数组结构
    last_hidden_state: jnp.ndarray = None

    # `past_key_values` 是一个字典，包含预先计算的隐藏状态（在注意力块中的键和值），用于快速自回归解码
    # 键和值的隐藏状态的形状为 `[batch_size, max_length]`
    past_key_values: Optional[Dict[str, jnp.ndarray]] = None

    # `hidden_states` 是一个元组，包含了模型每一层的隐藏状态
    # 第一个元素是嵌入层的输出，后续元素是每一层的输出，形状为 `(batch_size, sequence_length, hidden_size)`
    # 只有在传递参数 `output_hidden_states=True` 或者配置 `config.output_hidden_states=True` 时才返回
    hidden_states: Optional[Tuple[jnp.ndarray]] = None

    # `attentions` 是一个元组，包含了每一层的注意力权重
    # 每个元素是一个形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 JAX 数组
    # 只有在传递参数 `output_attentions=True` 或者配置 `config.output_attentions=True` 时才返回
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 使用 `flax.struct.dataclass` 装饰器定义一个数据类，该类继承自 `ModelOutput` 类，用于表示模型输出并包含最后隐藏状态的池化结果。
@flax.struct.dataclass
class FlaxBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
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

    # 定义类的属性，表示模型输出中的最后隐藏状态、池化输出、隐藏状态以及注意力权重
    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 `flax.struct.dataclass` 装饰器定义另一个数据类，表示模型输出并包含最后隐藏状态的池化结果以及交叉注意力权重。
@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    """

    # 该类继承自 `ModelOutput` 类，与 `FlaxBaseModelOutputWithPooling` 类似，但这里还包括交叉注意力权重的定义。
        Args:
            last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token) after further processing
                through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
                the classification token after processing through a linear layer and a tanh activation function. The linear
                layer weights are trained from the next sentence prediction (classification) objective during pretraining.
            hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
                for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
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
            past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
                `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
                encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
                `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
                input) to speed up sequential decoding.
        """

        last_hidden_state: jnp.ndarray = None
    # 定义一个变量 `pooler_output`，类型为 `jnp.ndarray`，初始值为 None
    pooler_output: jnp.ndarray = None
    # 定义一个变量 `hidden_states`，类型为 `Optional[Tuple[jnp.ndarray]]`，初始值为 None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义一个变量 `past_key_values`，类型为 `Optional[Tuple[Tuple[jnp.ndarray]]]`，初始值为 None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义一个变量 `attentions`，类型为 `Optional[Tuple[jnp.ndarray]]`，初始值为 None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义一个变量 `cross_attentions`，类型为 `Optional[Tuple[jnp.ndarray]]`，初始值为 None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
# 使用 @flax.struct.dataclass 装饰器声明一个数据类，该类继承自 ModelOutput 类
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

    # 定义类的属性，每个属性都有一个默认值为 None
    last_hidden_state: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义基于Flax的数据类，表示序列到序列模型的输出，继承自ModelOutput
@flax.struct.dataclass
class FlaxSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains pre-computed hidden states that can speed up sequential decoding.
    """

    # 最后一个隐藏状态，类型为jnp.ndarray，默认为None
    last_hidden_state: jnp.ndarray = None
    # 过去的键值对，类型为可选的元组，包含元组的元组，每个元组包含jnp.ndarray，默认为None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 解码器的隐藏状态，类型为可选的元组，包含jnp.ndarray，默认为None
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 解码器的注意力权重，类型为可选的元组，包含jnp.ndarray，默认为None
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 交叉注意力的权重，类型为可选的元组，包含jnp.ndarray，默认为None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 编码器最后一个隐藏状态，类型为可选的jnp.ndarray，默认为None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    # 编码器的隐藏状态，类型为可选的元组，包含jnp.ndarray，默认为None
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 编码器的注意力权重，类型为可选的元组，包含jnp.ndarray，默认为None
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义基于Flax的数据类，表示带有交叉注意力的因果语言模型输出，继承自ModelOutput
@flax.struct.dataclass
class FlaxCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """

    # 预测的logits，形状为(batch_size, sequence_length, config.vocab_size)的jnp.ndarray
    logits: jnp.ndarray
    # 隐藏状态的元组，包含embedding输出和每层输出的jnp.ndarray，形状为(batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 注意力权重的元组，每层一个jnp.ndarray，形状为(batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 交叉注意力权重的元组，每层一个jnp.ndarray，形状为(batch_size, num_heads, sequence_length, sequence_length)
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 过去的键值对的元组，每层一个jnp.ndarray元组，长度为config.n_layers，仅在使用缓存时有效，用于编码器-解码器设置
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义变量 logits，用于存储一个 NumPy 数组，初始值为 None
    logits: jnp.ndarray = None
    # 定义变量 past_key_values，类型为 Optional[Tuple[Tuple[jnp.ndarray]]]，可选的三重嵌套元组结构，初始值为 None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 定义变量 hidden_states，类型为 Optional[Tuple[jnp.ndarray]]，可选的元组结构，初始值为 None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量 attentions，类型为 Optional[Tuple[jnp.ndarray]]，可选的元组结构，初始值为 None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量 cross_attentions，类型为 Optional[Tuple[jnp.ndarray]]，可选的元组结构，初始值为 None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
@flax.struct.dataclass
class FlaxMaskedLMOutput(ModelOutput):
    """
    Masked语言模型输出的基类。

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `jnp.ndarray` 元组。

            模型在每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(jnp.ndarray)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `jnp.ndarray` 元组。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


FlaxCausalLMOutput = FlaxMaskedLMOutput


@flax.struct.dataclass
class FlaxSeq2SeqLMOutput(ModelOutput):
    """
    序列到序列语言模型输出的基类。

    """

    logits: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxNextSentencePredictorOutput(ModelOutput):
    """
    预测两个句子是否连续的模型输出的基类。

    """
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
    # 定义函数参数及其类型注释，描述函数接受的输入参数及其形状和类型信息

    logits: jnp.ndarray = None
    # 定义变量 logits，用于存储形状为(batch_size, 2)的 jnp.ndarray，表示下一个序列预测分类头部的预测分数（经过 SoftMax 之前的 True/False 连续性得分）。

    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量 hidden_states，可选的元组类型，包含 jnp.ndarray（当传递 output_hidden_states=True 或 config.output_hidden_states=True 时返回）。
    # 元组中的每个数组形状为(batch_size, sequence_length, hidden_size)，表示模型在每一层输出的隐藏状态以及初始嵌入输出。

    attentions: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量 attentions，可选的元组类型，包含 jnp.ndarray（当传递 output_attentions=True 或 config.output_attentions=True 时返回）。
    # 元组中的每个数组形状为(batch_size, num_heads, sequence_length, sequence_length)，
    # 表示注意力 softmax 后的注意力权重，用于计算自注意力头部中的加权平均值。
# 使用 @flax.struct.dataclass 装饰器声明一个数据类，用于表示序列分类模型的输出。
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
    
    # 分类或回归得分（未经 SoftMax 处理前）的 logits，形状为 `(batch_size, config.num_labels)`
    logits: jnp.ndarray = None
    # 模型每一层的输出的隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`，当 `output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 自注意力机制注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，当 `output_attentions=True` 时返回
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 @flax.struct.dataclass 装饰器声明一个数据类，用于表示序列到序列的句子分类模型的输出。
class FlaxSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    """
    
    # 分类得分 logits，形状为 `(batch_size, config.num_labels)`
    logits: jnp.ndarray = None
    # 用于存储过去键值的元组，形状为 `(batch_size, num_layers, 2, batch_size, num_heads, sequence_length, head_dim)`，当 `output_past_key_values=True` 时返回
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 解码器每一层的隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`，当 `output_hidden_states=True` 时返回
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 解码器每一层的注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，当 `output_attentions=True` 时返回
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 编码器解码器之间注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，当 `output_attentions=True` 时返回
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 编码器最后一层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    # 编码器每一层的隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`，当 `output_hidden_states=True` 时返回
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 编码器每一层的注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，当 `output_attentions=True` 时返回
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 @flax.struct.dataclass 装饰器声明一个数据类，用于表示多项选择模型的输出。
class FlaxMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.
    """
    
    # 此类暂未定义任何属性，因此无需添加额外的注释。
    """
    Args:
        logits (`jnp.ndarray` of shape `(batch_size, num_choices)`):
            分类器的输出分数（SoftMax 之前）。

        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含每层输出的元组，每个 `jnp.ndarray` 的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每一层的隐藏状态以及初始嵌入输出。

        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含每层注意力权重的元组，每个 `jnp.ndarray` 的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 使用 @flax.struct.dataclass 装饰器定义一个数据类，表示序列标注模型的输出
@flax.struct.dataclass
class FlaxTokenClassifierOutput(ModelOutput):
    """
    序列标注模型输出的基类。

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.num_labels)`):
            分类得分（SoftMax 之前）。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            包含多个 `jnp.ndarray` 的元组，形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(jnp.ndarray)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            包含多个 `jnp.ndarray` 的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 @flax.struct.dataclass 装饰器定义一个数据类，表示问答模型的输出
@flax.struct.dataclass
class FlaxQuestionAnsweringModelOutput(ModelOutput):
    """
    问答模型输出的基类。

    Args:
        start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            起始位置的得分（SoftMax 之前）。
        end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            终止位置的得分（SoftMax 之前）。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            包含多个 `jnp.ndarray` 的元组，形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(jnp.ndarray)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            包含多个 `jnp.ndarray` 的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    start_logits: jnp.ndarray = None
    end_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 使用 @flax.struct.dataclass 装饰器定义一个数据类，表示序列到序列问答模型的输出
@flax.struct.dataclass
class FlaxSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    序列到序列问答模型输出的基类。

    """

    start_logits: jnp.ndarray = None
    end_logits: jnp.ndarray = None
    # 初始化变量，用于存储模型解码器的相关状态和注意力权重
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    # 初始化变量，用于存储模型解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量，用于存储模型解码器的注意力权重
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量，用于存储模型交叉注意力的权重
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量，用于存储模型编码器的最后隐藏状态
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    # 初始化变量，用于存储模型编码器的隐藏状态的序列
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 初始化变量，用于存储模型编码器的注意力权重的序列
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None
```