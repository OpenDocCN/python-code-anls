# `.\generation\tf_utils.py`

```
# 导入所需的模块和库
import copy  # 导入 copy 模块，用于复制对象
import inspect  # 导入 inspect 模块，用于获取对象信息
import warnings  # 导入 warnings 模块，用于处理警告
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Any, Dict, Optional, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 库，并使用别名 np
import tensorflow as tf  # 导入 TensorFlow 库，并使用别名 tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice  # 导入特定函数

# 从相对路径中导入模型输出类
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
# 从相对路径中导入自动模型映射字典
from ..models.auto import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
)
# 从相对路径中导入 TensorFlow 工具函数和稳定 softmax 函数
from ..tf_utils import shape_list, stable_softmax
# 从相对路径中导入模型输出类和日志记录函数
from ..utils import ModelOutput, logging
# 从相对路径中导入生成配置类
from .configuration_utils import GenerationConfig
# 从相对路径中导入 TensorFlow 日志处理相关模块
from .tf_logits_process import (
    TFForcedBOSTokenLogitsProcessor,
    TFForcedEOSTokenLogitsProcessor,
    TFForceTokensLogitsProcessor,
    TFLogitsProcessorList,
    TFMinLengthLogitsProcessor,
    TFNoBadWordsLogitsProcessor,
    TFNoRepeatNGramLogitsProcessor,
    TFRepetitionPenaltyLogitsProcessor,
    TFSuppressTokensAtBeginLogitsProcessor,
    TFSuppressTokensLogitsProcessor,
    TFTemperatureLogitsWarper,
    TFTopKLogitsWarper,
    TFTopPLogitsWarper,
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 TFGreedySearchDecoderOnlyOutput 类，继承自 ModelOutput 基类
@dataclass
class TFGreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.
    """
    pass  # 类定义结束
    # 参数列表：
    # sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
    #     生成的序列。第二维 (sequence_length) 要么等于 `max_length`，要么在所有批次由于 `eos_token_id` 而提前结束时要短。
    # scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
    #     语言建模头部的处理预测分数（SoftMax 之前的每个词汇标记的分数）在每个生成步骤。元组的 `tf.Tensor`，最多包含 `max_new_tokens` 个元素（每个生成的标记一个元素），每个张量的形状为 `(batch_size, config.vocab_size)`。
    # attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
    #     元组（每个生成的标记一个元素）的元组（解码器每一层一个元素）的 `tf.Tensor`，形状为 `(batch_size, num_heads, generated_length, sequence_length)`。
    # hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
    #     元组（每个生成的标记一个元素）的元组（解码器每一层一个元素）的 `tf.Tensor`，形状为 `(batch_size, generated_length, hidden_size)`。

    sequences: tf.Tensor = None  # 初始化 sequences 变量为 None
    scores: Optional[Tuple[tf.Tensor]] = None  # 初始化 scores 变量为 None，类型为可选的元组
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None  # 初始化 attentions 变量为 None，类型为可选的元组的元组
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None  # 初始化 hidden_states 变量为 None，类型为可选的元组的元组
@dataclass
class TFGreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
    """

    sequences: tf.Tensor = None
    # 生成的序列，形状为(batch_size, sequence_length)，第二个维度(sequence_length)要么等于max_length，要么因为eos_token_id提前结束而较短
    scores: Optional[Tuple[tf.Tensor]] = None
    # 可选项，当传入output_scores=True或config.output_scores=True时返回，是语言建模头部的处理过的预测分数（SoftMax之前每个词汇标记的分数），每个生成步骤可能有多达max_new_tokens个元素，每个张量形状为(batch_size, config.vocab_size)
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    # 可选项，当传入output_attentions=True或config.output_attentions=True时返回，元组的每个元素对应解码器每层的注意力张量，形状为(batch_size, num_heads, sequence_length, sequence_length)
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    # 可选项，当传入output_hidden_states=True或config.output_hidden_states=True时返回，元组的每个元素对应嵌入层和每层输出的隐藏状态张量，形状为(batch_size, sequence_length, hidden_size)
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 可选项，当传入output_attentions=True或config.output_attentions=True时返回，元组的每个元素对应每个生成的标记，每层解码器的注意力张量元组，形状为(batch_size, num_heads, generated_length, sequence_length)
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 可选项，当传入output_attentions=True或config.output_attentions=True时返回，元组的每个元素对应每个生成的标记，每层解码器的交叉注意力张量元组，形状为(batch_size, num_heads, generated_length, sequence_length)
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 可选项，当传入output_hidden_states=True或config.output_hidden_states=True时返回，元组的每个元素对应每个生成的标记，每层解码器的隐藏状态张量元组，形状为(batch_size, generated_length, hidden_size)
    # 定义一个可选的变量，用于存储编码器的隐藏状态（Tensor 的元组）。初始值为 None。
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    
    # 定义一个可选的变量，用于存储解码器注意力权重（Tensor 元组的元组）。初始值为 None。
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    
    # 定义一个可选的变量，用于存储交叉注意力权重（Tensor 元组的元组）。初始值为 None。
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    
    # 定义一个可选的变量，用于存储解码器的隐藏状态（Tensor 元组的元组）。初始值为 None。
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
@dataclass
class TFSampleDecoderOnlyOutput(ModelOutput):
    """
    Decoder-only生成模型使用采样生成的输出的基类。

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            生成的序列。第二个维度（sequence_length）要么等于`max_length`，要么比`eos_token_id`提前结束。
        scores (`tuple(tf.Tensor)` *optional*, 当传入`output_scores=True`或者`config.output_scores=True`时返回):
            语言建模头部的处理过的预测分数（SoftMax之前的每个词汇标记的分数）在每个生成步骤中。
            元组中包含最多`max_new_tokens`个元素（每个生成的词汇标记一个元素），每个张量的形状为`(batch_size*num_return_sequences, config.vocab_size)`。
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, 当传入`output_attentions=True`或者`config.output_attentions=True`时返回):
            每个生成的词汇标记的元组（每个生成的词汇标记一个元素），其中包含解码器每一层的注意力张量。
            注意力张量的形状为`(num_return_sequences*batch_size, num_heads, generated_length, sequence_length)`。
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, 当传入`output_hidden_states=True`或者`config.output_hidden_states=True`时返回):
            每个生成的词汇标记的元组（每个生成的词汇标记一个元素），其中包含解码器每一层的隐藏状态张量。
            隐藏状态张量的形状为`(num_return_sequences*batch_size, generated_length, hidden_size)`。
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFSampleEncoderDecoderOutput(ModelOutput):
    """
    Encoder-decoder生成模型使用采样生成的输出的基类。可以通过encoder_attentions和encoder_hidden_states属性（分别通过decoder_attentions和decoder_hidden_states属性）访问解码器（分别是编码器）的隐藏状态和注意力权重。

    """
    # 定义函数的参数和它们的类型注解，这些参数用于接收生成的序列、预测分数、编码器注意力、编码器隐藏状态、
    # 解码器注意力、交叉注意力以及解码器隐藏状态。这些参数都是可选的，根据函数调用时传递的参数决定是否使用。
    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            生成的序列。第二维（sequence_length）要么等于 `max_length`，要么因为 `eos_token_id` 导致所有批次提前结束而更短。
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            语言建模头部处理后的预测分数（在SoftMax之前的每个词汇标记的分数），每一代步骤有一个元组，包含最多 `max_new_tokens` 个元素，
            每个张量的形状为 `(batch_size*num_return_sequences, config.vocab_size)`。
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            编码器注意力的元组（每个解码器层一个张量），形状为 `(batch_size*num_return_sequences, num_heads, sequence_length, sequence_length)`。
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            编码器隐藏状态的元组（每个解码器层一个张量），形状为 `(batch_size*num_return_sequences, sequence_length, hidden_size)`。
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            解码器注意力的元组（每个生成的令牌一个元组，每个解码器层一个张量），形状为 `(batch_size*num_return_sequences, num_heads, generated_length, sequence_length)`。
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            交叉注意力的元组（每个生成的令牌一个元组，每个解码器层一个张量），形状为 `(batch_size, num_heads, generated_length, sequence_length)`。
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            解码器隐藏状态的元组（每个生成的令牌一个元组，每个解码器层一个张量），形状为 `(batch_size*num_return_sequences, generated_length, hidden_size)`。
    
    # 初始化所有参数为 None，表示这些参数在函数调用时可以不传递，或者传递为 None，函数会根据需要进行处理。
    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
# 使用 dataclass 装饰器定义 TFBeamSearchDecoderOnlyOutput 类，表示仅解码器使用 beam search 生成模型的输出。
@dataclass
class TFBeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.
    解码器仅使用 beam search 生成模型的输出的基类。

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
            生成的序列。第二维度（sequence_length）要么等于 `max_length`，要么由于 `eos_token_id` 导致所有批次提前结束而更短。
        sequences_scores (`tf.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
            生成的 `sequences` 的最终 beam 分数。
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this
            beam. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
            每一代生成步骤中每个词汇标记的处理过的 beam 分数。包括每个词汇标记的 log softmax 分数和该 beam 中先前生成的标记的 log softmax 的总和。
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `tf.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
            每个生成步骤生成的标记 id 的 beam 索引。形状为 `(batch_size*num_return_sequences, sequence_length)` 的 `tf.Tensor`。
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
            每个生成的标记的元组（每个解码器层的一个元素）的元组（每个生成的标记的元素）的注意力张量。形状为 `(batch_size*num_beams, num_heads, generated_length, sequence_length)` 的 `tf.Tensor`。
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
            每个生成的标记的元组（每个解码器层的一个元素）的元组（每个生成的标记的元素）的隐藏状态张量。形状为 `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)` 的 `tf.Tensor`。
    """

    sequences: tf.Tensor = None  # 生成的序列
    sequences_scores: Optional[tf.Tensor] = None  # 生成序列的最终 beam 分数，可选
    scores: Optional[Tuple[tf.Tensor]] = None  # 每个生成步骤中每个词汇标记的处理过的 beam 分数，可选
    beam_indices: Optional[tf.Tensor] = None  # 每个生成步骤生成的标记 id 的 beam 索引，可选
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None  # 每个生成的标记的注意力张量，可选
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None  # 每个生成的标记的隐藏状态张量，可选


# 使用 dataclass 装饰器定义 TFBeamSearchEncoderDecoderOutput 类，表示编码器-解码器使用 beam search 生成模型的输出。
@dataclass
class TFBeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    编码器-解码器使用 beam search 生成模型的输出的基类。可以通过 encoder_attentions 和 encoder_hidden_states 访问解码器（或编码器）的隐藏状态和注意力权重。

    """
    # 定义一个包含多个属性的数据类，用于存储序列、分数、索引以及各种注意力和隐藏状态信息
    
    sequences: tf.Tensor = None
    # 序列数据，类型为 TensorFlow 的张量，初始值为 None
    sequences_scores: Optional[tf.Tensor] = None
    # 序列的分数数据，类型为可选的 TensorFlow 张量，初始值为 None
    scores: Optional[Tuple[tf.Tensor]] = None
    # 分数数据，类型为可选的 TensorFlow 张量元组，初始值为 None
    beam_indices: Optional[tf.Tensor] = None
    # Beam 搜索的索引数据，类型为可选的 TensorFlow 张量，初始值为 None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    # 编码器注意力数据，类型为可选的 TensorFlow 张量元组，初始值为 None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    # 编码器隐藏状态数据，类型为可选的 TensorFlow 张量元组，初始值为 None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 解码器注意力数据，类型为可选的嵌套元组的 TensorFlow 张量元组，初始值为 None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 交叉注意力数据，类型为可选的嵌套元组的 TensorFlow 张量元组，初始值为 None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 解码器隐藏状态数据，类型为可选的嵌套元组的 TensorFlow 张量元组，初始值为 None
@dataclass
class TFBeamSampleDecoderOnlyOutput(ModelOutput):
    """
    Decoder-only生成模型使用Beam采样的输出的基类。

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            生成的序列。第二维(sequence_length)要么等于`max_length`，要么因为`eos_token_id`导致所有批次提前结束而更短。
        sequences_scores (`tf.Tensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            生成的`sequences`的最终beam分数。
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            每个生成步骤中每个词汇标记的处理beam分数。每个元素为`tf.Tensor`的元组，最多有`max_new_tokens`个元素（每个生成的标记一个元素），每个张量的形状为`(batch_size*num_beams*num_return_sequences, config.vocab_size)`。
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            每个生成步骤生成的标记ID的beam索引。形状为`(batch_size*num_return_sequences, sequence_length)`的`tf.Tensor`。
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            每个生成的标记的注意力权重。元组（每个生成标记一个元素），元组（每个解码器层一个元素），`tf.Tensor`的元组，形状为`(batch_size*num_beams, num_heads, generated_length, sequence_length)`。
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            解码器每层的隐藏状态。元组（每个生成标记一个元素），元组（每个解码器层一个元素），`tf.Tensor`的元组，形状为`(batch_size*num_beams, generated_length, hidden_size)`。
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    beam_indices: Optional[tf.Tensor] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSampleEncoderDecoderOutput(ModelOutput):
    """
    Encoder-decoder生成模型使用Beam采样的输出的基类。可以通过encoder_attentions和encoder_hidden_states属性访问解码器（或者通过decoder_attentions和decoder_hidden_states属性访问编码器）的隐藏状态和注意力权重。

    """
    # 定义一个变量 sequences，类型为 tf.Tensor，初始值为 None
    sequences: tf.Tensor = None
    
    # 定义一个变量 sequences_scores，类型为 Optional[tf.Tensor]，初始值为 None
    sequences_scores: Optional[tf.Tensor] = None
    
    # 定义一个变量 scores，类型为 Optional[Tuple[tf.Tensor]]，初始值为 None
    scores: Optional[Tuple[tf.Tensor]] = None
    
    # 定义一个变量 beam_indices，类型为 Optional[tf.Tensor]，初始值为 None
    beam_indices: Optional[tf.Tensor] = None
    
    # 定义一个变量 encoder_attentions，类型为 Optional[Tuple[tf.Tensor]]，初始值为 None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    
    # 定义一个变量 encoder_hidden_states，类型为 Optional[Tuple[tf.Tensor]]，初始值为 None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    
    # 定义一个变量 decoder_attentions，类型为 Optional[Tuple[Tuple[tf.Tensor]]]，初始值为 None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    
    # 定义一个变量 cross_attentions，类型为 Optional[Tuple[Tuple[tf.Tensor]]]，初始值为 None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    
    # 定义一个变量 decoder_hidden_states，类型为 Optional[Tuple[Tuple[tf.Tensor]]]，初始值为 None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
@dataclass
class TFContrastiveSearchDecoderOnlyOutput(ModelOutput):
    """
    Decoder-only generation model output class for contrastive search.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFContrastiveSearchEncoderDecoderOutput(ModelOutput):
    """
    Encoder-decoder generation model output class for contrastive search.

    Base class for outputs of encoder-decoder generation models using contrastive search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
    """
    """
    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            生成的序列。第二个维度 (sequence_length) 可能等于 `max_length`，或者如果所有批次由于 `eos_token_id` 而提前结束，则会更短。
        scores (`tuple(tf.Tensor)` *optional*, 当 `output_scores=True` 传递或 `config.output_scores=True` 时返回):
            语言建模头部处理后的预测分数（SoftMax 前每个词汇标记的分数），每个生成步骤一个元组元素，元素数最多为 `max_new_tokens`，每个张量的形状为 `(batch_size, config.vocab_size)`。
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回):
            解码器每一层的注意力权重张量的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时返回):
            解码器每一层的隐藏状态张量的元组，形状为 `(batch_size, sequence_length, hidden_size)`，包含从嵌入层开始的所有层的输出。
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, 当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回):
            每个生成的标记一个元组元素，其中每个元素是解码器每一层的注意力权重张量的元组，形状为 `(batch_size, num_heads, generated_length, sequence_length)`。
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, 当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回):
            每个生成的标记一个元组元素，其中每个元素是解码器每一层与编码器的交叉注意力权重张量的元组，形状为 `(batch_size, num_heads, generated_length, sequence_length)`。
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, 当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时返回):
            每个生成的标记一个元组元素，其中每个元素是解码器每一层的隐藏状态张量的元组，形状为 `(batch_size, generated_length, hidden_size)`。
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
# 定义类型别名，表示不同的生成器输出类型
TFGreedySearchOutput = Union[TFGreedySearchEncoderDecoderOutput, TFGreedySearchDecoderOnlyOutput]
TFSampleOutput = Union[TFSampleEncoderDecoderOutput, TFSampleDecoderOnlyOutput]
TFBeamSearchOutput = Union[TFBeamSearchEncoderDecoderOutput, TFBeamSearchDecoderOnlyOutput]
TFBeamSampleOutput = Union[TFBeamSampleEncoderDecoderOutput, TFBeamSampleDecoderOnlyOutput]
TFContrastiveSearchOutput = Union[TFContrastiveSearchEncoderDecoderOutput, TFContrastiveSearchDecoderOnlyOutput]
# 定义一个类型别名，表示所有生成器可能的输出类型
TFGenerateOutput = Union[
    TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, TFContrastiveSearchOutput
]

class TFGenerationMixin:
    """
    包含支持生成的所有函数的类，用作[`TFPreTrainedModel`]中的混合类。

    该类公开[`~generation.TFGenerationMixin.generate`]，可以用于：
        - 当`num_beams=1`且`do_sample=False`时通过调用[`~generation.TFGenerationMixin.greedy_search`]进行*贪婪解码*
        - 当`penalty_alpha>0`且`top_k>1`时通过调用[`~generation.TFGenerationMixin.contrastive_search`]进行*对比搜索*
        - 当`num_beams=1`且`do_sample=True`时通过调用[`~generation.TFGenerationMixin.sample`]进行*多项式采样*
        - 当`num_beams>1`时通过调用[`~generation.TFGenerationMixin.beam_search`]进行*束搜索解码*

    不需要直接调用上述任何方法。而是将自定义参数值传递给 'generate' 方法。有关解码策略的更多信息，请参阅[text generation strategies guide](../generation_strategies)。
    """

    _seed_generator = None

    @property
    def seed_generator(self):
        # 警告：`seed_generator`已弃用，并将在未来版本中移除。
        warnings.warn("`seed_generator` is deprecated and will be removed in a future version.", UserWarning)
        if self._seed_generator is None:
            # 如果尚未初始化种子生成器，则从不确定状态创建一个随机生成器
            self._seed_generator = tf.random.Generator.from_non_deterministic_state()
        return self._seed_generator

    # 表示该类支持 XLA 生成
    supports_xla_generation = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # 如果模型类想要使用 `generate` 方法，需要定义 `prepare_inputs_for_generation` 方法
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`."
        )

    def compute_transition_scores(
        self,
        sequences: tf.Tensor,
        scores: Tuple[tf.Tensor],
        beam_indices: Optional[tf.Tensor] = None,
        normalize_logits: bool = False,
    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        # 检查当前模型类是否可以生成文本
        if not self.can_generate():
            # 定义兼容生成操作的模型映射列表
            generate_compatible_mappings = [
                TF_MODEL_FOR_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
                TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            # 遍历每个模型映射，检查当前模型类是否在其支持的模型中
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            # 构建异常消息，指示当前模型类不支持生成操作，并推荐可用的兼容模型类
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            # 抛出类型错误异常，包含详细的错误消息
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # 如果是编码-解码模型，排除在调用任何模型函数之前已处理的参数
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # 检查是否 prepare_inputs_for_generation 方法接受了 `kwargs` 或 `model_kwargs` 参数，以便处理可选的前向传递输入
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.call).parameters)
        # 检查每个传入的 model_kwargs 是否在模型参数中有对应的接收者
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            # 抛出数值错误异常，指示有未使用的 model_kwargs 参数
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )
        ) -> tf.Tensor:
        # 检查输入是否为 input_ids 类型且是二维的，并且数据类型为 tf.int32 或 tf.int64
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
        # 检查输入中是否存在 pad_token_id，并且在 inputs 中至少有一个元素等于 pad_token_id
        is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
        # 检查 pad_token_id 是否不等于 eos_token_id（如果 eos_token_id 为 None，则始终为 True）
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

        # 如果输入是 input_ids 且存在 pad_token_id 且 pad_token_id 不等于 eos_token_id，则生成 attention_mask
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32)
        else:
            # 否则返回一个全为 1 的 tensor，形状为 inputs.shape[:2]
            return tf.ones(inputs.shape[:2], dtype=tf.int32)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: tf.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. 获取编码器并存储编码器输出
        encoder = self.get_encoder()

        # 2. 从 model_kwargs 中准备编码器参数和编码器关键字参数
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        # 从 model_kwargs 中筛选出不以 irrelevant_prefix 开头的参数作为编码器参数
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # 检查编码器的调用签名，将符合签名的参数留下来
        encoder_signature = set(inspect.signature(encoder.call).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. 视觉模型不使用 `attention_mask`
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        # 如果 model_input_name 不是 self.main_input_name，在 Keras 中必须始终传递第一个输入
        if model_input_name != self.main_input_name:
            encoder_kwargs[self.main_input_name] = None
        # 调用编码器并将编码器输出存储在 model_kwargs 中的 "encoder_outputs" 键下
        encoder_outputs = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = encoder_outputs

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, tf.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. 检查用户是否手动定义了 `decoder_input_ids`。为了方便输入命名，如果编码器没有将其用作主输入，也允许用户通过 `input_ids` 参数传递。
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        # 如果 `input_ids` 在 `model_kwargs` 中，并且 `model_input_name` 不是 "input_ids"，则也将其用作 `decoder_input_ids`
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            # 否则，将 `decoder_input_ids` 设为 None
            decoder_input_ids = None

        # 2. 编码器-解码器模型期望 `decoder_input_ids` 以特殊标记开始。确保它符合这个要求。
        # 获取解码器的起始标记 ID
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        # 用 `decoder_start_token_id` 创建起始的 `decoder_input_ids` 张量
        decoder_input_ids_start = tf.ones((batch_size, 1), dtype=tf.int32) * decoder_start_token_id

        # 如果没有用户输入 -> 使用 `decoder_start_token_id` 作为 `decoder_input_ids`
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # 如果有用户输入但不以 `decoder_start_token_id` 开始 -> 在开头添加 `decoder_start_token_id`（并调整 `decoder_attention_mask` 如果提供了）
        elif tf.reduce_all(decoder_input_ids[:, 0] != decoder_start_token_id):
            decoder_input_ids = tf.concat([decoder_input_ids_start, decoder_input_ids], axis=-1)
            if "decoder_attention_mask" in model_kwargs:
                # 调整 `decoder_attention_mask`，在开头增加一个标记
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = tf.concat(
                    (tf.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    axis=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        # 返回处理后的 `decoder_input_ids` 和可能修改过的 `model_kwargs`
        return decoder_input_ids, model_kwargs

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # 检索编码器-解码器模型的解码器起始标记 ID
        # 如果需要，回退到 `bos_token_id`
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        # 如果 `decoder_start_token_id` 已定义，则返回它
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        # 否则，如果 `bos_token_id` 已定义，则返回它
        elif bos_token_id is not None:
            return bos_token_id
        # 如果两者都未定义，则引发 ValueError 异常
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[tf.Tensor] = None,
        expand_in_new_axis: bool = False,
        **model_kwargs,
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...] or [batch_size, expand_size, ...],
        depending on `expand_in_new_axis`. Beam-based approaches expect this function to be used with
        `expand_in_new_axis=True`
        """
        
        def _expand_tensor(tensor: tf.Tensor):
            # 根据 `expand_in_new_axis` 参数选择不同的扩展方式
            if expand_in_new_axis:
                shape = shape_list(tensor)
                return tf.broadcast_to(tensor[:, None], (shape[0], expand_size) + tuple(shape[1:]))
            else:
                return tf.repeat(tensor, expand_size, axis=0)

        def _expand_dict_for_generation(dict_to_expand):
            # 遍历字典中的每个值，如果是 Tensor 类型且非空，则调用 `_expand_tensor` 函数扩展
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], tf.Tensor):
                    dict_to_expand[key] = _expand_tensor(dict_to_expand[key])
            return dict_to_expand

        if input_ids is not None:
            # 如果 `input_ids` 不为空，则调用 `_expand_tensor` 函数扩展 `input_ids`
            input_ids = _expand_tensor(input_ids)

        # 调用 `_expand_dict_for_generation` 函数扩展 `model_kwargs`
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            # 如果是编码-解码模型，确保 `encoder_outputs` 在 `model_kwargs` 中定义
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            # 调用 `_expand_dict_for_generation` 函数扩展 `encoder_outputs`
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        # 返回扩展后的 `input_ids` 和 `model_kwargs`
        return input_ids, model_kwargs

    def _prepare_model_inputs(
        self,
        inputs: Optional[tf.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,
    ):
        """
        Prepares inputs for the model, optionally including a beginning-of-sequence token ID (`bos_token_id`).
        """
        # 此函数未提供实现，仅作为方法声明，用于准备模型的输入

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[tf.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,
    ):
        """
        Initializes `input_ids` for generation, optionally including a beginning-of-sequence token ID (`bos_token_id`).
        """
        # 此函数未提供实现，仅作为方法声明，用于为生成任务初始化 `input_ids`
    ) -> tf.Tensor:
        """Initializes input ids for generation, if necessary."""
        # 如果已经提供了输入，则直接返回输入
        if inputs is not None:
            return inputs

        # 获取模型参数中的编码器输出
        encoder_outputs = model_kwargs.get("encoder_outputs")
        # 如果是编码-解码模型并且有编码器输出，则创建一个全为-100的虚拟输入，以确保不会被用于编码
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            shape = encoder_outputs.last_hidden_state.shape[:-1]
            return tf.ones(shape, dtype=tf.int32) * -100

        # 如果未提供输入且未定义bos_token_id，则抛出异常
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # 如果在 `model_kwargs` 中有张量，则可以从中推断批量大小。这对于软提示或基于解码器的多模态实现很有帮助。
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, tf.Tensor):
                batch_size = value.shape[0]
                break
        # 创建一个形状为(batch_size, 1)的全为bos_token_id的张量，作为初始化的输入
        return tf.ones((batch_size, 1), dtype=tf.int32) * bos_token_id

    @staticmethod
    def _extract_past_from_model_output(outputs: ModelOutput):
        """Extracts past key values from model outputs."""
        past_key_values = None
        # 根据不同的输出结构，提取过去的键值
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        """Updates model keyword arguments for generation."""
        # 更新模型参数中的过去键值
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        # 更新注意力掩码
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                # 将一个形状为原注意力掩码+1列的张量与原注意力掩码拼接，用于后续生成过程中的扩展
                model_kwargs["attention_mask"] = tf.concat(
                    [attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
                )

        return model_kwargs

    def _update_model_kwargs_for_xla_generation(
        self,
        model_outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        cur_len: int,
        max_length: int,
        batch_size: int,
        is_encoder_decoder: bool = False,
        batch_axis: int = 0,
    ):
        """Updates model keyword arguments for XLA generation."""
        # 省略部分代码，未作注释要求的部分
        pass

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
        # 省略部分代码，未作注释要求的部分
        ):
        """Gets the logits warper for generation."""
        pass
        ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsWarper`]
        instances used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = TFLogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(generation_config.eos_token_id) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # Check if temperature warping is enabled and add warper accordingly
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TFTemperatureLogitsWarper(generation_config.temperature))
        
        # Check if top-k warping is enabled and add warper accordingly
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TFTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        
        # Check if top-p warping is enabled and add warper accordingly
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TFTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        
        # Return the list of warpers containing all configured logits processors
        return warpers
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # 创建一个空的处理器列表
        processors = TFLogitsProcessorList()

        # 如果设定了重复惩罚并且不等于默认值 1.0，则添加重复惩罚处理器
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(TFRepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        
        # 如果设定了禁止重复 n-gram 大小，并且大于 0，则添加禁止重复 n-gram 处理器
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(TFNoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        
        # 如果设定了要避免的词汇 ID 列表，则添加避免坏词汇处理器
        if generation_config.bad_words_ids is not None:
            processors.append(
                TFNoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        
        # 如果设定了最小生成长度、结束符号 ID，并且最小长度大于 0，则添加最小长度处理器
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(TFMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        
        # 如果设定了强制起始符号 ID，则添加强制起始符号处理器
        if generation_config.forced_bos_token_id is not None:
            processors.append(TFForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        
        # 如果设定了强制结束符号 ID，则添加强制结束符号处理器
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                TFForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        
        # 如果设定了要抑制的 token 列表，则添加抑制 token 处理器
        if generation_config.suppress_tokens is not None:
            processors.append(TFSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        
        # 如果设定了要在开头抑制的 token 列表，则添加在开头抑制 token 处理器
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                begin_index += generation_config.forced_decoder_ids[-1][
                    0
                ]  # generation starts after the last token that is forced
            processors.append(
                TFSuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        
        # 如果设定了强制生成的 token 列表，则添加强制生成 token 处理器
        if generation_config.forced_decoder_ids is not None:
            processors.append(TFForceTokensLogitsProcessor(generation_config.forced_decoder_ids))

        # 合并默认处理器列表和自定义处理器列表，并返回最终的处理器列表
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        return processors
    # 定义一个方法，接受一个自定义的 TFLogitsProcessorList 参数列表，并返回一个 TFLogitsProcessorList 对象
    def __init__(self, custom_list: List[TFLogitsProcessor] = []) -> TFLogitsProcessorList:
        # 如果 custom_list 是空的，则返回默认的 default_list
        if len(custom_list) == 0:
            return default_list
        # 遍历 default_list 中的每个元素
        for default in default_list:
            # 遍历 custom_list 中的每个元素
            for custom in custom_list:
                # 如果 custom 和 default 的类型相同
                if type(custom) is type(default):
                    # 设置对象类型为 "logits processor"
                    object_type = "logits processor"
                    # 抛出值错误异常，提醒用户 custom 对象已经存在
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        # 将 custom_list 中的元素扩展到 default_list 中
        default_list.extend(custom_list)
        # 返回扩展后的 default_list
        return default_list

    # 定义一个贪婪搜索方法，接受多个参数和关键字参数
    def greedy_search(
        self,
        input_ids: tf.Tensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ):
        # 方法实现省略

    # 定义一个采样方法，接受多个参数和关键字参数
    def sample(
        self,
        input_ids: tf.Tensor,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        seed: Optional[Tuple[int, int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ):
        # 方法实现省略

    @staticmethod
    def _gather_beams(nested, beam_indices, batch_axis=0):
        """Gathers the beam slices indexed by beam_indices into new beam array."""

        def gather_fn(tensor):
            # 如果 batch_axis 大于 0，则将 batch_axis 之前的所有维度移到最后，以便得到形状为 (batch, beam_id, ...) 的张量
            if batch_axis > 0:
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                tensor = tf.transpose(tensor, perm=perm)

            # 在 axis=1 上使用 beam_indices 进行 gather 操作，得到聚集后的张量
            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
            
            # 如果 batch_axis 大于 0，则将张量恢复到原始的维度顺序
            if batch_axis > 0:
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                perm = tf.math.invert_permutation(perm)
                gathered_tensor = tf.transpose(gathered_tensor, perm=perm)

            return gathered_tensor

        # 对 nested 结构中的每个张量应用 gather_fn 函数，并返回新的结构
        return tf.nest.map_structure(gather_fn, nested)
# 将给定的值按照批次索引散布到张量中
def scatter_values_on_batch_indices(values, batch_indices):
    # 获取批次索引张量的形状
    shape = shape_list(batch_indices)
    # 扩展批次维度以匹配形状
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # 将批次索引转换为对应的索引对
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # 根据索引对将值散布到目标形状中
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)


def sample_without_replacement(logits, num_samples):
    """
    不重复的分类采样当前尚未实现，现在使用Gumbel-Max技巧代替，请参见
    https://github.com/tensorflow/tensorflow/issues/9260 获取更多信息
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(shape_list(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return indices


def _ranking_fast(
    context_hidden: tf.Tensor,
    next_hidden: tf.Tensor,
    next_top_k_probs: tf.Tensor,
    alpha: float,
    beam_width: int,
) -> tf.Tensor:
    """
    根据文献《神经文本生成的对比框架》中描述的退化惩罚（与先前标记的余弦相似度）对top_k候选进行重新排序。
    返回批次中每行最佳候选的索引。
    """
    # 对上下文隐藏层进行归一化处理
    norm_context_hidden = context_hidden / tf.norm(context_hidden, axis=2, keepdims=True)
    # 对下一个隐藏层进行归一化处理
    norm_next_hidden = next_hidden / tf.norm(next_hidden, axis=2, keepdims=True)
    # 计算余弦相似度矩阵
    cosine_matrix = tf.squeeze(tf.linalg.matmul(norm_context_hidden, norm_next_hidden, transpose_b=True), axis=-1)
    # 计算最大余弦相似度的退化惩罚
    degeneration_penalty = tf.reduce_max(cosine_matrix, axis=-1)
    # 重塑下一个top_k概率
    next_top_k_probs = tf.reshape(next_top_k_probs, shape=[-1])
    # 计算对比分数，包括概率和退化惩罚
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = tf.reshape(contrastive_score, shape=[-1, beam_width])
    # 选择每行中最高对比分数的索引
    selected_idx = tf.argmax(contrastive_score, axis=1)
    return selected_idx
```