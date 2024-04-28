# `.\transformers\generation\tf_utils.py`

```
# 设置编码格式为 UTF-8
# 版权声明，版权归 Google AI Language Team 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证要求，否则不能使用此文件
# 可以在下面链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件分发时不附带任何形式的明示或暗示的担保或条件
# 详见许可证，此处省略
# 导入必要的库和模块
import copy  # 复制模块，用于对象的深度复制
import inspect  # 检查模块，用于检查对象的属性和方法
import warnings  # 警告模块，用于发出警告信息
from dataclasses import dataclass  # 数据类模块，用于创建数据类
from typing import Any, Dict, Optional, Tuple, Union  # 类型提示模块，用于指定函数的参数和返回值的类型

import numpy as np  # NumPy 库，用于处理数组和矩阵的数学运算
import tensorflow as tf  # TensorFlow 库，用于构建和训练深度学习模型
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice  # TensorFlow XLA 模块，用于动态更新切片

from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput  # 导入模型输出类
from ..models.auto import (  # 导入自动模型映射
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..tf_utils import shape_list, stable_softmax  # 导入 TensorFlow 工具函数
from ..utils import ModelOutput, logging  # 导入模型输出类和日志记录模块
from .configuration_utils import GenerationConfig  # 导入生成配置类
from .tf_logits_process import (  # 导入 TensorFlow logits 处理相关模块
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义数据类，用于存储解码器仅生成模型的输出
@dataclass
class TFGreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.
    """
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
    
    # 初始化变量，用于存储生成的序列、预测分数、注意力权重和隐藏状态
    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
@dataclass
class TFGreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None  # 存储生成的序列
    scores: Optional[Tuple[tf.Tensor]] = None  # 存储语言模型头部预测分数的元组，每个元素是一个张量，形状为(batch_size, config.vocab_size)
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None  # 存储编码器注意力权重的元组，每个元素是一个张量，形状为(batch_size, num_heads, sequence_length, sequence_length)
    # 定义一个可选的元组，用于存储编码器的隐藏状态
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    # 定义一个可选的元组，用于存储解码器的注意力权重
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 定义一个可选的元组，用于存储交叉注意力权重
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 定义一个可选的元组，用于存储解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
from dataclasses import dataclass
from typing import Optional, Tuple
import tensorflow as tf
from transformers.modeling_tf_outputs import ModelOutput

# 定义一个用于仅解码器生成模型输出的基类，采用采样方式生成
@dataclass
class TFSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(num_return_sequences*batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    # 生成的序列
    sequences: tf.Tensor = None
    # 语言建模头部的预测分数，在每一代的每一步生成时返回
    scores: Optional[Tuple[tf.Tensor]] = None
    # 注意力权重，返回时在每一代的每一步生成
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 隐藏状态，返回时在每一代的每一步生成
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


# 定义一个用于编码器-解码器生成模型输出的基类，采用采样方式生成
@dataclass
class TFSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
    the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes).
    """
```  
    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size*num_return_sequences,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_return_sequences, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_return_sequences, generated_length, hidden_size)`.

    """

    # 初始化变量，用于存储生成的序列、预测分数、编码器注意力、编码器隐藏状态、解码器注意力、交叉注意力、解码器隐藏状态
    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
# 导入必要的库
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_tf_outputs import ModelOutput
import tensorflow as tf

# 基于 Beam Search 的解码器生成的输出的基类
@dataclass
class TFBeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this
            beam. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `tf.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    # 生成的序列
    sequences: tf.Tensor = None
    # 生成序列的最终 Beam 分数
    sequences_scores: Optional[tf.Tensor] = None
    # 每一代的处理后的 Beam 分数
    scores: Optional[Tuple[tf.Tensor]] = None
    # 生成的每个标记的 Beam 索引
    beam_indices: Optional[tf.Tensor] = None
    # 生成序列时的注意力权重
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 生成序列时的隐藏状态
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


# 基于 Beam Search 的编码器-解码器生成的输出的基类
@dataclass
class TFBeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    """
    """
    定义一个包含各种属性的类，包括 decoder_attentions 和 decoder_hidden_states 属性
    """

    # 定义一个 TensorFlow 张量变量 sequences，初始值为 None
    sequences: tf.Tensor = None
    # 定义一个可选的 TensorFlow 张量变量 sequences_scores，初始值为 None
    sequences_scores: Optional[tf.Tensor] = None
    # 定义一个可选的包含 TensorFlow 张量元组的 scores 变量，初始值为 None
    scores: Optional[Tuple[tf.Tensor]] = None
    # 定义一个可选的 TensorFlow 张量变量 beam_indices，初始值为 None
    beam_indices: Optional[tf.Tensor] = None
    # 定义一个可选的包含 TensorFlow 张量元组的 encoder_attentions 变量，初始值为 None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    # 定义一个可选的包含 TensorFlow 张量元组的 encoder_hidden_states 变量，初始值为 None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    # 定义一个可选的包含 TensorFlow 张量元组的 decoder_attentions 变量，初始值为 None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 定义一个可选的包含 TensorFlow 张量元组的 cross_attentions 变量，初始值为 None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 定义一个可选的包含 TensorFlow 张量元组的 decoder_hidden_states 变量，初始值为 None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
# 定义了一个用于仅使用解码器生成模型输出的基类，该模型使用 beam sample 技术。
@dataclass
class TFBeamSampleDecoderOnlyOutput(ModelOutput):
    """
    仅使用解码器生成模型输出的基类。

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            生成的序列。第二维（sequence_length）要么等于 `max_length`，要么较短，如果所有批次由于 `eos_token_id` 提前结束。
        sequences_scores (`tf.Tensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            生成的 `sequences` 的最终 beam 分数。
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            每个生成步骤每个词汇标记的处理后 beam 分数。Beam 分数由每个词汇标记的 log softmax 分数和此 beam 中先前生成标记的 log softmax 之和组成。由多达 `max_new_tokens` 元素组成的 `tf.Tensor` 元组（每个生成标记一个元素），每个张量的形状为 `(batch_size*num_beams*num_return_sequences, config.vocab_size)`。
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            每个生成步骤生成的令牌 id 的 beam 索引。形状为 `(batch_size*num_return_sequences, sequence_length)` 的 `tf.Tensor`。
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            每个生成标记的元组（每个解码器层一个元素）的元组（每个生成标记一个元素）的注意力权重。由 `(batch_size*num_beams, num_heads, generated_length, sequence_length)` 形状的 `tf.Tensor` 元组组成。
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            每个生成标记的元组（每个解码器层一个元素）的元组（每个生成标记一个元素）的隐藏状态。由 `(batch_size*num_beams, generated_length, hidden_size)` 形状的 `tf.Tensor` 元组组成。
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    beam_indices: Optional[tf.Tensor] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


# 定义了一个用于使用编码器-解码器生成模型输出的基类，该模型使用 beam 抽样。解码器的隐藏状态和注意权重（分别是编码器的注意力和隐藏状态）可以通过 encoder_attentions 和 encoder_hidden_states 属性（分别是 decoder_attentions 和 decoder_hidden_states 属性）访问
@dataclass
class TFBeamSampleEncoderDecoderOutput(ModelOutput):
    """
    使用编码器-解码器生成模型输出的基类。

    Hidden states 和 attention 权重属于解码器（分别是编码器）可以通过 encoder_attentions 和 encoder_hidden_states 属性（分别是 decoder_attentions 和 decoder_hidden_states 属性）访问。

    """
    # 定义变量 sequences，用于存储张量（TensorFlow 的张量），初始化为 None
    sequences: tf.Tensor = None
    # 定义变量 sequences_scores，用于存储张量（TensorFlow 的张量），初始化为可选类型的 None
    sequences_scores: Optional[tf.Tensor] = None
    # 定义变量 scores，用于存储张量元组，其中元素为 TensorFlow 的张量，初始化为可选类型的 None
    scores: Optional[Tuple[tf.Tensor]] = None
    # 定义变量 beam_indices，用于存储张量（TensorFlow 的张量），初始化为可选类型的 None
    beam_indices: Optional[tf.Tensor] = None
    # 定义变量 encoder_attentions，用于存储张量元组，其中元素为 TensorFlow 的张量，初始化为可选类型的 None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    # 定义变量 encoder_hidden_states，用于存储张量元组，其中元素为 TensorFlow 的张量，初始化为可选类型的 None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    # 定义变量 decoder_attentions，用于存储张量元组，其中元素为张量元组，内部元素为 TensorFlow 的张量，初始化为可选类型的 None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 定义变量 cross_attentions，用于存储张量元组，其中元素为张量元组，内部元素为 TensorFlow 的张量，初始化为可选类型的 None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    # 定义变量 decoder_hidden_states，用于存储张量元组，其中元素为张量元组，内部元素为 TensorFlow 的张量，初始化为可选类型的 None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
# 定义一个数据类，用于存储仅解码器生成模型使用对比搜索的输出结果
class TFContrastiveSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using contrastive search.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            生成的序列。第二维（sequence_length）要么等于 `max_length`，要么比 `eos_token_id` 提前结束。
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            在每个生成步骤中，语言建模头的处理过的预测分数（SoftMax 之前的每个词汇标记的分数）。
            元组，包含最多 `max_new_tokens` 个元素（每个生成的标记一个元素），每个张量的形状为 `(batch_size, config.vocab_size)`。
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            元组（每个生成的标记一个元素），元组（解码器的每一层一个元素）的注意力张量，形状为 `(batch_size, num_heads, generated_length, sequence_length)`。
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组（每个生成的标记一个元素），元组（解码器的每一层一个元素）的隐藏状态张量，形状为 `(batch_size, generated_length, hidden_size)`。
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


# 定义一个数据类，用于存储使用对比搜索的编码器-解码器生成模型的输出结果。可以通过 encoder_attentions 和 encoder_hidden_states 属性（或 decoder_attentions 和 decoder_hidden_states 属性）访问解码器（或编码器）的隐藏状态和注意力权重。
class TFContrastiveSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using contrastive search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            生成的序列。第二维（sequence_length）要么等于 `max_length`，要么如果所有批次由于 `eos_token_id` 提前结束则会更短。
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            语言建模头部的处理过的预测分数（SoftMax 之前的每个词汇标记的分数）在每个生成步骤的情况下。包含最多 `max_new_tokens` 个元素的 `tf.Tensor` 元组（每个生成的词汇标记一个元素），每个张量的形状为 `(batch_size, config.vocab_size)`。
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            编码器注意力的元组（每个解码器层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor`。
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            编码器隐藏状态的元组（每个解码器层一个），形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor`（包含嵌入输出和每个层的输出）。
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            元组（每个生成的词汇标记一个）的元组（每个解码器层一个）的形状为 `(batch_size, num_heads, generated_length, sequence_length)` 的 `tf.Tensor`。
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            元组（每个生成的词汇标记一个）的元组（每个解码器层一个）的形状为 `(batch_size, num_heads, generated_length, sequence_length)` 的 `tf.Tensor`。
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组（每个生成的词汇标记一个）的元组（每个解码器层一个）的形状为 `(batch_size, generated_length, hidden_size)` 的 `tf.Tensor`。
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None
# 定义类型别名，表示生成器的输出可能是贪婪搜索、采样、束搜索、对比搜索中的一种
TFGreedySearchOutput = Union[TFGreedySearchEncoderDecoderOutput, TFGreedySearchDecoderOnlyOutput]
TFSampleOutput = Union[TFSampleEncoderDecoderOutput, TFSampleDecoderOnlyOutput]
TFBeamSearchOutput = Union[TFBeamSearchEncoderDecoderOutput, TFBeamSearchDecoderOnlyOutput]
TFBeamSampleOutput = Union[TFBeamSampleEncoderDecoderOutput, TFBeamSampleDecoderOnlyOutput]
TFContrastiveSearchOutput = Union[TFContrastiveSearchEncoderDecoderOutput, TFContrastiveSearchDecoderOnlyOutput]
TFGenerateOutput = Union[
    TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, TFContrastiveSearchOutput
]

# 定义一个混合类，包含所有支持生成的函数，用作 [`TFPreTrainedModel`] 中的一个 mixin
class TFGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in [`TFPreTrainedModel`].

    The class exposes [`~generation.TFGenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.TFGenerationMixin.greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.TFGenerationMixin.contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.TFGenerationMixin.sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.TFGenerationMixin.beam_search`] if `num_beams>1`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    # 种子生成器属性，标记为不推荐使用，并在将来的版本中将被移除
    _seed_generator = None

    # 获取种子生成器属性的方法
    @property
    def seed_generator(self):
        warnings.warn("`seed_generator` is deprecated and will be removed in a future version.", UserWarning)
        # 如果种子生成器属性为空，创建一个非确定性状态的随机生成器
        if self._seed_generator is None:
            self._seed_generator = tf.random.Generator.from_non_deterministic_state()
        return self._seed_generator

    # 是否支持 XLA 生成
    supports_xla_generation = True

    # 准备用于生成的输入数据的方法，需要由模型类定义
    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`."
        )

    # 计算转移分数的方法，用于束搜索
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
        # 检查模型类是否与生成兼容，如果不兼容，则引发异常指向正确的类
        if not self.can_generate():
            # 定义与生成兼容的映射列表
            generate_compatible_mappings = [
                TF_MODEL_FOR_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
                TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            # 遍历生成兼容的映射列表
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # 排除在调用任何模型函数之前处理的参数
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs`通常用于处理可选的前向传递输入，如`attention_mask`。如果`prepare_inputs_for_generation`不接受它们，则可以进行更严格的检查
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.call).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def generate(
        self,
        inputs: Optional[tf.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        seed=None,
        **kwargs,
    def _prepare_attention_mask_for_generation(
        self,
        inputs: tf.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
        # 检查输入是否为 input_ids，并且是否包含填充标记 -> 只有在这种情况下才定义 attention_mask
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
        is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            # 如果是 input_ids 并且包含填充标记，并且填充标记不等于 eos_token_id，则返回不等于填充标记的张量
            return tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32)
        else:
            # 否则返回形状为 inputs.shape[:2] 的全为1的张量
            return tf.ones(inputs.shape[:2], dtype=tf.int32)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: tf.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. 获取编码器并存储编码器输出
        encoder = self.get_encoder()

        # 2. 从模型参数中准备编码器参数和编码器关键字参数
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.call).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. 视觉模型不使用 `attention_mask`
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        if model_input_name != self.main_input_name:  # 在 Keras 中，第一个输入必须始终传递
            encoder_kwargs[self.main_input_name] = None
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
        # 1. 检查用户是否手动定义了 `decoder_input_ids`。为了方便输入命名，如果编码器不将其用作主要输入，我们也允许用户将其传递为 `input_ids`。
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # 如果用户在 model_kwargs 中定义了 `decoder_input_ids`，则将其取出
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            # 如果用户在 model_kwargs 中定义了 `input_ids`，且编码器不将其用作主要输入，则将其作为 `decoder_input_ids`
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            # 否则，将 `decoder_input_ids` 设为 None
            decoder_input_ids = None

        # 2. 编码器-解码器模型要求 `decoder_input_ids` 以特殊令牌开头。让我们确保如此。
        # 获取解码器的起始令牌 ID
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        # 用 decoder_start_token_id 创建 decoder_input_ids_start，其形状为(batch_size, 1)
        decoder_input_ids_start = tf.ones((batch_size, 1), dtype=tf.int32) * decoder_start_token_id

        # 如果没有用户输入 -> 使用 decoder_start_token_id 作为 decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # 如果有用户输入但不以 decoder_start_token_id 开头 -> 在其前面添加 decoder_start_token_id（并调整 decoder_attention_mask 如果提供）
        elif tf.reduce_all(decoder_input_ids[:, 0] != decoder_start_token_id):
            decoder_input_ids = tf.concat([decoder_input_ids_start, decoder_input_ids], axis=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = tf.concat(
                    (tf.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    axis=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # 检索编码器-解码器模型的解码器起始令牌 ID，如有必要，则退回到 bos_token_id
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        # 如果 decoder_start_token_id 已定义，则返回之
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        # 如果未定义 decoder_start_token_id，则返回 bos_token_id
        elif bos_token_id is not None:
            return bos_token_id
        # 否则抛出值错误
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    # 定义一个函数，用于扩展生成的输入张量的大小
    def _expand_inputs_for_generation(
        expand_size: int = 1,  # 扩展大小，默认为1
        is_encoder_decoder: bool = False,  # 是否为编码-解码模型，默认为False
        input_ids: Optional[tf.Tensor] = None,  # 输入张量，默认为None
        expand_in_new_axis: bool = False,  # 是否在新轴上扩展，默认为False
        **model_kwargs,  # 其他模型参数
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:  # 返回类型为元组，包含扩展后的输入张量和模型参数字典
        """
        Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...] or [batch_size, expand_size, ...],
        depending on `expand_in_new_axis`. Beam-based approaches expect this function to be used with
        `expand_in_new_axis=True`
        """

        # 定义一个函数，用于扩展张量
        def _expand_tensor(tensor: tf.Tensor):
            if expand_in_new_axis:  # 如果在新轴上扩展
                shape = shape_list(tensor)  # 获取张量的形状
                return tf.broadcast_to(tensor[:, None], (shape[0], expand_size) + tuple(shape[1:]))  # 在新轴上广播张量
            else:
                return tf.repeat(tensor, expand_size, axis=0)  # 沿指定轴重复张量expand_size次

        # 定义一个函数，用于扩展生成的字典
        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:  # 遍历字典的键
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], tf.Tensor):  # 如果值不为空且为张量
                    dict_to_expand[key] = _expand_tensor(dict_to_expand[key])  # 对值进行扩展
            return dict_to_expand  # 返回扩展后的字典

        if input_ids is not None:  # 如果输入张量不为空
            input_ids = _expand_tensor(input_ids)  # 对输入张量进行扩展

        model_kwargs = _expand_dict_for_generation(model_kwargs)  # 对模型参数字典进行扩展

        if is_encoder_decoder:  # 如果是编码-解码模型
            if model_kwargs.get("encoder_outputs") is None:  # 如果模型参数字典中没有"encoder_outputs"
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")  # 抛出数值错误
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])  # 对"encoder_outputs"进行扩展

        return input_ids, model_kwargs  # 返回扩展后的输入张量和模型参数字典

    # 定义一个函数，用于准备模型输入
    def _prepare_model_inputs(
        self,
        inputs: Optional[tf.Tensor] = None,  # 输入张量，默认为None
        bos_token_id: Optional[int] = None,  # 开始标记ID，默认为None
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,  # 模型参数字典，默认为None

    # 定义一个函数，用于为生成初始化输入张量
    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[tf.Tensor] = None,  # 输入张量，默认为None
        bos_token_id: Optional[int] = None,  # 开始标记ID，默认为None
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,  # 模型参数字典，默认为None
    ) -> tf.Tensor:
        """初始化生成的输入 id，如果需要的话。"""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # 创建带有值为 -100 的虚拟 input_ids，作为一种健全性检查，确保它们不会用于编码
            shape = encoder_outputs.last_hidden_state.shape[:-1]
            return tf.ones(shape, dtype=tf.int32) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` 在未提供 `input_ids` 时必须定义。")

        # 如果 `model_kwargs` 中有一些张量，我们可以从中推断出批次大小。这在软提示或建立在仅解码器语言模型之上的多模态实现中很有用。
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, tf.Tensor):
                batch_size = value.shape[0]
                break
        return tf.ones((batch_size, 1), dtype=tf.int32) * bos_token_id

    @staticmethod
    def _extract_past_from_model_output(outputs: ModelOutput):
        past_key_values = None
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
        # 更新 past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        # 更新注意力掩码
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
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
    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsWarper`]
        instances used for multinomial sampling.
        """

        # 实例化 warpers 列表
        warpers = TFLogitsProcessorList()

        # 在 beam 方法中，我们需要至少保留一个非 eos（end-of-sequence）标记，以便探索可能具有更好分数的延续（即保留 len(generation_config.eos_token_id) + 1）
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # 如果设置了温度参数且不等于1.0，则添加温度调整器
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TFTemperatureLogitsWarper(generation_config.temperature))
        # 如果设置了 top_k 参数且不等于0，则添加 top_k 调整器
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TFTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        # 如果设置了 top_p 参数且小于1.0，则添加 top_p 调整器
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TFTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        logits_processor: Optional[TFLogitsProcessorList],
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # 创建一个 TFLogitsProcessorList 对象来存储所有相关的 TFLogitsProcessor 实例，用于修改语言模型头部的分数
        processors = TFLogitsProcessorList()

        # 实例化处理器列表
        # 如果重复惩罚值不为 None 且不等于 1.0，则添加 TFRepetitionPenaltyLogitsProcessor 处理器
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(TFRepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        # 如果不重复 n 克大小不为 None 且大于 0，则添加 TFNoRepeatNGramLogitsProcessor 处理器
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(TFNoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        # 如果存在不良单词 ID，则添加 TFNoBadWordsLogitsProcessor 处理器
        if generation_config.bad_words_ids is not None:
            processors.append(
                TFNoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        # 如果最小长度不为 None 且结束标记 ID 不为 None 且最小长度大于 0，则添加 TFMinLengthLogitsProcessor 处理器
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(TFMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        # 如果强制开始标记 ID 不为 None，则添加 TFForcedBOSTokenLogitsProcessor 处理器
        if generation_config.forced_bos_token_id is not None:
            processors.append(TFForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        # 如果强制结束标记 ID 不为 None，则添加 TFForcedEOSTokenLogitsProcessor 处理器
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                TFForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        # 如果需要抑制的标记不为 None，则添加 TFSuppressTokensLogitsProcessor 处理器
        if generation_config.suppress_tokens is not None:
            processors.append(TFSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        # 如果需要在开头抑制的标记不为 None，则添加 TFSuppressTokensAtBeginLogitsProcessor 处理器
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
        # 如果强制解码器 ID 不为 None，则添加 TFForceTokensLogitsProcessor 处理器
        if generation_config.forced_decoder_ids is not None:
            processors.append(TFForceTokensLogitsProcessor(generation_config.forced_decoder_ids))

        # 合并默认处理器列表和自定义处理器列表
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: TFLogitsProcessorList,
        custom_list: TFLogitsProcessorList,
    # 定义一个静态方法，用于处理生成过程中的 logits
    @staticmethod
    # 定义函数参数和返回类型
    ) -> TFLogitsProcessorList:
        # 如果自定义的处理器列表为空，则直接返回默认的处理器列表
        if len(custom_list) == 0:
            return default_list
        # 遍历默认处理器列表
        for default in default_list:
            # 遍历自定义处理器列表
            for custom in custom_list:
                # 如果自定义处理器和默认处理器类型相同
                if type(custom) is type(default):
                    # 设置对象类型为"logits processor"
                    object_type = "logits processor"
                    # 抛出异常，提示用户自定义的处理器已经存在
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        # 将自定义处理器列表添加到默认处理器列表中
        default_list.extend(custom_list)
        # 返回合并后的处理器列表
        return default_list

    # 定义贪婪搜索方法
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
    # 定义采样方法
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
    # 静态方法声明结束
    @staticmethod
    def _gather_beams(nested, beam_indices, batch_axis=0):
        """Gathers the beam slices indexed by beam_indices into new beam array."""

        def gather_fn(tensor):
            if batch_axis > 0:
                # 将批次前面的所有维度推到最后，以便得到 (batch, beam_id, ...) 的形状
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                tensor = tf.transpose(tensor, perm=perm)

            # 使用指定索引从给定轴上收集张量的切片
            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
            if batch_axis > 0:
                # 将维度转换回原始的形状
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                perm = tf.math.invert_permutation(perm)
                gathered_tensor = tf.transpose(gathered_tensor, perm=perm)

            return gathered_tensor

        # 对嵌套结构中的每个张量应用 gather_fn 函数
        return tf.nest.map_structure(gather_fn, nested)

    def beam_search(
        self,
        input_ids: tf.Tensor,
        do_sample: bool = False,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[Union[bool, str]] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        num_return_sequences: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ):
        """
        Beam search algorithm for generating sequences.

        Args:
            input_ids: Tensor of token indices.
            do_sample: Whether to sample next token or use greedy strategy.
            max_length: The maximum length of the generated sequence.
            pad_token_id: The token id for padding.
            eos_token_id: The token id for end of sequence.
            length_penalty: Exponential penalty to the length of the generated sequence.
            early_stopping: If true, the generation stops as soon as all beams have finished generating sequences.
            logits_processor: A list of processors for modifying the logits during generation.
            logits_warper: A list of warpers for modifying the logits during generation.
            num_return_sequences: Number of sequences to return.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            output_scores: Whether to output scores.
            return_dict_in_generate: Whether to return a dictionary instead of a tuple in generation methods.

        Returns:
            A dictionary containing the generated sequences along with other optional outputs.
        """
        # 具体的生成方法在此实现
        pass

    def contrastive_search(
        self,
        input_ids: tf.Tensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ):
        """
        Contrastive search algorithm for generating sequences.

        Args:
            input_ids: Tensor of token indices.
            top_k: Number of top sequences to keep at each step.
            penalty_alpha: Penalty factor applied to the probability distribution.
            logits_processor: A list of processors for modifying the logits during generation.
            logits_warper: A list of warpers for modifying the logits during generation.
            max_length: The maximum length of the generated sequence.
            pad_token_id: The token id for padding.
            eos_token_id: The token id for end of sequence.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            output_scores: Whether to output scores.
            return_dict_in_generate: Whether to return a dictionary instead of a tuple in generation methods.

        Returns:
            A dictionary containing the generated sequences along with other optional outputs.
        """
        # 具体的生成方法在此实现
        pass
# 使用 top-k 和/或 nucleus (top-p) 过滤方法过滤 logits 分布
def tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            如果 > 0，仅保留概率最高的 top-k 个标记（top-k 过滤）
        top_p (`float`, *optional*, defaults to 1.0):
            如果 < 1.0，仅保留累积概率大于等于 top_p 的 top 标记（nucleus 过滤）。Nucleus 过滤见 Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            输出中每个批示例中要保留的最小标记数。

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    # 发出警告，提示 `tf_top_k_top_p_filtering` 将在 v4.39 中删除，建议使用 `TFTopKLogitsWarper` 和 `TFTopPLogitsWarper` 代替
    warnings.warn(
        "`tf_top_k_top_p_filtering` is scheduled for deletion in v4.39. Use `TFTopKLogitsWarper` and "
        "`TFTopPLogitsWarper` instead.",
        DeprecationWarning,
    )

    # 获取 logits 的形状
    logits_shape = shape_list(logits)

    # 如果 top_k 大于 0
    if top_k > 0:
        # 确保 top_k 大于等于 min_tokens_to_keep，并且不超过 logits 最后一个维度的大小
        top_k = min(max(top_k, min_tokens_to_keep), logits_shape[-1])  # Safety check
        # 移除概率小于 top-k 中最后一个标记的所有标记
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1, None]
        logits = tf.where(indices_to_remove, filter_value, logits)
    # 如果 top_p 小于 1.0
    if top_p < 1.0:
        # 根据 logits 排序得到的索引
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        # 根据排序后的索引获取 logits
        sorted_logits = tf.gather(
            logits, sorted_indices, axis=-1, batch_dims=1
        )  # expects logits to be of dim (batch_size, vocab_size)

        # 计算累积概率
        cumulative_probs = tf.math.cumsum(stable_softmax(sorted_logits, axis=-1), axis=-1)

        # 移除累积概率超过阈值的标记（概率为 0 的标记保留）
        sorted_indices_to_remove = cumulative_probs > top_p

        # 如果 min_tokens_to_keep 大于 1
        if min_tokens_to_keep > 1:
            # 至少保留 min_tokens_to_keep（减去 1 是因为我们在下面添加了第一个标记）
            sorted_indices_to_remove = tf.concat(
                [
                    tf.zeros_like(sorted_indices_to_remove[:, :min_tokens_to_keep]),
                    sorted_indices_to_remove[:, min_tokens_to_keep:],
                ],
                -1,
            )

        # 将索引向右移动以保留超过阈值的第一个标记
        sorted_indices_to_remove = tf.concat(
            [tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]],
            -1,
        )
        # 将排序后的张量散布到原始索引上
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        logits = tf.where(indices_to_remove, filter_value, logits)
    return logits


def scatter_values_on_batch_indices(values, batch_indices):
    # 获取 batch_indices 的形状
    shape = shape_list(batch_indices)
    # 广播批次维度到指定形状
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # 将批次索引转换为对应的配对索引
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # 将数值按照配对索引散布到指定形状的张量中
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)
# 对给定的logits进行无重复的分类抽样，目前尚未实现无重复的分类抽样，暂时使用gumbel-max技巧，详情请参考链接
def sample_without_replacement(logits, num_samples):
    # 生成服从均匀分布的随机数，用于gumbel-max技巧
    z = -tf.math.log(-tf.math.log(tf.random.uniform(shape_list(logits), 0, 1)))
    # 计算logits加上z后的top_k值和索引
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return indices

# 快速对top_k候选进行重新排序，基于退化惩罚（与先前标记的余弦相似度），如论文"A Contrastive Framework for Neural Text Generation"中所述
def _ranking_fast(
    context_hidden: tf.Tensor,
    next_hidden: tf.Tensor,
    next_top_k_probs: tf.Tensor,
    alpha: float,
    beam_width: int,
) -> tf.Tensor:
    # 对上下文隐藏层和下一个隐藏层进行归一化
    norm_context_hidden = context_hidden / tf.norm(context_hidden, axis=2, keepdims=True)
    norm_next_hidden = next_hidden / tf.norm(next_hidden, axis=2, keepdims=True)
    # 计算余弦相似度矩阵
    cosine_matrix = tf.squeeze(tf.linalg.matmul(norm_context_hidden, norm_next_hidden, transpose_b=True), axis=-1)
    # 计算最大余弦相似度，作为退化惩罚
    degeneration_penalty = tf.reduce_max(cosine_matrix, axis=-1)
    # 重塑next_top_k_probs的形状
    next_top_k_probs = tf.reshape(next_top_k_probs, shape=[-1])
    # 计算对比分数
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = tf.reshape(contrastive_score, shape=[-1, beam_width])
    # 选择最佳候选的索引
    selected_idx = tf.argmax(contrastive_score, axis=1)
    return selected_idx
```