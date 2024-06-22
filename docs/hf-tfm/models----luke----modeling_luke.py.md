# `.\transformers\models\luke\modeling_luke.py`

```py
# 指定编码格式为 UTF-8

# 引入必要的库
# 注意：这里使用了相对导入，即从当前包中导入模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从外部模块中导入一些函数和类
# 注意：这里是从父包(huggingface)中导入相关内容
from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
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

# 用于文档的配置和检查点说明
_CONFIG_FOR_DOC = "LukeConfig"
_CHECKPOINT_FOR_DOC = "studio-ousia/luke-base"

# 预训练模型的存档列表
LUKE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "studio-ousia/luke-base",
    "studio-ousia/luke-large",
    # 在 https://huggingface.co/models?filter=luke 查看所有 LUKE 模型
]

# 用于存储 LUKE 模型输出的基类，包含汇聚层结果
@dataclass
class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Base class for outputs of the LUKE model.
    # 函数参数说明：
    # last_hidden_state: 模型最后一层输出的隐藏状态序列，形状为(batch_size, sequence_length, hidden_size)
    # entity_last_hidden_state: 实体最后一层输出的隐藏状态序列，形状为(batch_size, entity_length, hidden_size)
    # pooler_output: 经过线性层和Tanh激活函数处理过的序列中第一个标记（分类标记）的最后一层隐藏状态，形状为(batch_size, hidden_size)
    # hidden_states: 可选参数，当output_hidden_states=True时返回，是一个元组，包含每一层的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    # entity_hidden_states: 可选参数，当output_hidden_states=True时返回，是一个元组，包含每一层实体的隐藏状态，形状为(batch_size, entity_length, hidden_size)
    # attentions: 可选参数，当output_attentions=True时返回，是一个元组，包含每一层的注意力权重，形状为(batch_size, num_heads, sequence_length + entity_length, sequence_length + entity_length)

    # 将实体的最后一层隐藏状态初始化为None
    entity_last_hidden_state: torch.FloatTensor = None
    # 将实体的隐藏状态初始化为None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 使用 @dataclass 装饰器定义一个数据类 BaseLukeModelOutput，继承自BaseModelOutput
class BaseLukeModelOutput(BaseModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            
        entity_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, entity_length, hidden_size)`):
            Sequence of entity hidden-states at the output of the last layer of the model.
            
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
            
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义类属性 entity_last_hidden_state，默认值为 None，数据类型为 torch.FloatTensor
    entity_last_hidden_state: torch.FloatTensor = None
    # 定义类属性 entity_hidden_states，默认值为 None，数据类型为 Optional[Tuple[torch.FloatTensor]]
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 使用 @dataclass 装饰器定义一个数据类 LukeMaskedLMOutput，继承自ModelOutput
class LukeMaskedLMOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    # 标记loss，当提供`labels`时返回
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        The sum of masked language modeling (MLM) loss and entity prediction loss.
    
    # 标记mlm_loss，当提供`labels`时返回
    mlm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Masked language modeling (MLM) loss.
    
    # 标记mep_loss，当提供`labels`时返回
    mep_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Masked entity prediction (MEP) loss.
    
    # 标记logits
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
    # 标记entity_logits
    entity_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the entity prediction head (scores for each entity vocabulary token before SoftMax).
    
    # 标记hidden_states，当`output_hidden_states=True`被传递或`config.output_hidden_states=True`时返回
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, sequence_length, hidden_size)`.
        
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
    # 标记entity_hidden_states，当`output_hidden_states=True`被传递或`config.output_hidden_states=True`时返回
    entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
        layer plus the initial entity embedding outputs.
    
    # 标记attentions，当`output_attentions=True`被传递或`config.output_attentions=True`时返回
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
        
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
    """

    # 标记loss，类型为Optional[torch.FloatTensor]
    loss: Optional[torch.FloatTensor] = None
    
    # 标记mlm_loss，类型为Optional[torch.FloatTensor]
    mlm_loss: Optional[torch.FloatTensor] = None
    
    # 标记mep_loss，类型为Optional[torch.FloatTensor]
    mep_loss: Optional[torch.FloatTensor] = None
    
    # 标记logits，类型为torch.FloatTensor
    logits: torch.FloatTensor = None
    
    # 标记entity_logits，类型为torch.FloatTensor
    entity_logits: torch.FloatTensor = None
    
    # 标记hidden_states，类型为Optional[Tuple[torch.FloatTensor]]
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 标记entity_hidden_states，类型为Optional[Tuple[torch.FloatTensor]]
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 标记attentions，类型为Optional[Tuple[torch.FloatTensor]]
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class EntityClassificationOutput(ModelOutput):
    """
    Outputs of entity classification models.

    定义实体分类模型的输出类

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class EntityPairClassificationOutput(ModelOutput):
    """
    Outputs of entity pair classification models.

    定义实体对分类模型的输出类
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.  # 分类损失
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).  # 分类分数（SoftMax之前）
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.  # 隐藏状态，每层的输出（包括初始嵌入输出）
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.  # 实体隐藏状态，每层的输出（包括初始实体嵌入输出）
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.  # 注意力权重，经过注意力Softmax后的值，用于计算自注意力头的加权平均值
    """

    loss: Optional[torch.FloatTensor] = None  # 分类损失，默认为None
    logits: torch.FloatTensor = None  # 分类分数，默认为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏状态，默认为None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 实体隐藏状态，默认为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，默认为None
# 定义了一个名为 EntitySpanClassificationOutput 的数据类，用于存储实体跨度分类模型的输出结果
@dataclass
class EntitySpanClassificationOutput(ModelOutput):
    """
    实体跨度分类模型的输出结果。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 当提供 `labels` 时返回):
            分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, entity_length, config.num_labels)`):
            分类得分（SoftMax 前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            元组 `torch.FloatTensor`（一个用于嵌入输出，每个层输出一个）的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每一层输出的隐藏状态，加上初始嵌入输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            元组 `torch.FloatTensor`（一个用于嵌入输出，每个层输出一个）的形状为 `(batch_size, entity_length, hidden_size)`。
            模型在每一层输出的实体隐藏状态，加上初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            元组 `torch.FloatTensor`（每层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在注意力 SoftMax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LukeSequenceClassifierOutput(ModelOutput):
    """
    句子分类模型的输出结果。
    """
```  
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或者当 `config.num_labels==1` 时是回归）损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或者当 `config.num_labels==1` 时是回归）得分（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            由 `torch.FloatTensor` 组成的元组（如果模型有嵌入层，则包含嵌入层的输出，以及每个层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每个层输出的隐藏状态以及可选的初始嵌入输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            由 `torch.FloatTensor` 组成的元组（包含嵌入层的输出和每个层的输出），形状为 `(batch_size, entity_length, hidden_size)`。模型在每个层输出的实体隐藏状态以及初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            由 `torch.FloatTensor` 组成的元组（每个层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，默认为空
    logits: torch.FloatTensor = None  # 分类（或者回归）得分，默认为空张量
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏状态，默认为空元组
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 实体隐藏状态，默认为空元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，默认为空元组
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import ModelOutput

# 定义一个数据类，用于存储标记分类模型的输出结果
@dataclass
class LukeTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss. 分类损失
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax). 分类分数（SoftMax 之前）
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每个层次的模型隐藏状态，以及可选的初始嵌入输出
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
            每个层次的实体隐藏状态，以及初始实体嵌入输出
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重在注意力 SoftMax 之后的值，用于计算自注意力头部的加权平均值
    """

    # 分类损失
    loss: Optional[torch.FloatTensor] = None
    # 分类分数（SoftMax 之前）
    logits: torch.FloatTensor = None
    # 每个层次的模型隐藏状态，以及可选的初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 每个层次的实体隐藏状态，以及初始实体嵌入输出
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重在注意力 SoftMax 之后的值，用于计算自注意力头部的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LukeQuestionAnsweringModelOutput(ModelOutput):
    """
    Outputs of question answering models.
    """

```py  
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        # loss参数，类型为torch.FloatTensor，形状为(1,)，可选参数，当提供`labels`时返回
        # 总跨度抽取损失是起始和结束位置的交叉熵之和。

        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        # start_logits参数，类型为torch.FloatTensor，形状为(batch_size, sequence_length)
        # 跨度的起始得分（softmax之前）。

        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        # end_logits参数，类型为torch.FloatTensor，形状为(batch_size, sequence_length)
        # 跨度的结束得分（softmax之前）。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            # 隐藏状态参数，类型为元组(torch.FloatTensor)，可选参数，在传递`output_hidden_states=True`或者`config.output_hidden_states=True`时返回
            # torch.FloatTensor元组，如果模型具有嵌入层，则输出嵌入的结果，以及每个层的输出
            # 形状为(batch_size, sequence_length, hidden_size)。

        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
            # 实体隐藏状态参数，类型为元组(torch.FloatTensor)，可选参数，在传递`output_hidden_states=True`或者`config.output_hidden_states=True`时返回
            # torch.FloatTensor元组，一个是嵌入的输出，另一个是每一层的输出，形状为(batch_size, entity_length, hidden_size)。
            # 模型在每一层的实体隐藏状态，还包括初始实体嵌入输出。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            # 注意力参数，类型为元组(torch.FloatTensor)，可选参数，在传递`output_attentions=True`或者`config.output_attentions=True`时返回
            # torch.FloatTensor元组，每层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)。
            # 注意力softmax后的权重，用于在自注意力头中计算加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    # loss参数，类型为Optional[torch.FloatTensor]，默认值为None

    start_logits: torch.FloatTensor = None
    # start_logits参数，类型为torch.FloatTensor，默认值为None

    end_logits: torch.FloatTensor = None
    # end_logits参数，类型为torch.FloatTensor，默认值为None

    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # hidden_states参数，类型为Optional[Tuple[torch.FloatTensor]]，默认值为None

    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # entity_hidden_states参数，类型为Optional[Tuple[torch.FloatTensor]]，默认值为None

    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # attentions参数，类型为Optional[Tuple[torch.FloatTensor]]，默认值为None
# 定义 LukeMultipleChoiceModelOutput 类，用于多项选择模型的输出
@dataclass
class LukeMultipleChoiceModelOutput(ModelOutput):
    """
    多项选择模型的输出。

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* 是输入张量的第二个维度。（参见上面的 *input_ids*）。

            分类分数（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（如果模型有嵌入层则为一个，以及每一层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态加上可选的初始嵌入输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（嵌入层的一个输出加上每一层的输出）的形状为 `(batch_size, entity_length, hidden_size)`。每层模型的实体隐藏状态加上初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义 LukeEmbeddings 类，与 BertEmbeddings 类相同，但对于位置嵌入的索引有微小的调整。
class LukeEmbeddings(nn.Module):
    """
    与 BertEmbeddings 类相同，但对于位置嵌入的索引有微小的调整。
    """
    # 初始化函数，接收一个配置参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 创建词嵌入层，根据词汇表大小、隐藏层大小和填充标记来初始化
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，根据最大位置数和隐藏层大小来初始化
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，根据类型词汇表大小和隐藏层大小来初始化
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 初始化 LayerNorm 层，根据隐藏层大小和层归一化项来初始化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化丢弃层，根据隐藏层丢弃概率来初始化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 终端拷贝
        # 初始化填充标记
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层，根据最大位置数、隐藏层大小和填充标记来初始化
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 正向传播函数
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        # 如果位置标记为空
        if position_ids is None:
            # 如果输入标记不为空
            if input_ids is not None:
                # 从输入标记创建位置标记。任何填充的标记仍然保持填充
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                # 从输入嵌入创建位置标记
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入标记不为空
        if input_ids is not None:
            # 获取输入形状
            input_shape = input_ids.size()
        else:
            # 获取输入嵌入形状
            input_shape = inputs_embeds.size()[:-1]

        # 如果标记类型为空
        if token_type_ids is None:
            # 创建一个全零的标记类型，其形状与输入相同
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为空
        if inputs_embeds is None:
            # 使用词嵌入层获取输入标记的嵌入
            inputs_embeds = self.word_embeddings(input_ids)

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)
        # 获取标记类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算嵌入向量
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 进行层归一化
        embeddings = self.LayerNorm(embeddings)
        # 进行丢弃操作
        embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings

    # 从输入嵌入创建位置标记的函数
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成连续的位置标记
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 返回位置标记
        return position_ids.unsqueeze(0).expand(input_shape)
class LukeEntityEmbeddings(nn.Module):
    # 定义 LukeEntityEmbeddings 类，继承自 nn.Module
    def __init__(self, config: LukeConfig):
        # 构造函数，接受 LukeConfig 类型的参数 config
        super().__init__()
        # 调用父类的构造函数

        self.config = config
        # 将传入的配置参数保存在类的成员变量中

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        # 创建一个嵌入层，输入为实体的词汇大小，输出为实体的嵌入大小，使用零作为填充值的索引

        if config.entity_emb_size != config.hidden_size:
            # 如果实体嵌入的大小与隐藏层大小不相等
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)
            # 创建一个线性变换层，将实体嵌入的大小转换成隐藏层大小，不使用偏置

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建一个位置编码的嵌入层，输入为最大位置编码数，输出为隐藏层大小

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 创建一个 token 类型的嵌入层，输入为 token 类型的词汇大小，输出为隐藏层大小

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 LayerNormalization 层，对隐藏层进行归一化处理，eps 参数为归一化时的偏移量

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个丢弃层，根据隐藏层的概率进行丢弃操作

    def forward(
        self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        # 正向传播函数，接受实体 ID、位置 ID、token 类型 ID 作为输入

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)
            # 如果 token 类型 ID 为空，则使用与实体 ID 相同大小的零张量代替

        entity_embeddings = self.entity_embeddings(entity_ids)
        # 通过实体 ID 查找实体嵌入

        if self.config.entity_emb_size != self.config.hidden_size:
            # 如果实体嵌入的大小不等于隐藏层大小
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)
            # 使用线性变换层将实体嵌入的大小转换成隐藏层大小

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        # 通过位置 ID 查找位置嵌入，并将负值位置 ID 设为零

        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        # 创建一个位置编码的掩码，排除特殊值-1，维度为位置嵌入的维度

        position_embeddings = position_embeddings * position_embedding_mask
        # 将位置嵌入与掩码相乘，过滤无效位置编码

        position_embeddings = torch.sum(position_embeddings, dim=-2)
        # 沿位置 ID 维度求和

        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)
        # 对位置编码做归一化处理

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 通过 token 类型 ID 查找 token 类型嵌入

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        # 将实体嵌入、位置嵌入和 token 类型嵌入相加得到整体嵌入

        embeddings = self.LayerNorm(embeddings)
        # 对整体嵌入进行 LayerNormalization 处理

        embeddings = self.dropout(embeddings)
        # 对处理后的嵌入进行丢弃操作

        return embeddings
        # 返回处理后的嵌入结果
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，如果不是且没有嵌入大小，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 设置是否使用实体感知注意力
        self.use_entity_aware_attention = config.use_entity_aware_attention

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 如果使用实体感知注意力，则创建额外的线性层
        if self.use_entity_aware_attention:
            self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建一个丢弃层，用于在注意力计算中应用丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量重排为适合注意力计算的形状
    def transpose_for_scores(self, x):
        # 计算新的形状，保持批次大小不变，但增加了注意力头数和每个头的大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重排张量以匹配新形状
        x = x.view(*new_x_shape)
        # 调换维度以匹配注意力计算所需的形状
        return x.permute(0, 2, 1, 3)

    # 正向传播函数，用于执行模型的前向计算
    def forward(
        self,
        word_hidden_states,  # 单词隐藏状态
        entity_hidden_states,  # 实体隐藏状态
        attention_mask=None,  # 注意力掩码
        head_mask=None,  # 头部掩码
        output_attentions=False,  # 是否输出注意力权重
# 从transformers.models.bert.modeling_bert.BertSelfOutput中复制代码
class LukeSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用全连接层处理输入，输出维度与配置中隐藏层大小相同
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm层，对隐藏层进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层，对隐藏层进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏层状态
        hidden_states = self.dense(hidden_states)
        # Dropout层处理隐藏层状态
        hidden_states = self.dropout(hidden_states)
        # LayerNorm层对处理后的隐藏层状态进行归一化，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertIntermediate中复制代码
class LukeIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用全连接层处理输入，输出维度为中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串，则使用预定义的激活函数映射表中的激活函数，否则使用配置中定义的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏层状态
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理全连接层输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput中复制代码
class LukeOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入大小为 config.hidden_size，eps 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个张量参数，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用 dense 层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 层处理 hidden_states
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 层对 hidden_states 和 input_tensor 进行残差连接和 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的 hidden_states
        return hidden_states
# LukeLayer 是一个 PyTorch 模块，它定义了一个 Transformer 层
class LukeLayer(nn.Module):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 feed-forward 子层的 chunk 大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度为 1
        self.seq_len_dim = 1
        # 创建注意力子层
        self.attention = LukeAttention(config)
        # 创建中间子层
        self.intermediate = LukeIntermediate(config)
        # 创建输出子层
        self.output = LukeOutput(config)

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 获取 word 隐状态的长度
        word_size = word_hidden_states.size(1)

        # 计算注意力输出
        self_attention_outputs = self.attention(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )

        # 根据是否有 entity 隐状态拼接注意力输出
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = torch.cat(self_attention_outputs[:2], dim=1)

        # 添加注意力权重输出（如果有的话）
        outputs = self_attention_outputs[2:]

        # 应用分块前馈神经网络
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concat_attention_output
        )
        # 分割 word 和 entity 输出
        word_layer_output = layer_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        # 组合输出
        outputs = (word_layer_output, entity_layer_output) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 计算中间层输出
        intermediate_output = self.intermediate(attention_output)
        # 计算输出层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# LukeEncoder 是一个 PyTorch 模块，它定义了 LUKE 模型的编码器
class LukeEncoder(nn.Module):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        self.config = config
        # 创建多个 LukeLayer 层
        self.layer = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置是否使用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 省略后续代码
        # 初始化保存所有单词隐藏状态的元组，如果不需要输出隐藏状态则为 None
        all_word_hidden_states = () if output_hidden_states else None
        # 初始化保存所有实体隐藏状态的元组，如果不需要输出隐藏状态则为 None
        all_entity_hidden_states = () if output_hidden_states else None
        # 初始化保存所有自注意力值的元组，如果不需要输出注意力值则为 None
        all_self_attentions = () if output_attentions else None

        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前步骤的隐藏状态保存到相应的元组中
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            # 获取当前层的头部遮罩
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果启用梯度检查点且处于训练状态，则使用梯度检查点函数计算当前层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向计算
                layer_outputs = layer_module(
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            # 更新当前步骤的单词隐藏状态
            word_hidden_states = layer_outputs[0]

            # 如果存在实体隐藏状态，则更新当前步骤的实体隐藏状态
            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            # 如果需要输出注意力值，则将当前步骤的注意力值保存到相应的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态保存到相应的元组中
        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

        # 如果不需要返回字典形式，则返回包含所有结果的元组
        if not return_dict:
            return tuple(
                v
                for v in [
                    word_hidden_states,
                    all_word_hidden_states,
                    all_self_attentions,
                    entity_hidden_states,
                    all_entity_hidden_states,
                ]
                if v is not None
            )
        # 返回以 BaseLukeModelOutput 格式组织的结果
        return BaseLukeModelOutput(
            last_hidden_state=word_hidden_states,
            hidden_states=all_word_hidden_states,
            attentions=all_self_attentions,
            entity_last_hidden_state=entity_hidden_states,
            entity_hidden_states=all_entity_hidden_states,
        )
# 这是一个 LukePooler 类，它继承自 nn.Module。它的作用是对输入的隐藏状态进行池化操作，获取第一个token的隐藏状态作为输出。
class LukePooler(nn.Module):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 定义一个线性层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 Tanh 激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出隐藏状态的第一个token的向量
        first_token_tensor = hidden_states[:, 0]
        # 通过线性层和激活函数处理第一个token的向量
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        # 返回处理后的向量
        return pooled_output


# EntityPredictionHeadTransform 类用于对实体预测头的隐藏状态进行变换
class EntityPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 定义一个线性层，输入维度是 config.hidden_size，输出维度是 config.entity_emb_size
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        # 获取激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 定义一个LayerNorm层
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 先通过线性层和激活函数处理输入
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # 再通过LayerNorm层处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回变换后的隐藏状态
        return hidden_states


# EntityPredictionHead 类用于实体预测任务的预测头
class EntityPredictionHead(nn.Module):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        self.config = config
        # 定义一个 EntityPredictionHeadTransform 模块
        self.transform = EntityPredictionHeadTransform(config)
        # 定义一个线性层，用于从 entity_emb_size 维度映射到 entity_vocab_size 维度
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        # 定义一个 bias 参数
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

    def forward(self, hidden_states):
        # 先通过 EntityPredictionHeadTransform 模块处理输入
        hidden_states = self.transform(hidden_states)
        # 再通过线性层和 bias 得到最终的预测结果
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


# LukePreTrainedModel 是一个抽象基类，用于处理模型权重初始化和加载预训练模型
class LukePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 指定配置类为 LukeConfig
    config_class = LukeConfig
    # 指定基模型前缀为 "luke"
    base_model_prefix = "luke"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 指定模型中不需要分割的模块
    _no_split_modules = ["LukeAttention", "LukeEntityEmbeddings"]
    # 初始化模型的权重
    def _init_weights(self, module: nn.Module):
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果偏置项存在，则将其设为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 如果是用于偏置参数的嵌入，则将权重设为0
            if module.embedding_dim == 1:  
                module.weight.data.zero_()
            # 否则，使用正态分布初始化权重
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引对应的权重设为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项设为0
            module.bias.data.zero_()
            # 将权重设为1
            module.weight.data.fill_(1.0)
# LUKE 模型的起始文档字符串，提供了有关该模型的一些常规信息
LUKE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LukeConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# LUKE 模型输入的文档字符串，描述了模型的输入
LUKE_INPUTS_DOCSTRING = r"""
"""

# 用于添加文档字符串的装饰器
@add_start_docstrings(
    "The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities without any"
    " specific head on top.",
    LUKE_START_DOCSTRING,
)
# LukeModel 类定义
class LukeModel(LukePreTrainedModel):
    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        # 调用父类的构造函数
        super().__init__(config)
        self.config = config

        # 创建 LUKE 词嵌入层
        self.embeddings = LukeEmbeddings(config)
        # 创建 LUKE 实体嵌入层
        self.entity_embeddings = LukeEntityEmbeddings(config)
        # 创建 LUKE 编码器
        self.encoder = LukeEncoder(config)

        # 根据配置决定是否创建池化层
        self.pooler = LukePooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取词嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置词嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 获取实体嵌入层
    def get_entity_embeddings(self):
        return self.entity_embeddings.entity_embeddings

    # 设置实体嵌入层
    def set_entity_embeddings(self, value):
        self.entity_embeddings.entity_embeddings = value

    # 不支持裁剪注意力头
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    # 添加文档字符串到模型的 forward 方法
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseLukeModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的词元 ID，可选的长整型张量，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，可选的浮点张量，默认为 None
        token_type_ids: Optional[torch.LongTensor] = None,  # 词元类型 ID，可选的长整型张量，默认为 None
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID，可选的长整型张量，默认为 None
        entity_ids: Optional[torch.LongTensor] = None,  # 实体 ID，可选的长整型张量，默认为 None
        entity_attention_mask: Optional[torch.FloatTensor] = None,  # 实体的注意力掩码，可选的浮点张量，默认为 None
        entity_token_type_ids: Optional[torch.LongTensor] = None,  # 实体的词元类型 ID，可选的长整型张量，默认为 None
        entity_position_ids: Optional[torch.LongTensor] = None,  # 实体的位置 ID，可选的长整型张量，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，可选的浮点张量，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入，可选的浮点张量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为 None
    # 定义一个方法用于生成扩展的注意力掩码
    def get_extended_attention_mask(
        self, word_attention_mask: torch.LongTensor, entity_attention_mask: Optional[torch.LongTensor]
    ):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            word_attention_mask (`torch.LongTensor`):
                Attention mask for word tokens with ones indicating tokens to attend to, zeros for tokens to ignore.
                词元的注意力掩码，其中 1 表示要关注的词元，0 表示要忽略的词元。
            entity_attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for entity tokens with ones indicating tokens to attend to, zeros for tokens to ignore.
                实体词元的注意力掩码，其中 1 表示要关注的词元，0 表示要忽略的词元。

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
            扩展后的注意力掩码，与 attention_mask.dtype 相同的数据类型。
        """
        # 将词元的注意力掩码赋值给注意力掩码
        attention_mask = word_attention_mask
        # 如果存在实体的注意力掩码
        if entity_attention_mask is not None:
            # 将实体的注意力掩码与词元的注意力掩码在最后一个维度上拼接起来
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=-1)

        # 如果注意力掩码的维度为 3
        if attention_mask.dim() == 3:
            # 在第二个维度上添加一个维度，形成广播后的注意力掩码
            extended_attention_mask = attention_mask[:, None, :, :]
        # 如果注意力掩码的维度为 2
        elif attention_mask.dim() == 2:
            # 在第二个和第三个维度上各添加一个维度，形成广播后的注意力掩码
            extended_attention_mask = attention_mask[:, None, None, :]
        # 如果注意力掩码的维度既不是 3 也不是 2
        else:
            # 抛出 ValueError 异常，提示注意力掩码的形状错误
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        # 将扩展后的注意力掩码转换为指定数据类型，确保与模型中的数据类型兼容（用于 fp16 兼容性）
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        # 将掩码中的 1 替换为负无穷，确保对未来词元和被屏蔽词元的忽略
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        # 返回扩展后的注意力掩码
        return extended_attention_mask
# 根据输入的input_ids和padding_idx创建位置ID：非填充符号替换为它们的位置编号，位置编号从padding_idx+1开始，忽略填充符号。这是从fairseq的`utils.make_positions`修改而来。
def create_position_ids_from_input_ids(input_ids, padding_idx):
    # 这里的一系列强制转换和类型转换是精心平衡的，既可以与ONNX导出一起工作，也可以与XLA一起工作。
    # 用input_ids中不等于padding_idx的元素创建掩码（mask）
    mask = input_ids.ne(padding_idx).int()
    # 先创建一个累积的掩码（mask）然后类型转换，再乘以掩码（mask）
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    # 返回长整型的incremental_indices和padding_idx的和
    return incremental_indices.long() + padding_idx


# 从transformers.models.roberta.modeling_roberta.RobertaLMHead复制的
# 用于遮蔽语言建模的Roberta头
class LukeLMHead(nn.Module):
    """用于遮蔽语言建模的Roberta头"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)  # 使用GELU激活函数
        x = self.layer_norm(x)

        # 通过偏置将其投影回词汇表的大小
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果这两个权重被断开连接（在TPU上或当偏置被调整大小时），将它们绑定在一起
        # 用于加速兼容性和不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    LUKE模型带有用于遮蔽语言建模和遮蔽实体预测的语言建模头和实体预测头。
    """,
    LUKE_START_DOCSTRING,
)
class LukeForMaskedLM(LukePreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias", "entity_predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)

        self.lm_head = LukeLMHead(config)
        self.entity_predictions = EntityPredictionHead(config)

        self.loss_fn = nn.CrossEntropyLoss()

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        super().tie_weights()
        # 绑定或克隆权重
        self._tie_or_clone_weights(self.entity_predictions.decoder, self.luke.entity_embeddings.entity_embeddings)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器替换返回文档字符串，指定输出类型和配置类
    @replace_return_docstrings(output_type=LukeMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，并接收以下参数

    # 输入的标记 ID，数据类型为可选的长整型张量
    input_ids: Optional[torch.LongTensor] = None,
    # 注意力遮罩，数据类型为可选的浮点张量
    attention_mask: Optional[torch.FloatTensor] = None,
    # 标记类型 ID，数据类型为可选的长整型张量
    token_type_ids: Optional[torch.LongTensor] = None,
    # 位置 ID，数据类型为可选的长整型张量
    position_ids: Optional[torch.LongTensor] = None,
    # 实体 ID，数据类型为可选的长整型张量
    entity_ids: Optional[torch.LongTensor] = None,
    # 实体注意力遮罩，数据类型为可选的长整型张量
    entity_attention_mask: Optional[torch.LongTensor] = None,
    # 实体标记类型 ID，数据类型为可选的长整型张量
    entity_token_type_ids: Optional[torch.LongTensor] = None,
    # 实体位置 ID，数据类型为可选的长整型张量
    entity_position_ids: Optional[torch.LongTensor] = None,
    # 标签，数据类型为可选的长整型张量
    labels: Optional[torch.LongTensor] = None,
    # 实体标签，数据类型为可选的长整型张量
    entity_labels: Optional[torch.LongTensor] = None,
    # 头部遮罩，数据类型为可选的浮点张量
    head_mask: Optional[torch.FloatTensor] = None,
    # 输入嵌入，数据类型为可选的浮点张量
    inputs_embeds: Optional[torch.FloatTensor] = None,
    # 输出注意力，数据类型为可选的布尔值
    output_attentions: Optional[bool] = None,
    # 输出隐藏状态，数据类型为可选的布尔值
    output_hidden_states: Optional[bool] = None,
    # 返回字典，数据类型为可选的布尔值
    return_dict: Optional[bool] = None,
# 在 LUKE 模型的基础上添加分类头部，用于实体分类任务，例如 Open Entity
@add_start_docstrings(
    """
    The LUKE model with a classification head on top (a linear layer on top of the hidden state of the first entity
    token) for entity classification tasks, such as Open Entity.
    """,
    LUKE_START_DOCSTRING,
)
class LukeForEntityClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 LUKE 模型
        self.luke = LukeModel(config)

        # 设置分类数目和 dropout 层
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器是一个线性层，将隐藏状态映射到标签空间
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多种输入，并返回输出
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntityClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 在 LUKE 模型的基础上添加分类头部，用于实体对分类任务，例如 TACRED
@add_start_docstrings(
    """
    The LUKE model with a classification head on top (a linear layer on top of the hidden states of the two entity
    tokens) for entity pair classification tasks, such as TACRED.
    """,
    LUKE_START_DOCSTRING,
)
class LukeForEntityPairClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 LUKE 模型
        self.luke = LukeModel(config)

        # 设置分类数目和 dropout 层
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器是一个线性层，将两个实体的隐藏状态映射到标签空间
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels, False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多种输入，并返回输出
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntityPairClassificationOutput, config_class=_CONFIG_FOR_DOC)
```  
    # 此方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列，类型为可选的长整型张量，默认为None
        attention_mask: Optional[torch.FloatTensor] = None,  # 用于指示哪些位置需要被忽略的注意力掩码，类型为可选的浮点数张量，默认为None
        token_type_ids: Optional[torch.LongTensor] = None,  # 输入的token类型ID序列，类型为可选的长整型张量，默认为None
        position_ids: Optional[torch.LongTensor] = None,  # 输入的token位置ID序列，类型为可选的长整型张量，默认为None
        entity_ids: Optional[torch.LongTensor] = None,  # 实体ID序列，类型为可选的长整型张量，默认为None
        entity_attention_mask: Optional[torch.FloatTensor] = None,  # 实体的注意力掩码，类型为可选的浮点数张量，默认为None
        entity_token_type_ids: Optional[torch.LongTensor] = None,  # 实体的token类型ID序列，类型为可选的长整型张量，默认为None
        entity_position_ids: Optional[torch.LongTensor] = None,  # 实体的token位置ID序列，类型为可选的长整型张量，默认为None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，类型为可选的浮点数张量，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入张量，类型为可选的浮点数张量，默认为None
        labels: Optional[torch.LongTensor] = None,  # 标签序列，类型为可选的长整型张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
# 为LUKE模型添加一个基于span分类的头部（线性层叠加在隐藏状态输出之上），用于诸如命名实体识别之类的任务
@add_start_docstrings(
    """
    The LUKE model with a span classification head on top (a linear layer on top of the hidden states output) for tasks
    such as named entity recognition.
    """,
    LUKE_START_DOCSTRING,
)
class LukeForEntitySpanClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化LUKE模型
        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntitySpanClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.LongTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        entity_start_positions: Optional[torch.LongTensor] = None,
        entity_end_positions: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 为LUKE模型添加一个基于序列分类/回归的头部（线性层叠加在汇总输出之上），用于GLUE任务等
@add_start_docstrings(
    """
    The LUKE Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LUKE_START_DOCSTRING,
)
class LukeForSequenceClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        # 初始化LUKE模型
        self.luke = LukeModel(config)
        # 配置Dropout层
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义forward方法，用于模型的前向传播
    def forward(
        # 输入的token id序列，类型为LongTensor，可选参数
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，用于指示模型在输入中要关注的位置，类型为FloatTensor，可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        # token类型id序列，用于指示不同句子的token属于哪个句子，类型为LongTensor，可选参数
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置id序列，用于指示每个token在输入中的位置，类型为LongTensor，可选参数
        position_ids: Optional[torch.LongTensor] = None,
        # 实体id序列，用于标识实体，类型为LongTensor，可选参数
        entity_ids: Optional[torch.LongTensor] = None,
        # 实体注意力掩码，用于指示模型在实体id位置要关注的位置，类型为FloatTensor，可选参数
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        # 实体token类型id序列，用于指示不同实体的token属于哪个实体，类型为LongTensor，可选参数
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        # 实体位置id序列，用于指示每个实体token在输入中的位置，类型为LongTensor，可选参数
        entity_position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，用于在多头注意力中指定哪些头部应该被忽略，类型为FloatTensor，可选参数
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入的嵌入向量，类型为FloatTensor，可选参数
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，用于计算损失，类型为FloatTensor，可选参数
        labels: Optional[torch.FloatTensor] = None,
        # 是否输出注意力权重，类型为bool，可选参数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为bool，可选参数
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回结果，类型为bool，可选参数
        return_dict: Optional[bool] = None,
# 导入所需模块或类
@add_start_docstrings(
    """
    The LUKE Model with a token classification head on top (a linear layer on top of the hidden-states output). To
    solve Named-Entity Recognition (NER) task using LUKE, `LukeForEntitySpanClassification` is more suitable than this
    class.
    """,
    LUKE_START_DOCSTRING,
)
# 定义一个继承自 LukePreTrainedModel 的类 LukeForTokenClassification
class LukeForTokenClassification(LukePreTrainedModel):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 LUKE 模型对象，不添加池化层
        self.luke = LukeModel(config, add_pooling_layer=False)
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个全连接层，用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数的类型和含义
    # 这个函数是一个 PyTorch 模型的前向传播函数。它接收一些输入张量，
    # 并返回预测结果和一些可选的辅助输出。
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        entity_attention_mask: Optional[torch.Tensor] = None,
        entity_token_type_ids: Optional[torch.Tensor] = None,
        entity_position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LukeTokenClassifierOutput]:
        # 根据输入参数的情况，设置是否使用返回字典的模式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入传递给 self.luke 模型,获得输出
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    
        # 从模型输出中获取序列输出
        sequence_output = outputs.last_hidden_state
    
        # 对序列输出应用dropout
        sequence_output = self.dropout(sequence_output)
    
        # 将序列输出传递给分类器,得到logits
        logits = self.classifier(sequence_output)
    
        # 如果提供了标签,则计算交叉熵损失
        loss = None
        if labels is not None:
            # 将标签移动到logits的设备上,以支持模型并行
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 根据返回模式,返回相应的输出
        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )
    
        return LukeTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )
# 在 LUKE 模型的基础上添加一个用于抽取式问答任务的跨度分类头部
# 包括一个线性层，用于计算“跨度起始对数”和“跨度终点对数”以及隐藏状态输出的线性层。
class LukeForQuestionAnswering(LukePreTrainedModel):
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        
        # 记录标签数量
        self.num_labels = config.num_labels

        # 初始化 LUKE 模型，并禁用添加池化层
        self.luke = LukeModel(config, add_pooling_layer=False)
        
        # 添加一个线性层用于 QA 输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self, ...



# 在 LUKE 模型的基础上添加一个用于多项选择分类任务头部
# 包括一个线性层，用于在汇总输出上进行多项选择分类的 softmax，例如 RocStories/SWAG 任务。
class LukeForMultipleChoice(LukePreTrainedModel):
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 初始化 LUKE 模型
        self.luke = LukeModel(config)
        
        # 添加一个丢弃层，使用配置中的分类器丢弃率或隐藏层丢弃率，如果没有指定则使用默认值
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        # 添加一个线性层用于分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个前向传播函数，接受多个输入参数，都是可选的 Torch 张量
    def forward(
        # 输入序列的 ID，是一个长整型 Torch 张量
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，是一个浮点数 Torch 张量
        attention_mask: Optional[torch.FloatTensor] = None,
        # 标记类型 ID，是一个长整型 Torch 张量
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 ID，是一个长整型 Torch 张量
        position_ids: Optional[torch.LongTensor] = None,
        # 实体 ID，是一个长整型 Torch 张量
        entity_ids: Optional[torch.LongTensor] = None,
        # 实体注意力掩码，是一个浮点数 Torch 张量
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        # 实体标记类型 ID，是一个长整型 Torch 张量
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        # 实体位置 ID，是一个长整型 Torch 张量
        entity_position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，是一个浮点数 Torch 张量
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入特征，是一个浮点数 Torch 张量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，是一个浮点数 Torch 张量
        labels: Optional[torch.FloatTensor] = None,
        # 是否输出注意力矩阵，是一个布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，是一个布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典结果，是一个布尔值
        return_dict: Optional[bool] = None,
```