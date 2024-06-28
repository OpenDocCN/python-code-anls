# `.\models\luke\modeling_luke.py`

```
# coding=utf-8
# Copyright Studio Ousia and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LUKE model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点
_CONFIG_FOR_DOC = "LukeConfig"
_CHECKPOINT_FOR_DOC = "studio-ousia/luke-base"

# 预训练模型存档列表
LUKE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "studio-ousia/luke-base",
    "studio-ousia/luke-large",
    # 查看所有 LUKE 模型：https://huggingface.co/models?filter=luke
]


@dataclass
class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Base class for outputs of the LUKE model.
    """
    # 定义函数的参数及其类型说明
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        entity_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, entity_length, hidden_size)`):
            实体的最后一层隐藏状态序列。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            经过线性层和Tanh激活函数处理过的序列中第一个标记（分类标记）的最后一层隐藏状态。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含模型每一层的隐藏状态（从嵌入层开始，每层一个张量），形状为 `(batch_size, sequence_length, hidden_size)`。
            当 `output_hidden_states=True` 时返回。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含实体的每一层隐藏状态（从嵌入层开始，每层一个张量），形状为 `(batch_size, entity_length, hidden_size)`。
            当 `output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含每一层的注意力权重张量，形状为 `(batch_size, num_heads, sequence_length + entity_length, sequence_length + entity_length)`。
            注意力权重经过 softmax 处理，用于计算自注意力头中的加权平均值。
    
    
    
    # 声明实体的最后隐藏状态和实体隐藏状态，类型分别为 torch.FloatTensor 和 Optional[Tuple[torch.FloatTensor, ...]]
    entity_last_hidden_state: torch.FloatTensor = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器定义一个名为 BaseLukeModelOutput 的数据类，作为模型输出的基类
@dataclass
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
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义实体最后隐藏状态，类型为 torch.FloatTensor，默认为 None
    entity_last_hidden_state: torch.FloatTensor = None
    # 定义实体隐藏状态的元组，类型为 Optional[Tuple[torch.FloatTensor, ...]]，默认为 None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            The sum of masked language modeling (MLM) loss and entity prediction loss.
            当提供 `labels` 参数时返回，表示掩码语言建模（MLM）损失和实体预测损失的总和。
        mlm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
            当提供 `labels` 参数时返回，表示掩码语言建模（MLM）损失。
        mep_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked entity prediction (MEP) loss.
            当提供 `labels` 参数时返回，表示掩码实体预测（MEP）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言建模头部的预测分数（SoftMax 之前的每个词汇标记的分数）。
        entity_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the entity prediction head (scores for each entity vocabulary token before SoftMax).
            实体预测头部的预测分数（SoftMax 之前的每个实体词汇标记的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            模型在每个层输出后的隐藏状态的元组，包括初始嵌入输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`.
            模型在每个层输出后的实体隐藏状态的元组，包括初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            注意力权重在经过注意力 softmax 后的结果，用于计算自注意力头部的加权平均值。
# 定义一个数据类 EntityClassificationOutput，继承自 ModelOutput
@dataclass
class EntityClassificationOutput(ModelOutput):
    """
    实体分类模型的输出结果。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类得分（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含每一层的输出的元组 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每一层的隐藏状态加上初始嵌入的输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            实体的隐藏状态的元组 `torch.FloatTensor`，形状为 `(batch_size, entity_length, hidden_size)`。
            模型每一层的实体隐藏状态加上初始实体嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力头中用于计算加权平均的注意力权重的元组 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力 softmax 后的注意力权重。

    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类损失。
            如果提供了 `labels` 参数，则返回分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类分数（SoftMax 之前的输出）。
            形状为 `(batch_size, config.num_labels)` 的分类分数。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型隐藏状态，包括每一层的输出和初始嵌入的输出。
            形状为 `(batch_size, sequence_length, hidden_size)` 的元组，第一个元素是嵌入的输出，后续元素是每一层的输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            实体的隐藏状态，包括每一层的输出和初始实体嵌入的输出。
            形状为 `(batch_size, entity_length, hidden_size)` 的元组，第一个元素是实体嵌入的输出，后续元素是每一层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重，用于计算自注意力头中的加权平均值。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的元组，每个元素对应一个层的注意力权重。
    """

    # 可选的分类损失，形状为 `(1,)` 的 `torch.FloatTensor`
    loss: Optional[torch.FloatTensor] = None
    # 分类分数（SoftMax 之前的输出），形状为 `(batch_size, config.num_labels)` 的 `torch.FloatTensor`
    logits: torch.FloatTensor = None
    # 可选的模型隐藏状态元组，包含每层的输出和初始嵌入的输出
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的实体隐藏状态元组，包含每层的输出和初始实体嵌入的输出
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的注意力权重元组，用于计算自注意力头中的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 定义一个数据类，用于存储实体跨度分类模型的输出结果
@dataclass
class EntitySpanClassificationOutput(ModelOutput):
    """
    实体跨度分类模型的输出结果。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 当提供 `labels` 时返回):
            分类损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, entity_length, config.num_labels)`):
            分类分数（SoftMax 之前的）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            由两部分组成的元组 `torch.FloatTensor` （一个用于嵌入的输出 + 一个用于每层输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每层输出结束时的隐藏状态，加上初始嵌入输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            由两部分组成的元组 `torch.FloatTensor` （一个用于嵌入的输出 + 一个用于每层输出），形状为 `(batch_size, entity_length, hidden_size)`。
            模型在每层输出结束时实体的隐藏状态，加上初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            由每一层的 `torch.FloatTensor` 组成的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """


@dataclass
class LukeSequenceClassifierOutput(ModelOutput):
    """
    句子分类模型的输出结果。
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 `config.num_labels==1`）的损失值。
            如果提供 `labels`，则返回损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 `config.num_labels==1`）的分数（SoftMax 之前）。
            模型的输出分数。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层的隐藏状态以及可选的初始嵌入输出的元组。
            形状为 `(batch_size, sequence_length, hidden_size)`。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            实体的隐藏状态，包括每一层的输出以及初始实体嵌入输出的元组。
            形状为 `(batch_size, entity_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，每个层级一个。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在自注意力机制中用于计算加权平均值的注意力权重。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器定义一个名为 LukeTokenClassifierOutput 的数据类，它继承自 ModelOutput。
@dataclass
class LukeTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
            分类损失，当提供 `labels` 参数时返回，数据类型为 `torch.FloatTensor`，形状为 `(1,)`。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
            分类分数（SoftMax 之前的输出），数据类型为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, config.num_labels)`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            可选项。当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
            返回一个元组，包含 `torch.FloatTensor` 类型的张量（如果模型有嵌入层，则包含嵌入层的输出，
            否则包含每一层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output of each
            layer plus the initial entity embedding outputs.
            可选项。当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
            返回一个元组，包含 `torch.FloatTensor` 类型的张量（一个用于嵌入层的输出，另一个包含每一层的输出），
            形状为 `(batch_size, entity_length, hidden_size)`。用于表示每一层的实体隐藏状态以及初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            可选项。当传入 `output_attentions=True` 或 `config.output_attentions=True` 时返回，
            返回一个元组，包含 `torch.FloatTensor` 类型的张量（每一层一个），形状为 `(batch_size, num_heads, sequence_length,
            sequence_length)`。表示经过注意力 softmax 后的注意力权重，用于计算自注意力头部的加权平均值。
    """
    
    # 分类损失，数据类型为 `torch.FloatTensor`，可选项。
    loss: Optional[torch.FloatTensor] = None
    # 分类分数（SoftMax 之前的输出），数据类型为 `torch.FloatTensor`，默认为 `None`。
    logits: torch.FloatTensor = None
    # 隐藏状态，数据类型为 `tuple`，包含 `torch.FloatTensor` 的元组，可选项。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 实体隐藏状态，数据类型为 `tuple`，包含 `torch.FloatTensor` 的元组，可选项。
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 注意力权重，数据类型为 `tuple`，包含 `torch.FloatTensor` 的元组，可选项。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 dataclass 装饰器定义一个名为 LukeQuestionAnsweringModelOutput 的数据类，它继承自 ModelOutput。
@dataclass
class LukeQuestionAnsweringModelOutput(ModelOutput):
    """
    Outputs of question answering models.
    """

    # 这里将不添加额外的注释，因为类本身并没有额外的字段或说明。
    # 损失函数（如果提供了`labels`则会返回），总体抽取损失由起始和结束位置的交叉熵之和组成
    loss: Optional[torch.FloatTensor] = None

    # 起始位置的分数（softmax之前的张量），形状为`(batch_size, sequence_length)`
    start_logits: torch.FloatTensor = None

    # 结束位置的分数（softmax之前的张量），形状为`(batch_size, sequence_length)`
    end_logits: torch.FloatTensor = None

    # 模型每层的隐藏状态的元组（如果设置了`output_hidden_states=True`或`config.output_hidden_states=True`时返回）
    # 包含嵌入层的输出（如果模型有嵌入层）和每层的输出，形状为`(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    # 实体每层的隐藏状态的元组（如果设置了`output_hidden_states=True`或`config.output_hidden_states=True`时返回）
    # 包含嵌入层的输出和每层的输出，形状为`(batch_size, entity_length, hidden_size)`
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    # 注意力权重的元组（如果设置了`output_attentions=True`或`config.output_attentions=True`时返回）
    # 每层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 定义一个数据类 LukeMultipleChoiceModelOutput，继承自 ModelOutput
@dataclass
class LukeMultipleChoiceModelOutput(ModelOutput):
    """
    多选模型的输出结果。

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            `num_choices` 是输入张量的第二个维度。参见上文中的 `input_ids`。

            分类分数（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor`（如果模型有嵌入层，则包含嵌入层输出，以及每一层的输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态，以及可选的初始嵌入输出。
        entity_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor`（包含嵌入层输出以及每一层的输出），
            形状为 `(batch_size, entity_length, hidden_size)`。

            模型在每一层输出的实体隐藏状态，以及初始实体嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `torch.FloatTensor`（每一层一个），
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力机制 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 可选的分类损失值
    loss: Optional[torch.FloatTensor] = None
    # 分类分数（SoftMax 之前）
    logits: torch.FloatTensor = None
    # 可选的隐藏状态元组
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的实体隐藏状态元组
    entity_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的注意力权重元组
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义一个类 LukeEmbeddings，与 BertEmbeddings 类相同，但稍作修改以支持位置嵌入索引
class LukeEmbeddings(nn.Module):
    """
    与 BertEmbeddings 类似，但稍作修改以支持位置嵌入索引。
    """
    def __init__(self, config):
        super().__init__()
        # 定义词嵌入层，根据配置参数创建词嵌入矩阵，大小为 vocab_size * hidden_size，其中使用 pad_token_id 进行填充
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，根据配置参数创建位置嵌入矩阵，大小为 max_position_embeddings * hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 定义类型嵌入层，根据配置参数创建类型嵌入矩阵，大小为 type_vocab_size * hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用 TensorFlow 模型变量名命名的 LayerNorm 层，用于标准化隐藏层输出，eps 为配置参数中的 layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义 Dropout 层，用于随机失活，丢弃概率为 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 设置 padding_idx 属性，用于输入序列中的填充标记
        self.padding_idx = config.pad_token_id
        # 重新定义位置嵌入层，使用与输入序列相同的参数配置，并指定填充标记
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # 根据输入的 token ids 创建位置 ids，保持填充的 token 仍然是填充状态
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                # 如果没有输入 token ids，则根据 inputs_embeds 创建位置 ids
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            # 如果没有指定 token_type_ids，则创建全零的 token 类型 ids
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果没有提供 inputs_embeds，则使用 word_embeddings 对 input_ids 进行嵌入
            inputs_embeds = self.word_embeddings(input_ids)

        # 根据位置 ids 获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 根据 token_type_ids 获取类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入的嵌入向量、位置嵌入向量和类型嵌入向量相加
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 对合并后的嵌入向量进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对处理后的向量进行随机失活
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成与输入嵌入向量维度相匹配的位置 ids
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
class LukeEntityEmbeddings(nn.Module):
    # Luke 实体嵌入模块，继承自 nn.Module
    def __init__(self, config: LukeConfig):
        super().__init__()
        self.config = config

        # 实体嵌入层，使用 nn.Embedding 创建一个词汇表大小为 config.entity_vocab_size，
        # 嵌入维度为 config.entity_emb_size 的嵌入层，padding 索引为 0
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        
        # 如果实体嵌入维度不等于隐藏层大小，创建一个线性层用于将实体嵌入维度转换到隐藏层大小
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        # 位置嵌入层，使用 nn.Embedding 创建一个最大位置嵌入大小为 config.max_position_embeddings，
        # 嵌入维度为 config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # 类型嵌入层，使用 nn.Embedding 创建一个类型词汇表大小为 config.type_vocab_size，
        # 嵌入维度为 config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Layer normalization 层，输入大小为 config.hidden_size，epsilon 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout 层，概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(
        self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        # 如果 token_type_ids 为 None，则初始化为全零的与 entity_ids 大小相同的张量
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        # 获取实体嵌入，通过实体嵌入层获取对应的实体嵌入向量
        entity_embeddings = self.entity_embeddings(entity_ids)
        
        # 如果实体嵌入维度不等于隐藏层大小，则通过线性层进行维度转换
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        # 获取位置嵌入，通过位置嵌入层获取对应的位置嵌入向量，同时根据位置索引进行裁剪和掩码处理
        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        # 获取类型嵌入，通过类型嵌入层获取对应的类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将实体嵌入、位置嵌入和类型嵌入进行相加得到最终的嵌入向量
        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        
        # 应用 Layer normalization 层
        embeddings = self.LayerNorm(embeddings)
        
        # 应用 Dropout 层
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量
        return embeddings
    # 初始化方法，接受一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        
        # 检查隐藏层大小是否是注意力头数的整数倍，且不存在嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不是整数倍则抛出数值错误异常
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        
        # 将注意力头数和每个注意力头的大小设置为配置中的值
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 根据配置设置是否使用实体感知注意力
        self.use_entity_aware_attention = config.use_entity_aware_attention

        # 创建用于查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 如果使用实体感知注意力，则额外创建实体到词的查询层和实体到实体的查询层
        if self.use_entity_aware_attention:
            self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建注意力概率的丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量转换为适合注意力分数计算的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播方法定义，接受词的隐藏状态、实体的隐藏状态等参数
    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput

class LukeSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，对输入的最后一个维度进行归一化，eps 是归一化的参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，根据 config.hidden_dropout_prob 概率丢弃输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 归一化并加上输入 tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # LUKE 自注意力机制模块
        self.self = LukeSelfAttention(config)
        # LUKE 自注意力输出模块
        self.output = LukeSelfOutput(config)
        # 头部剪枝集合，用于存储不参与注意力计算的头部
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # LUKE 不支持注意力头部的剪枝操作，抛出未实现错误
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 计算 word_hidden_states 的第二个维度大小
        word_size = word_hidden_states.size(1)
        # 使用 LUKE 自注意力机制进行计算
        self_outputs = self.self(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 如果 entity_hidden_states 为 None，则将 self_outputs 的第一个元素作为 concat_self_outputs
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_hidden_states
        else:
            # 否则将 self_outputs 的前两个元素在第二维度上连接，并将 hidden_states 连接在一起
            concat_self_outputs = torch.cat(self_outputs[:2], dim=1)
            concat_hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)

        # 输出注意力计算的结果
        attention_output = self.output(concat_self_outputs, concat_hidden_states)

        # 截取 word_attention_output 和 entity_attention_output
        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]

        # 如果输出注意力信息，则将其添加到输出元组中
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate

class LukeIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 中间激活函数，根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 中间激活函数变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput

class LukeOutput(nn.Module):
    # 该类未完整提供，应该包括更多的代码才能完全注释
    pass
    # 初始化函数，用于创建一个新的神经网络模块实例
    def __init__(self, config):
        # 调用父类的初始化方法，确保正确地初始化神经网络模块
        super().__init__()
        # 创建一个全连接层，将输入特征的维度转换为隐藏层的维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个层归一化层，用于标准化隐藏层的输出
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，用于在训练过程中随机丢弃部分隐藏层的输出，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收隐藏状态和输入张量作为输入，并返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层进行线性变换，将隐藏状态张量转换到隐藏层的维度
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态张量进行dropout操作，以随机丢弃部分输出，防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 使用层归一化层对dropout后的隐藏状态张量进行归一化处理，加上输入张量，得到最终的隐藏状态张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量作为前向传播的输出
        return hidden_states
# 定义了一个名为 LukeLayer 的神经网络模块，继承自 nn.Module
class LukeLayer(nn.Module):
    # 初始化函数，接收一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 设置当前层的前向传播中使用的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化注意力模块、中间层模块和输出层模块
        self.attention = LukeAttention(config)
        self.intermediate = LukeIntermediate(config)
        self.output = LukeOutput(config)

    # 前向传播函数，接收多个参数：word_hidden_states, entity_hidden_states, attention_mask, head_mask 等
    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 获取输入 word_hidden_states 的第二维度大小
        word_size = word_hidden_states.size(1)

        # 调用注意力模块的前向传播，获取自注意力的输出
        self_attention_outputs = self.attention(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        
        # 根据是否有实体隐藏状态，选择不同的连接方式
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = torch.cat(self_attention_outputs[:2], dim=1)

        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = self_attention_outputs[2:]  # 如果需要输出注意力权重，则添加自注意力的输出

        # 将前向传播应用于 chunk 分块，得到当前层的输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concat_attention_output
        )
        
        # 提取出词的层的输出
        word_layer_output = layer_output[:, :word_size, :]
        
        # 根据是否有实体隐藏状态，提取出实体的层的输出
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        # 将词层和实体层的输出，以及可能的注意力权重输出，放入 outputs 中返回
        outputs = (word_layer_output, entity_layer_output) + outputs

        return outputs

    # 定义 feed_forward_chunk 方法，用于前向传播中的块处理
    def feed_forward_chunk(self, attention_output):
        # 中间层的输出
        intermediate_output = self.intermediate(attention_output)
        # 输出层的输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# 定义了一个名为 LukeEncoder 的神经网络模块，继承自 nn.Module
class LukeEncoder(nn.Module):
    # 初始化函数，接收一个配置参数 config
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建多个 LukeLayer 层，根据配置中的层数
        self.layer = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # LukeEncoder 的前向传播函数，接收多个参数：word_hidden_states, entity_hidden_states, attention_mask 等
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
        # 逐层调用 LukeLayer 的前向传播函数
        for layer_module in self.layer:
            # 对每一层进行前向传播
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states,
                entity_hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )

        # 返回最终的词和实体隐藏状态
        return word_hidden_states, entity_hidden_states
        ):
        # 初始化空元组以存储所有层的隐藏状态（如果需要输出）
        all_word_hidden_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历模型的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到相应的元组中
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            # 获取当前层的头部掩码（如果有的话）
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 根据梯度检查点和训练模式选择不同的前向传播方法
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 普通的前向传播
                layer_outputs = layer_module(
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            # 更新当前层的词级隐藏状态
            word_hidden_states = layer_outputs[0]

            # 如果实体级别的隐藏状态不为None，则更新
            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终层的隐藏状态添加到相应的元组中
        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

        # 如果不需要返回字典形式的输出，则返回包含非None值的元组
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
        # 返回一个BaseLukeModelOutput对象，包含不同层级的最终隐藏状态和注意力权重
        return BaseLukeModelOutput(
            last_hidden_state=word_hidden_states,
            hidden_states=all_word_hidden_states,
            attentions=all_self_attentions,
            entity_last_hidden_state=entity_hidden_states,
            entity_hidden_states=all_entity_hidden_states,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler
# 从 transformers 库中的 BertPooler 类复制而来

class LukePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 通过获取第一个 token 对应的隐藏状态来实现模型的“池化”
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EntityPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class EntityPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class LukePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # LukePreTrainedModel 类的抽象基类，处理权重初始化和预训练模型的下载加载接口

    config_class = LukeConfig
    base_model_prefix = "luke"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LukeAttention", "LukeEntityEmbeddings"]
    # 初始化神经网络模块的权重
    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 如果嵌入维度为1，通常用于偏置参数的嵌入，将权重初始化为零
            if module.embedding_dim == 1:
                module.weight.data.zero_()
            else:
                # 否则使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1（通常用于缩放归一化的权重）
            module.weight.data.fill_(1.0)
# LUKE_START_DOCSTRING 定义了该模型的文档字符串，提供了关于模型的继承、参数初始化和基本使用的描述信息。
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

# LUKE_INPUTS_DOCSTRING 没有给出具体的文档字符串内容，在实际代码中应当补充相关描述信息。
LUKE_INPUTS_DOCSTRING = r"""
"""


# 使用装饰器 @add_start_docstrings 将以下类的文档字符串与额外的描述信息合并，提供了对 LUKE 模型转换器的描述
# 和 LUKE_START_DOCSTRING 中定义的模型文档信息。
@add_start_docstrings(
    "The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities without any"
    " specific head on top.",
    LUKE_START_DOCSTRING,
)
class LukeModel(LukePreTrainedModel):
    # 初始化方法，接收配置和是否添加池化层的参数，构建 LUKE 模型的组件。
    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        # 初始化模型的嵌入层、实体嵌入层和编码器。
        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = LukeEncoder(config)

        # 如果指定添加池化层，则初始化池化器。
        self.pooler = LukePooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理。
        self.post_init()

    # 获取输入嵌入层（词嵌入层）。
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层（词嵌入层）的值。
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 获取实体嵌入层。
    def get_entity_embeddings(self):
        return self.entity_embeddings.entity_embeddings

    # 设置实体嵌入层的值。
    def set_entity_embeddings(self, value):
        self.entity_embeddings.entity_embeddings = value

    # 实现了头部修剪的方法，但在 LUKE 中未实现头部的修剪操作。
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    # 使用装饰器 @add_start_docstrings_to_model_forward 将以下方法的输入描述与 LUKE_INPUTS_DOCSTRING 结合，
    # 提供了模型前向传播方法的输入描述和相关配置信息。
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseLukeModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Executes the forward pass for the model.

        Arguments:
            input_ids (`torch.LongTensor`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor`, *optional*):
                Mask to avoid performing attention on padding tokens.
            token_type_ids (`torch.LongTensor`, *optional*):
                Segment token indices to indicate first and second portions of the inputs.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            entity_ids (`torch.LongTensor`, *optional*):
                Indices of entity sequence tokens in the vocabulary.
            entity_attention_mask (`torch.FloatTensor`, *optional*):
                Mask to avoid performing attention on padding tokens in entity tokens.
            entity_token_type_ids (`torch.LongTensor`, *optional*):
                Segment token indices to indicate first and second portions of the entity inputs.
            entity_position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each entity sequence tokens in the position embeddings.
            head_mask (`torch.FloatTensor`, *optional*):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Optionally instead of input_ids, you can pass pre-computed embeddings.
            output_attentions (`bool`, *optional*):
                Whether to output the attentions tensors.
            output_hidden_states (`bool`, *optional*):
                Whether to output the hidden states tensors.
            return_dict (`bool`, *optional*):
                Whether to return a dict instead of a tuple.

        Returns:
            `torch.Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.
        """
        # Implementation of the forward pass for the model goes here
        # ...
        pass


    def get_extended_attention_mask(
        self, word_attention_mask: torch.LongTensor, entity_attention_mask: Optional[torch.LongTensor]
    ):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            word_attention_mask (`torch.LongTensor`):
                Attention mask for word tokens with ones indicating tokens to attend to, zeros for tokens to ignore.
            entity_attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for entity tokens with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # Initialize the attention_mask with word_attention_mask
        attention_mask = word_attention_mask

        # Concatenate entity_attention_mask if provided
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=-1)

        # Ensure attention_mask is extended properly based on its dimensionality
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        # Ensure dtype compatibility for fp16
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)

        # Handle floating-point precision compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        return extended_attention_mask
# LUKE 模型用于带有语言建模头和顶部实体预测头的任务，支持掩码语言建模和掩码实体预测。
@add_start_docstrings(
    """
    LUKE 模型带有语言建模头和顶部实体预测头，支持掩码语言建模和掩码实体预测。
    """,
    LUKE_START_DOCSTRING,
)
class LukeForMaskedLM(LukePreTrainedModel):
    # 用于绑定权重的键列表，包括语言建模头和实体预测头的权重
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias", "entity_predictions.decoder.weight"]

    def __init__(self, config):
        # 调用父类初始化方法，传入配置参数
        super().__init__(config)

        # 初始化 LUKE 模型
        self.luke = LukeModel(config)

        # 初始化语言建模头
        self.lm_head = LukeLMHead(config)

        # 初始化实体预测头
        self.entity_predictions = EntityPredictionHead(config)

        # 交叉熵损失函数用于计算损失
        self.loss_fn = nn.CrossEntropyLoss()

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        # 调用父类方法来绑定权重
        super().tie_weights()

        # 绑定或克隆实体预测头的权重到 LUKE 模型的实体嵌入
        self._tie_or_clone_weights(self.entity_predictions.decoder, self.luke.entity_embeddings.entity_embeddings)

    def get_output_embeddings(self):
        # 返回语言建模头的解码器权重
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入到语言建模头的解码器中
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 重写父类的前向方法，详细描述输入参数的文档字符串
    def forward(self, **kwargs):
    # 使用装饰器替换返回文档字符串，指定输出类型为LukeMaskedLMOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=LukeMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，类型为可选的浮点数张量
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs，类型为可选的长整型张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，类型为可选的长整型张量
        entity_ids: Optional[torch.LongTensor] = None,  # 实体 IDs，类型为可选的长整型张量
        entity_attention_mask: Optional[torch.LongTensor] = None,  # 实体注意力遮罩，类型为可选的长整型张量
        entity_token_type_ids: Optional[torch.LongTensor] = None,  # 实体 token 类型 IDs，类型为可选的长整型张量
        entity_position_ids: Optional[torch.LongTensor] = None,  # 实体位置 IDs，类型为可选的长整型张量
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量
        entity_labels: Optional[torch.LongTensor] = None,  # 实体标签，类型为可选的长整型张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩，类型为可选的浮点数张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入，类型为可选的浮点数张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态信息，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典，类型为可选的布尔值
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

        self.luke = LukeModel(config)  # 初始化 LUKE 模型

        self.num_labels = config.num_labels  # 类别数量
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 分类器线性层

        # 初始化权重并进行最终处理
        self.post_init()

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

        self.luke = LukeModel(config)  # 初始化 LUKE 模型

        self.num_labels = config.num_labels  # 类别数量
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout 层
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels, False)  # 用于实体对分类的线性层

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntityPairClassificationOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，用于模型的前向推断过程
    def forward(
        self,
        # 输入的token IDs，类型为torch.LongTensor，可选参数
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，指定哪些元素是padding的，类型为torch.FloatTensor，可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        # token类型IDs，用于区分两个句子的token，类型为torch.LongTensor，可选参数
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置IDs，指定每个token在序列中的位置，类型为torch.LongTensor，可选参数
        position_ids: Optional[torch.LongTensor] = None,
        # 实体IDs，用于指示实体的token序列，类型为torch.LongTensor，可选参数
        entity_ids: Optional[torch.LongTensor] = None,
        # 实体注意力掩码，指定实体token的padding，类型为torch.FloatTensor，可选参数
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        # 实体token类型IDs，区分实体token的两个句子，类型为torch.LongTensor，可选参数
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        # 实体位置IDs，指定每个实体token在序列中的位置，类型为torch.LongTensor，可选参数
        entity_position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，指定要执行注意力操作的头部，类型为torch.FloatTensor，可选参数
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入的嵌入表示，类型为torch.FloatTensor，可选参数
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，用于计算损失的真实标签，类型为torch.LongTensor，可选参数
        labels: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重，类型为bool，可选参数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为bool，可选参数
        output_hidden_states: Optional[bool] = None,
        # 是否返回一个字典格式的输出，类型为bool，可选参数
        return_dict: Optional[bool] = None,
# 使用自定义的文档字符串初始化 LUKE 模型，添加了一个在隐藏状态输出之上的跨度分类头部，用于诸如命名实体识别等任务
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

        # 初始化 LUKE 模型
        self.luke = LukeModel(config)

        # 获取标签数量和隐藏层的 dropout 概率
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 线性层用于分类，输入维度是隐藏层大小的三倍
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 方法，注解详细描述了模型的输入参数及其用途
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



# 使用自定义的文档字符串初始化 LUKE 模型，添加了一个在汇总输出之上的序列分类/回归头部，例如 GLUE 任务
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

        # 获取标签数量
        self.num_labels = config.num_labels

        # 初始化 LUKE 模型
        self.luke = LukeModel(config)

        # 根据配置使用分类器的 dropout 或者隐藏层的 dropout 概率
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        # 线性层用于分类，输入维度是隐藏层大小
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 方法，注解详细描述了模型的输入参数及其用途
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法，接受多个可选的输入参数，用于模型的输入和控制
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token IDs，类型为长整型Tensor，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，类型为浮点数Tensor，可选
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型IDs，类型为长整型Tensor，可选
        position_ids: Optional[torch.LongTensor] = None,  # 位置IDs，类型为长整型Tensor，可选
        entity_ids: Optional[torch.LongTensor] = None,  # 实体IDs，类型为长整型Tensor，可选
        entity_attention_mask: Optional[torch.FloatTensor] = None,  # 实体的注意力掩码，类型为浮点数Tensor，可选
        entity_token_type_ids: Optional[torch.LongTensor] = None,  # 实体的token类型IDs，类型为长整型Tensor，可选
        entity_position_ids: Optional[torch.LongTensor] = None,  # 实体的位置IDs，类型为长整型Tensor，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，类型为浮点数Tensor，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，类型为浮点数Tensor，可选
        labels: Optional[torch.FloatTensor] = None,  # 标签，类型为浮点数Tensor，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选布尔值
"""
The LUKE Model with a token classification head on top (a linear layer on top of the hidden-states output). To
solve Named-Entity Recognition (NER) task using LUKE, `LukeForEntitySpanClassification` is more suitable than this
class.
"""
@add_start_docstrings(
    """
    The LUKE Model with a token classification head on top (a linear layer on top of the hidden-states output). To
    solve Named-Entity Recognition (NER) task using LUKE, `LukeForEntitySpanClassification` is more suitable than this
    class.
    """,
    LUKE_START_DOCSTRING,
)
class LukeForTokenClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the LUKE model backbone without adding pooling layer
        self.luke = LukeModel(config, add_pooling_layer=False)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        # Linear layer for token classification, output dimension is num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> Union[Tuple, LukeTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 return_dict 是否为 None 来确定是否使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 LUKE 模型进行前向传播
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

        # 获取模型最后一层的隐藏状态作为序列输出
        sequence_output = outputs.last_hidden_state

        # 对序列输出进行 dropout 操作
        sequence_output = self.dropout(sequence_output)
        
        # 将 dropout 后的输出传入分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化 loss 为 None
        loss = None
        if labels is not None:
            # 将 labels 移动到正确的设备上，以支持模型并行计算
            labels = labels.to(logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回一个元组，包含 loss, logits, outputs.hidden_states,
        # outputs.entity_hidden_states, outputs.attentions 中不为 None 的部分
        if not return_dict:
            return tuple(
                v
                for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions]
                if v is not None
            )

        # 如果 return_dict 为 True，则返回 LukeTokenClassifierOutput 对象，包含 loss, logits,
        # outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions
        return LukeTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    The LUKE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LUKE_START_DOCSTRING,
)
class LukeForQuestionAnswering(LukePreTrainedModel):
    """
    LUKE模型，用于支持抽取式问答任务（如SQuAD），在隐藏状态输出的顶部增加一个用于计算“起始位置logits”和“结束位置logits”的线性层。
    继承自LukePreTrainedModel。
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        # 初始化LUKE模型，不添加池化层
        self.luke = LukeModel(config, add_pooling_layer=False)
        
        # QA输出层，用于生成答案的线性层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        正向传播方法，接收多种输入并生成相应的输出。
        参数与返回值的详细说明参见LUKE_INPUTS_DOCSTRING。
        """
        # 实现正向传播逻辑的具体内容
        pass


@add_start_docstrings(
    """
    The LUKE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    LUKE_START_DOCSTRING,
)
class LukeForMultipleChoice(LukePreTrainedModel):
    """
    LUKE模型，用于多选分类任务（例如RocStories/SWAG），在汇总输出的顶部增加一个线性层和softmax激活函数。
    继承自LukePreTrainedModel。
    """

    def __init__(self, config):
        super().__init__(config)

        # 初始化LUKE模型
        self.luke = LukeModel(config)
        
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        # 分类器线性层，输出为1，用于多选分类任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LukeMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接受多个可选的输入参数：
    # - input_ids: 输入的token IDs序列，类型为torch.LongTensor，默认为None
    # - attention_mask: 注意力掩码，类型为torch.FloatTensor，默认为None
    # - token_type_ids: token类型IDs，类型为torch.LongTensor，默认为None
    # - position_ids: 位置IDs，类型为torch.LongTensor，默认为None
    # - entity_ids: 实体IDs，类型为torch.LongTensor，默认为None
    # - entity_attention_mask: 实体的注意力掩码，类型为torch.FloatTensor，默认为None
    # - entity_token_type_ids: 实体的token类型IDs，类型为torch.LongTensor，默认为None
    # - entity_position_ids: 实体的位置IDs，类型为torch.LongTensor，默认为None
    # - head_mask: 头部掩码，类型为torch.FloatTensor，默认为None
    # - inputs_embeds: 输入的嵌入向量，类型为torch.FloatTensor，默认为None
    # - labels: 标签，类型为torch.FloatTensor，默认为None
    # - output_attentions: 是否输出注意力权重，类型为bool，默认为None
    # - output_hidden_states: 是否输出隐藏状态，类型为bool，默认为None
    # - return_dict: 是否以字典形式返回结果，类型为bool，默认为None
```