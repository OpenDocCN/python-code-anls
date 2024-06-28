# `.\models\longformer\modeling_longformer.py`

```
# coding=utf-8
# 版权归 The Allen Institute for AI team 和 The HuggingFace Inc. team 所有。
#
# 根据 Apache License, Version 2.0 授权使用本文件；
# 除非遵守许可证，否则不得使用此文件。
# 可在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据"现状"分发软件，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
"""PyTorch Longformer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_longformer import LongformerConfig

# 获取记录器，用于记录日志
logger = logging.get_logger(__name__)

# 用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "allenai/longformer-base-4096"
# 用于文档的配置名称
_CONFIG_FOR_DOC = "LongformerConfig"

# 预训练模型的存档列表
LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # 所有 Longformer 模型详见 https://huggingface.co/models?filter=longformer
]


@dataclass
class LongformerBaseModelOutput(ModelOutput):
    """
    Longformer 输出的基类，包含潜在的隐藏状态、本地和全局注意力。
    """
    pass  # 此处暂时未定义具体内容，仅声明基类
    # 定义函数参数 `last_hidden_state`，表示模型最后一层的隐藏状态，是一个形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor`。
    last_hidden_state: torch.FloatTensor
    # 定义可选参数 `hidden_states`，当 `output_hidden_states=True` 时返回，表示模型每一层的隐藏状态的元组。
    # 每个元素是一个形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor`。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


```    
    # 定义可选参数 `attentions`，当 `output_attentions=True` 时返回，表示模型每一层的本地注意力权重的元组。
    # 每个元素是一个形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)` 的 `torch.FloatTensor`。
    # 这些是经过注意力 softmax 后的本地注意力权重，用于计算自注意力头中的加权平均值。
    # 前 `x` 个值是对全局注意力掩码中的令牌的注意力权重，剩余的 `attention_window + 1` 个值是对注意力窗口中的令牌的注意力权重。
    # 注意，前 `x` 个值指的是文本中固定位置的令牌的注意力权重，但剩余的 `attention_window + 1` 个值是相对位置的注意力权重。
    # 如果注意力窗口包含具有全局注意力的令牌，则相应索引处的注意力权重设置为 0，其值应从第一个 `x` 个注意力权重中访问。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    # 定义可选参数 `global_attentions`，当 `output_attentions=True` 时返回，表示模型每一层的全局注意力权重的元组。
    # 每个元素是一个形状为 `(batch_size, num_heads, sequence_length, x)` 的 `torch.FloatTensor`。
    # 这些是经过注意力 softmax 后的全局注意力权重，用于计算自注意力头中的加权平均值。
    # 这些是从每个具有全局注意力的令牌到序列中每个令牌的注意力权重。
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 声明一个可选类型的变量 attentions，可以存储 torch.FloatTensor 类型的元组或者为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 声明一个可选类型的变量 global_attentions，可以存储 torch.FloatTensor 类型的元组或者为 None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 @dataclass 装饰器声明一个数据类，用于表示带池化的 Longformer 模型的基本输出
@dataclass
class LongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Longformer 模型的输出基类，同时包含最后隐藏状态的池化结果。

    """

    # 最后的隐藏状态，类型为 torch.FloatTensor
    last_hidden_state: torch.FloatTensor
    # 可选项：池化层的输出，类型为 torch.FloatTensor，默认为 None
    pooler_output: torch.FloatTensor = None
    # 可选项：隐藏状态的元组，包含多个 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：注意力分布的元组，包含多个 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：全局注意力的元组，包含多个 torch.FloatTensor
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 @dataclass 装饰器声明一个数据类，用于表示 Longformer 掩码语言模型的输出
@dataclass
class LongformerMaskedLMOutput(ModelOutput):
    """
    Longformer 掩码语言模型输出的基类。

    """

    # 可选项：损失值，类型为 torch.FloatTensor，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # 可选项：预测的 logits，类型为 torch.FloatTensor
    logits: torch.FloatTensor = None
    # 可选项：隐藏状态的元组，包含多个 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：注意力分布的元组，包含多个 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：全局注意力的元组，包含多个 torch.FloatTensor
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 @dataclass 装饰器声明一个数据类，用于表示 Longformer 问答模型的输出
@dataclass
class LongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Longformer 问答模型输出的基类。

    """

    # 可选项：损失值，类型为 torch.FloatTensor，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # 可选项：起始位置的 logits，类型为 torch.FloatTensor
    start_logits: torch.FloatTensor = None
    # 可选项：结束位置的 logits，类型为 torch.FloatTensor
    end_logits: torch.FloatTensor = None
    # 可选项：隐藏状态的元组，包含多个 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：注意力分布的元组，包含多个 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：全局注意力的元组，包含多个 torch.FloatTensor
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 @dataclass 装饰器声明一个数据类，用于表示 Longformer 序列分类模型的输出
@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    """
    Longformer 序列分类模型输出的基类。

    """

    # 可选项：损失值，类型为 torch.FloatTensor，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # 可选项：预测的 logits，类型为 torch.FloatTensor
    logits: torch.FloatTensor = None
    # 可选项：隐藏状态的元组，包含多个 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：注意力分布的元组，包含多个 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：全局注意力的元组，包含多个 torch.FloatTensor
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        Args:
            loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                分类（或者当`config.num_labels==1`时为回归）的损失值。
            logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
                分类（或者当`config.num_labels==1`时为回归）的分数（SoftMax 之前）。
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                包含模型每一层输出的隐藏状态的元组，每个元素是一个 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。

                模型每一层的输出的隐藏状态以及初始嵌入输出。
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                包含局部注意力权重的元组，每个元素是一个 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)`，其中 `x` 是具有全局注意力掩码的令牌数量。

                局部注意力softmax后的权重，用于计算自注意力头中的加权平均值。这些是每个令牌到具有全局注意力的每个令牌（前 `x` 个值）和到注意力窗口中的每个令牌（剩余的 `attention_window + 1` 个值）的注意力权重。
                注意，前 `x` 个值指的是文本中具有固定位置的令牌，但剩余的 `attention_window + 1` 个值指的是具有相对位置的令牌：一个令牌到自身的注意力权重位于索引 `x + attention_window / 2` 处，前 `attention_window / 2`（后 `attention_window / 2`）个值是指前 `attention_window / 2`（后 `attention_window / 2`）个令牌的注意力权重。
                如果注意力窗口中包含一个具有全局注意力的令牌，则相应索引处的注意力权重设为0；这些值应从前 `x` 个注意力权重中获取。如果一个令牌具有全局注意力，则到`attentions`中的所有其他令牌的注意力权重为0，这些值应从`global_attentions`中获取。
            global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                包含全局注意力权重的元组，每个元素是一个 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, x)`，其中 `x` 是具有全局注意力掩码的令牌数量。

                全局注意力softmax后的权重，用于计算自注意力头中的加权平均值。这些是每个具有全局注意力的令牌到序列中的每个令牌的注意力权重。
    # 定义一个可选的浮点数张量 loss，初始值为 None
    loss: Optional[torch.FloatTensor] = None
    
    # 定义一个浮点数张量 logits，初始值为 None
    logits: torch.FloatTensor = None
    
    # 定义一个可选的元组，包含多个浮点数张量，表示隐藏状态，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义一个可选的元组，包含多个浮点数张量，表示注意力机制，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义一个可选的元组，包含多个浮点数张量，表示全局注意力机制，初始值为 None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
@dataclass
class LongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice Longformer models.
    """

    # 可选项：损失值，用于存储模型的损失值（浮点数张量）
    loss: Optional[torch.FloatTensor] = None
    # 输出：逻辑回归值，模型的逻辑回归输出（浮点数张量）
    logits: torch.FloatTensor = None
    # 可选项：隐藏状态，包含模型的隐藏状态的元组（浮点数张量的元组）
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：注意力分布，包含模型的注意力分布的元组（浮点数张量的元组）
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项：全局注意力分布，包含模型的全局注意力分布的元组（浮点数张量的元组）
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class LongformerTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    """
    # 定义 loss 变量，用于存储分类损失（如果提供标签的话）
    loss: Optional[torch.FloatTensor] = None
    # 定义一个变量 logits，类型为 torch 的 FloatTensor，初始值为 None
    logits: torch.FloatTensor = None
    # 定义一个变量 hidden_states，类型为一个可选的元组，元组内包含多个 torch 的 FloatTensor 对象，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个变量 attentions，类型为一个可选的元组，元组内包含多个 torch 的 FloatTensor 对象，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个变量 global_attentions，类型为一个可选的元组，元组内包含多个 torch 的 FloatTensor 对象，初始值为 None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 计算输入中第一个 `sep_token_id` 的索引位置
def _get_question_end_index(input_ids, sep_token_id):
    # 找到所有 `sep_token_id` 的索引位置
    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]

    # 断言确保 `input_ids` 是二维的
    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    # 断言确保每个样本中恰好有三个分隔符 `sep_token_id`
    assert sep_token_indices.shape[0] == 3 * batch_size, (
        f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You"
        " might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
    )
    # 返回每个样本中第一个 `sep_token_id` 的索引
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]


# 计算全局注意力掩码，根据 `before_sep_token` 决定在 `sep_token_id` 之前或之后放置注意力
def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    # 获取问题结束的索引位置
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
    
    # 创建布尔类型的注意力掩码，全局注意力位置为 True
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        # 将小于 `question_end_index` 的位置设置为 True
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.bool)
    else:
        # 如果不在 `before_sep_token` 模式下，将 `sep_token_id` 之后的位置设置为 True
        attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(torch.bool) * (
            attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(torch.bool)

    return attention_mask


# 根据输入的 `input_ids` 和 `padding_idx` 创建位置编号
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # 创建掩码，标识非填充符号的位置
    mask = input_ids.ne(padding_idx).int()
    # 计算递增的位置编号，从 `padding_idx+1` 开始
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class LongformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 初始化函数，接受配置对象 config
    def __init__(self, config):
        # 调用父类构造函数初始化
        super().__init__()
        # 创建词嵌入层，根据配置中的词汇大小（vocab_size）和隐藏大小（hidden_size），并设置填充 token 的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建标记类型嵌入层，根据配置中的类型词汇大小（type_vocab_size）和隐藏大小（hidden_size）
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建 LayerNorm 层，用于规范化隐藏状态向量，保持与 TensorFlow 模型变量名称的一致性，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，用于在训练过程中随机丢弃部分隐藏状态向量，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 设置填充 token 的索引，以便后续使用
        self.padding_idx = config.pad_token_id
        # 创建位置嵌入层，根据配置中的最大位置嵌入数（max_position_embeddings）和隐藏大小（hidden_size），并设置填充 token 的索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数，接受输入参数 input_ids、token_type_ids、position_ids 和 inputs_embeds
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果未提供 position_ids 参数
        if position_ids is None:
            # 如果提供了 input_ids 参数
            if input_ids is not None:
                # 根据 input_ids 创建 position_ids，保持任何填充的 token 仍然是填充状态
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                # 否则，从 inputs_embeds 创建 position_ids
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果提供了 input_ids 参数
        if input_ids is not None:
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状，去掉最后一维（即序列长度维度）
            input_shape = inputs_embeds.size()[:-1]

        # 如果未提供 token_type_ids 参数，则创建全零的 token_type_ids，形状与 input_shape 相同
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

        # 如果未提供 inputs_embeds 参数，则根据 input_ids 获取对应的词嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据 position_ids 获取对应的位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 根据 token_type_ids 获取对应的标记类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入向量、位置嵌入向量和标记类型嵌入向量相加得到最终的嵌入表示
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 对嵌入向量进行 LayerNorm 规范化
        embeddings = self.LayerNorm(embeddings)
        # 对规范化后的向量进行 Dropout 操作
        embeddings = self.dropout(embeddings)
        # 返回嵌入向量作为模型的输出
        return embeddings

    # 根据 inputs_embeds 参数创建位置 ids 的函数
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor inputs_embeds:

        Returns: torch.Tensor
        """
        # 获取 inputs_embeds 的形状，去掉最后一维（即序列长度维度）
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成从 padding_idx + 1 开始，到 padding_idx + 1 + sequence_length 的序列作为位置 ids
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置 ids 扩展为与 input_shape 相同的形状，并返回
        return position_ids.unsqueeze(0).expand(input_shape)
    # LongformerSelfAttention 类的定义，继承自 nn.Module
    class LongformerSelfAttention(nn.Module):
        def __init__(self, config, layer_id):
            super().__init__()
            # 检查隐藏大小是否是注意力头数的倍数
            if config.hidden_size % config.num_attention_heads != 0:
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({config.num_attention_heads})"
                )
            # 设置注意力头数和头部维度
            self.num_heads = config.num_attention_heads
            self.head_dim = int(config.hidden_size / config.num_attention_heads)
            self.embed_dim = config.hidden_size

            # 为查询、键和值分别创建线性层
            self.query = nn.Linear(config.hidden_size, self.embed_dim)
            self.key = nn.Linear(config.hidden_size, self.embed_dim)
            self.value = nn.Linear(config.hidden_size, self.embed_dim)

            # 为具有全局注意力的令牌单独创建投影层
            self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
            self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
            self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

            # 设置注意力概率的 dropout 率
            self.dropout = config.attention_probs_dropout_prob

            self.layer_id = layer_id
            attention_window = config.attention_window[self.layer_id]
            # 确保 attention_window 是偶数
            assert (
                attention_window % 2 == 0
            ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
            # 确保 attention_window 是正数
            assert (
                attention_window > 0
            ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

            # 设置单侧注意力窗口大小
            self.one_sided_attn_window_size = attention_window // 2

            self.config = config

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            is_index_masked=None,
            is_index_global_attn=None,
            is_global_attn=None,
            output_attentions=False,
        ):
            # _pad_and_transpose_last_two_dims 方法：填充并转置最后两个维度的静态方法
            def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
                """pads rows and then flips rows and columns"""
                hidden_states_padded = nn.functional.pad(
                    hidden_states_padded, padding
                )  # padding value is not important because it will be overwritten
                hidden_states_padded = hidden_states_padded.view(
                    *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
                )
                return hidden_states_padded

            @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

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

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        # 获取输入张量的维度信息
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        # 在最后一维度上进行填充，增加 window_overlap + 1 个位置的填充
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        # 重新调整张量形状，将多维度张量转换为二维张量
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        # 去除最后一个维度上的部分数据，保留前面的数据
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        # 将二维张量转换为四维张量，重塑成对角化矩阵的形式
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        # 去除最后一个维度上多余的数据，保留有效的部分
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        # 返回处理后的张量结果
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """将隐藏状态转换为重叠的块。块大小 = 2w，重叠大小 = w"""
        if not onnx_export:
            # 对于非导出到ONNX的情况，生成大小为2w的非重叠块
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            )
            # 使用 `as_strided` 方法使块之间重叠，重叠大小为 window_overlap
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

        # 导出到ONNX时，使用以下逻辑，因为 `as_strided`、`unfold` 和二维张量索引在ONNX导出中不受支持（尚未支持）
        # 当 `unfold` 支持后，使用以下方式进行替换：
        # > return hidden_states.unfold(dimension=1, size=window_overlap * 2, step=window_overlap).transpose(2, 3)
        # 如果 hidden_states.size(1) == window_overlap * 2，则可以简单地返回 hidden_states.unsqueeze(1)，但这需要控制流

        chunk_size = [
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]

        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        # 创建一个影响序列长度为 affected_seq_len 的二维开始位置掩码
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        # 创建一个翻转的结束位置掩码，与开始位置掩码相反
        ending_mask = beginning_mask.flip(dims=(1, 3))

        # 对输入张量的开始位置进行掩码处理
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)

        # 对输入张量的结束位置进行掩码处理
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)
    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        # 获取 value 张量的维度信息
        batch_size, seq_len, num_heads, head_dim = value.size()

        # 断言确保 seq_len 可以被 window_overlap*2 整除
        assert seq_len % (window_overlap * 2) == 0
        # 断言确保 attn_probs 的前三个维度与 value 的前三个维度相同
        assert attn_probs.size()[:3] == value.size()[:3]
        # 断言确保 attn_probs 的第四个维度等于 2*window_overlap + 1
        assert attn_probs.size(3) == 2 * window_overlap + 1
        
        # 计算 chunk 的数量，即将 seq_len 分成大小为 window_overlap 的 chunk 的数量
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
        
        # 将 attn_probs 转置后重塑成形状为 (batch_size*num_heads, chunks_count, window_overlap, 2*window_overlap+1) 的张量
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # 将 value 转置后重塑成形状为 (batch_size*num_heads, seq_len, head_dim) 的张量
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # 在序列的开头和结尾各填充 window_overlap 个值为 -1 的元素
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # 将 padded_value 切分成大小为 3*window_overlap 的 chunk，重叠部分为 window_overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        # 对 chunked_attn_probs 执行 _pad_and_diagonalize 操作
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        # 使用 Einstein Summation (einsum) 进行张量乘法操作，得到 context 张量
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        
        # 调整 context 张量的形状，并将第二和第三维度交换位置
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    def _get_global_attn_indices(is_index_global_attn):
        """计算全局注意力索引，在前向传递中需要的索引"""
        # 计算每个样本中全局注意力索引的数量
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = num_global_attn_indices.max()

        # 全局注意力索引的位置
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # 辅助变量，表示是否是全局注意力的本地索引
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # 全局注意力索引中非零值的位置
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # 全局注意力索引中零值（即填充值）的位置
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # 仅创建全局键向量
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        # 将全局注意力索引对应的键向量填充到新创建的张量中
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        # 使用 Einstein Summation 计算全局键向量对应的注意力概率
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        # 由于 ONNX 导出仅支持连续索引，需要进行转置操作
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        # 将填充位置的注意力概率置为一个很小的数，以便在处理中被忽略
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min

        # 再次进行转置，以便输出与原始格式匹配
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        # 省略函数体，不在注释范围内
        ):
        # 获取批量大小
        batch_size = attn_probs.shape[0]

        # 仅保留全局注意力的局部注意力概率
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        
        # 仅获取全局注意力对应的数值向量
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # 使用 `matmul` 替代 `einsum`，因为在 fp16 下 `einsum` 有时会崩溃
        # 计算仅全局注意力的输出
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # 重塑非全局注意力的注意力概率
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # 使用滑动窗口方法计算包含全局和非全局注意力的注意力输出
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        
        # 返回全局注意力输出与非全局注意力输出的总和
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput

class LongformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，对输入进行归一化，eps 参数设置为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，按照 config.hidden_dropout_prob 概率随机丢弃输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层 dense 对输入 hidden_states 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行 Dropout
        hidden_states = self.dropout(hidden_states)
        # 对 Dropout 后的结果和输入 input_tensor 进行 LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回 LayerNorm 结果作为输出
        return hidden_states


class LongformerAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 创建 LongformerSelfAttention 对象，传入 config 和 layer_id
        self.self = LongformerSelfAttention(config, layer_id)
        # 创建 LongformerSelfOutput 对象，传入 config
        self.output = LongformerSelfOutput(config)
        # 初始化一个空集合，用于存储需要剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 根据给定的 heads 列表，寻找可剪枝的注意力头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对 self.self 中的 query、key、value 线性层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 对 self.output 中的 dense 线性层进行剪枝，dim=1 表示在第一个维度上进行剪枝
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        # 调用 self.self 的 forward 方法，传入相应参数，获取自注意力机制的输出
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        # 将 self_outputs[0] 和 hidden_states 作为输入，调用 self.output 进行后续处理
        attn_output = self.output(self_outputs[0], hidden_states)
        # 返回 attn_output 和 self_outputs 的其余部分作为输出
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate

class LongformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度是 config.hidden_size，输出维度是 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则选择对应的激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接收隐藏状态张量并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 应用激活函数到全连接层输出的隐藏状态张量
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertOutput
# 定义了 LongformerOutput 类，继承自 nn.Module
class LongformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征大小调整为隐藏状态大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建 LayerNorm 层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，用于随机丢弃隐藏状态中的一些元素，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受两个张量作为输入并返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行变换
        hidden_states = self.dense(hidden_states)
        # 在输出上应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 对变换后的隐藏状态应用 LayerNorm，并加上输入张量，形成残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states


# 定义了 LongformerLayer 类，继承自 nn.Module
class LongformerLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 创建 LongformerAttention 对象
        self.attention = LongformerAttention(config, layer_id)
        # 创建 LongformerIntermediate 对象
        self.intermediate = LongformerIntermediate(config)
        # 创建 LongformerOutput 对象
        self.output = LongformerOutput(config)
        # 设置前馈过程的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度
        self.seq_len_dim = 1

    # 前向传播函数，接受多个输入参数，并返回多个输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        # 使用注意力层进行处理，并获取注意力输出
        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        # 获取注意力输出的第一个元素作为注意力输出
        attn_output = self_attn_outputs[0]
        # 对注意力输出应用分块策略来进行前向传播
        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        # 将层输出和注意力输出的其余部分组合为输出元组
        outputs = (layer_output,) + self_attn_outputs[1:]
        # 返回最终的输出元组
        return outputs

    # 前馈分块函数，接受注意力输出并返回层输出
    def ff_chunk(self, attn_output):
        # 使用中间层进行处理
        intermediate_output = self.intermediate(attn_output)
        # 使用输出层进行处理，并返回层输出
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output


# 定义了 LongformerEncoder 类，继承自 nn.Module
class LongformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建 nn.ModuleList 来包含多个 LongformerLayer 对象，每个对象代表一层
        self.layer = nn.ModuleList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数，接受多个输入参数并返回多个输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 遍历每一层 LongformerLayer 并调用其 forward 方法进行处理
        for layer_module in self.layer:
            # 将当前层的输出作为下一层的输入
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask,
                output_attentions=output_attentions,
            )[0]  # 只保留每层的第一个输出
        # 返回最终的隐藏状态张量
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler
# 定义了 LongformerPooler 类，继承自 nn.Module
class LongformerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将隐藏状态大小映射回隐藏状态大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建激活函数层，使用 tanh 激活函数
        self.activation = nn.Tanh()
    # 定义一个方法 `forward`，接受一个名为 `hidden_states` 的张量作为输入，并返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个标记对应的隐藏状态来实现模型的"汇聚"操作
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态传递给全连接层 `self.dense`
        pooled_output = self.dense(first_token_tensor)
        # 将全连接层的输出应用激活函数 `self.activation`
        pooled_output = self.activation(pooled_output)
        # 返回经过激活函数处理后的汇聚输出张量
        return pooled_output
# 从transformers.models.roberta.modeling_roberta.RobertaLMHead中复制并修改为LongformerLMHead
class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization，输入维度为config.hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 解码器线性层，将config.hidden_size映射到config.vocab_size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 偏置项，用于解码器线性层的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将解码器的偏置设置为自定义的偏置项
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 全连接层的前向传播
        x = self.dense(features)
        # GELU激活函数
        x = gelu(x)
        # Layer normalization
        x = self.layer_norm(x)

        # 使用解码器将特征映射到词汇表大小的向量空间
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果解码器的偏置设备类型为"meta"，则将解码器的偏置与自定义的偏置项绑定
        # 否则，将自定义的偏置项与解码器的偏置绑定
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class LongformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为LongformerConfig
    config_class = LongformerConfig
    # 基础模型前缀为"longformer"
    base_model_prefix = "longformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不进行模块拆分的模块列表
    _no_split_modules = ["LongformerSelfAttention"]

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 略微不同于TF版本，使用正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，则将填充索引对应的权重置为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将LayerNorm层的偏置项置零，权重置为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 长格式开始文档字符串，描述了LongformerPreTrainedModel类的基本信息和用法
LONGFORMER_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`LongformerConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
LongformerModel 类的定义，继承自 LongformerPreTrainedModel 类，实现了具有长序列处理能力的自注意力机制。
"""

@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class LongformerModel(LongformerPreTrainedModel):
    """
    这个类是从 `RobertaModel` 复制的代码，并用长形自注意力机制覆盖了标准的自注意力机制，
    以提供处理长序列的能力，遵循 [Longformer: the Long-Document Transformer](https://arxiv.org/abs/2004.05150) 
    论文中描述的自注意力方法，由 Iz Beltagy、Matthew E. Peters 和 Arman Cohan 提出。
    
    Longformer 的自注意力结合了局部（滑动窗口）和全局注意力，可以在不增加 O(n^2) 内存和计算量的情况下扩展到长文档。

    这里实现的 `LongformerSelfAttention` 自注意力模块支持局部和全局注意力的结合，但不支持自回归注意力和扩展注意力。
    自回归和扩展注意力对于自回归语言建模比下游任务的微调更为重要。未来的版本将添加对自回归注意力的支持，
    但是对扩展注意力的支持需要一个定制的 CUDA 内核，以确保内存和计算效率。

    """

    def __init__(self, config, add_pooling_layer=True):
        """
        初始化函数，接受一个配置对象 `config` 和一个布尔值参数 `add_pooling_layer`。
        """
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # 为每一层设置一个值
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)  # 初始化 LongformerEmbeddings
        self.encoder = LongformerEncoder(config)  # 初始化 LongformerEncoder
        self.pooler = LongformerPooler(config) if add_pooling_layer else None  # 初始化 LongformerPooler，如果 add_pooling_layer 为 True 则初始化，否则为 None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        """
        返回输入嵌入层 `word_embeddings`。
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        设置输入嵌入层 `word_embeddings` 的值为 `value`。
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型的注意力头。heads_to_prune: dict，键为层号，值为要在该层中剪枝的注意力头列表。参见 PreTrainedModel 基类。
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # 获取注意力窗口大小，若为整数则直接使用，否则取最大值
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        # 断言确保 attention_window 是偶数
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        
        # 获取输入数据的形状信息
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        # 计算需要填充的长度，使得序列长度是 attention_window 的整数倍
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # 在 ONNX 导出时需要记录这个分支，即使 padding_len == 0 也是可以的
        if padding_len > 0:
            # 发出警告，说明输入的长度被自动填充到多个 attention_window 的倍数
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            # 如果存在 input_ids，则使用 nn.functional.pad 进行填充
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            # 如果存在 position_ids，则使用 nn.functional.pad 进行填充，填充值为 pad_token_id
            if position_ids is not None:
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            # 如果存在 inputs_embeds，则创建一个新的 padding 数据，填充值为 pad_token_id，并拼接在原 inputs_embeds 后面
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            # 使用 nn.functional.pad 在 attention_mask 上进行填充，填充值为 0，表示填充部分不考虑注意力
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens
            # 使用 nn.functional.pad 在 token_type_ids 上进行填充，填充值为 0
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        # 返回填充后的信息：padding_len 填充长度，以及可能被填充的 input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        
        # 如果传入的 attention_mask 不为空
        if attention_mask is not None:
            # 将 attention_mask 乘以 (global_attention_mask + 1)，生成最终的合并后的 attention_mask
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # 如果没有传入 attention_mask，则直接使用 global_attention_mask + 1 作为 attention_mask
            attention_mask = global_attention_mask + 1
        
        # 返回合并后的 attention_mask
        return attention_mask

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器为类添加文档字符串，描述其作为 Longformer 模型和语言建模头部的特性
@add_start_docstrings("""Longformer Model with a `language modeling` head on top.""", LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):
    # 定义用于共享权重的关键字列表
    _tied_weights_keys = ["lm_head.decoder"]

    # 初始化方法，接受一个配置对象，并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 创建 Longformer 模型，不包含池化层
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 创建 LongformerLMHead 对象作为语言建模头部
        self.lm_head = LongformerLMHead(config)

        # 调用初始化权重和应用最终处理的方法
        self.post_init()

    # 返回语言建模头部的解码器
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置语言建模头部的解码器
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 使用装饰器为前向方法添加文档字符串，描述其接受的输入参数和输出类型
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Mask filling example:

        ```python
        >>> from transformers import AutoTokenizer, LongformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        ```

        Let's try a very long input.

        ```python
        >>> TXT = (
        ...     "My friends are <mask> but they eat too many carbs."
        ...     + " That's why I decide not to eat with them." * 300
        ... )
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['healthy', 'skinny', 'thin', 'good', 'vegetarian']
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用Longformer模型进行预测
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 使用语言建模头部生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()

            # 将labels移动到与预测分数相同的设备上
            labels = labels.to(prediction_scores.device)
            # 计算masked语言建模的损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不使用return_dict，则返回额外的输出
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 使用LongformerMaskedLMOutput类来返回结果
        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
@add_start_docstrings(
    """
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForSequenceClassification(LongformerPreTrainedModel):
    """
    Longformer模型，顶部带有序列分类/回归头部（即在汇总输出之上的线性层），例如用于GLUE任务。
    继承自LongformerPreTrainedModel。
    """

    def __init__(self, config):
        """
        初始化方法，接收一个配置参数config。

        Args:
            config (LongformerConfig): 模型的配置对象。

        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # 创建Longformer模型，不包含汇总层
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 创建Longformer分类头部
        self.classifier = LongformerClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="jpwahle/longformer-base-plagiarism-detection",
        output_type=LongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'ORIGINAL'",
        expected_loss=5.44,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，接收多个输入和控制参数，并返回模型输出。

        Args:
            input_ids (torch.Tensor, optional): 输入的token IDs张量。Default: None
            attention_mask (torch.Tensor, optional): 注意力掩码张量。Default: None
            global_attention_mask (torch.Tensor, optional): 全局注意力掩码张量。Default: None
            head_mask (torch.Tensor, optional): 头部掩码张量。Default: None
            token_type_ids (torch.Tensor, optional): token类型IDs张量。Default: None
            position_ids (torch.Tensor, optional): 位置IDs张量。Default: None
            inputs_embeds (torch.Tensor, optional): 嵌入输入张量。Default: None
            labels (torch.Tensor, optional): 标签张量。Default: None
            output_attentions (bool, optional): 是否返回注意力权重。Default: None
            output_hidden_states (bool, optional): 是否返回隐藏状态。Default: None
            return_dict (bool, optional): 是否以字典形式返回输出。Default: None

        Returns:
            Various depending on the configuration (torch.Tensor or dict of torch.Tensor):
            根据配置返回不同类型的输出（torch.Tensor或torch.Tensor字典）。

        """
        ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，如果未提供则根据配置决定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果全局注意力掩码未提供，则发出警告并初始化全局注意力掩码，将第一个token设置为全局关注
        if global_attention_mask is None:
            logger.warning_once("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # 在CLS token上开启全局关注

        # 使用Longformer模型进行前向传播
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # 获取Longformer模型的序列输出
        logits = self.classifier(sequence_output)  # 使用分类器对序列输出进行分类得到logits

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)  # 将标签移到与logits相同的设备上

            # 确定问题类型（回归、单标签分类、多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择合适的损失函数进行计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典，则返回一个元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 组装输出元组
            return ((loss,) + output) if loss is not None else output

        # 返回Longformer模型的输出，作为LongformerSequenceClassifierOutput对象
        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义一个全连接层，输入输出维度为config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 定义一个dropout层，概率为config.hidden_dropout_prob
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # 定义一个全连接层，输入为config.hidden_size，输出为config.num_labels

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # 取hidden_states的第一个token（相当于[CLS]）
        hidden_states = self.dropout(hidden_states)  # 对hidden_states进行dropout处理
        hidden_states = self.dense(hidden_states)  # 将hidden_states输入全连接层进行线性变换
        hidden_states = torch.tanh(hidden_states)  # 对变换后的hidden_states应用tanh激活函数
        hidden_states = self.dropout(hidden_states)  # 再次对hidden_states进行dropout处理
        output = self.out_proj(hidden_states)  # 将处理后的hidden_states输入输出层进行线性变换得到最终输出
        return output


@add_start_docstrings(
    """
    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)  # 使用LongformerModel初始化一个Longformer层，不加池化层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 定义一个全连接层，输入维度为config.hidden_size，输出维度为config.num_labels

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理步骤

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForTokenClassification(LongformerPreTrainedModel):
    # 该类为基于Longformer的标记分类模型，用于例如命名实体识别（NER）任务
    # 初始化方法，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 将配置对象中的标签数量赋给实例变量 num_labels
        self.num_labels = config.num_labels

        # 使用配置对象初始化 Longformer 模型，不添加池化层
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 使用配置对象中的隐藏层 dropout 概率初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 使用配置对象中的隐藏大小和标签数量初始化线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用自定义的后初始化方法，用于初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法，根据输入计算输出结果
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="brad1141/Longformer-finetuned-norm",
        output_type=LongformerTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=(
            "['Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence',"
            " 'Evidence', 'Evidence', 'Evidence', 'Evidence']"
        ),
        expected_loss=0.63,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, LongformerTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否返回字典格式的输出，如果未指定则根据配置决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给Longformer模型进行处理
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 对序列输出应用dropout操作
        sequence_output = self.dropout(sequence_output)
        
        # 将dropout后的序列输出传递给分类器得到logits
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()

            # 将标签移到与logits相同的设备上
            labels = labels.to(logits.device)
            
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典格式的输出
        if not return_dict:
            # 构造输出元组，包括logits和可能的额外输出状态
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回LongformerTokenClassifierOutput对象，其中包括损失、logits、隐藏状态和注意力权重
        return LongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
"""
Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.
"""
# 继承自 LongformerPreTrainedModel 的 Longformer 多选分类模型
class LongformerForMultipleChoice(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Longformer 模型
        self.longformer = LongformerModel(config)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器，线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LongformerMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
```