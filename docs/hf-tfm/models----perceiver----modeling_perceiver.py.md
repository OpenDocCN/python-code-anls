# `.\transformers\models\perceiver\modeling_perceiver.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 Deepmind 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证，否则不得使用该文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 在适用法律或书面同意的情况下，根据许可证分发的软件是基于“原样”分发的，没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证了解具体语言控制权限和限制事项
"""PyTorch Perceiver 模型。"""

# 导入必要的库
import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射表
from ...activations import ACT2FN
# 导入模型输出类
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
# 导入工具函数和类
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 Perceiver 配置类
from .configuration_perceiver import PerceiverConfig

# 定义类型别名
ModalitySizeType = Mapping[str, int]
PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
PostprocessorType = Callable[..., Any]

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于生成文档的检查点和配置
_CHECKPOINT_FOR_DOC = "deepmind/language-perceiver"
_CONFIG_FOR_DOC = "PerceiverConfig"

# 预训练 Perceiver 模型存档列表
PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "deepmind/language-perceiver",
    # 您可以在 https://huggingface.co/models?filter=perceiver 查看所有 Perceiver 模型
]

# 定义 Perceiver 模型输出类
@dataclass
class PerceiverModelOutput(ModelOutput):
    """
    Base class for Perceiver base model's outputs, with potential hidden states, attentions and cross-attentions.
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            分类（如果 config.num_labels==1 则为回归）得分（SoftMax 之前）。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组形式的 `torch.FloatTensor`（包含嵌入输出和每层输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            每层模型的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组形式的 `torch.FloatTensor`（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在自注意力头中使用的注意力 softmax 后的注意力权重，用于计算加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组形式的 `torch.FloatTensor`（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在解码器的跨注意力层中使用的注意力 softmax 后的注意力权重，用于计算加权平均值。
    """

    # 初始化 logits 变量，类型为 torch.FloatTensor，形状为 `(batch_size, num_labels)`，默认为 None
    logits: torch.FloatTensor = None
    # 初始化 last_hidden_state 变量，类型为 torch.FloatTensor，形状为 `(batch_size, sequence_length, hidden_size)`，默认为 None
    last_hidden_state: torch.FloatTensor = None
    # 初始化 hidden_states 变量，类型为 `Optional[Tuple[torch.FloatTensor]]`，默认为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化 attentions 变量，类型为 `Optional[Tuple[torch.FloatTensor]]`，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化 cross_attentions 变量，类型为 `Optional[Tuple[torch.FloatTensor]]`，默认为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于Perceiver解码器的输出，包括潜在的交叉注意力
@dataclass
class PerceiverDecoderOutput(ModelOutput):
    """
    Base class for Perceiver decoder outputs, with potential cross-attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Output of the basic decoder.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    logits: torch.FloatTensor = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义一个数据类，用于Perceiver的掩码语言模型输出
@dataclass
class PerceiverMaskedLMOutput(ModelOutput):
    """
    Base class for Perceiver's masked language model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_latents,
            num_latents)`. Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义一个数据类，用于Perceiver分类器的输出
@dataclass
class PerceiverClassifierOutput(ModelOutput):
    """
    Perceiver模型的输出的基类，包括序列/图像分类模型、光流和多模态自编码。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果config.num_labels==1，则为回归）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（如果config.num_labels==1，则为回归）分数（SoftMax之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组`torch.FloatTensor`（一个用于嵌入输出 + 一个用于每个层的输出）的形状为`(batch_size, sequence_length, hidden_size)`。模型在每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            元组`torch.FloatTensor`（每一层一个）的形状为`(batch_size, num_heads, sequence_length, sequence_length)`。注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            元组`torch.FloatTensor`（每一层一个）的形状为`(batch_size, num_heads, sequence_length, sequence_length)`。解码器的交叉注意力层的注意力权重，注意力softmax之后，用于计算交叉注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None  # 分类（或回归）损失，可选
    logits: torch.FloatTensor = None  # 分类（或回归）分数（SoftMax之前）
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 每一层的模型隐藏状态，包括嵌入输出
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，用于计算��注意力头中的加权平均值
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 交叉注意力权重，用于计算交叉注意力头中的加权平均值
class PerceiverEmbeddings(nn.Module):
    """Construct the latent embeddings."""

    def __init__(self, config):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.d_latents))

    def forward(self, batch_size: int):
        return self.latents.expand(batch_size, -1, -1)  # Thanks, Phil Wang


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        # Layer normalization for query and key tensors
        self.layernorm1 = nn.LayerNorm(q_dim)
        # Layer normalization for value tensor if it's cross attention, otherwise an identity layer
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        # Linear transformation for queries
        self.query = nn.Linear(q_dim, qk_channels)
        # Linear transformation for keys
        self.key = nn.Linear(kv_dim, qk_channels)
        # Linear transformation for values
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        # Reshape the tensor for multi-head attention computation
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        # Permute dimensions to facilitate multi-head attention computation
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        # 使用 Layernorm1 层对 hidden_states 进行标准化处理
        hidden_states = self.layernorm1(hidden_states)
        # 使用 Layernorm2 层对 inputs 进行标准化处理
        inputs = self.layernorm2(inputs)

        # 判断是否是跨注意力模块
        is_cross_attention = inputs is not None
        # 将 hidden_states 传入 query 层得到 queries
        queries = self.query(hidden_states)

        if is_cross_attention:
            # 如果是跨注意力模块，将 inputs 分别传入 key 层和 value 层，得到 keys 和 values
            keys = self.key(inputs)
            values = self.value(inputs)
            # 将输入的注意力掩码赋值给 attention_mask
            attention_mask = inputs_mask
        else:
            # 如果不是跨注意力模块，将 hidden_states 分别传入 key 层和 value 层，得到 keys 和 values
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # 重塑通道以进行多头注意力
        # 将输入数据从 (batch_size, time, channels) 重塑为 (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # 计算查询与键之间的点积以获得原始注意力得分
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        # 获取 queries 的形状信息
        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        # 获取 values 的形状信息
        _, _, _, v_head_dim = values.shape
        # 计算隐藏层数
        hiddens = self.num_heads * v_head_dim

        # 对 attention_scores 进行缩放处理
        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if attention_mask is not None:
            # 应用预先计算的注意力掩码
            attention_scores = attention_scores + attention_mask

        # 对 attention_scores 进行 softmax 处理，得到注意力概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 使用 dropout 层对 attention_probs 进行处理
        attention_probs = self.dropout(attention_probs)

        # 如果存在 head_mask，则对 attention_probs 进行头部掩码处理
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，即注意力概率与 values 的矩阵乘积
        context_layer = torch.matmul(attention_probs, values)

        # 将 context_layer 进行维度变换
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据需要返回 outputs，包括 context_layer 和 attention_probs
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class PerceiverSelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels):
        # 初始化方法，设置 PerceiverSelfOutput 类的属性
        super().__init__()
        # 创建一个线性层
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，通过线性层处理输入数据
        hidden_states = self.dense(hidden_states)
        return hidden_states


class PerceiverAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        # 初始化方法，设置 PerceiverAttention 类的属性
        super().__init__()
        # 多头注意力机制的实现
        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {config.cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        # 创建 PerceiverSelfAttention 实例
        self.self = PerceiverSelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        # 创建输出层
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels
        self.output = PerceiverSelfOutput(config, input_channels=self.self.v_channels, output_channels=output_channels)
        self.use_query_residual = use_query_residual
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 剪枝头部的方法
        if len(heads) == 0:
            return
        # 找到可剪枝的头的下标
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    # 前向传播函数，接受隐藏状态、注意力掩码、头部掩码、输入、输入掩码、输出注意力等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 self-attention 层处理隐藏状态等参数，返回 self-attention 层的输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # 输出投影层，将 self-attention 层的输出进行投影处理
        attention_output = self.output(self_outputs[0])

        # 可选择将原始查询加入到注意力输出中，仅当查询与输出的语义相同时可使用
        # 例如查询是位置信息，输出是像素信息时，可以考虑不使用原始查询
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        # 将注意力输出与可能的注意力结果一起组成输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了注意力结果，则添加到输出中
        return outputs
class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, config, input_size, widening_factor):
        super().__init__()
        # 创建一个线性层，输入维度为 input_size，输出维度为 widening_factor * input_size
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        # 判断 hidden_act 是否为字符串类型，如果是则使用指定的激活函数，否则使用 config 中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个线性层，输入维度为 widening_factor * input_size，输出维度为 input_size
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入数据进行线性变换
        hidden_states = self.dense1(hidden_states)
        # 使用激活函数处理 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对处理后的数据再次进行线性变换
        hidden_states = self.dense2(hidden_states)
        # 返回处理后的数据
        return hidden_states


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        # 设置前馈模块的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # 创建一个 PerceiverAttention 实例
        self.attention = PerceiverAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        # 对输入进行层归一化
        self.layernorm = nn.LayerNorm(q_dim)
        # 创建一个 PerceiverMLP 实例
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 通���注意力模块获取注意力输出
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # 如果输出注意力权重，则添加到输出中

        # 将前馈操作分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # 加上残差连接
        layer_output = layer_output + attention_output 

        # 构建输出
        outputs = (layer_output,) + outputs

        return outputs

    # 前馈操作的分块处理
    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""
    def __init__(self, config, kv_dim=None):
        super().__init__()
        self.config = config

        # 检查我们是否可以使用这些形状的多头注意力。
        # 检查潜在变量的维度是否能够被自注意力头数整除。
        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        # 检查潜在变量的维度是否能够被交叉注意力头数整除。
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        # 构建交叉注意力层。
        self.cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            num_heads=config.num_cross_attention_heads,
            q_dim=config.d_latents,
            kv_dim=kv_dim,
            widening_factor=config.cross_attention_widening_factor,
            use_query_residual=config.use_query_residual,
        )

        # 构建自注意力层的单个块。
        # 通过多次应用此块，我们得到更深的体系结构。
        self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = PerceiverLayer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            self_attention_layers.append(layer)

        # 将自注意力层放入模块列表中。
        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        # 如果不输出隐藏状态，则初始化为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化为空元组
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # 对 latents（hidden_states）和输入之间应用交叉注意力：
        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_cross_attentions 中
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # 多次应用自注意力层块：
        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不返回字典，则返回非空元素的元组
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        # 返回带有交叉注意力的基础模型输出
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class PerceiverPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为PerceiverConfig
    config_class = PerceiverConfig
    # 设置基础模型前缀为"perceiver"
    base_model_prefix = "perceiver"
    # 设置主输入名称为"inputs"
    main_input_name = "inputs"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块有"latents"属性
        elif hasattr(module, "latents"):
            # 使用正态分布初始化latents属性
            module.latents.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果模块有"position_embeddings"属性且是PerceiverTrainablePositionEncoding的实例
        elif hasattr(module, "position_embeddings") and isinstance(module, PerceiverTrainablePositionEncoding):
            # 使用正态分布初始化position_embeddings属性
            module.position_embeddings.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果模块是参数字典
        elif isinstance(module, nn.ParameterDict):
            # 遍历参数字典的键，使用正态分布初始化值
            for modality in module.keys():
                module[modality].data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将对应位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为0，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


PERCEIVER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PERCEIVER_MODEL_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    # 参数说明：
    # config ([`PerceiverConfig`]): 模型配置类，包含模型的所有参数。
    # 初始化时使用配置文件不会加载与模型相关的权重，只加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # decoder (*DecoderType*, *optional*): 可选的解码器，用于解码编码器的潜在表示。示例包括
    # *transformers.models.perceiver.modeling_perceiver.PerceiverBasicDecoder*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder*。
    # input_preprocessor (*PreprocessorType*, *optional*): 可选的输入预处理器。示例包括
    # *transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverTextPreprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor*。
    # output_postprocessor (*PostprocessorType*, *optional*): 可选的输出后处理器。示例包括
    # *transformers.models.perceiver.modeling_perceiver.PerceiverImagePostprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor*,
    # *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor*。
    # 
    # 请注意，您可以定义自己的解码器、预处理器和/或后处理器以适应您的用例。
"""

PERCEIVER_INPUTS_DOCSTRING = r"""
    Args:
        inputs (`torch.FloatTensor`):
            Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
        attention_mask (`torch.FloatTensor` of shape `{0}`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The Perceiver: a scalable, fully attentional architecture.""",
    PERCEIVER_MODEL_START_DOCSTRING,
)
class PerceiverModel(PerceiverPreTrainedModel):
    def __init__(
        self,
        config,
        decoder=None,
        input_preprocessor: PreprocessorType = None,
        output_postprocessor: PostprocessorType = None,
    ):
        # 初始化PerceiverModel类
        super().__init__(config)
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(
            config, kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else config.d_model
        )
        self.decoder = decoder

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入
        return self.embeddings.latents

    def set_input_embeddings(self, value):
        # 设置输入嵌入
        self.embeddings.latents = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=PerceiverModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播函数，接受输入张量、注意力掩码、下采样输出点、头部掩码、输出注意力、输出隐藏状态、返回字典等参数
    def forward(
        self,
        inputs: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        subsampled_output_points: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 为遮蔽语言建模示例使用 Perceiver 添加文档字符串
# 继承自 PerceiverPreTrainedModel 类
class PerceiverForMaskedLM(PerceiverPreTrainedModel):
    # 初始化函数，接受 PerceiverConfig 类型的参数
    def __init__(self, config: PerceiverConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建 PerceiverTextPreprocessor 对象
        text_preprocessor = PerceiverTextPreprocessor(config)

        # 定义可训练的位置编码参数
        trainable_position_encoding_kwargs_decoder = {
            "num_channels": text_preprocessor.num_channels,
            "index_dims": config.max_position_embeddings,
        }

        # 创建 PerceiverModel 对象
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=text_preprocessor,
            decoder=PerceiverBasicDecoder(
                config,
                output_num_channels=config.d_latents,
                output_index_dims=config.max_position_embeddings,  # 需要预先定义输入的序列长度
                num_channels=text_preprocessor.num_channels,
                qk_channels=8 * 32,
                v_channels=text_preprocessor.num_channels,
                num_heads=8,
                use_query_residual=False,
                final_project=False,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
            ),
        )
        # 创建 PerceiverEmbeddingDecoder 对象
        self.embedding_decoder = PerceiverEmbeddingDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传播添加文档字符串
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        input_ids: Optional[torch.Tensor] = None,



# 为文本分类示例使用 Perceiver 添加文档字符串
# 继承自 PerceiverPreTrainedModel 类
class PerceiverForSequenceClassification(PerceiverPreTrainedModel):
    # 初始化函数，接受 PerceiverConfig 类型的参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 定义可训练的位置编码参数
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 PerceiverModel 对象
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverTextPreprocessor(config),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # 初始化权重并应用最终处理
        self.post_init()
    # 添加模型前向传播的文档字符串，包含Perceiver模型输入的说明
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值的文档字符串，指定输出类型为PerceiverClassifierOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        input_ids: Optional[torch.Tensor] = None,
# 导入必要的库
@add_start_docstrings(
    """
Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses learned position embeddings. In other words, this model is not given any privileged information about
the structure of images. As shown in the paper, this model can achieve a top-1 accuracy of 72.7 on ImageNet.

[`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]
(with `prep_type="conv1x1"`) to preprocess the input images, and
[`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of
[`PerceiverModel`] into classification logits.
""",
    PERCEIVER_START_DOCSTRING,
)
# 定义 PerceiverForImageClassificationLearned 类，继承自 PerceiverPreTrainedModel
class PerceiverForImageClassificationLearned(PerceiverPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)

        # 定义可训练的位置编码参数
        trainable_position_encoding_kwargs_preprocessor = {"num_channels": 256, "index_dims": config.image_size**2}
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 PerceiverModel 模型
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="trainable",
                concat_or_add_pos="concat",
                project_pos_dim=256,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,



@add_start_docstrings(
    """
Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses fixed 2D Fourier position embeddings. As shown in the paper, this model can achieve a top-1 accuracy of
79.0 on ImageNet, and 84.5 when pre-trained on a large-scale dataset (i.e. JFT).
""",
    PERCEIVER_START_DOCSTRING,
)
# 使用PerceiverForImageClassificationLearned类，使用PerceiverImagePreprocessor对输入图像进行预处理，使用PerceiverClassificationDecoder将PerceiverModel的潜在表示解码为分类logits
# 初始化PerceiverForImageClassificationFourier类，继承自PerceiverPreTrainedModel
class PerceiverForImageClassificationFourier(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设置Fourier位置编码的预处理器参数
        fourier_position_encoding_kwargs_preprocessor = {
            "concat_pos": True,
            "max_resolution": (224, 224),
            "num_bands": 64,
            "sine_only": False,
        }
        # 设置可训练位置编码的解码器参数
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # 获取类别数量和PerceiverModel对象
        self.num_labels = config.num_labels
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="pixels",
                spatial_downsample=1,
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,



# 使用PerceiverForImageClassificationConvProcessing类，示例用于图像分类，如ImageNet
# 该模型使用2D conv+maxpool预处理网络。如论文所示，该模型在ImageNet上可以达到82.1的top-1准确率
# 使用PerceiverForImageClassificationLearned类，使用PerceiverImagePreprocessor对输入图像进行预处理，使用PerceiverClassificationDecoder将PerceiverModel的潜在表示解码为分类logits
# 初始化PerceiverForImageClassificationConvProcessing类，继承自PerceiverPreTrainedModel
class PerceiverForImageClassificationConvProcessing(PerceiverPreTrainedModel):
    # 初始化函数，接受配置参数并调用父类的初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 定义傅立叶位置编码的预处理参数
        fourier_position_encoding_kwargs_preprocessor = {
            "concat_pos": True,
            "max_resolution": (56, 56),
            "num_bands": 64,
            "sine_only": False,
        }
        # 定义可训练位置编码的解码器参数
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # 设置类别数量
        self.num_labels = config.num_labels
        # 初始化感知器模型
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv",
                spatial_downsample=1,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数并返回输出结果
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
# 使用 Perceiver 处理光流的示例，适用于 Sintel 和 KITTI 等任务。PerceiverForOpticalFlow 使用 PerceiverImagePreprocessor（prep_type="patches"）对输入图像进行预处理，并使用 PerceiverOpticalFlowDecoder 解码 PerceiverModel 的潜在表示。

# 作为输入，将两个连续帧沿通道维度连接起来，并在每个像素周围提取一个 3 x 3 的补丁（导致每个像素有 54 个值）。使用固定的傅立叶位置编码来编码每个像素在补丁中的位置。接下来，应用 Perceiver 编码器。为了解码，使用与输入相同的编码查询潜在表示。

@add_start_docstrings(
    """
Example use of Perceiver for optical flow, for tasks such as Sintel and KITTI. [`PerceiverForOpticalFlow`] uses
[`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`] (with *prep_type="patches"*) to preprocess the
input images, and [`~models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder`] to decode the latent
representation of [`PerceiverModel`].

As input, one concatenates 2 subsequent frames along the channel dimension and extract a 3 x 3 patch around each pixel
(leading to 3 x 3 x 3 x 2 = 54 values for each pixel). Fixed Fourier position encodings are used to encode the position
of each pixel in the patch. Next, one applies the Perceiver encoder. To decode, one queries the latent representation
using the same encoding used for the input.
""",
    PERCEIVER_START_DOCSTRING,
)

# PerceiverForOpticalFlow 类继承自 PerceiverPreTrainedModel
class PerceiverForOpticalFlow(PerceiverPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 预处理器的傅立叶位置编码参数
        fourier_position_encoding_kwargs_preprocessor = {
            "num_bands": 64,
            "max_resolution": config.train_size,
            "sine_only": False,
            "concat_pos": True,
        }
        # 解码器的傅立叶位置编码参数
        fourier_position_encoding_kwargs_decoder = {
            "concat_pos": True,
            "max_resolution": config.train_size,
            "num_bands": 64,
            "sine_only": False,
        }

        # 创建图像预处理器
        image_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="patches",
            spatial_downsample=1,
            conv_after_patching=True,
            conv_after_patching_in_channels=54,
            temporal_downsample=2,
            position_encoding_type="fourier",
            # 位置编码参数
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
        )

        # 创建 PerceiverModel
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=image_preprocessor,
            decoder=PerceiverOpticalFlowDecoder(
                config,
                num_channels=image_preprocessor.num_channels,
                output_image_shape=config.train_size,
                rescale_factor=100.0,
                # 解码器参数
                use_query_residual=False,
                output_num_channels=2,
                # 使用第一帧特征而不是标准解码器位置编码来查询解码器
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
            ),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,  # 输入数据张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签张量，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
    ) -> Union[Tuple, PerceiverClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the optical flow loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import PerceiverForOpticalFlow
        >>> import torch

        >>> model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")

        >>> # in the Perceiver IO paper, the authors extract a 3 x 3 patch around each pixel,
        >>> # leading to 3 x 3 x 3 = 27 values for each pixel (as each pixel also has 3 color channels)
        >>> # patches have shape (batch_size, num_frames, num_channels, height, width)
        >>> # the authors train on resolutions of 368 x 496
        >>> patches = torch.randn(1, 2, 27, 368, 496)
        >>> outputs = model(inputs=patches)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 368, 496, 2]
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict不为None，则使用return_dict，否则使用self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]  # 如果return_dict为True，则使用outputs.logits，否则使用outputs的第一个元素

        loss = None
        if labels is not None:
            raise NotImplementedError("Optical flow training is not yet supported")  # 如果labels不为None，则抛出未实现错误

        if not return_dict:
            output = (logits,) + outputs[2:]  # 如果不返回字典，则将logits和outputs的第三个元素之后的元素组成元组
            return ((loss,) + output) if loss is not None else output  # 如果loss不为None，则将loss和output组成元组返回，否则返回output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
# 导入必要的库和模块
@add_start_docstrings(
    """
Example use of Perceiver for multimodal (video) autoencoding, for tasks such as Kinetics-700.

[`PerceiverForMultimodalAutoencoding`] uses [`~models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor`] to
preprocess the 3 modalities: images, audio and class labels. This preprocessor uses modality-specific preprocessors to
preprocess every modality separately, after which they are concatenated. Trainable position embeddings are used to pad
each modality to the same number of channels to make concatenation along the time dimension possible. Next, one applies
the Perceiver encoder.

[`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] is used to decode the latent representation of
[`PerceiverModel`]. This decoder uses each modality-specific decoder to construct queries. The decoder queries are
created based on the inputs after preprocessing. However, autoencoding an entire video in a single forward pass is
computationally infeasible, hence one only uses parts of the decoder queries to do cross-attention with the latent
representation. This is determined by the subsampled indices for each modality, which can be provided as additional
input to the forward pass of [`PerceiverForMultimodalAutoencoding`].

[`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] also pads the decoder queries of the different
modalities to the same number of channels, in order to concatenate them along the time dimension. Next, cross-attention
is performed with the latent representation of [`PerceiverModel`].

Finally, [`~models.perceiver.modeling_perceiver.PerceiverMultiModalPostprocessor`] is used to turn this tensor into an
actual video. It first splits up the output into the different modalities, and then applies the respective
postprocessor for each modality.

Note that, by masking the classification label during evaluation (i.e. simply providing a tensor of zeros for the
"label" modality), this auto-encoding model becomes a Kinetics 700 video classifier.
""",
    PERCEIVER_START_DOCSTRING,
)
# 定义 PerceiverForMultimodalAutoencoding 类，继承自 PerceiverPreTrainedModel
class PerceiverForMultimodalAutoencoding(PerceiverPreTrainedModel):
    # 重写 forward 方法
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 forward 方法的参数和返回值
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        subsampled_output_points: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
# Below: position encodings

# 定义构建位置编码的函数
def build_position_encoding(
    position_encoding_type,
    out_channels=None,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    # 定义一个参数，用于传递傅立叶位置编码的参数，默认为None
    """
    Builds the position encoding.

    Args:
    - out_channels: refers to the number of channels of the position encodings.
    - project_pos_dim: if specified, will project the position encodings to this dimension.

    """

    # 根据位置编码类型选择不同的位置编码方式
    if position_encoding_type == "trainable":
        # 如果选择可训练的位置编码，则使用PerceiverTrainablePositionEncoding类
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = PerceiverTrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # 如果选择傅立叶位置编码，则使用PerceiverFourierPositionEncoding类
        # 我们不使用index_dims参数，因为这只在前向传递期间才知道
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = PerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    # 可选地，将位置编码投影到目标维度
    positions_projection = nn.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()

    return output_pos_enc, positions_projection


# 以下是Perceiver解码器


class PerceiverAbstractDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver抽象解码器。"""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_query_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, query, z, query_mask=None):
        raise NotImplementedError


class PerceiverProjectionDecoder(PerceiverAbstractDecoder):
    """
    基线投影解码器（无交叉注意力）。

    Args:
        config ([`PerceiverConfig`]):
            模型配置。
    """

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.d_latents, config.num_labels)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return None

    def forward(
        self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        # (batch_size, num_latents, d_latents) -> (batch_size, d_latents)
        z = torch.mean(z, dim=1)
        # (batch_size, d_latents) -> (batch_size, config.num_labels)
        logits = self.classifier(z)
        return logits


class PerceiverBasicDecoder(PerceiverAbstractDecoder):
    """
    基于交叉注意力的解码器。此类可用于使用交叉注意力操作解码潜在状态的最终隐藏状态，其中潜在状态生成键和值。

    此类的输出形状取决于如何定义输出查询（也称为解码器查询）。
    # 定义了一个 PerceiverConvQueryGenerator 类的构造函数
    def __init__(
        self,
        # 接受 PerceiverConfig 类型的配置参数
        config: PerceiverConfig,
        # 指定输出通道数量
        output_num_channels: int,
        # 指定位置编码的类型，可以是"trainable"、"fourier"或"none"
        position_encoding_type: Optional[str] = "trainable",
        # 如果 position_encoding_type 不为"none"，则指定输出查询的维度数量
        output_index_dims: Optional[int] = None,
        # 如果 position_encoding_type 不为"none"，则指定查询通道数量
        num_channels: Optional[int] = 128,
        # 子采样后的索引维度数量
        subsampled_index_dims: Optional[int] = None,
        # 交叉注意力层中查询和键的通道数量
        qk_channels: Optional[int] = None,
        # 交叉注意力层中值的通道数量
        v_channels: Optional[int] = None,
        # 交叉注意力层中注意力头的数量
        num_heads: Optional[int] = 1,
        # 交叉注意力层的拓宽因子
        widening_factor: Optional[int] = 1,
        # 是否使用查询的残差连接
        use_query_residual: Optional[bool] = False,
        # 是否将预处理的输入拼接到查询上
        concat_preprocessed_input: Optional[bool] = False,
        # 是否将交叉注意力层的输出投射到目标维度
        final_project: Optional[bool] = True,
        # 是否仅用于定义输出查询
        position_encoding_only: Optional[bool] = False,
        # 位置编码其他参数
        **position_encoding_kwargs,
    ):
    # 指定方法的返回值类型为 None
    ) -> None:
        # 调用父类的构造函数
        super().__init__()

        # 设置输出的通道数
        self.output_num_channels = output_num_channels
        # 如果设置为 `none`，则解码器不会构建任何位置编码
        # 在查询解码器时，您应该自行构建位置编码
        self.output_position_encodings = None
        # 位置编码类型
        self.position_encoding_type = position_encoding_type
        # 位置编码的参数
        self.position_encoding_kwargs = position_encoding_kwargs
        # 如果位置编码类型不为 "none"
        if position_encoding_type != "none":
            # 构建位置编码
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )

        # 输出索引维度
        self.output_index_dims = output_index_dims
        # 通道数
        self.num_channels = num_channels
        # 如果下采样的索引维度为 None，则设为输出索引维度
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        # 是否在预处理输入时进行连接
        self.concat_preprocessed_input = concat_preprocessed_input
        # 是否进行最终投影
        self.final_project = final_project
        # 仅包含位置编码
        self.position_encoding_only = position_encoding_only

        # 对于多模态自编码，我们不需要解码器的交叉注意力和最终层
        # 因此将 position_encoding_only 设置为 True
        if not self.position_encoding_only:
            # 解码器交叉注意力
            self.decoding_cross_attention = PerceiverLayer(
                config,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                num_heads=num_heads,
                q_dim=num_channels,
                kv_dim=config.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
            )
            # 最终层
            self.final_layer = nn.Linear(num_channels, output_num_channels) if final_project else nn.Identity()

    # 查询通道数的属性访问器
    @property
    def num_query_channels(self) -> int:
        # 如果位置编码类型为 "none"，则抛出异常
        if self.position_encoding_type == "none":  # 查询来自其他地方
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        # ��果仅包含位置编码
        if self.position_encoding_only:
            # 如果在位置编码参数中包含 "project_pos_dim"，则返回该值
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            # 返回输出的位置编码大小
            return self.output_position_encodings.output_size()
        # 如果最终进行投影
        if self.final_project:
            return self.output_num_channels
        # 返回通道数
        return self.num_channels
```      
    # 该函数用于根据输入生成解码器查询
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        # 如果位置编码类型设置为"none"，则不能构建解码器查询
        if self.position_encoding_type == "none":
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
        
        # 如果提供了子采样点
        if subsampled_points is not None:
            # 使用 unravel_index 函数获取未展平的数组的索引
            indices = [torch.from_numpy(x) for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)]
            # 将索引张量堆叠为 [n, d] 形状
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # 将坐标映射到 [-1, 1] 区间
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            
            # 构建位置编码
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, dtype=inputs.dtype, pos=pos
                )
            
            # 将位置编码投影到目标维度
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]
            
            # 构建位置编码
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    index_dims, batch_size, device=inputs.device, dtype=inputs.dtype
                )
            
            # 将位置编码投影到目标维度
            pos_emb = self.positions_projection(pos_emb)
        
        # 如果concat_preprocessed_input为True，则将输入与位置编码连接
        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)
        
        return pos_emb
    
    # 该函数用于执行前向传播
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        ...
    ) -> PerceiverDecoderOutput:
        # 使用交叉注意力机制进行解码
        # key, value: 形状为 B x N x K; query: 形状为 B x M x K
        # 注意力图 -> 形状为 B x N x M
        # 输出 -> 形状为 B x M x K
        cross_attentions = () if output_attentions else None  # 根据是否需要输出注意力，初始化交叉注意力变量

        layer_outputs = self.decoding_cross_attention(
            query,  # 输入查询数据，形状为 B x M x K
            attention_mask=query_mask,  # 查询掩码
            head_mask=None,  # 没有头掩码
            inputs=z,  # 输入数据，形状为 B x N x K
            inputs_mask=None,  # 输入掩码
            output_attentions=output_attentions,  # 指示是否输出注意力
        )
        output = layer_outputs[0]  # 从层输出中获取解码后的输出

        # 如果需要输出注意力，则将其添加到交叉注意力中
        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        logits = self.final_layer(output)  # 使用最终层处理输出数据，生成 logits

        # 返回解码器输出，包含 logits 和交叉注意力信息
        return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)
class PerceiverClassificationDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention based classification decoder. Light-weight wrapper of [`PerceiverBasicDecoder`] for logit output.
    Will turn the output of the Perceiver encoder which is of shape (batch_size, num_latents, d_latents) to a tensor of
    shape (batch_size, num_labels). The queries are of shape (batch_size, 1, num_labels).

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config, **decoder_kwargs):
        super().__init__()

        self.num_labels = config.num_labels  # 获取分类的数量
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=self.num_labels,
            output_index_dims=1,  # Predict a single logit array.
            **decoder_kwargs,
        )  # 初始化基本解码器

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels  # 返回查询通道的数量

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )  # 解码查询

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> PerceiverDecoderOutput:
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)  # 获取解码器输出

        # B x 1 x num_classes -> B x num_classes
        logits = decoder_outputs.logits[:, 0, :]  # 转化为正确的形状

        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)  # 返回解码器输出


class PerceiverOpticalFlowDecoder(PerceiverAbstractDecoder):
    """Cross-attention based optical flow decoder."""

    def __init__(self, config, output_image_shape, output_num_channels=2, rescale_factor=100.0, **decoder_kwargs):
        super().__init__()

        self.output_image_shape = output_image_shape  # 输出图像的形状
        self.output_num_channels = output_num_channels  # 输出的通道数量
        self.rescale_factor = rescale_factor  # 重新缩放因子
        self.decoder = PerceiverBasicDecoder(config, output_num_channels=output_num_channels, **decoder_kwargs)  # 初始化基本解码器

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels  # 返回查询通道的数量

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")  # 抛出不支持子采样的异常
        return inputs

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    
# 定义 PerceiverBasicVideoAutoencodingDecoder 类，继承自 PerceiverAbstractDecoder 类
class PerceiverBasicVideoAutoencodingDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention based video-autoencoding decoder. Light-weight wrapper of [*PerceiverBasicDecoder*] with video
    reshaping logic.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_shape (`List[int]`):
            Shape of the output as (batch_size, num_frames, height, width), excluding the channel dimension.
        position_encoding_type (`str`):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
    """

    # 初始化方法
    def __init__(
        self, config: PerceiverConfig, output_shape: List[int], position_encoding_type: str, **decoder_kwargs
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 检查输出形状是否为 rank 4
        if len(output_shape) != 4:  # B, T, H, W
            raise ValueError(f"Expected rank 4 output_shape, got {output_shape}.")
        # 存储输出形状和输出通道数
        self.output_shape = output_shape
        self.output_num_channels = decoder_kwargs["output_num_channels"]

        # 构建解码器组件
        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=self.output_shape[1:4],  # T*H*W
            position_encoding_type=position_encoding_type,
            **decoder_kwargs,
        )

    # 返回查询通道数的属性
    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    # 解码器查询方法
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs,
            modality_sizes=modality_sizes,
            inputs_without_pos=inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    # 前向传播方法
    def forward(
        self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor] = None
    ) -> PerceiverDecoderOutput:
        # 解码器输出
        decoder_outputs = self.decoder(query, z)
        logits = decoder_outputs.logits

        # 重新整形 logits
        logits = torch.reshape(logits, self.output_shape + [logits.shape[-1]])
        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)


# 重新结构化方法
def restructure(modality_sizes: ModalitySizeType, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """
    Partitions a [B, N, C] tensor into tensors for each modality.

    Args:
        modality_sizes
            dict specifying the size of the modality
        inputs:
            input tensor

    Returns:
        dict mapping name of modality to its associated tensor.
    """
    outputs = {}
    index = 0
    # 对模态应用可预测的排序
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


# 定义 PerceiverMultimodalDecoder 类，继承自 PerceiverAbstractDecoder 类
class PerceiverMultimodalDecoder(PerceiverAbstractDecoder):
    """
    # 多模态解码器，通过组合单模态解码器实现。构造函数的 *modalities* 参数是一个字典，将模态名称映射到该模态的解码器。该解码器将用于构建该模态的查询。特定于模态的查询在可训练的模态特定参数填充后进行填充，然后沿时间维度进行连接。
    
    # 接下来，对所有模态进行共享的交叉注意力操作。
    
    # 参数:
    #     config ([*PerceiverConfig*]):
    #         模型配置。
    #     modalities (`Dict[str, PerceiverAbstractDecoder]`):
    #         将模态名称映射到该模态的解码器的字典。
    #     num_outputs (`int`):
    #         解码器的输出数量。
    #     output_num_channels (`int`):
    #         输出中的通道数。
    #     min_padding_size (`int`, *可选*, 默认为2):
    #         所有模态的最小填充大小。最终输出的通道数等于所有模态中最大的通道数加上 min_padding_size。
    #     subsampled_index_dims (`Dict[str, PerceiverAbstractDecoder]`, *可选*):
    #         将模态名称映射到该模态的解码器查询要使用的子采样索引维度的字典。
    # """
    
    def __init__(
        self,
        config: PerceiverConfig,
        modalities: Dict[str, PerceiverAbstractDecoder],
        num_outputs: int,
        output_num_channels: int,
        min_padding_size: Optional[int] = 2,
        subsampled_index_dims: Optional[Dict[str, PerceiverAbstractDecoder]] = None,
        **decoder_kwargs,
    ) -> None:
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)  # 创建模态解码器字典模块
        self.subsampled_index_dims = subsampled_index_dims  # 存储子采样索引维度
        self.min_padding_size = min_padding_size  # 存储最小填充大小
        self.output_num_channels = output_num_channels  # 存储输出中的通道数
        self.num_outputs = num_outputs  # 存储解码器的输出数量
        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=(num_outputs,),
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            num_channels=self.num_query_channels,
            **decoder_kwargs,
        )  # 创建基础解码器
        self.padding = nn.ParameterDict(  # 创建包含模态填充参数的字典
            {
                modality: nn.Parameter(torch.randn(1, self.num_query_channels - decoder.num_query_channels))
                for modality, decoder in modalities.items()
            }
        )
    
    @property
    def num_query_channels(self) -> int:
        max_channel_size = max(decoder.num_query_channels for _, decoder in self.modalities.items())  # 获取模态中查询通道数的最大值
        common_channel_size = max_channel_size + self.min_padding_size  # 计算通用通道数
        return common_channel_size  # 返回通用通道数
    # 该函数用于根据不同的模态大小对输入进行划分，并获取每个模态的解码器查询
    def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None):
        # 根据模态大小对输入进行重组
        inputs = restructure(modality_sizes, inputs)
    
        # 获取每个模态对应的解码器查询
        subsampled_points = subsampled_points or {}
        decoder_queries = {}
        for modality, decoder in self.modalities.items():
            # 获取当前模态的输入（不包含位置编码）
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            # 获取当前模态的解码器查询
            query = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,
                inputs_without_pos=input_without_pos,
                subsampled_points=subsampled_points.get(modality, None),
            )
            decoder_queries[modality] = query
    
        # 使用可训练的位置编码填充所有查询，使其具有相同的通道数
        def embed(modality, x):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = self.padding[modality]
            pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
            return torch.cat([x, pos], dim=2)
    
        # 以排序的方式连接所有模态的解码器查询
        return torch.cat(
            [embed(modality, decoder_queries[modality]) for modality in sorted(self.modalities.keys())], dim=1
        )
    
    # 该函数用于根据输入的查询和隐藏状态输出解码结果
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        # 将 B x 1 x num_classes 的查询转换为 B x num_classes 的输出
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        return decoder_outputs
# 下面是为 Perceiver 设计的 IO 预处理和后处理类。
def space_to_depth(frames: torch.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1) -> torch.Tensor:
    """
    空间转深度变换。将空间数据块重新排列为深度数据块。

    此函数假设通道在前，但在变换后会将通道置于最后。

    基于 https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15。
    """
    # 检查输入张量的维度
    if len(frames.shape) == 4:
        # 获取批次大小、通道数、高度和宽度
        batch_size, num_channels, height, width = frames.shape
        # 将图像按空间块大小切分
        frames = frames.view(
            batch_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # 将块移到最后一个维度：(批次大小, H//bs, W//bs, bs, bs, 通道数)
        frames = frames.permute(0, 2, 4, 3, 5, 1).contiguous()
        # 沿通道维度连接块：(批次大小, H//bs, W//bs, bs*bs*通道数)
        frames = frames.view(
            batch_size,
            height // spatial_block_size,
            width // spatial_block_size,
            (spatial_block_size**2) * num_channels,
        )
        return frames
    elif len(frames.shape) == 5:
        # 获取批次大小、时间步长、通道数、高度和宽度
        batch_size, time, num_channels, height, width = frames.shape
        # 将图像按时间和空间块大小切分
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            temporal_block_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # 将块移到最后一个维度：(批次大小, T//ts, H//bs, W//bs, ts, bs, bs, 通道数)
        frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        # 沿通道维度连接块：(批次大小, T//ts, H//bs, W//bs, ts*bs*bs*通道数)
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            height // spatial_block_size,
            width // spatial_block_size,
            temporal_block_size * (spatial_block_size**2) * num_channels,
        )
        return frames
    else:
        # 抛出异常，输入张量的维度不正确
        raise ValueError(
            "Frames should be of rank 4 (batch, channels, height, width)"
            " or rank 5 (batch, time, channels, height, width)"
        )


class Conv2dSamePadding(nn.Conv2d):
    """
    带有 padding="same" 支持的 Conv2d 层。来源：
    https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
    """
    # 定义一个名为Conv2dSamePadding的类，继承自父类nn.Module
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        # 计算得到每一维度的padding大小，并创建ZeroPad2d对象
        self.zero_pad_2d = nn.ZeroPad2d(
            # 遍历卷积核的尺寸，计算每一维度的padding大小，并将其累加起来
            reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        )
    
    # 定义前向传播方法
    def forward(self, input):
        # 对输入的input进行零填充，并使用零填充后的input，权重和偏执参数进行卷积操作
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
class Conv2DDownsample(nn.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(
        self,
        num_layers: int = 1,
        in_channels: int = 3,
        out_channels: int = 64,
        use_batchnorm: bool = True,
    ):
        """
        Constructs a Conv2DDownsample model.

        Args:
          in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
          out_channels (`int`, *optional*, defaults to 64):
            The number of conv output channels.
          use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batchnorm.
        """
        super().__init__()

        # Define a 2D convolution layer with kernel size 7 and stride 2
        self.conv = Conv2dSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, bias=False
        )
        # Add batch normalization if specified, otherwise use identity
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()  # ReLU activation function
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)  # Max pooling layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)  # Pass inputs through the convolutional layer
        out = self.batchnorm(out)  # Apply batch normalization
        out = self.relu(out)  # Apply ReLU activation
        out = self.max_pool(out)  # Perform max pooling
        return out


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.

    Args:
      pos (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`):
        The Tensor containing the position of n points in d dimensional space.
      num_bands (`int`):
        The number of frequency bands (K) to use.
      max_resolution (`Tuple[int]`, *optional*, defaults to (224, 224)):
        The maximum resolution (i.e. the number of pixels per dim). A tuple representing resolution for each dimension.
      concat_pos (`bool`, *optional*, defaults to `True`):
        Whether to concatenate the input position encoding to the Fourier features.
      sine_only (`bool`, *optional*, defaults to `False`):
        Whether to use a single phase (sin) or two (sin/cos) for each frequency band.

    Returns:
      `torch.FloatTensor` of shape `(batch_size, sequence_length, n_channels)`: The Fourier position embeddings. If
      `concat_pos` is `True` and `sine_only` is `False`, output dimensions are ordered as: [dim_1, dim_2, ..., dim_d,
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1),..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]  # Get the batch size

    min_freq = 1.0  # Define the minimum frequency
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # 生成大小为 [n, d * num_bands] 的特征向量
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    # 重新调整特征向量的形状为 [-1, 所有列的乘积]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # 使用正弦函数处理特征向量，输出大小为 [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # 使用正弦和余弦函数处理特征向量，输出大小为 [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # 拼接原始输入位置信息
    if concat_pos:
        # 在编码中添加 d 个频段
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    # 返回处理后的特征向量
    return per_pos_features
# 生成一个 N 维输入数组的位置索引数组
def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    # 定义一个函数来创建 1 维的线性间隔数组
    def _linspace(n_xels_per_dim):
        # 创建一个从 output_range 的最小值到最大值之间的等间隔线性数组
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)
    
    # 遍历每个维度的单元格数，对每个维度创建线性间隔数组
    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    
    # 使用 meshgrid 函数创建一个 N 维的网格，每个维度对应一个线性间隔数组
    array_index_grid = meshgrid(*dim_ranges, indexing="ij")
    
    # 将 N 维网格堆叠起来，形成最终的位置索引数组
    return torch.stack(array_index_grid, dim=-1)


# 感知器抽象位置编码模块
class PerceiverAbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    # 获取位置编码的维度数
    @property
    @abc.abstractmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError
    
    # 获取位置编码的输出大小
    @abc.abstractmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError
    
    # 执行位置编码的前向传播
    @abc.abstractmethod
    def forward(self, batch_size, pos):
        raise NotImplementedError


# 可训练的位置编码模块
class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        # 设置位置编码的通道数
        self._num_channels = num_channels
        # 保存输入数组的维度信息
        self._index_dims = index_dims
        # 计算总的单元格数
        index_dim = np.prod(index_dims)
        # 创建可训练的位置嵌入向量
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channels))
    
    # 获取位置编码的维度数
    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)
    
    # 获取位置编码的输出大小
    def output_size(self, *args, **kwargs) -> int:
        return self._num_channels
    
    # 执行位置编码的前向传播
    def forward(self, batch_size: int) -> torch.Tensor:
        # 获取位置嵌入向量
        position_embeddings = self.position_embeddings
        
        # 如果有 batch 维度，则复制嵌入向量到 batch 维度
        if batch_size is not None:
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        
        return position_embeddings


# 检查或构建空间位置特征
def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    检查或构建空间位置特征 (x, y, ...).

    参数:
      pos (`torch.FloatTensor`):
        None 或位置特征数组。如果为 None，则构建位置特征。否则，检查其大小。
      index_dims (`List[int]`):
        要素化数据的空间/索引大小的可迭代对象。
      batch_size (`int`):
        要素化数据的批量大小。

    返回:
        `torch.FloatTensor` of shape `(batch_size, prod(index_dims))` 位置特征数组.
    """
    # 如果未提供位置信息，则根据索引维度构建线性位置信息
    if pos is None:
        pos = build_linear_positions(index_dims)
        # 相当于 `torch.broadcast_to(pos[None], (batch_size,) + pos.shape)`，
        # 但 `torch.broadcast_to` 无法转换为 ONNX
        # 将位置信息扩展到与批次维度相同的形状
        pos = pos[None].expand((batch_size,) + pos.shape)
        # 重新整形位置信息，以 [batch_size, 索引维度的乘积, -1] 形式表示
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        # 警告：您可能不希望空间特征与位置坐标系具有不同的空间布局。
        # 如果您认为会有效果，请随意覆盖！
        # 如果位置信息的最后一个维度与索引维度的长度不相等，则引发错误
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    # 返回位置信息
    return pos
class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 Fourier 位置编码的参数
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        # 返回最大分辨率的维度数
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        # 计算位置编码的输出维度
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(
        self,
        index_dims: List[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        pos: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        # 检查或构建空间位置
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        # 生成 Fourier 特征
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device=device, dtype=dtype)
        return fourier_pos_enc


class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
        # 抽象方法，需要在子类中实现
        raise NotImplementedError()


class PerceiverTextPreprocessor(AbstractPreprocessor):
    """
    Text preprocessing for Perceiver Encoder. Can be used to embed `inputs` and add positional encodings.

    The dimensionality of the embeddings is determined by the `d_model` attribute of the configuration.

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config = config
        # 创建词嵌入层和位置编码层
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

    @property
    def num_channels(self) -> int:
        # 返回词嵌入的维度大小
        return self.config.d_model

    def forward(self, inputs: torch.LongTensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # 获取输入的词嵌入
        embeddings_without_pos = self.embeddings(inputs)

        # 获取序列长度
        seq_length = inputs.shape[1]
        # 构建位置索引
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        # 添加位置编码到词嵌入上
        embeddings = embeddings_without_pos + self.position_embeddings(position_ids)

        return embeddings, None, embeddings_without_pos


class PerceiverEmbeddingDecoder(nn.Module):
    """
    Module to decode embeddings (for masked language modeling).
    """
    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """
    # 初始化函数，接收一个PerceiverConfig类型的参数并设置配置
    def __init__(self, config: PerceiverConfig) -> None:
        # 调用父类初始化函数
        super().__init__()
        # 保存配置参数
        self.config = config
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 初始化偏置
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))

    # 前向传播函数，接收隐藏状态和嵌入层作为参数，并返回预测输出
    def forward(self, hidden_states: torch.Tensor, embedding_layer: torch.Tensor) -> torch.Tensor:
        # 获取批量大小、序列长度和模型维度
        batch_size, seq_len, d_model = hidden_states.shape
        # 将隐藏状态展平
        output = torch.matmul(hidden_states.reshape([-1, d_model]), embedding_layer.weight.transpose(0, 1))
        # 加上偏置
        output = output + self.bias
        # 将输出恢复成原始形状
        return output.reshape([batch_size, seq_len, self.vocab_size])
class PerceiverMultimodalPostprocessor(nn.Module):
    """
    Multimodal postprocessing for Perceiver. Can be used to combine modality-specific postprocessors into a single
    postprocessor.

    Args:
          modalities (`Mapping[str, PostprocessorType]`):
            Dictionary mapping modality name to postprocessor class for that modality.
          input_is_dict (`bool`, *optional*, defaults to `False`):
            If True, input is assumed to be dictionary structured, and outputs keep the same dictionary shape. If
            False, input is a tensor which is sliced up during postprocessing by *modality_sizes*.
    """

    def __init__(self, modalities: Mapping[str, PostprocessorType], input_is_dict: bool = False):
        super().__init__()
        # 使用给定的模态列表创建一个模块字典
        self.modalities = nn.ModuleDict(modalities)
        # 指示输入是否为字典结构
        self.input_is_dict = input_is_dict

    def forward(
        self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None
    ) -> Mapping[str, torch.Tensor]:
        if not self.input_is_dict:
            # 如果输入不是字典结构，则按照给定的模态大小对输入进行切片
            if modality_sizes is None:
                raise ValueError("Modality sizes should be specified if input is not a dictionary.")
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)

        # 对每个模态应用对应的后处理器，生成输出字典
        outputs = {
            modality: postprocessor(inputs[modality], pos=pos, modality_sizes=None)
            for modality, postprocessor in self.modalities.items()
        }
        return outputs


class PerceiverClassificationPostprocessor(nn.Module):
    """
    Classification postprocessing for Perceiver. Can be used to convert the decoder output to classification logits.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        in_channels (`int`):
            Number of channels in the input.
    """

    def __init__(self, config: PerceiverConfig, in_channels: int) -> None:
        super().__init__()
        # 创建一个线性层作为分类器
        self.classifier = nn.Linear(in_channels, config.num_labels)

    def forward(self, inputs, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        # 将输入通过分类器转换为分类的logits
        logits = self.classifier(inputs)
        return logits[:, 0, :]


class PerceiverAudioPostprocessor(nn.Module):
    """
    Audio postprocessing for Perceiver. Can be used to convert the decoder output to audio features.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        in_channels (`int`):
            Number of channels in the input.
        postproc_type (`str`, *optional*, defaults to `"patches"`):
            Postprocessor type to use. Currently, only "patches" is supported.
    """
    # 定义 PerceiverClassifier 类，继承自 torch.nn.Module
        def __init__(self, config: PerceiverConfig, in_channels: int, postproc_type: str = "patches") -> None:
            # 调用父类 __init__ 方法
            super().__init__()
    
            # 检查 postproc_type 是否是有效值
            if postproc_type not in ("patches",):  # to be supported: 'conv', 'patches', 'pixels'
                raise ValueError("Invalid postproc_type!")
    
            # 设置分类器参数
            self.classifier = nn.Linear(in_channels, config.samples_per_patch)
    
        # 定义前向传播方法
        def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
            # 使用分类器计算输入的 logits
            logits = self.classifier(inputs)
            # 根据输入的 shape 重新组织输出的 logits
            return torch.reshape(logits, [inputs.shape[0], -1])
class PerceiverProjectionPostprocessor(nn.Module):
    """
    Projection postprocessing for Perceiver. Can be used to project the channels of the decoder output to a lower
    dimension.

    Args:
        in_channels (`int`):
            Number of channels in the input.
        out_channels (`int`):
            Number of channels in the output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 创建一个线性层，用于将输入的通道数投影到输出的通道数
        self.classifier = nn.Linear(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        # 使用线性层对输入进行投影得到logits
        logits = self.classifier(inputs)
        return logits


class PerceiverImagePreprocessor(AbstractPreprocessor):
    """
    Image preprocessing for Perceiver Encoder.

    Note: the *out_channels* argument refers to the output channels of a convolutional layer, if *prep_type* is set to
    "conv1x1" or "conv". If one adds absolute position embeddings, one must make sure the *num_channels* of the
    position encoding kwargs are set equal to the *out_channels*.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"conv"`):
            Preprocessing type. Can be "conv1x1", "conv", "patches", "pixels".
        spatial_downsample (`int`, *optional*, defaults to 4):
            Spatial downsampling factor.
        temporal_downsample (`int`, *optional*, defaults to 1):
            Temporal downsampling factor (only relevant in case a time dimension is present).
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Position encoding type. Can be "fourier" or "trainable".
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input.
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        conv_after_patching (`bool`, *optional*, defaults to `False`):
            Whether to apply a convolutional layer after patching.
        conv_after_patching_in_channels (`int`, *optional*, defaults to 54):
            Number of channels in the input of the convolutional layer after patching.
        conv2d_use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batch normalization in the convolutional layer.
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        # config: 配置参数，用于初始化模型
        config,
        # prep_type: 数据准备类型，默认为卷积（"conv"）
        prep_type="conv",
        # spatial_downsample: 空间下采样倍数，默认为4
        spatial_downsample: int = 4,
        # temporal_downsample: 时间下采样倍数，默认为1
        temporal_downsample: int = 1,
        # position_encoding_type: 位置编码类型，默认为"fourier"
        position_encoding_type: str = "fourier",
        # in_channels: 输入通道数，默认为3
        in_channels: int = 3,
        # out_channels: 输出通道数，默认为64
        out_channels: int = 64,
        # conv_after_patching: 是否在打补丁后进行卷积，默认为False
        conv_after_patching: bool = False,
        # conv_after_patching_in_channels: 在conv_after_patching为True时有效的输入通道数，默认为54
        conv_after_patching_in_channels: int = 54,  # only relevant when conv_after_patching = True
        # conv2d_use_batchnorm: 是否使用批量归一化，默认为True
        conv2d_use_batchnorm: bool = True,
        # concat_or_add_pos: 拼接或相加位置编码，默认为"concat"
        concat_or_add_pos: str = "concat",
        # project_pos_dim: 位置维度投影，默认为-1
        project_pos_dim: int = -1,
        # position_encoding_kwargs: 其他位置编码参数
        **position_encoding_kwargs,
        # 调用父类构造函数
        super().__init__()
        # 将配置信息存储在对象中
        self.config = config

        # 如果预处理类型不在指定的范围内，则引发数值错误
        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        # 如果连接或添加位置不是指定的值，则引发数值错误
        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        # 存储输入通道数和其他指定属性
        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        # 如果预处理类型为"conv"，则执行以下代码块
        if self.prep_type == "conv":
            # 对于使用 conv 进行下采样目前有限制
            # 计算卷积层数
            convnet_num_layers = math.log(spatial_downsample, 4)
            # 检查卷积层数是否为整数
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            # 如果卷积层数不是整数或者时间下采样不等于1，则引发数值错误
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial and 1 expected for temporal downsampling with conv."
                )
            # 创建 Conv2DDownsample 对象来进行下采样
            self.convnet = Conv2DDownsample(
                in_channels=in_channels,
                num_layers=int(convnet_num_layers),
                out_channels=out_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )

        # 如果预处理类型为"conv1x1"，则执行以下代码块
        elif self.prep_type == "conv1x1":
            # 如果时间下采样不等于1，则引发数值错误
            if temporal_downsample != 1:
                raise ValueError("Conv1x1 does not downsample in time.")
            # 创建一个 1x1 卷积进行处理
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                # 对于 1x1 卷积，空间下采样没有限制
                stride=(spatial_downsample, spatial_downsample),
            )

        # 存储位置编码相关的属性
        self.project_pos_dim = project_pos_dim
        # 构建位置编码
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

        # 可选的在补丁之后的卷积层
        self.conv_after_patches = (
            nn.Linear(conv_after_patching_in_channels, self.out_channels) if conv_after_patching else nn.Identity()
        )

    @property
    def num_channels(self) -> int:
        # 返回通道数
        
        # 检查输入数据的分辨率数是否为2或3，如果处理图像则为2，处理视频则为3
        is_temporal = self.position_embeddings.num_dimensions > 2
        
        # 位置嵌入
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim
        
        # 输入
        # 根据不同的预处理类型确定输入维度
        if self.conv_after_patching or self.prep_type in ("conv1x1", "conv"):
            inp_dim = self.out_channels
        elif self.prep_type == "pixels":
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == "patches":
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample**2
                if is_temporal:
                    inp_dim *= self.temporal_downsample
        
        # 返回通道数
        return inp_dim + pos_dim
    
    def _build_network_inputs(self, inputs: torch.Tensor, network_input_is_1d: bool = True):
        """
        构建最终输入，包括位置编码。

        此方法期望输入始终具有通道作为最后一个维度。
        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # 如果输入维度大于3且网络输入是1D，则将输入特征展平为1D索引维度
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # 构建位置编码
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)
        
        # 可选择地将位置编码投影到目标维度
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # 如果网络接受非1D输入，则调整位置以匹配输入特征形状
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc
        return inputs_with_pos, inputs
    # 定义一个前向传播函数，接受输入张量 inputs，位置张量 pos（可选），network_input_is_1d 参数（默认为 True）
    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # 如果准备类型 prep_type 为 "conv"
        if self.prep_type == "conv":
            # 卷积神经网络图像特征提取
            # 空间下采样因子为4
            inputs = self.convnet(inputs)

        # 如果准备类型 prep_type 为 "conv1x1"
        elif self.prep_type == "conv1x1":
            # 将输入映射为 self.out_channels
            inputs = self.convnet_1x1(inputs)

        # 如果准备类型 prep_type 为 "pixels"
        elif self.prep_type == "pixels":
            # 如果要求的话，以最简单的方式进行空间下采样
            if inputs.ndim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :, :: self.temporal_downsample, :, :: self.spatial_downsample, :: self.spatial_downsample
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        # 如果准备类型 prep_type 为 "patches"
        elif self.prep_type == "patches":
            # Space2depth 特征化
            # 视频：B x T x C x H x W
            inputs = space_to_depth(
                inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample
            )

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # 用于光流
                inputs = inputs.squeeze(dim=1)

            # 可选地应用卷积层
            inputs = self.conv_after_patches(inputs)

        # 如果准备类型不是 "patches"
        if self.prep_type != "patches":
            # 将通道移动到最后一个维度，因为下面的 _build_network_inputs 方法期望这样的结构
            if inputs.ndim == 4:
                inputs = inputs.permute(0, 2, 3, 1)
            elif inputs.ndim == 5:
                inputs = inputs.permute(0, 1, 3, 4, 2)
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        # 调用 _build_network_inputs 方法，构建网络输入，同时返回没有位置信息的输入
        inputs, inputs_without_pos = self._build_network_inputs(inputs, network_input_is_1d)
        modality_sizes = None  # 每个模态的尺寸，仅在多模态时需要

        # 返回输入、模态尺寸和没有位置信息的输入
        return inputs, modality_sizes, inputs_without_pos
class PerceiverOneHotPreprocessor(AbstractPreprocessor):
    """
    One-hot preprocessor for Perceiver Encoder. Can be used to add a dummy index dimension to the input.

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config: PerceiverConfig = config

    @property
    def num_channels(self) -> int:
        return self.config.num_labels

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # Add a dummy index dimension.
        inputs = inputs[:, None, :]

        # No position encodings, so the 1st (input) and 3rd (inputs_without_pos)
        # outputs are identical.
        return inputs, None, inputs


class PerceiverAudioPreprocessor(AbstractPreprocessor):
    """
    Audio preprocessing for Perceiver Encoder.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"patches"`):
            Preprocessor type to use. Only "patches" is supported.
        samples_per_patch (`int`, *optional*, defaults to 96):
            Number of samples per patch.
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Type of position encoding to use. Can be "trainable" or "fourier".
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(
        self,
        config,
        prep_type: str = "patches",
        samples_per_patch: int = 96,
        position_encoding_type: str = "fourier",
        concat_or_add_pos: str = "concat",
        out_channels=64,
        project_pos_dim=-1,
        **position_encoding_kwargs,
    ):
        # 调用父类的构造函数初始化
        super().__init__()
        # 设置配置参数
        self.config = config

        # 检查预处理类型是否有效
        if prep_type not in ("patches",):
            raise ValueError(f"Prep_type {prep_type} is invalid, can only be 'patches'.")

        # 检查连接或加法位置是否有效
        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'.")

        # 设置每个补丁的样本数
        self.samples_per_patch = samples_per_patch
        # 设置位置编码类型
        self.position_encoding_type = position_encoding_type
        # 设置连接或加法位置
        self.concat_or_add_pos = concat_or_add_pos
        # 设置投影位置的维度
        self.project_pos_dim = project_pos_dim

        # 构建位置嵌入
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

    @property
    def num_channels(self) -> int:
        # 位置嵌入
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        # 如果连接位置方式为加法，则返回位置维度
        if self.concat_or_add_pos == "add":
            return pos_dim
        # 否则返回每个补丁的样本数加上位置维度
        return self.samples_per_patch + pos_dim

    def _build_network_inputs(self, inputs):
        """构建最终输入，包括位置编码。"""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # 构建位置编码
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)

        # 可选地将它们投影到目标维度
        pos_enc = self.positions_projection(pos_enc)

        # 如果连接位置方式为拼接，则将位置编码与输入拼接在一起
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        # 如果连接位置方式为加法，则将位置编码与输入相加
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # 将输入重塑为[batch_size, 补丁数, 每个补丁的样本数]
        inputs = torch.reshape(inputs, [inputs.shape[0], -1, self.samples_per_patch])

        # 构建具有位置编码的网络输入
        inputs, inputs_without_pos = self._build_network_inputs(inputs)
        modality_sizes = None  # 每种模态的大小，仅对多模态需要

        return inputs, modality_sizes, inputs_without_pos
class PerceiverMultimodalPreprocessor(AbstractPreprocessor):
    """
    Multimodal preprocessing for Perceiver Encoder.

    Inputs for each modality are preprocessed, then padded with trainable position embeddings to have the same number
    of channels.

    Args:
        modalities (`Mapping[str, PreprocessorType]`):
            Dict mapping modality name to preprocessor.
        mask_probs (`Dict[str, float]`):
            Dict mapping modality name to masking probability of that modality.
        min_padding_size (`int`, *optional*, defaults to 2):
            The minimum padding size for all modalities. The final output will have num_channels equal to the maximum
            channels across all modalities plus min_padding_size.
    """

    def __init__(
        self,
        modalities: Mapping[str, PreprocessorType],
        mask_probs: Optional[Mapping[str, float]] = None,
        min_padding_size: int = 2,
    ):
        # 初始化方法，设置各种属性
        super().__init__()
        # 保存传入的各个模态预处理器
        self.modalities = nn.ModuleDict(modalities)
        # 设置最小填充大小
        self.min_padding_size = min_padding_size
        # 如果提供了掩码概率，则保存，否则设为空字典
        self.mask_probs = mask_probs if mask_probs is not None else {}
        # 初始化填充参数，为每个模态分别生成填充参数
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_channels - preprocessor.num_channels))
                for modality, preprocessor in modalities.items()
            }
        )
        # 初始化掩码参数，为每个模态分别生成掩码参数
        self.mask = nn.ParameterDict(
            {modality: nn.Parameter(torch.randn(1, self.num_channels)) for modality, _ in self.mask_probs.items()}
        )

    @property
    def num_channels(self) -> int:
        # 计算所有模态中通道数的最大值，并加上最小填充大小，作为总通道数
        max_channel_size = max(processor.num_channels for _, processor in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def forward(
        self, inputs: Mapping[str, torch.Tensor], pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True
        # 前向传播方法，对输入进行预处理和填充
    # 定义函数的输入和输出类型
    ) -> PreprocessorOutputType:
        # 创建空字典用于存储填充后的数据
        padded = {}
        # 创建空字典用于存储每个模态的尺寸
        modality_sizes = {}
        # 创建空字典用于存储没有位置信息的输入数据
        inputs_without_pos = {}
        # 遍历所有模态和对应的预处理器
        for modality, preprocessor in self.modalities.items():
            # 使用相应的预处理器预处理每个模态
            output, _, inputs_without_pos[modality] = preprocessor(
                inputs[modality], pos=pos, network_input_is_1d=network_input_is_1d
            )

            # 对输出进行填充以保证通道数一致
            batch_size, num_samples, num_channels = output.shape
            pos_enc = self.padding[modality].expand(batch_size, -1, -1)
            padding = torch.broadcast_to(
                pos_enc,
                [batch_size, num_samples, self.num_channels - num_channels],
            )
            output_padded = torch.cat([output, padding], dim=2)

            # 如果需要，进行遮罩处理
            if modality in self.mask_probs:
                mask_token = self.mask[modality].expand(batch_size, -1, -1)
                mask_prob = self.mask_probs[modality]
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2).to(mask_token.device)
                output_padded = (1 - mask) * output_padded + mask * mask_token

            # 将填充后的数据存入字典
            padded[modality] = output_padded
            # 记录填充后数据的尺寸
            modality_sizes[modality] = output_padded.shape[1]

        # 对模态提供一致的顺序
        padded_ls = [padded[k] for k in sorted(padded.keys())]

        # 最终沿着时间维度进行拼接
        final_inputs = torch.cat(padded_ls, dim=1)

        # 返回填充后的最终输入数据，模态尺寸和无位置信息的输入数据
        return final_inputs, modality_sizes, inputs_without_pos
```