# `.\models\perceiver\modeling_perceiver.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，声明代码版权归 Deepmind 和 HuggingFace Inc. 团队所有，保留所有权利

# 根据 Apache 许可证版本 2.0 进行许可
# 在遵守许可证的情况下，您可以使用此文件，详细信息请参见许可证
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0

# 除非法律另有规定或书面同意，否则不得以任何方式使用此软件
# 此软件按"原样"提供，不提供任何明示或暗示的保证或条件
# 请参阅许可证以获取特定于语言的权限和限制
""" PyTorch Perceiver 模型。"""

# 导入必要的库和模块
import abc  # 抽象基类模块
import math  # 数学函数模块
from dataclasses import dataclass  # 用于定义数据类的装饰器
from functools import reduce  # 函数工具模块中的reduce函数
from operator import __add__  # 运算符模块中的add函数
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union  # 类型提示

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 深度学习库
import torch.utils.checkpoint  # PyTorch 的 checkpoint 模块，用于内存优化
from torch import nn  # PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # PyTorch 的损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import BaseModelOutputWithCrossAttentions  # 模型输出类，包含交叉注意力
from ...modeling_utils import PreTrainedModel  # 预训练模型基类
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer  # PyTorch 工具函数
from ...utils import (  # 通用工具函数和类
    ModelOutput,  # 模型输出基类
    add_start_docstrings,  # 为函数添加文档字符串的装饰器
    add_start_docstrings_to_model_forward,  # 为模型前向方法添加文档字符串的装饰器
    logging,  # 日志记录模块
    replace_return_docstrings,  # 替换返回文档字符串的工具函数
)
from .configuration_perceiver import PerceiverConfig  # 导入 Perceiver 模型的配置类

# 类型别名定义
ModalitySizeType = Mapping[str, int]  # 模态大小类型别名，映射字符串到整数
PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]  # 预处理器输出类型别名
PreprocessorType = Callable[..., PreprocessorOutputType]  # 预处理器类型别名
PostprocessorType = Callable[..., Any]  # 后处理器类型别名

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置常量
_CHECKPOINT_FOR_DOC = "deepmind/language-perceiver"  # 模型检查点用于文档
_CONFIG_FOR_DOC = "PerceiverConfig"  # Perceiver 模型配置用于文档

# 预训练模型的存档列表
PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "deepmind/language-perceiver",  # Deepmind 的语言 Perceiver 模型
    # 更多 Perceiver 模型存档可以在此处查看 https://huggingface.co/models?filter=perceiver
]

@dataclass
class PerceiverModelOutput(ModelOutput):
    """
    Perceiver 模型输出的基类，包含可能的隐藏状态、注意力和交叉注意力。
    
    这个类使用 dataclass 装饰器来定义数据类，它是一个轻量级的数据结构，用于表示简单的值对象。
    """

    # 该类用于描述 Perceiver 模型的输出，包含了可能的隐藏状态、注意力和交叉注意力等信息
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
            分类（或回归，如果config.num_labels==1）分数，即SoftMax之前的分数。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
            `torch.FloatTensor`元组（一个用于嵌入输出 + 一个用于每层输出），形状为`(batch_size, sequence_length, hidden_size)`。
            模型每层的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
            `torch.FloatTensor`元组（每层一个），形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
            自注意力头中注意力权重的 softmax 后的结果，用于计算加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
            `torch.FloatTensor`元组（每层一个），形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
            解码器的交叉注意力层中注意力权重的 softmax 后的结果，用于计算加权平均值。
    """

    logits: torch.FloatTensor = None  # 分类（或回归）分数的张量，形状为`(batch_size, num_labels)`
    last_hidden_state: torch.FloatTensor = None  # 模型最后一层输出的隐藏状态张量，形状为`(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型每层的隐藏状态的元组张量，形状为`(batch_size, sequence_length, hidden_size)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 自注意力头中的注意力权重的元组张量，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 交叉注意力头中的注意力权重的元组张量，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
    # 定义 PerceiverDecoderOutput 类，继承自 ModelOutput 类
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

        # 定义 logits 属性，类型为 torch.FloatTensor，形状为 (batch_size, num_labels)，存储基本解码器的输出
        logits: torch.FloatTensor = None
        # 定义 cross_attentions 属性，类型为可选的元组，如果传入参数 output_attentions=True 或者 config.output_attentions=True 则会返回，存储解码器的跨注意力层的注意力权重
        cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


    # 定义 PerceiverMaskedLMOutput 类，继承自 ModelOutput 类
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

        # 定义 loss 属性，类型为可选的 torch.FloatTensor，形状为 (1,)，当提供 labels 参数时返回，存储掩码语言模型（MLM）损失
        loss: Optional[torch.FloatTensor] = None
        # 定义 logits 属性，类型为 torch.FloatTensor，形状为 (batch_size, sequence_length, config.vocab_size)，存储语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）
        logits: torch.FloatTensor = None
        # 定义 hidden_states 属性，类型为可选的元组，如果传入参数 output_hidden_states=True 或者 config.output_hidden_states=True 则会返回，存储模型在每一层输出之后的隐藏状态 plus 初始嵌入输出
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        # 定义 attentions 属性，类型为可选的元组，如果传入参数 output_attentions=True 或者 config.output_attentions=True 则会返回，存储注意力softmax后的注意权重，用于计算自注意力头中的加权平均值
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        # 定义 cross_attentions 属性，类型为可选的元组，如果传入参数 output_attentions=True 或者 config.output_attentions=True 则会返回，存储解码器的跨注意力层的注意力权重
        cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


    # 定义 PerceiverClassifierOutput 类，继承自 ModelOutput 类
    @dataclass
    class PerceiverClassifierOutput(ModelOutput):
    """
    Perceiver 模型的输出基类，适用于序列/图像分类模型、光流和多模态自编码。
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 `config.num_labels==1`）的损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 `config.num_labels==1`）的分数（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含 `torch.FloatTensor` 元组的隐藏状态（如果传递了 `output_hidden_states=True` 或 `config.output_hidden_states=True`）。
            形状为 `(batch_size, sequence_length, hidden_size)`，模型在每一层输出后的隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含 `torch.FloatTensor` 元组的注意力权重（如果传递了 `output_attentions=True` 或 `config.output_attentions=True`）。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含 `torch.FloatTensor` 元组的交叉注意力权重（如果传递了 `output_attentions=True` 或 `config.output_attentions=True`）。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，解码器的交叉注意力层的注意力权重，经过注意力 softmax 后用于计算加权平均值。
    """
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    """实现Perceiver模型的自注意力机制模块。可以用于编码器和解码器中。"""

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
        # Q和K必须具有相同数量的通道。
        # 默认保持Q的输入形状。
        if qk_channels is None:
            qk_channels = q_dim
        # V的通道数确定了QKV-attention输出的形状。
        # 默认使用与键-查询操作中使用的通道数相同。
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels})必须能被num_heads ({num_heads})整除。")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels})必须能被num_heads ({num_heads})整除。")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # 层归一化
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # 投影矩阵
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        """将张量重塑为注意力分数计算所需的形状。"""
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        hidden_states = self.layernorm1(hidden_states)
        # 对隐藏状态进行 Layer Normalization

        inputs = self.layernorm2(inputs)
        # 对输入进行 Layer Normalization

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        # 判断是否为跨注意力模块

        queries = self.query(hidden_states)
        # 从隐藏状态计算查询

        if is_cross_attention:
            keys = self.key(inputs)
            # 如果是跨注意力模块，从输入计算键
            values = self.value(inputs)
            # 如果是跨注意力模块，从输入计算值
            attention_mask = inputs_mask
            # 如果是跨注意力模块，使用输入的注意力掩码
        else:
            keys = self.key(hidden_states)
            # 如果不是跨注意力模块，从隐藏状态计算键
            values = self.value(hidden_states)
            # 如果不是跨注意力模块，从隐藏状态计算值

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        # 调整查询张量以进行多头注意力计算
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        # 调整键张量以进行多头注意力计算
        values = self.transpose_for_scores(values, self.v_channels_per_head)
        # 调整值张量以进行多头注意力计算

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        # 计算查询和键的点积以获得原始注意力分数

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim
        # 计算中间变量

        attention_scores = attention_scores / math.sqrt(q_head_dim)
        # 缩放注意力分数

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask
            # 应用注意力掩码

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # 将注意力分数归一化为概率

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # 使用 dropout 随机丢弃整个 token 的注意力概率，这种做法源自于原始的 Transformer 论文

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            # 如果有头部掩码，应用头部掩码

        context_layer = torch.matmul(attention_probs, values)
        # 计算上下文张量

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 调整上下文张量的维度顺序

        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 调整上下文张量的形状

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # 准备输出

        return outputs
        # 返回计算结果
class PerceiverSelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels):
        super().__init__()
        # 初始化一个全连接层，输入通道数为input_channels，输出通道数为output_channels
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层进行线性变换
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
        super().__init__()
        
        # 根据是否是交叉注意力机制和参数配置设置查询键值通道数和值通道数
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
        
        # 初始化自注意力层
        self.self = PerceiverSelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        
        # 设置输出层，根据是否是交叉注意力机制确定输出通道数
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
        # 如果没有需要剪枝的头部，直接返回
        if len(heads) == 0:
            return
        
        # 寻找可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 self.self 方法处理输入的隐藏状态，返回处理后的输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # 将 self_outputs[0] 通过 self.output 进行输出投影
        attention_output = self.output(self_outputs[0])

        # 如果指定使用查询残差连接
        if self.use_query_residual:
            # 将 attention_output 添加到原始隐藏状态 hidden_states 上
            attention_output = attention_output + hidden_states

        # 组装最终输出，包括 attention_output 和可能的其他输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，也加入到 outputs 中
        return outputs
class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, config, input_size, widening_factor):
        super().__init__()
        # 第一层全连接层，将输入特征大小映射到扩展因子倍数的输入特征大小
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 第二层全连接层，将扩展后的特征映射回原始输入特征大小
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播：全连接层1
        hidden_states = self.dense1(hidden_states)
        # 前向传播：激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 前向传播：全连接层2
        hidden_states = self.dense2(hidden_states)
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
        # 分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 注意力机制
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
        # Layer normalization
        self.layernorm = nn.LayerNorm(q_dim)
        # MLP层
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
        # 调用注意力机制的前向传播
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # 如果输出注意力权重，则添加

        # 对注意力输出应用分块处理，返回分块后的输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # 残差连接
        layer_output = layer_output + attention_output

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        # Layer normalization
        layer_output = self.layernorm(attention_output)
        # MLP层
        layer_output = self.mlp(layer_output)
        return layer_output


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""
    def __init__(self, config, kv_dim=None):
        super().__init__()
        self.config = config

        # Check that we can use multihead-attention with these shapes.
        # 检查是否可以使用这些形状进行多头注意力
        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        # Construct the cross attention layer.
        # 构建跨注意力层
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

        # Construct a single block of self-attention layers.
        # 构建一个自注意力层块
        # 通过多次应用这个块，可以得到更深的网络结构
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
        # 如果不需要输出隐藏状态，设置为空元组；否则初始化为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，设置为空元组；否则初始化为 None
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重，设置为空元组；否则初始化为 None
        all_cross_attentions = () if output_attentions else None

        # 对 latent（hidden_states）和 inputs 之间进行交叉注意力计算：
        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        # 更新 hidden_states 为交叉注意力计算的输出的第一个元素
        hidden_states = layer_outputs[0]

        # 如果需要输出注意力权重，将本次计算的注意力权重添加到 all_cross_attentions 中
        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # 多次应用自注意力层块：
        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                # 如果需要输出隐藏状态，将当前 hidden_states 添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 获取当前层的头部掩码
                layer_head_mask = head_mask[i] if head_mask is not None else None

                # 执行当前自注意力层的前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                # 更新 hidden_states 为当前自注意力层的输出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，将本次计算的注意力权重添加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果需要输出隐藏状态，将当前 hidden_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，将结果以元组形式返回，过滤掉值为 None 的项
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        # 返回一个 BaseModelOutputWithCrossAttentions 对象，包含最后隐藏状态、所有隐藏状态、自注意力和交叉注意力
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

    # 设置默认的配置类为PerceiverConfig
    config_class = PerceiverConfig
    # 指定基础模型前缀为"perceiver"
    base_model_prefix = "perceiver"
    # 指定主要输入名称为"inputs"
    main_input_name = "inputs"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果module是Linear或者Conv2d类型
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重数据，均值为0，标准差为config中指定的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将偏置数据初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果module具有"latents"属性
        elif hasattr(module, "latents"):
            # 使用正态分布初始化latents属性数据，均值为0，标准差为config中指定的initializer_range
            module.latents.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果module具有"position_embeddings"属性并且是PerceiverTrainablePositionEncoding类型
        elif hasattr(module, "position_embeddings") and isinstance(module, PerceiverTrainablePositionEncoding):
            # 使用正态分布初始化position_embeddings数据，均值为0，标准差为config中指定的initializer_range
            module.position_embeddings.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果module是ParameterDict类型
        elif isinstance(module, nn.ParameterDict):
            # 对于每个modality，使用正态分布初始化数据，均值为0，标准差为config中指定的initializer_range
            for modality in module.keys():
                module[modality].data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果module是Embedding类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重数据，均值为0，标准差为config中指定的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了padding_idx，则将该索引位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果module是LayerNorm类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置数据初始化为零
            module.bias.data.zero_()
            # 将权重数据初始化为1
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
    #                           通过配置文件初始化不会加载模型的权重，仅加载配置。
    #                           查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # decoder (*DecoderType*, *optional*):
    #         可选的解码器，用于解码编码器的潜在表示。示例包括
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverBasicDecoder*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder*。
    # input_preprocessor (*PreprocessorType*, *optional*):
    #         可选的输入预处理器。示例包括
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverTextPreprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor*。
    # output_postprocessor (*PostprocessorType*, *optional*):
    #         可选的输出后处理器。示例包括
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverImagePostprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor*,
    #         *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor*。

    # 注意：您可以定义自己的解码器、预处理器和/或后处理器以适应您的使用案例。
"""
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
    """
    The PerceiverModel class implements a perceiver architecture for various input modalities.

    Args:
        config (PretrainedConfig):
            The model configuration class instance.
        decoder (Optional):
            Optional decoder for the model.
        input_preprocessor (PreprocessorType, Optional):
            Optional input preprocessor for handling input data.
        output_postprocessor (PostprocessorType, Optional):
            Optional output postprocessor for handling model outputs.
    """

    def __init__(
        self,
        config,
        decoder=None,
        input_preprocessor: PreprocessorType = None,
        output_postprocessor: PostprocessorType = None,
    ):
        """
        Initialize the PerceiverModel with given configuration and optional components.

        Args:
            config (PretrainedConfig):
                The model configuration class instance.
            decoder (Optional):
                Optional decoder for the model.
            input_preprocessor (PreprocessorType, Optional):
                Optional input preprocessor for handling input data.
            output_postprocessor (PostprocessorType, Optional):
                Optional output postprocessor for handling model outputs.
        """
        super().__init__(config)
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(
            config, kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else config.d_model
        )
        self.decoder = decoder

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the latent embeddings used as inputs to the perceiver model.

        Returns:
            torch.Tensor: The latent embeddings tensor.
        """
        return self.embeddings.latents

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings of the perceiver model.

        Args:
            value (torch.Tensor): The new input embeddings tensor.
        """
        self.embeddings.latents = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (dict):
                Dictionary of {layer_num: list of heads to prune in this layer}. See base class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=PerceiverModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, **inputs):
        """
        Perform a forward pass through the PerceiverModel.

        Args:
            **inputs (keyword arguments):
                The input data. Can contain various inputs depending on the model configuration.

        Returns:
            PerceiverModelOutput or tuple:
                The model outputs. Can contain attentions, hidden states, and additional model-specific outputs.
        """
        return super().forward(**inputs)
    # 定义神经网络模型的前向传播函数
    def forward(
        self,
        # 输入数据张量，通常是浮点型张量
        inputs: torch.FloatTensor,
        # 注意力掩码张量，可选，用于控制注意力机制的作用范围
        attention_mask: Optional[torch.FloatTensor] = None,
        # 子采样输出点的字典，可选，包含不同子样本输出的张量
        subsampled_output_points: Optional[Dict[str, torch.Tensor]] = None,
        # 头部掩码张量，可选，用于掩盖特定头部的注意力权重
        head_mask: Optional[torch.FloatTensor] = None,
        # 是否输出注意力权重信息，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态信息，可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，可选
        return_dict: Optional[bool] = None,
@add_start_docstrings("""Example use of Perceiver for masked language modeling.""", PERCEIVER_START_DOCSTRING)
class PerceiverForMaskedLM(PerceiverPreTrainedModel):
    def __init__(self, config: PerceiverConfig):
        super().__init__(config)

        # 实例化文本预处理器
        text_preprocessor = PerceiverTextPreprocessor(config)

        # 定义用于解码器的可训练位置编码参数
        trainable_position_encoding_kwargs_decoder = {
            "num_channels": text_preprocessor.num_channels,
            "index_dims": config.max_position_embeddings,
        }

        # 创建 PerceiverModel 实例，配置输入预处理器和解码器
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

        # 实例化 PerceiverEmbeddingDecoder
        self.embedding_decoder = PerceiverEmbeddingDecoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
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
        # 具体的模型前向传播方法，参见 PERCEIVER_INPUTS_DOCSTRING 的格式说明
        # 输出类型为 PerceiverMaskedLMOutput，配置类为 _CONFIG_FOR_DOC



@add_start_docstrings("""Example use of Perceiver for text classification.""", PERCEIVER_START_DOCSTRING)
class PerceiverForSequenceClassification(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 定义用于解码器的可训练位置编码参数
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # 设置分类数量
        self.num_labels = config.num_labels

        # 创建 PerceiverModel 实例，配置输入预处理器和分类解码器
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

        # 初始化权重并进行最终处理
        self.post_init()
    # 将模型的输入格式的文档字符串添加到前向传播方法上，描述其参数是批量大小和序列长度
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值的文档字符串，指定输出类型为PerceiverClassifierOutput，并使用_CONFIG_FOR_DOC作为配置类
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,  # 模型的输入张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签张量，默认为None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回，默认为None
        input_ids: Optional[torch.Tensor] = None,  # 输入ID张量，默认为None
@add_start_docstrings(
    """
Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses fixed 2D Fourier position embeddings. As shown in the paper, this model can achieve a top-1 accuracy of
79.0 on ImageNet, and 84.5 when pre-trained on a large-scale dataset (i.e. JFT).
""",
    PERCEIVER_START_DOCSTRING,
)
class PerceiverForImageClassificationFixed(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Define kwargs for trainable position encoding in preprocessor and decoder
        trainable_position_encoding_kwargs_preprocessor = {"num_channels": 256, "index_dims": config.image_size ** 2}
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # Initialize number of labels from config
        self.num_labels = config.num_labels

        # Initialize Perceiver model with fixed 2D Fourier position embeddings
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="fourier",
                concat_or_add_pos="add",
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

        # Initialize weights and apply final processing
        self.post_init()

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
        ):
        """
        Perform forward pass of the PerceiverForImageClassificationFixed model.

        Args:
            inputs (torch.Tensor, optional): Input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Mask tensor indicating which elements should be attended to.
            head_mask (torch.Tensor, optional): Mask tensor for attention heads.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            labels (torch.Tensor, optional): Labels tensor for classification.
            return_dict (bool, optional): Whether to return outputs as a dictionary.
            pixel_values (torch.Tensor, optional): Pixel values tensor for image input.

        Returns:
            PerceiverClassifierOutput or torch.Tensor: Output of the model, depending on return_dict.

        """
        # Forward pass through the Perceiver model
        return self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )
"""
[`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]
(with `prep_type="pixels"`) to preprocess the input images, and
[`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of
[`PerceiverModel`] into classification logits.
""",
PERCEIVER_START_DOCSTRING,
)
class PerceiverForImageClassificationFourier(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设置傅里叶位置编码的预处理器参数
        fourier_position_encoding_kwargs_preprocessor = {
            "concat_pos": True,  # 是否将位置编码与输入数据连接
            "max_resolution": (224, 224),  # 输入图像的最大分辨率
            "num_bands": 64,  # 傅里叶变换中的频带数量
            "sine_only": False,  # 是否只使用正弦函数作为位置编码的基础
        }
        # 可训练位置编码解码器的参数
        trainable_position_encoding_kwargs_decoder = {
            "num_channels": config.d_latents,  # 潜在表示的通道数
            "index_dims": 1,  # 位置索引的维度
        }

        self.num_labels = config.num_labels
        # 创建Perceiver模型，指定输入预处理器和分类解码器
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="pixels",  # 使用像素级别的预处理方式
                spatial_downsample=1,  # 空间下采样因子
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,  # 使用查询残差连接
            ),
        )

        # 初始化权重并应用最终处理
        self.post_init()

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

    This model uses a 2D conv+maxpool preprocessing network. As shown in the paper, this model can achieve a top-1 accuracy
    of 82.1 on ImageNet.

    [`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]
    (with `prep_type="conv"`) to preprocess the input images, and
    [`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of
    [`PerceiverModel`] into classification logits.
    """,
    PERCEIVER_START_DOCSTRING,
)
class PerceiverForImageClassificationConvProcessing(PerceiverPreTrainedModel):



"""
注释：
- `PerceiverForImageClassificationLearned` 类使用 `PerceiverImagePreprocessor` 来预处理输入图像（使用 `prep_type="pixels"`），并使用 `PerceiverClassificationDecoder` 来将 `PerceiverModel` 的潜在表示解码为分类 logits。
- `PerceiverForImageClassificationConvProcessing` 类示例用于图像分类任务（例如 ImageNet），使用 2D 卷积+最大池化预处理网络，可以在 ImageNet 上达到82.1%的top-1准确率。
"""
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 定义用于预处理的傅里叶位置编码的参数字典
        fourier_position_encoding_kwargs_preprocessor = {
            "concat_pos": True,  # 是否在输入中连接位置编码
            "max_resolution": (56, 56),  # 最大分辨率
            "num_bands": 64,  # 傅里叶变换中使用的波段数
            "sine_only": False,  # 是否只使用正弦函数
        }
        
        # 定义用于解码器的可训练位置编码的参数字典
        trainable_position_encoding_kwargs_decoder = {"num_channels": config.d_latents, "index_dims": 1}

        # 设置实例变量：标签的数量
        self.num_labels = config.num_labels
        
        # 初始化感知器模型，配置输入预处理器和解码器
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv",  # 预处理类型为卷积
                spatial_downsample=1,  # 空间下采样因子
                position_encoding_type="fourier",  # 位置编码类型为傅里叶
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,  # 傅里叶位置编码参数
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,  # 解码器的通道数
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,  # 可训练位置编码参数
                use_query_residual=True,  # 是否使用查询残差
            ),
        )

        # 调用初始化后的处理函数
        self.post_init()

    # 重写的前向传播函数，接受多种输入参数并返回模型输出
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
class PerceiverForOpticalFlow(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        fourier_position_encoding_kwargs_preprocessor = {
            "num_bands": 64,
            "max_resolution": config.train_size,
            "sine_only": False,
            "concat_pos": True,
        }
        fourier_position_encoding_kwargs_decoder = {
            "concat_pos": True,
            "max_resolution": config.train_size,
            "num_bands": 64,
            "sine_only": False,
        }

        # Initialize the image preprocessor for the Perceiver model
        image_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="patches",
            spatial_downsample=1,
            conv_after_patching=True,
            conv_after_patching_in_channels=54,
            temporal_downsample=2,
            position_encoding_type="fourier",
            # Set Fourier position encoding parameters for preprocessor
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
        )

        # Initialize the Perceiver model with image preprocessor and optical flow decoder
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=image_preprocessor,
            decoder=PerceiverOpticalFlowDecoder(
                config,
                num_channels=image_preprocessor.num_channels,
                output_image_shape=config.train_size,
                rescale_factor=100.0,
                # Set decoder parameters including position encoding
                use_query_residual=False,
                output_num_channels=2,
                # Specify using Fourier position encoding for decoder
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
            ),
        )

        # Initialize weights and perform post-initialization steps
        self.post_init()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,  # 输入张量，用于模型的前向传播
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，用于控制模型关注的位置
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，用于屏蔽特定的注意力头部
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        labels: Optional[torch.Tensor] = None,  # 目标标签张量，用于光流损失的计算
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出结果
    ) -> Union[Tuple, PerceiverClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the optical flow loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:
            根据参数设置返回不同形式的输出结果。

        Examples:
            代码示例，展示了如何使用Perceiver模型处理光流问题。

        ```
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
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 根据配置确定是否使用字典形式的返回结果

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 调用Perceiver模型进行前向传播，获取模型输出

        logits = outputs.logits if return_dict else outputs[0]  # 根据是否返回字典形式选择相应的logits输出方式

        loss = None  # 初始化损失为None
        if labels is not None:
            raise NotImplementedError("Optical flow training is not yet supported")  # 如果标签不为空，抛出未实现错误，暂不支持光流训练

        if not return_dict:
            output = (logits,) + outputs[2:]  # 如果不返回字典形式，组合输出为(logits, hidden_states, attentions, cross_attentions)
            return ((loss,) + output) if loss is not None else output  # 返回输出结果，包括损失信息

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )  # 返回以PerceiverClassifierOutput形式封装的输出结果
# 导入所需模块和函数
@add_start_docstrings(
    """
    Perceiver 用于多模态（视频）自编码的示例用法，例如 Kinetics-700 数据集。

    [`PerceiverForMultimodalAutoencoding`] 使用 [`~models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor`] 来
    预处理三种模态：图像、音频和类标签。这个预处理器使用模态特定的预处理器来单独处理每种模态，然后将它们连接起来。使用可训练的位置编码来
    将每种模态填充到相同数量的通道，以便在时间维度上进行串联。接下来，应用 Perceiver 编码器。

    [`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] 用于解码 [`PerceiverModel`] 的潜在表示。
    这个解码器使用每种模态特定的解码器来构建查询。解码器的查询基于预处理后的输入。然而，在单个前向传递中自动编码整个视频在计算上是不可行的，
    因此只使用部分解码器查询与潜在表示进行交叉注意力。这由每种模态的子采样索引决定，可以作为额外输入提供给 [`PerceiverForMultimodalAutoencoding`] 的前向传递。

    [`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] 还将不同模态的解码器查询填充到相同数量的通道，以便在时间维度上进行串联。接下来，使用 [`PerceiverModel`] 的潜在表示进行交叉注意力。

    最后，[`~models.perceiver.modeling_perceiver.PerceiverMultiModalPostprocessor`] 用于将这个张量转换成实际的视频。
    它首先将输出分割成不同的模态，然后为每种模态应用相应的后处理器。

    请注意，在评估过程中通过掩盖分类标签（即简单地为"label"模态提供零张量）时，这个自编码模型变成了 Kinetics 700 视频分类器。
    """,
    PERCEIVER_START_DOCSTRING,
)
# 使用 PerceiverPreTrainedModel 作为基类定义 PerceiverForMultimodalAutoencoding 类
class PerceiverForMultimodalAutoencoding(PerceiverPreTrainedModel):
    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 重写前向传递函数 forward
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
    ):
        # 下面是位置编码
    # 设置一个参数变量用来传递傅立叶位置编码的参数，默认为None
# 构建位置编码器的函数，根据指定的参数生成不同类型的位置编码

def build_position_encoding(
    out_channels,  # 输出通道数，表示位置编码的通道数目
    project_pos_dim=None,  # 如果指定，将位置编码投影到这个维度
    position_encoding_type="trainable",  # 位置编码的类型，默认为可训练的位置编码
    trainable_position_encoding_kwargs=None,  # 可训练位置编码的额外参数
    fourier_position_encoding_kwargs=None  # 傅立叶位置编码的额外参数
):
    """
    Builds the position encoding.

    Args:
    - out_channels: refers to the number of channels of the position encodings.
    - project_pos_dim: if specified, will project the position encodings to this dimension.
    - position_encoding_type: specifies the type of position encoding to use.
    - trainable_position_encoding_kwargs: additional kwargs for trainable position encoding.
    - fourier_position_encoding_kwargs: additional kwargs for Fourier position encoding.

    Returns:
    - output_pos_enc: the constructed position encoding object.
    - positions_projection: optional projection layer for position encoding.
    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = PerceiverTrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = PerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    # Optionally, project the position encoding to a target dimension:
    positions_projection = nn.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()

    return output_pos_enc, positions_projection


# Below: Perceiver decoders


class PerceiverAbstractDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver abstract decoder."""

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
    Baseline projection decoder (no cross-attention).

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
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
    Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
    cross-attention operation, in which the latents produce keys and values.

    The shape of the output of this class depends on how one defines the output queries (also called decoder queries).
    """
    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_num_channels (`int`, *optional*):
            The number of channels in the output. Will only be used in case *final_project* is set to `True`.
        position_encoding_type (`str`, *optional*, defaults to "trainable"):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
        output_index_dims (`int`, *optional*):
            The number of dimensions of the output queries. Ignored if 'position_encoding_type' == 'none'.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels of the decoder queries. Ignored if 'position_encoding_type' == 'none'.
        subsampled_index_dims (`int`, *optional*):
            The number of dimensions of the subsampled indices. Ignored if 'position_encoding_type' == 'none'.
        qk_channels (`int`, *optional*):
            The number of channels of the queries and keys in the cross-attention layer.
        v_channels (`int`, *optional*):
            The number of channels of the values in the cross-attention layer.
        num_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the cross-attention layer.
        widening_factor (`int`, *optional*, defaults to 1):
            The widening factor of the cross-attention layer.
        use_query_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a residual connection between the query and the output of the cross-attention layer.
        concat_preprocessed_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the preprocessed input to the query.
        final_project (`bool`, *optional*, defaults to `True`):
            Whether to project the output of the cross-attention layer to a target dimension.
        position_encoding_only (`bool`, *optional*, defaults to `False`):
            Whether to only use this class to define output queries.
    ) -> None:
        super().__init__()
        
        self.output_num_channels = output_num_channels
        # 如果为 `none`，则解码器不会构建任何位置编码。
        # 当查询解码器时，您应该自行构建位置编码。
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != "none":
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )
        
        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only
        
        # 对于多模态自编码，我们不需要解码器的交叉注意力和最终层
        # 因此，将 position_encoding_only 设置为 True
        if not self.position_encoding_only:
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
            self.final_layer = nn.Linear(num_channels, output_num_channels) if final_project else nn.Identity()

    @property
    def num_query_channels(self) -> int:
        if self.position_encoding_type == "none":  # 查询来自其他地方
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        if self.position_encoding_only:
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            return self.output_position_encodings.output_size()
        if self.final_project:
            return self.output_num_channels
        return self.num_channels
    # 定义一个方法用于解码查询，接受多个输入参数
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        # 如果位置编码类型为"none"，则抛出数值错误，不允许构建解码查询
        if self.position_encoding_type == "none":
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
        
        # 如果给定了子采样点（subsampled_points）
        if subsampled_points is not None:
            # subsampled_points 是输入在扁平化后的索引，使用unravel_index获取非扁平化后的数组索引
            indices = [torch.from_numpy(x) for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)]
            # 将索引堆叠成 [n, d] 的坐标张量
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # 将这些坐标映射到 [-1, 1] 的范围
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            # 广播位置张量，使其与输入数据形状相匹配
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            
            # 构建位置编码
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, dtype=inputs.dtype, pos=pos
                )

            # 可选地将位置编码投影到目标维度
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            # 如果没有提供子采样点，获取输入的批次大小和索引维度
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # 构建位置编码
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    index_dims, batch_size, device=inputs.device, dtype=inputs.dtype
                )

            # 可选地将位置编码投影到目标维度
            pos_emb = self.positions_projection(pos_emb)

        # 如果设置了 concat_preprocessed_input 标志，则将预处理的输入与位置编码连接起来
        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)

        # 返回位置编码张量作为方法的输出
        return pos_emb
    ) -> PerceiverDecoderOutput:
        # 定义函数签名，指定返回类型为 PerceiverDecoderOutput

        # 执行交叉注意力解码。
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        # 如果不需要输出注意力权重，则将 cross_attentions 设置为 None
        cross_attentions = () if output_attentions else None

        # 调用解码器的交叉注意力层
        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
            output_attentions=output_attentions,
        )
        # 获取解码器层输出的第一个元素，即解码器的输出
        output = layer_outputs[0]

        # 如果需要输出注意力权重，将当前层的注意力权重添加到 cross_attentions 中
        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        # 将解码器的输出传入最终的输出层，得到最终的 logits
        logits = self.final_layer(output)

        # 返回 PerceiverDecoderOutput 对象，包含 logits 和可能的 cross_attentions
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

        self.num_labels = config.num_labels  # 设置分类标签的数量
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=self.num_labels,  # 输出通道数设置为分类标签的数量
            output_index_dims=1,  # 预测单一logit数组
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels  # 返回解码器的查询通道数量

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points  # 返回解码器的查询结果
        )

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> PerceiverDecoderOutput:
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        # B x 1 x num_classes -> B x num_classes
        logits = decoder_outputs.logits[:, 0, :]  # 从解码器输出中提取logits

        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)  # 返回解码器的输出结果


class PerceiverOpticalFlowDecoder(PerceiverAbstractDecoder):
    """Cross-attention based optical flow decoder."""

    def __init__(self, config, output_image_shape, output_num_channels=2, rescale_factor=100.0, **decoder_kwargs):
        super().__init__()

        self.output_image_shape = output_image_shape  # 设置输出图像的形状
        self.output_num_channels = output_num_channels  # 设置输出图像的通道数
        self.rescale_factor = rescale_factor  # 设置光流的重新缩放因子
        self.decoder = PerceiverBasicDecoder(config, output_num_channels=output_num_channels, **decoder_kwargs)

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels  # 返回解码器的查询通道数量

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")  # 如果有子采样点，则引发错误
        return inputs  # 返回输入数据，用于光流解码器的查询

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> PerceiverDecoderOutput:
        # 此处应有更多代码，但已截断
        pass  # 占位符，实际应该返回光流解码器的输出结果
    ) -> PerceiverDecoderOutput:
        # 调用解码器生成输出，传入查询向量 query 和编码器输出 z，选择是否返回注意力权重
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        # 从解码器输出中提取预测的 logits
        preds = decoder_outputs.logits
        # 对预测结果进行缩放，使用预定义的缩放因子 self.rescale_factor
        preds /= self.rescale_factor
        # 调整预测结果的形状为 [batch_size, output_height, output_width, num_classes]
        preds = preds.reshape([preds.shape[0]] + list(self.output_image_shape) + [preds.shape[-1]])
        # 返回经过解码器处理后的输出，包括 logits 和可能的交叉注意力权重
        return PerceiverDecoderOutput(logits=preds, cross_attentions=decoder_outputs.cross_attentions)
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

    def __init__(
        self, config: PerceiverConfig, output_shape: List[int], position_encoding_type: str, **decoder_kwargs
    ) -> None:
        super().__init__()
        # Validate the shape of output_shape to ensure it's rank 4 (batch_size, num_frames, height, width)
        if len(output_shape) != 4:  # B, T, H, W
            raise ValueError(f"Expected rank 4 output_shape, got {output_shape}.")
        # Initialize the decoder components:
        self.output_shape = output_shape
        self.output_num_channels = decoder_kwargs["output_num_channels"]

        # Create an instance of PerceiverBasicDecoder tailored for video decoding:
        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=self.output_shape[1:4],  # T*H*W
            position_encoding_type=position_encoding_type,
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        # Return the number of query channels from the decoder:
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        # Delegate the decoder_query method to the underlying PerceiverBasicDecoder instance:
        return self.decoder.decoder_query(
            inputs,
            modality_sizes=modality_sizes,
            inputs_without_pos=inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    def forward(
        self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor] = None
    ) -> PerceiverDecoderOutput:
        # Forward pass through the decoder:
        decoder_outputs = self.decoder(query, z)
        logits = decoder_outputs.logits

        # Reshape logits to match the specified output shape:
        logits = torch.reshape(logits, self.output_shape + [logits.shape[-1]])
        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)


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
    # Apply a predictable ordering to the modalities by iterating over sorted keys
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        # Slice the input tensor to extract the portion corresponding to the current modality
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


class PerceiverMultimodalDecoder(PerceiverAbstractDecoder):
    """
    Placeholder class for a multimodal decoder based on the Perceiver architecture.
    """
    """
    Multimodal decoding by composing uni-modal decoders. The *modalities* argument of the constructor is a dictionary
    mapping modality name to the decoder of that modality. That decoder will be used to construct queries for that
    modality. Modality-specific queries are padded with trainable modality-specific parameters, after which they are
    concatenated along the time dimension.

    Next, there is a shared cross attention operation across all modalities.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        modalities (`Dict[str, PerceiverAbstractDecoder]`):
            Dictionary mapping modality name to the decoder of that modality.
        num_outputs (`int`):
            The number of outputs of the decoder.
        output_num_channels (`int`):
            The number of channels in the output.
        min_padding_size (`int`, *optional*, defaults to 2):
            The minimum padding size for all modalities. The final output will have num_channels equal to the maximum
            channels across all modalities plus min_padding_size.
        subsampled_index_dims (`Dict[str, PerceiverAbstractDecoder]`, *optional*):
            Dictionary mapping modality name to the subsampled index dimensions to use for the decoder query of that
            modality.
    """

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
        """
        Constructor method for the MultimodalPerceiverDecoder class.

        Args:
            config (PerceiverConfig): Model configuration.
            modalities (Dict[str, PerceiverAbstractDecoder]): Dictionary mapping modality name to the decoder.
            num_outputs (int): The number of outputs of the decoder.
            output_num_channels (int): The number of channels in the output.
            min_padding_size (int, optional): The minimum padding size for all modalities.
            subsampled_index_dims (Dict[str, PerceiverAbstractDecoder], optional): Dictionary mapping modality name to
                subsampled index dimensions for the decoder query.
            **decoder_kwargs: Additional keyword arguments for the decoder.
        """
        super().__init__()
        # Initialize the modalities as a ModuleDict
        self.modalities = nn.ModuleDict(modalities)
        # Store the subsampled index dimensions
        self.subsampled_index_dims = subsampled_index_dims
        # Store the minimum padding size
        self.min_padding_size = min_padding_size
        # Store the number of output channels
        self.output_num_channels = output_num_channels
        # Store the number of outputs
        self.num_outputs = num_outputs
        # Initialize the decoder with given configuration and arguments
        self.decoder = PerceiverBasicDecoder(
            config,
            output_index_dims=(num_outputs,),
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            num_channels=self.num_query_channels,
            **decoder_kwargs,
        )
        # Initialize padding parameters for each modality
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_query_channels - decoder.num_query_channels))
                for modality, decoder in modalities.items()
            }
        )

    @property
    def num_query_channels(self) -> int:
        """
        Calculate the number of query channels based on the modalities.

        Returns:
            int: Number of query channels.
        """
        # Determine the maximum number of query channels among modalities
        max_channel_size = max(decoder.num_query_channels for _, decoder in self.modalities.items())
        # Ensure common channel size includes minimum padding size
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size
    def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None):
        # 将扁平化的输入数据按照不同的感知模态进行分割重组
        inputs = restructure(modality_sizes, inputs)

        # 获取各个感知模态的解码器查询
        subsampled_points = subsampled_points or {}

        # 存储每个模态的解码器查询结果
        decoder_queries = {}
        for modality, decoder in self.modalities.items():
            # 如果存在输入数据不包含位置信息，则获取当前模态的无位置信息的输入数据
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            # 调用解码器的查询函数，获取当前模态的查询结果
            query = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,  # 此处未使用 modality_sizes 参数，可能为函数签名未更新的遗留
                inputs_without_pos=input_without_pos,
                subsampled_points=subsampled_points.get(modality, None),
            )
            decoder_queries[modality] = query

        # 使用可训练的位置编码填充所有查询结果，以保证它们具有相同的通道数

        def embed(modality, x):
            # 将输入张量 x 重塑为 [batch_size, 总特征数, 通道数] 的形状
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            # 获取当前模态的填充位置编码
            pos = self.padding[modality]
            # 将位置编码广播到与 x 相同的形状
            pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
            # 在通道维度上连接 x 和位置编码
            return torch.cat([x, pos], dim=2)

        # 对模态按照可预测的顺序进行排序，并连接它们的查询结果
        return torch.cat(
            [embed(modality, decoder_queries[modality]) for modality in sorted(self.modalities.keys())], dim=1
        )

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        # B x 1 x num_classes -> B x num_classes
        # 调用解码器模块进行前向传播，生成解码器的输出结果
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        return decoder_outputs
# Below: IO pre- and post-processor classes for Perceiver.

# 定义一个函数，实现空间到深度的转换，用于重新排列空间数据块到深度
def space_to_depth(frames: torch.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1) -> torch.Tensor:
    """
    Space to depth transform. Rearranges blocks of spatial data, into depth.

    This function assumes the channels to be first, but will place the channels last after transformation.

    Based on https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15.
    """
    # 检查输入张量的维度是否为4
    if len(frames.shape) == 4:
        batch_size, num_channels, height, width = frames.shape
        # 将空间数据块按照指定的空间块大小进行分割
        frames = frames.view(
            batch_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # 将分割后的块移动到最后一个维度：(batch_size, H//bs, W//bs, bs, bs, C)
        frames = frames.permute(0, 2, 4, 3, 5, 1).contiguous()
        # 沿着通道维度连接块：(batch_size, H//bs, W//bs, bs*bs*C)
        frames = frames.view(
            batch_size,
            height // spatial_block_size,
            width // spatial_block_size,
            (spatial_block_size**2) * num_channels,
        )
        return frames
    # 检查输入张量的维度是否为5
    elif len(frames.shape) == 5:
        batch_size, time, num_channels, height, width = frames.shape
        # 将时间维度和空间维度按照指定的块大小进行分割
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
        # 将分割后的块移动到最后一个维度：(batch_size, T//ts, H//bs, W//bs, ts, bs, bs, C)
        frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        # 沿着通道维度连接块：(batch_size, T//ts, H//bs, W//bs, ts*bs*bs*C)
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            height // spatial_block_size,
            width // spatial_block_size,
            temporal_block_size * (spatial_block_size**2) * num_channels,
        )
        return frames
    else:
        # 抛出异常，如果输入张量的维度既不是4也不是5
        raise ValueError(
            "Frames should be of rank 4 (batch, channels, height, width)"
            " or rank 5 (batch, time, channels, height, width)"
        )


# 定义一个继承自 nn.Conv2d 的类，支持 padding="same"
class Conv2dSamePadding(nn.Conv2d):
    """
    Conv2d layer with padding="same" support. Source:
    https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
    """
    # 初始化方法，继承父类 Conv2dSamePadding
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        # 创建 ZeroPad2d 层，用于实现“same” padding
        self.zero_pad_2d = nn.ZeroPad2d(
            # 计算每个维度的 padding 数量，使得卷积操作后大小不变
            reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        )

    # 前向传播方法
    def forward(self, input):
        # 对输入进行 zero padding，保证卷积输出大小与输入相同
        padded_input = self.zero_pad_2d(input)
        # 执行卷积操作，使用权重 self.weight 和偏置 self.bias
        return self._conv_forward(padded_input, self.weight, self.bias)
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

        # Define a 2D convolution layer with same padding
        self.conv = Conv2dSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, bias=False
        )
        
        # Batch normalization layer if `use_batchnorm` is True, otherwise an identity layer
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels) if use_batchnorm else nn.Identity()
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Max pooling layer with kernel size 3 and stride 2
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Forward pass through the layers
        out = self.conv(inputs)  # Apply convolution
        out = self.batchnorm(out)  # Apply batch normalization or identity
        out = self.relu(out)  # Apply ReLU activation
        out = self.max_pool(out)  # Apply max pooling
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
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # (This part of the function calculates frequency bands based on the given maximum resolution and number of bands)
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    # Reshape per_pos_features into a flattened shape
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        # Apply sine transformation to per_pos_features
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        # Apply both sine and cosine transformations to per_pos_features
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        # Concatenate pos and per_pos_features along the last dimension
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    
    # Return the final per_pos_features tensor
    return per_pos_features
# 生成一个线性位置索引数组，用于 N 维输入数组。

def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """
    Generate an array of position indices for an N-D input array.

    Args:
      index_dims (`List[int]`):
        The shape of the index dimensions of the input array.
      output_range (`Tuple[float]`, *optional*, defaults to `(-1.0, 1.0)`):
        The min and max values taken by each input index dimension.

    Returns:
      `torch.FloatTensor` of shape `(index_dims[0], index_dims[1], .., index_dims[-1], N)`.
    """

    def _linspace(n_xels_per_dim):
        # 使用 torch.linspace 生成指定范围和步长的一维张量
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    # 生成每个维度的线性分布的张量数组
    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    # 使用 meshgrid 函数创建多维网格，表示每个位置的坐标
    array_index_grid = meshgrid(*dim_ranges, indexing="ij")

    return torch.stack(array_index_grid, dim=-1)


class PerceiverAbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver abstract position encoding."""

    @property
    @abc.abstractmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch_size, pos):
        raise NotImplementedError


class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        # 创建一个形状为 (index_dim, num_channels) 的可训练的位置嵌入参数
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channels))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        # 返回位置编码器的输出大小，即 num_channels
        return self._num_channels

    def forward(self, batch_size: int) -> torch.Tensor:
        position_embeddings = self.position_embeddings

        if batch_size is not None:
            # 如果指定了批量大小，扩展位置嵌入参数的第一维度为 batch_size
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        return position_embeddings


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    Checks or builds spatial position features (x, y, ...).

    Args:
      pos (`torch.FloatTensor`):
        None, or an array of position features. If None, position features are built. Otherwise, their size is checked.
      index_dims (`List[int]`):
        An iterable giving the spatial/index size of the data to be featurized.
      batch_size (`int`):
        The batch size of the data to be featurized.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, prod(index_dims))` an array of position features.
    """
    # 如果 pos 参数为 None，则根据 index_dims 构建线性位置信息
    if pos is None:
        pos = build_linear_positions(index_dims)
        # 相当于 `torch.broadcast_to(pos[None], (batch_size,) + pos.shape)`
        # 但是 `torch.broadcast_to` 不能转换为 ONNX 格式
        pos = pos[None].expand((batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        # 警告：你可能不希望你的空间特征与 pos 坐标系的空间布局不同。
        # 如果你认为可以，请随意覆盖这一段代码！
        
        # 检查 pos 的最后一个维度是否与 index_dims 的长度相同
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    # 返回 pos 变量，其中包含位置信息
    return pos
class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands  # 设置频带数量
        self.max_resolution = max_resolution  # 设置最大分辨率
        self.concat_pos = concat_pos  # 是否连接位置编码
        self.sine_only = sine_only  # 是否只使用正弦编码

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)  # 返回最大分辨率的维度数

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)  # 获取最大分辨率的维度数
        encoding_size = self.num_bands * num_dims  # 计算编码的大小
        if not self.sine_only:
            encoding_size *= 2  # 如果不仅使用正弦编码，则大小加倍
        if self.concat_pos:
            encoding_size += self.num_dimensions  # 如果连接位置编码，则增加维度数

        return encoding_size  # 返回编码的最后一个维度大小

    def forward(
        self,
        index_dims: List[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        pos: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)  # 检查或构建空间位置
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device=device, dtype=dtype)  # 生成傅里叶特征编码，并将其转移到指定设备和数据类型
        return fourier_pos_enc


class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
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
        self.config = config  # 设置模型配置
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)  # 创建词嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)  # 创建位置编码层

    @property
    def num_channels(self) -> int:
        return self.config.d_model  # 返回模型配置中的 d_model 大小

    def forward(self, inputs: torch.LongTensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        embeddings_without_pos = self.embeddings(inputs)  # 获取不包含位置编码的词嵌入

        seq_length = inputs.shape[1]  # 获取序列长度
        position_ids = torch.arange(0, seq_length, device=inputs.device)  # 在指定设备上创建位置索引
        embeddings = embeddings_without_pos + self.position_embeddings(position_ids)  # 添加位置编码到词嵌入

        return embeddings, None, embeddings_without_pos


class PerceiverEmbeddingDecoder(nn.Module):
    """
    Module to decode embeddings (for masked language modeling).
    """
    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """
    # 定义 Perceiver 模型类，继承自 nn.Module
    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        # 保存模型配置
        self.config = config
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 初始化偏置项，维度为词汇表大小，作为可学习参数
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(self, hidden_states: torch.Tensor, embedding_layer: torch.Tensor) -> torch.Tensor:
        # 获取输入的张量维度信息
        batch_size, seq_len, d_model = hidden_states.shape
        # 将隐藏状态张量展平（flatten）为二维张量，进行矩阵乘法
        output = torch.matmul(hidden_states.reshape([-1, d_model]), embedding_layer.weight.transpose(0, 1))
        # 添加偏置项到输出张量
        output = output + self.bias
        # 将输出张量重新形状为原始的三维张量形状
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
        # 初始化时将各个模态的后处理器组成一个模块字典
        self.modalities = nn.ModuleDict(modalities)
        # 标记输入是否为字典形式
        self.input_is_dict = input_is_dict

    def forward(
        self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None
    ) -> Mapping[str, torch.Tensor]:
        if not self.input_is_dict:
            # 如果输入不是字典形式，根据模态大小重新组织输入数据
            if modality_sizes is None:
                raise ValueError("Modality sizes should be specified if input is not a dictionary.")
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)

        # 对每个模态使用对应的后处理器进行处理，并输出结果字典
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
        # 使用线性层将输入通道数映射为分类标签数
        self.classifier = nn.Linear(in_channels, config.num_labels)

    def forward(self, inputs, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        # 使用分类器线性层计算分类 logits
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
    # 使用给定的配置和输入通道数初始化模型
    def __init__(self, config: PerceiverConfig, in_channels: int, postproc_type: str = "patches") -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 检查后处理类型是否在支持的范围内，目前支持 'patches' 类型
        if postproc_type not in ("patches",):  # to be supported: 'conv', 'patches', 'pixels'
            # 如果不在支持的类型中，则抛出数值错误异常
            raise ValueError("Invalid postproc_type!")

        # 架构参数:
        # 创建一个线性分类器，输入通道数为 in_channels，输出通道数为 config.samples_per_patch
        self.classifier = nn.Linear(in_channels, config.samples_per_patch)

    # 前向传播函数，接收输入张量 inputs，可选的位置张量 pos 和模态大小 modality_sizes
    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        # 使用分类器进行前向计算，得到 logits
        logits = self.classifier(inputs)
        # 对 logits 进行形状变换，将其变为 [batch_size, -1] 的形状
        return torch.reshape(logits, [inputs.shape[0], -1])
# 定义了一个名为 PerceiverProjectionPostprocessor 的神经网络模块，用于处理 Perceiver 模型的投影后处理，
# 可以将解码器输出的通道投影到较低维度。

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
        # 使用线性层进行投影，将输入通道数投影到输出通道数
        self.classifier = nn.Linear(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, modality_sizes=None) -> torch.Tensor:
        # 将输入数据通过线性层进行投影
        logits = self.classifier(inputs)
        # 返回投影后的结果
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
        config,
        prep_type="conv",  # 预处理类型，默认为卷积
        spatial_downsample: int = 4,  # 空间下采样因子，默认为4
        temporal_downsample: int = 1,  # 时间下采样因子，默认为1
        position_encoding_type: str = "fourier",  # 位置编码类型，默认为傅里叶
        in_channels: int = 3,  # 输入通道数，默认为3
        out_channels: int = 64,  # 输出通道数，默认为64
        conv_after_patching: bool = False,  # 是否在打补丁后进行卷积，默认为False
        conv_after_patching_in_channels: int = 54,  # 仅在conv_after_patching为True时 relevant 的输入通道数
        conv2d_use_batchnorm: bool = True,  # 是否在卷积层后使用批量归一化，默认为True
        concat_or_add_pos: str = "concat",  # 位置编码添加方式，默认为拼接
        project_pos_dim: int = -1,  # 位置维度投影，默认为-1
        **position_encoding_kwargs,  # 其他位置编码的关键字参数
        ):
        # 调用父类的构造函数
        super().__init__()
        # 将配置参数保存到实例变量中
        self.config = config

        # 检查预处理类型是否合法
        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        # 检查拼接或添加位置的选项是否合法
        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        # 初始化实例变量
        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        # 如果预处理类型为 "conv"
        if self.prep_type == "conv":
            # 使用对数函数计算需要的卷积层数，要求空间下采样为4的幂次方
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            # 检查空间和时间下采样是否符合要求
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial and 1 expected for temporal downsampling with conv."
                )
            # 创建卷积下采样网络
            self.convnet = Conv2DDownsample(
                in_channels=in_channels,
                num_layers=int(convnet_num_layers),
                out_channels=out_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )

        # 如果预处理类型为 "conv1x1"
        elif self.prep_type == "conv1x1":
            # 对于 conv1x1，只允许空间下采样，不允许时间下采样
            if temporal_downsample != 1:
                raise ValueError("Conv1x1 does not downsample in time.")
            # 创建 1x1 卷积层
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(spatial_downsample, spatial_downsample),  # 空间下采样步幅设置
            )

        # 构建位置编码
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

        # 可选的卷积层，用于在提取补丁之后进行处理
        self.conv_after_patches = (
            nn.Linear(conv_after_patching_in_channels, self.out_channels) if conv_after_patching else nn.Identity()
        )
    def num_channels(self) -> int:
        # 假设输入数据的分辨率在图像预处理的上下文中是2或3，
        # 取决于我们是处理图像还是视频。为了方便起见，
        # 我们定义一个 is_temporal 变量，用于表示数据是否具有时间维度。
        is_temporal = self.position_embeddings.num_dimensions > 2

        # 位置嵌入
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()

        # 如果使用“add”模式连接位置编码，则返回位置维度
        if self.concat_or_add_pos == "add":
            return pos_dim

        # 输入维度
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

        # 返回输入维度加上位置维度的结果
        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs: torch.Tensor, network_input_is_1d: bool = True):
        """
        构建最终输入，包括位置编码。

        该方法假设输入始终将通道作为最后一个维度。

        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # 如果输入维度大于3且网络输入是1维，则将输入特征展平为1维索引维度。
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # 构建位置编码
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)

        # 可选择将位置编码投影到目标维度。
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # 如果网络接受非1维输入，则重新整形位置编码以匹配输入特征形状。
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])

        # 根据连接或加法模式将位置编码与输入合并或相加，并返回结果。
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs
    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # 根据 self.prep_type 的不同进行不同的数据预处理
        if self.prep_type == "conv":
            # 如果预处理类型为 "conv"，则使用卷积神经网络进行图像特征提取
            # 空间下采样因子为4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # 如果预处理类型为 "conv1x1"，则将输入映射到 self.out_channels 维度
            inputs = self.convnet_1x1(inputs)

        elif self.prep_type == "pixels":
            # 如果预处理类型为 "pixels"，根据输入的维度进行最简单的下采样处理
            if inputs.ndim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :, :: self.temporal_downsample, :, :: self.spatial_downsample, :: self.spatial_downsample
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        elif self.prep_type == "patches":
            # 如果预处理类型为 "patches"，进行 Space2depth 特征化处理
            # 视频数据格式为 B x T x C x H x W
            inputs = space_to_depth(
                inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample
            )

            # 如果数据维度为5且第二个维度为1，则为光流数据，进行压缩处理
            if inputs.ndim == 5 and inputs.shape[1] == 1:
                inputs = inputs.squeeze(dim=1)

            # 可选择应用卷积层
            inputs = self.conv_after_patches(inputs)

        if self.prep_type != "patches":
            # 将通道移动到最后一个维度，因为下面的 _build_network_inputs 方法需要这种格式
            if inputs.ndim == 4:
                inputs = inputs.permute(0, 2, 3, 1)
            elif inputs.ndim == 5:
                inputs = inputs.permute(0, 1, 3, 4, 2)
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        # 调用 _build_network_inputs 方法构建网络输入
        inputs, inputs_without_pos = self._build_network_inputs(inputs, network_input_is_1d)
        modality_sizes = None  # 每种模态的大小，仅在多模态情况下需要

        return inputs, modality_sizes, inputs_without_pos
# 定义一个用于Perceiver Encoder的One-hot预处理器，用于将一个虚拟的索引维度添加到输入中。
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
        # 返回配置中定义的标签数，作为通道数
        return self.config.num_labels

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # 添加一个虚拟的索引维度到输入张量中
        inputs = inputs[:, None, :]

        # 由于没有位置编码，因此第一个（输入）和第三个（没有位置编码的输入）输出是相同的
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
    ):
        super().__init__()
        self.config = config

        # 检查预处理类型是否合法，只能是 "patches"
        if prep_type not in ("patches",):
            raise ValueError(f"Prep_type {prep_type} is invalid, can only be 'patches'.")

        # 检查连接或添加位置编码的方式是否合法，只能是 "concat" 或 "add"
        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'.")

        # 设置样本每个补丁的数量
        self.samples_per_patch = samples_per_patch
        # 设置位置编码类型
        self.position_encoding_type = position_encoding_type
        # 设置连接或添加位置编码的方式
        self.concat_or_add_pos = concat_or_add_pos
        # 设置位置编码的投影维度
        self.project_pos_dim = project_pos_dim

        # 构建位置编码和位置投影
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

    @property
    def num_channels(self) -> int:
        # 位置编码维度
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        # 根据连接或添加位置编码的方式确定通道数
        if self.concat_or_add_pos == "add":
            return pos_dim
        return self.samples_per_patch + pos_dim

    def _build_network_inputs(self, inputs):
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # 构建位置编码
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)

        # 可选择性地将位置编码投影到目标维度
        pos_enc = self.positions_projection(pos_enc)

        # 根据连接或添加位置编码的方式，合并输入数据和位置编码
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs  # 返回带位置编码和原始输入的数据

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        inputs = torch.reshape(inputs, [inputs.shape[0], -1, self.samples_per_patch])

        # 构建网络的输入，包括位置编码
        inputs, inputs_without_pos = self._build_network_inputs(inputs)
        modality_sizes = None  # 用于多模态的每个模态的大小

        return inputs, modality_sizes, inputs_without_pos
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
        super().__init__()
        # 使用 nn.ModuleDict 封装各个模态的预处理器
        self.modalities = nn.ModuleDict(modalities)
        # 设置最小填充大小
        self.min_padding_size = min_padding_size
        # 如果提供了遮罩概率字典，则使用该字典；否则为空字典
        self.mask_probs = mask_probs if mask_probs is not None else {}
        # 初始化填充参数，为每个模态创建一个可训练的位置填充向量
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_channels - preprocessor.num_channels))
                for modality, preprocessor in modalities.items()
            }
        )
        # 初始化遮罩参数，为每个模态创建一个可训练的遮罩向量
        self.mask = nn.ParameterDict(
            {modality: nn.Parameter(torch.randn(1, self.num_channels)) for modality, _ in self.mask_probs.items()}
        )

    @property
    def num_channels(self) -> int:
        # 计算所有模态中最大通道数，并加上最小填充大小，得到公共通道数
        max_channel_size = max(processor.num_channels for _, processor in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def forward(
        self, inputs: Mapping[str, torch.Tensor], pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True
    ):
        # 实现前向传播的方法，处理输入数据和位置信息
    ) -> PreprocessorOutputType:
        # 初始化空字典用于存储填充后的输出
        padded = {}
        # 初始化空字典用于存储每个模态的输出大小
        modality_sizes = {}
        # 初始化空字典用于存储没有位置编码的输入
        inputs_without_pos = {}

        # 遍历每个模态和其对应的预处理器
        for modality, preprocessor in self.modalities.items():
            # 使用对应的预处理器处理每个模态的输入
            # 获取预处理后的输出、位置编码和没有位置编码的输入
            output, _, inputs_without_pos[modality] = preprocessor(
                inputs[modality], pos=pos, network_input_is_1d=network_input_is_1d
            )

            # 对输出进行填充到相同的 common_channel_size
            batch_size, num_samples, num_channels = output.shape
            # 扩展位置编码以匹配输出的形状
            pos_enc = self.padding[modality].expand(batch_size, -1, -1)

            # 使用广播方式创建填充张量，使其与输出的通道数匹配
            padding = torch.broadcast_to(
                pos_enc,
                [batch_size, num_samples, self.num_channels - num_channels],
            )
            # 在通道维度上连接输出和填充部分
            output_padded = torch.cat([output, padding], dim=2)

            # 如果需要，进行掩码操作
            if modality in self.mask_probs:
                # 获取模态对应的掩码标记并扩展以匹配输出形状
                mask_token = self.mask[modality].expand(batch_size, -1, -1)
                mask_prob = self.mask_probs[modality]
                # 使用伯努利分布生成掩码
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2).to(mask_token.device)
                # 应用掩码到填充后的输出
                output_padded = (1 - mask) * output_padded + mask * mask_token

            # 将填充后的输出存储到对应的模态键下
            padded[modality] = output_padded
            # 记录每个模态填充后的输出大小
            modality_sizes[modality] = output_padded.shape[1]

        # 将填充后的输出按照模态键排序形成列表
        padded_ls = [padded[k] for k in sorted(padded.keys())]

        # 最终将所有模态的填充输出沿时间维度连接起来
        final_inputs = torch.cat(padded_ls, dim=1)

        # 返回最终的填充后的输入、每个模态的输出大小和没有位置编码的输入
        return final_inputs, modality_sizes, inputs_without_pos
```