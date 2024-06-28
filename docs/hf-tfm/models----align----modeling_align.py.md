# `.\models\align\modeling_align.py`

```py
# coding=utf-8
# Copyright 2023 The Google Research Team Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch ALIGN model."""

import math  # 导入数学函数库
from dataclasses import dataclass  # 导入用于创建数据类的装饰器
from typing import Any, Optional, Tuple, Union  # 导入类型提示相关库

import torch  # 导入PyTorch深度学习库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint模块
from torch import nn  # 导入神经网络模块

from ...activations import ACT2FN  # 导入激活函数相关定义
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,  # 导入无注意力机制的基础模型输出
    BaseModelOutputWithPastAndCrossAttentions,  # 导入带过去和交叉注意力的基础模型输出
    BaseModelOutputWithPoolingAndCrossAttentions,  # 导入带池化和交叉注意力的基础模型输出
    BaseModelOutputWithPoolingAndNoAttention,  # 导入带池化但无注意力机制的基础模型输出
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer  # 导入PyTorch工具函数
from ...utils import (
    ModelOutput,  # 导入模型输出定义
    add_start_docstrings,  # 导入添加文档字符串的函数
    add_start_docstrings_to_model_forward,  # 导入添加模型前向传播文档字符串的函数
    logging,  # 导入日志记录功能
    replace_return_docstrings,  # 导入替换返回文档字符串的函数
)
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig  # 导入配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "kakaobrain/align-base"  # 预训练模型的检查点名称
_CONFIG_FOR_DOC = "AlignConfig"  # 配置类的名称


ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kakaobrain/align-base",  # 预训练模型存档列表，包括基础模型
    # 查看所有ALIGN模型：https://huggingface.co/models?filter=align
]


ALIGN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`AlignConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ALIGN_TEXT_INPUTS_DOCSTRING = r"""
    Parameters:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.ALIGNTokenizer`.
            See :class:`~transformers.PreTrainedTokenizer` for more information.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            See :class:`~transformers.ALIGNTokenizer` for more information.
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            See :class:`~transformers.ALIGNTokenizer` for more information.
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Indices of positions of each input token in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            See :class:`~transformers.ALIGNTokenizer` for more information.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, optional):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert input tokens into embeddings before feeding them
            to the model.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, optional):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            See :func:`~transformers.modeling_utils.create_mask_from_input_mask` for more information.
        inputs_kwargs:
            Additional dictionary of keyword arguments for specific settings of the model encoder (usually for adding
            special features in model-specific encoders).

    Returns:
        :class:`~transformers.ModelOutput`: A dictionary (if the model has more than one output) or a single tensor
        (if the model has only one output) with the model outputs.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的词汇索引。默认情况下会忽略填充部分。
            # 可以使用 AutoTokenizer 获取索引。详见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__。

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于在填充的令牌索引上避免注意力操作。取值范围为 `[0, 1]`：

            - 1 表示**未遮罩**的令牌，
            - 0 表示**遮罩**的令牌。

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列令牌在位置嵌入中的位置索引。取值范围为 `[0, config.max_position_embeddings - 1]`。

            [What are position IDs?](../glossary#position-ids)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段令牌索引，用于指示输入的第一部分和第二部分。索引选取范围为 `[0, 1]`：

            - 0 对应 *句子 A* 的令牌，
            - 1 对应 *句子 B* 的令牌。

            [What are token type IDs?](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选定头部的掩码。取值范围为 `[0, 1]`：

            - 1 表示**未遮罩**的头部，
            - 0 表示**遮罩**的头部。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您希望更精确地控制如何将 `input_ids` 索引转换为相关联的向量，则这很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是一个普通的元组。
# ALIGN_VISION_INPUTS_DOCSTRING 是一个原始字符串，用于描述 AlignVisionModel 类中 align_vision 模块的输入参数和返回值
ALIGN_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`EfficientNetImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# ALIGN_INPUTS_DOCSTRING 是一个空字符串，可能用于将来扩展
ALIGN_INPUTS_DOCSTRING = r"""
"""


@dataclass
class AlignVisionModelOutput(ModelOutput):
    """
    AlignVisionModelOutput 是一个数据类，用于存储视觉模型输出，包含图像嵌入和最后一层隐藏状态的汇总。

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class AlignTextModelOutput(ModelOutput):
    """
    AlignTextModelOutput 是一个数据类，用于存储文本模型的输出，包含最后一层隐藏状态的汇总。

    Base class for text model's outputs that also contains a pooling of the last hidden states.
    """
    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # Optional: Text embeddings of shape (batch_size, output_dim) if projection layer is used.
    text_embeds: Optional[torch.FloatTensor] = None
    # Required: Hidden states of shape (batch_size, sequence_length, hidden_size) from the last model layer.
    last_hidden_state: torch.FloatTensor = None
    # Optional: Tuple of hidden states from all layers, including embeddings if present, of shape (batch_size, sequence_length, hidden_size).
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Optional: Tuple of attention weights for each layer of shape (batch_size, num_heads, sequence_length, sequence_length).
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class AlignOutput(ModelOutput):
    """
    AlignOutput 类，继承自 ModelOutput，用于保存对齐模型的输出结果。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            对比损失，用于衡量图像-文本的相似度。
        logits_per_image: (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            图像嵌入向量与文本嵌入向量之间的点积得分。表示图像-文本之间的相似度分数。
        logits_per_text: (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            文本嵌入向量与图像嵌入向量之间的点积得分。表示文本-图像之间的相似度分数。
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            通过投影层应用到 [`AlignTextModel`] 的汇总输出得到的文本嵌入向量。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            [`AlignVisionModel`] 的输出。
        text_model_output (`BaseModelOutputWithPoolingAndCrossAttentions`):
            [`AlignTextModel`] 的输出，包含池化和交叉注意力。
        vision_model_output (`BaseModelOutputWithPoolingAndNoAttention`):
            [`AlignVisionModel`] 的输出，包含池化但没有注意力。
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = None

    def to_tuple(self) -> Tuple[Any]:
        """
        将对象转换为元组形式，用于序列化。

        Returns:
            Tuple[Any]: 对象的元组表示。
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html 改编而来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    计算对比损失。

    Args:
        logits (torch.Tensor): 输入的 logits 张量。

    Returns:
        torch.Tensor: 计算得到的对比损失。
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device), label_smoothing=0.1)


def align_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    计算对齐损失，结合了对比损失函数的结果。

    Args:
        similarity (torch.Tensor): 图像-文本或文本-图像之间的相似度矩阵。

    Returns:
        torch.Tensor: 计算得到的对齐损失。
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


# 从 transformers.models.efficientnet.modeling_efficientnet.round_filters 复制而来，用于 AlignVision
def round_filters(config: AlignVisionConfig, num_channels: int):
    """
    根据深度乘数调整滤波器数量。

    Args:
        config (AlignVisionConfig): 包含配置信息的对象，如深度因子。
        num_channels (int): 当前的通道数量。

    Returns:
        int: 调整后的通道数量。
    """
    divisor = config.depth_divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # 确保向下舍入不会降低超过 10%。
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)
# Copied from transformers.models.efficientnet.modeling_efficientnet.correct_pad
# 定义一个函数，用于计算深度卷积的填充值
def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
    """
    Utility function to get the tuple padding value for the depthwise convolution.

    Args:
        kernel_size (`int` or `tuple`):
            Kernel size of the convolution layers.
        adjust (`bool`, *optional*, defaults to `True`):
            Adjusts padding value to apply to right and bottom sides of the input.
    """
    # 如果 `kernel_size` 是 `int` 类型，则转换为元组形式
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # 计算正确的填充值，使得深度卷积的输出尺寸与输入尺寸相同
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        # 如果需要调整填充值，则返回调整后的填充元组
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    else:
        # 否则返回未调整的填充元组
        return (correct[1], correct[1], correct[0], correct[0])


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetEmbeddings with EfficientNet->AlignVision
# 定义一个用于视觉对齐的嵌入模块，类似于原始实现中的干节点模块
class AlignVisionEmbeddings(nn.Module):
    """
    A module that corresponds to the stem module of the original work.
    """

    def __init__(self, config: AlignVisionConfig):
        super().__init__()

        # 计算输出维度，根据配置文件中的信息
        self.out_dim = round_filters(config, 32)
        # 添加零填充层，填充的方式为 (0, 1, 0, 1)
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        # 定义二维卷积层，用于提取特征
        self.convolution = nn.Conv2d(
            config.num_channels, self.out_dim, kernel_size=3, stride=2, padding="valid", bias=False
        )
        # 批归一化层，用于规范化数据分布
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        # 激活函数，根据配置文件中指定的激活函数类型选择
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 对输入的像素值进行填充
        features = self.padding(pixel_values)
        # 应用卷积操作
        features = self.convolution(features)
        # 应用批归一化
        features = self.batchnorm(features)
        # 应用激活函数
        features = self.activation(features)

        return features


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetDepthwiseConv2d with EfficientNet->AlignVision
# 定义一个深度卷积层，继承自 PyTorch 的二维卷积层
class AlignVisionDepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        # 计算输出通道数，根据输入通道数和深度倍增因子
        out_channels = in_channels * depth_multiplier
        # 调用父类的初始化方法，定义深度卷积层
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 使用组卷积，每个输入通道对应一个卷积核
            bias=bias,
            padding_mode=padding_mode,
        )


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetExpansionLayer with EfficientNet->AlignVision
# 定义一个扩展层模块，对应原始实现中每个块的扩展阶段
class AlignVisionExpansionLayer(nn.Module):
    """
    This corresponds to the expansion phase of each block in the original implementation.
    """
    # 初始化函数，用于创建一个卷积神经网络模块
    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        # 定义一个1x1的卷积层，用于扩展输入通道数到输出通道数
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",  # 设定填充方式为 "same"，即保持输入输出尺寸相同
            bias=False,  # 不使用偏置项
        )
        # 定义扩展后的批归一化层，对输出通道数进行归一化处理
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        # 根据配置选择激活函数，ACT2FN 是一个预定义的激活函数字典
        self.expand_act = ACT2FN[config.hidden_act]

    # 前向传播函数
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 执行扩展阶段的前向传播
        # 将输入的 hidden_states 通过扩展卷积层进行卷积操作
        hidden_states = self.expand_conv(hidden_states)
        # 对卷积结果进行扩展批归一化处理
        hidden_states = self.expand_bn(hidden_states)
        # 应用预定义的激活函数到批归一化后的结果
        hidden_states = self.expand_act(hidden_states)

        # 返回经过扩展阶段处理后的隐藏状态数据
        return hidden_states
# 从 EfficientNet 的模型定义中复制而来，用于实现 AlignVision 的深度卷积层
class AlignVisionDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ):
        super().__init__()
        self.stride = stride
        # 根据步长选择是否使用 'valid' 或 'same' 的填充方式
        conv_pad = "valid" if self.stride == 2 else "same"
        # 计算正确的填充大小
        padding = correct_pad(kernel_size, adjust=adjust_padding)

        # 深度卷积层的零填充
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        # 深度卷积层的定义
        self.depthwise_conv = AlignVisionDepthwiseConv2d(
            in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False
        )
        # 批归一化层
        self.depthwise_norm = nn.BatchNorm2d(
            num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 激活函数的选择
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 深度卷积操作
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


# 从 EfficientNet 的模型定义中复制而来，用于实现 AlignVision 的挤压激活层
class AlignVisionSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, expand_dim: int, expand: bool = False):
        super().__init__()
        # 根据是否扩展选择维度
        self.dim = expand_dim if expand else in_dim
        # 计算挤压激活层的输出维度
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))

        # 挤压阶段：全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        # 激活阶段：降维卷积
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        # 激活阶段：扩展卷积
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=1,
            padding="same",
        )
        # 降维卷积后的激活函数
        self.act_reduce = ACT2FN[config.hidden_act]
        # 扩展卷积后的激活函数
        self.act_expand = nn.Sigmoid()
    # 定义前向传播方法，接受隐藏状态作为输入并返回张量
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 将输入赋给局部变量inputs
        inputs = hidden_states
        # 使用squeeze方法对隐藏状态进行压缩操作
        hidden_states = self.squeeze(hidden_states)
        # 使用reduce方法对压缩后的隐藏状态进行进一步处理
        hidden_states = self.reduce(hidden_states)
        # 对进一步处理后的隐藏状态应用激活函数
        hidden_states = self.act_reduce(hidden_states)

        # 使用expand方法对处理后的隐藏状态进行扩展操作
        hidden_states = self.expand(hidden_states)
        # 对扩展后的隐藏状态应用激活函数
        hidden_states = self.act_expand(hidden_states)
        # 将原始输入与扩展后的隐藏状态进行逐元素乘法操作
        hidden_states = torch.mul(inputs, hidden_states)

        # 返回经过处理后的隐藏状态张量
        return hidden_states
        super().__init__()
        # 初始化函数，调用父类的初始化方法

        self.apply_dropout = stride == 1 and not id_skip
        # 根据参数确定是否应用 dropout，条件是 stride 为 1 且 id_skip 为 False

        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        # 创建一个卷积层，用于将输入通道数 in_dim 转换为输出通道数 out_dim，
        # 使用 1x1 的卷积核，padding 设置为 "same"，不包含偏置项

        self.project_bn = nn.BatchNorm2d(
            num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 创建一个批归一化层，对输出通道数为 out_dim 的特征图进行批归一化，
        # 使用配置类 config 中指定的批归一化参数 eps 和 momentum

        self.dropout = nn.Dropout(p=drop_rate)
        # 创建一个 dropout 层，使用指定的 dropout rate drop_rate

    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 前向传播函数，输入为 embeddings 和 hidden_states，输出为 torch.Tensor

        hidden_states = self.project_conv(hidden_states)
        # 将 hidden_states 通过之前定义的卷积层 project_conv 进行卷积操作

        hidden_states = self.project_bn(hidden_states)
        # 将卷积后的 hidden_states 通过批归一化层 project_bn 进行批归一化操作

        if self.apply_dropout:
            # 如果 apply_dropout 为 True，则执行以下操作
            hidden_states = self.dropout(hidden_states)
            # 对 hidden_states 应用 dropout 操作
            hidden_states = hidden_states + embeddings
            # 将 dropout 后的 hidden_states 与输入的 embeddings 相加

        return hidden_states
        # 返回处理后的 hidden_states
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置扩展比例
        self.expand_ratio = expand_ratio
        # 根据扩展比例确定是否需要扩展操作
        self.expand = True if self.expand_ratio != 1 else False
        # 计算扩展后的输入维度
        expand_in_dim = in_dim * expand_ratio

        # 如果需要扩展，则创建扩展层对象
        if self.expand:
            self.expansion = AlignVisionExpansionLayer(
                config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride
            )

        # 创建深度卷积层对象
        self.depthwise_conv = AlignVisionDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )
        # 创建挤压激活层对象
        self.squeeze_excite = AlignVisionSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )
        # 创建投影层对象
        self.projection = AlignVisionFinalBlockLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
            id_skip=id_skip,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 将输入的隐藏状态作为嵌入向量
        embeddings = hidden_states
        # 执行扩展和深度卷积阶段
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)

        # 执行挤压激活阶段
        hidden_states = self.squeeze_excite(hidden_states)
        # 执行投影阶段
        hidden_states = self.projection(embeddings, hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class AlignVisionEncoder(nn.Module):
    r"""
    Forward propogates the embeddings through each vision encoder (EfficientNet) block.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
    """

    def __init__(self, config: AlignVisionConfig):
        super().__init__()
        self.depth_coefficient = config.depth_coefficient

        def round_repeats(repeats):
            # Round number of block repeats based on depth multiplier.
            return int(math.ceil(self.depth_coefficient * repeats))

        num_base_blocks = len(config.in_channels)
        num_blocks = sum(round_repeats(n) for n in config.num_block_repeats)

        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]

            for j in range(round_repeats(config.num_block_repeats[i])):
                id_skip = True if j == 0 else False
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = False if curr_block_num in config.depthwise_padding else True
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                block = AlignVisionBlock(
                    config=config,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                    id_skip=id_skip,
                    adjust_padding=adjust_padding,
                )
                blocks.append(block)
                curr_block_num += 1

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # Iterate through each block and perform forward pass
        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        # Return output based on return_dict flag
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert->AlignText
class AlignTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，使用 nn.Embedding 类，设置词汇表大小、隐藏大小，并指定填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，使用 nn.Embedding 类，设置最大位置编码和隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，使用 nn.Embedding 类，设置标记类型词汇表大小和隐藏大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用 nn.LayerNorm 创建层归一化层，设置隐藏大小和 epsilon 参数
        # self.LayerNorm 的命名方式不使用蛇形命名法，以保持与 TensorFlow 模型变量名的一致性，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，设置隐藏单元的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册一个缓冲区 tensor，存储从 0 到 config.max_position_embeddings-1 的整数序列，形状为 (1, max_position_embeddings)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册一个缓冲区 tensor，存储全零的 token_type_ids，形状与 position_ids 相同，数据类型为 long 型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数，接受多个输入参数，输出模型的前向传播结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token 序列的 ID，数据类型为 LongTensor，可选
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型的 ID，数据类型为 LongTensor，可选
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码的 ID，数据类型为 LongTensor，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，数据类型为 FloatTensor，可选
        past_key_values_length: int = 0,  # 过去的键值对长度，整数类型，默认为 0
    ) -> torch.Tensor:
        # 如果给定了 input_ids，则获取其形状作为 input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，从 inputs_embeds 获取形状，去除最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为 None，则从 self.position_ids 中获取一部分，匹配当前序列的长度
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置 token_type_ids 为构造函数中注册的缓冲区，通常是全零，用于在模型追踪时帮助用户，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，将 token_type_ids 初始化为全零张量，与输入形状匹配
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为 None，则使用 word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据 token_type_ids 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将 inputs_embeds 和 token_type_embeddings 相加作为最终的 embeddings
        embeddings = inputs_embeds + token_type_embeddings

        # 如果使用绝对位置编码，则加上 position_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对 embeddings 进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回最终的 embeddings
        return embeddings
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->AlignText
class AlignTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear transformation for query, key, and value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout layer
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Position embedding type handling
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # Flag indicating if the module is used as a decoder
        self.is_decoder = config.is_decoder

    # Reshape and permute the input tensor for attention scores computation
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Forward pass logic will compute attention scores and apply attention mechanisms
        pass

# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->AlignText
class AlignTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Fully connected layer for output transformation
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # Apply dense layer, dropout, layer normalization, and residual connection
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertAttention 复制并修改为 AlignTextAttention 类
class AlignTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 AlignTextSelfAttention 层
        self.self = AlignTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 AlignTextSelfOutput 层
        self.output = AlignTextSelfOutput(config)
        # 存储被修剪的注意力头部的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的注意力头部和相应索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self 层的 forward 方法，进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将 self 输出传递给 output 层，并结合输入的 hidden_states 计算注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 AlignTextIntermediate 类
class AlignTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义线性层，将隐藏状态映射到中间状态
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性映射
        hidden_states = self.dense(hidden_states)
        # 应用中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 AlignTextOutput 类
class AlignTextOutput(nn.Module):
    # 初始化函数，用于创建一个新的实例对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，对输入进行归一化，设置epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，以config.hidden_dropout_prob的概率随机将输入设置为0
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，处理输入的hidden_states和input_tensor，返回处理后的hidden_states
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对hidden_states进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 对dropout后的hidden_states和input_tensor进行残差连接，并进行LayerNorm归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的hidden_states作为最终的输出结果
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制的代码，将Bert->AlignText
class AlignTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前馈传递的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度设定为1
        self.seq_len_dim = 1
        # 初始化AlignTextAttention模块
        self.attention = AlignTextAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨注意力，确保作为解码器模型使用
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置嵌入类型初始化跨注意力模块
            self.crossattention = AlignTextAttention(config, position_embedding_type="absolute")
        # 初始化AlignTextIntermediate模块
        self.intermediate = AlignTextIntermediate(config)
        # 初始化AlignTextOutput模块
        self.output = AlignTextOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention computation using the given inputs and past key/values
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # Retrieve the attention output from the self-attention computation
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Extract outputs excluding the first (attention_output) and last (present_key_value)
            outputs = self_attention_outputs[1:-1]
            # Retrieve the present key/values from self-attention computation
            present_key_value = self_attention_outputs[-1]
        else:
            # Include self attentions in outputs if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention computation using the given inputs and past key/values
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # Retrieve the attention output from the cross-attention computation
            attention_output = cross_attention_outputs[0]
            # Add cross attentions to outputs if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # Append cross-attn cache to present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking to forward computation for feed forward layer
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # Prepare final outputs including layer output
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # Return the computed outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        # Compute intermediate output using the attention output
        intermediate_output = self.intermediate(attention_output)
        # Compute final layer output using intermediate output and attention output
        layer_output = self.output(intermediate_output, attention_output)
        # Return the final layer output
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码，并将Bert->AlignText
class AlignTextEncoder(nn.Module):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        super().__init__()
        # 将配置参数保存在实例中
        self.config = config
        # 创建一个包含多个AlignTextLayer模块的层列表，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([AlignTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志默认为False
        self.gradient_checkpointing = False

    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不需要输出隐藏状态，则初始化空元组；否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化空元组；否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出跨层注意力权重或者模型配置未开启跨层注意力，则初始化空元组；否则为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点并且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 为 True，警告并设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果 use_cache 为 True，则初始化空元组；否则为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在头部掩码，则获取当前层的头部掩码；否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果存在过去的键值对，则获取当前层的过去键值对；否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播函数来计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果 use_cache 为 True，则将当前层的缓存状态加入 next_decoder_cache
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的自注意力权重加入 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置开启了跨层注意力，则将当前层的跨层注意力权重加入 all_cross_attentions
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不使用返回字典形式，则返回一个元组，包含需要的输出项
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则返回一个包含所有输出的对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert -> AlignText
# 定义一个池化层的模块，用于处理ALIGN模型的隐藏状态
class AlignTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出大小都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从隐藏状态中取出第一个标记对应的隐藏状态张量
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态张量通过全连接层进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 将线性变换后的结果应用双曲正切激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


class AlignPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    config_class = AlignConfig
    base_model_prefix = "align"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对线性层和卷积层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AlignModel):
            # 对AlignModel模块的text_projection部分进行权重初始化（使用Xavier均匀分布）
            nn.init.xavier_uniform_(module.text_projection.weight)
            # 将text_projection的偏置项初始化为零
            module.text_projection.bias.data.zero_()
            # 设置_is_hf_initialized标志为True，表示已经初始化
            module.text_projection._is_hf_initialized = True
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.LayerNorm):
            # 对LayerNorm层的偏置项初始化为零
            module.bias.data.zero_()
            # 对LayerNorm层的权重初始化为全1
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    """The text model from ALIGN without any head or projection on top.""",
    ALIGN_START_DOCSTRING,
)
class AlignTextModel(AlignPreTrainedModel):
    config_class = AlignTextConfig

    def __init__(self, config: AlignTextConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        # 初始化AlignTextModel，包括嵌入层和编码器
        self.config = config

        self.embeddings = AlignTextEmbeddings(config)
        self.encoder = AlignTextEncoder(config)

        # 如果需要添加池化层，则初始化池化层
        self.pooler = AlignTextPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置嵌入层的词嵌入为给定的值
        self.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=AlignTextConfig)
    # 定义神经网络模型的前向传播方法，用于生成预测结果或特征
    def forward(
        # 输入序列的 token IDs，可选的 Torch 张量
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，指示模型在计算注意力时应忽略的位置，可选的 Torch 张量
        attention_mask: Optional[torch.Tensor] = None,
        # 标识 token 的类型，例如区分两个句子的情况，可选的 Torch 张量
        token_type_ids: Optional[torch.Tensor] = None,
        # 标识 token 在序列中的位置，可选的 Torch 张量
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码，用于指定哪些注意力头部应该被掩盖，可选的 Torch 张量
        head_mask: Optional[torch.Tensor] = None,
        # 输入的嵌入表示，直接传入而不是使用输入 token IDs 进行嵌入查找，可选的 Torch 张量
        inputs_embeds: Optional[torch.Tensor] = None,
        # 是否返回注意力权重，可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否返回隐藏状态，可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回输出结果，可选的布尔值
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串，描述这是一个来自ALIGN模型的视觉模型，不含任何头部或顶部投影
@add_start_docstrings(
    """The vision model from ALIGN without any head or projection on top.""",
    ALIGN_START_DOCSTRING,
)
# 定义AlignVisionModel类，继承自AlignPreTrainedModel类
class AlignVisionModel(AlignPreTrainedModel):
    # 指定配置类为AlignVisionConfig
    config_class = AlignVisionConfig
    # 定义主要输入名称为"pixel_values"
    main_input_name = "pixel_values"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    # 初始化方法，接受一个AlignVisionConfig类型的config参数
    def __init__(self, config: AlignVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将config保存到实例中
        self.config = config
        # 初始化嵌入层对象AlignVisionEmbeddings
        self.embeddings = AlignVisionEmbeddings(config)
        # 初始化编码器对象AlignVisionEncoder
        self.encoder = AlignVisionEncoder(config)

        # 最终的池化层
        if config.pooling_type == "mean":
            # 如果配置中的池化类型为均值池化，则使用AvgPool2d进行初始化
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            # 如果配置中的池化类型为最大池化，则使用MaxPool2d进行初始化
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            # 如果配置中的池化类型不是'mean'或'max'，则抛出数值错误异常
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回视觉模型的输入嵌入层对象
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.convolution

    # 覆盖模型的forward方法，接受pixel_values、output_hidden_states和return_dict作为参数
    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=AlignVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果用户未指定 output_hidden_states，则使用模型配置中的默认值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果用户未指定 return_dict，则使用模型配置中的默认设置

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果未提供 pixel_values，则抛出数值错误异常

        embedding_output = self.embeddings(pixel_values)
        # 将像素值输入到嵌入层中进行嵌入编码

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用编码器处理嵌入输出，可选地返回隐藏状态和字典格式输出

        # 应用池化操作
        last_hidden_state = encoder_outputs[0]
        # 取编码器输出的第一个元素作为最终的隐藏状态

        pooled_output = self.pooler(last_hidden_state)
        # 使用池化器处理最终隐藏状态，得到池化输出（通常是 CLS 标记）

        # 重新调整形状 (batch_size, projection_dim, 1 , 1) -> (batch_size, projection_dim)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        if not return_dict:
            # 如果未设置返回字典格式，则返回一个元组
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            # 如果设置了返回字典格式，则返回包含所有信息的特定输出对象
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 定义 AlignModel 类，继承自 AlignPreTrainedModel
@add_start_docstrings(ALIGN_START_DOCSTRING)
class AlignModel(AlignPreTrainedModel):
    # 指定配置类为 AlignConfig
    config_class = AlignConfig

    # 初始化函数，接受一个 AlignConfig 类型的 config 对象作为参数
    def __init__(self, config: AlignConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 检查 config.text_config 是否为 AlignTextConfig 类型，否则抛出 ValueError 异常
        if not isinstance(config.text_config, AlignTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type AlignTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 AlignVisionConfig 类型，否则抛出 ValueError 异常
        if not isinstance(config.vision_config, AlignVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type AlignVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将 text_config 和 vision_config 存储在局部变量中
        text_config = config.text_config
        vision_config = config.vision_config

        # 存储配置中的 projection_dim 和 text_embed_dim 到当前对象的属性中
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        # 创建 AlignTextModel 和 AlignVisionModel 的实例，分别使用 text_config 和 vision_config 作为参数
        self.text_model = AlignTextModel(text_config)
        self.vision_model = AlignVisionModel(vision_config)

        # 创建一个线性层，用于文本嵌入维度到投影维度的转换
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        
        # 创建一个可学习的参数 temperature，并使用 config 中的 temperature_init_value 进行初始化
        self.temperature = nn.Parameter(torch.tensor(self.config.temperature_init_value))

        # 调用 post_init 方法，用于初始化权重和应用最终处理
        self.post_init()

    # 在模型前向传播时，添加文档字符串说明，描述输入参数的作用
    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`AlignTextModel`].

        Examples:

        ```
        >>> from transformers import AutoTokenizer, AlignModel

        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input parameters to the text_model of ALIGN model and retrieve outputs
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the pooled output (first element) from text_model outputs
        last_hidden_state = text_outputs[0][:, 0, :]
        # Apply text projection layer to obtain text features
        text_features = self.text_projection(last_hidden_state)

        # Return the computed text features
        return text_features
        ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`AlignVisionModel`].

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AlignModel

        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        # 设置是否返回隐藏状态，默认为 ALIGN 模型配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典格式的输出，默认为 ALIGN 模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型来获取视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉输出中取出汇总输出作为图像特征
        image_features = vision_outputs[1]  # pooled_output

        # 返回图像特征
        return image_features

    @add_start_docstrings_to_model_forward(ALIGN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AlignOutput, config_class=AlignConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```