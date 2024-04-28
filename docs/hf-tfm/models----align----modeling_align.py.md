# `.\transformers\models\align\modeling_align.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，对代码进行许可
# 你可以在遵守许可证的情况下使用此文件
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" PyTorch ALIGN model."""

# 导入所需的库
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入相关模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "kakaobrain/align-base"
_CONFIG_FOR_DOC = "AlignConfig"

# 预训练模型存档列表
ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kakaobrain/align-base",
    # 查看所有 ALIGN 模型 https://huggingface.co/models?filter=align
]

# ALIGN 模型的起始文档字符串
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

# ALIGN 文本输入的文档字符串
ALIGN_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
            # 可以使用 [`AutoTokenizer`] 获取这些索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            [输入 ID 是什么？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮盖掩码，避免在填充标记索引上执行注意力操作。选择在 `[0, 1]` 范围内的掩码值：
            # - 1 表示 **未被掩盖** 的标记，
            # - 0 表示 **被掩盖** 的标记。
            [注意力掩码是什么？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。
            [位置 ID 是什么？](../glossary#position-ids)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 中：
            # - 0 对应 *句子 A* 标记，
            # - 1 对应 *句子 B* 标记。
            [令牌类型 ID 是什么？](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意模块的选定头部置空的掩码。在 `[0, 1]` 中选择掩码值：
            # - 1 表示 **未被掩盖** 的头部，
            # - 0 表示 **被掩盖** 的头部。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示，而不是传递 `input_ids`。如果您想更精确地控制如何将 `input_ids` 索引转换为相关联的向量，
            # 这很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
"""

# ALIGN_VISION_INPUTS_DOCSTRING 是一个文档字符串，用于描述 AlignVisionModel 的输入参数
ALIGN_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`EfficientNetImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# ALIGN_INPUTS_DOCSTRING 是一个文档字符串，用于描述 AlignTextModel 的输入参数
ALIGN_INPUTS_DOCSTRING = r"""
"""


# AlignVisionModelOutput 类，用于存储视觉模型的输出结果
@dataclass
class AlignVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

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


# AlignTextModelOutput 类，用于存储文本模型的输出结果
@dataclass
class AlignTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.
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

    # 定义可选的文本嵌入向量，形状为(batch_size, output_dim)，在使用`with_projection=True`初始化模型时返回
    text_embeds: Optional[torch.FloatTensor] = None
    # 定义最后一层模型的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    last_hidden_state: torch.FloatTensor = None
    # 定义隐藏状态的元组，包含每一层的隐藏状态，如果模型有嵌入层，则还包含嵌入层的输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义注意力的元组，包含每一层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类 AlignOutput，继承自 ModelOutput
@dataclass
class AlignOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`AlignTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The output of [`AlignVisionModel`].
        text_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`AlignTextModel`].
        vision_model_output(`BaseModelOutputWithPoolingAndNoAttention`):
            The output of the [`AlignVisionModel`].
    """

    # 定义类的属性
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = None

    # 将类转换为元组的方法
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html 改编而来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device), label_smoothing=0.1)


# 对齐损失函数
def align_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算文本损失
    caption_loss = contrastive_loss(similarity)
    # 计算图像损失
    image_loss = contrastive_loss(similarity.t())
    # 返回文本损失和图像损失的平均值
    return (caption_loss + image_loss) / 2.0


# 从 transformers.models.efficientnet.modeling_efficientnet.round_filters 复制的函数，将 EfficientNet 改为 AlignVision
def round_filters(config: AlignVisionConfig, num_channels: int):
    r"""
    Round number of filters based on depth multiplier.
    """
    # 获取深度因子
    divisor = config.depth_divisor
    # 根据宽度系数调整通道数
    num_channels *= config.width_coefficient
    # 对通道数进行四舍五入
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # 确保四舍五入不会减少超过 10%
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)
# 从 efficientnet.modeling_efficientnet 模块中复制了 correct_pad 函数
def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
    r"""
    Utility function to get the tuple padding value for the depthwise convolution.

    Args:
        kernel_size (`int` or `tuple`):
            Kernel size of the convolution layers.
        adjust (`bool`, *optional*, defaults to `True`):
            Adjusts padding value to apply to right and bottom sides of the input.
    """
    # 如果 kernel_size 是整数，转换为元组
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # 计算正确的填充值
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    # 如果需要调整填充值，则返回调整后的填充值，否则返回原始填充值
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    else:
        return (correct[1], correct[1], correct[0], correct[0])


# 从 efficientnet.modeling_efficientnet 模块中复制了 AlignVisionEmbeddings 类
class AlignVisionEmbeddings(nn.Module):
    r"""
    A module that corresponds to the stem module of the original work.
    """

    def __init__(self, config: AlignVisionConfig):
        super().__init__()

        # 计算输出维度
        self.out_dim = round_filters(config, 32)
        # 使用 ZeroPad2d 进行填充
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        # 创建卷积层
        self.convolution = nn.Conv2d(
            config.num_channels, self.out_dim, kernel_size=3, stride=2, padding="valid", bias=False
        )
        # 创建批归一化层
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        # 使用激活函数
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 对输入进行填充
        features = self.padding(pixel_values)
        # 进行卷积操作
        features = self.convolution(features)
        # 进行批归一化操作
        features = self.batchnorm(features)
        # 应用激活函数
        features = self.activation(features)

        return features


# 从 efficientnet.modeling_efficientnet 模块中复制了 AlignVisionDepthwiseConv2d 类
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
        # 计算输出通道数
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


# 从 efficientnet.modeling_efficientnet 模块中复制了 AlignVisionExpansionLayer 类
class AlignVisionExpansionLayer(nn.Module):
    r"""
    This corresponds to the expansion phase of each block in the original implementation.
    """
```  
    # 初始化函数，用于创建一个 ExpandLayer 类的实例
    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个卷积层，用于将输入特征图维度扩展为输出特征图维度
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",  # 使用相同的填充
            bias=False,  # 不使用偏置
        )
        # 创建一个批归一化层，用于对输出特征图进行批归一化处理
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        # 根据配置选择激活函数
        self.expand_act = ACT2FN[config.hidden_act]

    # 前向传播函数，用于执行扩展层的前向传播过程
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Expand phase（扩展阶段）
        # 通过卷积层对输入特征图进行维度扩展
        hidden_states = self.expand_conv(hidden_states)
        # 对输出特征图进行批归一化处理
        hidden_states = self.expand_bn(hidden_states)
        # 应用激活函数
        hidden_states = self.expand_act(hidden_states)

        # 返回处理后的特征图
        return hidden_states
# 从EfficientNet的模型中复制过来的深度可分离卷积层，用于AlignVision模型
class AlignVisionDepthwiseLayer(nn.Module):
    r"""
    这对应于原始实现中每个块的深度可分离卷积阶段。
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
        # 初始化深度可分离卷积层，根据不同的步幅选择不同的填充方式
        self.stride = stride
        conv_pad = "valid" if self.stride == 2 else "same"
        padding = correct_pad(kernel_size, adjust=adjust_padding)

        # 设置深度可分离卷积的零填充层
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        # 设置深度可分离卷积层
        self.depthwise_conv = AlignVisionDepthwiseConv2d(
            in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False
        )
        # 设置深度可分离卷积层后的批归一化层
        self.depthwise_norm = nn.BatchNorm2d(
            num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 设置深度可分离卷积层后的激活函数
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 深度可分离卷积
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


# 从EfficientNet的模型中复制过来的挤压激励层，用于AlignVision模型
class AlignVisionSqueezeExciteLayer(nn.Module):
    r"""
    这对应于原始实现中每个块的挤压激励阶段。
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, expand_dim: int, expand: bool = False):
        super().__init__()
        # 确定挤压激励阶段的维度
        self.dim = expand_dim if expand else in_dim
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))

        # 设置挤压层，将输入特征图压缩到1x1
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        # 设置减少通道数的卷积层
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        # 设置扩张通道数的卷积层
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=1,
            padding="same",
        )
        # 设置减少通道数的激活函数
        self.act_reduce = ACT2FN[config.hidden_act]
        # 设置扩张通道数的激活函数
        self.act_expand = nn.Sigmoid()
    # 定义一个前向传播函数，接受隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 将输入赋值给变量inputs
        inputs = hidden_states
        # 对隐藏状态进行压缩操作
        hidden_states = self.squeeze(hidden_states)
        # 对压缩后的隐藏状态进行降维操作
        hidden_states = self.reduce(hidden_states)
        # 对降维后的隐藏状态进行激活函数处理

        hidden_states = self.act_reduce(hidden_states)
        # 对隐藏状态进行扩展操作
        hidden_states = self.expand(hidden_states)
        # 对扩展后的隐藏状态进行激活函数处理
        hidden_states = self.act_expand(hidden_states)
        # 将输入与处理后的隐藏状态进行逐元素相乘
        hidden_states = torch.mul(inputs, hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class AlignVisionFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """

    def __init__(
        self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool
    ):
        # 初始化函数，设置模块的属性
        super().__init__()
        # 根据条件确定是否应用 dropout
        self.apply_dropout = stride == 1 and not id_skip
        # 创建 1x1 卷积层，用于投影
        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        # 创建批归一化层
        self.project_bn = nn.BatchNorm2d(
            num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 创建 dropout 层
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 对隐藏状态进行投影卷积
        hidden_states = self.project_conv(hidden_states)
        # 对投影后的隐藏状态进行批归一化
        hidden_states = self.project_bn(hidden_states)

        # 如果需要应用 dropout，则进行 dropout 操作并与嵌入向量相加
        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings

        return hidden_states


class AlignVisionBlock(nn.Module):
    r"""
    This corresponds to the block module of original the EfficientNet vision encoder implementation.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """

    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
        id_skip: bool,
        adjust_padding: bool,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置扩展比例
        self.expand_ratio = expand_ratio
        # 如果扩展比例不为1，则设置self.expand为True，否则为False
        self.expand = True if self.expand_ratio != 1 else False
        # 计算扩展后的维度
        expand_in_dim = in_dim * expand_ratio

        # 如果需要扩展
        if self.expand:
            # 创建对齐视觉扩展层对象
            self.expansion = AlignVisionExpansionLayer(
                config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride
            )

        # 创建对齐视觉深度卷积层对象
        self.depthwise_conv = AlignVisionDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )
        # 创建对齐视觉压缩激励层对象
        self.squeeze_excite = AlignVisionSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )
        # 创建对齐视觉最终块层对象
        self.projection = AlignVisionFinalBlockLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
            id_skip=id_skip,
        )

    # 前向传播函数
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 将隐藏状态作为嵌入
        embeddings = hidden_states
        # 扩展和深度卷积阶段
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)

        # 压缩和激励阶段
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.projection(embeddings, hidden_states)
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
            # 根据深度系数四舍五入重复块的数量。
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

        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# 从transformers.models.bert.modeling_bert.BertEmbeddings复制并修改为Bert->AlignText
class AlignTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建词嵌入层，用于将词索引映射成词向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，用于表示词的位置信息
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，用于表示词的类型信息
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 用于归一化隐藏状态的层
        # self.LayerNorm 不使用蛇形命名法以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 随机失活层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，绝对或相对
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置 ID 缓冲区，用于存储位置嵌入的位置索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册标记类型 ID 缓冲区，用于存储标记类型嵌入的标记类型索引
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数，用于计算模型的输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 定义一个方法，用于生成 Transformer 模型的输入嵌入张量
    ) -> torch.Tensor:
        # 如果输入的是 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 否则，获取 inputs_embeds 的形状，去掉最后一个维度（通常是 batch_size）
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从 self.position_ids 中获取一部分，保持与输入序列相同长度
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置 token_type_ids 为注册在构造函数中的缓冲区，其中全部为零，通常在自动生成时发生。
        # 注册的缓冲区可帮助用户在不传递 token_type_ids 的情况下跟踪模型，解决问题＃5664
        if token_type_ids is None:
            # 如果存在 self.token_type_ids 属性
            if hasattr(self, "token_type_ids"):
                # 从已注册的缓冲区中获取 token_type_ids 的一部分，保持与序列相同的长度
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 将 buffered_token_type_ids 扩展为与输入形状相同的形状
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 如果不存在 self.token_type_ids 属性，则创建一个全部为零的 token_type_ids 张量
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则使用 word_embeddings 方法生成
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据 token_type_ids 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将 inputs_embeds 和 token_type_embeddings 相加，得到嵌入向量
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为 "absolute"，则添加位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对嵌入向量进行 LayerNormalization
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 Dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制代码，并将Bert->AlignText
class AlignTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的倍数，如果不是则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量转换为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,



# 从transformers.models.bert.modeling_bert.BertSelfOutput复制代码，并将Bert->AlignText
class AlignTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层、LayerNorm和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertAttention复制代码，并将Bert->AlignText
class AlignTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化AlignTextSelfAttention和AlignTextSelfOutput
        self.self = AlignTextSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = AlignTextSelfOutput(config)
        self.pruned_heads = set()

    # 剪枝头部
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播
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
        # 调用self的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用output的前向传播
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力，则添加注意力
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制代码，并将Bert->AlignText
class AlignTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层���激活函数
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性层传播
        hidden_states = self.dense(hidden_states)
        # 激活函数传播
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput复制代码，并将Bert->AlignText
class AlignTextOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个张量参数，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states传入全连接层，得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的输出与input_tensor相加，然后传入LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回LayerNorm层的输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制代码，并将Bert->AlignText
class AlignTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化AlignTextLayer类，设置一些属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AlignTextAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果需要添加跨注意力机制
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨注意力机制
            self.crossattention = AlignTextAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = AlignTextIntermediate(config)
        # 初始化输出层
        self.output = AlignTextOutput(config)

    # 前向传播函数
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
        # 如果过去的键/值不为空，则将decoder单向自注意力的缓存键/值元组放在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self.attention进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
          
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组在过去键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用crossattention进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到现在的键/值元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块技术对前向传播进行处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 使用中间层进行前向传播
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层进行前向传播
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制而来，用于处理文本对齐的编码器
class AlignTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化AlignTextEncoder类，设置配置参数
        self.config = config
        # 创建一个由AlignTextLayer对象组成的层列表，列表长度为配置中的隐藏层数量
        self.layer = nn.ModuleList([AlignTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点
        self.gradient_checkpointing = False

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
    # 定义函数的返回类型为 Tuple[torch.Tensor] 或 BaseModelOutputWithPastAndCrossAttentions 中的一种
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为空元组，否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化 all_self_attentions 为空元组，否则设为 None
        all_self_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力权重，并且配置中指定添加交叉注意力，则初始化 all_cross_attentions 为空元组，否则设为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了渐变检查点且处于训练状态
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 为 True，则给出警告，并将其设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果 use_cache 为 True，则初始化 next_decoder_cache 为空元组，否则设为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果没有指定则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对，如果没有则设为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了渐变检查点且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用渐变检查点的函数计算当前层的输出
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
                # 否则直接调用当前层的 __call__ 方法计算输出
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
            # 如果 use_cache 为 True，则将当前层的输出的最后一个元素添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中指定添加交叉注意力，则将当前层的交叉注意力权重添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 返回非空元素的元组
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
        # 返回 BaseModelOutputWithPastAndCrossAttentions 类的实例，包含最终的输出结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制代码，并将Bert改为AlignText
class AlignTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地取第一个标记对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AlignPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和下载预训练模型的简单接口的抽象类。
    """

    config_class = AlignConfig
    base_model_prefix = "align"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AlignModel):
            nn.init.xavier_uniform_(module.text_projection.weight)
            module.text_projection.bias.data.zero_()
            module.text_projection._is_hf_initialized = True
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    """从ALIGN中获取的文本模型，没有任何头部或顶部的投影。""",
    ALIGN_START_DOCSTRING,
)
class AlignTextModel(AlignPreTrainedModel):
    config_class = AlignTextConfig

    def __init__(self, config: AlignTextConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = AlignTextEmbeddings(config)
        self.encoder = AlignTextEncoder(config)

        self.pooler = AlignTextPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=AlignTextConfig)
    # Transformer 模型的前向传播函数，用于生成模型的输出
    def forward(
        # 输入的 token IDs，表示输入文本中每个 token 的编号
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，用于指示哪些位置需要被注意，哪些位置可以被忽略
        attention_mask: Optional[torch.Tensor] = None,
        # token 类型 IDs，用于区分不同句子或不同部分的 token
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 IDs，用于表示每个 token 的位置信息
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码，用于控制每个注意力头部的作用程度
        head_mask: Optional[torch.Tensor] = None,
        # 输入的嵌入向量，用于直接输入而不是通过 token IDs
        inputs_embeds: Optional[torch.Tensor] = None,
        # 是否返回注意力权重信息
        output_attentions: Optional[bool] = None,
        # 是否返回隐藏状态信息
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回结果
        return_dict: Optional[bool] = None,
# 添加起始文档字符串到类的注释
@add_start_docstrings(
    """The vision model from ALIGN without any head or projection on top.""",
    ALIGN_START_DOCSTRING,
)
# 定义 AlignVisionModel 类，继承自 AlignPreTrainedModel
class AlignVisionModel(AlignPreTrainedModel):
    # 指定配置类为 AlignVisionConfig
    config_class = AlignVisionConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    # 初始化方法
    def __init__(self, config: AlignVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置
        self.config = config
        # 创建 AlignVisionEmbeddings 对象
        self.embeddings = AlignVisionEmbeddings(config)
        # 创建 AlignVisionEncoder 对象
        self.encoder = AlignVisionEncoder(config)

        # 最终池化层
        if config.pooling_type == "mean":
            # 使用平均池化
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            # 使用最大池化
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            # 抛出数值错误异常
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.convolution

    # 前向传播方法
    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=AlignVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AlignVisionModel

        >>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        # 检查是否应返回隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否应返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查像素值是否已提供
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 计算嵌入输出
        embedding_output = self.embeddings(pixel_values)
        # 使用编码器生成输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 应用池化
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        # 重塑形状 (batch_size, projection_dim, 1 , 1) -> (batch_size, projection_dim)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带池化且无注意力的基础模型输出
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 为 AlignModel 类添加文档字符串
@add_start_docstrings(ALIGN_START_DOCSTRING)
class AlignModel(AlignPreTrainedModel):
    # 指定配置类为 AlignConfig
    config_class = AlignConfig

    # 初始化函数，接受一个 AlignConfig 实例作为参数
    def __init__(self, config: AlignConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查文本配置和视觉配置是否符合预期类型
        if not isinstance(config.text_config, AlignTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type AlignTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, AlignVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type AlignVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 提取文本配置和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化模型的属性
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        # 创建文本模型和视觉模型实例
        self.text_model = AlignTextModel(text_config)
        self.vision_model = AlignVisionModel(vision_config)

        # 创建文本特征投影层
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        # 创建温度参数作为模型的属性
        self.temperature = nn.Parameter(torch.tensor(self.config.temperature_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型前向传播函数添加文档字符串
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

        ```python
        >>> from transformers import AutoTokenizer, AlignModel

        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        # 设置输出注意力、隐藏状态和返回字典的配置，使用 ALIGN 模型的配置而不是视觉和文本组件的配置（如果指定了的话）。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本模型以获取文本输出
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

        # 从文本输出中提取最后一个隐藏状态，并通过文本投影层获得文本特征
        last_hidden_state = text_outputs[0][:, 0, :]
        text_features = self.text_projection(last_hidden_state)

        # 返回文本特征
        return text_features

    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`AlignVisionModel`].

        Examples:

        ```python
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
        # 如果指定了输出隐藏状态，则使用ALIGN模型的配置中的一些字段，而不是视觉和文本组件中的字段。
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了返回字典，则使用ALIGN模型的配置中的返回字典标志。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用视觉模型处理输入像素值
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取视觉输出的第二个元素，即池化后的输出
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
```