# `.\transformers\models\bit\modeling_bit.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，包括版权信息和许可证信息
# 此模型由 Google AI 和 HuggingFace Inc. 团队版权所有
# 根据 Apache 许可证 2.0 进行许可
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，无任何明示或暗示的担保
# 有关许可证的详细信息，请参阅许可证

"""PyTorch BiT 模型。还支持 ViT 混合的骨干。"""

# 导入必要的库
import collections
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关的模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "BitConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "google/bit-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "google/bit-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

# 预训练模型存档列表
BIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/bit-50",
    # 查看所有 BiT 模型 https://huggingface.co/models?filter=bit
]

# 获取填充值的实用函数
def get_padding_value(padding=None, kernel_size=7, stride=1, dilation=1) -> Tuple[Tuple, bool]:
    r"""
    给定 kernel_size 和 padding，获取填充值的元组的实用函数。

    Args:
        padding (Union[`str`, `int`], *optional*):
            填充值，可以是 `"same"`，`"valid"`。如果提供了其他值，则使用 PyTorch 的默认填充。
        kernel_size (`int`, *optional*, defaults to 7):
            卷积层的内核大小。
        stride (`int`, *optional*, defaults to 1):
            卷积层的步长值。
        dilation (`int`, *optional*, defaults to 1):
            卷积层的膨胀值。
    """
    dynamic = False
    if padding is None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        return padding, dynamic
    # 检查 padding 是否为字符串类型
    if isinstance(padding, str):
        # 将 padding 转换为小写形式
        padding = padding.lower()
        # 如果 padding 为 "same"
        if padding == "same":
            # TF 兼容的 'SAME' padding，对性能和 GPU 内存分配有影响
            if stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0:
                # 静态情况下，没有额外的开销
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
            else:
                # 动态 'SAME' padding，有运行时/GPU 内存开销
                padding = 0
                dynamic = True
        # 如果 padding 为 "valid"
        elif padding == "valid":
            # 'VALID' padding，等同于 padding=0
            padding = 0
        else:
            # 默认为 PyTorch 风格的类似 "same" 的对称 padding
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    # 返回 padding 和 dynamic 变量的值
    return padding, dynamic
class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Includes TensorFlow compatible SAME padding. Used for ViT Hybrid model.

    Paper: [Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization](https://arxiv.org/abs/1903.10520v2)
    """

    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-6,
    ):
        # 获取填充值和是否为动态填充
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        # 调用父类的初始化方法
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # 根据是否为动态填充，初始化填充层
        if is_dynamic:
            self.pad = DynamicPad2d(kernel_size, stride, dilation)
        else:
            self.pad = None
        self.eps = eps

    def forward(self, hidden_state):
        # 如果存在填充层，则对隐藏状态进行填充
        if self.pad is not None:
            hidden_state = self.pad(hidden_state)
        # 对权重进行标准化
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, training=True, momentum=0.0, eps=self.eps
        ).reshape_as(self.weight)
        # 使用卷积操作
        hidden_state = nn.functional.conv2d(
            hidden_state, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return hidden_state


class BitGroupNormActivation(nn.GroupNorm):
    r"""
    A module that combines group normalization with an activation function.
    """

    def __init__(self, config, num_channels, eps=1e-5, affine=True, apply_activation=True):
        # 调用父类的初始化方法
        super(BitGroupNormActivation, self).__init__(config.num_groups, num_channels, eps=eps, affine=affine)
        # 根据是否应用激活函数，初始化激活函数
        if apply_activation:
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = nn.Identity()

    def forward(self, hidden_state):
        # 对隐藏状态进行组归一化
        hidden_state = nn.functional.group_norm(hidden_state, self.num_groups, self.weight, self.bias, self.eps)
        # 应用激活函数
        hidden_state = self.activation(hidden_state)
        return hidden_state


class DynamicPad2d(nn.Module):
    r"""
    A module that wraps dynamic padding of any input, given the parameters of the convolutional layer and the input
    hidden states.
    """
    # 初始化函数，设置卷积核大小、步长、膨胀率和默认值
    def __init__(self, kernel_size, stride, dilation, value=0):
        # 调用父类的初始化函数
        super().__init__()
        # 检查卷积核大小是否为整数，如果是则转换为元组
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # 检查步长是否为整数，如果是则转换为元组
        if isinstance(stride, int):
            stride = (stride, stride)

        # 检查膨胀率是否为整数，如果是则转换为元组
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        # 设置卷积核大小、步长、膨胀率和默认值
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.value = value

        # 定义计算填充值的函数
        def compute_padding(x, kernel_size, stride, dilation):
            return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

        # 将计算填充值的函数赋值给对象属性
        self.compute_padding = compute_padding

    # 调用函数，对输入进行填充
    def __call__(self, input):
        # 获取输入的宽度和高度
        input_height, input_width = input.size()[-2:]

        # 计算填充值
        padding_height = self.compute_padding(input_height, self.kernel_size[0], self.stride[0], self.dilation[0])
        padding_width = self.compute_padding(input_width, self.kernel_size[1], self.stride[1], self.dilation[1])

        # 如果需要填充，则对输入进行填充
        if padding_height > 0 or padding_width > 0:
            input = nn.functional.pad(
                input,
                [
                    padding_width // 2,
                    padding_width - padding_width // 2,
                    padding_height // 2,
                    padding_height - padding_height // 2,
                ],
                value=self.value,
            )
        # 返回填充后的输入
        return input
class BitMaxPool2d(nn.MaxPool2d):
    """定义一个继承自 nn.MaxPool2d 的类，实现类似于 Tensorflow 中 'SAME' 功能的 2D 最大池化"""

    def __init__(
        self,
        kernel_size: int,
        stride=None,
        dilation=1,
        ceil_mode=False,
        padding=(0, 0),
        padding_value=0,
        use_dynamic_padding=True,
    ):
        # 将 kernel_size 转换为元组形式
        kernel_size = kernel_size if isinstance(kernel_size, collections.abc.Iterable) else (kernel_size, kernel_size)
        # 将 stride 转换为元组形式
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        # 将 dilation 转换为元组形式
        dilation = dilation if isinstance(dilation, collections.abc.Iterable) else (dilation, dilation)
        # 调用父类的初始化方法
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode)
        # 根据 use_dynamic_padding 决定是否使用动态填充
        if use_dynamic_padding:
            self.pad = DynamicPad2d(kernel_size, stride, dilation, padding_value)
        else:
            self.pad = nn.Identity()

    def forward(self, hidden_states):
        # 对输入进行填充
        hidden_states = self.pad(hidden_states)
        # 返回最大池化结果
        return nn.functional.max_pool2d(
            hidden_states, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode
        )


class BitEmbeddings(nn.Module):
    """
    BiT Embeddings (stem) 由一个激进的卷积层组成。
    """

    def __init__(self, config: BitConfig):
        super().__init__()

        # 定义一个权重标准化的卷积层
        self.convolution = WeightStandardizedConv2d(
            config.num_channels,
            config.embedding_size,
            kernel_size=7,
            stride=2,
            eps=1e-8,
            padding=config.global_padding,
        )

        # 定义一个 BitMaxPool2d 实例
        self.pooler = BitMaxPool2d(kernel_size=3, stride=2, use_dynamic_padding=config.embedding_dynamic_padding)

        # 根据配置使用相同的填充策略
        if config.global_padding is not None and config.global_padding.upper() == "SAME":
            self.pad = nn.Identity()
        else:
            self.pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0)

        # 根据配置选择是否使用 BitGroupNormActivation 或者 nn.Identity
        if not config.layer_type == "preactivation":
            self.norm = BitGroupNormActivation(config, num_channels=config.embedding_size)
        else:
            self.norm = nn.Identity()

        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        # 检查通道数是否匹配配置中的通道数
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 进行卷积操作
        embedding = self.convolution(pixel_values)

        # 对卷积结果进行填充
        embedding = self.pad(embedding)

        # 对填充后的结果进行规范化
        embedding = self.norm(embedding)

        # 对规范化后的结果进行池化
        embedding = self.pooler(embedding)

        return embedding


# 从 transformers.models.convnext.modeling_convnext.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    对每个样本进行路径丢弃（随机深度）（应用于残差块的主路径）。
    # 如果 dropout 概率为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的神经元概率
    keep_prob = 1 - drop_prob
    # 确定随机张量的形状，以便适应不同维度的张量，而不仅仅是 2D 卷积网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成与输入形状相同的随机张量，并将其类型设定为与输入相同的数据类型，并放置在与输入相同的设备上
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量二值化
    random_tensor.floor_()
    # 对输入进行 dropout 处理，并乘以随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出
    return output
# 从transformers.models.beit.modeling_beit.BeitDropPath复制并修改为BitDropPath
class BitDropPath(nn.Module):
    """对每个样本进行路径丢弃（随机深度），应用于残差块的主路径中。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        # 设置丢弃概率
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用drop_path函数进行路径丢弃
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


def make_div(value, divisor=8):
    # 设置最小值
    min_value = divisor
    # 对value进行修正，确保可以被divisor整除
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 如果修正后的值小于原值的90%，则增加divisor以避免过小
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class BitPreActivationBottleneckLayer(nn.Module):
    """预激活（v2）瓶颈块。
    遵循"Identity Mappings in Deep Residual Networks"的实现：
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    不同之处在于当可用时将步长放在3x3卷积上。
    """

    def __init__(
        self,
        config,
        in_channels,
        out_channels=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        drop_path_rate=0.0,
        is_first_layer=False,
    ):
        super().__init__()

        first_dilation = first_dilation or dilation

        out_channels = out_channels or in_channels
        mid_channels = make_div(out_channels * bottle_ratio)

        # 如果是第一层，则设置下采样
        if is_first_layer:
            self.downsample = BitDownsampleConv(
                config,
                in_channels,
                out_channels,
                stride=stride,
                preact=True,
            )
        else:
            self.downsample = None

        # 第一个预激活卷积层
        self.norm1 = BitGroupNormActivation(config, in_channels)
        self.conv1 = WeightStandardizedConv2d(in_channels, mid_channels, 1, eps=1e-8, padding=config.global_padding)

        # 第二个预激活卷积层
        self.norm2 = BitGroupNormActivation(config, num_channels=mid_channels)
        self.conv2 = WeightStandardizedConv2d(
            mid_channels, mid_channels, 3, stride=stride, groups=groups, eps=1e-8, padding=config.global_padding
        )

        # 第三个预激活卷积层
        self.norm3 = BitGroupNormActivation(config, mid_channels)
        self.conv3 = WeightStandardizedConv2d(mid_channels, out_channels, 1, eps=1e-8, padding=config.global_padding)

        # 设置路径丢弃
        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行预处理
        hidden_states_preact = self.norm1(hidden_states)

        # 备选分支
        shortcut = hidden_states
        # 如果存在下采样函数，则对隐藏状态进行下采样
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states_preact)

        # 残差分支
        hidden_states = self.conv1(hidden_states_preact)
        hidden_states = self.conv2(self.norm2(hidden_states))
        hidden_states = self.conv3(self.norm3(hidden_states))
        hidden_states = self.drop_path(hidden_states)
        # 返回残差分支结果与备选分支结果的和
        return hidden_states + shortcut
class BitBottleneckLayer(nn.Module):
    """Non Pre-activation bottleneck block, equivalent to V1.5/V1b bottleneck. Used for ViT Hybrid."""

    def __init__(
        self,
        config,
        in_channels,
        out_channels=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        drop_path_rate=0.0,
        is_first_layer=False,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation  # 如果未指定，则使用 dilation 参数

        out_channels = out_channels or in_channels  # 如果未指定输出通道数，则与输入通道数相同
        mid_chs = make_div(out_channels * bottle_ratio)  # 计算中间通道数，确保为整数

        if is_first_layer:  # 如果是第一层
            self.downsample = BitDownsampleConv(  # 创建下采样模块
                config,
                in_channels,
                out_channels,
                stride=stride,  # 设置步幅
                preact=False,  # 不进行预激活
            )
        else:
            self.downsample = None  # 否则不进行下采样

        self.conv1 = WeightStandardizedConv2d(in_channels, mid_chs, 1, eps=1e-8, padding=config.global_padding)  # 第一个卷积层
        self.norm1 = BitGroupNormActivation(config, num_channels=mid_chs)  # 第一个规范化和激活层
        self.conv2 = WeightStandardizedConv2d(  # 第二个卷积层
            mid_chs,
            mid_chs,
            3,
            stride=stride,  # 设置步幅
            dilation=first_dilation,  # 设置膨胀率
            groups=groups,  # 设置分组数
            eps=1e-8,
            padding=config.global_padding,
        )
        self.norm2 = BitGroupNormActivation(config, num_channels=mid_chs)  # 第二个规范化和激活层
        self.conv3 = WeightStandardizedConv2d(mid_chs, out_channels, 1, eps=1e-8, padding=config.global_padding)  # 第三个卷积层
        self.norm3 = BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)  # 第三个规范化层
        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()  # 设置丢弃路径，如果丢弃率大于0，则使用 BitDropPath，否则使用恒等映射

        self.activation = ACT2FN[config.hidden_act]  # 设置激活函数

    def forward(self, hidden_states):
        # shortcut branch
        shortcut = hidden_states  # 按捷径分支连接隐藏状态
        if self.downsample is not None:  # 如果存在下采样模块
            shortcut = self.downsample(hidden_states)  # 通过下采样模块处理隐藏状态

        # residual
        hidden_states = self.conv1(hidden_states)  # 第一个卷积层
        hidden_states = self.norm1(hidden_states)  # 第一个规范化和激活层

        hidden_states = self.conv2(hidden_states)  # 第二个卷积层
        hidden_states = self.norm2(hidden_states)  # 第二个规范化和激活层

        hidden_states = self.conv3(hidden_states)  # 第三个卷积层
        hidden_states = self.norm3(hidden_states)  # 第三个规范化层

        hidden_states = self.drop_path(hidden_states)  # 丢弃路径处理隐藏状态
        hidden_states = self.activation(hidden_states + shortcut)  # 添加捷径并应用激活函数
        return hidden_states


class BitDownsampleConv(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        stride=1,
        preact=True,
    ):
        super().__init__()
        self.conv = WeightStandardizedConv2d(  # 初始化卷积层
            in_channels, out_channels, 1, stride=stride, eps=1e-8, padding=config.global_padding
        )
        self.norm = (  # 初始化规范化和激活层，如果预激活为真则使用恒等映射，否则使用 BitGroupNormActivation
            nn.Identity()
            if preact
            else BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)
        )
    # 定义前向传播方法，接受输入张量 x，返回经过卷积和归一化处理后的结果
    def forward(self, x):
        # 将输入张量 x 经过卷积操作后得到的结果传递给归一化层，并返回结果
        return self.norm(self.conv(x))
class BitStage(nn.Module):
    """
    A ResNet v2 stage composed by stacked layers.
    """

    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        stride,
        dilation,
        depth,
        bottle_ratio=0.25,
        layer_dropout=None,
    ):
        super().__init__()

        # Set dilation for the first layer based on given dilation value
        first_dilation = 1 if dilation in (1, 2) else 2

        # Determine the type of layer to be used based on configuration
        if config.layer_type == "bottleneck":
            layer_cls = BitBottleneckLayer
        else:
            layer_cls = BitPreActivationBottleneckLayer

        prev_chs = in_channels
        # Create a sequential container to hold the layers
        self.layers = nn.Sequential()
        # Iterate over the specified depth to create layers
        for layer_idx in range(depth):
            # Get updated hyper-parameters for the current layer
            stride, drop_path_rate, is_first_layer = self._get_updated_hyperparameters(
                layer_idx, stride, layer_dropout
            )

            # Add the layer to the sequential container
            self.layers.add_module(
                str(layer_idx),
                layer_cls(
                    config,
                    prev_chs,
                    out_channels,
                    stride=stride,
                    dilation=dilation,
                    bottle_ratio=bottle_ratio,
                    first_dilation=first_dilation,
                    drop_path_rate=drop_path_rate,
                    is_first_layer=is_first_layer,
                ),
            )
            # Update previous channels for the next layer
            prev_chs = out_channels
            first_dilation = dilation

    def _get_updated_hyperparameters(self, layer_idx, stride, layer_dropout):
        """
        Get the new hyper-parameters with respect to the previous ones and the index of the current layer.
        """
        # Determine drop path rate if layer dropout is specified
        if layer_dropout:
            drop_path_rate = layer_dropout[layer_idx]
        else:
            drop_path_rate = 0.0

        # If not the first layer, set stride to 1
        if layer_idx != 0:
            stride = 1

        # Determine if the current layer is the first layer
        is_first_layer = layer_idx == 0

        return stride, drop_path_rate, is_first_layer

    def forward(self, input: Tensor) -> Tensor:
        # Forward pass through the layers sequentially
        hidden_state = input
        for _, layer in enumerate(self.layers):
            hidden_state = layer(hidden_state)
        return hidden_state


class BitEncoder(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config: BitConfig):
        # 调用父类初始化方法
        super().__init__()
        # 初始化阶段列表
        self.stages = nn.ModuleList([])

        # 初始化输入通道数为嵌入尺寸
        prev_chs = config.embedding_size

        # 这些需要硬编码
        current_stride = 4  # 当前步幅为4
        dilation = 1  # 膨胀率为1

        # 计算每个阶段的丢弃率
        layer_dropouts = [
            x.tolist()
            for x in torch.Tensor(np.linspace(0, config.drop_path_rate, sum(config.depths))).split(config.depths)
        ]

        # 遍历配置中的深度、隐藏大小和丢弃率
        for stage_idx, (current_depth, current_hidden_size, layer_dropout) in enumerate(
            zip(config.depths, config.hidden_sizes, layer_dropouts)
        ):
            # 获取更新后的超参数
            out_channels, stride, dilation = self._get_updated_hyperparameters(
                stage_idx, current_stride, current_hidden_size, dilation, config
            )

            # 创建 BitStage 模块
            stage = BitStage(
                config,
                prev_chs,
                out_channels,
                stride=stride,
                dilation=dilation,
                depth=current_depth,
                layer_dropout=layer_dropout,
            )

            # 更新输入通道数
            prev_chs = out_channels
            # 更新当前步幅
            current_stride *= stride

            # 将阶段模块添加到阶段列表中
            self.stages.add_module(str(stage_idx), stage)

    # 获取更新后的超参数
    def _get_updated_hyperparameters(self, stage_idx, current_stride, current_hidden_size, dilation, config):
        # 计算输出通道数
        out_channels = make_div(current_hidden_size * config.width_factor)
        # 如果是第一个阶段，则步幅为1，否则为2
        stride = 1 if stage_idx == 0 else 2
        # 如果当前步幅大于等于输出步幅，则更新膨胀率和步幅
        if current_stride >= config.output_stride:
            dilation *= stride
            stride = 1
        return out_channels, stride, dilation

    # 前向传播方法
    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        # 如果需要输出隐藏状态，则初始化一个空元组
        hidden_states = () if output_hidden_states else None

        # 遍历所有阶段模块
        for stage_module in self.stages:
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到隐藏状态元组中
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            # 将当前隐藏状态传递给阶段模块
            hidden_state = stage_module(hidden_state)

        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        # 如果不需要返回字典，则返回隐藏状态和隐藏状态元组中不为空的值
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回一个带有最后隐藏状态和隐藏状态元组的基本模型输出
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )
class BitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # BitPreTrainedModel 类的配置类为 BitConfig
    config_class = BitConfig
    # 模型的基本名称前缀为 "bit"
    base_model_prefix = "bit"
    # 主输入的名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化模型权重的函数
    def _init_weights(self, module):
        # 如果是卷积层，则使用 Kaiming 初始化权重
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # 如果是批标准化层或分组标准化层，则将权重初始化为1，偏置初始化为0
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


BIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`BitImageProcessor.__call__`]
            for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare BiT model outputting raw features without any specific head on top.",
    BIT_START_DOCSTRING,
)
# BitModel 类继承自 BitPreTrainedModel 类
class BitModel(BitPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 创建 BitEmbeddings 对象
        self.embedder = BitEmbeddings(config)

        # 创建 BitEncoder 对象
        self.encoder = BitEncoder(config)
        # 根据配置选择是否添加标准化层
        self.norm = (
            BitGroupNormActivation(config, num_channels=config.hidden_sizes[-1])
            if config.layer_type == "preactivation"
            else nn.Identity()
        )

        # 创建自适应平均池化层
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 前向传播函数
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    # 定义函数的返回类型为BaseModelOutputWithPoolingAndNoAttention
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        # 如果output_hidden_states为None，则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict为None，则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用embedder对pixel_values进行嵌入
        embedding_output = self.embedder(pixel_values)

        # 使用encoder对嵌入的输出进行编码
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        # 获取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后隐藏状态进行规范化
        last_hidden_state = self.norm(last_hidden_state)

        # 使用pooler对最后隐藏状态进行池化
        pooled_output = self.pooler(last_hidden_state)

        # 如果return_dict为False，则返回元组(last_hidden_state, pooled_output)和encoder_outputs[1:]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果return_dict为True，则返回BaseModelOutputWithPoolingAndNoAttention对象
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 使用 BiT 模型进行图像分类，包含一个线性层作为分类头部，例如用于 ImageNet 数据集
@add_start_docstrings(
    """
    BiT Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    BIT_START_DOCSTRING,
)
class BitForImageClassification(BitPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 BiT 模型
        self.bit = BitModel(config)
        # 分类头部
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 如果有标签数量，则创建线性层，否则创建恒等映射
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(BIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用 return_dict，否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Bit model 进行前向传播
        outputs = self.bit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 False，则使用 outputs[1] 作为 pooled_output，否则使用 outputs.pooler_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对 pooled_output 进行分类
        logits = self.classifier(pooled_output)

        loss = None

        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 根据问题类型确定损失函数
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
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

        # 如果 return_dict 为 False，则返回 logits 和 outputs[2:]，否则返回损失和 logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        # 返回 ImageClassifierOutputWithNoAttention 对象
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
# 导入必要的库函数和类
@add_start_docstrings(
    """
    BiT backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    BIT_START_DOCSTRING,
)
class BitBackbone(BitPreTrainedModel, BackboneMixin):
    # 初始化 BitBackbone 类
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 初始化网络的骨干部分
        super()._init_backbone(config)

        # 创建 BitModel 实例作为骨干网络的主体
        self.bit = BitModel(config)
        # 计算输出特征的维度列表
        self.num_features = [config.embedding_size] + config.hidden_sizes

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(BIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("google/resnetnv2-50")
        >>> model = AutoBackbone.from_pretrained("google/resnetnv2-50")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```py"""
        # 设置是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 执行 BitModel 的前向传播，并返回隐藏状态
        outputs = self.bit(pixel_values, output_hidden_states=True, return_dict=True)

        # 提取隐藏状态
        hidden_states = outputs.hidden_states

        # 定义特征图元组
        feature_maps = ()
        # 遍历网络各个阶段的名称和隐藏状态
        for idx, stage in enumerate(self.stage_names):
            # 如果阶段在输出特征中
            if stage in self.out_features:
                # 将隐藏状态添加到特征图中
                feature_maps += (hidden_states[idx],)

        # 如果不返回字典
        if not return_dict:
            # 构建输出
            output = (feature_maps,)
            # 如果需要输出隐藏状态，则添加隐藏状态到输出中
            if output_hidden_states:
                output += (outputs.hidden_states,)
            # 返回输出
            return output

        # 如果返回字典
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```