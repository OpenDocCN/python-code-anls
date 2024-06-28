# `.\models\bit\modeling_bit.py`

```
# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BiT model. Also supports backbone for ViT hybrid."""

import collections
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "BitConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/bit-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/bit-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

BIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/bit-50",
    # See all BiT models at https://huggingface.co/models?filter=bit
]


def get_padding_value(padding=None, kernel_size=7, stride=1, dilation=1) -> Tuple[Tuple, bool]:
    r"""
    Utility function to get the tuple padding value given the kernel_size and padding.

    Args:
        padding (Union[`str`, `int`], *optional*):
            Padding value, can be either `"same"`, `"valid"`. If a different value is provided the default padding from
            PyTorch is used.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size of the convolution layers.
        stride (`int`, *optional*, defaults to 1):
            Stride value of the convolution layers.
        dilation (`int`, *optional*, defaults to 1):
            Dilation value of the convolution layers.
    """
    # Determine if padding should be dynamically calculated
    dynamic = False
    # If padding is not provided, calculate it based on convolution parameters
    if padding is None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    # Return the calculated padding and a boolean indicating if it's dynamic
    return padding, dynamic
    # 如果 padding 是字符串类型，将其转换为小写
    if isinstance(padding, str):
        # 如果 padding 是字符串 "same"
        padding = padding.lower()
        if padding == "same":
            # 对于 TF 兼容的 'SAME' padding，会影响性能和 GPU 内存分配
            # 当 stride 等于 1 并且 (dilation * (kernel_size - 1)) % 2 等于 0 时，静态情况下没有额外开销
            if stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0:
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
            else:
                # 动态 'SAME' padding，会有运行时和 GPU 内存的额外开销
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding，相当于 padding=0
            padding = 0
        else:
            # 默认使用类似 PyTorch 风格的对称 'same' padding
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    # 返回计算后的 padding 值和 dynamic 变量
    return padding, dynamic
# 定义一个继承自 nn.Conv2d 的类，用于实现带有权重标准化的二维卷积操作
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
        # 根据 padding 参数获取填充值及是否动态计算标志
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        # 调用父类的初始化方法，设置卷积层的各个参数
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
        # 根据是否动态填充选择相应的填充方法
        if is_dynamic:
            self.pad = DynamicPad2d(kernel_size, stride, dilation)
        else:
            self.pad = None
        self.eps = eps

    def forward(self, hidden_state):
        # 如果存在动态填充方法，则对输入进行填充操作
        if self.pad is not None:
            hidden_state = self.pad(hidden_state)
        # 对卷积核进行权重标准化操作
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, training=True, momentum=0.0, eps=self.eps
        ).reshape_as(self.weight)
        # 执行卷积操作，并返回处理后的 hidden_state
        hidden_state = nn.functional.conv2d(
            hidden_state, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return hidden_state


class BitGroupNormActivation(nn.GroupNorm):
    r"""
    A module that combines group normalization with an activation function.
    """

    def __init__(self, config, num_channels, eps=1e-5, affine=True, apply_activation=True):
        # 调用父类 nn.GroupNorm 的初始化方法，设置组归一化的参数
        super(BitGroupNormActivation, self).__init__(config.num_groups, num_channels, eps=eps, affine=affine)
        # 根据 apply_activation 参数选择激活函数
        if apply_activation:
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = nn.Identity()

    def forward(self, hidden_state):
        # 执行组归一化操作，并应用选择的激活函数
        hidden_state = nn.functional.group_norm(hidden_state, self.num_groups, self.weight, self.bias, self.eps)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class DynamicPad2d(nn.Module):
    r"""
    A module that wraps dynamic padding of any input, given the parameters of the convolutional layer and the input
    hidden states.
    """

    # 此处省略了初始化方法的注释，根据示例中不需要对该部分进行注释
    def __init__(self, kernel_size, stride, dilation, value=0):
        super().__init__()
        # Safety checkers
        # 如果 kernel_size 是整数，则转换为元组 (kernel_size, kernel_size)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # 如果 stride 是整数，则转换为元组 (stride, stride)
        if isinstance(stride, int):
            stride = (stride, stride)

        # 如果 dilation 是整数，则转换为元组 (dilation, dilation)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        # 将参数存储到对象的属性中
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.value = value

        # 定义一个内部方法 compute_padding，用于计算填充值
        def compute_padding(x, kernel_size, stride, dilation):
            return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

        # 将 compute_padding 方法存储到对象的属性中
        self.compute_padding = compute_padding

    def __call__(self, input):
        # 获取输入张量的高度和宽度
        input_height, input_width = input.size()[-2:]

        # 计算高度方向的填充值
        padding_height = self.compute_padding(input_height, self.kernel_size[0], self.stride[0], self.dilation[0])

        # 计算宽度方向的填充值
        padding_width = self.compute_padding(input_width, self.kernel_size[1], self.stride[1], self.dilation[1])

        # 如果需要进行填充
        if padding_height > 0 or padding_width > 0:
            # 使用 nn.functional.pad 对输入张量进行填充
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
        # 返回填充后的输入张量
        return input
# 定义一个自定义的2D最大池化层，实现类似于TensorFlow中'SAME'的功能
class BitMaxPool2d(nn.MaxPool2d):
    """Tensorflow like 'SAME' wrapper for 2D max pooling"""

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
        # 如果kernel_size不是可迭代对象，则转换为元组形式
        kernel_size = kernel_size if isinstance(kernel_size, collections.abc.Iterable) else (kernel_size, kernel_size)
        # 如果stride不是可迭代对象，则转换为元组形式
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        # 如果dilation不是可迭代对象，则转换为元组形式
        dilation = dilation if isinstance(dilation, collections.abc.Iterable) else (dilation, dilation)
        # 调用父类的初始化方法，设置kernel_size, stride, padding, dilation, ceil_mode
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode)
        # 如果使用动态填充
        if use_dynamic_padding:
            # 初始化一个动态填充对象DynamicPad2d，并赋值给self.pad
            self.pad = DynamicPad2d(kernel_size, stride, dilation, padding_value)
        else:
            # 否则使用nn.Identity()作为填充层
            self.pad = nn.Identity()

    def forward(self, hidden_states):
        # 对输入的hidden_states进行填充
        hidden_states = self.pad(hidden_states)
        # 使用nn.functional.max_pool2d进行最大池化操作，返回池化后的结果
        return nn.functional.max_pool2d(
            hidden_states, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode
        )


class BitEmbeddings(nn.Module):
    """
    BiT Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: BitConfig):
        # 调用父类的初始化方法
        super().__init__()

        # 定义一个权重标准化的2D卷积层，作为BiT嵌入的第一层
        self.convolution = WeightStandardizedConv2d(
            config.num_channels,
            config.embedding_size,
            kernel_size=7,
            stride=2,
            eps=1e-8,
            padding=config.global_padding,
        )

        # 定义一个BitMaxPool2d对象作为池化层，处理卷积层输出的特征图
        self.pooler = BitMaxPool2d(kernel_size=3, stride=2, use_dynamic_padding=config.embedding_dynamic_padding)

        # 如果全局填充策略为'SAME'，则使用nn.Identity()作为填充层，否则使用常数填充
        if config.global_padding is not None and config.global_padding.upper() == "SAME":
            self.pad = nn.Identity()
        else:
            self.pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0)

        # 如果层类型不是'preactivation'，则使用BitGroupNormActivation进行归一化和激活，否则使用nn.Identity()
        if not config.layer_type == "preactivation":
            self.norm = BitGroupNormActivation(config, num_channels=config.embedding_size)
        else:
            self.norm = nn.Identity()

        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        # 检查输入的通道维度是否与配置中设置的通道数相匹配
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 对输入的像素值进行卷积操作，得到嵌入表示
        embedding = self.convolution(pixel_values)

        # 对卷积层的输出进行填充
        embedding = self.pad(embedding)

        # 对填充后的特征图进行归一化和激活处理
        embedding = self.norm(embedding)

        # 对归一化后的特征图进行池化操作
        embedding = self.pooler(embedding)

        # 返回最终的嵌入表示
        return embedding


# 从transformers.models.convnext.modeling_convnext.drop_path中复制的drop_path函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    # 省略部分，实现dropout功能，但此处未提供完整实现，仅有文档字符串和函数声明
    # 如果 drop_prob 为 0 或者不处于训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 确定随机张量的形状，保证适用于不同维度的张量，而不仅仅是二维卷积网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成与输入张量相同设备和数据类型的随机张量
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 计算输出值，通过随机张量进行 Dropout 操作
    output = input.div(keep_prob) * random_tensor
    # 返回 Dropout 后的输出张量
    return output
# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Bit
class BitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob  # 初始化时设置 dropout 的概率

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数对输入的 hidden_states 进行随机深度(drop path)操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回模块的额外描述信息，包括 dropout 的概率
        return "p={}".format(self.drop_prob)


def make_div(value, divisor=8):
    # 计算 value 的最接近的大于 divisor 的整数倍的数
    min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class BitPreActivationBottleneckLayer(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
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

        if is_first_layer:
            # 如果是第一层，则初始化 downsample 为 BitDownsampleConv 对象
            self.downsample = BitDownsampleConv(
                config,
                in_channels,
                out_channels,
                stride=stride,
                preact=True,
            )
        else:
            self.downsample = None  # 否则 downsample 为 None

        self.norm1 = BitGroupNormActivation(config, in_channels)  # 第一个规范化与激活层
        self.conv1 = WeightStandardizedConv2d(in_channels, mid_channels, 1, eps=1e-8, padding=config.global_padding)  # 第一个卷积层

        self.norm2 = BitGroupNormActivation(config, num_channels=mid_channels)  # 第二个规范化与激活层
        self.conv2 = WeightStandardizedConv2d(
            mid_channels, mid_channels, 3, stride=stride, groups=groups, eps=1e-8, padding=config.global_padding
        )  # 第二个卷积层，带有可能的步幅和分组设置

        self.norm3 = BitGroupNormActivation(config, mid_channels)  # 第三个规范化与激活层
        self.conv3 = WeightStandardizedConv2d(mid_channels, out_channels, 1, eps=1e-8, padding=config.global_padding)  # 第三个卷积层

        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()  # 设置随机深度(drop path)模块
    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化处理
        hidden_states_preact = self.norm1(hidden_states)

        # 生成快捷路径分支
        shortcut = hidden_states
        # 如果定义了下采样函数，则对层归一化后的隐藏状态进行下采样
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states_preact)

        # 残差分支
        # 对归一化后的隐藏状态进行第一次卷积操作
        hidden_states = self.conv1(hidden_states_preact)
        # 对第一次卷积后的结果进行第二次卷积和归一化处理
        hidden_states = self.conv2(self.norm2(hidden_states))
        # 对第二次卷积后的结果进行第三次卷积和归一化处理
        hidden_states = self.conv3(self.norm3(hidden_states))
        # 执行 dropout 路径
        hidden_states = self.drop_path(hidden_states)
        # 将残差分支的输出与快捷路径的输出相加作为最终的隐藏状态输出
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
        first_dilation = first_dilation or dilation  # 如果未指定，则使用 dilation 参数的值作为 first_dilation

        out_channels = out_channels or in_channels  # 如果未指定输出通道数，则使用输入通道数
        mid_chs = make_div(out_channels * bottle_ratio)  # 计算中间通道数，通过 make_div 函数调整为可被整除的数

        if is_first_layer:
            # 如果是第一层，则使用 BitDownsampleConv 类进行下采样
            self.downsample = BitDownsampleConv(
                config,
                in_channels,
                out_channels,
                stride=stride,
                preact=False,  # 不进行预激活
            )
        else:
            self.downsample = None  # 否则不进行下采样

        # 第一层卷积
        self.conv1 = WeightStandardizedConv2d(in_channels, mid_chs, 1, eps=1e-8, padding=config.global_padding)
        self.norm1 = BitGroupNormActivation(config, num_channels=mid_chs)  # 中间层的规范化和激活

        # 第二层卷积
        self.conv2 = WeightStandardizedConv2d(
            mid_chs,
            mid_chs,
            3,
            stride=stride,
            dilation=first_dilation,
            groups=groups,
            eps=1e-8,
            padding=config.global_padding,
        )
        self.norm2 = BitGroupNormActivation(config, num_channels=mid_chs)  # 第二层的规范化和激活

        # 第三层卷积
        self.conv3 = WeightStandardizedConv2d(mid_chs, out_channels, 1, eps=1e-8, padding=config.global_padding)
        self.norm3 = BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)  # 输出层的规范化，不进行激活

        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()  # Dropout 路径

        self.activation = ACT2FN[config.hidden_act]  # 激活函数选择

    def forward(self, hidden_states):
        # shortcut 分支，即残差连接
        shortcut = hidden_states
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states)  # 如果有下采样，则应用到 shortcut 上

        # 残差连接
        hidden_states = self.conv1(hidden_states)  # 第一层卷积
        hidden_states = self.norm1(hidden_states)  # 第一层规范化和激活

        hidden_states = self.conv2(hidden_states)  # 第二层卷积
        hidden_states = self.norm2(hidden_states)  # 第二层规范化和激活

        hidden_states = self.conv3(hidden_states)  # 第三层卷积
        hidden_states = self.norm3(hidden_states)  # 输出层规范化，不进行激活

        hidden_states = self.drop_path(hidden_states)  # 应用 dropout 路径

        hidden_states = self.activation(hidden_states + shortcut)  # 加上残差连接后应用激活函数
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
        self.conv = WeightStandardizedConv2d(
            in_channels, out_channels, 1, stride=stride, eps=1e-8, padding=config.global_padding
        )
        self.norm = (
            nn.Identity()
            if preact
            else BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)
        )  # 如果 preact 是 True，则使用 nn.Identity()，否则使用 BitGroupNormActivation 进行规范化
    # 定义一个前向传播的方法，接受输入 x
    def forward(self, x):
        # 先将输入 x 经过卷积操作 conv
        conv_output = self.conv(x)
        # 然后对卷积输出进行归一化操作 norm
        normalized_output = self.norm(conv_output)
        # 返回归一化后的结果
        return normalized_output
# 定义一个 ResNet v2 阶段，由堆叠的层组成
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

        # 根据 dilation 参数设置第一个层的扩张率
        first_dilation = 1 if dilation in (1, 2) else 2

        # 根据配置选择使用 Bottleneck 层或者 PreActivationBottleneckLayer 层
        if config.layer_type == "bottleneck":
            layer_cls = BitBottleneckLayer
        else:
            layer_cls = BitPreActivationBottleneckLayer

        prev_chs = in_channels
        self.layers = nn.Sequential()
        # 创建指定深度的层堆叠
        for layer_idx in range(depth):
            # 获取当前层的超参数
            stride, drop_path_rate, is_first_layer = self._get_updated_hyperparameters(
                layer_idx, stride, layer_dropout
            )

            # 将当前层添加到层序列中
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
            prev_chs = out_channels
            first_dilation = dilation

    # 获取更新后的超参数的内部方法
    def _get_updated_hyperparameters(self, layer_idx, stride, layer_dropout):
        """
        Get the new hyper-parameters with respect to the previous ones and the index of the current layer.
        """
        # 如果存在层的 dropout 设置，则获取当前层的 dropout rate
        if layer_dropout:
            drop_path_rate = layer_dropout[layer_idx]
        else:
            drop_path_rate = 0.0

        # 如果不是第一层，则将 stride 设置为 1
        if layer_idx != 0:
            stride = 1

        # 判断当前层是否是第一层
        is_first_layer = layer_idx == 0

        return stride, drop_path_rate, is_first_layer

    # 前向传播方法，遍历每一层并依次传递输入
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for _, layer in enumerate(self.layers):
            hidden_state = layer(hidden_state)
        return hidden_state
    # 初始化方法，接受一个BitConfig类型的配置对象作为参数
    def __init__(self, config: BitConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个空的神经网络模块列表
        self.stages = nn.ModuleList([])

        # 初始通道数设为配置对象中的嵌入大小
        prev_chs = config.embedding_size

        # 固定设定的当前步幅为4，膨胀率为1
        current_stride = 4
        dilation = 1

        # 计算每个层的丢弃率列表
        layer_dropouts = [
            x.tolist()
            for x in torch.Tensor(np.linspace(0, config.drop_path_rate, sum(config.depths))).split(config.depths)
        ]

        # 遍历每个阶段，其中current_depth为层深度，current_hidden_size为隐藏层大小，layer_dropout为层丢弃率
        for stage_idx, (current_depth, current_hidden_size, layer_dropout) in enumerate(
            zip(config.depths, config.hidden_sizes, layer_dropouts)
        ):
            # 获取更新后的超参数
            out_channels, stride, dilation = self._get_updated_hyperparameters(
                stage_idx, current_stride, current_hidden_size, dilation, config
            )

            # 创建BitStage模块，并添加到self.stages中
            stage = BitStage(
                config,
                prev_chs,
                out_channels,
                stride=stride,
                dilation=dilation,
                depth=current_depth,
                layer_dropout=layer_dropout,
            )

            prev_chs = out_channels
            current_stride *= stride

            self.stages.add_module(str(stage_idx), stage)

    # 获取更新后的超参数方法
    def _get_updated_hyperparameters(self, stage_idx, current_stride, current_hidden_size, dilation, config):
        # 计算输出通道数，确保是可被整除的
        out_channels = make_div(current_hidden_size * config.width_factor)
        # 首个阶段步幅设为1，其余为2
        stride = 1 if stage_idx == 0 else 2
        # 若当前步幅超过设定的输出步幅，则调整膨胀率和步幅
        if current_stride >= config.output_stride:
            dilation *= stride
            stride = 1
        # 返回更新后的输出通道数、步幅和膨胀率
        return out_channels, stride, dilation

    # 前向传播方法
    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        # 如果需要输出隐藏状态，则初始化一个空的隐藏状态元组
        hidden_states = () if output_hidden_states else None

        # 遍历每个BitStage模块进行前向传播
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        # 如果不需要以字典形式返回结果，则按需返回隐藏状态和隐藏状态元组
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 以BaseModelOutputWithNoAttention类的实例形式返回结果
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )
@add_start_docstrings(
    "The bare BiT model outputting raw features without any specific head on top.",
    BIT_START_DOCSTRING,
)
class BitModel(BitPreTrainedModel):
    """
    BiT 模型的抽象类，负责权重初始化、预训练模型的下载和加载接口。
    """

    def __init__(self, config):
        """
        初始化函数，设置模型结构及参数。

        Args:
            config (BitConfig): 模型的配置类，包含模型的所有参数。

        Attributes:
            embedder (BitEmbeddings): BiT 模型的嵌入层。
            encoder (BitEncoder): BiT 模型的编码器。
            norm (nn.Module): 规范化层，根据配置决定是分组规范化还是身份映射。
            pooler (nn.Module): 自适应平均池化层，用于汇总特征。
        """
        super().__init__(config)
        self.config = config

        self.embedder = BitEmbeddings(config)
        self.encoder = BitEncoder(config)
        self.norm = (
            BitGroupNormActivation(config, num_channels=config.hidden_sizes[-1])
            if config.layer_type == "preactivation"
            else nn.Identity()
        )
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
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> Union[ModelOutput, Tuple[Tensor]]:
        """
        BiT 模型的前向传播函数。

        Args:
            pixel_values (Tensor): 输入的像素值张量，形状为(batch_size, num_channels, height, width)。
            output_hidden_states (bool, optional): 是否返回所有层的隐藏状态。
            return_dict (bool, optional): 是否返回一个 ModelOutput 而不是普通元组。

        Returns:
            Union[ModelOutput, Tuple[Tensor]]: 根据 return_dict 参数返回不同形式的输出。

        Raises:
            NotImplementedError: 如果未指定返回格式，则抛出错误。
        """
        raise NotImplementedError
    # 定义方法的返回类型为BaseModelOutputWithPoolingAndNoAttention
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        # 如果output_hidden_states参数不为None，则使用传入的值；否则使用self.config.output_hidden_states的默认配置值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict参数不为None，则使用传入的值；否则使用self.config.use_return_dict的默认配置值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用embedder方法将pixel_values转换为嵌入表示
        embedding_output = self.embedder(pixel_values)

        # 使用encoder方法对嵌入表示进行编码，根据参数output_hidden_states和return_dict是否返回字典
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        # 获取编码器输出的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后一个隐藏状态进行归一化处理
        last_hidden_state = self.norm(last_hidden_state)

        # 使用pooler方法对归一化后的最后一个隐藏状态进行池化
        pooled_output = self.pooler(last_hidden_state)

        # 如果return_dict为False，则返回一个元组，包含最后一个隐藏状态、池化输出以及其他编码器输出的隐藏状态列表
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则，返回一个BaseModelOutputWithPoolingAndNoAttention对象，包含最后一个隐藏状态、池化输出以及所有隐藏状态的列表
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 使用装饰器为类添加文档字符串，描述这是一个在图像分类任务上使用的 BiT 模型，例如用于 ImageNet
@add_start_docstrings(
    """
    BiT Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    BIT_START_DOCSTRING,  # 引用全局定义的 BiT 模型文档字符串模板
)
class BitForImageClassification(BitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量
        self.bit = BitModel(config)  # 创建 BiT 模型实例
        # 分类头部，根据配置决定输出层结构
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 将输入展平
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),  # 添加线性分类层或者恒等映射层
        )
        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为前向传播函数添加模型文档字符串，描述输入和输出的格式
    @add_start_docstrings_to_model_forward(BIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,  # 引用全局的模型检查点信息
        output_type=ImageClassifierOutputWithNoAttention,  # 引用全局的输出类型定义
        config_class=_CONFIG_FOR_DOC,  # 引用全局的配置类信息
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,  # 引用全局的预期输出信息
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 输入像素值的张量，可选
        labels: Optional[torch.LongTensor] = None,  # 真实标签的张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的张量，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选
        # 注意：函数定义没有被完全展示，继续在下文中
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 `return_dict` 不为 None，则使用它；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.bit 方法，传入像素值 `pixel_values`，根据 `output_hidden_states` 和 `return_dict` 的值返回输出
        outputs = self.bit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，则使用 outputs.pooler_output 作为 pooled_output，否则使用 outputs 的第二个元素
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将 pooled_output 输入分类器 self.classifier，得到 logits
        logits = self.classifier(pooled_output)

        # 初始化 loss 为 None
        loss = None

        # 如果 labels 不为 None，则计算损失函数
        if labels is not None:
            # 如果 self.config.problem_type 为 None，则根据情况设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单个标签的回归任务，计算 logits 和 labels 的均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算 logits 和 labels 的均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带 logits 的二进制交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则构建输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            # 返回损失与输出元组，如果损失不为 None，则包含损失
            return (loss,) + output if loss is not None else output

        # 返回 ImageClassifierOutputWithNoAttention 对象，包含损失、logits 和 hidden_states
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
@add_start_docstrings(
    """
    BiT backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    BIT_START_DOCSTRING,
)
class BitBackbone(BitPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 调用父类的初始化背骨方法
        super()._init_backbone(config)

        # 创建 BiT 模型实例
        self.bit = BitModel(config)
        # 计算特征维度列表，包括嵌入大小和隐藏层大小
        self.num_features = [config.embedding_size] + config.hidden_sizes

        # 初始化权重并应用最终处理
        self.post_init()

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
        ```"""
        # 如果未提供返回字典参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果未提供输出隐藏状态参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 调用 BiT 模型的前向传播，返回输出结果
        outputs = self.bit(pixel_values, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            # 如果当前阶段在输出特征名称列表中
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        # 如果不要求返回字典，则返回元组形式的输出
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回包含特征映射、隐藏状态（如果需要）、注意力（默认为None）的 BackboneOutput 对象
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```