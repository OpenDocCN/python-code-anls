# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_31.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
"""
# 代码来源于以下链接:
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/layers/conv_layer.py
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/backbones/resnet31_ocr.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

# 定义可以导出的模块
__all__ = ["ResNet31"]

# 定义一个 3x3 卷积层
def conv3x3(in_channel, out_channel, stride=1, conv_weight_attr=None):
    return nn.Conv2D(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=conv_weight_attr,
        bias_attr=False)

# 定义基本的残差块
class BasicBlock(nn.Layer):
    expansion = 1
    # 初始化函数，定义了一个残差块，包括卷积层、批归一化层、激活函数等
    def __init__(self, in_channels, channels, stride=1, downsample=False, conv_weight_attr=None, bn_weight_attr=None):
        super().__init__()
        # 创建3x3卷积层，输入通道数为in_channels，输出通道数为channels，步长为stride
        self.conv1 = conv3x3(in_channels, channels, stride, 
            conv_weight_attr=conv_weight_attr)
        # 创建2D批归一化层，输入通道数为channels
        self.bn1 = nn.BatchNorm2D(channels, weight_attr=bn_weight_attr)
        # 创建ReLU激活函数
        self.relu = nn.ReLU()
        # 创建第二个3x3卷积层，输入通道数为channels，输出通道数为channels
        self.conv2 = conv3x3(channels, channels,
            conv_weight_attr=conv_weight_attr)
        # 创建第二个2D批归一化层，输入通道数为channels
        self.bn2 = nn.BatchNorm2D(channels, weight_attr=bn_weight_attr)
        # 是否进行下采样
        self.downsample = downsample
        # 如果需要下采样
        if downsample:
            # 创建下采样层，包括1x1卷积层和批归一化层
            self.downsample = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    channels * self.expansion,
                    1,
                    stride,
                    weight_attr=conv_weight_attr,
                    bias_attr=False),
                nn.BatchNorm2D(channels * self.expansion, weight_attr=bn_weight_attr))
        else:
            # 如果不需要下采样，则创建空的Sequential对象
            self.downsample = nn.Sequential()
        # 步长
        self.stride = stride

    # 前向传播函数
    def forward(self, x):
        # 保存输入x作为残差
        residual = x

        # 第一个卷积层、批归一化层、激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层、批归一化层
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，则对输入x进行下采样
        if self.downsample:
            residual = self.downsample(x)

        # 将残差加到输出上
        out += residual
        out = self.relu(out)

        return out
class ResNet31(nn.Layer):
    '''
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
        init_type (None | str): the config to control the initialization.
    '''
    # 定义 ResNet31 类，包含参数说明

    def _make_layer(self, input_channels, output_channels, blocks, conv_weight_attr=None, bn_weight_attr=None):
        # 定义内部方法 _make_layer，用于创建网络层
        layers = []
        # 初始化空列表 layers
        for _ in range(blocks):
            # 循环创建 blocks 个 BasicBlock
            downsample = None
            # 初始化 downsample 为 None
            if input_channels != output_channels:
                # 如果输入通道数不等于输出通道数
                downsample = nn.Sequential(
                    nn.Conv2D(
                        input_channels,
                        output_channels,
                        kernel_size=1,
                        stride=1,
                        weight_attr=conv_weight_attr,
                        bias_attr=False),
                    nn.BatchNorm2D(output_channels, weight_attr=bn_weight_attr))
                # 创建包含卷积层和批归一化层的 downsample

            layers.append(
                BasicBlock(
                    input_channels, output_channels, downsample=downsample, 
                    conv_weight_attr=conv_weight_attr, bn_weight_attr=bn_weight_attr))
            # 将 BasicBlock 添加到 layers 列表中
            input_channels = output_channels
            # 更新输入通道数为输出通道数
        return nn.Sequential(*layers)
        # 返回包含所有 BasicBlock 的序列
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 第一层卷积操作
        x = self.conv1_1(x)
        # 第一层批归一化操作
        x = self.bn1_1(x)
        # 第一层激活函数操作
        x = self.relu1_1(x)

        # 第二层卷积操作
        x = self.conv1_2(x)
        # 第二层批归一化操作
        x = self.bn1_2(x)
        # 第二层激活函数操作
        x = self.relu1_2(x)

        # 初始化一个空列表用于存储输出
        outs = []
        # 遍历四次
        for i in range(4):
            # 计算当前层的索引
            layer_index = i + 2
            # 获取当前层的池化层
            pool_layer = getattr(self, f'pool{layer_index}')
            # 获取当前层的块层
            block_layer = getattr(self, f'block{layer_index}')
            # 获取当前层的卷积层
            conv_layer = getattr(self, f'conv{layer_index}')
            # 获取当前层的批归一化层
            bn_layer = getattr(self, f'bn{layer_index}')
            # 获取当前层的激活函数层
            relu_layer = getattr(self, f'relu{layer_index}')

            # 如果当前层有池化层，则进行池化操作
            if pool_layer is not None:
                x = pool_layer(x)
            # 进行块层操作
            x = block_layer(x)
            # 进行卷积层操作
            x = conv_layer(x)
            # 进行批归一化操作
            x = bn_layer(x)
            # 进行激活函数操作
            x = relu_layer(x)

            # 将当前层的输出添加到输出列表中
            outs.append(x)

        # 如果指定了输出索引，则返回指定索引的输出
        if self.out_indices is not None:
            return tuple([outs[i] for i in self.out_indices])

        # 否则返回最后一层的输出
        return x
```