# `.\PaddleOCR\ppocr\modeling\backbones\det_resnet.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

# 导入必要的库
from paddle.vision.ops import DeformConv2D
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant, XavierUniform
from .det_resnet_vd import DeformableConvV2, ConvBNLayer

# 定义 BottleneckBlock 类
class BottleneckBlock(nn.Layer):
    # 定义 BottleneckBlock 类，继承自父类
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 is_dcn=False):
        # 调用父类的初始化方法
        super(BottleneckBlock, self).__init__()
    
        # 创建第一个卷积层，包括通道数、滤波器数、卷积核大小等参数
        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            act="relu", )
        # 创建第二个卷积层，包括通道数、滤波器数、卷积核大小等参数
        self.conv1 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=1, )
        # 创建第三个卷积层，包括通道数、滤波器数、卷积核大小等参数
        self.conv2 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            act=None, )
    
        # 如果没有快捷连接，则创建一个快捷连接层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                stride=stride, )
    
        # 设置是否使用快捷连接
        self.shortcut = shortcut
    
        # 记录输出通道数
        self._num_channels_out = num_filters * 4
    
    # 定义前向传播方法
    def forward(self, inputs):
        # 第一个卷积层
        y = self.conv0(inputs)
        # 第二个卷积层
        conv1 = self.conv1(y)
        # 第三个卷积层
        conv2 = self.conv2(conv1)
    
        # 如果使用快捷连接，则直接将输入作为快捷连接
        if self.shortcut:
            short = inputs
        # 否则通过快捷连接层计算快捷连接
        else:
            short = self.short(inputs)
    
        # 将快捷连接和卷积结果相加
        y = paddle.add(x=short, y=conv2)
        # 使用 ReLU 激活函数
        y = F.relu(y)
        # 返回结果
        return y
# 定义一个基本的残差块类，继承自 nn.Layer 类
class BasicBlock(nn.Layer):
    # 初始化函数，接受输入通道数、输出通道数、步长、是否使用快捷连接和名称作为参数
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__()
        # 设置步长
        self.stride = stride
        # 创建第一个卷积层和批归一化层
        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu")
        # 创建第二个卷积层和批归一化层
        self.conv1 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            act=None)

        # 如果不使用快捷连接，则创建一个额外的卷积层和批归一化层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                stride=stride)

        # 设置是否使用快捷连接
        self.shortcut = shortcut

    # 前向传播函数，接受输入数据
    def forward(self, inputs):
        # 第一个卷积层
        y = self.conv0(inputs)
        # 第二个卷积层
        conv1 = self.conv1(y)

        # 如果使用快捷连接，则直接将输入作为快捷连接
        if self.shortcut:
            short = inputs
        # 如果不使用快捷连接，则通过额外的卷积层和批归一化层得到快捷连接
        else:
            short = self.short(inputs)
        # 将快捷连接和卷积结果相加
        y = paddle.add(x=short, y=conv1)
        # 使用 ReLU 激活函数
        y = F.relu(y)
        # 返回结果
        return y

# 定义一个 ResNet 类，继承自 nn.Layer 类
class ResNet(nn.Layer):
    # 前向传播函数，接受输入数据
    def forward(self, inputs):
        # 经过卷积层
        y = self.conv(inputs)
        # 最大池化层
        y = self.pool2d_max(y)
        # 存储每个阶段的输出
        out = []
        # 遍历每个残差块
        for i, block in enumerate(self.stages):
            # 经过残差块
            y = block(y)
            # 如果当前残差块在输出索引列表中，则将其输出添加到结果列表中
            if i in self.out_indices:
                out.append(y)
        # 返回结果列表
        return out
```