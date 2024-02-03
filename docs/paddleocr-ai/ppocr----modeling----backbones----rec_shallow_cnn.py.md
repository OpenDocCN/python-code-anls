# `.\PaddleOCR\ppocr\modeling\backbones\rec_shallow_cnn.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
# 代码来源：
# https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/backbones/shallow_cnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import MaxPool2D
from paddle.nn.initializer import KaimingNormal, Uniform, Constant

# 定义 ConvBNLayer 类，包含卷积、批归一化和激活函数
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        # 创建卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充、分组数等参数
        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        # 创建批归一化层，设置输出通道数、权重初始化方式、偏置初始化方式
        self.bn = nn.BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(initializer=Uniform(0, 1)),
            bias_attr=ParamAttr(initializer=Constant(0)))
        # 创建激活函数层，使用 ReLU 激活函数
        self.relu = nn.ReLU()
    # 定义一个前向传播函数，接收输入并返回输出
    def forward(self, inputs):
        # 将输入数据通过卷积层处理得到输出
        y = self.conv(inputs)
        # 将卷积层的输出通过批量归一化层处理
        y = self.bn(y)
        # 将批量归一化层的输出通过激活函数处理
        y = self.relu(y)
        # 返回处理后的输出
        return y
# 定义一个浅层卷积神经网络类 ShallowCNN，继承自 nn.Layer
class ShallowCNN(nn.Layer):
    # 初始化函数，设置输入通道数和隐藏层维度
    def __init__(self, in_channels=1, hidden_dim=512):
        super().__init__()
        # 断言输入通道数和隐藏层维度为整数类型
        assert isinstance(in_channels, int)
        assert isinstance(hidden_dim, int)

        # 创建第一个卷积层对象，输入通道数为 in_channels，输出通道数为 hidden_dim // 2，卷积核大小为 3，步长为 1，填充为 1
        self.conv1 = ConvBNLayer(
            in_channels, 3, hidden_dim // 2, stride=1, padding=1)
        # 创建第二个卷积层对象，输入通道数为 hidden_dim // 2，输出通道数为 hidden_dim，卷积核大小为 3，步长为 1，填充为 1
        self.conv2 = ConvBNLayer(
            hidden_dim // 2, 3, hidden_dim, stride=1, padding=1)
        # 创建最大池化层对象，池化核大小为 2，步长为 2，填充为 0
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # 设置输出通道数为 hidden_dim

    # 前向传播函数
    def forward(self, x):
        # 第一个卷积层的前向传播
        x = self.conv1(x)
        # 最大池化层的前向传播
        x = self.pool(x)

        # 第二个卷积层的前向传播
        x = self.conv2(x)
        # 最大池化层的前向传播
        x = self.pool(x)

        # 返回结果
        return x
```