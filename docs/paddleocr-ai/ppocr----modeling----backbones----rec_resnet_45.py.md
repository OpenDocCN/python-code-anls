# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_45.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
此代码参考自：
https://github.com/FangShancheng/ABINet/tree/main/modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math

__all__ = ["ResNet45"]


def conv1x1(in_planes, out_planes, stride=1):
    # 创建一个 1x1 的卷积层
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        weight_attr=ParamAttr(initializer=KaimingNormal()),
        bias_attr=False)


def conv3x3(in_channel, out_channel, stride=1):
    # 创建一个 3x3 的卷积层
    return nn.Conv2D(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=ParamAttr(initializer=KaimingNormal()),
        bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        # 第一个卷积层，1x1 卷积
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2D(channels)
        self.relu = nn.ReLU()
        # 第二个卷积层，3x3 卷积
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2D(channels)
        self.downsample = downsample
        self.stride = stride
    # 定义神经网络的前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 保存为残差项
        residual = x

        # 第一层卷积操作
        out = self.conv1(x)
        # 对第一层卷积结果进行批量归一化
        out = self.bn1(out)
        # 对第一层卷积结果应用激活函数 ReLU
        out = self.relu(out)

        # 第二层卷积操作
        out = self.conv2(out)
        # 对第二层卷积结果进行批量归一化
        out = self.bn2(out)

        # 如果存在下采样操作
        if self.downsample is not None:
            # 则对输入 x 进行下采样
            residual = self.downsample(x)
        # 将残差项加到当前输出上
        out += residual
        # 对输出应用激活函数 ReLU
        out = self.relu(out)

        # 返回最终输出
        return out
# 定义 ResNet45 类，继承自 nn.Layer
class ResNet45(nn.Layer):
    # 初始化函数，设置默认参数值
    def __init__(self,
                 in_channels=3,
                 block=BasicBlock,
                 layers=[3, 4, 6, 6, 3],
                 strides=[2, 1, 2, 1, 1]):
        # 初始化输入通道数
        self.inplanes = 32
        # 调用父类的初始化函数
        super(ResNet45, self).__init__()
        # 创建第一个卷积层
        self.conv1 = nn.Conv2D(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        # 创建第一个批归一化层
        self.bn1 = nn.BatchNorm2D(32)
        # 创建激活函数层
        self.relu = nn.ReLU()

        # 创建 ResNet 的五个阶段
        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[3])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[4])
        # 设置输出通道数
        self.out_channels = 512

    # 定义创建 ResNet 层的函数
    def _make_layer(self, block, planes, blocks, stride=1):
        # 初始化下采样层
        downsample = None
        # 判断是否需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建下采样层
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion), )

        # 初始化层列表
        layers = []
        # 添加第一个 block
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新输入通道数
        self.inplanes = planes * block.expansion
        # 循环添加剩余的 block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 返回包含所有 block 的序列
        return nn.Sequential(*layers)
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 使用第一个卷积层对输入 x 进行卷积操作
        x = self.conv1(x)
        # 对卷积结果进行批量归一化处理
        x = self.bn1(x)
        # 对归一化后的结果进行激活函数处理
        x = self.relu(x)
        # 通过第一个残差块的层进行处理
        x = self.layer1(x)
        # 通过第二个残差块的层进行处理
        x = self.layer2(x)
        # 通过第三个残差块的层进行处理
        x = self.layer3(x)
        # 通过第四个残差块的层进行处理
        x = self.layer4(x)
        # 通过第五个残差块的层进行处理
        x = self.layer5(x)
        # 返回处理后的结果
        return x
```