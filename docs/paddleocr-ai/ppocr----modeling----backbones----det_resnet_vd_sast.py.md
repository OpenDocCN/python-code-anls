# `.\PaddleOCR\ppocr\modeling\backbones\det_resnet_vd_sast.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发
# 没有任何形式的保证或条件，无论是明示的还是暗示的
# 有关特定语言的权限和限制，请参阅许可证
#
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

# 导出 ResNet_SAST 类
__all__ = ["ResNet_SAST"]

# 定义 ConvBNLayer 类，继承自 nn.Layer
class ConvBNLayer(nn.Layer):
    # 初始化卷积和批归一化层
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        # 调用父类的初始化方法
        super(ConvBNLayer, self).__init__()

        # 设置是否使用 VD 模式
        self.is_vd_mode = is_vd_mode
        # 创建 2D 平均池化层
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        # 根据名称设置批归一化层的参数
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # 创建批归一化层
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    # 前向传播函数
    def forward(self, inputs):
        # 如果使用 VD 模式，则对输入进行平均池化
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        # 经过卷积层
        y = self._conv(inputs)
        # 经过批归一化层
        y = self._batch_norm(y)
        return y
# 定义 BottleneckBlock 类，继承自 nn.Layer
class BottleneckBlock(nn.Layer):
    # 初始化方法，接受输入通道数、输出通道数、步长、是否使用快捷连接、是否为第一个块、名称等参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        # 调用父类的初始化方法
        super(BottleneckBlock, self).__init__()

        # 创建第一个卷积层，1x1卷积
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        # 创建第二个卷积层，3x3卷积
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        # 创建第三个卷积层，1x1卷积
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        # 如果不使用快捷连接，则创建额外的卷积层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        # 记录是否使用快捷连接
        self.shortcut = shortcut

    # 前向传播方法
    def forward(self, inputs):
        # 第一个卷积层
        y = self.conv0(inputs)
        # 第二个卷积层
        conv1 = self.conv1(y)
        # 第三个卷积层
        conv2 = self.conv2(conv1)

        # 判断是否使用快捷连接
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        # 快捷连接和卷积结果相加
        y = paddle.add(x=short, y=conv2)
        # 激活函数ReLU
        y = F.relu(y)
        # 返回结果
        return y


# 定义 BasicBlock 类
class BasicBlock(nn.Layer):
    # 初始化 BasicBlock 类，设置输入通道数、输出通道数、步长、是否使用 shortcut、是否为第一个 BasicBlock、名称
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        # 调用父类的初始化方法
        super(BasicBlock, self).__init__()
        # 设置步长
        self.stride = stride
        # 创建第一个卷积层和 BN 层
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        # 创建第二个卷积层和 BN 层
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")

        # 如果不使用 shortcut，则创建 shortcut 卷积层和 BN 层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        # 设置是否使用 shortcut
        self.shortcut = shortcut

    # 前向传播函数
    def forward(self, inputs):
        # 进行第一个卷积操作
        y = self.conv0(inputs)
        # 进行第二个卷积操作
        conv1 = self.conv1(y)

        # 如果使用 shortcut，则直接将输入作为 shortcut
        if self.shortcut:
            short = inputs
        # 如果不使用 shortcut，则通过 shortcut 卷积层和 BN 层得到 shortcut
        else:
            short = self.short(inputs)
        # 将 shortcut 和第二个卷积结果相加
        y = paddle.add(x=short, y=conv1)
        # 对相加结果进行激活函数处理
        y = F.relu(y)
        # 返回结果
        return y
# 定义一个名为 ResNet_SAST 的类，继承自 nn.Layer 类
class ResNet_SAST(nn.Layer):
    # 定义类的前向传播方法，接收输入参数 inputs
    def forward(self, inputs):
        # 初始化一个列表 out，将输入参数 inputs 添加到列表中
        out = [inputs]
        # 对输入参数 inputs 进行卷积操作 conv1_1
        y = self.conv1_1(inputs)
        # 对上一步的结果 y 进行卷积操作 conv1_2
        y = self.conv1_2(y)
        # 对上一步的结果 y 进行卷积操作 conv1_3
        y = self.conv1_3(y)
        # 将上一步的结果 y 添加到列表 out 中
        out.append(y)
        # 对结果 y 进行最大池化操作
        y = self.pool2d_max(y)
        # 遍历 self.stages 中的每个 block，对结果 y 进行操作并添加到列表 out 中
        for block in self.stages:
            y = block(y)
            out.append(y)
        # 返回列表 out
        return out
```