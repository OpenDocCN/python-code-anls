# `.\PaddleOCR\ppocr\modeling\backbones\e2e_resnet_vd_pg.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的保证或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 Paddle 库
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

# 定义可以导出的类
__all__ = ["ResNet"]

# 定义 ConvBNLayer 类，继承自 nn.Layer 类
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

        # 设置是否为 VD 模式
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
        # 根据卷积层名称确定批归一化层名称
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
        # 进行卷积操作
        y = self._conv(inputs)
        # 进行批归一化操作
        y = self._batch_norm(y)
        # 返回结果
        return y
# 定义 BottleneckBlock 类，继承自 nn.Layer 类
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

        # 创建 1x1 卷积层和 BN 层，用于降维
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        # 创建 3x3 卷积层和 BN 层
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        # 创建 1x1 卷积层和 BN 层，用于升维
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        # 如果不使用快捷连接，则创建额外的卷积层和 BN 层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        # 记录是否使用快捷连接
        self.shortcut = shortcut

    # 前向传播方法，接受输入数据
    def forward(self, inputs):
        # 进行第一个 1x1 卷积操作
        y = self.conv0(inputs)
        # 进行第二个 3x3 卷积操作
        conv1 = self.conv1(y)
        # 进行第三个 1x1 卷积操作
        conv2 = self.conv2(conv1)

        # 如果使用快捷连接，则直接将输入作为短接
        if self.shortcut:
            short = inputs
        # 如果不使用快捷连接，则通过额外的卷积层和 BN 层得到短接
        else:
            short = self.short(inputs)
        # 将短接和卷积结果相加
        y = paddle.add(x=short, y=conv2)
        # 使用 ReLU 激活函数
        y = F.relu(y)
        # 返回结果
        return y


# 定义 BasicBlock 类
class BasicBlock(nn.Layer):
    # 初始化函数，定义 BasicBlock 类的构造函数
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        # 调用父类的初始化函数
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

        # 如果没有快捷连接，则创建快捷连接
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        # 设置是否有快捷连接
        self.shortcut = shortcut

    # 前向传播函数
    def forward(self, inputs):
        # 第一个卷积层
        y = self.conv0(inputs)
        # 第二个卷积层
        conv1 = self.conv1(y)

        # 如果有快捷连接，则直接将输入作为快捷连接
        if self.shortcut:
            short = inputs
        # 如果没有快捷连接，则通过快捷连接层处理输入
        else:
            short = self.short(inputs)
        # 将快捷连接和卷积结果相加
        y = paddle.add(x=short, y=conv1)
        # 使用 ReLU 激活函数
        y = F.relu(y)
        # 返回结果
        return y
# 定义 ResNet 类，继承自 nn.Layer
class ResNet(nn.Layer):
    # 前向传播函数，接收输入 inputs
    def forward(self, inputs):
        # 初始化输出列表，将输入添加到列表中
        out = [inputs]
        # 对输入进行第一次卷积操作
        y = self.conv1_1(inputs)
        # 将卷积结果添加到输出列表中
        out.append(y)
        # 对卷积结果进行最大池化操作
        y = self.pool2d_max(y)
        # 遍历网络中的每个残差块
        for block in self.stages:
            # 对当前残差块进行操作
            y = block(y)
            # 将操作后的结果添加到输出列表中
            out.append(y)
        # 返回所有操作后的结果列表
        return out
```