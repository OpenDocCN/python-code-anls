# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_vd.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的保证或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 库
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

# 定义模块的导出列表
__all__ = ["ResNet"]

# 定义 ConvBNLayer 类，继承自 nn.Layer 类
class ConvBNLayer(nn.Layer):
    # 初始化函数，定义卷积层和批归一化层
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
        # 调用父类的初始化函数
        super(ConvBNLayer, self).__init__()

        # 是否为可变形卷积模式
        self.is_vd_mode = is_vd_mode
        # 创建平均池化层
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
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
        # 如果是可变形卷积模式，对输入进行平均池化
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

        # 创建一个 1x1 卷积层和 BN 层，用于降维
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        # 创建一个 3x3 卷积层和 BN 层
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        # 创建一个 1x1 卷积层和 BN 层，用于升维
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        # 如果不使用快捷连接，则创建一个额外的卷积层和 BN 层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        # 记录是否使用快捷连接
        self.shortcut = shortcut

    # 前向传播方法
    def forward(self, inputs):
        # 对输入数据进行 1x1 卷积和 BN 操作
        y = self.conv0(inputs)

        # 对上一步结果进行 3x3 卷积和 BN 操作
        conv1 = self.conv1(y)
        # 对上一步结果进行 1x1 卷积和 BN 操作
        conv2 = self.conv2(conv1)

        # 如果使用快捷连接，则直接将输入作为快捷连接
        if self.shortcut:
            short = inputs
        # 否则，对输入数据进行额外的卷积和 BN 操作
        else:
            short = self.short(inputs)
        # 将快捷连接和卷积结果相加
        y = paddle.add(x=short, y=conv2)
        # 对相加结果进行 ReLU 激活函数操作
        y = F.relu(y)
        # 返回结果
        return y


# 定义 BasicBlock 类
class BasicBlock(nn.Layer):
    # 初始化 BasicBlock 类，设置输入通道数、输出通道数、步长、是否使用快捷连接、是否为第一个 BasicBlock、名称
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

        # 如果不使用快捷连接，则创建快捷连接层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        # 设置是否使用快捷连接
        self.shortcut = shortcut

    # 前向传播函数
    def forward(self, inputs):
        # 第一个卷积层
        y = self.conv0(inputs)
        # 第二个卷积层
        conv1 = self.conv1(y)

        # 如果使用快捷连接，则直接将输入作为快捷连接
        if self.shortcut:
            short = inputs
        # 如果不使用快捷连接，则通过快捷连接层计算快捷连接
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
    # 定义前向传播函数，接收输入 inputs
    def forward(self, inputs):
        # 使用 conv1_1 卷积层处理输入数据
        y = self.conv1_1(inputs)
        # 使用 conv1_2 卷积层处理上一层输出数据
        y = self.conv1_2(y)
        # 使用 conv1_3 卷积层处理上一层输出数据
        y = self.conv1_3(y)
        # 使用最大池化层处理上一层输出数据
        y = self.pool2d_max(y)
        # 遍历 self.block_list 中的每个块，对 y 进行处理
        for block in self.block_list:
            y = block(y)
        # 使用 out_pool 池化层处理最终输出数据
        y = self.out_pool(y)
        # 返回处理后的输出数据
        return y
```