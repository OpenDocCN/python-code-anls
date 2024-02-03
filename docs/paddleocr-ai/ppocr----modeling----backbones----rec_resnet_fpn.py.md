# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_fpn.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
#
# 导入必要的库和模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn, ParamAttr
from paddle.nn import functional as F
import paddle
import numpy as np

# 定义模块的导出列表
__all__ = ["ResNetFPN"]

# 定义 ResNetFPN 类，继承自 nn.Layer
class ResNetFPN(nn.Layer):
    # 定义类的调用方法，接受输入 x
    def __call__(self, x):
        # 对输入 x 进行卷积操作
        x = self.conv(x)
        # 初始化 FPN 列表和 F 列表
        fpn_list = []
        F = []
        # 遍历深度列表，计算累积和并添加到 fpn_list 中
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[:i + 1]))

        # 遍历块列表中的每个块
        for i, block in enumerate(self.block_list):
            # 对输入 x 进行块操作
            x = block(x)
            # 遍历 fpn_list 中的数值
            for number in fpn_list:
                # 如果 i + 1 等于 number，则将 x 添加到 F 中
                if i + 1 == number:
                    F.append(x)
        # 将最后一个 F 添加到 base 中
        base = F[-1]

        # 初始化 j 为 0
        j = 0
        # 遍历基础块列表中的每个块
        for i, block in enumerate(self.base_block):
            # 如果 i 是 3 的倍数且小于 6
            if i % 3 == 0 and i < 6:
                j = j + 1
                # 获取 F 的形状信息
                b, c, w, h = F[-j - 1].shape
                # 如果 F 的宽高与 base 的宽高相同，则保持 base 不变
                if [w, h] == list(base.shape[2:]):
                    base = base
                else:
                    # 否则，对 base 进行转置卷积和批归一化操作
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                # 在通道维度上拼接 base 和 F[-j-1]
                base = paddle.concat([base, F[-j - 1]], axis=1)
            # 对 base 进行块操作
            base = block(base)
        # 返回 base
        return base

# 定义 ConvBNLayer 类，继承自 nn.Layer
class ConvBNLayer(nn.Layer):
    # 初始化函数，定义卷积层和批归一化层
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        # 调用父类的初始化函数
        super(ConvBNLayer, self).__init__()
        # 创建卷积层对象
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 if stride == (1, 1) else kernel_size,
            dilation=2 if stride == (1, 1) else 1,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + '.conv2d.output.1.w_0'),
            bias_attr=False, )

        # 根据名称确定批归一化层的名称
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # 创建批归一化层对象
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name=name + '.output.1.w_0'),
            bias_attr=ParamAttr(name=name + '.output.1.b_0'),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")

    # 调用函数，对输入进行卷积和批归一化操作
    def __call__(self, x):
        # 卷积操作
        x = self.conv(x)
        # 批归一化操作
        x = self.bn(x)
        # 返回处理后的结果
        return x
# 定义一个名为 ShortCut 的类，继承自 nn.Layer 类
class ShortCut(nn.Layer):
    # 初始化方法，接受输入通道数、输出通道数、步幅、名称和是否为第一层的标志
    def __init__(self, in_channels, out_channels, stride, name, is_first=False):
        # 调用父类的初始化方法
        super(ShortCut, self).__init__()
        # 初始化一个标志位，表示是否使用卷积层
        self.use_conv = True

        # 判断是否需要使用卷积层
        if in_channels != out_channels or stride != 1 or is_first == True:
            # 如果步幅为 (1, 1)，则创建一个 1x1 的卷积层
            if stride == (1, 1):
                self.conv = ConvBNLayer(
                    in_channels, out_channels, 1, 1, name=name)
            else:  # 如果步幅为 (2, 2)，则创建一个 1x1 的卷积层
                self.conv = ConvBNLayer(
                    in_channels, out_channels, 1, stride, name=name)
        else:
            # 如果不需要使用卷积层，则将标志位设为 False
            self.use_conv = False

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 如果需要使用卷积层，则对输入 x 进行卷积操作
        if self.use_conv:
            x = self.conv(x)
        # 返回处理后的结果
        return x


# 定义一个名为 BottleneckBlock 的类，继承自 nn.Layer 类
class BottleneckBlock(nn.Layer):
    # 初始化方法，接受输入通道数、输出通道数、步幅和名称
    def __init__(self, in_channels, out_channels, stride, name):
        # 调用父类的初始化方法
        super(BottleneckBlock, self).__init__()
        # 创建一个 1x1 的卷积层
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        # 创建一个 3x3 的卷积层
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        # 创建一个 1x1 的卷积层
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")
        # 创建一个 ShortCut 实例
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            stride=stride,
            is_first=False,
            name=name + "_branch1")
        # 记录输出通道数
        self.out_channels = out_channels * 4

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 对输入 x 进行卷积操作
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        # 将卷积结果与 ShortCut 的结果相加
        y = y + self.short(x)
        # 对结果应用 ReLU 激活函数
        y = F.relu(y)
        # 返回处理后的结果
        return y


# 定义一个名为 BasicBlock 的类，继承自 nn.Layer 类
class BasicBlock(nn.Layer):
    # 初始化 BasicBlock 类，设置输入通道数、输出通道数、步长、名称和是否为第一个块的标志
    def __init__(self, in_channels, out_channels, stride, name, is_first):
        # 调用父类的初始化方法
        super(BasicBlock, self).__init__()
        # 创建第一个卷积层和 BN 层，设置输入通道数、输出通道数、卷积核大小、激活函数为 relu、步长和名称
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        # 创建第二个卷积层和 BN 层，设置输入通道数为上一个卷积层的输出通道数、输出通道数、卷积核大小、不使用激活函数和名称
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")
        # 创建 Shortcut 层，设置输入通道数、输出通道数、步长、是否为第一个块和名称
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            is_first=is_first,
            name=name + "_branch1")
        # 设置输出通道数
        self.out_channels = out_channels

    # 前向传播函数
    def forward(self, x):
        # 经过第一个卷积层和 BN 层
        y = self.conv0(x)
        # 经过第二个卷积层和 BN 层
        y = self.conv1(y)
        # 将第二个卷积层的输出与 Shortcut 层的输出相加
        y = y + self.short(x)
        # 使用 relu 激活函数
        return F.relu(y)
```