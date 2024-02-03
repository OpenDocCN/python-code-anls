# `.\PaddleOCR\ppocr\modeling\backbones\det_mobilenet_v3.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 Paddle 库
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

# 定义导出的类列表
__all__ = ['MobileNetV3']

# 定义函数，将输入的值调整为可被除数整除的值
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 定义 MobileNetV3 类，继承自 nn.Layer
class MobileNetV3(nn.Layer):
    # 前向传播函数
    def forward(self, x):
        # 对输入进行卷积操作
        x = self.conv(x)
        # 初始化输出列表
        out_list = []
        # 遍历每个阶段
        for stage in self.stages:
            # 对输入进行当前阶段的操作
            x = stage(x)
            # 将当前阶段的输出添加到输出列表中
            out_list.append(x)
        # 返回输出列表
        return out_list

# 定义卷积和批归一化层类，继承自 nn.Layer
class ConvBNLayer(nn.Layer):
    # 初始化函数，定义卷积层和批归一化层
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        # 调用父类的初始化函数
        super(ConvBNLayer, self).__init__()
        # 是否需要激活函数
        self.if_act = if_act
        # 激活函数类型
        self.act = act
        # 创建卷积层对象
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        # 创建批归一化层对象
        self.bn = nn.BatchNorm(num_channels=out_channels, act=None)

    # 前向传播函数
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # 批归一化操作
        x = self.bn(x)
        # 如果需要激活函数
        if self.if_act:
            # 判断激活函数类型
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                # 打印错误信息并退出程序
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        # 返回处理后的数据
        return x
# 定义残差单元类，继承自 nn.Layer
class ResidualUnit(nn.Layer):
    # 初始化函数，接受输入通道数、中间通道数、输出通道数、卷积核大小、步长、是否使用 SE 模块、激活函数等参数
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        # 调用父类的初始化函数
        super(ResidualUnit, self).__init__()
        # 判断是否需要添加 shortcut 连接
        self.if_shortcut = stride == 1 and in_channels == out_channels
        # 判断是否使用 SE 模块
        self.if_se = use_se

        # 扩展卷积层，用于将输入通道数扩展到中间通道数
        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        # 瓶颈卷积层，用于减少通道数并保持特征图大小
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act)
        # 如果使用 SE 模块，则添加 SE 模块
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        # 线性卷积层，将中间通道数转换为输出通道数
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    # 前向传播函数，接受输入数据，依次经过各层操作后返回结果
    def forward(self, inputs):
        # 经过扩展卷积层
        x = self.expand_conv(inputs)
        # 经过瓶颈卷积层
        x = self.bottleneck_conv(x)
        # 如果使用 SE 模块，则经过 SE 模块
        if self.if_se:
            x = self.mid_se(x)
        # 经过线性卷积层
        x = self.linear_conv(x)
        # 如果需要添加 shortcut 连接，则将输入数据与输出数据相加
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        # 返回结果
        return x


# 定义 SE 模块类，继承自 nn.Layer
class SEModule(nn.Layer):
    # 初始化方法，接受输入通道数和缩减比例作为参数
    def __init__(self, in_channels, reduction=4):
        # 调用父类的初始化方法
        super(SEModule, self).__init__()
        # 创建一个自适应平均池化层，输出大小为1
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # 创建一个卷积层，用于通道间的信息交互，输入通道数为in_channels，输出通道数为in_channels // reduction
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        # 创建一个卷积层，用于还原通道数，输入通道数为in_channels // reduction，输出通道数为in_channels
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    # 前向传播方法，接受输入数据作为参数
    def forward(self, inputs):
        # 对输入数据进行平均池化
        outputs = self.avg_pool(inputs)
        # 通过第一个卷积层处理数据
        outputs = self.conv1(outputs)
        # 对输出数据进行ReLU激活函数处理
        outputs = F.relu(outputs)
        # 通过第二个卷积层处理数据
        outputs = self.conv2(outputs)
        # 对输出数据进行硬Sigmoid激活函数处理，使用斜率为0.2和偏移为0.5
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        # 返回输入数据与处理后的输出数据的乘积
        return inputs * outputs
```