# `.\PaddleOCR\ppocr\modeling\backbones\rec_micronet.py`

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
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用的代码来源于以下链接：
# https://github.com/liyunsheng13/micronet/blob/main/backbone/micronet.py
# https://github.com/liyunsheng13/micronet/blob/main/backbone/activation.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 导入 paddle 中的神经网络模块
import paddle.nn as nn

# 从 ppocr.modeling.backbones.det_mobilenet_v3 模块中导入 make_divisible 函数

M0_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
    [2, 1, 8, 3, 2, 2, 0, 4, 8, 2, 2, 2, 0, 1, 1],
    [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 2, 1, 1],
    [2, 1, 16, 5, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 4, 4, 4, 32, 4, 4, 2, 2, 1, 1],
    [2, 1, 64, 5, 1, 4, 8, 8, 64, 8, 8, 2, 2, 1, 1],
    [1, 1, 96, 3, 1, 4, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 384, 3, 1, 4, 12, 12, 0, 0, 0, 2, 2, 1, 2],
]
M1_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1, 8, 3, 2, 2, 0, 6, 8, 2, 2, 2, 0, 1, 1],
    [2, 1, 16, 3, 2, 2, 0, 8, 16, 4, 4, 2, 2, 1, 1],
    [2, 1, 16, 5, 2, 2, 0, 16, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 6, 4, 4, 32, 4, 4, 2, 2, 1, 1],
    [2, 1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1, 1],
    [1, 1, 96, 3, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 576, 3, 1, 6, 12, 12, 0, 0, 0, 2, 2, 1, 2],
]
M2_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 0, 1, 1],
    # 创建一个包含整数列表的二维数组
    [2, 1, 16, 3, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 2, 2, 1, 1],
    [2, 1, 32, 5, 1, 6, 6, 6, 32, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 2, 2, 1, 2],
    [1, 1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1, 2],
    [2, 1, 96, 5, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 128, 3, 1, 6, 12, 12, 128, 8, 8, 2, 2, 1, 2],
    [1, 1, 768, 3, 1, 6, 16, 16, 0, 0, 0, 2, 2, 1, 2],
# 定义了一系列 M3 网络的配置参数，每个参数列表包含了 s, n, c, ks, c1, c2, g1, g2, c3, g3, g4 等参数
M3_cfgs = [
    [2, 1, 16, 3, 2, 2, 0, 12, 16, 4, 4, 0, 2, 0, 1],
    [2, 1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 0, 2, 0, 1],
    [1, 1, 24, 3, 2, 2, 0, 24, 24, 4, 4, 0, 2, 0, 1],
    [2, 1, 32, 5, 1, 6, 6, 6, 32, 4, 4, 0, 2, 0, 1],
    [1, 1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 0, 2, 0, 2],
    [1, 1, 64, 5, 1, 6, 8, 8, 48, 8, 8, 0, 2, 0, 2],
    [1, 1, 80, 5, 1, 6, 8, 8, 80, 8, 8, 0, 2, 0, 2],
    [1, 1, 80, 5, 1, 6, 10, 10, 80, 8, 8, 0, 2, 0, 2],
    [1, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2],
    [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2],
    [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2],
    [1, 1, 432, 3, 1, 3, 12, 12, 0, 0, 0, 0, 2, 0, 2],
]

# 定义一个函数，根据传入的模式返回对应的配置参数列表
def get_micronet_config(mode):
    return eval(mode + '_cfgs')


# 定义一个自定义的最大组池化层
class MaxGroupPooling(nn.Layer):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = x.shape

        # reshape
        y = paddle.reshape(x, [b, c // self.channel_per_group, -1, h, w])
        out = paddle.max(y, axis=2)
        return out


# 定义一个空间分离卷积层
class SpatialSepConvSF(nn.Layer):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                oup1, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0),
                bias_attr=False,
                groups=1),
            nn.BatchNorm2D(oup1),
            nn.Conv2D(
                oup1,
                oup1 * oup2, (1, kernel_size), (1, stride),
                (0, kernel_size // 2),
                bias_attr=False,
                groups=oup1),
            nn.BatchNorm2D(oup1 * oup2),
            ChannelShuffle(oup1), )
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 将输入 x 通过卷积层 conv 处理得到输出 out
        out = self.conv(x)
        # 返回输出 out
        return out
class ChannelShuffle(nn.Layer):
    # 定义通道混洗层，用于对输入的通道进行重新排列
    def __init__(self, groups):
        # 初始化函数，接受参数groups表示分组数
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        # 前向传播函数，接受输入x
        b, c, h, w = x.shape

        channels_per_group = c // self.groups

        # 重新调整输入x的形状
        x = paddle.reshape(x, [b, self.groups, channels_per_group, h, w])

        x = paddle.transpose(x, (0, 2, 1, 3, 4))
        out = paddle.reshape(x, [b, -1, h, w])

        return out


class StemLayer(nn.Layer):
    # 定义干细胞层，用于网络的初始部分
    def __init__(self, inp, oup, stride, groups=(4, 4)):
        # 初始化函数，接受输入通道数inp，输出通道数oup，步长stride，分组数groups
        super(StemLayer, self).__init()

        g1, g2 = groups
        self.stem = nn.Sequential(
            SpatialSepConvSF(inp, groups, 3, stride),
            MaxGroupPooling(2) if g1 * g2 == 2 * oup else nn.ReLU6())

    def forward(self, x):
        # 前向传播函数，接受输入x
        out = self.stem(x)
        return out


class DepthSpatialSepConv(nn.Layer):
    # 定义深度空间分离卷积层
    def __init__(self, inp, expand, kernel_size, stride):
        # 初始化函数，接受输入通道数inp，扩展参数expand，卷积核大小kernel_size，步长stride
        super(DepthSpatialSepConv, self).__init()

        exp1, exp2 = expand

        hidden_dim = inp * exp1
        oup = inp * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                inp * exp1, (kernel_size, 1), (stride, 1),
                (kernel_size // 2, 0),
                bias_attr=False,
                groups=inp),
            nn.BatchNorm2D(inp * exp1),
            nn.Conv2D(
                hidden_dim,
                oup, (1, kernel_size),
                1, (0, kernel_size // 2),
                bias_attr=False,
                groups=hidden_dim),
            nn.BatchNorm2D(oup))

    def forward(self, x):
        # 前向传播函数，接受输入x
        x = self.conv(x)
        return x


class GroupConv(nn.Layer):
    # 定义分组卷积层
    # 初始化 GroupConv 类，设置输入通道数、输出通道数和分组数
    def __init__(self, inp, oup, groups=2):
        # 调用父类的初始化方法
        super(GroupConv, self).__init__()
        # 设置输入通道数
        self.inp = inp
        # 设置输出通道数
        self.oup = oup
        # 设置分组数
        self.groups = groups
        # 创建卷积层和批归一化层的序列
        self.conv = nn.Sequential(
            # 创建卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充、是否有偏置、分组数
            nn.Conv2D(
                inp, oup, 1, 1, 0, bias_attr=False, groups=self.groups[0]),
            # 创建批归一化层，设置输出通道数
            nn.BatchNorm2D(oup))

    # 前向传播函数
    def forward(self, x):
        # 将输入数据经过卷积和批归一化层处理
        x = self.conv(x)
        # 返回处理后的数据
        return x
# 定义深度卷积层类，继承自 nn.Layer
class DepthConv(nn.Layer):
    # 初始化方法，接受输入通道数、输出通道数、卷积核大小和步长作为参数
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        # 定义卷积层和批归一化层的序列
        self.conv = nn.Sequential(
            # 创建二维卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充大小、是否有偏置、分组数
            nn.Conv2D(
                inp,
                oup,
                kernel_size,
                stride,
                kernel_size // 2,
                bias_attr=False,
                groups=inp),
            # 创建二维批归一化层，设置输出通道数
            nn.BatchNorm2D(oup))

    # 前向传播方法，接受输入张量 x，经过卷积和批归一化后返回输出张量
    def forward(self, x):
        # 将输入张量 x 经过卷积和批归一化操作得到输出张量 out
        out = self.conv(x)
        # 返回输出张量
        return out


# 定义 DYShiftMax 类，继承自 nn.Layer
class DYShiftMax(nn.Layer):
    # 初始化函数，接受输入、输出维度，压缩比例，激活函数最大值，是否使用ReLU激活函数，初始化参数，是否在池化之前使用ReLU激活函数，分组卷积参数，是否进行扩展
    def __init__(self,
                 inp,
                 oup,
                 reduction=4,
                 act_max=1.0,
                 act_relu=True,
                 init_a=[0.0, 0.0],
                 init_b=[0.0, 0.0],
                 relu_before_pool=False,
                 g=None,
                 expansion=False):
        # 调用父类的初始化函数
        super(DYShiftMax, self).__init__()
        # 初始化输出维度
        self.oup = oup
        # 设置激活函数最大值
        self.act_max = act_max * 2
        # 是否使用ReLU激活函数
        self.act_relu = act_relu
        # 创建平均池化层，根据是否在池化之前使用ReLU激活函数来决定
        self.avg_pool = nn.Sequential(nn.ReLU() if relu_before_pool == True else
                                      nn.Sequential(), nn.AdaptiveAvgPool2D(1))

        # 根据是否使用ReLU激活函数来确定扩展参数
        self.exp = 4 if act_relu else 2
        # 初始化参数a
        self.init_a = init_a
        # 初始化参数b
        self.init_b = init_b

        # 确定压缩维度
        squeeze = make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4

        # 创建全连接层
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(), nn.Linear(squeeze, oup * self.exp), nn.Hardsigmoid())

        # 如果g为None，则设置为1
        if g is None:
            g = 1
        # 设置分组卷积参数
        self.g = g[1]
        # 如果分组卷积参数不为1且进行扩展，则重新设置分组卷积参数
        if self.g != 1 and expansion:
            self.g = inp // self.g

        # 计算分组卷积参数
        self.gc = inp // self.g
        # 创建索引张量
        index = paddle.to_tensor([range(inp)])
        index = paddle.reshape(index, [1, inp, 1, 1])
        index = paddle.reshape(index, [1, self.g, self.gc, 1, 1])
        indexgs = paddle.split(index, [1, self.g - 1], axis=1)
        indexgs = paddle.concat((indexgs[1], indexgs[0]), axis=1)
        indexs = paddle.split(indexgs, [1, self.gc - 1], axis=2)
        indexs = paddle.concat((indexs[1], indexs[0]), axis=2)
        # 设置索引
        self.index = paddle.reshape(indexs, [inp])
        # 是否进行扩展
        self.expansion = expansion
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 复制输入 x 到 x_in 和 x_out
        x_in = x
        x_out = x

        # 获取输入 x_in 的形状信息
        b, c, _, _ = x_in.shape
        # 对输入 x_in 进行平均池化操作
        y = self.avg_pool(x_in)
        # 重新调整 y 的形状为 [b, c]
        y = paddle.reshape(y, [b, c])
        # 使用全连接层对 y 进行处理
        y = self.fc(y)
        # 重新调整 y 的形状为 [b, self.oup * self.exp, 1, 1]
        y = paddle.reshape(y, [b, self.oup * self.exp, 1, 1])
        # 对 y 进行缩放和平移操作
        y = (y - 0.5) * self.act_max

        # 获取输出 x_out 的形状信息
        n2, c2, h2, w2 = x_out.shape
        # 从 x_out 中提取特定索引的数据，转换为张量 x2
        x2 = paddle.to_tensor(x_out.numpy()[:, self.index.numpy(), :, :])

        # 根据 self.exp 的值进行不同的处理
        if self.exp == 4:
            # 获取 y 的形状信息
            temp = y.shape
            # 将 y 拆分为两部分 a1 和 b1
            a1, b1, a2, b2 = paddle.split(y, temp[1] // self.oup, axis=1)
            # 对 a1 和 a2 进行平移操作
            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]
            # 对 b1 和 b2 进行平移操作
            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]
            # 计算两个部分的加权和
            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2
            # 取两者的最大值作为输出
            out = paddle.maximum(z1, z2)

        elif self.exp == 2:
            # 获取 y 的形状信息
            temp = y.shape
            # 将 y 拆分为两部分 a1 和 b1
            a1, b1 = paddle.split(y, temp[1] // self.oup, axis=1)
            # 对 a1 进行平移操作
            a1 = a1 + self.init_a[0]
            # 对 b1 进行平移操作
            b1 = b1 + self.init_b[0]
            # 计算加权和作为输出
            out = x_out * a1 + x2 * b1

        # 返回最终输出结果
        return out
class DYMicroBlock(nn.Layer):
    # 定义一个继承自 nn.Layer 的 DYMicroBlock 类
    def forward(self, x):
        # 将输入 x 赋值给 identity
        identity = x
        # 将输入 x 传入 layers 方法得到输出 out
        out = self.layers(x)

        # 如果有 identity 参数
        if self.identity:
            # 将 out 与 identity 相加
            out = out + identity

        # 返回输出 out
        return out


class MicroNet(nn.Layer):
    """
        the MicroNet backbone network for recognition module.
        Args:
            mode(str): {'M0', 'M1', 'M2', 'M3'} 
                Four models are proposed based on four different computational costs (4M, 6M, 12M, 21M MAdds)
                Default: 'M3'.
    """
    # 定义一个 MicroNet 类，用于识别模块的主干网络
    def forward(self, x):
        # 将输入 x 传入 features 方法得到输出 x
        x = self.features(x)
        # 将输出 x 传入 pool 方法得到输出 x
        x = self.pool(x)
        # 返回输出 x
        return x
```