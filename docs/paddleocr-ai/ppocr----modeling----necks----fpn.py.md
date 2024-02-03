# `.\PaddleOCR\ppocr\modeling\necks\fpn.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""
# 代码来源于:
# https://github.com/whai362/PSENet/blob/python3/models/neck/fpn.py

# 导入所需的库
import paddle.nn as nn
import paddle
import math
import paddle.nn.functional as F

# 定义一个类，包含卷积、批归一化和激活函数
class Conv_BN_ReLU(nn.Layer):
    # 初始化函数，定义了一个包含卷积、批归一化和ReLU激活函数的模块
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        # 调用父类的初始化函数
        super(Conv_BN_ReLU, self).__init__()
        # 创建卷积层，设置输入通道数、输出通道数、卷积核大小、步长和填充
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
        # 创建批归一化层，设置输出通道数和动量参数
        self.bn = nn.BatchNorm2D(out_planes, momentum=0.1)
        # 创建ReLU激活函数层
        self.relu = nn.ReLU()

        # 遍历模块的子层
        for m in self.sublayers():
            # 如果子层是卷积层
            if isinstance(m, nn.Conv2D):
                # 计算权重参数的初始化标准差
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                # 初始化卷积层的权重参数
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Normal(
                        0, math.sqrt(2. / n)))
            # 如果子层是批归一化层
            elif isinstance(m, nn.BatchNorm2D):
                # 初始化批归一化层的权重参数为1.0
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(1.0))
                # 初始化批归一化层的偏置参数为0.0
                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(0.0))

    # 前向传播函数，定义了模块的前向计算过程
    def forward(self, x):
        # 先进行卷积操作，然后批归一化，最后经过ReLU激活函数
        return self.relu(self.bn(self.conv(x)))
class FPN(nn.Layer):
    # FPN 构造函数，接受输入通道数和输出通道数
    def __init__(self, in_channels, out_channels):
        # 调用父类构造函数
        super(FPN, self).__init__()

        # 顶层特征图处理
        self.toplayer_ = Conv_BN_ReLU(
            in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)
        # 侧边特征图处理
        self.latlayer1_ = Conv_BN_ReLU(
            in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(
            in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(
            in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)

        # 平滑处理
        self.smooth1_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 输出通道数为输入通道数的四倍
        self.out_channels = out_channels * 4
        # 遍历子层，初始化权重和偏置
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                # 初始化卷积层权重
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Normal(
                        0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                # 初始化批归一化层权重和偏置
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(0.0))
    # 定义一个函数用于上采样输入张量 x，可以指定缩放比例，默认为 1
    def _upsample(self, x, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear')

    # 定义一个函数用于上采样输入张量 x，并将结果与另一个张量 y 相加，可以指定缩放比例，默认为 1
    def _upsample_add(self, x, y, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear') + y

    # 定义前向传播函数，接收输入张量 x
    def forward(self, x):
        # 将输入张量 x 拆分为 f2, f3, f4, f5 四个部分
        f2, f3, f4, f5 = x
        # 对 f5 进行顶层处理
        p5 = self.toplayer_(f5)

        # 对 f4 进行横向处理
        f4 = self.latlayer1_(f4)
        # 将 p5 上采样并与 f4 相加，缩放比例为 2
        p4 = self._upsample_add(p5, f4, 2)
        # 对 p4 进行平滑处理
        p4 = self.smooth1_(p4)

        # 对 f3 进行横向处理
        f3 = self.latlayer2_(f3)
        # 将 p4 上采样并与 f3 相加，缩放比例为 2
        p3 = self._upsample_add(p4, f3, 2)
        # 对 p3 进行平滑处理
        p3 = self.smooth2_(p3)

        # 对 f2 进行横向处理
        f2 = self.latlayer3_(f2)
        # 将 p3 上采样并与 f2 相加，缩放比例为 2
        p2 = self._upsample_add(p3, f2, 2)
        # 对 p2 进行平滑处理
        p2 = self.smooth3_(p2)

        # 对 p3 进行上采样，缩放比例为 2
        p3 = self._upsample(p3, 2)
        # 对 p4 进行上采样，缩放比例为 4
        p4 = self._upsample(p4, 4)
        # 对 p5 进行上采样，缩放比例为 8
        p5 = self._upsample(p5, 8)

        # 将 p2, p3, p4, p5 沿着 axis=1 进行拼接
        fuse = paddle.concat([p2, p3, p4, p5], axis=1)
        # 返回拼接后的结果
        return fuse
```