# `.\PaddleOCR\ppocr\modeling\necks\ct_fpn.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import os
import sys

import math
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
# 创建常量初始化器
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录路径添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录路径的上级目录路径添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

# 定义一个名为 Conv_BN_ReLU 的类，继承自 nn.Layer
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
        # 创建批归一化层，设置输出通道数
        self.bn = nn.BatchNorm2D(out_planes)
        # 创建ReLU激活函数层
        self.relu = nn.ReLU()

        # 遍历模块的子层
        for m in self.sublayers():
            # 如果子层是卷积层
            if isinstance(m, nn.Conv2D):
                # 计算权重初始化的标准差
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_ = Normal(mean=0.0, std=math.sqrt(2. / n))
                # 对卷积层的权重进行正态分布初始化
                normal_(m.weight)
            # 如果子层是批归一化层
            elif isinstance(m, nn.BatchNorm2D):
                # 将批归一化层的偏置项初始化为0
                zeros_(m.bias)
                # 将批归一化层的权重初始化为1
                ones_(m.weight)

    # 前向传播函数，定义了模块的前向计算过程
    def forward(self, x):
        # 返回经过卷积、批归一化和ReLU激活函数的结果
        return self.relu(self.bn(self.conv(x)))
class FPEM(nn.Layer):
    # 定义 FPEM 类，继承自 nn.Layer
    def __init__(self, in_channels, out_channels):
        # 初始化函数，接受输入通道数和输出通道数作为参数
        planes = out_channels
        # 设置 planes 变量为输出通道数
        self.dwconv3_1 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias_attr=False)
        # 创建深度可分离卷积层 dwconv3_1
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)
        # 创建平滑层 smooth_layer3_1

        self.dwconv2_1 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias_attr=False)
        # 创建深度可分离卷积层 dwconv2_1
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)
        # 创建平滑层 smooth_layer2_1

        self.dwconv1_1 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias_attr=False)
        # 创建深度可分离卷积层 dwconv1_1
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)
        # 创建平滑层 smooth_layer1_1

        self.dwconv2_2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias_attr=False)
        # 创建深度可分离卷积层 dwconv2_2
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)
        # 创建平滑层 smooth_layer2_2

        self.dwconv3_2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias_attr=False)
        # 创建深度可分离卷积层 dwconv3_2
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)
        # 创建平滑层 smooth_layer3_2

        self.dwconv4_2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias_attr=False)
        # 创建深度可分离卷积层 dwconv4_2
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)
        # 创建平滑层 smooth_layer4_2

    def _upsample_add(self, x, y):
        # 定义上采样加法函数，接受两个参数 x 和 y
        return F.upsample(x, scale_factor=2, mode='bilinear') + y
        # 对 x 进行双线性插值上采样，然后与 y 相加
    # 定义一个前向传播函数，接受四个特征图作为输入
    def forward(self, f1, f2, f3, f4):
        # 上采样并进行卷积操作，然后通过平滑层处理特征图 f3
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        # 上采样并进行卷积操作，然后通过平滑层处理特征图 f2
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        # 上采样并进行卷积操作，然后通过平滑层处理特征图 f1

        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))

        # 下采样并进行卷积操作，然后通过平滑层处理特征图 f2
        f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
        # 下采样并进行卷积操作，然后通过平滑层处理特征图 f3
        f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
        # 下采样并进行卷积操作，然后通过平滑层处理特征图 f4
        f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))

        # 返回处理后的特征图 f1, f2, f3, f4
        return f1, f2, f3, f4
class CTFPN(nn.Layer):
    # 定义 CTFPN 类，继承自 nn.Layer
    def __init__(self, in_channels, out_channel=128):
        # 初始化函数，接受输入通道数和输出通道数，默认为128
        super(CTFPN, self).__init__()
        # 调用父类的初始化函数
        self.out_channels = out_channel * 4
        # 设置输出通道数为输入的四倍

        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        # 创建第一个减少通道的层，输入通道数为in_channels[0]，输出通道数为128
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        # 创建第二个减少通道的层，输入通道数为in_channels[1]，输出通道数为128
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        # 创建第三个减少通道的层，输入通道数为in_channels[2]，输出通道数为128
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        # 创建第四个减少通道的层，输入通道数为in_channels[3]，输出通道数为128

        self.fpem1 = FPEM(in_channels=(64, 128, 256, 512), out_channels=128)
        # 创建第一个 FPEM 模块，输入通道数为(64, 128, 256, 512)，输出通道数为128
        self.fpem2 = FPEM(in_channels=(64, 128, 256, 512), out_channels=128)
        # 创建第二个 FPEM 模块，输入通道数为(64, 128, 256, 512)，输出通道数为128

    def _upsample(self, x, scale=1):
        # 定义上采样函数，对输入 x 进行上采样，缩放因子为 scale，默认为1
        return F.upsample(x, scale_factor=scale, mode='bilinear')

    def forward(self, f):
        # 前向传播函数，接受输入 f
        # reduce channel
        # 减少通道
        f1 = self.reduce_layer1(f[0])  # N,64,160,160    --> N, 128, 160, 160
        # 对输入 f 的第一个部分进行通道减少操作，得到 f1
        f2 = self.reduce_layer2(f[1])  # N, 128, 80, 80  --> N, 128, 80, 80
        # 对输入 f 的第二个部分进行通道减少操作，得到 f2
        f3 = self.reduce_layer3(f[2])  # N, 256, 40, 40  --> N, 128, 40, 40
        # 对输入 f 的第三个部分进行通道减少操作，得到 f3
        f4 = self.reduce_layer4(f[3])  # N, 512, 20, 20  --> N, 128, 20, 20
        # 对输入 f 的第四个部分进行通道减少操作，得到 f4

        # FPEM
        # FPEM 模块
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        # 使用第一个 FPEM 模块处理 f1, f2, f3, f4 得到 f1_1, f2_1, f3_1, f4_1
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)
        # 使用第二个 FPEM 模块处理 f1_1, f2_1, f3_1, f4_1 得到 f1_2, f2_2, f3_2, f4_2

        # FFM
        # FFM 模块
        f1 = f1_1 + f1_2
        # 将 f1_1 和 f1_2 相加
        f2 = f2_1 + f2_2
        # 将 f2_1 和 f2_2 相加
        f3 = f3_1 + f3_2
        # 将 f3_1 和 f3_2 相加
        f4 = f4_1 + f4_2
        # 将 f4_1 和 f4_2 相加

        f2 = self._upsample(f2, scale=2)
        # 对 f2 进行上采样，缩放因子为2
        f3 = self._upsample(f3, scale=4)
        # 对 f3 进行上采样，缩放因子为4
        f4 = self._upsample(f4, scale=8)
        # 对 f4 进行上采样，缩放因子为8
        ff = paddle.concat((f1, f2, f3, f4), 1)  # N,512, 160,160
        # 将 f1, f2, f3, f4 沿着通道维度拼接
        return ff
        # 返回结果 ff
```