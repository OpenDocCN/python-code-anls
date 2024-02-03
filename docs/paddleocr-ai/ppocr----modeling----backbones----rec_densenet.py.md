# `.\PaddleOCR\ppocr\modeling\backbones\rec_densenet.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用的代码来源于：
# https://github.com/LBH1024/CAN/models/densenet.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义 Bottleneck 类，继承自 nn.Layer
class Bottleneck(nn.Layer):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        # 第一个 BatchNorm2D 层
        self.bn1 = nn.BatchNorm2D(interChannels)
        # 第一个卷积层
        self.conv1 = nn.Conv2D(
            nChannels, interChannels, kernel_size=1,
            bias_attr=None)  # Xavier initialization
        # 第二个 BatchNorm2D 层
        self.bn2 = nn.BatchNorm2D(growthRate)
        # 第二个卷积层
        self.conv2 = nn.Conv2D(
            interChannels, growthRate, kernel_size=3, padding=1,
            bias_attr=None)  # Xavier initialization
        self.use_dropout = use_dropout
        # 是否使用 dropout
        self.dropout = nn.Dropout(p=0.2)

    # 前向传播函数
    def forward(self, x):
        # 第一个卷积层后的激活函数和 BatchNorm2D 层
        out = F.relu(self.bn1(self.conv1(x)))
        # 如果使用 dropout，则应用 dropout
        if self.use_dropout:
            out = self.dropout(out)
        # 第二个卷积层后的激活函数和 BatchNorm2D 层
        out = F.relu(self.bn2(self.conv2(out)))
        # 如果使用 dropout，则应用 dropout
        if self.use_dropout:
            out = self.dropout(out)
        # 将输入 x 和输出 out 进行拼接
        out = paddle.concat([x, out], 1)
        return out


# 定义 SingleLayer 类
class SingleLayer(nn.Layer):
    # 初始化函数，接受通道数、增长率和是否使用 dropout 作为参数
    def __init__(self, nChannels, growthRate, use_dropout):
        # 调用父类的初始化函数
        super(SingleLayer, self).__init__()
        # Batch normalization 层，对输入数据进行标准化
        self.bn1 = nn.BatchNorm2D(nChannels)
        # 卷积层，使用 nChannels 个输入通道和 growthRate 个输出通道，3x3 的卷积核，padding 为 1，不使用偏置
        self.conv1 = nn.Conv2D(
            nChannels, growthRate, kernel_size=3, padding=1, bias_attr=False)

        # 是否使用 dropout
        self.use_dropout = use_dropout
        # Dropout 层，以 0.2 的概率将输入置为 0
        self.dropout = nn.Dropout(p=0.2)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 对输入 x 进行 ReLU 激活函数和卷积操作
        out = self.conv1(F.relu(x))
        # 如果使用 dropout，则对输出进行 dropout 操作
        if self.use_dropout:
            out = self.dropout(out)

        # 将输入 x 和处理后的输出 out 进行拼接，沿着通道维度拼接
        out = paddle.concat([x, out], 1)
        # 返回拼接后的结果
        return out
class Transition(nn.Layer):
    # 定义过渡层类，用于连接不同密集块之间的层
    def __init__(self, nChannels, out_channels, use_dropout):
        # 初始化过渡层对象
        super(Transition, self).__init__()
        # 初始化 Batch Normalization 层
        self.bn1 = nn.BatchNorm2D(out_channels)
        # 初始化卷积层
        self.conv1 = nn.Conv2D(
            nChannels, out_channels, kernel_size=1, bias_attr=False)
        # 设置是否使用 dropout
        self.use_dropout = use_dropout
        # 初始化 dropout 层
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 前向传播函数
        # 使用 ReLU 激活函数对卷积层和 Batch Normalization 层的输出进行激活
        out = F.relu(self.bn1(self.conv1(x)))
        # 如果使用 dropout，则对输出进行 dropout 处理
        if self.use_dropout:
            out = self.dropout(out)
        # 对输出进行平均池化
        out = F.avg_pool2d(out, 2, ceil_mode=True, exclusive=False)
        return out


class DenseNet(nn.Layer):
    # 定义 DenseNet 类
    def __init__(self, growthRate, reduction, bottleneck, use_dropout,
                 input_channel, **kwargs):
        # 初始化 DenseNet 对象
        super(DenseNet, self).__init__()

        nDenseBlocks = 16
        nChannels = 2 * growthRate

        # 初始化第一个卷积层
        self.conv1 = nn.Conv2D(
            input_channel,
            nChannels,
            kernel_size=7,
            padding=3,
            stride=2,
            bias_attr=False)
        # 构建第一个密集块
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        # 构建第一个过渡层
        self.trans1 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        # 构建第二个密集块
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        # 构建第二个过渡层
        self.trans2 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        # 构建第三个密集块
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        self.out_channels = out_channels
    # 创建稠密块，根据参数设置创建多个层
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck,
                    use_dropout):
        # 初始化层列表
        layers = []
        # 循环创建指定数量的稠密块
        for i in range(int(nDenseBlocks)):
            # 如果使用瓶颈结构
            if bottleneck:
                # 添加瓶颈层到层列表
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                # 添加单层到层列表
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            # 更新通道数
            nChannels += growthRate
        # 返回一个包含所有层的序列
        return nn.Sequential(*layers)

    # 前向传播函数
    def forward(self, inputs):
        # 解包输入数据
        x, x_m, y = inputs
        # 第一层卷积
        out = self.conv1(x)
        # ReLU 激活函数
        out = F.relu(out)
        # 最大池化
        out = F.max_pool2d(out, 2, ceil_mode=True)
        # 第一个稠密块
        out = self.dense1(out)
        # 第一个过渡层
        out = self.trans1(out)
        # 第二个稠密块
        out = self.dense2(out)
        # 第二个过渡层
        out = self.trans2(out)
        # 第三个稠密块
        out = self.dense3(out)
        # 返回输出、中间数据和标签
        return out, x_m, y
```