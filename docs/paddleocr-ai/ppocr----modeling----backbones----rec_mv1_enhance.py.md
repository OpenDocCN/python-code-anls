# `.\PaddleOCR\ppocr\modeling\backbones\rec_mv1_enhance.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 本代码参考自: https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_lcnet.py

# 导入必要的库和模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
from paddle import ParamAttr, reshape, transpose
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay
from paddle.nn.functional import hardswish, hardsigmoid

# 定义 ConvBNLayer 类
class ConvBNLayer(nn.Layer):
    # 定义一个卷积层和批量归一化层的类
    class ConvBNLayer(nn.Layer):
        # 初始化函数，接受卷积层的参数和激活函数等参数
        def __init__(self,
                     num_channels,
                     filter_size,
                     num_filters,
                     stride,
                     padding,
                     channels=None,
                     num_groups=1,
                     act='hard_swish'):
            # 调用父类的初始化函数
            super(ConvBNLayer, self).__init__()
    
            # 创建一个卷积层对象
            self._conv = Conv2D(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                groups=num_groups,
                weight_attr=ParamAttr(initializer=KaimingNormal()),
                bias_attr=False)
    
            # 创建一个批量归一化层对象
            self._batch_norm = BatchNorm(
                num_filters,
                act=act,
                param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
    
        # 前向传播函数，接受输入数据，经过卷积和批量归一化后返回结果
        def forward(self, inputs):
            # 经过卷积操作
            y = self._conv(inputs)
            # 经过批量归一化操作
            y = self._batch_norm(y)
            # 返回结果
            return y
class DepthwiseSeparable(nn.Layer):
    # 定义深度可分离卷积层
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 dw_size=3,
                 padding=1,
                 use_se=False):
        # 初始化函数
        super(DepthwiseSeparable, self).__init__()
        # 设置是否使用 SE 模块
        self.use_se = use_se
        # 创建深度卷积层对象
        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=dw_size,
            stride=stride,
            padding=padding,
            num_groups=int(num_groups * scale))
        # 如果使用 SE 模块，创建 SE 模块对象
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        # 创建逐点卷积层对象
        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    # 前向传播函数
    def forward(self, inputs):
        # 深度卷积层前向传播
        y = self._depthwise_conv(inputs)
        # 如果使用 SE 模块，应用 SE 模块
        if self.use_se:
            y = self._se(y)
        # 逐点卷积层前向传播
        y = self._pointwise_conv(y)
        return y


class MobileNetV1Enhance(nn.Layer):
    # 定义 MobileNetV1Enhance 类
    def forward(self, inputs):
        # 执行卷积操作
        y = self.conv1(inputs)
        # 执行块列表操作
        y = self.block_list(y)
        # 执行池化操作
        y = self.pool(y)
        return y


class SEModule(nn.Layer):
    # 定义 SE 模块类
    def __init__(self, channel, reduction=4):
        # 初始化函数
        super(SEModule, self).__init__()
        # 创建全局平均池化层
        self.avg_pool = AdaptiveAvgPool2D(1)
        # 创建第一个卷积层
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())
        # 创建第二个卷积层
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())
    # 定义前向传播函数，接收输入并返回输出
    def forward(self, inputs):
        # 对输入进行平均池化操作
        outputs = self.avg_pool(inputs)
        # 对平均池化后的结果进行第一次卷积操作
        outputs = self.conv1(outputs)
        # 对第一次卷积后的结果进行ReLU激活函数操作
        outputs = F.relu(outputs)
        # 对ReLU激活后的结果进行第二次卷积操作
        outputs = self.conv2(outputs)
        # 对第二次卷积后的结果进行硬Sigmoid激活函数操作
        outputs = hardsigmoid(outputs)
        # 返回输入和输出的元素级乘积结果
        return paddle.multiply(x=inputs, y=outputs)
```