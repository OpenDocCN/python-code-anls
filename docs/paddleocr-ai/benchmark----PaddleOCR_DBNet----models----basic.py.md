# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\basic.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/12/6 11:19
# @Author  : zhoujun
# 导入 paddle 中的 nn 模块
from paddle import nn

# 定义一个名为 ConvBnRelu 的类，继承自 nn.Layer
class ConvBnRelu(nn.Layer):
    # 初始化函数，接受多个参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 inplace=True):
        super().__init__()
        # 创建一个卷积层对象
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
            padding_mode=padding_mode)
        # 创建一个二维批归一化层对象
        self.bn = nn.BatchNorm2D(out_channels)
        # 创建一个 ReLU 激活函数层对象
        self.relu = nn.ReLU()

    # 前向传播函数，接受输入 x，返回处理后的结果
    def forward(self, x):
        # 通过卷积层处理输入 x
        x = self.conv(x)
        # 通过批归一化层处理 x
        x = self.bn(x)
        # 通过 ReLU 激活函数处理 x
        x = self.relu(x)
        # 返回处理后的结果 x
        return x
```