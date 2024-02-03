# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\neck\FPN.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 10:29
# @Author  : zhoujun
# 导入 paddle 模块
import paddle
# 导入 paddle 中的 nn 模块
import paddle.nn.functional as F
from paddle import nn

# 导入自定义模块 models.basic 中的 ConvBnRelu 类
from models.basic import ConvBnRelu

# 定义 FPN 类，继承自 nn.Layer 类
class FPN(nn.Layer):
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置 inplace 变量为 True
        inplace = True
        # 设置内部卷积层的输出通道数
        self.conv_out = inner_channels
        # 将内部通道数减少为原来的四分之一
        inner_channels = inner_channels // 4
        # reduce layers
        # 创建减少通道数的卷积层对象，用于处理不同层级的特征图
        self.reduce_conv_c2 = ConvBnRelu(
            in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(
            in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(
            in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(
            in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        # 创建平滑层对象，用于平滑处理不同层级的特征图
        self.smooth_p4 = ConvBnRelu(
            inner_channels,
            inner_channels,
            kernel_size=3,
            padding=1,
            inplace=inplace)
        self.smooth_p3 = ConvBnRelu(
            inner_channels,
            inner_channels,
            kernel_size=3,
            padding=1,
            inplace=inplace)
        self.smooth_p2 = ConvBnRelu(
            inner_channels,
            inner_channels,
            kernel_size=3,
            padding=1,
            inplace=inplace)

        # 创建卷积层对象，用于最终输出
        self.conv = nn.Sequential(
            nn.Conv2D(
                self.conv_out,
                self.conv_out,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.BatchNorm2D(self.conv_out),
            nn.ReLU())
        # 设置输出通道数
        self.out_channels = self.conv_out
    # 定义一个前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 拆分为 c2, c3, c4, c5 四个部分
        c2, c3, c4, c5 = x
        # 从最高层到最底层进行特征融合和上采样
        # 通过 reduce_conv_c5 函数对 c5 进行降维
        p5 = self.reduce_conv_c5(c5)
        # 将 p5 上采样并与 reduce_conv_c4(c4) 相加
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        # 对 p4 进行平滑处理
        p4 = self.smooth_p4(p4)
        # 将 p4 上采样并与 reduce_conv_c3(c3) 相加
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        # 对 p3 进行平滑处理
        p3 = self.smooth_p3(p3)
        # 将 p3 上采样并与 reduce_conv_c2(c2) 相加
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        # 对 p2 进行平滑处理
        p2 = self.smooth_p2(p2)

        # 将 p2, p3, p4, p5 拼接在一起并进行上采样
        x = self._upsample_cat(p2, p3, p4, p5)
        # 经过一个卷积层处理
        x = self.conv(x)
        # 返回处理后的结果
        return x

    # 定义一个上采样并相加的函数，接收输入 x 和 y
    def _upsample_add(self, x, y):
        # 对 x 进行上采样使其大小与 y 相同，然后与 y 相加
        return F.interpolate(x, size=y.shape[2:]) + y

    # 定义一个上采样并拼接的函数，接收输入 p2, p3, p4, p5
    def _upsample_cat(self, p2, p3, p4, p5):
        # 获取 p2 的高度和宽度
        h, w = p2.shape[2:]
        # 将 p3, p4, p5 上采样至与 p2 相同的大小
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        # 在通道维度上拼接 p2, p3, p4, p5
        return paddle.concat([p2, p3, p4, p5], axis=1)
```