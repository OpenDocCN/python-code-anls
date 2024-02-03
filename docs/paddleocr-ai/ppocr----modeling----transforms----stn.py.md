# `.\PaddleOCR\ppocr\modeling\transforms\stn.py`

```
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
"""
# 引用来源
# https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/stn_head.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np

from .tps_spatial_transformer import TPSSpatialTransformer

# 定义一个 3x3 的卷积块
def conv3x3_block(in_channels, out_channels, stride=1):
    # 计算权重初始化的标准差
    n = 3 * 3 * out_channels
    w = math.sqrt(2. / n)
    # 创建一个卷积层，使用正态分布初始化权重，常数初始化偏置
    conv_layer = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=nn.initializer.Normal(
            mean=0.0, std=w),
        bias_attr=nn.initializer.Constant(0))
    # 创建一个包含卷积层、批归一化层和激活函数的序列
    block = nn.Sequential(conv_layer, nn.BatchNorm2D(out_channels), nn.ReLU())
    return block

# 空间变换网络类
class STN(nn.Layer):
    # 初始化空间变换网络(STN)模块，设置输入通道数、控制点数量和激活函数类型
    def __init__(self, in_channels, num_ctrlpoints, activation='none'):
        # 调用父类的初始化方法
        super(STN, self).__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置控制点数量
        self.num_ctrlpoints = num_ctrlpoints
        # 设置激活函数类型
        self.activation = activation
        # 定义STN的卷积神经网络结构
        self.stn_convnet = nn.Sequential(
            # 使用3x3的卷积块，输入通道数为in_channels，输出通道数为32
            conv3x3_block(in_channels, 32),  #32x64
            # 最大池化层，核大小为2，步长为2
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            # 使用3x3的卷积块，输入通道数为32，输出通道数为64
            conv3x3_block(32, 64),  #16x32
            # 最大池化层，核大小为2，步长为2
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            # 使用3x3的卷积块，输入通道数为64，输出通道数为128
            conv3x3_block(64, 128),  # 8*16
            # 最大池化层，核大小为2，步长为2
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            # 使用3x3的卷积块，输入通道数为128，输出通道数为256
            conv3x3_block(128, 256),  # 4*8
            # 最大池化层，核大小为2，步长为2
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            # 使用3x3的卷积块，输入通道数为256，输出通道数为256
            conv3x3_block(256, 256),  # 2*4,
            # 最大池化层，核大小为2，步长为2
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            # 使用3x3的卷积块，输入通道数为256，输出通道数为256
            conv3x3_block(256, 256))  # 1*2
        # 定义STN的全连接层1
        self.stn_fc1 = nn.Sequential(
            # 全连接层，输入维度为2*256，输出维度为512
            nn.Linear(
                2 * 256,
                512,
                weight_attr=nn.initializer.Normal(0, 0.001),
                bias_attr=nn.initializer.Constant(0)),
            # 一维批量归一化层
            nn.BatchNorm1D(512),
            # ReLU激活函数
            nn.ReLU())
        # 初始化STN的全连接层2的偏置
        fc2_bias = self.init_stn()
        # 定义STN的全连接层2
        self.stn_fc2 = nn.Linear(
            # 全连接层，输入维度为512，输出维度为num_ctrlpoints * 2
            512,
            num_ctrlpoints * 2,
            weight_attr=nn.initializer.Constant(0.0),
            bias_attr=nn.initializer.Assign(fc2_bias))
    # 初始化空间变换网络的控制点
    def init_stn(self):
        # 设置边距
        margin = 0.01
        # 计算每侧采样点的数量
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        # 在 x 轴上均匀采样控制点
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        # 在顶部设置控制点的 y 值
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        # 在底部设置控制点的 y 值
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        # 组合顶部和底部的控制点
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        # 根据激活函数类型对控制点进行处理
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        # 将控制点转换为张量
        ctrl_points = paddle.to_tensor(ctrl_points)
        # 重塑控制点张量
        fc2_bias = paddle.reshape(
            ctrl_points, shape=[ctrl_points.shape[0] * ctrl_points.shape[1])
        return fc2_bias

    # 空间变换网络的前向传播
    def forward(self, x):
        # 经过卷积网络
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.shape
        # 重塑张量形状
        x = paddle.reshape(x, shape=(batch_size, -1))
        # 经过全连接层1
        img_feat = self.stn_fc1(x)
        # 经过全连接层2
        x = self.stn_fc2(0.1 * img_feat)
        # 根据激活函数类型对输出进行处理
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        # 重塑输出张量形状
        x = paddle.reshape(x, shape=[-1, self.num_ctrlpoints, 2])
        return img_feat, x
class STN_ON(nn.Layer):
    # 定义 STN_ON 类，继承自 nn.Layer
    def __init__(self, in_channels, tps_inputsize, tps_outputsize,
                 num_control_points, tps_margins, stn_activation):
        # 初始化函数，接受输入通道数、TPS 输入大小、TPS 输出大小、控制点数量、TPS 边距、STN 激活函数
        super(STN_ON, self).__init__()
        # 调用父类的初始化函数
        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(tps_outputsize),
            num_control_points=num_control_points,
            margins=tuple(tps_margins))
        # 创建 TPSSpatialTransformer 实例，设置输出图像大小、控制点数量、边距
        self.stn_head = STN(in_channels=in_channels,
                            num_ctrlpoints=num_control_points,
                            activation=stn_activation)
        # 创建 STN 实例，设置输入通道数、控制点数量、激活函数
        self.tps_inputsize = tps_inputsize
        # 设置 TPS 输入大小
        self.out_channels = in_channels
        # 设置输出通道数为输入通道数

    def forward(self, image):
        # 前向传播函数，接受输入图像
        stn_input = paddle.nn.functional.interpolate(
            image, self.tps_inputsize, mode="bilinear", align_corners=True)
        # 对输入图像进行插值，调整大小为 TPS 输入大小，使用双线性插值，保持角点对齐
        stn_img_feat, ctrl_points = self.stn_head(stn_input)
        # 使用 STN 头部处理插值后的图像，得到特征图和控制点
        x, _ = self.tps(image, ctrl_points)
        # 使用 TPS 对输入图像和控制点进行变换
        return x
        # 返回变换后的图像
```