# `.\PaddleOCR\ppocr\modeling\transforms\tsrn.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码来源于：
# https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/model/tsrn.py

# 导入必要的库
import math
import paddle
import paddle.nn.functional as F
from paddle import nn
from collections import OrderedDict
import sys
import numpy as np
import warnings
import math, copy
import cv2

# 忽略警告信息
warnings.filterwarnings("ignore")

# 导入自定义模块
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn import STN as STN_model
from ppocr.modeling.heads.sr_rensnet_transformer import Transformer

# 定义 TSRN 类，继承自 nn.Layer
class TSRN(nn.Layer):
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 初始化输出字典
        output = {}
        # 如果处于推断模式
        if self.infer_mode:
            # 将输入 x 存入输出字典中的 "lr_img" 键
            output["lr_img"] = x
            # y 等于 x
            y = x
        else:
            # 将输入 x 的第一个元素存入输出字典中的 "lr_img" 键
            output["lr_img"] = x[0]
            # 将输入 x 的第二个元素存入输出字典中的 "hr_img" 键
            output["hr_img"] = x[1]
            # y 等于 x 的第一个元素
            y = x[0]
        # 如果启用空间变换网络并且处于训练模式
        if self.stn and self.training:
            # 获取空间变换网络的控制点
            _, ctrl_points_x = self.stn_head(y)
            # 对输入 y 进行空间变换
            y, _ = self.tps(y, ctrl_points_x)
        # 创建字典，存储每个块的输出
        block = {'1': self.block1(y)}
        # 循环遍历每个残差块
        for i in range(self.srb_nums + 1):
            # 将每个残差块的输出存入字典中
            block[str(i + 2)] = getattr(self,
                                        'block%d' % (i + 2))(block[str(i + 1)])

        # 计算最后一个残差块的输出
        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))

        # 对最终的超分辨率图像进行 tanh 激活
        sr_img = paddle.tanh(block[str(self.srb_nums + 3)])

        # 将超分辨率图像存入输出字典中的 "sr_img" 键
        output["sr_img"] = sr_img

        # 如果处于训练模式
        if self.training:
            # 获取输入 x 的第二个元素作为高分辨率图像
            hr_img = x[1]
            # 获取输入 x 的第三个元素作为长度
            length = x[2]
            # 获取输入 x 的第四个元素作为输入张量
            input_tensor = x[3]

            # 使用 R34 Transformer 处理超分辨率图像和高分辨率图像
            sr_pred, word_attention_map_pred, _ = self.r34_transformer(
                sr_img, length, input_tensor)

            hr_pred, word_attention_map_gt, _ = self.r34_transformer(
                hr_img, length, input_tensor)

            # 将高分辨率图像、高分辨率预测、注意力图等存入输出字典中
            output["hr_img"] = hr_img
            output["hr_pred"] = hr_pred
            output["word_attention_map_gt"] = word_attention_map_gt
            output["sr_pred"] = sr_pred
            output["word_attention_map_pred"] = word_attention_map_pred

        # 返回输出字典
        return output
class RecurrentResidualBlock(nn.Layer):
    # 定义一个循环残差块的类，继承自nn.Layer
    def __init__(self, channels):
        # 初始化函数，接受通道数作为参数
        super(RecurrentResidualBlock, self).__init__()
        # 调用父类的初始化函数
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        # 定义一个卷积层，输入输出通道数相同，卷积核大小为3，填充为1
        self.bn1 = nn.BatchNorm2D(channels)
        # 定义一个2D批归一化层
        self.gru1 = GruBlock(channels, channels)
        # 定义一个GRU块
        self.prelu = mish()
        # 定义一个激活函数
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        # 定义另一个卷积层
        self.bn2 = nn.BatchNorm2D(channels)
        # 定义另一个2D批归一化层
        self.gru2 = GruBlock(channels, channels)
        # 定义另一个GRU块

    def forward(self, x):
        # 定义前向传播函数，接受输入x
        residual = self.conv1(x)
        # 使用第一个卷积层对输入x进行卷积
        residual = self.bn1(residual)
        # 对卷积结果进行批归一化
        residual = self.prelu(residual)
        # 使用激活函数对结果进行激活
        residual = self.conv2(residual)
        # 使用第二个卷积层对结果进行卷积
        residual = self.bn2(residual)
        # 对卷积结果进行批归一化
        residual = self.gru1(residual.transpose([0, 1, 3, 2])).transpose([0, 1, 3, 2])
        # 对卷积结果进行转置操作，并将结果输入到第一个GRU块中
        return self.gru2(x + residual)
        # 返回第二个GRU块的输出结果与输入x加和的结果


class UpsampleBLock(nn.Layer):
    # 定义一个上采样块的类，继承自nn.Layer
    def __init__(self, in_channels, up_scale):
        # 初始化函数，接受输入通道数和上采样比例作为参数
        super(UpsampleBLock, self).__init__()
        # 调用父类的初始化函数
        self.conv = nn.Conv2D(in_channels, in_channels * up_scale**2, kernel_size=3, padding=1)
        # 定义一个卷积层，输入通道数不变，输出通道数为输入通道数乘以上采样比例的平方，卷积核大小为3，填充为1
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # 定义一个像素重排层，上采样比例为up_scale
        self.prelu = mish()
        # 定义一个激活函数

    def forward(self, x):
        # 定义前向传播函数，接受输入x
        x = self.conv(x)
        # 使用卷积层对输入x进行卷积
        x = self.pixel_shuffle(x)
        # 使用像素重排层对卷积结果进行像素重排
        x = self.prelu(x)
        # 使用激活函数对结果进行激活
        return x
        # 返回结果


class mish(nn.Layer):
    # 定义一个mish激活函数的类，继承自nn.Layer
    def __init__(self, ):
        # 初始化函数
        super(mish, self).__init__()
        # 调用父类的初始化函数
        self.activated = True
        # 初始化一个激活标志为True

    def forward(self, x):
        # 定义前向传播函数，接受输入x
        if self.activated:
            # 如果激活标志为True
            x = x * (paddle.tanh(F.softplus(x)))
            # 对输入x进行mish激活函数操作
        return x
        # 返回结果


class GruBlock(nn.Layer):
    # 定义一个GRU块的类，继承自nn.Layer
    def __init__(self, in_channels, out_channels):
        # 初始化函数，接受输入通道数和输出通道数作为参数
        super(GruBlock, self).__init__()
        # 调用父类的初始化函数
        assert out_channels % 2 == 0
        # 断言输出通道数为偶数
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        # 定义一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为1，填充为0
        self.gru = nn.GRU(out_channels, out_channels // 2, direction='bidirectional')
        # 定义一个双向GRU层，输入输出通道数为out_channels，隐藏单元数为out_channels的一半
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # x: b, c, w, h，表示输入的形状为 batch_size, channels, width, height
        # 使用卷积层处理输入 x
        x = self.conv1(x)
        # 将 x 的维度进行转置，变为 b, w, h, c
        x = x.transpose([0, 2, 3, 1])
        # 获取转置后 x 的形状信息
        batch_size, w, h, c = x.shape
        # 将 x 重塑为 b*w, h, c 的形状
        x = x.reshape([-1, h, c])
        # 使用 GRU 神经网络处理 x
        x, _ = self.gru(x)
        # 将处理后的 x 重新调整为 b, w, h, c 的形状
        x = x.reshape([-1, w, h, c])
        # 将 x 的维度再次进行转置，变为 b, c, w, h
        x = x.transpose([0, 3, 1, 2])
        # 返回处理后的 x
        return x
```