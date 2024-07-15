# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\layers_33966KB.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数模块，如 F.interpolate 和 nn.functional
import torch.nn.functional as F
# 从 torch 中导入 nn 模块
from torch import nn
# 从当前目录下导入 spec_utils 模块
from . import spec_utils


# 定义一个卷积层、批标准化、激活函数的模块类
class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            # 二维卷积层，使用指定的参数
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            # 二维批标准化层，标准化输出特征图
            nn.BatchNorm2d(nout),
            # 指定的激活函数，如 ReLU
            activ(),
        )

    def __call__(self, x):
        # 对输入 x 应用定义好的卷积层、批标准化、激活函数
        return self.conv(x)


# 定义一个可分离卷积层、批标准化、激活函数的模块类
class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            # 深度可分离卷积，首先在通道维度上进行卷积
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False,
            ),
            # 再在空间维度上进行卷积，将通道混合特征映射为输出特征映射
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),
            # 二维批标准化层，标准化输出特征图
            nn.BatchNorm2d(nout),
            # 指定的激活函数，如 ReLU
            activ(),
        )

    def __call__(self, x):
        # 对输入 x 应用定义好的可分离卷积层、批标准化、激活函数
        return self.conv(x)


# 定义一个编码器模块类，包含两个卷积层、批标准化、激活函数
class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        # 第一个卷积层模块，输入 nin，输出 nout
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # 第二个卷积层模块，输入 nout，输出 nout
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        # 应用第一个卷积层模块，得到 skip 连接的输出特征图
        skip = self.conv1(x)
        # 应用第二个卷积层模块，得到最终的编码器输出特征图 h
        h = self.conv2(skip)

        return h, skip


# 定义一个解码器模块类，包含一个卷积层、批标准化、激活函数以及可能的 dropout
class Decoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        # 卷积层、批标准化、激活函数的组合
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # 如果 dropout=True，则使用二维 dropout
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        # 使用双线性插值进行上采样
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # 如果提供了 skip 连接，则进行中心裁剪并与上采样结果连接
        if skip is not None:
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        # 应用卷积层、批标准化、激活函数
        h = self.conv(x)

        # 如果启用 dropout，则应用 dropout
        if self.dropout is not None:
            h = self.dropout(h)

        return h


# 定义 ASPP（空间金字塔池化）模块类，用于语义分割任务
class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 16, 32, 64), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        # 第一个卷积层包括自适应平均池化和一个 1x1 的卷积操作，用于多尺度空洞卷积的ASPP模块
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 自适应平均池化层，水平方向为1，垂直方向无限制
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),  # 1x1的卷积层，激活函数为ReLU
        )
        # 第二个卷积层，包含一个 1x1 的卷积操作
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        # 第三个卷积层，包含一个空洞卷积操作，使用第一个空洞率
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )
        # 第四个卷积层，包含一个空洞卷积操作，使用第二个空洞率
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )
        # 第五个卷积层，包含一个空洞卷积操作，使用第三个空洞率
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        # 第六个卷积层，包含一个空洞卷积操作，使用第四个空洞率
        self.conv6 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[3], dilations[3], activ=activ
        )
        # 第七个卷积层，包含一个空洞卷积操作，使用第五个空洞率
        self.conv7 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[4], dilations[4], activ=activ
        )
        # 瓶颈层，包含一个 1x1 的卷积操作和一个 dropout 操作
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 7, nout, 1, 1, 0, activ=activ),  # 1x1的卷积层，输出通道数为nout
            nn.Dropout2d(0.1)  # 二维dropout操作，丢弃率为0.1
        )

    def forward(self, x):
        _, _, h, w = x.size()
        # 对输入进行自适应平均池化后插值到原始尺寸，作为第一个特征
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        # 第二个特征，通过第二个卷积层得到
        feat2 = self.conv2(x)
        # 第三个特征，通过第三个卷积层得到
        feat3 = self.conv3(x)
        # 第四个特征，通过第四个卷积层得到
        feat4 = self.conv4(x)
        # 第五个特征，通过第五个卷积层得到
        feat5 = self.conv5(x)
        # 第六个特征，通过第六个卷积层得到
        feat6 = self.conv6(x)
        # 第七个特征，通过第七个卷积层得到
        feat7 = self.conv7(x)
        # 将所有特征在通道维度上连接起来
        out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        # 通过瓶颈层得到最终输出
        bottle = self.bottleneck(out)
        return bottle
```