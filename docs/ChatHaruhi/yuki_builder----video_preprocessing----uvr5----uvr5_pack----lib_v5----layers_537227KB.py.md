# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\layers_537227KB.py`

```py
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch中导入神经网络模块
import torch.nn.functional as F  # 从PyTorch中导入函数操作模块

from . import spec_utils  # 从当前目录导入特定工具模块


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,  # 卷积核大小
                stride=stride,  # 步长
                padding=pad,  # 填充
                dilation=dilation,  # 膨胀
                bias=False,  # 是否包含偏置
            ),
            nn.BatchNorm2d(nout),  # 批标准化层
            activ(),  # 激活函数
        )

    def __call__(self, x):
        return self.conv(x)  # 返回卷积操作的结果


class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,  # 卷积核大小
                stride=stride,  # 步长
                padding=pad,  # 填充
                dilation=dilation,  # 膨胀
                groups=nin,  # 分组卷积数
                bias=False,  # 是否包含偏置
            ),
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),  # 1x1卷积核
            nn.BatchNorm2d(nout),  # 批标准化层
            activ(),  # 激活函数
        )

    def __call__(self, x):
        return self.conv(x)  # 返回分离卷积操作的结果


class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)  # 第一个卷积层
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)  # 第二个卷积层

    def __call__(self, x):
        skip = self.conv1(x)  # 第一个卷积层的输出
        h = self.conv2(skip)  # 第二个卷积层的输出

        return h, skip  # 返回第二个卷积层的输出和第一个卷积层的输出（跳跃连接）


class Decoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)  # 卷积层
        self.dropout = nn.Dropout2d(0.1) if dropout else None  # 可选的2D dropout层

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)  # 双线性插值上采样
        if skip is not None:
            skip = spec_utils.crop_center(skip, x)  # 裁剪中心部分以匹配尺寸
            x = torch.cat([x, skip], dim=1)  # 拼接张量
        h = self.conv(x)  # 卷积操作

        if self.dropout is not None:
            h = self.dropout(h)  # dropout操作

        return h  # 返回结果张量


class ASPPModule(nn.Module):
    pass  # 空的ASPP模块类，待实现具体功能
    def __init__(self, nin, nout, dilations=(4, 8, 16, 32, 64), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        # 第一个卷积层，使用自适应平均池化作为输入，然后进行 1x1 的卷积
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),
        )
        # 第二个卷积层，进行 1x1 的卷积操作
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        # 第三个卷积层，使用可分离卷积（depthwise separable convolution），包括扩张率参数
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )
        # 第四个卷积层，同样使用可分离卷积，不同的扩张率参数
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )
        # 第五个卷积层，使用可分离卷积，另外的扩张率参数
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        # 第六个卷积层，使用可分离卷积，与第五个层相同的扩张率参数
        self.conv6 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        # 第七个卷积层，使用可分离卷积，与第五个层相同的扩张率参数
        self.conv7 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        # 瓶颈层，将所有卷积层的输出合并后进行 1x1 的卷积，再加上 dropout 操作
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 7, nout, 1, 1, 0, activ=activ), nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        # 对输入进行自适应平均池化后插值回原始大小，并进行双线性插值
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        # 直接通过第二个卷积层
        feat2 = self.conv2(x)
        # 第三到第七个卷积层的输出
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        feat6 = self.conv6(x)
        feat7 = self.conv7(x)
        # 将所有特征图拼接在一起
        out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        # 通过瓶颈层得到最终输出
        bottle = self.bottleneck(out)
        return bottle
```