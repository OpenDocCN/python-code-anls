# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\layers_537238KB.py`

```py
# 导入 PyTorch 库
import torch
# 导入神经网络模块和函数模块
from torch import nn
import torch.nn.functional as F

# 从当前目录下的 spec_utils 模块中导入全部内容
from . import spec_utils


# 定义包含卷积、批归一化和激活函数的模块
class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(nout),  # 添加批归一化层
            activ(),  # 添加激活函数
        )

    def __call__(self, x):
        return self.conv(x)


# 定义包含深度可分离卷积、批归一化和激活函数的模块
class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,  # 设置分组卷积的组数
                bias=False,
            ),
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),  # 添加 1x1 卷积层
            nn.BatchNorm2d(nout),  # 添加批归一化层
            activ(),  # 添加激活函数
        )

    def __call__(self, x):
        return self.conv(x)


# 定义编码器模块
class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        # 第一个卷积层，包含卷积、批归一化和激活函数
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # 第二个卷积层，包含卷积、批归一化和激活函数
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        # 运行第一个卷积层
        skip = self.conv1(x)
        # 运行第二个卷积层
        h = self.conv2(skip)

        return h, skip


# 定义解码器模块
class Decoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        # 卷积、批归一化和激活函数
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # 可选的 2D Dropout 层
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        # 使用双线性插值对输入的特征图进行上采样
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            # 利用 spec_utils 模块中的函数对 skip 进行裁剪
            skip = spec_utils.crop_center(skip, x)
            # 将裁剪后的 skip 和上采样后的特征图 x 拼接在一起
            x = torch.cat([x, skip], dim=1)
        # 应用卷积、批归一化和激活函数
        h = self.conv(x)

        if self.dropout is not None:
            # 如果存在 Dropout 层，应用 Dropout
            h = self.dropout(h)

        return h


class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 16, 32, 64), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        
        # 第一个卷积模块，包括自适应平均池化层和卷积层
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 自适应平均池化到大小为 (1, None)
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),  # 1x1卷积层
        )
        
        # 第二个卷积模块，简单的1x1卷积层
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        
        # 第三个卷积模块，深度可分离卷积层，使用给定的扩张率
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )
        
        # 第四个卷积模块，深度可分离卷积层，使用给定的扩张率
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )
        
        # 第五个卷积模块，深度可分离卷积层，使用给定的扩张率
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        
        # 第六个卷积模块，深度可分离卷积层，使用给定的扩张率（与第五个相同）
        self.conv6 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        
        # 第七个卷积模块，深度可分离卷积层，使用给定的扩张率（与第五个相同）
        self.conv7 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        
        # 瓶颈层，包括1x1卷积层和2D Dropout层
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 7, nout, 1, 1, 0, activ=activ),  # 1x1卷积层
            nn.Dropout2d(0.1)  # 2D Dropout层
        )

    def forward(self, x):
        _, _, h, w = x.size()
        
        # 对输入特征进行自适应平均池化，然后使用双线性插值恢复到原始大小
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        
        # 第二个卷积模块的输出
        feat2 = self.conv2(x)
        
        # 第三个卷积模块的输出
        feat3 = self.conv3(x)
        
        # 第四个卷积模块的输出
        feat4 = self.conv4(x)
        
        # 第五个卷积模块的输出
        feat5 = self.conv5(x)
        
        # 第六个卷积模块的输出
        feat6 = self.conv6(x)
        
        # 第七个卷积模块的输出
        feat7 = self.conv7(x)
        
        # 将所有特征拼接在一起，沿着通道维度
        out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        
        # 通过瓶颈层进行最终的特征变换
        bottle = self.bottleneck(out)
        
        # 返回最终输出
        return bottle
```