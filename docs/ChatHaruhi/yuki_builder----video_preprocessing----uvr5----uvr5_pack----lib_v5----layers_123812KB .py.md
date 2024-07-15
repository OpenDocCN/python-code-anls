# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\layers_123812KB .py`

```py
# 导入 PyTorch 库
import torch
# 导入神经网络模块和函数模块
from torch import nn
import torch.nn.functional as F

# 从当前目录导入 spec_utils 模块
from . import spec_utils


# 定义一个包含卷积、批归一化和激活函数的模块
class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        # 依次构建卷积层、批归一化层和激活函数层
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
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        # 对输入 x 应用 Conv2DBNActiv 模块
        return self.conv(x)


# 定义一个包含深度可分离卷积、批归一化和激活函数的模块
class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        # 依次构建深度可分离卷积层、标准卷积层、批归一化层和激活函数层
        self.conv = nn.Sequential(
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
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        # 对输入 x 应用 SeperableConv2DBNActiv 模块
        return self.conv(x)


# 定义一个编码器模块
class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        # 构建第一个卷积、批归一化和激活函数模块
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # 构建第二个卷积、批归一化和激活函数模块
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        # 对输入 x 进行编码器操作，返回中间结果和跳跃连接
        skip = self.conv1(x)
        h = self.conv2(skip)

        return h, skip


# 定义一个解码器模块
class Decoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        # 构建卷积、批归一化和激活函数模块
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # 如果需要，添加 dropout 层
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        # 对输入 x 进行解码器操作
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            # 利用 spec_utils 模块进行中心裁剪和连接
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h


class ASPPModule(nn.Module):
    # 初始化 ASPPModule 类，设置输入通道数 nin、输出通道数 nout 和扩张率 dilations，默认激活函数为 nn.ReLU
    def __init__(self, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        # 调用父类构造函数初始化模块
        super(ASPPModule, self).__init__()

        # 第一个卷积层序列，包含自适应平均池化层和一个 1x1 卷积层
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 自适应平均池化层，输出尺寸为 (1, None)
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),  # 1x1 卷积层，输入输出通道数均为 nin
        )

        # 第二个卷积层，一个简单的 1x1 卷积层
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)  # 1x1 卷积层，输入输出通道数均为 nin

        # 第三个卷积层，深度可分离卷积，带有扩张率 dilations[0]
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )  # 3x3 深度可分离卷积，输入输出通道数均为 nin，扩张率为 dilations[0]

        # 第四个卷积层，深度可分离卷积，带有扩张率 dilations[1]
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )  # 3x3 深度可分离卷积，输入输出通道数均为 nin，扩张率为 dilations[1]

        # 第五个卷积层，深度可分离卷积，带有扩张率 dilations[2]
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )  # 3x3 深度可分离卷积，输入输出通道数均为 nin，扩张率为 dilations[2]

        # 瓶颈层序列，包含一个 1x1 卷积层和一个 2D Dropout 层
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 5, nout, 1, 1, 0, activ=activ),  # 1x1 卷积层，输入通道数为 nin*5，输出通道数为 nout
            nn.Dropout2d(0.1)  # 二维 Dropout 层，丢弃概率为 0.1
        )

    # 定义前向传播方法，接受输入 x，返回处理后的结果 bottle
    def forward(self, x):
        _, _, h, w = x.size()  # 获取输入 x 的高度 h 和宽度 w
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )  # 对输入 x 进行卷积操作后，进行双线性插值上采样，尺寸为 (h, w)
        feat2 = self.conv2(x)  # 对输入 x 进行第二个卷积操作
        feat3 = self.conv3(x)  # 对输入 x 进行第三个卷积操作
        feat4 = self.conv4(x)  # 对输入 x 进行第四个卷积操作
        feat5 = self.conv5(x)  # 对输入 x 进行第五个卷积操作
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)  # 在通道维度上连接 feat1 到 feat5
        bottle = self.bottleneck(out)  # 将连接结果 out 输入到瓶颈层进行处理
        return bottle  # 返回瓶颈层处理后的结果
```