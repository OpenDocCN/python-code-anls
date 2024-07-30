# `.\yolov8\ultralytics\nn\modules\block.py`

```py
# 导入PyTorch库中的必要模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入自定义的工具函数fuse_conv_and_bn
from ultralytics.utils.torch_utils import fuse_conv_and_bn

# 从当前目录下的conv.py文件中导入以下模块
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad

# 从当前目录下的transformer.py文件中导入TransformerBlock模块
from .transformer import TransformerBlock

# 定义模块的公开接口，列出所有公开的类和函数
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
)

# 定义DFL类，继承自nn.Module，实现分布式焦点损失的核心模块
class DFL(nn.Module):
    """
    Distribution Focal Loss（DFL）的核心模块。

    提出于《Generalized Focal Loss》https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """
        初始化具有指定输入通道数的卷积层。

        Args:
            c1 (int): 输入通道数
        """
        super().__init__()
        # 创建一个卷积层，输入通道数为c1，输出通道数为1，卷积核大小为1x1，没有偏置项
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # 初始化卷积核权重为1到c1的浮点数序列
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        对输入张量'x'应用转换层，并返回张量。

        Args:
            x (tensor): 输入张量，维度为[b, c1, 4, a]

        Returns:
            tensor: 转换后的张量，维度为[b, 4, a]
        """
        b, _, a = x.shape  # 批量大小，通道数，锚点数
        # 对输入进行形状变换，然后进行softmax操作，最后再次进行形状变换
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

# 定义Proto类，继承自nn.Module，用于YOLOv8的掩膜Proto模块，适用于分割模型
class Proto(nn.Module):
    """
    YOLOv8分割模型的掩膜Proto模块。
    """

    def __init__(self, c1, c_=256, c2=32):
        """
        初始化YOLOv8掩膜Proto模块，指定原型和掩膜的数量。

        Args:
            c1 (int): 输入通道数
            c_ (int): 原型通道数，默认为256
            c2 (int): 掩膜通道数，默认为32
        """
        super().__init__()
        # 第一个卷积层，输入通道数为c1，输出通道数为c_，卷积核大小为3x3
        self.cv1 = Conv(c1, c_, k=3)
        # 上采样层，将输入特征图上采样，输出通道数不变，采样因子为2
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        # 第二个卷积层，输入通道数和输出通道数都为c_，卷积核大小为3x3
        self.cv2 = Conv(c_, c_, k=3)
        # 第三个卷积层，输入通道数为c_，输出通道数为c2，卷积核大小为3x3
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        通过各层进行前向传播，使用上采样的输入图像。

        Args:
            x (tensor): 输入张量

        Returns:
            tensor: 经过Proto模块处理后的张量
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    PPHGNetV2的StemBlock，包含5个卷积层和一个maxpool2d层。

    参考：https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    def __init__(self, c1, cm, c2):
        """
        Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling.
        """
        super().__init__()  # 调用父类的初始化方法

        # 第一个卷积层，输入通道数为c1，输出通道数为cm，卷积核大小为3x3，步长为2，激活函数为ReLU
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())

        # 第二个卷积层分支a，输入通道数为cm，输出通道数为cm//2，卷积核大小为2x2，步长为1，无填充，激活函数为ReLU
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())

        # 第二个卷积层分支b，输入通道数为cm//2，输出通道数为cm，卷积核大小为2x2，步长为1，无填充，激活函数为ReLU
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())

        # 第三个卷积层，输入通道数为2*cm，输出通道数为cm，卷积核大小为3x3，步长为2，激活函数为ReLU
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())

        # 第四个卷积层，输入通道数为cm，输出通道数为c2，卷积核大小为1x1，步长为1，激活函数为ReLU
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())

        # 最大池化层，核大小为2x2，步长为1，无填充，且在计算输出大小时采用ceil模式
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """
        Forward pass of a PPHGNetV2 backbone layer.
        """
        x = self.stem1(x)  # 经过第一个卷积层
        x = F.pad(x, [0, 1, 0, 1])  # 在x的右侧和底部各填充1列/行零
        x2 = self.stem2a(x)  # 经过第二个卷积层分支a
        x2 = F.pad(x2, [0, 1, 0, 1])  # 在x2的右侧和底部各填充1列/行零
        x2 = self.stem2b(x2)  # 经过第二个卷积层分支b
        x1 = self.pool(x)  # 对x进行最大池化
        x = torch.cat([x1, x2], dim=1)  # 将x1和x2在通道维度上拼接
        x = self.stem3(x)  # 经过第三个卷积层
        x = self.stem4(x)  # 经过第四个卷积层
        return x  # 返回处理后的结果x
# HG_Block类定义，用于PaddleDetection中PPHGNetV2的一个模块，包含两个卷积层和LightConv。
class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        # 创建包含n个LightConv或Conv模块的ModuleList，第一个模块使用c1通道，其余使用cm通道
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        # squeeze conv，将输入c1和n*cm的通道数合并为c2/2，使用1x1卷积进行处理
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        # excitation conv，将c2/2通道数处理为c2，使用1x1卷积进行处理
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        # 如果设置了shortcut且输入输出通道数相同，则将输入添加到输出中
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        # 对输入x进行一系列的n次卷积操作，并将结果存储在列表y中
        y.extend(m(y[-1]) for m in self.m)
        # 将列表y中的所有特征图连接起来，然后依次经过squeeze conv和excitation conv操作得到最终输出y
        y = self.ec(self.sc(torch.cat(y, 1)))
        # 如果add为True，则将最终输出y与输入x相加作为最终输出；否则直接返回y
        return y + x if self.add else y


# SPP类定义，实现空间金字塔池化（SPP）层，参考论文https://arxiv.org/abs/1406.4729
class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # 第一个卷积层，将输入通道c1压缩为c_通道
        self.cv1 = Conv(c1, c_, 1, 1)
        # 第二个卷积层，将c_*(len(k)+1)个通道压缩为c2个通道
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 创建一系列的最大池化层，每个层的kernel_size分别为k中的元素，stride和padding设置为1
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        # 经过第一个卷积层cv1处理输入x
        x = self.cv1(x)
        # 将经过cv1的特征图x与一系列池化结果连接起来，并经过第二个卷积层cv2处理，得到最终的输出特征图
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# SPPF类定义，实现快速的空间金字塔池化（SPPF）层，用于YOLOv5
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # 第一个卷积层，将输入通道c1压缩为c_通道
        self.cv1 = Conv(c1, c_, 1, 1)
        # 第二个卷积层，将c_*4个通道压缩为c2个通道
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # 创建一个最大池化层，kernel_size为k，stride和padding设置为k//2
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        # 经过第一个卷积层cv1处理输入x
        y = [self.cv1(x)]
        # 对cv1处理后的特征图进行3次最大池化操作，并将结果连接起来，然后经过第二个卷积层cv2处理，得到最终的输出特征图
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


# C1类定义，实现具有1个卷积的CSP Bottleneck
class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        # 第一个卷积层，将输入通道c1压缩为c2通道
        self.cv1 = Conv(c1, c2, 1, 1)
        # 创建一个序列模块，包含n个3x3卷积层
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        # 经过第一个卷积层cv1处理输入x，然后经过一系列的3x3卷积层m，将结果与cv1的输出相加作为最终输出
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """This class is incomplete and requires further implementation."""

    # 此处为未完整的类定义，需要进一步实现
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化具有两个卷积层的CSP Bottleneck模块，参数包括输入通道数c1、输出通道数c2、重复次数n、是否使用shortcut、分组数g、扩展系数e。"""
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 第一个卷积层，输入c1通道，输出2*self.c通道，卷积核大小1x1
        self.cv2 = Conv(2 * self.c, c2, 1)  # 第二个卷积层，输入2*self.c通道，输出c2通道，卷积核大小1x1（可选使用激活函数FReLU(c2)）
        # self.attention = ChannelAttention(2 * self.c)  # 或者使用空间注意力（SpatialAttention）
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
        # 使用n个重复的Bottleneck块，每个块输入输出都是self.c通道，是否使用shortcut由参数shortcut决定，分组数为g，内部卷积核大小为(3x3, 3x3)，扩展系数为1.0

    def forward(self, x):
        """通过CSP Bottleneck进行前向传播，包含两个卷积层。"""
        a, b = self.cv1(x).chunk(2, 1)  # 将第一个卷积层的输出分为两部分，分别为a和b
        return self.cv2(torch.cat((self.m(a), b), 1))
        # 将a经过重复的Bottleneck块处理后与b拼接，再经过第二个卷积层cv2处理，返回结果
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # First convolution layer
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Second convolution layer with optional activation
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split output of first convolution into 2 chunks
        y.extend(m(y[-1]) for m in self.m)  # Apply each Bottleneck module to the last chunk
        return self.cv2(torch.cat(y, 1))  # Concatenate all chunks and pass through second convolution

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))  # Split output of first convolution into 2 parts
        y.extend(m(y[-1]) for m in self.m)  # Apply each Bottleneck module to the last part
        return self.cv2(torch.cat(y, 1))  # Concatenate all parts and pass through second convolution


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # First convolution layer
        self.cv2 = Conv(c1, c_, 1, 1)  # Second convolution layer
        self.cv3 = Conv(2 * c_, c2, 1)  # Third convolution layer with optional activation
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))  # Sequence of Bottleneck modules

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))  # Concatenate outputs and pass through third convolution


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))  # Sequence of cross Bottleneck modules


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)  # First convolution layer
        self.cv2 = Conv(c1, c2, 1, 1)  # Second convolution layer
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])  # Sequence of RepConv modules
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()  # Third convolution layer or Identity if channels match

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))  # Combine outputs using addition and pass through third convolution


class C3TR(C3):
    """C3 module with TransformerBlock()."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3Ghost module with GhostBottleneck().
        """
        # 调用父类的构造函数进行初始化
        super().__init__(c1, c2, n, shortcut, g, e)
        # 计算降维后的通道数
        c_ = int(c2 * e)
        # 创建一个 TransformerBlock 对象，并赋值给 self.m
        self.m = TransformerBlock(c_, c_, 4, n)
class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # 使用GhostBottleneck构建一个序列模块，重复n次
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        # 构建一个序列的卷积模块
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            # 如果stride为2，则使用深度卷积层；否则使用恒等映射
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        # 如果stride为2，则创建一个深度卷积和卷积的序列作为shortcut；否则使用恒等映射
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        # 应用卷积操作和shortcut，然后将结果相加
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 创建两个卷积层，然后根据shortcut标志决定是否添加输入和输出的相加
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # 如果add标志为True，则添加输入和输出；否则直接返回输出
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 创建四个卷积层和一个Batch Normalization层，并应用激活函数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # 使用Bottleneck构建一个序列模块，重复n次
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        # 对输入x应用CSP bottleneck操作，返回处理后的张量
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""
    # 定义一个卷积块的初始化方法，接受输入参数 c1, c2，可选参数 s 和 e，默认值为 s=1, e=4
    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        # 调用父类的初始化方法
        super().__init__()
        # 计算第三个卷积层的输出通道数
        c3 = e * c2
        # 定义第一个卷积层，1x1 卷积，输入通道数为 c1，输出通道数为 c2，步长为 1，激活函数为 ReLU
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        # 定义第二个卷积层，3x3 卷积，输入输出通道数均为 c2，步长为 s，填充为 1，激活函数为 ReLU
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        # 定义第三个卷积层，1x1 卷积，输入通道数为 c2，输出通道数为 c3，不使用激活函数
        self.cv3 = Conv(c2, c3, k=1, act=False)
        # 根据是否需要进行下采样（s != 1 或者 c1 != c3），定义 shortcut 分支
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        # 实现 ResNet 块的前向传播过程
        # 先经过 cv1 -> cv2 -> cv3 的卷积处理，然后加上 shortcut 分支的结果，最后使用 ReLU 激活函数
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))
class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            # 如果是第一层，则使用序列模块包含一个卷积层和最大池化层
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            # 如果不是第一层，则创建多个ResNetBlock组成的序列模块
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))
    # 定义一个方法 forward_split，用于执行前向传播，使用 split() 替代 chunk() 方法。
    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        # 将输入 x 经过第一个卷积层 self.cv1，并使用 split 方法按照指定大小分割，得到 y 列表
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 对 y 列表中最后一个元素应用每个模块 self.m 中的函数，并将结果扩展到 y 列表中
        y.extend(m(y[-1]) for m in self.m)
        # 将 self.attn 应用于 y 列表中的最后一个元素和指南 guide，将结果添加到 y 列表
        y.append(self.attn(y[-1], guide))
        # 将 y 列表中的所有元素在维度 1 上进行拼接，然后经过第二个卷积层 self.cv2 处理，并返回结果
        return self.cv2(torch.cat(y, 1))
class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        # Define layers for query, key, and value transformations
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        
        # Projection layer to transform enhanced embeddings back to original dimension
        self.proj = nn.Linear(ec, ct)
        
        # Scaling factor for attention weights
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        
        # Convolutional projections for image features
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        
        # Adaptive max pooling layers for image feature pooling
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        
        # Store other parameters
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]  # Batch size
        assert len(x) == self.nf  # Ensure correct number of image features
        num_patches = self.k**2  # Number of patches in each image feature
        
        # Process each image feature with projection and pooling
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)  # Concatenate and transpose for attention computation
        
        # Transform text input using query, key, and value networks
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # Reshape query, key, and value tensors for batched matrix multiplication
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        # Compute attention weights using matrix multiplication and softmax
        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        # Compute attended output using weighted sum of values
        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))  # Project back to original embedding dimension
        return x * self.scale + text  # Scale and add residual connection with text embeddings


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # Initialize bias for contrastive head
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # Initialize logit scale for contrastive head
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)  # Normalize input embeddings
        w = F.normalize(w, dim=-1, p=2)  # Normalize text embeddings
        x = torch.einsum("bchw,bkc->bkhw", x, w)  # Compute contrastive scores
        return x * self.logit_scale.exp() + self.bias  # Scale and add bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """
    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # 初始化一个二维批归一化层，用于处理输入特征图的通道维度
        self.norm = nn.BatchNorm2d(embed_dims)
        
        # NOTE: 使用 -10.0 来保持初始的类别损失与其他损失一致
        # 初始化一个偏置参数，用于调整模型预测中的偏差
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        
        # use -1.0 is more stable
        # 初始化一个对数尺度参数，用于缩放模型输出的对数概率
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        # 对输入特征图进行批归一化处理
        x = self.norm(x)
        
        # 对输入的权重向量进行 L2 归一化处理
        w = F.normalize(w, dim=-1, p=2)
        
        # 使用 Einstein Summation (einsum) 完成张量乘法，计算特征图与权重的点积
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        
        # 返回加权后的特征图，通过指数函数和偏置进行缩放和平移
        return x * self.logit_scale.exp() + self.bias
class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio."""
        super().__init__(c1, c2, shortcut, g, k, e)
        # Calculate the number of hidden channels
        c_ = int(c2 * e)  # hidden channels
        # Initialize RepConv module with input channels c1, output channels c_, and kernel size k[0]
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        # Calculate the number of hidden channels
        c_ = int(c2 * e)  # hidden channels
        # Create a sequence of RepBottleneck modules with input channels c_, output channels c_, and other parameters
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        # Calculate the number of channels for the first convolution layer
        self.c = c3 // 2
        # Initialize the first convolution layer with input channels c1 and output channels c3
        self.cv1 = Conv(c1, c3, 1, 1)
        # Create a sequence of RepCSP module followed by a Conv module
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        # Create another sequence of RepCSP module followed by a Conv module
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        # Initialize the last convolution layer with input channels c3 + 2 * c4 and output channels c2
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        # Apply the first convolution layer and split its output into two parts
        y = list(self.cv1(x).chunk(2, 1))
        # Apply self.cv2 and self.cv3 sequentially on the last split part of y
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        # Concatenate all parts of y and apply the last convolution layer
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        # Apply the first convolution layer and split its output into two parts
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # Apply self.cv2 and self.cv3 sequentially on the last split part of y
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        # Concatenate all parts of y and apply the last convolution layer
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        # Calculate the number of channels for the first convolution layer
        self.c = c3 // 2
        # Initialize the first convolution layer with input channels c1 and output channels c3
        self.cv1 = Conv(c1, c3, 1, 1)
        # Initialize the second convolution layer with input channels c3 // 2 and output channels c4
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        # Initialize the third convolution layer with input and output channels both as c4
        self.cv3 = Conv(c4, c4, 3, 1)
        # Initialize the last convolution layer with input channels c3 + 2 * c4 and output channels c2
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        # Initialize convolution layer with input channels c1, output channels c2, kernel size 3x3, stride 2, padding 1
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        # Apply average pooling to input x with kernel size 2x2 and stride 2
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        # Apply the initialized convolution layer
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        # Calculate the number of channels for the first convolution layer
        self.c = c2 // 2
        # Initialize the first convolution layer to downsample input from c1 // 2 channels to self.c channels
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        # Initialize the second convolution layer with input channels c1 // 2, output channels self.c, kernel size 1x1, stride 1, padding 0
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
    def forward(self, x):
        """Forward pass through ADown layer."""
        # 对输入 x 进行 2x2 平均池化操作，步长为1，填充为0，保持输入大小
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        # 将经过平均池化后的 x 在第二个维度上分割成两部分 x1 和 x2
        x1, x2 = x.chunk(2, 1)
        # 对 x1 应用 self.cv1 网络层
        x1 = self.cv1(x1)
        # 对 x2 进行 3x3 最大池化操作，步长为2，填充为1
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        # 对池化后的 x2 应用 self.cv2 网络层
        x2 = self.cv2(x2)
        # 将经过处理的 x1 和 x2 沿着第二个维度拼接起来
        return torch.cat((x1, x2), 1)
class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)  # 使用 Conv 函数创建一个 1x1 的卷积层
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 使用 MaxPool2d 创建一个最大池化层
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 使用 MaxPool2d 创建一个最大池化层
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 使用 MaxPool2d 创建一个最大池化层
        self.cv5 = Conv(4 * c3, c2, 1, 1)  # 使用 Conv 函数创建一个 1x1 的卷积层

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]  # 对输入 x 进行第一个卷积操作
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])  # 对 y[-1] 分别应用 cv2, cv3, cv4 的池化操作，并将结果添加到 y 中
        return self.cv5(torch.cat(y, 1))  # 将 y 中的结果在通道维度上拼接，然后应用 cv5 的卷积操作


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)  # 使用 Conv2d 创建一个卷积层

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)  # 对输入 x 应用卷积操作后，按照通道维度将结果分割为 c2s 指定的通道数


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx  # 保存传入的 idx 参数作为属性

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]  # 获取 xs 中最后一个张量的空间维度
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]  # 对 xs[:-1] 中的张量进行插值操作，并根据 idx 选择相应的通道
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)  # 将插值后的结果与 xs 的最后一个张量相加，并在维度 0 上求和


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)  # 使用 Conv 函数创建一个 7x7 的卷积层，g=ed 表示组卷积
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)  # 使用 Conv 函数创建一个 3x3 的卷积层，g=ed 表示组卷积
        self.dim = ed  # 保存输入的 ed 参数作为属性
        self.act = nn.SiLU()  # 创建 SiLU 激活函数对象

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))  # 对输入 x 分别应用 conv 和 conv1 操作，然后将结果相加并使用 SiLU 激活函数

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))  # 对输入 x 只应用 conv 操作，并使用 SiLU 激活函数

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        # 融合第一个卷积层和其对应的批标准化层
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        # 融合第二个卷积层和其对应的批标准化层
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        # 获取融合后的第一个卷积层的权重和偏置
        conv_w = conv.weight
        conv_b = conv.bias
        # 获取融合后的第二个卷积层的权重和偏置
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        # 对第二个卷积层的权重进行填充，使用2个单位宽度的零填充
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        # 计算最终融合后的卷积层的权重和偏置
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        # 将融合后的权重和偏置复制到第一个卷积层对象中
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        # 更新 self.conv 为融合后的第一个卷积层对象，删除原始的第二个卷积层对象 self.conv1
        self.conv = conv
        del self.conv1
class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),  # 3x3 convolution with c1 channels
            Conv(c1, 2 * c_, 1),    # 1x1 convolution increasing channels to 2*c_
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),  # RepVGGDW or 3x3 convolution based on lk
            Conv(2 * c_, c2, 1),    # 1x1 convolution reducing channels to c2
            Conv(c2, c2, 3, g=c2),  # 3x3 convolution with c2 channels
        )

        self.add = shortcut and c1 == c2  # Determine whether to add shortcut connection

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.
    """
    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 计算每个注意力头的维度
        self.head_dim = dim // num_heads
        # 根据比例计算注意力键的维度
        self.key_dim = int(self.head_dim * attn_ratio)
        # 缩放因子，用于注意力分数的缩放
        self.scale = self.key_dim**-0.5
        # 计算 qkv 的输入通道数
        nh_kd = self.key_dim * num_heads
        # 计算 qkv 模块的输出通道数
        h = dim + nh_kd * 2
        # 创建计算查询、键、值的卷积层 qkv
        self.qkv = Conv(dim, h, 1, act=False)
        # 创建投影注意力值的卷积层 proj
        self.proj = Conv(dim, dim, 1, act=False)
        # 创建用于位置编码的卷积层 pe
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        # 获取输入张量的形状信息
        B, C, H, W = x.shape
        # 计算位置数量 N
        N = H * W
        # 计算查询、键、值的结果
        qkv = self.qkv(x)
        # 按照注意力头、注意力键维度、注意力值维度拆分 qkv 张量
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        # 计算注意力分数
        attn = (q.transpose(-2, -1) @ k) * self.scale
        # 对注意力分数进行 softmax 归一化
        attn = attn.softmax(dim=-1)
        # 计算自注意力结果并加上位置编码
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        # 对注意力结果进行投影
        x = self.proj(x)
        return x
class PSA(nn.Module):
    """
    Position-wise Spatial Attention module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention module, and feed-forward network with channel reduction."""
        super().__init__()
        assert c1 == c2
        # 计算中间通道数
        self.c = int(c1 * e)
        # 第一个 1x1 卷积层，将输入通道数减少到 2*c
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # 第二个 1x1 卷积层，将输出通道数减少到 c
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 空间注意力模块
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        # 前馈网络模块
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Forward pass of the PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        # 对输入进行第一次卷积，然后按通道数分割为 a 和 b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # 应用空间注意力模块到 b
        b = b + self.attn(b)
        # 应用前馈网络到 b
        b = b + self.ffn(b)
        # 将 a 和更新后的 b 拼接后通过第二个卷积层 cv2
        return self.cv2(torch.cat((a, b), 1))


class SCDown(nn.Module):
    """Spatial Channel Downsample (SCDown) module for reducing spatial and channel dimensions."""

    def __init__(self, c1, c2, k, s):
        """
        Spatial Channel Downsample (SCDown) module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolutional layer.
            s (int): Stride for the convolutional layer.
        """
        super().__init__()
        # 第一个 1x1 卷积层，用于降低输入通道数到 c2
        self.cv1 = Conv(c1, c2, 1, 1)
        # 第二个卷积层，用于进一步降维，包括空间和通道
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Forward pass of the SCDown module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the SCDown module.
        """
        # 应用两个卷积层来实现空间和通道的降维
        return self.cv2(self.cv1(x))
```