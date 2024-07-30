# `.\yolov8\ultralytics\nn\modules\conv.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    # 计算实际的卷积核大小，当 dilation 大于 1 时
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    # 自动计算 padding 大小，如果未指定
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # 默认激活函数为 SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 创建卷积层，设置相关参数
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 批归一化层
        self.bn = nn.BatchNorm2d(c2)
        # 激活函数，默认为 SiLU
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        # 添加额外的 1x1 卷积层
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # 使用融合后的卷积核进行前向传播
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        # 合并并更新卷积核权重
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        # 删除 cv2 属性，更新 forward 方法为融合后的版本
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        # 调用父类构造函数，初始化神经网络层
        super().__init__()
        # 创建第一个卷积层，1x1卷积，不使用激活函数
        self.conv1 = Conv(c1, c2, 1, act=False)
        # 创建深度可分离卷积层，输入输出通道数相同，卷积核大小为k，使用指定的激活函数
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # 将输入张量x先经过conv1进行卷积，再经过conv2进行深度可分离卷积
        return self.conv2(self.conv1(x))
class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """Repeated Convolution."""

    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        """Initialize repeated convolution with specified parameters."""
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s, g=g, act=act)
        self.conv2 = Conv(c2, c2, k, s, g=g, act=act)

    def forward(self, x):
        """Apply repeated convolution on input tensor and return the result."""
        return self.conv2(self.conv1(x))
    """
    RepConv 是一个基本的重复风格块，包括训练和部署状态。
    
    这个模块用于 RT-DETR。
    基于 https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    
    default_act = nn.SiLU()  # 默认激活函数为 SiLU
    
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """使用给定的输入、输出和可选的激活函数初始化轻量卷积层。"""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
        # 如果启用了批归一化（bn=True）且满足条件，初始化批归一化层
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        # 初始化第一个卷积层，使用自定义的 Conv 类
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        # 初始化第二个卷积层，使用自定义的 Conv 类
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
    
    def forward_fuse(self, x):
        """前向传播过程。"""
        return self.act(self.conv(x))
    
    def forward(self, x):
        """前向传播过程。"""
        # 如果未使用批归一化，id_out 为 0；否则，id_out 为经过批归一化的 x
        id_out = 0 if self.bn is None else self.bn(x)
        # 返回第一个卷积层、第二个卷积层和可能的批归一化的叠加结果
        return self.act(self.conv1(x) + self.conv2(x) + id_out)
    
    def get_equivalent_kernel_bias(self):
        """通过将 3x3 卷积核、1x1 卷积核和身份卷积核及其偏置相加，返回等效的卷积核和偏置。"""
        # 获取第一个卷积层的等效卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        # 获取第二个卷积层的等效卷积核和偏置
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        # 获取批归一化层的等效卷积核和偏置
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        # 返回相加后的等效卷积核和偏置
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """将 1x1 卷积核填充为 3x3 卷积核。"""
        # 如果 1x1 卷积核为 None，则返回 0
        if kernel1x1 is None:
            return 0
        else:
            # 使用 torch.nn.functional.pad 函数对 1x1 卷积核进行填充
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        # 如果分支为空，返回0和0
        if branch is None:
            return 0, 0
        # 如果分支是Conv类型
        if isinstance(branch, Conv):
            # 获取卷积核
            kernel = branch.conv.weight
            # 获取BatchNorm层的running_mean
            running_mean = branch.bn.running_mean
            # 获取BatchNorm层的running_var
            running_var = branch.bn.running_var
            # 获取BatchNorm层的gamma（权重）
            gamma = branch.bn.weight
            # 获取BatchNorm层的beta（偏置）
            beta = branch.bn.bias
            # 获取BatchNorm层的eps
            eps = branch.bn.eps
        # 如果分支是nn.BatchNorm2d类型
        elif isinstance(branch, nn.BatchNorm2d):
            # 如果没有id_tensor属性，创建一个对角矩阵作为id_tensor
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            # 使用已经存在的id_tensor作为kernel
            kernel = self.id_tensor
            # 获取BatchNorm2d层的running_mean
            running_mean = branch.running_mean
            # 获取BatchNorm2d层的running_var
            running_var = branch.running_var
            # 获取BatchNorm2d层的gamma（权重）
            gamma = branch.weight
            # 获取BatchNorm2d层的beta（偏置）
            beta = branch.bias
            # 获取BatchNorm2d层的eps
            eps = branch.eps
        # 计算标准差
        std = (running_var + eps).sqrt()
        # 计算t的值，用于归一化
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 返回融合了BN的卷积核和偏置
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        # 如果已经存在conv属性，直接返回
        if hasattr(self, "conv"):
            return
        # 获取等效的卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
        # 创建新的卷积层，并设置其参数
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        # 将融合后的卷积核和偏置赋值给新的卷积层
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # 将所有参数设置为不需要梯度
        for para in self.parameters():
            para.detach_()
        # 删除不再需要的属性
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        # 创建一个自适应平均池化层，输出大小为 (1, 1)，用于对输入进行全局平均池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 创建一个卷积层，对输入进行通道间的卷积，输出通道数与输入相同
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # 创建一个 Sigmoid 激活函数实例
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        # 对输入 x 进行全局平均池化，然后通过卷积和 Sigmoid 激活函数处理，返回加权后的特征
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        # 断言 kernel_size 必须是 3 或 7
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        # 根据 kernel_size 创建卷积层，用于空间注意力计算
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # 创建一个 Sigmoid 激活函数实例
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        # 对输入 x 进行通道和空间注意力的计算，返回加权后的特征
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        # 创建通道注意力模块和空间注意力模块
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        # 通过通道注意力模块和空间注意力模块，对输入 x 进行特征加权处理
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        # 指定要进行拼接的维度
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        # 沿指定维度对输入 x 中的张量进行拼接
        return torch.cat(x, self.d)
```