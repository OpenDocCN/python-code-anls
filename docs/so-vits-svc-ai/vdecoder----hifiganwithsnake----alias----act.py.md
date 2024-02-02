# `so-vits-svc\vdecoder\hifiganwithsnake\alias\act.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的神经网络模块
import torch.nn as nn
# 导入 torch 中的函数模块
import torch.nn.functional as F
# 从 torch 中导入 pow 和 sin 函数
from torch import pow, sin
# 从 torch.nn 中导入 Parameter 类
from torch.nn import Parameter
# 从当前目录下的 resample.py 文件中导入 DownSample1d 和 UpSample1d 类
from .resample import DownSample1d, UpSample1d

# 定义 Activation1d 类，继承自 nn.Module 类
class Activation1d(nn.Module):
    # 初始化函数
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        # 设置上采样比例
        self.up_ratio = up_ratio
        # 设置下采样比例
        self.down_ratio = down_ratio
        # 设置激活函数
        self.act = activation
        # 创建上采样对象
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        # 创建下采样对象
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # 前向传播函数
    # x: [B,C,T]
    def forward(self, x):
        # 上采样
        x = self.upsample(x)
        # 激活函数
        x = self.act(x)
        # 下采样
        x = self.downsample(x)

        return x

# 定义 SnakeBeta 类，继承自 nn.Module 类
class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    # 初始化 SnakeBeta 类
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        初始化。
        输入：
            - in_features: 输入的形状
            - alpha：控制频率的可训练参数
            - beta：控制幅度的可训练参数
            alpha 默认初始化为 1，值越高表示频率越高。
            beta 默认初始化为 1，值越高表示幅度越高。
            alpha 将会和模型的其余部分一起训练。
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        # 初始化 alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # 对数尺度的 alpha 初始化为零
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # 线性尺度的 alpha 初始化为一
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    # SnakeBeta 函数的前向传播
    def forward(self, x):
        '''
        函数的前向传播。
        对输入进行逐元素操作。
        SnakeBeta = x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(
            0).unsqueeze(-1)  # 与 x 对齐到 [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x
# 定义一个名为Mish的神经网络模块，实现Mish激活函数，参考论文"Mish: A Self Regularized Non-Monotonic Neural Activation Function"，链接https://arxiv.org/abs/1908.08681
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    # 前向传播函数，对输入x应用Mish激活函数
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# 定义一个名为SnakeAlias的神经网络模块
class SnakeAlias(nn.Module):
    # 初始化函数，接受通道数、上采样比例、下采样比例、上采样核大小、下采样核大小和C作为参数
    def __init__(self,
                 channels,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12,
                 C = None):
        super().__init__()
        # 初始化上采样比例、下采样比例、激活函数和上采样、下采样模块
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = SnakeBeta(channels, alpha_logscale=True)
        self.upsample = UpSample1d(up_ratio, up_kernel_size, C)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, C)

    # 前向传播函数，接受输入x和C作为参数
    # x: [B,C,T]
    def forward(self, x, C=None):
        # 对输入x进行上采样
        x = self.upsample(x, C)
        # 对上采样结果应用激活函数
        x = self.act(x)
        # 对激活函数结果进行下采样
        x = self.downsample(x)

        return x
```