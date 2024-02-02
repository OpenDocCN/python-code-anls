# `so-vits-svc\vdecoder\hifiganwithsnake\alias\filter.py`

```py
# 导入所需的库
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查是否存在 torch.sinc 函数，如果不存在则定义 sinc 函数
if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # 从 adefossez 的 julius.core.sinc 中采用代码，根据 MIT 许可证
    # https://adefossez.github.io/julius/julius/core.html
    #   许可证在 incl_licenses 目录中
    def sinc(x: torch.Tensor):
        """
        实现 sinc 函数，即 sin(pi * x) / (pi * x)
        __警告__: 与 julius.sinc 不同，输入被乘以了 `pi`！
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)

# 从 adefossez 的 julius.lowpass.LowPassFilters 中采用代码，根据 MIT 许可证
# https://adefossez.github.io/julius/julius/lowpass.html
#   许可证在 incl_licenses 目录中
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # 返回滤波器 [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    # 对于 kaiser 窗口
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        # 计算滤波器
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # 将滤波器归一化，使其总和为1，否则输入信号中会有一个小的泄漏常量分量
        filter_ /= filter_.sum()
        # 将滤波器转换成1x1xkernel_size的视图
        filter = filter_.view(1, 1, kernel_size)

    # 返回滤波器
    return filter
class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12,
                 C=None):
        # 初始化函数，定义低通滤波器的参数
        # kernel_size 应该是偶数，但在这个实现中，奇数也是可能的
        super().__init__()
        # 如果截止频率小于0，则引发数值错误
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        # 如果截止频率大于0.5，则引发数值错误
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        # 设置滤波器的参数
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        # 创建 kaiser_sinc_filter1d 滤波器
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        # 将滤波器注册为模型的缓冲区
        self.register_buffer("filter", filter)
        self.conv1d_block = None
        # 如果 C 不为空，则创建 1D 卷积块
        if C is not None:
            self.conv1d_block = [nn.Conv1d(C,C,kernel_size,stride=self.stride, groups=C, bias=False),]
            self.conv1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1))
            self.conv1d_block[0].requires_grad_(False)

    # 输入 [B, C, T]
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 如果第一个 1D 卷积块的权重所在设备与输入 x 的设备不同
        if self.conv1d_block[0].weight.device != x.device:
            # 将第一个 1D 卷积块的权重移动到输入 x 所在的设备
            self.conv1d_block[0] = self.conv1d_block[0].to(x.device)
        # 如果没有定义 1D 卷积块
        if self.conv1d_block is None:
            # 获取输入 x 的形状信息
            _, C, _ = x.shape
            # 如果需要填充
            if self.padding:
                # 对输入 x 进行填充
                x = F.pad(x, (self.pad_left, self.pad_right),
                            mode=self.padding_mode)
            # 执行 1D 卷积操作
            out = F.conv1d(x, self.filter.expand(C, -1, -1),
                            stride=self.stride, groups=C)
        else:
            # 如果需要填充
            if self.padding:
                # 对输入 x 进行填充
                x = F.pad(x, (self.pad_left, self.pad_right),
                            mode=self.padding_mode)
            # 执行定义好的 1D 卷积块操作
            out = self.conv1d_block[0](x)
        # 返回输出结果
        return out
```