# `.\lucidrains\logavgexp-torch\logavgexp_pytorch\logavgexp_pytorch.py`

```
import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from unfoldNd import unfoldNd

# helper functions

# 检查变量是否存在
def exists(t):
    return t is not None

# 对张量取对数
def log(t, eps = 1e-20):
    return torch.log(t + eps)

# 将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 计算卷积输出形状
def calc_conv_output(shape, kernel_size, padding, stride):
    return tuple(map(lambda x: int((x[0] - x[1] + 2 * x[2]) / x[3] + 1), zip(shape, kernel_size, padding, stride))

# main function

# 对输入张量进行 logavgexp 操作
def logavgexp(
    t,
    mask = None,
    dim = -1,
    eps = 1e-20,
    temp = 0.01,
    keepdim = False
):
    if exists(mask):
        mask_value = -torch.finfo(t.dtype).max
        t = t.masked_fill(~mask, mask_value)
        n = mask.sum(dim = dim)
        norm = torch.log(n)
    else:
        n = t.shape[dim]
        norm = math.log(n)

    t = t / temp
    max_t = t.amax(dim = dim).detach()
    t_exp = (t - max_t.unsqueeze(dim)).exp()
    avg_exp = t_exp.sum(dim = dim).clamp(min = eps) / n
    out = log(avg_exp, eps = eps) + max_t - norm
    out = out * temp

    out = out.unsqueeze(dim) if keepdim else out
    return out

# learned temperature - logavgexp class

# LogAvgExp 类，用于 logavgexp 操作
class LogAvgExp(nn.Module):
    def __init__(
        self,
        dim = -1,
        eps = 1e-20,
        temp = 0.01,
        keepdim = False,
        learned_temp = False
    ):
        super().__init__()
        assert temp >= 0 and temp <= 1., 'temperature must be between 0 and 1'

        self.learned_temp = learned_temp

        if learned_temp:
            self.temp = nn.Parameter(torch.ones((1,)) * math.log(temp))
        else:
            self.temp = temp

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, mask = None, eps = 1e-8):
        if not self.learned_temp:
            temp = self.temp
        else:
            temp = self.temp.exp().clamp(min = eps)

        return logavgexp(
            x,
            mask = mask,
            dim = self.dim,
            temp = temp,
            keepdim = self.keepdim
        )

# logavgexp 2d

# LogAvgExp2D 类，用于 2D logavgexp 操作
class LogAvgExp2D(nn.Module):
    def __init__(
        self,
        kernel_size,
        *,
        padding = 0,
        stride = 1,
        temp = 0.01,
        learned_temp = True,
        eps = 1e-20,
        **kwargs
    ):
        super().__init__()
        self.padding = cast_tuple(padding, 2)
        self.stride = cast_tuple(stride, 2)
        self.kernel_size = cast_tuple(kernel_size, 2)

        self.unfold = nn.Unfold(self.kernel_size, padding = self.padding, stride = self.stride)
        self.logavgexp = LogAvgExp(dim = -1, eps = eps, learned_temp = learned_temp, temp = temp)

    def forward(self, x):
        """
        b - batch
        c - channels
        h - height
        w - width
        j - reducing dimension
        """

        b, c, h, w = x.shape
        out_h, out_w = calc_conv_output((h, w), self.kernel_size, self.padding, self.stride)

        # calculate mask for padding, if needed

        mask = None
        if any([i > 0 for i in self.padding]):
            mask = torch.ones((b, 1, h, w), device = x.device)
            mask = self.unfold(mask)
            mask = rearrange(mask, 'b j (h w) -> b 1 h w j', h = out_h, w = out_w)
            mask = mask == 1.

        x = self.unfold(x)
        x = rearrange(x, 'b (c j) (h w) -> b c h w j', h = out_h, w = out_w, c = c)
        return self.logavgexp(x, mask = mask)

# logavgexp 3d

# LogAvgExp3D 类，用于 3D logavgexp 操作
class LogAvgExp3D(nn.Module):
    def __init__(
        self,
        kernel_size,
        *,
        padding = 0,
        stride = 1,
        temp = 0.01,
        learned_temp = True,
        eps = 1e-20,
        **kwargs
    # 初始化函数，设置填充、步幅和卷积核大小
    def __init__(
        super().__init__()
        # 将填充、步幅和卷积核大小转换为元组形式
        self.padding = cast_tuple(padding, 3)
        self.stride = cast_tuple(stride, 3)
        self.kernel_size = cast_tuple(kernel_size, 3)

        # 部分应用 unfoldNd 函数，设置卷积核大小、填充和步幅
        self.unfold = partial(unfoldNd, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        # 初始化 LogAvgExp 函数
        self.logavgexp = LogAvgExp(dim = -1, eps = eps, learned_temp = learned_temp, temp = temp)

    # 前向传播函数
    def forward(self, x):
        """
        b - batch
        c - channels
        f - depth
        h - height
        w - width
        j - reducing dimension
        """

        # 获取输入张量的形状
        b, c, f, h, w = x.shape
        # 计算卷积输出的深度、高度和宽度
        out_f, out_h, out_w = calc_conv_output((f, h, w), self.kernel_size, self.padding, self.stride)

        # 计算是否需要填充的掩码

        mask = None
        if any([i > 0 for i in self.padding]):
            mask = torch.ones((b, 1, f, h, w), device = x.device)
            mask = self.unfold(mask)
            mask = rearrange(mask, 'b j (f h w) -> b 1 f h w j', f = out_f, h = out_h, w = out_w)
            mask = mask == 1.

        # 对输入张量进行展开操作
        x = self.unfold(x)
        x = rearrange(x, 'b (c j) (f h w) -> b c f h w j', f = out_f, h = out_h, w = out_w, c = c)
        # 调用 logavgexp 函数进行计算，传入掩码
        return self.logavgexp(x, mask = mask)
```