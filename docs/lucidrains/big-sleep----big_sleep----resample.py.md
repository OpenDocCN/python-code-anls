# `.\lucidrains\big-sleep\big_sleep\resample.py`

```
"""Good differentiable image resampling for PyTorch."""

# 导入所需的库
from functools import update_wrapper
import math

import torch
from torch.nn import functional as F


# 定义 sinc 函数
def sinc(x):
    return torch.where(x != 0,  torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


# 定义 lanczos 函数
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


# 定义 ramp 函数
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# 定义 odd 函数
def odd(fn):
    return update_wrapper(lambda x: torch.sign(x) * fn(abs(x)), fn)


# 定义将输入转换为线性 sRGB 的函数
def _to_linear_srgb(input):
    cond = input <= 0.04045
    a = input / 12.92
    b = ((input + 0.055) / 1.055)**2.4
    return torch.where(cond, a, b)


# 定义将输入转换为非线性 sRGB 的函数
def _to_nonlinear_srgb(input):
    cond = input <= 0.0031308
    a = 12.92 * input
    b = 1.055 * input**(1/2.4) - 0.055
    return torch.where(cond, a, b)


# 使用 odd 函数包装 _to_linear_srgb 函数和 _to_nonlinear_srgb 函数
to_linear_srgb = odd(_to_linear_srgb)
to_nonlinear_srgb = odd(_to_nonlinear_srgb)


# 定义 resample 函数
def resample(input, size, align_corners=True, is_srgb=False):
    n, c, h, w = input.shape
    dh, dw = size

    # 如果 is_srgb 为 True，则将输入转换为线性 sRGB
    if is_srgb:
        input = to_linear_srgb(input)

    input = input.view([n * c, 1, h, w])

    # 如果目标高度小于原始高度
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 3), 3).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    # 如果目标宽度小于原始宽度
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 3), 3).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    input = F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

    # 如果 is_srgb 为 True，则将输出转换为非线性 sRGB
    if is_srgb:
        input = to_nonlinear_srgb(input)

    return input
```