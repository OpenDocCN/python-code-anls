# `.\pytorch\torch\_numpy\fft.py`

```py
# mypy: ignore-errors
# 忽略类型检查错误

from __future__ import annotations
# 导入将来版本的类型注解特性，用于函数参数注解

import functools
# 导入 functools 模块，用于创建装饰器函数

import torch
# 导入 PyTorch 库

from . import _dtypes_impl, _util
# 从当前包中导入 _dtypes_impl 和 _util 模块

from ._normalizations import ArrayLike, normalizer
# 从当前包的 _normalizations 模块中导入 ArrayLike 类和 normalizer 装饰器函数

def upcast(func):
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""
    # upcast 装饰器函数：将输入数据类型转换为 64 位并返回 64 位结果

    @functools.wraps(func)
    # 使用 functools.wraps 保留原始函数的元数据
    def wrapped(tensor, *args, **kwds):
        # 包装函数，接受 tensor 和其他参数

        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        # 根据 tensor 是否为复数确定目标数据类型为复数或浮点数

        tensor = _util.cast_if_needed(tensor, target_dtype)
        # 根据需要将 tensor 转换为目标数据类型

        return func(tensor, *args, **kwds)
        # 调用原始函数并返回结果

    return wrapped
    # 返回包装后的函数

@normalizer
@upcast
# 使用 normalizer 和 upcast 装饰器修饰以下函数，先执行 normalizer，再执行 upcast

def fft(a: ArrayLike, n=None, axis=-1, norm=None):
    # FFT 变换函数，对数组 a 进行傅里叶变换

    return torch.fft.fft(a, n, dim=axis, norm=norm)
    # 调用 PyTorch 的 FFT 函数进行计算，并返回结果

# 后续函数 ifft, rfft, irfft 等均类似，只是调用了不同的 PyTorch FFT 变换函数
# 函数名和参数注释可以类似地进行解释
```