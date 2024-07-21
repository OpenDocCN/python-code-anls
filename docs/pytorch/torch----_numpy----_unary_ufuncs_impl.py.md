# `.\pytorch\torch\_numpy\_unary_ufuncs_impl.py`

```py
# 忽略类型检查错误，这是为了告知静态类型检查工具不要报告这些代码中的错误
"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `_numpy/_ufuncs.py` module.
"""
# 导入 torch 库
import torch

# 从 torch 中导入一系列函数，并进行重命名以匹配 numpy 的命名风格。注释 `noqa: F401` 用于告知 linter 忽略未使用的导入警告。

from torch import (  # noqa: F401
    absolute as fabs,  # noqa: F401
    arccos,  # noqa: F401
    arccosh,  # noqa: F401
    arcsin,  # noqa: F401
    arcsinh,  # noqa: F401
    arctan,  # noqa: F401
    arctanh,  # noqa: F401
    bitwise_not,  # noqa: F401
    bitwise_not as invert,  # noqa: F401
    ceil,  # noqa: F401
    conj_physical as conjugate,  # noqa: F401
    cos,  # noqa: F401
    cosh,  # noqa: F401
    deg2rad,  # noqa: F401
    deg2rad as radians,  # noqa: F401
    exp,  # noqa: F401
    exp2,  # noqa: F401
    expm1,  # noqa: F401
    floor,  # noqa: F401
    isfinite,  # noqa: F401
    isinf,  # noqa: F401
    isnan,  # noqa: F401
    log,  # noqa: F401
    log10,  # noqa: F401
    log1p,  # noqa: F401
    log2,  # noqa: F401
    logical_not,  # noqa: F401
    negative,  # noqa: F401
    rad2deg,  # noqa: F401
    rad2deg as degrees,  # noqa: F401
    reciprocal,  # noqa: F401
    round as fix,  # noqa: F401
    round as rint,  # noqa: F401
    sign,  # noqa: F401
    signbit,  # noqa: F401
    sin,  # noqa: F401
    sinh,  # noqa: F401
    sqrt,  # noqa: F401
    square,  # noqa: F401
    tan,  # noqa: F401
    tanh,  # noqa: F401
    trunc,  # noqa: F401
)

# 特殊情况：torch 并未导出以下这些函数名
# 计算 x 的立方根
def cbrt(x):
    return torch.pow(x, 1 / 3)

# 返回 x 的正值
def positive(x):
    return +x

# 计算 x 的绝对值
def absolute(x):
    # 由于 torch.absolute 对布尔类型不适用，这里进行了一个变通的实现
    if x.dtype == torch.bool:
        return x
    return torch.absolute(x)

# TODO 设置 __name__ 和 __qualname__
# 将 absolute 和 conjugate 函数分别赋值给 abs 和 conj
abs = absolute
conj = conjugate
```