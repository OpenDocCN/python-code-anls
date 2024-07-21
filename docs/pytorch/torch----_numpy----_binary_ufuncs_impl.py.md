# `.\pytorch\torch\_numpy\_binary_ufuncs_impl.py`

```py
# mypy: ignore-errors
# 忽略类型检查错误的标记

"""Export torch work functions for binary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch._numpy/_ufuncs.py` module.
"""
# 导出 torch 的二元通用函数工作函数，重命名/调整以匹配 numpy。
# 这些函数进一步导出到 `torch._numpy/_ufuncs.py` 模块的公共符号中。

import torch

from torch import (  # noqa: F401
    add,  # noqa: F401
    arctan2,  # noqa: F401
    bitwise_and,  # noqa: F401
    bitwise_left_shift as left_shift,  # noqa: F401
    bitwise_or,  # noqa: F401
    bitwise_right_shift as right_shift,  # noqa: F401
    bitwise_xor,  # noqa: F401
    copysign,  # noqa: F401
    divide,  # noqa: F401
    eq as equal,  # noqa: F401
    float_power,  # noqa: F401
    floor_divide,  # noqa: F401
    fmax,  # noqa: F401
    fmin,  # noqa: F401
    fmod,  # noqa: F401
    gcd,  # noqa: F401
    greater,  # noqa: F401
    greater_equal,  # noqa: F401
    heaviside,  # noqa: F401
    hypot,  # noqa: F401
    lcm,  # noqa: F401
    ldexp,  # noqa: F401
    less,  # noqa: F401
    less_equal,  # noqa: F401
    logaddexp,  # noqa: F401
    logaddexp2,  # noqa: F401
    logical_and,  # noqa: F401
    logical_or,  # noqa: F401
    logical_xor,  # noqa: F401
    maximum,  # noqa: F401
    minimum,  # noqa: F401
    multiply,  # noqa: F401
    nextafter,  # noqa: F401
    not_equal,  # noqa: F401
    pow as power,  # noqa: F401
    remainder,  # noqa: F401
    remainder as mod,  # noqa: F401
    subtract,  # noqa: F401
    true_divide,  # noqa: F401
)

from . import _dtypes_impl, _util

# 导入本地模块 _dtypes_impl 和 _util

# work around torch limitations w.r.t. numpy
def matmul(x, y):
    # 解决 Torch 对于 numpy 的限制
    # - RuntimeError: expected scalar type Int but found Double
    # - RuntimeError: "addmm_impl_cpu_" not implemented for 'Bool'
    # - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    
    # 确定输出的数据类型
    dtype = _dtypes_impl.result_type_impl(x, y)
    
    # 检查是否是布尔类型
    is_bool = dtype == torch.bool
    
    # 检查是否是半精度类型
    is_half = (x.dtype == torch.float16 or y.dtype == torch.float16) and (
        x.is_cpu or y.is_cpu
    )

    # 根据情况选择工作的数据类型
    work_dtype = dtype
    if is_bool:
        work_dtype = torch.uint8
    if is_half:
        work_dtype = torch.float32

    # 根据需要转换输入张量的数据类型
    x = _util.cast_if_needed(x, work_dtype)
    y = _util.cast_if_needed(y, work_dtype)

    # 使用 Torch 的 matmul 函数计算结果
    result = torch.matmul(x, y)

    # 如果输出数据类型与所需数据类型不一致，则进行转换
    if work_dtype != dtype:
        result = result.to(dtype)

    return result


# a stub implementation of divmod, should be improved after
# https://github.com/pytorch/pytorch/issues/90820 is fixed in pytorch
# divmod 的存根实现，应在修复 PyTorch 中的问题后进行改进

def divmod(x, y):
    return x // y, x % y
```