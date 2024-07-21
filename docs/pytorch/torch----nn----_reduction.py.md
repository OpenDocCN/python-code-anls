# `.\pytorch\torch\nn\_reduction.py`

```py
# 导入警告模块，用于在需要时发出警告
import warnings
# 导入可选类型，用于指定参数的可选性
from typing import Optional

# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h

# 根据指定的字符串 reduction 返回对应的整数值
def get_enum(reduction: str) -> int:
    # 如果 reduction 是 "none"，返回 0
    if reduction == "none":
        ret = 0
    # 如果 reduction 是 "mean"，返回 1
    elif reduction == "mean":
        ret = 1
    # 如果 reduction 是 "elementwise_mean"
    elif reduction == "elementwise_mean":
        # 发出警告，提示该选项已经不推荐使用
        warnings.warn(
            "reduction='elementwise_mean' is deprecated. "
            "Please use reduction='mean' instead."
        )
        # 返回 1，对应于 "mean"
        ret = 1
    # 如果 reduction 是 "sum"，返回 2
    elif reduction == "sum":
        ret = 2
    else:
        # 如果 reduction 是未知值，则返回 -1，并抛出 ValueError 异常
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret

# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now

# We use these functions in torch/legacy as well, in which case we'll silence the warning
# 在支持之前版本的同时，接受布尔类型的 size_average 和 reduce 参数，并暂时转换为新的常量

# 定义函数 legacy_get_string，用于获取遗留接口的字符串表示
def legacy_get_string(
    size_average: Optional[bool],
    reduce: Optional[bool],
    emit_warning: bool = True,
) -> str:
    # 警告信息模板
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    # 如果 size_average 为 None，默认为 True
    if size_average is None:
        size_average = True
    # 如果 reduce 为 None，默认为 True
    if reduce is None:
        reduce = True

    # 根据 size_average 和 reduce 的值确定返回的字符串表示
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    
    # 如果 emit_warning 为 True，发出警告，提示用户将要废弃的参数转换为新的 reduction 参数
    if emit_warning:
        warnings.warn(warning.format(ret))
    
    return ret

# 定义函数 legacy_get_enum，用于获取遗留接口的枚举表示
def legacy_get_enum(
    size_average: Optional[bool],
    reduce: Optional[bool],
    emit_warning: bool = True,
) -> int:
    # 调用 get_enum 函数，将 legacy_get_string 的结果转换为整数枚举值
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
```