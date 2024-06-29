# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\np_datetime.pyi`

```
# 导入 NumPy 库，并将其命名为 np
import numpy as np

# 导入 pandas 库的类型定义模块 npt
from pandas._typing import npt

# 定义自定义异常类 OutOfBoundsDatetime，继承自 ValueError
class OutOfBoundsDatetime(ValueError): ...

# 定义自定义异常类 OutOfBoundsTimedelta，继承自 ValueError
class OutOfBoundsTimedelta(ValueError): ...

# 仅用于测试目的的函数，用于从给定的 np.dtype 中获取单位信息
def py_get_unit_from_dtype(dtype: np.dtype): ...

# 将纳秒时间增量 td64 转换为时间结构体字典，使用给定的单位
def py_td64_to_tdstruct(td64: int, unit: int) -> dict: ...

# 将数组 values 转换为指定 dtype 类型，处理溢出情况，返回处理后的数组
def astype_overflowsafe(
    values: np.ndarray,
    dtype: np.dtype,
    copy: bool = ...,
    round_ok: bool = ...,
    is_coerce: bool = ...,
) -> np.ndarray: ...

# 检查给定 dtype 是否是无单位的类型
def is_unitless(dtype: np.dtype) -> bool: ...

# 比较左右两个 ndarray 的分辨率是否不匹配，返回布尔类型的 ndarray
def compare_mismatched_resolutions(
    left: np.ndarray, right: np.ndarray, op
) -> npt.NDArray[np.bool_]: ...

# 对两个 int64 类型的 ndarray 执行安全的加法，返回结果为 int64 类型的 ndarray
def add_overflowsafe(
    left: npt.NDArray[np.int64],
    right: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]: ...

# 获取支持的 dtype 类型，返回一个 np.dtype 对象
def get_supported_dtype(dtype: np.dtype) -> np.dtype: ...

# 检查给定的 dtype 是否是支持的 dtype 类型，返回布尔值
def is_supported_dtype(dtype: np.dtype) -> bool: ...
```