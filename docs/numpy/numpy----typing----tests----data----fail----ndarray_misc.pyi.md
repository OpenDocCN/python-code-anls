# `.\numpy\numpy\typing\tests\data\fail\ndarray_misc.pyi`

```py
"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.
"""

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型注解模块

f8: np.float64  # 声明一个类型为 np.float64 的变量 f8
AR_f8: npt.NDArray[np.float64]  # 声明一个 np.float64 类型的 NumPy 数组 AR_f8
AR_M: npt.NDArray[np.datetime64]  # 声明一个 np.datetime64 类型的 NumPy 数组 AR_M
AR_b: npt.NDArray[np.bool]  # 声明一个 np.bool 类型的 NumPy 数组 AR_b

ctypes_obj = AR_f8.ctypes  # 获取 AR_f8 数组的 ctypes 对象

reveal_type(ctypes_obj.get_data())  # E: has no attribute
reveal_type(ctypes_obj.get_shape())  # E: has no attribute
reveal_type(ctypes_obj.get_strides())  # E: has no attribute
reveal_type(ctypes_obj.get_as_parameter())  # E: has no attribute

f8.argpartition(0)  # E: has no attribute
f8.diagonal()  # E: has no attribute
f8.dot(1)  # E: has no attribute
f8.nonzero()  # E: has no attribute
f8.partition(0)  # E: has no attribute
f8.put(0, 2)  # E: has no attribute
f8.setfield(2, np.float64)  # E: has no attribute
f8.sort()  # E: has no attribute
f8.trace()  # E: has no attribute

AR_M.__int__()  # E: Invalid self argument
AR_M.__float__()  # E: Invalid self argument
AR_M.__complex__()  # E: Invalid self argument
AR_b.__index__()  # E: Invalid self argument

AR_f8[1.5]  # E: No overload variant
AR_f8["field_a"]  # E: No overload variant
AR_f8[["field_a", "field_b"]]  # E: Invalid index type

AR_f8.__array_finalize__(object())  # E: incompatible type
```