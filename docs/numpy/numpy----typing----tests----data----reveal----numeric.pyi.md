# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\numeric.pyi`

```py
"""
Tests for :mod:`_core.numeric`.

Does not include tests which fall under ``array_constructors``.
"""

# 导入系统模块sys和类型提示模块Any
import sys
from typing import Any

# 导入numpy库，并使用简称np
import numpy as np
# 导入numpy.typing模块，并使用简称npt
import numpy.typing as npt

# 根据Python版本导入assert_type函数
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 定义一个SubClass类，继承自npt.NDArray[np.int64]，但没有实现具体功能
class SubClass(npt.NDArray[np.int64]):
    ...

# 定义变量i8为np.int64类型
i8: np.int64

# 定义变量AR_b到AR_O为不同类型的npt.NDArray数组类型
AR_b: npt.NDArray[np.bool]
AR_u8: npt.NDArray[np.uint64]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_O: npt.NDArray[np.object_]

# 定义变量B为int类型的列表
B: list[int]
# 定义变量C为SubClass类的实例

C: SubClass

# 对不同函数调用进行类型断言
assert_type(np.count_nonzero(i8), int)
assert_type(np.count_nonzero(AR_i8), int)
assert_type(np.count_nonzero(B), int)
assert_type(np.count_nonzero(AR_i8, keepdims=True), Any)
assert_type(np.count_nonzero(AR_i8, axis=0), Any)

assert_type(np.isfortran(i8), bool)
assert_type(np.isfortran(AR_i8), bool)

assert_type(np.argwhere(i8), npt.NDArray[np.intp])
assert_type(np.argwhere(AR_i8), npt.NDArray[np.intp])

assert_type(np.flatnonzero(i8), npt.NDArray[np.intp])
assert_type(np.flatnonzero(AR_i8), npt.NDArray[np.intp])

assert_type(np.correlate(B, AR_i8, mode="valid"), npt.NDArray[np.signedinteger[Any]])
assert_type(np.correlate(AR_i8, AR_i8, mode="same"), npt.NDArray[np.signedinteger[Any]])
assert_type(np.correlate(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.correlate(AR_b, AR_u8), npt.NDArray[np.unsignedinteger[Any]])
assert_type(np.correlate(AR_i8, AR_b), npt.NDArray[np.signedinteger[Any]])
assert_type(np.correlate(AR_i8, AR_f8), npt.NDArray[np.floating[Any]])
assert_type(np.correlate(AR_i8, AR_c16), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.correlate(AR_i8, AR_m), npt.NDArray[np.timedelta64])
assert_type(np.correlate(AR_O, AR_O), npt.NDArray[np.object_])

assert_type(np.convolve(B, AR_i8, mode="valid"), npt.NDArray[np.signedinteger[Any]])
assert_type(np.convolve(AR_i8, AR_i8, mode="same"), npt.NDArray[np.signedinteger[Any]])
assert_type(np.convolve(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.convolve(AR_b, AR_u8), npt.NDArray[np.unsignedinteger[Any]])
assert_type(np.convolve(AR_i8, AR_b), npt.NDArray[np.signedinteger[Any]])
assert_type(np.convolve(AR_i8, AR_f8), npt.NDArray[np.floating[Any]])
assert_type(np.convolve(AR_i8, AR_c16), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.convolve(AR_i8, AR_m), npt.NDArray[np.timedelta64])
assert_type(np.convolve(AR_O, AR_O), npt.NDArray[np.object_])

assert_type(np.outer(i8, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.outer(B, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.outer(AR_i8, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.outer(AR_i8, AR_i8, out=C), SubClass)
assert_type(np.outer(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.outer(AR_b, AR_u8), npt.NDArray[np.unsignedinteger[Any]])
assert_type(np.outer(AR_i8, AR_b), npt.NDArray[np.signedinteger[Any]])
# 确保 np.convolve 返回一个包含任意浮点数的多维数组
assert_type(np.convolve(AR_i8, AR_f8), npt.NDArray[np.floating[Any]])

# 确保 np.outer 返回一个包含复数浮点数的二维数组
assert_type(np.outer(AR_i8, AR_c16), npt.NDArray[np.complexfloating[Any, Any]])

# 确保 np.outer 返回一个包含时间增量的二维数组
assert_type(np.outer(AR_i8, AR_m), npt.NDArray[np.timedelta64])

# 确保 np.outer 返回一个包含对象的二维数组
assert_type(np.outer(AR_O, AR_O), npt.NDArray[np.object_])

# 确保 np.tensordot 返回一个包含任意有符号整数的多维数组
assert_type(np.tensordot(B, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.tensordot(AR_i8, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.tensordot(AR_i8, AR_i8, axes=0), npt.NDArray[np.signedinteger[Any]])
assert_type(np.tensordot(AR_i8, AR_i8, axes=(0, 1)), npt.NDArray[np.signedinteger[Any]])

# 确保 np.tensordot 返回一个包含布尔值的多维数组
assert_type(np.tensordot(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.tensordot(AR_b, AR_u8), npt.NDArray[np.unsignedinteger[Any]])
assert_type(np.tensordot(AR_i8, AR_b), npt.NDArray[np.signedinteger[Any]])

# 确保 np.tensordot 返回一个包含任意浮点数的多维数组
assert_type(np.tensordot(AR_i8, AR_f8), npt.NDArray[np.floating[Any]])
assert_type(np.tensordot(AR_i8, AR_c16), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.tensordot(AR_i8, AR_m), npt.NDArray[np.timedelta64])
assert_type(np.tensordot(AR_O, AR_O), npt.NDArray[np.object_])

# 确保 np.isscalar 返回一个布尔值
assert_type(np.isscalar(i8), bool)
assert_type(np.isscalar(AR_i8), bool)
assert_type(np.isscalar(B), bool)

# 确保 np.roll 返回一个包含 int64 类型的数组
assert_type(np.roll(AR_i8, 1), npt.NDArray[np.int64])
assert_type(np.roll(AR_i8, (1, 2)), npt.NDArray[np.int64])
assert_type(np.roll(B, 1), npt.NDArray[Any])

# 确保 np.rollaxis 返回一个包含 int64 类型的数组
assert_type(np.rollaxis(AR_i8, 0, 1), npt.NDArray[np.int64])

# 确保 np.moveaxis 返回一个包含 int64 类型的数组
assert_type(np.moveaxis(AR_i8, 0, 1), npt.NDArray[np.int64])
assert_type(np.moveaxis(AR_i8, (0, 1), (1, 2)), npt.NDArray[np.int64])

# 确保 np.cross 返回一个包含任意有符号整数的多维数组
assert_type(np.cross(B, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.cross(AR_i8, AR_i8), npt.NDArray[np.signedinteger[Any]])
assert_type(np.cross(AR_b, AR_u8), npt.NDArray[np.unsignedinteger[Any]])
assert_type(np.cross(AR_i8, AR_b), npt.NDArray[np.signedinteger[Any]])

# 确保 np.cross 返回一个包含任意浮点数的多维数组
assert_type(np.cross(AR_i8, AR_f8), npt.NDArray[np.floating[Any]])
assert_type(np.cross(AR_i8, AR_c16), npt.NDArray[np.complexfloating[Any, Any]])
assert_type(np.cross(AR_O, AR_O), npt.NDArray[np.object_])

# 确保 np.indices 返回一个包含 int_ 类型的多维数组
assert_type(np.indices([0, 1, 2]), npt.NDArray[np.int_])

# 确保 np.indices 返回一个包含 int_ 类型的元组
assert_type(np.indices([0, 1, 2], sparse=True), tuple[npt.NDArray[np.int_], ...])

# 确保 np.indices 返回一个包含 float64 类型的多维数组
assert_type(np.indices([0, 1, 2], dtype=np.float64), npt.NDArray[np.float64])

# 确保 np.indices 返回一个包含 float64 类型的元组
assert_type(np.indices([0, 1, 2], sparse=True, dtype=np.float64), tuple[npt.NDArray[np.float64], ...])

# 确保 np.indices 返回一个包含任意类型的多维数组
assert_type(np.indices([0, 1, 2], dtype=float), npt.NDArray[Any])

# 确保 np.indices 返回一个包含任意类型的元组
assert_type(np.indices([0, 1, 2], sparse=True, dtype=float), tuple[npt.NDArray[Any], ...])

# 确保 np.binary_repr 返回一个字符串
assert_type(np.binary_repr(1), str)

# 确保 np.base_repr 返回一个字符串
assert_type(np.base_repr(1), str)

# 确保 np.allclose 返回一个布尔值
assert_type(np.allclose(i8, AR_i8), bool)
assert_type(np.allclose(B, AR_i8), bool)
assert_type(np.allclose(AR_i8, AR_i8), bool)

# 确保 np.isclose 返回一个布尔值
assert_type(np.isclose(i8, i8), np.bool)
assert_type(np.isclose(i8, AR_i8), npt.NDArray[np.bool])
assert_type(np.isclose(B, AR_i8), npt.NDArray[np.bool])
assert_type(np.isclose(AR_i8, AR_i8), npt.NDArray[np.bool])

# 确保 np.array_equal 返回一个布尔值
assert_type(np.array_equal(i8, AR_i8), bool)
# 断言：验证两个 NumPy 数组 B 和 AR_i8 是否相等，并返回布尔值结果
assert_type(np.array_equal(B, AR_i8), bool)

# 断言：验证一个 NumPy 数组 AR_i8 是否与自身相等，并返回布尔值结果
assert_type(np.array_equal(AR_i8, AR_i8), bool)

# 断言：验证两个 NumPy 数组 i8 和 AR_i8 是否在数值意义上等价，并返回布尔值结果
assert_type(np.array_equiv(i8, AR_i8), bool)

# 断言：验证两个 NumPy 数组 B 和 AR_i8 是否在数值意义上等价，并返回布尔值结果
assert_type(np.array_equiv(B, AR_i8), bool)

# 断言：验证一个 NumPy 数组 AR_i8 是否与自身在数值意义上等价，并返回布尔值结果
assert_type(np.array_equiv(AR_i8, AR_i8), bool)
```