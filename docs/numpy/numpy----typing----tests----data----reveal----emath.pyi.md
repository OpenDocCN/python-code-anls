# `.\numpy\numpy\typing\tests\data\reveal\emath.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型标注模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # Python 版本大于等于 3.11 使用标准库中的 assert_type
else:
    from typing_extensions import assert_type  # 否则使用 typing_extensions 库中的 assert_type

AR_f8: npt.NDArray[np.float64]  # 声明 AR_f8 变量为 np.float64 类型的 NumPy 数组
AR_c16: npt.NDArray[np.complex128]  # 声明 AR_c16 变量为 np.complex128 类型的 NumPy 数组
f8: np.float64  # 声明 f8 变量为 np.float64 类型
c16: np.complex128  # 声明 c16 变量为 np.complex128 类型

assert_type(np.emath.sqrt(f8), Any)  # 断言 np.emath.sqrt(f8) 的返回类型为 Any
assert_type(np.emath.sqrt(AR_f8), npt.NDArray[Any])  # 断言 np.emath.sqrt(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.sqrt(c16), np.complexfloating[Any, Any])  # 断言 np.emath.sqrt(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.sqrt(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.sqrt(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.log(f8), Any)  # 断言 np.emath.log(f8) 的返回类型为 Any
assert_type(np.emath.log(AR_f8), npt.NDArray[Any])  # 断言 np.emath.log(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.log(c16), np.complexfloating[Any, Any])  # 断言 np.emath.log(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.log(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.log(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.log10(f8), Any)  # 断言 np.emath.log10(f8) 的返回类型为 Any
assert_type(np.emath.log10(AR_f8), npt.NDArray[Any])  # 断言 np.emath.log10(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.log10(c16), np.complexfloating[Any, Any])  # 断言 np.emath.log10(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.log10(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.log10(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.log2(f8), Any)  # 断言 np.emath.log2(f8) 的返回类型为 Any
assert_type(np.emath.log2(AR_f8), npt.NDArray[Any])  # 断言 np.emath.log2(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.log2(c16), np.complexfloating[Any, Any])  # 断言 np.emath.log2(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.log2(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.log2(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.logn(f8, 2), Any)  # 断言 np.emath.logn(f8, 2) 的返回类型为 Any
assert_type(np.emath.logn(AR_f8, 4), npt.NDArray[Any])  # 断言 np.emath.logn(AR_f8, 4) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.logn(f8, 1j), np.complexfloating[Any, Any])  # 断言 np.emath.logn(f8, 1j) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.logn(AR_c16, 1.5), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.logn(AR_c16, 1.5) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.power(f8, 2), Any)  # 断言 np.emath.power(f8, 2) 的返回类型为 Any
assert_type(np.emath.power(AR_f8, 4), npt.NDArray[Any])  # 断言 np.emath.power(AR_f8, 4) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.power(f8, 2j), np.complexfloating[Any, Any])  # 断言 np.emath.power(f8, 2j) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.power(AR_c16, 1.5), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.power(AR_c16, 1.5) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.arccos(f8), Any)  # 断言 np.emath.arccos(f8) 的返回类型为 Any
assert_type(np.emath.arccos(AR_f8), npt.NDArray[Any])  # 断言 np.emath.arccos(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.arccos(c16), np.complexfloating[Any, Any])  # 断言 np.emath.arccos(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.arccos(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.arccos(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.arcsin(f8), Any)  # 断言 np.emath.arcsin(f8) 的返回类型为 Any
assert_type(np.emath.arcsin(AR_f8), npt.NDArray[Any])  # 断言 np.emath.arcsin(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.arcsin(c16), np.complexfloating[Any, Any])  # 断言 np.emath.arcsin(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.arcsin(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.arcsin(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.emath.arctanh(f8), Any)  # 断言 np.emath.arctanh(f8) 的返回类型为 Any
assert_type(np.emath.arctanh(AR_f8), npt.NDArray[Any])  # 断言 np.emath.arctanh(AR_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.emath.arctanh(c16), np.complexfloating[Any, Any])  # 断言 np.emath.arctanh(c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.emath.arctanh(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.emath.arctanh(AR_c16) 的返回类型为 npt.NDArray[np.complexfloating[Any, Any]]
```