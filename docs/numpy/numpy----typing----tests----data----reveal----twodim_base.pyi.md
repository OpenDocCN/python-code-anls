# `.\numpy\numpy\typing\tests\data\reveal\twodim_base.pyi`

```py
import sys
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt  # 导入 NumPy 类型提示模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # Python 版本大于等于 3.11 使用标准库中的 assert_type
else:
    from typing_extensions import assert_type  # 否则使用 typing_extensions 中的 assert_type

_SCT = TypeVar("_SCT", bound=np.generic)  # 定义泛型类型 _SCT，限制为 NumPy 泛型类型

def func1(ar: npt.NDArray[_SCT], a: int) -> npt.NDArray[_SCT]:
    pass  # 函数 func1 接受一个 NumPy 数组 ar 和一个整数 a，返回一个与 ar 类型相同的 NumPy 数组

def func2(ar: npt.NDArray[np.number[Any]], a: str) -> npt.NDArray[np.float64]:
    pass  # 函数 func2 接受一个元素类型为 np.number[Any] 的 NumPy 数组 ar 和一个字符串 a，返回一个 np.float64 类型的 NumPy 数组

AR_b: npt.NDArray[np.bool]  # 声明一个布尔类型的 NumPy 数组 AR_b
AR_u: npt.NDArray[np.uint64]  # 声明一个无符号 64 位整数类型的 NumPy 数组 AR_u
AR_i: npt.NDArray[np.int64]  # 声明一个有符号 64 位整数类型的 NumPy 数组 AR_i
AR_f: npt.NDArray[np.float64]  # 声明一个双精度浮点数类型的 NumPy 数组 AR_f
AR_c: npt.NDArray[np.complex128]  # 声明一个复数类型的 NumPy 数组 AR_c
AR_O: npt.NDArray[np.object_]  # 声明一个对象类型的 NumPy 数组 AR_O

AR_LIKE_b: list[bool]  # 声明一个布尔类型的列表 AR_LIKE_b

assert_type(np.fliplr(AR_b), npt.NDArray[np.bool])  # 对 np.fliplr(AR_b) 的返回类型进行断言为 np.bool 类型的 NumPy 数组
assert_type(np.fliplr(AR_LIKE_b), npt.NDArray[Any])  # 对 np.fliplr(AR_LIKE_b) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.flipud(AR_b), npt.NDArray[np.bool])  # 对 np.flipud(AR_b) 的返回类型进行断言为 np.bool 类型的 NumPy 数组
assert_type(np.flipud(AR_LIKE_b), npt.NDArray[Any])  # 对 np.flipud(AR_LIKE_b) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.eye(10), npt.NDArray[np.float64])  # 对 np.eye(10) 的返回类型进行断言为 np.float64 类型的 NumPy 数组
assert_type(np.eye(10, M=20, dtype=np.int64), npt.NDArray[np.int64])  # 对 np.eye(10, M=20, dtype=np.int64) 的返回类型进行断言为 np.int64 类型的 NumPy 数组
assert_type(np.eye(10, k=2, dtype=int), npt.NDArray[Any])  # 对 np.eye(10, k=2, dtype=int) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.diag(AR_b), npt.NDArray[np.bool])  # 对 np.diag(AR_b) 的返回类型进行断言为 np.bool 类型的 NumPy 数组
assert_type(np.diag(AR_LIKE_b, k=0), npt.NDArray[Any])  # 对 np.diag(AR_LIKE_b, k=0) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.diagflat(AR_b), npt.NDArray[np.bool])  # 对 np.diagflat(AR_b) 的返回类型进行断言为 np.bool 类型的 NumPy 数组
assert_type(np.diagflat(AR_LIKE_b, k=0), npt.NDArray[Any])  # 对 np.diagflat(AR_LIKE_b, k=0) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.tri(10), npt.NDArray[np.float64])  # 对 np.tri(10) 的返回类型进行断言为 np.float64 类型的 NumPy 数组
assert_type(np.tri(10, M=20, dtype=np.int64), npt.NDArray[np.int64])  # 对 np.tri(10, M=20, dtype=np.int64) 的返回类型进行断言为 np.int64 类型的 NumPy 数组
assert_type(np.tri(10, k=2, dtype=int), npt.NDArray[Any])  # 对 np.tri(10, k=2, dtype=int) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.tril(AR_b), npt.NDArray[np.bool])  # 对 np.tril(AR_b) 的返回类型进行断言为 np.bool 类型的 NumPy 数组
assert_type(np.tril(AR_LIKE_b, k=0), npt.NDArray[Any])  # 对 np.tril(AR_LIKE_b, k=0) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.triu(AR_b), npt.NDArray[np.bool])  # 对 np.triu(AR_b) 的返回类型进行断言为 np.bool 类型的 NumPy 数组
assert_type(np.triu(AR_LIKE_b, k=0), npt.NDArray[Any])  # 对 np.triu(AR_LIKE_b, k=0) 的返回类型进行断言为任意类型的 NumPy 数组

assert_type(np.vander(AR_b), npt.NDArray[np.signedinteger[Any]])  # 对 np.vander(AR_b) 的返回类型进行断言为 np.signedinteger[Any] 类型的 NumPy 数组
assert_type(np.vander(AR_u), npt.NDArray[np.signedinteger[Any]])  # 对 np.vander(AR_u) 的返回类型进行断言为 np.signedinteger[Any] 类型的 NumPy 数组
assert_type(np.vander(AR_i, N=2), npt.NDArray[np.signedinteger[Any]])  # 对 np.vander(AR_i, N=2) 的返回类型进行断言为 np.signedinteger[Any] 类型的 NumPy 数组
assert_type(np.vander(AR_f, increasing=True), npt.NDArray[np.floating[Any]])  # 对 np.vander(AR_f, increasing=True) 的返回类型进行断言为 np.floating[Any] 类型的 NumPy 数组
assert_type(np.vander(AR_c), npt.NDArray[np.complexfloating[Any, Any]])  # 对 np.vander(AR_c) 的返回类型进行断言为 np.complexfloating[Any, Any] 类型的 NumPy 数组
assert_type(np.vander(AR_O), npt.NDArray[np.object_])  # 对 np.vander(AR_O) 的返回类型进行断言为 np.object_ 类型的 NumPy 数组

assert_type(
    np.histogram2d(AR_i, AR_b),
    tuple[
        npt.NDArray[np.float64],  # 对 np.histogram2d(AR_i, AR_b) 的返回类型的第一个元素进行断言为 np.float64 类型的 NumPy 数组
        npt.NDArray[np.floating[Any]],  # 对 np.histogram2d(AR_i, AR_b) 的返回类型的第二个元素进行断言为 np.floating[Any] 类型的 NumPy 数组
        npt.NDArray[np.floating[Any]],  # 对 np.histogram2d(AR_i, AR_b) 的返回类型的第三个元素进行断言为 np.floating[Any] 类型的 NumPy 数组
    ],
)
assert_type(
    np.histogram2d(AR_f, AR_f),
    tuple[
        npt.NDArray[np.float64],  # 对 np.histogram2d(AR_f, AR_f) 的返回类型的第一个元素进行断言为 np.float64 类型的 NumPy 数组
        npt.NDArray[np.floating[Any]],  # 对 np.histogram2d(AR_f, AR_f) 的返回类型的第二个元素进行断言为 np.floating[Any] 类型的 NumPy 数组
        npt.NDArray[np.floating[Any]],  # 对 np.histogram2d(AR_f, AR_f) 的返回类型的第三个元素进行断言为 np.floating[Any] 类型的 NumPy 数组
    ],
# 使用 NumPy 函数 np.triu_indices 生成一个包含给定大小的上三角矩阵的索引元组，并断言其类型为 (NumPy 整数数组, NumPy 整数数组)
assert_type(np.triu_indices(10), tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]])

# 使用 NumPy 函数 np.triu_indices_from 生成一个与给定数组 AR_b 大小相同的上三角矩阵的索引元组，并断言其类型为 (NumPy 整数数组, NumPy 整数数组)
assert_type(np.triu_indices_from(AR_b), tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]])
```