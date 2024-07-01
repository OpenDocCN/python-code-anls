# `.\numpy\numpy\typing\tests\data\reveal\shape_base.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型，表示任意类型

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型注解模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # Python 版本大于等于 3.11，使用标准库中的 assert_type
else:
    from typing_extensions import assert_type  # Python 版本小于 3.11，使用 typing_extensions 中的 assert_type

i8: np.int64  # 定义 i8 为 np.int64 类型的注解
f8: np.float64  # 定义 f8 为 np.float64 类型的注解

AR_b: npt.NDArray[np.bool]  # 定义 AR_b 为布尔类型的 NumPy 数组
AR_i8: npt.NDArray[np.int64]  # 定义 AR_i8 为 np.int64 类型的 NumPy 数组
AR_f8: npt.NDArray[np.float64]  # 定义 AR_f8 为 np.float64 类型的 NumPy 数组

AR_LIKE_f8: list[float]  # 定义 AR_LIKE_f8 为浮点数类型的列表

assert_type(np.take_along_axis(AR_f8, AR_i8, axis=1), npt.NDArray[np.float64])  # 断言 np.take_along_axis 的返回类型为 npt.NDArray[np.float64]
assert_type(np.take_along_axis(f8, AR_i8, axis=None), npt.NDArray[np.float64])  # 断言 np.take_along_axis 的返回类型为 npt.NDArray[np.float64]

assert_type(np.put_along_axis(AR_f8, AR_i8, "1.0", axis=1), None)  # 断言 np.put_along_axis 的返回类型为 None

assert_type(np.expand_dims(AR_i8, 2), npt.NDArray[np.int64])  # 断言 np.expand_dims 的返回类型为 npt.NDArray[np.int64]
assert_type(np.expand_dims(AR_LIKE_f8, 2), npt.NDArray[Any])  # 断言 np.expand_dims 的返回类型为 npt.NDArray[Any]

assert_type(np.column_stack([AR_i8]), npt.NDArray[np.int64])  # 断言 np.column_stack 的返回类型为 npt.NDArray[np.int64]
assert_type(np.column_stack([AR_LIKE_f8]), npt.NDArray[Any])  # 断言 np.column_stack 的返回类型为 npt.NDArray[Any]

assert_type(np.dstack([AR_i8]), npt.NDArray[np.int64])  # 断言 np.dstack 的返回类型为 npt.NDArray[np.int64]
assert_type(np.dstack([AR_LIKE_f8]), npt.NDArray[Any])  # 断言 np.dstack 的返回类型为 npt.NDArray[Any]

assert_type(np.array_split(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])  # 断言 np.array_split 的返回类型为 list[npt.NDArray[np.int64]]]
assert_type(np.array_split(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])  # 断言 np.array_split 的返回类型为 list[npt.NDArray[Any]]

assert_type(np.split(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])  # 断言 np.split 的返回类型为 list[npt.NDArray[np.int64]]
assert_type(np.split(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])  # 断言 np.split 的返回类型为 list[npt.NDArray[Any]]

assert_type(np.hsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])  # 断言 np.hsplit 的返回类型为 list[npt.NDArray[np.int64]]
assert_type(np.hsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])  # 断言 np.hsplit 的返回类型为 list[npt.NDArray[Any]]

assert_type(np.vsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])  # 断言 np.vsplit 的返回类型为 list[npt.NDArray[np.int64]]
assert_type(np.vsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])  # 断言 np.vsplit 的返回类型为 list[npt.NDArray[Any]]

assert_type(np.dsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])  # 断言 np.dsplit 的返回类型为 list[npt.NDArray[np.int64]]
assert_type(np.dsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])  # 断言 np.dsplit 的返回类型为 list[npt.NDArray[Any]]

assert_type(np.kron(AR_b, AR_b), npt.NDArray[np.bool])  # 断言 np.kron 的返回类型为 npt.NDArray[np.bool]
assert_type(np.kron(AR_b, AR_i8), npt.NDArray[np.signedinteger[Any]])  # 断言 np.kron 的返回类型为 npt.NDArray[np.signedinteger[Any]]
assert_type(np.kron(AR_f8, AR_f8), npt.NDArray[np.floating[Any]])  # 断言 np.kron 的返回类型为 npt.NDArray[np.floating[Any]]

assert_type(np.tile(AR_i8, 5), npt.NDArray[np.int64])  # 断言 np.tile 的返回类型为 npt.NDArray[np.int64]
assert_type(np.tile(AR_LIKE_f8, [2, 2]), npt.NDArray[Any])  # 断言 np.tile 的返回类型为 npt.NDArray[Any]
```