# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\arraysetops.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 类型标注模块
from numpy.lib._arraysetops_impl import (  # 导入 NumPy 库中的数组操作函数
    UniqueAllResult, UniqueCountsResult, UniqueInverseResult
)

if sys.version_info >= (3, 11):
    from typing import assert_type  # 根据 Python 版本导入不同的类型断言函数
else:
    from typing_extensions import assert_type  # 引入类型断言函数的扩展模块

AR_b: npt.NDArray[np.bool]  # 定义布尔类型的 NumPy 数组标注
AR_i8: npt.NDArray[np.int64]  # 定义 int64 类型的 NumPy 数组标注
AR_f8: npt.NDArray[np.float64]  # 定义 float64 类型的 NumPy 数组标注
AR_M: npt.NDArray[np.datetime64]  # 定义 datetime64 类型的 NumPy 数组标注
AR_O: npt.NDArray[np.object_]  # 定义 object 类型的 NumPy 数组标注

AR_LIKE_f8: list[float]  # 定义元素类型为 float 的列表

assert_type(np.ediff1d(AR_b), npt.NDArray[np.int8])  # 断言 ediff1d 函数返回类型为 int8 的 NumPy 数组
assert_type(np.ediff1d(AR_i8, to_end=[1, 2, 3]), npt.NDArray[np.int64])  # 断言 ediff1d 函数返回类型为 int64 的 NumPy 数组
assert_type(np.ediff1d(AR_M), npt.NDArray[np.timedelta64])  # 断言 ediff1d 函数返回类型为 timedelta64 的 NumPy 数组
assert_type(np.ediff1d(AR_O), npt.NDArray[np.object_])  # 断言 ediff1d 函数返回类型为 object 的 NumPy 数组
assert_type(np.ediff1d(AR_LIKE_f8, to_begin=[1, 1.5]), npt.NDArray[Any])  # 断言 ediff1d 函数返回类型为 Any 的 NumPy 数组

assert_type(np.intersect1d(AR_i8, AR_i8), npt.NDArray[np.int64])  # 断言 intersect1d 函数返回类型为 int64 的 NumPy 数组
assert_type(np.intersect1d(AR_M, AR_M, assume_unique=True), npt.NDArray[np.datetime64])  # 断言 intersect1d 函数返回类型为 datetime64 的 NumPy 数组
assert_type(np.intersect1d(AR_f8, AR_i8), npt.NDArray[Any])  # 断言 intersect1d 函数返回类型为 Any 的 NumPy 数组
assert_type(np.intersect1d(AR_f8, AR_f8, return_indices=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]])  # 断言 intersect1d 函数返回元组，包含三个 NumPy 数组类型

assert_type(np.setxor1d(AR_i8, AR_i8), npt.NDArray[np.int64])  # 断言 setxor1d 函数返回类型为 int64 的 NumPy 数组
assert_type(np.setxor1d(AR_M, AR_M, assume_unique=True), npt.NDArray[np.datetime64])  # 断言 setxor1d 函数返回类型为 datetime64 的 NumPy 数组
assert_type(np.setxor1d(AR_f8, AR_i8), npt.NDArray[Any])  # 断言 setxor1d 函数返回类型为 Any 的 NumPy 数组

assert_type(np.isin(AR_i8, AR_i8), npt.NDArray[np.bool])  # 断言 isin 函数返回类型为 bool 的 NumPy 数组
assert_type(np.isin(AR_M, AR_M, assume_unique=True), npt.NDArray[np.bool])  # 断言 isin 函数返回类型为 bool 的 NumPy 数组
assert_type(np.isin(AR_f8, AR_i8), npt.NDArray[np.bool])  # 断言 isin 函数返回类型为 bool 的 NumPy 数组
assert_type(np.isin(AR_f8, AR_LIKE_f8, invert=True), npt.NDArray[np.bool])  # 断言 isin 函数返回类型为 bool 的 NumPy 数组

assert_type(np.union1d(AR_i8, AR_i8), npt.NDArray[np.int64])  # 断言 union1d 函数返回类型为 int64 的 NumPy 数组
assert_type(np.union1d(AR_M, AR_M), npt.NDArray[np.datetime64])  # 断言 union1d 函数返回类型为 datetime64 的 NumPy 数组
assert_type(np.union1d(AR_f8, AR_i8), npt.NDArray[Any])  # 断言 union1d 函数返回类型为 Any 的 NumPy 数组

assert_type(np.setdiff1d(AR_i8, AR_i8), npt.NDArray[np.int64])  # 断言 setdiff1d 函数返回类型为 int64 的 NumPy 数组
assert_type(np.setdiff1d(AR_M, AR_M, assume_unique=True), npt.NDArray[np.datetime64])  # 断言 setdiff1d 函数返回类型为 datetime64 的 NumPy 数组
assert_type(np.setdiff1d(AR_f8, AR_i8), npt.NDArray[Any])  # 断言 setdiff1d 函数返回类型为 Any 的 NumPy 数组

assert_type(np.unique(AR_f8), npt.NDArray[np.float64])  # 断言 unique 函数返回类型为 float64 的 NumPy 数组
assert_type(np.unique(AR_LIKE_f8, axis=0), npt.NDArray[Any])  # 断言 unique 函数返回类型为 Any 的 NumPy 数组
assert_type(np.unique(AR_f8, return_index=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含两个 NumPy 数组类型
assert_type(np.unique(AR_LIKE_f8, return_index=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含两个 NumPy 数组类型
assert_type(np.unique(AR_f8, return_inverse=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含两个 NumPy 数组类型
assert_type(np.unique(AR_LIKE_f8, return_inverse=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含两个 NumPy 数组类型
assert_type(np.unique(AR_f8, return_counts=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含两个 NumPy 数组类型
assert_type(np.unique(AR_LIKE_f8, return_counts=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含两个 NumPy 数组类型
assert_type(np.unique(AR_f8, return_index=True, return_inverse=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]])  # 断言 unique 函数返回元组，包含三个 NumPy 数组类型
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp], npt.NDArray[np.intp]])
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_f8, return_index=True, return_counts=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]])
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_counts=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp], npt.NDArray[np.intp]])
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_f8, return_inverse=True, return_counts=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]])
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_LIKE_f8, return_inverse=True, return_counts=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp], npt.NDArray[np.intp]])
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_f8, return_index=True, return_inverse=True, return_counts=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]])
# 断言调用 np.unique 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True, return_counts=True), tuple[npt.NDArray[Any], npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]])

# 断言调用 np.unique_all 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_all(AR_f8), UniqueAllResult[np.float64])
# 断言调用 np.unique_all 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_all(AR_LIKE_f8), UniqueAllResult[Any])
# 断言调用 np.unique_counts 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_counts(AR_f8), UniqueCountsResult[np.float64])
# 断言调用 np.unique_counts 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_counts(AR_LIKE_f8), UniqueCountsResult[Any])
# 断言调用 np.unique_inverse 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_inverse(AR_f8), UniqueInverseResult[np.float64])
# 断言调用 np.unique_inverse 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_inverse(AR_LIKE_f8), UniqueInverseResult[Any])
# 断言调用 np.unique_values 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_values(AR_f8), npt.NDArray[np.float64])
# 断言调用 np.unique_values 函数，并验证返回结果的类型符合指定的类型
assert_type(np.unique_values(AR_LIKE_f8), npt.NDArray[Any])
```