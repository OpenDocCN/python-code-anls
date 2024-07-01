# `.\numpy\numpy\typing\tests\data\reveal\fromnumeric.pyi`

```py
"""Tests for :mod:`_core.fromnumeric`."""

# 导入系统和类型相关的模块
import sys
from typing import Any

# 导入 NumPy 库及其类型提示模块
import numpy as np
import numpy.typing as npt

# 根据 Python 版本导入不同的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 定义一个 NumPy 的子类 NDArraySubclass，继承自 npt.NDArray[np.complex128]
class NDArraySubclass(npt.NDArray[np.complex128]):
    ...

# 声明一些 NumPy 数组的类型注解
AR_b: npt.NDArray[np.bool]      # 布尔类型的 NumPy 数组
AR_f4: npt.NDArray[np.float32]  # 单精度浮点数类型的 NumPy 数组
AR_c16: npt.NDArray[np.complex128]  # 复数类型的 NumPy 数组
AR_u8: npt.NDArray[np.uint64]   # 64 位无符号整数类型的 NumPy 数组
AR_i8: npt.NDArray[np.int64]    # 64 位有符号整数类型的 NumPy 数组
AR_O: npt.NDArray[np.object_]   # Python 对象类型的 NumPy 数组
AR_subclass: NDArraySubclass    # 自定义的 NumPy 数组子类 NDArraySubclass

# 声明一些普通的 Python 变量
b: np.bool       # 布尔类型变量
f4: np.float32   # 单精度浮点数类型变量
i8: np.int64     # 64 位有符号整数类型变量
f: float         # 浮点数类型变量

# 使用 assert_type 函数验证不同 NumPy 函数返回值的类型
assert_type(np.take(b, 0), np.bool)
assert_type(np.take(f4, 0), np.float32)
assert_type(np.take(f, 0), Any)
assert_type(np.take(AR_b, 0), np.bool)
assert_type(np.take(AR_f4, 0), np.float32)
assert_type(np.take(AR_b, [0]), npt.NDArray[np.bool])
assert_type(np.take(AR_f4, [0]), npt.NDArray[np.float32])
assert_type(np.take([1], [0]), npt.NDArray[Any])
assert_type(np.take(AR_f4, [0], out=AR_subclass), NDArraySubclass)

assert_type(np.reshape(b, 1), npt.NDArray[np.bool])
assert_type(np.reshape(f4, 1), npt.NDArray[np.float32])
assert_type(np.reshape(f, 1), npt.NDArray[Any])
assert_type(np.reshape(AR_b, 1), npt.NDArray[np.bool])
assert_type(np.reshape(AR_f4, 1), npt.NDArray[np.float32])

assert_type(np.choose(1, [True, True]), Any)
assert_type(np.choose([1], [True, True]), npt.NDArray[Any])
assert_type(np.choose([1], AR_b), npt.NDArray[np.bool])
assert_type(np.choose([1], AR_b, out=AR_f4), npt.NDArray[np.float32])

assert_type(np.repeat(b, 1), npt.NDArray[np.bool])
assert_type(np.repeat(f4, 1), npt.NDArray[np.float32])
assert_type(np.repeat(f, 1), npt.NDArray[Any])
assert_type(np.repeat(AR_b, 1), npt.NDArray[np.bool])
assert_type(np.repeat(AR_f4, 1), npt.NDArray[np.float32])

# TODO: array_bdd tests for np.put()

assert_type(np.swapaxes([[0, 1]], 0, 0), npt.NDArray[Any])
assert_type(np.swapaxes(AR_b, 0, 0), npt.NDArray[np.bool])
assert_type(np.swapaxes(AR_f4, 0, 0), npt.NDArray[np.float32])

assert_type(np.transpose(b), npt.NDArray[np.bool])
assert_type(np.transpose(f4), npt.NDArray[np.float32])
assert_type(np.transpose(f), npt.NDArray[Any])
assert_type(np.transpose(AR_b), npt.NDArray[np.bool])
assert_type(np.transpose(AR_f4), npt.NDArray[np.float32])

assert_type(np.partition(b, 0, axis=None), npt.NDArray[np.bool])
assert_type(np.partition(f4, 0, axis=None), npt.NDArray[np.float32])
assert_type(np.partition(f, 0, axis=None), npt.NDArray[Any])
assert_type(np.partition(AR_b, 0), npt.NDArray[np.bool])
assert_type(np.partition(AR_f4, 0), npt.NDArray[np.float32])

assert_type(np.argpartition(b, 0), npt.NDArray[np.intp])
assert_type(np.argpartition(f4, 0), npt.NDArray[np.intp])
assert_type(np.argpartition(f, 0), npt.NDArray[np.intp])
assert_type(np.argpartition(AR_b, 0), npt.NDArray[np.intp])
assert_type(np.argpartition(AR_f4, 0), npt.NDArray[np.intp])

assert_type(np.sort([2, 1], 0), npt.NDArray[Any])
assert_type(np.sort(AR_b, 0), npt.NDArray[np.bool])
# 确保 AR_f4 按列排序后的类型为 np.float32 的 NumPy 数组
assert_type(np.sort(AR_f4, 0), npt.NDArray[np.float32])

# 确保 AR_b 按列排序后的类型为 np.intp 的 NumPy 数组
assert_type(np.argsort(AR_b, 0), npt.NDArray[np.intp])
# 确保 AR_f4 按列排序后的类型为 np.intp 的 NumPy 数组
assert_type(np.argsort(AR_f4, 0), npt.NDArray[np.intp])

# 确保 AR_b 的最大值索引的类型为 np.intp
assert_type(np.argmax(AR_b), np.intp)
# 确保 AR_f4 的最大值索引的类型为 np.intp
assert_type(np.argmax(AR_f4), np.intp)
# 确保 AR_b 按列最大值索引的类型为 Any
assert_type(np.argmax(AR_b, axis=0), Any)
# 确保 AR_f4 按列最大值索引的类型为 Any
assert_type(np.argmax(AR_f4, axis=0), Any)
# 确保 AR_f4 按列最大值索引并将结果存入 AR_subclass 的类型为 NDArraySubclass
assert_type(np.argmax(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保 AR_b 的最小值索引的类型为 np.intp
assert_type(np.argmin(AR_b), np.intp)
# 确保 AR_f4 的最小值索引的类型为 np.intp
assert_type(np.argmin(AR_f4), np.intp)
# 确保 AR_b 按列最小值索引的类型为 Any
assert_type(np.argmin(AR_b, axis=0), Any)
# 确保 AR_f4 按列最小值索引的类型为 Any
assert_type(np.argmin(AR_f4, axis=0), Any)
# 确保 AR_f4 按列最小值索引并将结果存入 AR_subclass 的类型为 NDArraySubclass
assert_type(np.argmin(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保在 AR_b[0] 中搜索 0 的索引的类型为 np.intp
assert_type(np.searchsorted(AR_b[0], 0), np.intp)
# 确保在 AR_f4[0] 中搜索 0 的索引的类型为 np.intp
assert_type(np.searchsorted(AR_f4[0], 0), np.intp)
# 确保在 AR_b[0] 中搜索 [0] 的索引的类型为 npt.NDArray[np.intp]
assert_type(np.searchsorted(AR_b[0], [0]), npt.NDArray[np.intp])
# 确保在 AR_f4[0] 中搜索 [0] 的索引的类型为 npt.NDArray[np.intp]
assert_type(np.searchsorted(AR_f4[0], [0]), npt.NDArray[np.intp])

# 确保将 b 调整为 (5, 5) 大小的数组的类型为 npt.NDArray[np.bool]
assert_type(np.resize(b, (5, 5)), npt.NDArray[np.bool])
# 确保将 f4 调整为 (5, 5) 大小的数组的类型为 npt.NDArray[np.float32]
assert_type(np.resize(f4, (5, 5)), npt.NDArray[np.float32])
# 确保将 f 调整为 (5, 5) 大小的数组的类型为 npt.NDArray[Any]
assert_type(np.resize(f, (5, 5)), npt.NDArray[Any])
# 确保将 AR_b 调整为 (5, 5) 大小的数组的类型为 npt.NDArray[np.bool]
assert_type(np.resize(AR_b, (5, 5)), npt.NDArray[np.bool])
# 确保将 AR_f4 调整为 (5, 5) 大小的数组的类型为 npt.NDArray[np.float32]
assert_type(np.resize(AR_f4, (5, 5)), npt.NDArray[np.float32])

# 确保将 b 去除所有维度为 1 的条目后的类型为 np.bool
assert_type(np.squeeze(b), np.bool)
# 确保将 f4 去除所有维度为 1 的条目后的类型为 np.float32
assert_type(np.squeeze(f4), np.float32)
# 确保将 f 去除所有维度为 1 的条目后的类型为 npt.NDArray[Any]
assert_type(np.squeeze(f), npt.NDArray[Any])
# 确保将 AR_b 去除所有维度为 1 的条目后的类型为 npt.NDArray[np.bool]
assert_type(np.squeeze(AR_b), npt.NDArray[np.bool])
# 确保将 AR_f4 去除所有维度为 1 的条目后的类型为 npt.NDArray[np.float32]
assert_type(np.squeeze(AR_f4), npt.NDArray[np.float32])

# 确保返回 AR_b 的对角线元素的类型为 npt.NDArray[np.bool]
assert_type(np.diagonal(AR_b), npt.NDArray[np.bool])
# 确保返回 AR_f4 的对角线元素的类型为 npt.NDArray[np.float32]
assert_type(np.diagonal(AR_f4), npt.NDArray[np.float32])

# 确保返回 AR_b 的对角线元素的和的类型为 Any
assert_type(np.trace(AR_b), Any)
# 确保返回 AR_f4 的对角线元素的和的类型为 Any
assert_type(np.trace(AR_f4), Any)
# 确保返回 AR_f4 的对角线元素的和并将结果存入 AR_subclass 的类型为 NDArraySubclass
assert_type(np.trace(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保将 b 扁平化后的类型为 npt.NDArray[np.bool]
assert_type(np.ravel(b), npt.NDArray[np.bool])
# 确保将 f4 扁平化后的类型为 npt.NDArray[np.float32]
assert_type(np.ravel(f4), npt.NDArray[np.float32])
# 确保将 f 扁平化后的类型为 npt.NDArray[Any]
assert_type(np.ravel(f), npt.NDArray[Any])
# 确保将 AR_b 扁平化后的类型为 npt.NDArray[np.bool]
assert_type(np.ravel(AR_b), npt.NDArray[np.bool])
# 确保将 AR_f4 扁平化后的类型为 npt.NDArray[np.float32]
assert_type(np.ravel(AR_f4), npt.NDArray[np.float32])

# 确保返回 b 中非零元素的索引的类型为 tuple[npt.NDArray[np.intp], ...]
assert_type(np.nonzero(b), tuple[npt.NDArray[np.intp], ...])
# 确保返回 f4 中非零元素的索引的类型为 tuple[npt.NDArray[np.intp], ...]
assert_type(np.nonzero(f4), tuple[npt.NDArray[np.intp], ...])
# 确保返回 f 中非零元素的索引的类型为 tuple[npt.NDArray[np.intp], ...]
assert_type(np.nonzero(f), tuple[npt.NDArray[np.intp], ...])
# 确保返回 AR_b 中非零元素的索引的类型为 tuple[npt.NDArray[np.intp], ...]
assert_type(np.nonzero(AR_b), tuple[npt.NDArray[np.intp], ...])
# 确保返回 AR_f4 中非零元素的索引的类型为 tuple[npt.NDArray[np.intp], ...]
assert_type(np.nonzero(AR_f4), tuple[npt.NDArray[np.intp], ...])

# 确保返回 b 的形状的类型为 tuple[int, ...]
assert_type(np.shape(b), tuple[int, ...])
# 确保返回 f4 的形状的类型为 tuple[int, ...]
assert_type(np.shape(f4), tuple[int, ...])
# 确保返回 f 的形状的类型为 tuple[int, ...]
assert_type(np.shape(f), tuple[int, ...])
# 确保返回 AR_b 的形状的类型为 tuple[int, ...]
assert_type(np.shape(AR_b), tuple[int, ...])
# 确保返回 AR_f4 的形状的类型为 tuple[int, ...]
assert_type(np
# 断言：检查 np.clip 函数对 AR_b 进行裁剪后的输出类型是否为 NDArraySubclass 类型
assert_type(np.clip(AR_b, 0, 1, out=AR_subclass), NDArraySubclass)

# 断言：检查 np.sum 函数对数组 b 的输出类型是否为 np.bool 类型
assert_type(np.sum(b), np.bool)
# 断言：检查 np.sum 函数对数组 f4 的输出类型是否为 np.float32 类型
assert_type(np.sum(f4), np.float32)
# 断言：检查 np.sum 函数对数组 f 的输出类型是否为 Any（任意类型）
assert_type(np.sum(f), Any)
# 断言：检查 np.sum 函数对数组 AR_b 的输出类型是否为 np.bool 类型
assert_type(np.sum(AR_b), np.bool)
# 断言：检查 np.sum 函数对数组 AR_f4 的输出类型是否为 np.float32 类型
assert_type(np.sum(AR_f4), np.float32)
# 断言：检查 np.sum 函数对数组 AR_b 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.sum(AR_b, axis=0), Any)
# 断言：检查 np.sum 函数对数组 AR_f4 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.sum(AR_f4, axis=0), Any)
# 断言：检查 np.sum 函数对数组 AR_f4 输出到指定的 AR_subclass 对象后的输出类型是否为 NDArraySubclass 类型
assert_type(np.sum(AR_f4, out=AR_subclass), NDArraySubclass)

# 断言：检查 np.all 函数对数组 b 的输出类型是否为 np.bool 类型
assert_type(np.all(b), np.bool)
# 断言：检查 np.all 函数对数组 f4 的输出类型是否为 np.bool 类型
assert_type(np.all(f4), np.bool)
# 断言：检查 np.all 函数对数组 f 的输出类型是否为 np.bool 类型
assert_type(np.all(f), np.bool)
# 断言：检查 np.all 函数对数组 AR_b 的输出类型是否为 np.bool 类型
assert_type(np.all(AR_b), np.bool)
# 断言：检查 np.all 函数对数组 AR_f4 的输出类型是否为 np.bool 类型
assert_type(np.all(AR_f4), np.bool)
# 断言：检查 np.all 函数对数组 AR_b 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.all(AR_b, axis=0), Any)
# 断言：检查 np.all 函数对数组 AR_f4 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.all(AR_f4, axis=0), Any)
# 断言：检查 np.all 函数对数组 AR_b 保持维度（keepdims=True）后的输出类型是否为 Any（任意类型）
assert_type(np.all(AR_b, keepdims=True), Any)
# 断言：检查 np.all 函数对数组 AR_f4 保持维度（keepdims=True）后的输出类型是否为 Any（任意类型）
assert_type(np.all(AR_f4, keepdims=True), Any)
# 断言：检查 np.all 函数对数组 AR_f4 输出到指定的 AR_subclass 对象后的输出类型是否为 NDArraySubclass 类型
assert_type(np.all(AR_f4, out=AR_subclass), NDArraySubclass)

# 断言：检查 np.any 函数对数组 b 的输出类型是否为 np.bool 类型
assert_type(np.any(b), np.bool)
# 断言：检查 np.any 函数对数组 f4 的输出类型是否为 np.bool 类型
assert_type(np.any(f4), np.bool)
# 断言：检查 np.any 函数对数组 f 的输出类型是否为 np.bool 类型
assert_type(np.any(f), np.bool)
# 断言：检查 np.any 函数对数组 AR_b 的输出类型是否为 np.bool 类型
assert_type(np.any(AR_b), np.bool)
# 断言：检查 np.any 函数对数组 AR_f4 的输出类型是否为 np.bool 类型
assert_type(np.any(AR_f4), np.bool)
# 断言：检查 np.any 函数对数组 AR_b 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.any(AR_b, axis=0), Any)
# 断言：检查 np.any 函数对数组 AR_f4 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.any(AR_f4, axis=0), Any)
# 断言：检查 np.any 函数对数组 AR_b 保持维度（keepdims=True）后的输出类型是否为 Any（任意类型）
assert_type(np.any(AR_b, keepdims=True), Any)
# 断言：检查 np.any 函数对数组 AR_f4 保持维度（keepdims=True）后的输出类型是否为 Any（任意类型）
assert_type(np.any(AR_f4, keepdims=True), Any)
# 断言：检查 np.any 函数对数组 AR_f4 输出到指定的 AR_subclass 对象后的输出类型是否为 NDArraySubclass 类型
assert_type(np.any(AR_f4, out=AR_subclass), NDArraySubclass)

# 断言：检查 np.cumsum 函数对数组 b 的输出类型是否为 npt.NDArray[np.bool] 类型
assert_type(np.cumsum(b), npt.NDArray[np.bool])
# 断言：检查 np.cumsum 函数对数组 f4 的输出类型是否为 npt.NDArray[np.float32] 类型
assert_type(np.cumsum(f4), npt.NDArray[np.float32])
# 断言：检查 np.cumsum 函数对数组 f 的输出类型是否为 npt.NDArray[Any] 类型
assert_type(np.cumsum(f), npt.NDArray[Any])
# 断言：检查 np.cumsum 函数对数组 AR_b 的输出类型是否为 npt.NDArray[np.bool] 类型
assert_type(np.cumsum(AR_b), npt.NDArray[np.bool])
# 断言：检查 np.cumsum 函数对数组 AR_f4 的输出类型是否为 npt.NDArray[np.float32] 类型
assert_type(np.cumsum(AR_f4), npt.NDArray[np.float32])
# 断言：检查 np.cumsum 函数对数组 f 指定数据类型为 float 后的输出类型是否为 npt.NDArray[Any] 类型
assert_type(np.cumsum(f, dtype=float), npt.NDArray[Any])
# 断言：检查 np.cumsum 函数对数组 f 指定数据类型为 np.float64 后的输出类型是否为 npt.NDArray[np.float64] 类型
assert_type(np.cumsum(f, dtype=np.float64), npt.NDArray[np.float64])
# 断言：检查 np.cumsum 函数对数组 AR_f4 输出到指定的 AR_subclass 对象后的输出类型是否为 NDArraySubclass 类型
assert_type(np.cumsum(AR_f4, out=AR_subclass), NDArraySubclass)

# 断言：检查 np.ptp 函数对数组 b 的输出类型是否为 np.bool 类型
assert_type(np.ptp(b), np.bool)
# 断言：检查 np.ptp 函数对数组 f4 的输出类型是否为 np.float32 类型
assert_type(np.ptp(f4), np.float32)
# 断言：检查 np.ptp 函数对数组 f 的输出类型是否为 Any（任意类型）
assert_type(np.ptp(f), Any)
# 断言：检查 np.ptp 函数对数组 AR_b 的输出类型是否为 np.bool 类型
assert_type(np.ptp(AR_b), np.bool)
# 断言：检查 np.ptp 函数对数组 AR_f4 的输出类型是否为 np.float32 类型
assert_type(np.ptp(AR_f4), np.float32)
# 断言：检查 np.ptp 函数对数组 AR_b 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.ptp(AR_b, axis=0), Any)
# 断言：检查 np.ptp 函数对数组 AR_f4 指定轴向（axis=0）进行操作后的输出类型是否为 Any（任意类型）
assert_type(np.ptp(AR_f4, axis=0), Any)
#
# 确保 np.prod 返回的结果类型为 np.floating[Any]
assert_type(np.prod(AR_f4), np.floating[Any])
# 确保 np.prod 返回的结果类型为 np.complexfloating[Any, Any]
assert_type(np.prod(AR_c16), np.complexfloating[Any, Any])
# 确保 np.prod 返回的结果类型为 Any
assert_type(np.prod(AR_O), Any)
# 确保 np.prod 返回的结果类型为 Any
assert_type(np.prod(AR_f4, axis=0), Any)
# 确保 np.prod 返回的结果类型为 Any，保持维度
assert_type(np.prod(AR_f4, keepdims=True), Any)
# 确保 np.prod 返回的结果类型为 np.float64
assert_type(np.prod(AR_f4, dtype=np.float64), np.float64)
# 确保 np.prod 返回的结果类型为 Any，使用 float 作为数据类型
assert_type(np.prod(AR_f4, dtype=float), Any)
# 确保 np.prod 返回的结果类型为 NDArraySubclass，并将结果输出到 AR_subclass
assert_type(np.prod(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.int_]
assert_type(np.cumprod(AR_b), npt.NDArray[np.int_])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.uint64]
assert_type(np.cumprod(AR_u8), npt.NDArray[np.uint64])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.int64]
assert_type(np.cumprod(AR_i8), npt.NDArray[np.int64])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.floating[Any]]
assert_type(np.cumprod(AR_f4), npt.NDArray[np.floating[Any]])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(np.cumprod(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.object_]
assert_type(np.cumprod(AR_O), npt.NDArray[np.object_])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.floating[Any]]，指定轴向
assert_type(np.cumprod(AR_f4, axis=0), npt.NDArray[np.floating[Any]])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[np.float64]，指定数据类型
assert_type(np.cumprod(AR_f4, dtype=np.float64), npt.NDArray[np.float64])
# 确保 np.cumprod 返回的结果类型为 npt.NDArray[Any]，使用 float 作为数据类型
assert_type(np.cumprod(AR_f4, dtype=float), npt.NDArray[Any])
# 确保 np.cumprod 返回的结果类型为 NDArraySubclass，并将结果输出到 AR_subclass
assert_type(np.cumprod(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保 np.ndim 返回的结果类型为 int
assert_type(np.ndim(b), int)
assert_type(np.ndim(f4), int)
assert_type(np.ndim(f), int)
assert_type(np.ndim(AR_b), int)
assert_type(np.ndim(AR_f4), int)

# 确保 np.size 返回的结果类型为 int
assert_type(np.size(b), int)
assert_type(np.size(f4), int)
assert_type(np.size(f), int)
assert_type(np.size(AR_b), int)
assert_type(np.size(AR_f4), int)

# 确保 np.around 返回的结果类型为 np.float16
assert_type(np.around(b), np.float16)
# 确保 np.around 返回的结果类型为 Any
assert_type(np.around(f), Any)
# 确保 np.around 返回的结果类型为 np.int64
assert_type(np.around(i8), np.int64)
# 确保 np.around 返回的结果类型为 np.float32
assert_type(np.around(f4), np.float32)
# 确保 np.around 返回的结果类型为 npt.NDArray[np.float16]
assert_type(np.around(AR_b), npt.NDArray[np.float16])
# 确保 np.around 返回的结果类型为 npt.NDArray[np.int64]
assert_type(np.around(AR_i8), npt.NDArray[np.int64])
# 确保 np.around 返回的结果类型为 npt.NDArray[np.float32]
assert_type(np.around(AR_f4), npt.NDArray[np.float32])
# 确保 np.around 返回的结果类型为 npt.NDArray[Any]
assert_type(np.around([1.5]), npt.NDArray[Any])
# 确保 np.around 返回的结果类型为 NDArraySubclass，并将结果输出到 AR_subclass
assert_type(np.around(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保 np.mean 返回的结果类型为 np.floating[Any]
assert_type(np.mean(AR_b), np.floating[Any])
assert_type(np.mean(AR_i8), np.floating[Any])
assert_type(np.mean(AR_f4), np.floating[Any])
assert_type(np.mean(AR_c16), np.complexfloating[Any, Any])
# 确保 np.mean 返回的结果类型为 Any
assert_type(np.mean(AR_O), Any)
# 确保 np.mean 返回的结果类型为 Any，指定轴向
assert_type(np.mean(AR_f4, axis=0), Any)
# 确保 np.mean 返回的结果类型为 Any，保持维度
assert_type(np.mean(AR_f4, keepdims=True), Any)
# 确保 np.mean 返回的结果类型为 Any，使用 float 作为数据类型
assert_type(np.mean(AR_f4, dtype=float), Any)
# 确保 np.mean 返回的结果类型为 np.float64
assert_type(np.mean(AR_f4, dtype=np.float64), np.float64)
# 确保 np.mean 返回的结果类型为 NDArraySubclass，并将结果输出到 AR_subclass
assert_type(np.mean(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保 np.std 返回的结果类型为 np.floating[Any]
assert_type(np.std(AR_b), np.floating[Any])
assert_type(np.std(AR_i8), np.floating[Any])
assert_type(np.std(AR_f4), np.floating[Any])
assert_type(np.std(AR_c16), np.floating[Any])
# 确保 np.std 返回的结果类型为 Any
assert_type(np.std(AR_O), Any)
# 确保 np.std 返回的结果类型为 Any，指定轴向
assert_type(np.std(AR_f4, axis=0), Any)
# 确保 np.std 返回的结果类型为 Any，保持维度
assert_type(np.std(AR_f4, keepdims=True), Any)
# 确保 np.std 返回的结果类型为 Any，使用 float 作为数据类型
assert_type(np.std(AR_f4, dtype=float), Any)
# 确保 np.std 返回的结果类型为 np.float64
assert_type(np.std(AR_f4, dtype=np.float64), np.float64)
# 确保 np.std 返回的结果类型为 NDArraySubclass，并将结果输出到 AR_subclass
assert_type(np.std(AR_f4, out=AR_subclass), NDArraySubclass)

# 确保 np.var 返回的结果类型为 np.floating[Any]
assert_type(np.var(AR_b), np.floating[Any])
assert_type(np.var(AR_i8), np.floating[Any])
assert_type(np.var(AR_f4), np.floating[Any])
assert_type(np.var(AR_c16), np.floating[Any])
# 确保 np.var 返回的结果类型为 Any
assert_type(np.var(AR_O), Any)
# 确保 np.var 返回的结果类型为 Any，指定轴向
assert_type(np.var(AR_f4, axis=0), Any)
# 确保 np.var 返回的结果类型为 Any，保持维度
assert_type(np.var(AR_f4, keepdims=True), Any)
# 使用 np.var 函数计算数组 AR_f4 的方差，并检查结果类型是否为 Any
assert_type(np.var(AR_f4, dtype=float), Any)

# 使用 np.var 函数计算数组 AR_f4 的方差，并检查结果类型是否为 np.float64
assert_type(np.var(AR_f4, dtype=np.float64), np.float64)

# 使用 np.var 函数计算数组 AR_f4 的方差，并将结果存储在 AR_subclass 中，
# 然后检查 AR_subclass 的类型是否为 NDArraySubclass
assert_type(np.var(AR_f4, out=AR_subclass), NDArraySubclass)
```