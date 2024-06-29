# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\lib_function_base.pyi`

```
import sys
from typing import Any  # 导入 Any 类型
from collections.abc import Callable  # 导入 Callable 抽象基类

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 类型提示

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果 Python 版本 >= 3.11，导入 assert_type
else:
    from typing_extensions import assert_type  # 否则，从 typing_extensions 导入 assert_type

vectorized_func: np.vectorize  # 声明 vectorized_func 为 np.vectorize 类型

f8: np.float64  # 声明 f8 为 np.float64 类型
AR_LIKE_f8: list[float]  # 声明 AR_LIKE_f8 为 float 类型的列表

AR_i8: npt.NDArray[np.int64]  # 声明 AR_i8 为 np.int64 类型的 NumPy 数组
AR_f8: npt.NDArray[np.float64]  # 声明 AR_f8 为 np.float64 类型的 NumPy 数组
AR_c16: npt.NDArray[np.complex128]  # 声明 AR_c16 为 np.complex128 类型的 NumPy 数组
AR_m: npt.NDArray[np.timedelta64]  # 声明 AR_m 为 np.timedelta64 类型的 NumPy 数组
AR_M: npt.NDArray[np.datetime64]  # 声明 AR_M 为 np.datetime64 类型的 NumPy 数组
AR_O: npt.NDArray[np.object_]  # 声明 AR_O 为 np.object_ 类型的 NumPy 数组
AR_b: npt.NDArray[np.bool]  # 声明 AR_b 为 np.bool 类型的 NumPy 数组
AR_U: npt.NDArray[np.str_]  # 声明 AR_U 为 np.str_ 类型的 NumPy 数组
CHAR_AR_U: np.char.chararray[Any, np.dtype[np.str_]]  # 声明 CHAR_AR_U 为 np.char.chararray 类型的数组，元素类型为 np.str_

def func(*args: Any, **kwargs: Any) -> Any: ...  # 定义一个 func 函数，接受任意参数，返回任意类型的值

assert_type(vectorized_func.pyfunc, Callable[..., Any])  # 断言 vectorized_func.pyfunc 的类型为 Callable[..., Any]
assert_type(vectorized_func.cache, bool)  # 断言 vectorized_func.cache 的类型为 bool
assert_type(vectorized_func.signature, None | str)  # 断言 vectorized_func.signature 的类型为 None 或 str
assert_type(vectorized_func.otypes, None | str)  # 断言 vectorized_func.otypes 的类型为 None 或 str
assert_type(vectorized_func.excluded, set[int | str])  # 断言 vectorized_func.excluded 的类型为 set[int | str]
assert_type(vectorized_func.__doc__, None | str)  # 断言 vectorized_func.__doc__ 的类型为 None 或 str
assert_type(vectorized_func([1]), Any)  # 断言 vectorized_func([1]) 的类型为 Any
assert_type(np.vectorize(int), np.vectorize)  # 断言 np.vectorize(int) 的类型为 np.vectorize

assert_type(
    np.vectorize(int, otypes="i", doc="doc", excluded=(), cache=True, signature=None),
    np.vectorize,
)  # 断言调用 np.vectorize 函数的返回类型为 np.vectorize

assert_type(np.rot90(AR_f8, k=2), npt.NDArray[np.float64])  # 断言 np.rot90(AR_f8, k=2) 的返回类型为 npt.NDArray[np.float64]
assert_type(np.rot90(AR_LIKE_f8, axes=(0, 1)), npt.NDArray[Any])  # 断言 np.rot90(AR_LIKE_f8, axes=(0, 1)) 的返回类型为 npt.NDArray[Any]

assert_type(np.flip(f8), np.float64)  # 断言 np.flip(f8) 的返回类型为 np.float64
assert_type(np.flip(1.0), Any)  # 断言 np.flip(1.0) 的返回类型为 Any
assert_type(np.flip(AR_f8, axis=(0, 1)), npt.NDArray[np.float64])  # 断言 np.flip(AR_f8, axis=(0, 1)) 的返回类型为 npt.NDArray[np.float64]
assert_type(np.flip(AR_LIKE_f8, axis=0), npt.NDArray[Any])  # 断言 np.flip(AR_LIKE_f8, axis=0) 的返回类型为 npt.NDArray[Any]

assert_type(np.iterable(1), bool)  # 断言 np.iterable(1) 的返回类型为 bool
assert_type(np.iterable([1]), bool)  # 断言 np.iterable([1]) 的返回类型为 bool

assert_type(np.average(AR_f8), np.floating[Any])  # 断言 np.average(AR_f8) 的返回类型为 np.floating[Any]
assert_type(np.average(AR_f8, weights=AR_c16), np.complexfloating[Any, Any])  # 断言 np.average(AR_f8, weights=AR_c16) 的返回类型为 np.complexfloating[Any, Any]
assert_type(np.average(AR_O), Any)  # 断言 np.average(AR_O) 的返回类型为 Any
assert_type(np.average(AR_f8, returned=True), tuple[np.floating[Any], np.floating[Any]])  # 断言 np.average(AR_f8, returned=True) 的返回类型为 tuple[np.floating[Any], np.floating[Any]]
assert_type(np.average(AR_f8, weights=AR_c16, returned=True), tuple[np.complexfloating[Any, Any], np.complexfloating[Any, Any]])  # 断言 np.average(AR_f8, weights=AR_c16, returned=True) 的返回类型为 tuple[np.complexfloating[Any, Any], np.complexfloating[Any, Any]]
assert_type(np.average(AR_O, returned=True), tuple[Any, Any])  # 断言 np.average(AR_O, returned=True) 的返回类型为 tuple[Any, Any]
assert_type(np.average(AR_f8, axis=0), Any)  # 断言 np.average(AR_f8, axis=0) 的返回类型为 Any
assert_type(np.average(AR_f8, axis=0, returned=True), tuple[Any, Any])  # 断言 np.average(AR_f8, axis=0, returned=True) 的返回类型为 tuple[Any, Any]

assert_type(np.asarray_chkfinite(AR_f8), npt.NDArray[np.float64])  # 断言 np.asarray_chkfinite(AR_f8) 的返回类型为 npt.NDArray[np.float64]
assert_type(np.asarray_chkfinite(AR_LIKE_f8), npt.NDArray[Any])  # 断言 np.asarray_chkfinite(AR_LIKE_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.asarray_chkfinite(AR_f8, dtype=np.float64), npt.NDArray[np.float64])  # 断言 np.asarray_chkfinite(AR_f8, dtype=np.float64) 的返回类型为 npt.NDArray[np.float64]
assert_type(np.asarray_chkfinite(AR_f8, dtype=float), npt.NDArray[Any])  # 断言 np.asarray_chkfinite(AR_f8, dtype=float) 的返回类型为 npt.NDArray[Any]

assert_type(np.piecewise(AR_f8, AR_b, [func]), npt.NDArray[np.float64])  # 断言 np.piecewise(AR_f8, AR_b, [func]) 的返回类型为 npt.NDArray[np.float64]
assert_type(np.piecewise(AR_LIKE_f8, AR_b, [func]), npt.NDArray[Any])  # 断言 np.piecewise(AR_LIKE_f8, AR_b, [func]) 的返回类型为 npt.NDArray[Any]

assert_type(np.select([AR_f8], [AR_f8]), npt.NDArray[Any])  # 断言 np.select([AR_f8], [AR_f8]) 的返回类型为 npt.NDArray[Any]

assert_type(np.copy(AR_LIKE_f8), npt.NDArray[Any])  # 断言 np.copy(AR_LIKE_f8) 的返回类型为 npt.NDArray[Any]
assert_type(np.copy(AR_U), npt.NDArray[np.str_])  # 断言 np.copy(AR_U) 的返回类型为 npt.NDArray[np.str_]
assert_type(np.copy(CHAR_AR_U), np.ndarray[Any, Any])  # 断言 np.copy(CHAR_AR_U) 的返回类型为 np.ndarray[Any, Any]
assert_type(np.copy(CHAR_AR_U, "K", subok=True), np.char.chararray[Any, np.dtype[np.str_
# 检查 np.gradient 函数返回值的类型是否为 Any 类型
assert_type(np.gradient(AR_LIKE_f8, edge_order=2), Any)

# 检查 np.diff 函数返回值的类型是否为 str 类型
assert_type(np.diff("bob", n=0), str)
# 检查 np.diff 函数返回值的类型是否为 npt.NDArray[Any] 类型
assert_type(np.diff(AR_f8, axis=0), npt.NDArray[Any])
# 检查 np.diff 函数返回值的类型是否为 npt.NDArray[Any] 类型
assert_type(np.diff(AR_LIKE_f8, prepend=1.5), npt.NDArray[Any])

# 检查 np.angle 函数返回值的类型是否为 np.floating[Any] 类型
assert_type(np.angle(f8), np.floating[Any])
# 检查 np.angle 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.angle(AR_f8), npt.NDArray[np.floating[Any]])
# 检查 np.angle 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.angle(AR_c16, deg=True), npt.NDArray[np.floating[Any]])
# 检查 np.angle 函数返回值的类型是否为 npt.NDArray[np.object_] 类型
assert_type(np.angle(AR_O), npt.NDArray[np.object_])

# 检查 np.unwrap 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.unwrap(AR_f8), npt.NDArray[np.floating[Any]])
# 检查 np.unwrap 函数返回值的类型是否为 npt.NDArray[np.object_] 类型
assert_type(np.unwrap(AR_O), npt.NDArray[np.object_])

# 检查 np.sort_complex 函数返回值的类型是否为 npt.NDArray[np.complexfloating[Any, Any]] 类型
assert_type(np.sort_complex(AR_f8), npt.NDArray[np.complexfloating[Any, Any]])

# 检查 np.trim_zeros 函数返回值的类型是否为 npt.NDArray[np.float64] 类型
assert_type(np.trim_zeros(AR_f8), npt.NDArray[np.float64])
# 检查 np.trim_zeros 函数返回值的类型是否为 list[float] 类型
assert_type(np.trim_zeros(AR_LIKE_f8), list[float])

# 检查 np.extract 函数返回值的类型是否为 npt.NDArray[np.float64] 类型
assert_type(np.extract(AR_i8, AR_f8), npt.NDArray[np.float64])
# 检查 np.extract 函数返回值的类型是否为 npt.NDArray[Any] 类型
assert_type(np.extract(AR_i8, AR_LIKE_f8), npt.NDArray[Any])

# 检查 np.place 函数返回值的类型是否为 None 类型
assert_type(np.place(AR_f8, mask=AR_i8, vals=5.0), None)

# 检查 np.cov 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.cov(AR_f8, bias=True), npt.NDArray[np.floating[Any]])
# 检查 np.cov 函数返回值的类型是否为 npt.NDArray[np.complexfloating[Any, Any]] 类型
assert_type(np.cov(AR_f8, AR_c16, ddof=1), npt.NDArray[np.complexfloating[Any, Any]])
# 检查 np.cov 函数返回值的类型是否为 npt.NDArray[np.float32] 类型
assert_type(np.cov(AR_f8, aweights=AR_f8, dtype=np.float32), npt.NDArray[np.float32])
# 检查 np.cov 函数返回值的类型是否为 npt.NDArray[Any] 类型
assert_type(np.cov(AR_f8, fweights=AR_f8, dtype=float), npt.NDArray[Any])

# 检查 np.corrcoef 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.corrcoef(AR_f8, rowvar=True), npt.NDArray[np.floating[Any]])
# 检查 np.corrcoef 函数返回值的类型是否为 npt.NDArray[np.complexfloating[Any, Any]] 类型
assert_type(np.corrcoef(AR_f8, AR_c16), npt.NDArray[np.complexfloating[Any, Any]])
# 检查 np.corrcoef 函数返回值的类型是否为 npt.NDArray[np.float32] 类型
assert_type(np.corrcoef(AR_f8, dtype=np.float32), npt.NDArray[np.float32])
# 检查 np.corrcoef 函数返回值的类型是否为 npt.NDArray[Any] 类型
assert_type(np.corrcoef(AR_f8, dtype=float), npt.NDArray[Any])

# 检查 np.blackman 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.blackman(5), npt.NDArray[np.floating[Any]])
# 检查 np.bartlett 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.bartlett(6), npt.NDArray[np.floating[Any]])
# 检查 np.hanning 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.hanning(4.5), npt.NDArray[np.floating[Any]])
# 检查 np.hamming 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.hamming(0), npt.NDArray[np.floating[Any]])
# 检查 np.i0 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.i0(AR_i8), npt.NDArray[np.floating[Any]])
# 检查 np.kaiser 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.kaiser(4, 5.9), npt.NDArray[np.floating[Any]])

# 检查 np.sinc 函数返回值的类型是否为 np.floating[Any] 类型
assert_type(np.sinc(1.0), np.floating[Any])
# 检查 np.sinc 函数返回值的类型是否为 np.complexfloating[Any, Any] 类型
assert_type(np.sinc(1j), np.complexfloating[Any, Any])
# 检查 np.sinc 函数返回值的类型是否为 npt.NDArray[np.floating[Any]] 类型
assert_type(np.sinc(AR_f8), npt.NDArray[np.floating[Any]])
# 检查 np.sinc 函数返回值的类型是否为 npt.NDArray[np.complexfloating[Any, Any]] 类型
assert_type(np.sinc(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])

# 检查 np.median 函数返回值的类型是否为 np.floating[Any] 类型
assert_type(np.median(AR_f8, keepdims=False), np.floating[Any])
# 检查 np.median 函数返回值的类型是否为 np.complexfloating[Any, Any] 类型
assert_type(np.median(AR_c16, overwrite_input=True), np.complexfloating[Any, Any])
# 检查 np.median 函数返回值的类型是否为 np.timedelta64 类型
assert_type(np.median(AR_m), np.timedelta64)
# 检查 np.median 函数返回值的类型是否为 Any 类型
assert_type(np.median(AR_O), Any)
# 检查 np.median 函数返回值的类型是否为 Any 类型
assert_type(np.median(AR_f8, keepdims=True), Any)
# 检查 np.median 函数返回值的类型是否为 Any 类型
assert_type(np.median(AR_c16, axis=0), Any)
# 检查 np.median 函数返回值的类型是否为 npt.NDArray[np.complex128] 类型
assert_type(np.median(AR_LIKE_f8, out=AR_c16), npt.NDArray[np.complex128])

# 检查 np.percentile 函数返回值的类型是否为 np.floating[Any] 类型
assert_type(np.percentile(AR_f8, 50), np.floating[Any])
# 检查 np.percentile 函数返回值的类型是否为 np.complexfloating[Any, Any] 类型
assert_type(np.percentile(AR_c16, 50), np.complexfloating[Any, Any])
# 检查 np.percentile 函数返回值的类型是否为 np.timedelta64 类型
assert_type(np.percentile(AR_m, 50), np.timedelta64
# 确保 np.percentile 返回的结果是一个包含 np.timedelta64 元素的 NumPy 数组
assert_type(np.percentile(AR_m, [50]), npt.NDArray[np.timedelta64])

# 确保 np.percentile 返回的结果是一个包含 np.datetime64 元素的 NumPy 数组，使用 nearest 方法进行近似
assert_type(np.percentile(AR_M, [50], method="nearest"), npt.NDArray[np.datetime64])

# 确保 np.percentile 返回的结果是一个包含 np.object_ 元素的 NumPy 数组
assert_type(np.percentile(AR_O, [50]), npt.NDArray[np.object_])

# 确保 np.percentile 返回的结果是一个任意类型的 NumPy 数组，keepdims=True 保持维度信息
assert_type(np.percentile(AR_f8, [50], keepdims=True), Any)

# 确保 np.percentile 返回的结果是一个任意类型的 NumPy 数组，指定 axis=1 进行计算
assert_type(np.percentile(AR_f8, [50], axis=[1]), Any)

# 确保 np.percentile 返回的结果是一个包含 np.complex128 元素的 NumPy 数组，将结果输出到 AR_c16 中
assert_type(np.percentile(AR_f8, [50], out=AR_c16), npt.NDArray[np.complex128])

# 确保 np.quantile 返回的结果是一个包含任意类型的浮点数的 NumPy 数组，quantile 指定为 0.5
assert_type(np.quantile(AR_f8, 0.5), np.floating[Any])

# 确保 np.quantile 返回的结果是一个包含 np.complexfloating 元素的 NumPy 数组，quantile 指定为 0.5
assert_type(np.quantile(AR_c16, 0.5), np.complexfloating[Any, Any])

# 确保 np.quantile 返回的结果是一个包含 np.timedelta64 元素的 NumPy 数组，quantile 指定为 0.5
assert_type(np.quantile(AR_m, 0.5), np.timedelta64)

# 确保 np.quantile 返回的结果是一个包含 np.datetime64 元素的 NumPy 数组，quantile 指定为 0.5，并覆盖输入数组
assert_type(np.quantile(AR_M, 0.5, overwrite_input=True), np.datetime64)

# 确保 np.quantile 返回的结果是一个包含任意类型的元素的 NumPy 数组，quantile 指定为 0.5
assert_type(np.quantile(AR_O, 0.5), Any)

# 确保 np.quantile 返回的结果是一个包含 np.floating[Any] 元素的 NumPy 数组，quantile 指定为 [0.5]
assert_type(np.quantile(AR_f8, [0.5]), npt.NDArray[np.floating[Any]])

# 确保 np.quantile 返回的结果是一个包含 np.complexfloating[Any, Any] 元素的 NumPy 数组，quantile 指定为 [0.5]
assert_type(np.quantile(AR_c16, [0.5]), npt.NDArray[np.complexfloating[Any, Any]])

# 确保 np.quantile 返回的结果是一个包含 np.timedelta64 元素的 NumPy 数组，quantile 指定为 [0.5]
assert_type(np.quantile(AR_m, [0.5]), npt.NDArray[np.timedelta64])

# 确保 np.quantile 返回的结果是一个包含 np.datetime64 元素的 NumPy 数组，quantile 指定为 [0.5]，使用 nearest 方法
assert_type(np.quantile(AR_M, [0.5], method="nearest"), npt.NDArray[np.datetime64])

# 确保 np.quantile 返回的结果是一个包含 np.object_ 元素的 NumPy 数组，quantile 指定为 [0.5]
assert_type(np.quantile(AR_O, [0.5]), npt.NDArray[np.object_])

# 确保 np.quantile 返回的结果是一个任意类型的 NumPy 数组，keepdims=True 保持维度信息
assert_type(np.quantile(AR_f8, [0.5], keepdims=True), Any)

# 确保 np.quantile 返回的结果是一个任意类型的 NumPy 数组，指定 axis=[1] 进行计算
assert_type(np.quantile(AR_f8, [0.5], axis=[1]), Any)

# 确保 np.quantile 返回的结果是一个包含 np.complex128 元素的 NumPy 数组，将结果输出到 AR_c16 中
assert_type(np.quantile(AR_f8, [0.5], out=AR_c16), npt.NDArray[np.complex128])

# 确保 np.meshgrid 返回的结果是一个元组，包含任意类型的 NumPy 数组
assert_type(np.meshgrid(AR_f8, AR_i8, copy=False), tuple[npt.NDArray[Any], ...])

# 确保 np.meshgrid 返回的结果是一个元组，包含任意类型的 NumPy 数组，使用 indexing="ij"
assert_type(np.meshgrid(AR_f8, AR_i8, AR_c16, indexing="ij"), tuple[npt.NDArray[Any], ...])

# 确保 np.delete 返回的结果是一个包含 np.float64 元素的 NumPy 数组，删除 np.s_[:5] 所指定的切片
assert_type(np.delete(AR_f8, np.s_[:5]), npt.NDArray[np.float64])

# 确保 np.delete 返回的结果是一个包含任意类型的元素的 NumPy 数组，删除 axis=0 上指定的索引 [0, 4, 9]
assert_type(np.delete(AR_LIKE_f8, [0, 4, 9], axis=0), npt.NDArray[Any])

# 确保 np.insert 返回的结果是一个包含 np.float64 元素的 NumPy 数组，在 np.s_[:5] 所指定的位置插入 5
assert_type(np.insert(AR_f8, np.s_[:5], 5), npt.NDArray[np.float64])

# 确保 np.insert 返回的结果是一个包含任意类型的元素的 NumPy 数组，在 axis=0 上指定的索引 [0, 4, 9] 处插入 [0.5, 9.2, 7]
assert_type(np.insert(AR_LIKE_f8, [0, 4, 9], [0.5, 9.2, 7], axis=0), npt.NDArray[Any])

# 确保 np.append 返回的结果是一个包含任意类型的元素的 NumPy 数组，在末尾追加 5
assert_type(np.append(AR_f8, 5), npt.NDArray[Any])

# 确保 np.append 返回的结果是一个包含任意类型的元素的 NumPy 数组，在 axis=0 上追加复数单位 1j
assert_type(np.append(AR_LIKE_f8, 1j, axis=0), npt.NDArray[Any])

# 确保 np.digitize 返回的结果是一个包含 np.intp 元素的 NumPy 数组，将 4.5 插入到 bins=[1] 中
assert_type(np.digitize(4.5, [1]), np.intp)

# 确保 np.digitize 返回的结果是一个包含 np.intp 元素的 NumPy 数组，将 AR_f8 插入到 bins=[1, 2, 3] 中
assert_type(np.digitize(AR_f8, [1, 2, 3]), npt.NDArray[np.intp])
```