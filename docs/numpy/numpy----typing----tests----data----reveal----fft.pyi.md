# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\fft.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型定义模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果 Python 版本大于等于 3.11，则导入 assert_type 函数
else:
    from typing_extensions import assert_type  # 否则导入 typing_extensions 模块中的 assert_type 函数

AR_f8: npt.NDArray[np.float64]  # 定义 AR_f8 变量的类型注解为 np.float64 类型的 NumPy 数组
AR_c16: npt.NDArray[np.complex128]  # 定义 AR_c16 变量的类型注解为 np.complex128 类型的 NumPy 数组
AR_LIKE_f8: list[float]  # 定义 AR_LIKE_f8 变量的类型注解为 float 类型的列表

assert_type(np.fft.fftshift(AR_f8), npt.NDArray[np.float64])  # 断言 np.fft.fftshift 函数返回的类型符合 npt.NDArray[np.float64]
assert_type(np.fft.fftshift(AR_LIKE_f8, axes=0), npt.NDArray[Any])  # 断言 np.fft.fftshift 函数返回的类型符合 npt.NDArray[Any]

assert_type(np.fft.ifftshift(AR_f8), npt.NDArray[np.float64])  # 断言 np.fft.ifftshift 函数返回的类型符合 npt.NDArray[np.float64]
assert_type(np.fft.ifftshift(AR_LIKE_f8, axes=0), npt.NDArray[Any])  # 断言 np.fft.ifftshift 函数返回的类型符合 npt.NDArray[Any]

assert_type(np.fft.fftfreq(5, AR_f8), npt.NDArray[np.floating[Any]])  # 断言 np.fft.fftfreq 函数返回的类型符合 npt.NDArray[np.floating[Any]]
assert_type(np.fft.fftfreq(np.int64(), AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.fft.fftfreq 函数返回的类型符合 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.fft.fftfreq(5, AR_f8), npt.NDArray[np.floating[Any]])  # 断言 np.fft.fftfreq 函数返回的类型符合 npt.NDArray[np.floating[Any]]
assert_type(np.fft.fftfreq(np.int64(), AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 断言 np.fft.fftfreq 函数返回的类型符合 npt.NDArray[np.complexfloating[Any, Any]]

assert_type(np.fft.fft(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.fft 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.ifft(AR_f8, axis=1), npt.NDArray[np.complex128])  # 断言 np.fft.ifft 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.rfft(AR_f8, n=None), npt.NDArray[np.complex128])  # 断言 np.fft.rfft 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.irfft(AR_f8, norm="ortho"), npt.NDArray[np.float64])  # 断言 np.fft.irfft 函数返回的类型符合 npt.NDArray[np.float64]
assert_type(np.fft.hfft(AR_f8, n=2), npt.NDArray[np.float64])  # 断言 np.fft.hfft 函数返回的类型符合 npt.NDArray[np.float64]
assert_type(np.fft.ihfft(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.ihfft 函数返回的类型符合 npt.NDArray[np.complex128]

assert_type(np.fft.fftn(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.fftn 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.ifftn(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.ifftn 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.rfftn(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.rfftn 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.irfftn(AR_f8), npt.NDArray[np.float64])  # 断言 np.fft.irfftn 函数返回的类型符合 npt.NDArray[np.float64]

assert_type(np.fft.rfft2(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.rfft2 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.ifft2(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.ifft2 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.fft2(AR_f8), npt.NDArray[np.complex128])  # 断言 np.fft.fft2 函数返回的类型符合 npt.NDArray[np.complex128]
assert_type(np.fft.irfft2(AR_f8), npt.NDArray[np.float64])  # 断言 np.fft.irfft2 函数返回的类型符合 npt.NDArray[np.float64]
```