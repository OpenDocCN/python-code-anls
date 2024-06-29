# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\arithmetic.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型，表示可以接受任何类型的参数

import numpy as np  # 导入 NumPy 库，并命名为 np
import numpy.typing as npt  # 导入 NumPy 的类型注解模块
from numpy._typing import _32Bit, _64Bit, _128Bit  # 导入 NumPy 中的特定位数类型

if sys.version_info >= (3, 11):
    from typing import assert_type  # 根据 Python 版本导入不同的 assert_type 函数
else:
    from typing_extensions import assert_type  # 使用 typing_extensions 模块中的 assert_type 函数

# 无法直接导入 `np.float128`，因为它不在所有平台上都可用
f16: np.floating[_128Bit]  # 声明一个名为 f16 的浮点数类型变量，具体类型为 np.floating[_128Bit]

c16 = np.complex128()  # 创建一个复数类型为 np.complex128 的变量 c16
f8 = np.float64()  # 创建一个浮点数类型为 np.float64 的变量 f8
i8 = np.int64()  # 创建一个整数类型为 np.int64 的变量 i8
u8 = np.uint64()  # 创建一个无符号整数类型为 np.uint64 的变量 u8

c8 = np.complex64()  # 创建一个复数类型为 np.complex64 的变量 c8
f4 = np.float32()  # 创建一个浮点数类型为 np.float32 的变量 f4
i4 = np.int32()  # 创建一个整数类型为 np.int32 的变量 i4
u4 = np.uint32()  # 创建一个无符号整数类型为 np.uint32 的变量 u4

dt = np.datetime64(0, "D")  # 创建一个日期时间类型为 np.datetime64 的变量 dt，初始化为 0 天
td = np.timedelta64(0, "D")  # 创建一个时间间隔类型为 np.timedelta64 的变量 td，初始化为 0 天

b_ = np.bool()  # 创建一个布尔类型为 np.bool 的变量 b_

b = bool()  # 创建一个 Python 内置的布尔类型变量 b
c = complex()  # 创建一个 Python 内置的复数类型变量 c
f = float()  # 创建一个 Python 内置的浮点数类型变量 f
i = int()  # 创建一个 Python 内置的整数类型变量 i

AR_b: npt.NDArray[np.bool]  # 声明一个 NumPy 数组类型变量 AR_b，元素类型为 np.bool
AR_u: npt.NDArray[np.uint32]  # 声明一个 NumPy 数组类型变量 AR_u，元素类型为 np.uint32
AR_i: npt.NDArray[np.int64]  # 声明一个 NumPy 数组类型变量 AR_i，元素类型为 np.int64
AR_f: npt.NDArray[np.float64]  # 声明一个 NumPy 数组类型变量 AR_f，元素类型为 np.float64
AR_c: npt.NDArray[np.complex128]  # 声明一个 NumPy 数组类型变量 AR_c，元素类型为 np.complex128
AR_m: npt.NDArray[np.timedelta64]  # 声明一个 NumPy 数组类型变量 AR_m，元素类型为 np.timedelta64
AR_M: npt.NDArray[np.datetime64]  # 声明一个 NumPy 数组类型变量 AR_M，元素类型为 np.datetime64
AR_O: npt.NDArray[np.object_]  # 声明一个 NumPy 数组类型变量 AR_O，元素类型为 np.object_
AR_number: npt.NDArray[np.number[Any]]  # 声明一个 NumPy 数组类型变量 AR_number，元素类型为 np.number[Any]

AR_LIKE_b: list[bool]  # 声明一个 Python 列表类型变量 AR_LIKE_b，元素类型为 bool
AR_LIKE_u: list[np.uint32]  # 声明一个 Python 列表类型变量 AR_LIKE_u，元素类型为 np.uint32
AR_LIKE_i: list[int]  # 声明一个 Python 列表类型变量 AR_LIKE_i，元素类型为 int
AR_LIKE_f: list[float]  # 声明一个 Python 列表类型变量 AR_LIKE_f，元素类型为 float
AR_LIKE_c: list[complex]  # 声明一个 Python 列表类型变量 AR_LIKE_c，元素类型为 complex
AR_LIKE_m: list[np.timedelta64]  # 声明一个 Python 列表类型变量 AR_LIKE_m，元素类型为 np.timedelta64
AR_LIKE_M: list[np.datetime64]  # 声明一个 Python 列表类型变量 AR_LIKE_M，元素类型为 np.datetime64
AR_LIKE_O: list[np.object_]  # 声明一个 Python 列表类型变量 AR_LIKE_O，元素类型为 np.object_

# 数组相减

assert_type(AR_number - AR_number, npt.NDArray[np.number[Any]])  # 断言 AR_number 和 AR_number 相减后的类型为 npt.NDArray[np.number[Any]]

assert_type(AR_b - AR_LIKE_u, npt.NDArray[np.unsignedinteger[Any]])  # 断言 AR_b 和 AR_LIKE_u 相减后的类型为 npt.NDArray[np.unsignedinteger[Any]]
assert_type(AR_b - AR_LIKE_i, npt.NDArray[np.signedinteger[Any]])  # 断言 AR_b 和 AR_LIKE_i 相减后的类型为 npt.NDArray[np.signedinteger[Any]]
assert_type(AR_b - AR_LIKE_f, npt.NDArray[np.floating[Any]])  # 断言 AR_b 和 AR_LIKE_f 相减后的类型为 npt.NDArray[np.floating[Any]]
assert_type(AR_b - AR_LIKE_c, npt.NDArray[np.complexfloating[Any, Any]])  # 断言 AR_b 和 AR_LIKE_c 相减后的类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(AR_b - AR_LIKE_m, npt.NDArray[np.timedelta64])  # 断言 AR_b 和 AR_LIKE_m 相减后的类型为 npt.NDArray[np.timedelta64]
assert_type(AR_b - AR_LIKE_O, Any)  # 断言 AR_b 和 AR_LIKE_O 相减后的类型为 Any

assert_type(AR_LIKE_u - AR_b, npt.NDArray[np.unsignedinteger[Any]])  # 断言 AR_LIKE_u 和 AR_b 相减后的类型为 npt.NDArray[np.unsignedinteger[Any]]
assert_type(AR_LIKE_i - AR_b, npt.NDArray[np.signedinteger[Any]])  # 断言 AR_LIKE_i 和 AR_b 相减后的类型为 npt.NDArray[np.signedinteger[Any]]
assert_type(AR_LIKE_f - AR_b, npt.NDArray[np.floating[Any]])  # 断言 AR_LIKE_f 和 AR_b 相减后的类型为 npt.NDArray[np.floating[Any]]
assert_type(AR_LIKE_c - AR_b, npt.NDArray[np.complexfloating[Any, Any]])  # 断言 AR_LIKE_c 和 AR_b 相减后的类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(AR_LIKE_m - AR_b, npt.NDArray[np.timedelta64])  # 断言 AR_LIKE_m 和 AR_b 相减后的类型为 npt.NDArray[np.timedelta64]
assert_type(AR_LIKE_M - AR_b, npt.NDArray[np.datetime64])  # 断言 AR_LIKE_M 和 AR_b 相减后的类型为 npt.NDArray[np.datetime64]
assert_type(AR_LIKE_O - AR_b, Any)  # 断言 AR_LIKE_O 和 AR_b 相减后的类型为 Any

assert_type(AR_u - AR_LIKE_b, npt.NDArray[np.unsignedinteger[Any]])  # 断言 AR_u 和 AR_LIKE_b 相减后的类型为 npt.NDArray[np.unsignedinteger[Any]]
assert_type(AR_u - AR_LIKE_u, npt.NDArray[np.unsignedinteger[Any]])  # 断言 AR_u 和 AR_LIKE_u 相减后的类型为 npt.NDArray[np.unsignedinteger[Any]]
assert_type(AR_u - AR_LIKE_i, npt.NDArray[np.signed
# 确保 AR_i 与 AR_LIKE_u 的元素类型为带符号整数的 NumPy 数组
assert_type(AR_i - AR_LIKE_u, npt.NDArray[np.signedinteger[Any]])

# 确保 AR_i 与 AR_LIKE_i 的元素类型为带符号整数的 NumPy 数组
assert_type(AR_i - AR_LIKE_i, npt.NDArray[np.signedinteger[Any]])

# 确保 AR_i 与 AR_LIKE_f 的元素类型为浮点数的 NumPy 数组
assert_type(AR_i - AR_LIKE_f, npt.NDArray[np.floating[Any]])

# 确保 AR_i 与 AR_LIKE_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_i - AR_LIKE_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_i 与 AR_LIKE_m 的元素类型为时间间隔的 NumPy 数组
assert_type(AR_i - AR_LIKE_m, npt.NDArray[np.timedelta64])

# 确保 AR_i 与 AR_LIKE_O 的元素类型为任意类型
assert_type(AR_i - AR_LIKE_O, Any)

# 确保 AR_LIKE_b 与 AR_i 的元素类型为带符号整数的 NumPy 数组
assert_type(AR_LIKE_b - AR_i, npt.NDArray[np.signedinteger[Any]])

# 确保 AR_LIKE_u 与 AR_i 的元素类型为带符号整数的 NumPy 数组
assert_type(AR_LIKE_u - AR_i, npt.NDArray[np.signedinteger[Any]])

# 确保 AR_LIKE_i 与 AR_i 的元素类型为带符号整数的 NumPy 数组
assert_type(AR_LIKE_i - AR_i, npt.NDArray[np.signedinteger[Any]])

# 确保 AR_LIKE_f 与 AR_i 的元素类型为浮点数的 NumPy 数组
assert_type(AR_LIKE_f - AR_i, npt.NDArray[np.floating[Any]])

# 确保 AR_LIKE_c 与 AR_i 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_c - AR_i, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_m 与 AR_i 的元素类型为时间间隔的 NumPy 数组
assert_type(AR_LIKE_m - AR_i, npt.NDArray[np.timedelta64])

# 确保 AR_LIKE_M 与 AR_i 的元素类型为日期时间的 NumPy 数组
assert_type(AR_LIKE_M - AR_i, npt.NDArray[np.datetime64])

# 确保 AR_LIKE_O 与 AR_i 的元素类型为任意类型
assert_type(AR_LIKE_O - AR_i, Any)

# 确保 AR_f 与 AR_LIKE_b 的元素类型为浮点数的 NumPy 数组
assert_type(AR_f - AR_LIKE_b, npt.NDArray[np.floating[Any]])

# 确保 AR_f 与 AR_LIKE_u 的元素类型为浮点数的 NumPy 数组
assert_type(AR_f - AR_LIKE_u, npt.NDArray[np.floating[Any]])

# 确保 AR_f 与 AR_LIKE_i 的元素类型为浮点数的 NumPy 数组
assert_type(AR_f - AR_LIKE_i, npt.NDArray[np.floating[Any]])

# 确保 AR_f 与 AR_LIKE_f 的元素类型为浮点数的 NumPy 数组
assert_type(AR_f - AR_LIKE_f, npt.NDArray[np.floating[Any]])

# 确保 AR_f 与 AR_LIKE_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_f - AR_LIKE_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_f 与 AR_LIKE_O 的元素类型为任意类型
assert_type(AR_f - AR_LIKE_O, Any)

# 确保 AR_LIKE_b 与 AR_f 的元素类型为浮点数的 NumPy 数组
assert_type(AR_LIKE_b - AR_f, npt.NDArray[np.floating[Any]])

# 确保 AR_LIKE_u 与 AR_f 的元素类型为浮点数的 NumPy 数组
assert_type(AR_LIKE_u - AR_f, npt.NDArray[np.floating[Any]])

# 确保 AR_LIKE_i 与 AR_f 的元素类型为浮点数的 NumPy 数组
assert_type(AR_LIKE_i - AR_f, npt.NDArray[np.floating[Any]])

# 确保 AR_LIKE_f 与 AR_f 的元素类型为浮点数的 NumPy 数组
assert_type(AR_LIKE_f - AR_f, npt.NDArray[np.floating[Any]])

# 确保 AR_LIKE_c 与 AR_f 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_c - AR_f, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_O 与 AR_f 的元素类型为任意类型
assert_type(AR_LIKE_O - AR_f, Any)

# 确保 AR_c 与 AR_LIKE_b 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_c - AR_LIKE_b, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_c 与 AR_LIKE_u 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_c - AR_LIKE_u, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_c 与 AR_LIKE_i 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_c - AR_LIKE_i, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_c 与 AR_LIKE_f 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_c - AR_LIKE_f, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_c 与 AR_LIKE_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_c - AR_LIKE_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_c 与 AR_LIKE_O 的元素类型为任意类型
assert_type(AR_c - AR_LIKE_O, Any)

# 确保 AR_LIKE_b 与 AR_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_b - AR_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_u 与 AR_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_u - AR_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_i 与 AR_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_i - AR_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_f 与 AR_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_f - AR_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_c 与 AR_c 的元素类型为复数浮点数的 NumPy 数组
assert_type(AR_LIKE_c - AR_c, npt.NDArray[np.complexfloating[Any, Any]])

# 确保 AR_LIKE_O 与 AR_c 的元素类型为任意类型
assert_type(AR_LIKE_O - AR_c, Any)

# 确保 AR_m 与 AR_LIKE_b 的元素类型为时间间隔的 NumPy 数组
assert_type(AR_m - AR_LIKE_b, npt.NDArray[np.timedelta64])

# 确保 AR_m 与 AR_LIKE_u 的元素类型为时间间隔的 NumPy 数组
assert_type(AR_m - AR_LIKE_u, npt.NDArray[np.timedelta64])

# 确保 AR_m 与 AR_LIKE_i 的元素类型为时间间隔的 NumPy 数组
assert_type(AR_m - AR_LIKE_i, npt.NDArray[np.timedelta64])

# 确保 AR_m 与 AR_LIKE_m 的元素类型为时间间隔的 NumPy 数组
assert
# 确保 AR_M 减去 AR_LIKE_b 后的结果为 np.datetime64 类型
assert_type(AR_M - AR_LIKE_b, npt.NDArray[np.datetime64])
# 确保 AR_M 减去 AR_LIKE_u 后的结果为 np.datetime64 类型
assert_type(AR_M - AR_LIKE_u, npt.NDArray[np.datetime64])
# 确保 AR_M 减去 AR_LIKE_i 后的结果为 np.datetime64 类型
assert_type(AR_M - AR_LIKE_i, npt.NDArray[np.datetime64])
# 确保 AR_M 减去 AR_LIKE_m 后的结果为 np.datetime64 类型
assert_type(AR_M - AR_LIKE_m, npt.NDArray[np.datetime64])
# 确保 AR_M 减去 AR_LIKE_M 后的结果为 np.timedelta64 类型
assert_type(AR_M - AR_LIKE_M, npt.NDArray[np.timedelta64])
# 确保 AR_M 减去 AR_LIKE_O 后的结果为任何类型
assert_type(AR_M - AR_LIKE_O, Any)

# 确保 AR_LIKE_M 减去 AR_M 后的结果为 np.timedelta64 类型
assert_type(AR_LIKE_M - AR_M, npt.NDArray[np.timedelta64])
# 确保 AR_LIKE_O 减去 AR_M 后的结果为任何类型
assert_type(AR_LIKE_O - AR_M, Any)

# 确保 AR_O 减去 AR_LIKE_b 后的结果为任何类型
assert_type(AR_O - AR_LIKE_b, Any)
# 确保 AR_O 减去 AR_LIKE_u 后的结果为任何类型
assert_type(AR_O - AR_LIKE_u, Any)
# 确保 AR_O 减去 AR_LIKE_i 后的结果为任何类型
assert_type(AR_O - AR_LIKE_i, Any)
# 确保 AR_O 减去 AR_LIKE_f 后的结果为任何类型
assert_type(AR_O - AR_LIKE_f, Any)
# 确保 AR_O 减去 AR_LIKE_c 后的结果为任何类型
assert_type(AR_O - AR_LIKE_c, Any)
# 确保 AR_O 减去 AR_LIKE_m 后的结果为任何类型
assert_type(AR_O - AR_LIKE_m, Any)
# 确保 AR_O 减去 AR_LIKE_M 后的结果为任何类型
assert_type(AR_O - AR_LIKE_M, Any)
# 确保 AR_O 减去 AR_LIKE_O 后的结果为任何类型
assert_type(AR_O - AR_LIKE_O, Any)

# 确保 AR_LIKE_b 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_b - AR_O, Any)
# 确保 AR_LIKE_u 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_u - AR_O, Any)
# 确保 AR_LIKE_i 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_i - AR_O, Any)
# 确保 AR_LIKE_f 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_f - AR_O, Any)
# 确保 AR_LIKE_c 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_c - AR_O, Any)
# 确保 AR_LIKE_m 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_m - AR_O, Any)
# 确保 AR_LIKE_M 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_M - AR_O, Any)
# 确保 AR_LIKE_O 减去 AR_O 后的结果为任何类型
assert_type(AR_LIKE_O - AR_O, Any)

# 数组的整数除法操作

# 确保 AR_b 除以 AR_LIKE_b 后的结果为 np.int8 类型
assert_type(AR_b // AR_LIKE_b, npt.NDArray[np.int8])
# 确保 AR_b 除以 AR_LIKE_u 后的结果为 np.unsignedinteger[Any] 类型的数组
assert_type(AR_b // AR_LIKE_u, npt.NDArray[np.unsignedinteger[Any]])
# 确保 AR_b 除以 AR_LIKE_i 后的结果为 np.signedinteger[Any] 类型的数组
assert_type(AR_b // AR_LIKE_i, npt.NDArray[np.signedinteger[Any]])
# 确保 AR_b 除以 AR_LIKE_f 后的结果为 np.floating[Any] 类型的数组
assert_type(AR_b // AR_LIKE_f, npt.NDArray[np.floating[Any]])
# 确保 AR_b 除以 AR_LIKE_O 后的结果为任何类型
assert_type(AR_b // AR_LIKE_O, Any)

# 确保 AR_LIKE_b 除以 AR_b 后的结果为 np.int8 类型的数组
assert_type(AR_LIKE_b // AR_b, npt.NDArray[np.int8])
# 确保 AR_LIKE_u 除以 AR_b 后的结果为 np.unsignedinteger[Any] 类型的数组
assert_type(AR_LIKE_u // AR_b, npt.NDArray[np.unsignedinteger[Any]])
# 确保 AR_LIKE_i 除以 AR_b 后的结果为 np.signedinteger[Any] 类型的数组
assert_type(AR_LIKE_i // AR_b, npt.NDArray[np.signedinteger[Any]])
# 确保 AR_LIKE_f 除以 AR_b 后的结果为 np.floating[Any] 类型的数组
assert_type(AR_LIKE_f // AR_b, npt.NDArray[np.floating[Any]])
# 确保 AR_LIKE_O 除以 AR_b 后的结果为任何类型
assert_type(AR_LIKE_O // AR_b, Any)

# 确保 AR_u 除以 AR_LIKE_b 后的结果为 np.unsignedinteger[Any] 类型的数组
assert_type(AR_u // AR_LIKE_b, npt.NDArray[np.unsignedinteger[Any]])
# 确保 AR_u 除以 AR_LIKE_u 后的结果为 np.unsignedinteger[Any] 类型的数组
assert_type(AR_u // AR_LIKE_u, npt.NDArray[np.unsignedinteger[Any]])
# 确保 AR_u 除以 AR_LIKE_i 后的结果为 np.signedinteger[Any] 类型的数组
assert_type(AR_u // AR_LIKE_i, npt.NDArray[np.signedinteger[Any]])
# 确保 AR_u 除以 AR_LIKE_f 后的结果为 np.floating[Any] 类型的数组
assert_type(AR_u // AR_LIKE_f, npt.NDArray[np.floating[Any]])
# 确保 AR_u 除以 AR_LIKE_O 后的结果为任何类型
assert_type(AR_u // AR_LIKE_O, Any)

# 确保 AR_LIKE_b 除以 AR_u 后的结果为 np.unsignedinteger[Any] 类型的数组
assert_type(AR_LIKE_b // AR_u, npt.NDArray[np.unsignedinteger[Any]])
# 确保 AR_LIKE_u 除以 AR_u 后的结果为 np.unsignedinteger[Any] 类型的数组
assert_type(AR_LIKE_u // AR_u, npt.NDArray[np.unsignedinteger[Any]])
# 确保 AR_LIKE_i 除以 AR_u 后的结果为 np.signedinteger[Any] 类型的数组
assert_type(AR_LIKE_i // AR_u, npt.NDArray[np.signedinteger[Any]])
# 确保 AR_LIKE_f 除以 AR_u 后的结果为 np.floating[Any] 类型的数组
assert_type(AR_LIKE_f // AR_u, npt.NDArray[np.floating[Any]])
# 确保 AR_LIKE_m 除以 AR_u 后的结果为 np.timedelta64 类型的数组
assert_type(AR_LIKE_m // AR_u, npt.NDArray[np.timedelta64])
# 确保 AR_LIKE_O 除以 AR_u 后的结果为任何类型
assert_type(AR_LIKE_O // AR_u, Any)

# 确保 AR_i 除以 AR_LIKE_b 后的结果为 np.signedinteger[Any] 类型的数组
assert_type(AR_i // AR_LIKE_b, npt.NDArray[np.signedinteger[Any]])
# 确保 AR_i 除以 AR_LIKE_u 后的结果为 np.signedinteger[Any] 类型的数组
assert_type(AR_i // AR_LIKE_u, npt.NDArray[np.signedinteger[Any]])
# 确保 AR_i 除以
# 断言表达式，验证 AR_f 与 AR_LIKE_u 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_f // AR_LIKE_u, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_f 与 AR_LIKE_i 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_f // AR_LIKE_i, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_f 与 AR_LIKE_f 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_f // AR_LIKE_f, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_f 与 AR_LIKE_O 之间的整除操作结果类型为任意类型
assert_type(AR_f // AR_LIKE_O, Any)

# 断言表达式，验证 AR_LIKE_b 与 AR_f 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_LIKE_b // AR_f, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_LIKE_u 与 AR_f 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_LIKE_u // AR_f, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_LIKE_i 与 AR_f 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_LIKE_i // AR_f, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_LIKE_f 与 AR_f 之间的整除操作结果类型为 numpy 中的浮点数数组
assert_type(AR_LIKE_f // AR_f, npt.NDArray[np.floating[Any]])
# 断言表达式，验证 AR_LIKE_m 与 AR_f 之间的整除操作结果类型为 numpy 中的时间间隔数组
assert_type(AR_LIKE_m // AR_f, npt.NDArray[np.timedelta64])
# 断言表达式，验证 AR_LIKE_O 与 AR_f 之间的整除操作结果类型为任意类型
assert_type(AR_LIKE_O // AR_f, Any)

# 断言表达式，验证 AR_m 与 AR_LIKE_u 之间的整除操作结果类型为 numpy 中的时间间隔数组
assert_type(AR_m // AR_LIKE_u, npt.NDArray[np.timedelta64])
# 断言表达式，验证 AR_m 与 AR_LIKE_i 之间的整除操作结果类型为 numpy 中的时间间隔数组
assert_type(AR_m // AR_LIKE_i, npt.NDArray[np.timedelta64])
# 断言表达式，验证 AR_m 与 AR_LIKE_f 之间的整除操作结果类型为 numpy 中的时间间隔数组
assert_type(AR_m // AR_LIKE_f, npt.NDArray[np.timedelta64])
# 断言表达式，验证 AR_m 与 AR_LIKE_m 之间的整除操作结果类型为 numpy 中的整数数组
assert_type(AR_m // AR_LIKE_m, npt.NDArray[np.int64])
# 断言表达式，验证 AR_m 与 AR_LIKE_O 之间的整除操作结果类型为任意类型
assert_type(AR_m // AR_LIKE_O, Any)

# 断言表达式，验证 AR_LIKE_m 与 AR_m 之间的整除操作结果类型为 numpy 中的整数数组
assert_type(AR_LIKE_m // AR_m, npt.NDArray[np.int64])
# 断言表达式，验证 AR_LIKE_O 与 AR_m 之间的整除操作结果类型为任意类型
assert_type(AR_LIKE_O // AR_m, Any)

# 断言表达式，验证 AR_O 与 AR_LIKE_b 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_b, Any)
# 断言表达式，验证 AR_O 与 AR_LIKE_u 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_u, Any)
# 断言表达式，验证 AR_O 与 AR_LIKE_i 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_i, Any)
# 断言表达式，验证 AR_O 与 AR_LIKE_f 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_f, Any)
# 断言表达式，验证 AR_O 与 AR_LIKE_m 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_m, Any)
# 断言表达式，验证 AR_O 与 AR_LIKE_M 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_M, Any)
# 断言表达式，验证 AR_O 与 AR_LIKE_O 之间的整除操作结果类型为任意类型
assert_type(AR_O // AR_LIKE_O, Any)

# 断言表达式，验证 f16 取负操作的结果类型为 numpy 中的浮点数，精度为 128 位
assert_type(-f16, np.floating[_128Bit])
# 断言表达式，验证 c16 取负操作的结果类型为 numpy 中的复数，精度为 complex128
assert_type(-c16, np.complex128)
# 断言表达式，验证 c8 取负操作的结果类型为 numpy 中的复数，精度为 complex64
assert_type(-c8, np.complex64)
# 断言表达式，验证 f8 取负操作的结果类型为 numpy 中的浮点数，精度为 float64
assert_type(-f8, np.float64)
# 断言表达式，验证 f4 取负操作的结果类型为 numpy 中的浮点数，精度为 float32
assert_type(-f4, np.float32)
# 断言表达式，验证 i8 取负操作的结果类型为 numpy 中的整数，精度为 int64
assert_type(-i8, np.int64)
# 断言表达式，验证 i4 取负操作的结果类型为 numpy 中的整数，精度为 int32
assert_type(-i4, np.int32)
# 断言表达式，验证 u8 取负操作的结果类型为 numpy 中的无符号整数，精度为 uint64
assert_type(-u8, np.uint64)
# 断言表达式，验证 u4 取负操作的结果类型为 numpy 中的无符号整数，精度为 uint32
assert_type(-u4, np.uint32)
# 断言表达式，验证 td 取负操作的结果类型为 numpy 中的时间间隔，精度由 td 决定
assert_type(-td, np.timedelta64)
# 断言表达式，验证 AR_f 取负操作的结果类型为 numpy 中的浮点数数组，精度为 float64
assert_type(-AR_f, npt.NDArray[np.float64])

# 断言表达式，验证 f16 取正操作的结果类型为 numpy 中的浮点数，精度为 128 位
assert_type(+f16, np.floating[_128Bit])
# 断言表达式，验证 c16 取正操作的结果类型为 numpy 中的复数，精度为 complex128
assert_type(+c16, np.complex128)
# 断言表达式，验证 c8 取正操作的结果类型为 numpy 中的复数，精度为 complex64
assert_type(+c8, np.complex64)
# 断言表达式，验证 f8 取正操作的结果类型为 numpy 中的浮点数，精度为 float64
assert_type(+f8, np.float64)
# 断言表达式，验证 f4 取正操作的结果类型为 numpy 中的浮点数，精度为 float32
assert_type(+f4, np.float32)
# 断言表达式，验证 i8 取正操作的结果类型为 numpy 中的整数，精度为 int64
assert_type(+i8, np.int64)
# 断言表达式，验证 i4 取正操作的结果类型为 numpy 中的整数，精度为 int32
assert_type(+i4, np.int32)
# 断言表达式，验证 u8 取正操作的结果类型为 numpy
# Assert that the result of subtracting two timedelta64 objects is of type np.timedelta64
assert_type(td - td, np.timedelta64)
# Assert that the result of subtracting a timedelta64 object from an integer is of type np.timedelta64
assert_type(td - i, np.timedelta64)
# Assert that the result of subtracting a timedelta64 object from an int32 is of type np.timedelta64
assert_type(td - i4, np.timedelta64)
# Assert that the result of subtracting a timedelta64 object from an int64 is of type np.timedelta64
assert_type(td - i8, np.timedelta64)
# Assert that the result of dividing a timedelta64 object by a float is of type np.timedelta64
assert_type(td / f, np.timedelta64)
# Assert that the result of dividing a timedelta64 object by a float32 is of type np.timedelta64
assert_type(td / f4, np.timedelta64)
# Assert that the result of dividing a timedelta64 object by a float64 is of type np.timedelta64
assert_type(td / f8, np.timedelta64)
# Assert that the result of dividing a timedelta64 object by another timedelta64 object is of type np.float64
assert_type(td / td, np.float64)
# Assert that the result of floor dividing a timedelta64 object by another timedelta64 object is of type np.int64
assert_type(td // td, np.int64)

# Boolean operations

# Assert that the result of dividing a boolean object by a boolean object is of type np.float64
assert_type(b_ / b, np.float64)
# Assert that the result of dividing a boolean object by another boolean object is of type np.float64
assert_type(b_ / b_, np.float64)
# Assert that the result of dividing a boolean object by an integer is of type np.float64
assert_type(b_ / i, np.float64)
# Assert that the result of dividing a boolean object by an int64 is of type np.float64
assert_type(b_ / i8, np.float64)
# Assert that the result of dividing a boolean object by an int32 is of type np.float64
assert_type(b_ / i4, np.float64)
# Assert that the result of dividing a boolean object by an uint64 is of type np.float64
assert_type(b_ / u8, np.float64)
# Assert that the result of dividing a boolean object by an uint32 is of type np.float64
assert_type(b_ / u4, np.float64)
# Assert that the result of dividing a boolean object by a float is of type np.float64
assert_type(b_ / f, np.float64)
# Assert that the result of dividing a boolean object by a float128 is of type np.floating[_128Bit]
assert_type(b_ / f16, np.floating[_128Bit])
# Assert that the result of dividing a boolean object by a float64 is of type np.float64
assert_type(b_ / f8, np.float64)
# Assert that the result of dividing a boolean object by a float32 is of type np.float32
assert_type(b_ / f4, np.float32)
# Assert that the result of dividing a boolean object by a complex128 object is of type np.complex128
assert_type(b_ / c, np.complex128)
# Assert that the result of dividing a boolean object by a complex256 object is of type np.complex128
assert_type(b_ / c16, np.complex128)
# Assert that the result of dividing a boolean object by a complex128 object is of type np.complex64
assert_type(b_ / c8, np.complex64)

# Assert that the result of dividing an integer by a boolean object is of type np.float64
assert_type(b / b_, np.float64)
# Assert that the result of dividing a boolean object by another boolean object is of type np.float64
assert_type(b_ / b_, np.float64)
# Assert that the result of dividing an integer by a boolean object is of type np.float64
assert_type(i / b_, np.float64)
# Assert that the result of dividing an int64 by a boolean object is of type np.float64
assert_type(i8 / b_, np.float64)
# Assert that the result of dividing an int32 by a boolean object is of type np.float64
assert_type(i4 / b_, np.float64)
# Assert that the result of dividing an uint64 by a boolean object is of type np.float64
assert_type(u8 / b_, np.float64)
# Assert that the result of dividing an uint32 by a boolean object is of type np.float64
assert_type(u4 / b_, np.float64)
# Assert that the result of dividing a float by a boolean object is of type np.float64
assert_type(f / b_, np.float64)
# Assert that the result of dividing a float128 by a boolean object is of type np.floating[_128Bit]
assert_type(f16 / b_, np.floating[_128Bit])
# Assert that the result of dividing a float64 by a boolean object is of type np.float64
assert_type(f8 / b_, np.float64)
# Assert that the result of dividing a float32 by a boolean object is of type np.float32
assert_type(f4 / b_, np.float32)
# Assert that the result of dividing a complex128 by a boolean object is of type np.complex128
assert_type(c / b_, np.complex128)
# Assert that the result of dividing a complex256 by a boolean object is of type np.complex128
assert_type(c16 / b_, np.complex128)
# Assert that the result of dividing a complex128 by a boolean object is of type np.complex64
assert_type(c8 / b_, np.complex64)

# Complex operations

# Assert that the result of adding a complex256 object to a float128 object is of type np.complexfloating[_64Bit | _128Bit, _64Bit | _128Bit]
assert_type(c16 + f16, np.complexfloating[_64Bit | _128Bit, _64Bit | _128Bit])
# Assert that the result of adding two complex128 objects is of type np.complex128
assert_type(c16 + c16, np.complex128)
# Assert that the result of adding a complex128 object to a float64 object is of type np.complex128
assert_type(c16 + f8, np.complex128)
# Assert that the result of adding a complex128 object to an int64 object is of type np.complex128
assert_type(c16 + i8, np.complex128)
# Assert that the result of adding two complex64 objects is of type np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c16 + c8, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])
# Assert that the result of adding a complex64 object to a float32 object is of type np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c16 + f4, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])
# Assert that the result of adding a complex64 object to an int32 object is of type np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c16 + i4, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])
# Assert that the result of adding a complex128 object to a boolean object is of type np.complex128
assert_type(c16 + b_, np.complex128)
# Assert that the result of adding a complex128 object to a boolean object is of type np.complex128
assert_type(c16 + b, np.complex128)
# Assert that the result of adding a complex128 object to a complex64 object is of type np.complex128
assert_type(c16 + c, np.complex128)
# Assert that the result of adding a complex128 object to a float object is of type np.complex128
assert_type(c16 + f, np.complex128)
# Assert that the result of adding a complex128 object to an array of float objects is of type npt.NDArray[np.complexfloating[Any, Any]]

# Assert that the result of adding a float128 object to a complex256 object is of type np.complexfloating[_64Bit | _128Bit, _64Bit | _128Bit]
assert_type(f16 + c16, np.complexfloating[_64Bit | _128Bit, _64Bit | _128Bit])
# Assert that the result of adding two complex128 objects is of type np.complex128
assert_type(c16 + c16, np.complex128)
# Assert that the result of adding a float64 object to a complex128 object is of type np.complex128
assert_type(f8 + c16, np.complex128)
# Assert that the result of adding an int64 object to a complex128 object is of type np.complex128
assert_type(i8 + c16, np.complex128)
# Assert that the result of adding a complex64 object to a complex128 object is of type np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c8 + c16, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])
# Assert that the result of adding a float32 object to a complex128 object is of type np.complexfloating[_
# 断言两个复数相加后的类型为 np.complex64
assert_type(c8 + b_, np.complex64)

# 断言两个复数相加后的类型为 np.complex64
assert_type(c8 + b, np.complex64)

# 断言一个复数与复数浮点数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c8 + c, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言一个复数与复数浮点数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c8 + f, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言一个复数与任意复数数组相加后的类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(c8 + AR_f, npt.NDArray[np.complexfloating[Any, Any]])

# 断言一个浮点数与复数相加后的类型为 np.complexfloating[_32Bit | _128Bit, _32Bit | _128Bit]
assert_type(f16 + c8, np.complexfloating[_32Bit | _128Bit, _32Bit | _128Bit])

# 断言两个复数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c16 + c8, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言一个浮点数与复数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(f8 + c8, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言一个整数与复数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(i8 + c8, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言两个复数相加后的类型为 np.complex64
assert_type(c8 + c8, np.complex64)

# 断言一个单精度浮点数与复数相加后的类型为 np.complex64
assert_type(f4 + c8, np.complex64)

# 断言一个整数与复数相加后的类型为 np.complex64
assert_type(i4 + c8, np.complex64)

# 断言一个布尔值与复数相加后的类型为 np.complex64
assert_type(b_ + c8, np.complex64)

# 断言一个布尔值与复数相加后的类型为 np.complex64
assert_type(b + c8, np.complex64)

# 断言一个复数与复数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(c + c8, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言一个浮点数与复数相加后的类型为 np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit]
assert_type(f + c8, np.complexfloating[_32Bit | _64Bit, _32Bit | _64Bit])

# 断言一个任意复数数组与复数相加后的类型为 npt.NDArray[np.complexfloating[Any, Any]]
assert_type(AR_f + c8, npt.NDArray[np.complexfloating[Any, Any]])

# 断言两个双精度浮点数相加后的类型为 np.floating[_64Bit | _128Bit]
assert_type(f8 + f16, np.floating[_64Bit | _128Bit])

# 断言两个双精度浮点数相加后的类型为 np.float64
assert_type(f8 + f8, np.float64)

# 断言一个双精度浮点数与整数相加后的类型为 np.float64
assert_type(f8 + i8, np.float64)

# 断言一个双精度浮点数与单精度浮点数相加后的类型为 np.floating[_32Bit | _64Bit]
assert_type(f8 + f4, np.floating[_32Bit | _64Bit])

# 断言一个双精度浮点数与整数相加后的类型为 np.floating[_32Bit | _64Bit]
assert_type(f8 + i4, np.floating[_32Bit | _64Bit])

# 断言一个双精度浮点数与布尔值相加后的类型为 np.float64
assert_type(f8 + b_, np.float64)

# 断言一个双精度浮点数与布尔值相加后的类型为 np.float64
assert_type(f8 + b, np.float64)

# 断言一个双精度浮点数与复数相加后的类型为 np.complex128
assert_type(f8 + c, np.complex128)

# 断言一个双精度浮点数与浮点数相加后的类型为 np.float64
assert_type(f8 + f, np.float64)

# 断言一个双精度浮点数与任意浮点数数组相加后的类型为 npt.NDArray[np.floating[Any]]
assert_type(f8 + AR_f, npt.NDArray[np.floating[Any]])

# 断言一个双精度浮点数与双精度浮点数相加后的类型为 np.floating[_64Bit | _128Bit]
assert_type(f16 + f8, np.floating[_64Bit | _128Bit])

# 断言一个双精度浮点数与整数相加后的类型为 np.float64
assert_type(i8 + f8, np.float64)

# 断言一个单精度浮点数与双精度浮点数相加后的类型为 np.floating[_32Bit | _64Bit]
assert_type(f4 + f8, np.floating[_32Bit | _64Bit])

# 断言一个单精度浮点数与整数相加后的类型为 np.floating[_32Bit | _64Bit]
assert_type(i4 + f8, np.floating[_32Bit | _64Bit])

# 断言一个布尔值与双精度浮点数相加后的类型为 np.float64
assert_type(b_ + f8, np.float64)

# 断言一个布尔值与双精度浮点数相加后的类型为 np.float64
assert_type(b + f8, np.float64)

# 断言一个复数与双精度浮点数相加后的类型为 np.complex128
assert_type(c + f8, np.complex128)

# 断言一个浮点数与双精度浮点数相加后的类型为 np.float64
assert_type(f + f8, np.float64)

# 断言一个任意浮点数数组与双精度浮点数相加后的类型为 npt.NDArray[np.floating[Any]]
assert_type(AR_f + f8, npt.NDArray[np.floating[Any]])

# 断言一个单精度浮点数与单精度浮点数相加后的类型为 np.floating[_32Bit | _128Bit]
assert_type(f4 + f16, np.floating[_32Bit | _128Bit])

# 断言一个单精度浮点数与双精度浮点数相加后的类型为 np.floating[_32Bit | _64Bit]
assert_type(f4 + f8, np.floating[_32Bit | _64Bit])

# 断言一个单精度浮点数与整数相加后的类型为 np.floating[_32Bit | _64Bit]
assert_type(f4 + i8, np.floating[_32Bit | _64Bit])

# 断言两个单精度浮点数相加后的类型为 np.float32
assert_type(f4 + f4, np.float32)

# 断言一个单精度浮点数与整数相加后的类型为 np.float32
assert_type(f4 + i4, np.float32)

# 断言一个单精度浮点数与布尔值相加后的类型为 np.float32
assert_type(f4 + b_, np.float32)

#
# 检查 i8 与 f 的相加结果的数据类型是否为 np.float64
assert_type(i8 + f, np.float64)

# 检查 i8 与 AR_f 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(i8 + AR_f, npt.NDArray[np.floating[Any]])

# 检查 u8 与 u8 的相加结果的数据类型是否为 np.uint64
assert_type(u8 + u8, np.uint64)

# 检查 u8 与 i4 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(u8 + i4, Any)

# 检查 u8 与 u4 的相加结果的数据类型是否为 np.unsignedinteger[_32Bit | _64Bit]
assert_type(u8 + u4, np.unsignedinteger[_32Bit | _64Bit])

# 检查 u8 与 b_ 的相加结果的数据类型是否为 np.uint64
assert_type(u8 + b_, np.uint64)

# 检查 u8 与 b 的相加结果的数据类型是否为 np.uint64
assert_type(u8 + b, np.uint64)

# 检查 u8 与 c 的相加结果的数据类型是否为 np.complex128
assert_type(u8 + c, np.complex128)

# 检查 u8 与 f 的相加结果的数据类型是否为 np.float64
assert_type(u8 + f, np.float64)

# 检查 u8 与 AR_f 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(u8 + AR_f, npt.NDArray[np.floating[Any]])

# 检查 i8 与 i8 的相加结果的数据类型是否为 np.int64
assert_type(i8 + i8, np.int64)

# 检查 u8 与 i8 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(u8 + i8, Any)

# 检查 i4 与 i8 的相加结果的数据类型是否为 np.signedinteger[_32Bit | _64Bit]
assert_type(i4 + i8, np.signedinteger[_32Bit | _64Bit])

# 检查 u4 与 i8 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(u4 + i8, Any)

# 检查 b_ 与 i8 的相加结果的数据类型是否为 np.int64
assert_type(b_ + i8, np.int64)

# 检查 b 与 i8 的相加结果的数据类型是否为 np.int64
assert_type(b + i8, np.int64)

# 检查 c 与 i8 的相加结果的数据类型是否为 np.complex128
assert_type(c + i8, np.complex128)

# 检查 f 与 i8 的相加结果的数据类型是否为 np.float64
assert_type(f + i8, np.float64)

# 检查 AR_f 与 i8 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(AR_f + i8, npt.NDArray[np.floating[Any]])

# 检查 u8 与 u8 的相加结果的数据类型是否为 np.uint64
assert_type(u8 + u8, np.uint64)

# 检查 i4 与 u8 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(i4 + u8, Any)

# 检查 u4 与 u8 的相加结果的数据类型是否为 np.unsignedinteger[_32Bit | _64Bit]
assert_type(u4 + u8, np.unsignedinteger[_32Bit | _64Bit])

# 检查 b_ 与 u8 的相加结果的数据类型是否为 np.uint64
assert_type(b_ + u8, np.uint64)

# 检查 b 与 u8 的相加结果的数据类型是否为 np.uint64
assert_type(b + u8, np.uint64)

# 检查 c 与 u8 的相加结果的数据类型是否为 np.complex128
assert_type(c + u8, np.complex128)

# 检查 f 与 u8 的相加结果的数据类型是否为 np.float64
assert_type(f + u8, np.float64)

# 检查 AR_f 与 u8 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(AR_f + u8, npt.NDArray[np.floating[Any]])

# 检查 i4 与 i8 的相加结果的数据类型是否为 np.signedinteger[_32Bit | _64Bit]
assert_type(i4 + i8, np.signedinteger[_32Bit | _64Bit])

# 检查 i4 与 i4 的相加结果的数据类型是否为 np.int32
assert_type(i4 + i4, np.int32)

# 检查 i4 与 b_ 的相加结果的数据类型是否为 np.int32
assert_type(i4 + b_, np.int32)

# 检查 i4 与 b 的相加结果的数据类型是否为 np.int32
assert_type(i4 + b, np.int32)

# 检查 i4 与 AR_f 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(i4 + AR_f, npt.NDArray[np.floating[Any]])

# 检查 u4 与 i8 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(u4 + i8, Any)

# 检查 u4 与 i4 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(u4 + i4, Any)

# 检查 u4 与 u8 的相加结果的数据类型是否为 np.unsignedinteger[_32Bit | _64Bit]
assert_type(u4 + u8, np.unsignedinteger[_32Bit | _64Bit])

# 检查 u4 与 u4 的相加结果的数据类型是否为 np.uint32
assert_type(u4 + u4, np.uint32)

# 检查 u4 与 b_ 的相加结果的数据类型是否为 np.uint32
assert_type(u4 + b_, np.uint32)

# 检查 u4 与 b 的相加结果的数据类型是否为 np.uint32
assert_type(u4 + b, np.uint32)

# 检查 u4 与 AR_f 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(u4 + AR_f, npt.NDArray[np.floating[Any]])

# 检查 i8 与 i4 的相加结果的数据类型是否为 np.signedinteger[_32Bit | _64Bit]
assert_type(i8 + i4, np.signedinteger[_32Bit | _64Bit])

# 检查 i4 与 i4 的相加结果的数据类型是否为 np.int32
assert_type(i4 + i4, np.int32)

# 检查 b_ 与 i4 的相加结果的数据类型是否为 np.int32
assert_type(b_ + i4, np.int32)

# 检查 b 与 i4 的相加结果的数据类型是否为 np.int32
assert_type(b + i4, np.int32)

# 检查 AR_f 与 i4 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(AR_f + i4, npt.NDArray[np.floating[Any]])

# 检查 i8 与 u4 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(i8 + u4, Any)

# 检查 i4 与 u4 的相加结果的数据类型是否为 Any（即没有特定限制）
assert_type(i4 + u4, Any)

# 检查 u8 与 u4 的相加结果的数据类型是否为 np.unsignedinteger[_32Bit | _64Bit]
assert_type(u8 + u4, np.unsignedinteger[_32Bit | _64Bit])

# 检查 u4 与 u4 的相加结果的数据类型是否为 np.uint32
assert_type(u4 + u4, np.uint32)

# 检查 b_ 与 u4 的相加结果的数据类型是否为 np.uint32
assert_type(b_ + u4, np.uint32)

# 检查 b 与 u4 的相加结果的数据类型是否为 np.uint32
assert_type(b + u4, np.uint32)

# 检查 AR_f 与 u4 的相加结果的数据类型是否为 npt.NDArray[np.floating[Any]]
assert_type(AR_f + u4, npt.NDArray[np.f
```