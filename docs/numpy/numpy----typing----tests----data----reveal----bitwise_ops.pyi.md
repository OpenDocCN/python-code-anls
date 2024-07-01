# `.\numpy\numpy\typing\tests\data\reveal\bitwise_ops.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型注解模块
from numpy._typing import _64Bit, _32Bit  # 导入 64 位和 32 位整数类型注解

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果 Python 版本大于等于 3.11，则使用标准库中的 assert_type
else:
    from typing_extensions import assert_type  # 否则，从 typing_extensions 中导入 assert_type

i8 = np.int64(1)  # 创建一个 NumPy 的 64 位有符号整数对象
u8 = np.uint64(1)  # 创建一个 NumPy 的 64 位无符号整数对象

i4 = np.int32(1)  # 创建一个 NumPy 的 32 位有符号整数对象
u4 = np.uint32(1)  # 创建一个 NumPy 的 32 位无符号整数对象

b_ = np.bool(1)  # 创建一个 NumPy 的布尔类型对象

b = bool(1)  # 创建一个 Python 内置的布尔类型对象
i = int(1)  # 创建一个 Python 内置的整数对象

AR = np.array([0, 1, 2], dtype=np.int32)  # 创建一个 NumPy 的整数数组对象，数据类型为 32 位有符号整数
AR.setflags(write=False)  # 设置数组为只读模式，禁止写入操作

# 下面是一系列的类型断言，用来验证不同操作的返回类型

assert_type(i8 << i8, np.int64)  # 断言 i8 左移 i8 的结果为 64 位有符号整数类型
assert_type(i8 >> i8, np.int64)  # 断言 i8 右移 i8 的结果为 64 位有符号整数类型
assert_type(i8 | i8, np.int64)  # 断言 i8 按位或 i8 的结果为 64 位有符号整数类型
assert_type(i8 ^ i8, np.int64)  # 断言 i8 按位异或 i8 的结果为 64 位有符号整数类型
assert_type(i8 & i8, np.int64)  # 断言 i8 按位与 i8 的结果为 64 位有符号整数类型

assert_type(i8 << AR, npt.NDArray[np.signedinteger[Any]])  # 断言 i8 左移 AR 数组的结果为特定类型的 NumPy 数组
assert_type(i8 >> AR, npt.NDArray[np.signedinteger[Any]])  # 断言 i8 右移 AR 数组的结果为特定类型的 NumPy 数组
assert_type(i8 | AR, npt.NDArray[np.signedinteger[Any]])  # 断言 i8 按位或 AR 数组的结果为特定类型的 NumPy 数组
assert_type(i8 ^ AR, npt.NDArray[np.signedinteger[Any]])  # 断言 i8 按位异或 AR 数组的结果为特定类型的 NumPy 数组
assert_type(i8 & AR, npt.NDArray[np.signedinteger[Any]])  # 断言 i8 按位与 AR 数组的结果为特定类型的 NumPy 数组

assert_type(i4 << i4, np.int32)  # 断言 i4 左移 i4 的结果为 32 位有符号整数类型
assert_type(i4 >> i4, np.int32)  # 断言 i4 右移 i4 的结果为 32 位有符号整数类型
assert_type(i4 | i4, np.int32)  # 断言 i4 按位或 i4 的结果为 32 位有符号整数类型
assert_type(i4 ^ i4, np.int32)  # 断言 i4 按位异或 i4 的结果为 32 位有符号整数类型
assert_type(i4 & i4, np.int32)  # 断言 i4 按位与 i4 的结果为 32 位有符号整数类型

assert_type(i8 << i4, np.signedinteger[_32Bit | _64Bit])  # 断言 i8 左移 i4 的结果为特定类型的 NumPy 数组
assert_type(i8 >> i4, np.signedinteger[_32Bit | _64Bit])  # 断言 i8 右移 i4 的结果为特定类型的 NumPy 数组
assert_type(i8 | i4, np.signedinteger[_32Bit | _64Bit])  # 断言 i8 按位或 i4 的结果为特定类型的 NumPy 数组
assert_type(i8 ^ i4, np.signedinteger[_32Bit | _64Bit])  # 断言 i8 按位异或 i4 的结果为特定类型的 NumPy 数组
assert_type(i8 & i4, np.signedinteger[_32Bit | _64Bit])  # 断言 i8 按位与 i4 的结果为特定类型的 NumPy 数组

assert_type(i8 << b_, np.int64)  # 断言 i8 左移 b_ 的结果为 64 位有符号整数类型
assert_type(i8 >> b_, np.int64)  # 断言 i8 右移 b_ 的结果为 64 位有符号整数类型
assert_type(i8 | b_, np.int64)  # 断言 i8 按位或 b_ 的结果为 64 位有符号整数类型
assert_type(i8 ^ b_, np.int64)  # 断言 i8 按位异或 b_ 的结果为 64 位有符号整数类型
assert_type(i8 & b_, np.int64)  # 断言 i8 按位与 b_ 的结果为 64 位有符号整数类型

assert_type(i8 << b, np.int64)  # 断言 i8 左移 b 的结果为 64 位有符号整数类型
assert_type(i8 >> b, np.int64)  # 断言 i8 右移 b 的结果为 64 位有符号整数类型
assert_type(i8 | b, np.int64)  # 断言 i8 按位或 b 的结果为 64 位有符号整数类型
assert_type(i8 ^ b, np.int64)  # 断言 i8 按位异或 b 的结果为 64 位有符号整数类型
assert_type(i8 & b, np.int64)  # 断言 i8 按位与 b 的结果为 64 位有符号整数类型

assert_type(u8 << u8, np.uint64)  # 断言 u8 左移 u8 的结果为 64 位无符号整数类型
assert_type(u8 >> u8, np.uint64)  # 断言 u8 右移 u8 的结果为 64 位无符号整数类型
assert_type(u8 | u8, np.uint64)  # 断言 u8 按位或 u8 的结果为 64 位无符号整数类型
assert_type(u8 ^ u8, np.uint64)  # 断言 u8 按位异或 u8 的结果为 64 位无符号整数类型
assert_type(u8 & u8, np.uint64)  # 断言 u8 按位与 u8 的结果为 64 位无符号整数类型

assert_type(u8 << AR, npt.NDArray[np.signedinteger[Any]])  # 断言 u8 左移 AR 数组的结果为特定类型的 NumPy 数组
assert_type(u8 >> AR, npt.NDArray[np.signedinteger[Any]])  # 断言 u8 右移 AR 数组的结果为特定类型的 NumPy 数组
assert_type(u8 | AR, npt.NDArray[np.signedinteger[Any]])  # 断言 u8 按位或 AR 数组的结果为特定类型的 NumPy 数组
assert_type(u8 ^ AR, npt.NDArray[np.signedinteger[Any]])  # 断言 u8 按位异或 AR 数组的结果为特定类型的 NumPy 数组
assert_type(u8 & AR, npt.NDArray[np.signedinteger[Any]])  # 断言 u8 按位与 AR 数组的结果为特定类型的 NumPy 数
# 对比两者进行左移位运算，断言结果类型为 np.int8
assert_type(b_ << b_, np.int8)
# 对比两者进行右移位运算，断言结果类型为 np.int8
assert_type(b_ >> b_, np.int8)
# 对比两者进行按位或运算，断言结果类型为 np.bool
assert_type(b_ | b_, np.bool)
# 对比两者进行按位异或运算，断言结果类型为 np.bool
assert_type(b_ ^ b_, np.bool)
# 对比两者进行按位与运算，断言结果类型为 np.bool
assert_type(b_ & b_, np.bool)

# 对比一个变量与数组元素进行左移位运算，断言结果类型为指定数组的有符号整数类型
assert_type(b_ << AR, npt.NDArray[np.signedinteger[Any]])
# 对比一个变量与数组元素进行右移位运算，断言结果类型为指定数组的有符号整数类型
assert_type(b_ >> AR, npt.NDArray[np.signedinteger[Any]])
# 对比一个变量与数组元素进行按位或运算，断言结果类型为指定数组的有符号整数类型
assert_type(b_ | AR, npt.NDArray[np.signedinteger[Any]])
# 对比一个变量与数组元素进行按位异或运算，断言结果类型为指定数组的有符号整数类型
assert_type(b_ ^ AR, npt.NDArray[np.signedinteger[Any]])
# 对比一个变量与数组元素进行按位与运算，断言结果类型为指定数组的有符号整数类型
assert_type(b_ & AR, npt.NDArray[np.signedinteger[Any]])

# 对比两者进行左移位运算，断言结果类型为 np.int8
assert_type(b_ << b, np.int8)
# 对比两者进行右移位运算，断言结果类型为 np.int8
assert_type(b_ >> b, np.int8)
# 对比两者进行按位或运算，断言结果类型为 np.bool
assert_type(b_ | b, np.bool)
# 对比两者进行按位异或运算，断言结果类型为 np.bool
assert_type(b_ ^ b, np.bool)
# 对比两者进行按位与运算，断言结果类型为 np.bool
assert_type(b_ & b, np.bool)

# 对比一个变量与整数进行左移位运算，断言结果类型为 np.int_
assert_type(b_ << i, np.int_)
# 对比一个变量与整数进行右移位运算，断言结果类型为 np.int_
assert_type(b_ >> i, np.int_)
# 对比一个变量与整数进行按位或运算，断言结果类型为 np.int_
assert_type(b_ | i, np.int_)
# 对比一个变量与整数进行按位异或运算，断言结果类型为 np.int_
assert_type(b_ ^ i, np.int_)
# 对比一个变量与整数进行按位与运算，断言结果类型为 np.int_
assert_type(b_ & i, np.int_)

# 对比取反运算，断言结果类型为 np.int64
assert_type(~i8, np.int64)
# 对比取反运算，断言结果类型为 np.int32
assert_type(~i4, np.int32)
# 对比取反运算，断言结果类型为 np.uint64
assert_type(~u8, np.uint64)
# 对比取反运算，断言结果类型为 np.uint32
assert_type(~u4, np.uint32)
# 对比取反运算，断言结果类型为 np.bool
assert_type(~b_, np.bool)
# 对比取反运算，断言结果类型为指定数组的有符号整数类型
assert_type(~AR, npt.NDArray[np.int32])
```