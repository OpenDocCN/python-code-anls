# `.\numpy\numpy\typing\tests\data\pass\arithmetic.py`

```py
from __future__ import annotations
# 引入未来的语法特性，确保代码在较老版本的 Python 中也能运行

from typing import Any, Optional
# 导入类型提示相关的模块

import numpy as np
# 导入 NumPy 库并使用别名 np

import pytest
# 导入 pytest 模块用于单元测试

c16 = np.complex128(1)
# 创建一个复数类型的 NumPy 变量 c16，精度为 128 位复数

f8 = np.float64(1)
# 创建一个浮点数类型的 NumPy 变量 f8，精度为 64 位浮点数

i8 = np.int64(1)
# 创建一个整数类型的 NumPy 变量 i8，精度为 64 位整数

u8 = np.uint64(1)
# 创建一个无符号整数类型的 NumPy 变量 u8，精度为 64 位无符号整数

c8 = np.complex64(1)
# 创建一个复数类型的 NumPy 变量 c8，精度为 64 位复数

f4 = np.float32(1)
# 创建一个浮点数类型的 NumPy 变量 f4，精度为 32 位浮点数

i4 = np.int32(1)
# 创建一个整数类型的 NumPy 变量 i4，精度为 32 位整数

u4 = np.uint32(1)
# 创建一个无符号整数类型的 NumPy 变量 u4，精度为 32 位无符号整数

dt = np.datetime64(1, "D")
# 创建一个表示日期的 NumPy 变量 dt，精度为一天

td = np.timedelta64(1, "D")
# 创建一个表示时间间隔的 NumPy 变量 td，精度为一天

b_ = np.bool(1)
# 创建一个布尔类型的 NumPy 变量 b_，值为 True

b = bool(1)
# 创建一个标准布尔类型的 Python 变量 b，值为 True

c = complex(1)
# 创建一个标准复数类型的 Python 变量 c，值为 1+0j

f = float(1)
# 创建一个标准浮点数类型的 Python 变量 f，值为 1.0

i = int(1)
# 创建一个标准整数类型的 Python 变量 i，值为 1

class Object:
    # 定义一个名为 Object 的类

    def __array__(self, dtype: Optional[np.typing.DTypeLike] = None,
                  copy: Optional[bool] = None) -> np.ndarray[Any, np.dtype[np.object_]]:
        # 定义 __array__ 方法，使对象可以被转换为 NumPy 数组
        ret = np.empty((), dtype=object)
        ret[()] = self
        return ret

    def __sub__(self, value: Any) -> Object:
        # 定义 __sub__ 方法，支持对象的减法操作
        return self

    def __rsub__(self, value: Any) -> Object:
        # 定义 __rsub__ 方法，支持减法操作的右侧反向运算
        return self

    def __floordiv__(self, value: Any) -> Object:
        # 定义 __floordiv__ 方法，支持对象的整数除法操作
        return self

    def __rfloordiv__(self, value: Any) -> Object:
        # 定义 __rfloordiv__ 方法，支持整数除法操作的右侧反向运算
        return self

    def __mul__(self, value: Any) -> Object:
        # 定义 __mul__ 方法，支持对象的乘法操作
        return self

    def __rmul__(self, value: Any) -> Object:
        # 定义 __rmul__ 方法，支持乘法操作的右侧反向运算
        return self

    def __pow__(self, value: Any) -> Object:
        # 定义 __pow__ 方法，支持对象的乘方操作
        return self

    def __rpow__(self, value: Any) -> Object:
        # 定义 __rpow__ 方法，支持乘方操作的右侧反向运算
        return self

AR_b: np.ndarray[Any, np.dtype[np.bool]] = np.array([True])
# 创建一个 NumPy 布尔类型的数组 AR_b，包含一个值为 True 的元素

AR_u: np.ndarray[Any, np.dtype[np.uint32]] = np.array([1], dtype=np.uint32)
# 创建一个 NumPy 无符号整数类型的数组 AR_u，包含一个值为 1 的元素

AR_i: np.ndarray[Any, np.dtype[np.int64]] = np.array([1])
# 创建一个 NumPy 整数类型的数组 AR_i，包含一个值为 1 的元素

AR_f: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0])
# 创建一个 NumPy 浮点数类型的数组 AR_f，包含一个值为 1.0 的元素

AR_c: np.ndarray[Any, np.dtype[np.complex128]] = np.array([1j])
# 创建一个 NumPy 复数类型的数组 AR_c，包含一个值为 1j 的元素

AR_m: np.ndarray[Any, np.dtype[np.timedelta64]] = np.array([np.timedelta64(1, "D")])
# 创建一个 NumPy 时间间隔类型的数组 AR_m，包含一个时间间隔为一天的元素

AR_M: np.ndarray[Any, np.dtype[np.datetime64]] = np.array([np.datetime64(1, "D")])
# 创建一个 NumPy 日期时间类型的数组 AR_M，包含一个日期为一天的元素

AR_O: np.ndarray[Any, np.dtype[np.object_]] = np.array([Object()])
# 创建一个 NumPy 对象类型的数组 AR_O，包含一个 Object 类的实例对象

AR_LIKE_b = [True]
# 创建一个 Python 布尔类型的列表 AR_LIKE_b，包含一个值为 True 的元素

AR_LIKE_u = [np.uint32(1)]
# 创建一个 Python 无符号整数类型的列表 AR_LIKE_u，包含一个值为 1 的元素

AR_LIKE_i = [1]
# 创建一个 Python 整数类型的列表 AR_LIKE_i，包含一个值为 1 的元素

AR_LIKE_f = [1.0]
# 创建一个 Python 浮点数类型的列表 AR_LIKE_f，包含一个值为 1.0 的元素

AR_LIKE_c = [1j]
# 创建一个 Python 复数类型的列表 AR_LIKE_c，包含一个值为 1j 的元素

AR_LIKE_m = [np.timedelta64(1, "D")]
# 创建一个 Python 时间间隔类型的列表 AR_LIKE_m，包含一个时间间隔为一天的元素

AR_LIKE_M = [np.datetime64(1, "D")]
# 创建一个 Python 日期时间类型的列表 AR_LIKE_M，包含一个日期为一天的元素

AR_LIKE_O = [Object()]
# 创建一个 Python 对象类型的列表 AR_LIKE_O，包含一个 Object 类的实例对象

# Array subtractions

AR_b - AR_LIKE_u
# 对 AR_b 和 AR_LIKE_u 执行数组减法操作

AR_b - AR_LIKE_i
# 对 AR_b 和 AR_LIKE_i 执行数组减法操作

AR_b - AR_LIKE_f
# 对 AR_b 和 AR_LIKE_f 执行数组减法操作

AR_b - AR_LIKE_c
# 对 AR_b 和 AR_LIKE_c 执行数组减法操作

AR_b - AR_LIKE_m
# 对 AR_b 和 AR_LIKE_m 执行数组减法操作

AR_b - AR_LIKE_O
# 对 AR_b 和 AR_LIKE_O 执行数组减法操作

AR_LIKE_u - AR_b
# 对 AR_LIKE_u 和 AR_b 执行数组减法操作

AR_LIKE_i - AR_b
# 对 AR_LIKE_i 和 AR_b 执行数组减法操作

AR_LIKE_f - AR_b
# 对 AR_LIKE_f 和 AR_b 执行数组减法操作

AR_LIKE_c - AR_b
# 对 AR_LIKE_c 和 AR_b 执行数组减法操作

AR_LIKE_m - AR_b
# 对 AR_LIKE_m 和 AR_b 执行数组减法操作

AR_LIKE_M - AR_b
# 对 AR_LIKE_M 和 AR_b 执行数组减法操作

AR_LIKE_O - AR_b
# 对 AR_LIKE
# Subtract AR_c from AR_LIKE_b
AR_LIKE_b - AR_c
# Subtract AR_c from AR_LIKE_u
AR_LIKE_u - AR_c
# Subtract AR_c from AR_LIKE_i
AR_LIKE_i - AR_c
# Subtract AR_c from AR_LIKE_f
AR_LIKE_f - AR_c
# Subtract AR_c from AR_LIKE_c
AR_LIKE_c - AR_c
# Subtract AR_c from AR_LIKE_O
AR_LIKE_O - AR_c

# Subtract AR_LIKE_b from AR_m
AR_m - AR_LIKE_b
# Subtract AR_LIKE_u from AR_m
AR_m - AR_LIKE_u
# Subtract AR_LIKE_i from AR_m
AR_m - AR_LIKE_i
# Subtract AR_LIKE_m from AR_m
AR_m - AR_LIKE_m

# Subtract AR_m from AR_LIKE_b
AR_LIKE_b - AR_m
# Subtract AR_m from AR_LIKE_u
AR_LIKE_u - AR_m
# Subtract AR_m from AR_LIKE_i
AR_LIKE_i - AR_m
# Subtract AR_m from AR_LIKE_m
AR_LIKE_m - AR_m
# Subtract AR_m from AR_LIKE_M
AR_LIKE_M - AR_m

# Subtract AR_LIKE_b from AR_M
AR_M - AR_LIKE_b
# Subtract AR_LIKE_u from AR_M
AR_M - AR_LIKE_u
# Subtract AR_LIKE_i from AR_M
AR_M - AR_LIKE_i
# Subtract AR_LIKE_m from AR_M
AR_M - AR_LIKE_m
# Subtract AR_LIKE_M from AR_M
AR_M - AR_LIKE_M

# Subtract AR_M from AR_LIKE_b
AR_LIKE_b - AR_M
# Subtract AR_M from AR_LIKE_u
AR_LIKE_u - AR_M
# Subtract AR_M from AR_LIKE_i
AR_LIKE_i - AR_M
# Subtract AR_M from AR_LIKE_m
AR_LIKE_m - AR_M
# Subtract AR_M from AR_LIKE_M
AR_LIKE_M - AR_M

# Subtract AR_M from AR_LIKE_M
AR_LIKE_M - AR_M

# Subtract AR_LIKE_b from AR_O
AR_O - AR_LIKE_b
# Subtract AR_LIKE_u from AR_O
AR_O - AR_LIKE_u
# Subtract AR_LIKE_i from AR_O
AR_O - AR_LIKE_i
# Subtract AR_LIKE_f from AR_O
AR_O - AR_LIKE_f
# Subtract AR_LIKE_c from AR_O
AR_O - AR_LIKE_c
# Subtract AR_LIKE_O from AR_O
AR_O - AR_LIKE_O

# Subtract AR_O from AR_LIKE_b
AR_LIKE_b - AR_O
# Subtract AR_O from AR_LIKE_u
AR_LIKE_u - AR_O
# Subtract AR_O from AR_LIKE_i
AR_LIKE_i - AR_O
# Subtract AR_O from AR_LIKE_f
AR_LIKE_f - AR_O
# Subtract AR_O from AR_LIKE_c
AR_LIKE_c - AR_O
# Subtract AR_O from AR_LIKE_O
AR_LIKE_O - AR_O

# Increment AR_u by AR_b
AR_u += AR_b
# Increment AR_u by AR_u
AR_u += AR_u
# Increment AR_u by 1 (allowed during runtime as long as AR_u is 0D and >= 0)

# Perform floor division of AR_b by AR_LIKE_b
AR_b // AR_LIKE_b
# Perform floor division of AR_b by AR_LIKE_u
AR_b // AR_LIKE_u
# Perform floor division of AR_b by AR_LIKE_i
AR_b // AR_LIKE_i
# Perform floor division of AR_b by AR_LIKE_f
AR_b // AR_LIKE_f
# Perform floor division of AR_b by AR_LIKE_O
AR_b // AR_LIKE_O

# Perform floor division of AR_LIKE_b by AR_b
AR_LIKE_b // AR_b
# Perform floor division of AR_LIKE_u by AR_b
AR_LIKE_u // AR_b
# Perform floor division of AR_LIKE_i by AR_b
AR_LIKE_i // AR_b
# Perform floor division of AR_LIKE_f by AR_b
AR_LIKE_f // AR_b
# Perform floor division of AR_LIKE_O by AR_b
AR_LIKE_O // AR_b

# Perform floor division of AR_u by AR_LIKE_b
AR_u // AR_LIKE_b
# Perform floor division of AR_u by AR_LIKE_u
AR_u // AR_LIKE_u
# Perform floor division of AR_u by AR_LIKE_i
AR_u // AR_LIKE_i
# Perform floor division of AR_u by AR_LIKE_f
AR_u // AR_LIKE_f
# Perform floor division of AR_u by AR_LIKE_O
AR_u // AR_LIKE_O

# Perform floor division of AR_LIKE_b by AR_u
AR_LIKE_b // AR_u
# Perform floor division of AR_LIKE_u by AR_u
AR_LIKE_u // AR_u
# Perform floor division of AR_LIKE_i by AR_u
AR_LIKE_i // AR_u
# Perform floor division of AR_LIKE_f by AR_u
AR_LIKE_f // AR_u
# Perform floor division of AR_LIKE_m by AR_u
AR_LIKE_m // AR_u
# Perform floor division of AR_LIKE_O by AR_u
AR_LIKE_O // AR_u

# Perform floor division of AR_i by AR_LIKE_b
AR_i // AR_LIKE_b
# Perform floor division of AR_i by AR_LIKE_u
AR_i // AR_LIKE_u
# Perform floor division of AR_i by AR_LIKE_i
AR_i // AR_LIKE_i
# Perform floor division of AR_i by AR_LIKE_f
AR_i // AR_LIKE_f
# Perform floor division of AR_i by AR_LIKE_O
AR_i // AR_LIKE_O

# Perform floor division of AR_LIKE_b by AR_i
AR_LIKE_b // AR_i
# Perform floor division of AR_LIKE_u by AR_i
AR_LIKE_u // AR_i
# Perform floor division of AR_LIKE_i by AR_i
AR_LIKE_i // AR_i
# Perform floor division of AR_LIKE_f by AR_i
AR_LIKE_f // AR_i
# Perform floor division of AR_LIKE_m by AR_i
AR_LIKE_m // AR_i
# Perform floor division of AR_LIKE_O by AR_i
AR_LIKE_O // AR_i

# Perform floor division of AR_f by AR_LIKE_b
AR_f // AR_LIKE_b
# Perform floor division of AR_f by AR_LIKE_u
AR_f // AR_LIKE_u
# Perform floor division of AR_f by AR_LIKE_i
AR_f // AR_LIKE_i
# Perform floor division of AR_f by AR_LIKE_f
AR_f // AR_LIKE_f
# Perform floor division of AR_f by AR_LIKE_O
AR_f // AR_LIKE_O

# Perform floor division of AR_LIKE_b by AR_f
AR_LIKE_b // AR_f
# Perform floor division of AR_LIKE_u by AR_f
AR_LIKE_u // AR_f
# Perform floor division of AR_LIKE_i by AR_f
AR_LIKE_i // AR_f
# Perform floor division of AR_LIKE_f by AR_f
AR_LIKE_f // AR_f
# Perform floor division of AR_LIKE_m by AR_f
AR_LIKE_m // AR_f
# Perform floor division of AR_LIKE_O by AR_f
AR_LIKE_O // AR_f

# Perform floor division of AR_m by AR_LIKE_u
AR_m // AR_LIKE_u
# Perform floor division of AR_m by AR_LIKE_i
AR_m // AR_LIKE_i
# Perform floor division of AR_m by AR_LIKE_f
AR_m // AR_LIKE_f
# Perform floor division of AR_m by AR_LIKE_m
AR_m // AR_LIKE_m

# Perform floor division of AR_LIKE_m by AR_m
AR_LIKE_m // AR_m

# Perform floor division of AR_O by AR_LIKE_b
AR_O // AR_LIKE_b
# Perform floor division of AR_O by AR_LIKE_u
AR_O // AR_LIKE_u
# Perform floor division of AR_O by AR_LIKE_i
AR_O // AR_LIKE_i
# Perform floor division of AR_O by AR_LIKE_f
AR_O // AR_LIKE_f
# Perform floor division of AR_O by AR_LIKE_O
AR_O // AR_LIKE_O

# Perform floor division of AR_LIKE_b by AR_O
AR_LIKE_b // AR_O
# Perform floor division of AR_LIKE_u by AR_O
AR_LIKE_u // AR_O
# Perform floor division of AR_LIKE_i by AR_O
AR_LIKE_i // AR_O
# Perform floor division of AR_LIKE_f by AR_O
AR_LIKE_f // AR_O
# Perform floor division of AR_LIKE_O by AR_O
AR_LIKE_O // AR_O

# Inplace multiplication of AR_LIKE_b by AR_b
AR_b *= AR_LIKE_b

# Inplace multiplication of AR_LIKE_b by AR_u
AR_u *= AR_LIKE_b
# Inplace multiplication of AR_LIKE_u by AR_u
AR_u *= AR_LIKE_u

# Inplace multiplication of AR_LIKE_b by AR_i
AR_i *= AR_LIKE_b
# Inplace multiplication of
# boolean

# 布尔类型之间的除法操作，结果为浮点数
b_ / b
b_ / b_
b_ / i
b_ / i8
b_ / i4
b_ / u8
b_ / u4
b_ / f
b_ / f8
b_ / f4
b_ / c
b_ / c16
b_ / c8

# 整数除以布尔类型，结果为整数
b / b_
b_ / b_
i / b_
i8 / b_
i4 / b_
u8 / b_
u4 / b_
f / b_
f8 / b_
f4 / b_
c / b_
c16 / b_
c8 / b_

# Complex

# 复数类型 c16 与各种类型的加法操作
c16 + c16
c16 + f8
c16 + i8
c16 + c8
c16 + f4
c16 + i4
c16 + b_
c16 + b
c16 + c
c16 + f
c16 + i
c16 + AR_f

# 各种类型与复数类型 c16 的加法操作
c16 + c16
f8 + c16
i8 + c16
c8 + c16
f4 + c16
i4 + c16
b_ + c16
b + c16
c + c16
f + c16
i + c16
AR_f + c16

c8 + c16
c8 + f8
c8 + i8
c8 + c8
c8 + f4
c8 + i4
c8 + b_
c8 + b
c8 + c
c8 + f
c8 + i
c8 + AR_f

c16 + c8
f8 + c8
i8 + c8
c8 + c8
f4 + c8
i4 + c8
b_ + c8
b + c8
c + c8
f + c8
i + c8
AR_f + c8

# Float

# 浮点类型 f8 与各种类型的加法操作
f8 + f8
f8 + i8
f8 + f4
f8 + i4
f8 + b_
f8 + b
f8 + c
f8 + f
f8 + i
f8 + AR_f

# 各种类型与浮点类型 f8 的加法操作
f8 + f8
i8 + f8
f4 + f8
i4 + f8
b_ + f8
b + f8
c + f8
f + f8
i + f8
AR_f + f8

f4 + f8
f4 + i8
f4 + f4
f4 + i4
f4 + b_
f4 + b
f4 + c
f4 + f
f4 + i
f4 + AR_f

f8 + f4
i8 + f4
f4 + f4
i4 + f4
b_ + f4
b + f4
c + f4
f + f4
i + f4
AR_f + f4

# Int

# 整数类型 i8 与各种类型的加法操作
i8 + i8
i8 + u8
i8 + i4
i8 + u4
i8 + b_
i8 + b
i8 + c
i8 + f
i8 + i
i8 + AR_f

u8 + u8
u8 + i4
u8 + u4
u8 + b_
u8 + b
u8 + c
u8 + f
u8 + i
u8 + AR_f

i8 + i8
u8 + i8
i4 + i8
u4 + i8
b_ + i8
b + i8
c + i8
f + i8
i + i8
AR_f + i8

u8 + u8
i4 + u8
u4 + u8
b_ + u8
b + u8
c + u8
f + u8
i + u8
AR_f + u8

i4 + i8
i4 + i4
i4 + i
i4 + b_
i4 + b
i4 + AR_f

u4 + i8
u4 + i4
u4 + u8
u4 + u4
u4 + i
u4 + b_
u4 + b
u4 + AR_f

i8 + i4
i4 + i4
i + i4
b_ + i4
b + i4
AR_f + i4

i8 + u4
i4 + u4
u8 + u4
u4 + u4
b_ + u4
b + u4
i + u4
AR_f + u4
```