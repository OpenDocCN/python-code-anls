# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\arithmetic.pyi`

```py
from typing import Any

import numpy as np
import numpy.typing as npt

b_ = np.bool()
dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

AR_b: npt.NDArray[np.bool]    # 定义一个布尔类型的 NumPy 数组
AR_u: npt.NDArray[np.uint32]  # 定义一个无符号 32 位整数类型的 NumPy 数组
AR_i: npt.NDArray[np.int64]   # 定义一个 64 位整数类型的 NumPy 数组
AR_f: npt.NDArray[np.float64] # 定义一个双精度浮点数类型的 NumPy 数组
AR_c: npt.NDArray[np.complex128]  # 定义一个复数类型的 NumPy 数组
AR_m: npt.NDArray[np.timedelta64]  # 定义一个时间差类型的 NumPy 数组
AR_M: npt.NDArray[np.datetime64]   # 定义一个日期时间类型的 NumPy 数组

ANY: Any  # 任意类型的对象

AR_LIKE_b: list[bool]               # 布尔类型的列表
AR_LIKE_u: list[np.uint32]          # 无符号 32 位整数类型的列表
AR_LIKE_i: list[int]                # 整数类型的列表
AR_LIKE_f: list[float]              # 浮点数类型的列表
AR_LIKE_c: list[complex]            # 复数类型的列表
AR_LIKE_m: list[np.timedelta64]     # 时间差类型的列表
AR_LIKE_M: list[np.datetime64]      # 日期时间类型的列表

# 数组减法操作

# 注意：mypys `NoReturn` 错误报告不是很好
_1 = AR_b - AR_LIKE_b  # E: 需要类型注解
_2 = AR_LIKE_b - AR_b  # E: 需要类型注解
AR_i - bytes()  # E: 没有匹配的重载变量

AR_f - AR_LIKE_m  # E: 不支持的操作数类型
AR_f - AR_LIKE_M  # E: 不支持的操作数类型
AR_c - AR_LIKE_m  # E: 不支持的操作数类型
AR_c - AR_LIKE_M  # E: 不支持的操作数类型

AR_m - AR_LIKE_f  # E: 不支持的操作数类型
AR_M - AR_LIKE_f  # E: 不支持的操作数类型
AR_m - AR_LIKE_c  # E: 不支持的操作数类型
AR_M - AR_LIKE_c  # E: 不支持的操作数类型

AR_m - AR_LIKE_M  # E: 不支持的操作数类型
AR_LIKE_m - AR_M  # E: 不支持的操作数类型

# 数组整除操作

AR_M // AR_LIKE_b  # E: 不支持的操作数类型
AR_M // AR_LIKE_u  # E: 不支持的操作数类型
AR_M // AR_LIKE_i  # E: 不支持的操作数类型
AR_M // AR_LIKE_f  # E: 不支持的操作数类型
AR_M // AR_LIKE_c  # E: 不支持的操作数类型
AR_M // AR_LIKE_m  # E: 不支持的操作数类型
AR_M // AR_LIKE_M  # E: 不支持的操作数类型

AR_b // AR_LIKE_M  # E: 不支持的操作数类型
AR_u // AR_LIKE_M  # E: 不支持的操作数类型
AR_i // AR_LIKE_M  # E: 不支持的操作数类型
AR_f // AR_LIKE_M  # E: 不支持的操作数类型
AR_c // AR_LIKE_M  # E: 不支持的操作数类型
AR_m // AR_LIKE_M  # E: 不支持的操作数类型
AR_M // AR_LIKE_M  # E: 不支持的操作数类型

_3 = AR_m // AR_LIKE_b  # E: 需要类型注解
AR_m // AR_LIKE_c  # E: 不支持的操作数类型

AR_b // AR_LIKE_m  # E: 不支持的操作数类型
AR_u // AR_LIKE_m  # E: 不支持的操作数类型
AR_i // AR_LIKE_m  # E: 不支持的操作数类型
AR_f // AR_LIKE_m  # E: 不支持的操作数类型
AR_c // AR_LIKE_m  # E: 不支持的操作数类型

# 数组乘法操作

AR_b *= AR_LIKE_u  # E: 不兼容的类型
AR_b *= AR_LIKE_i  # E: 不兼容的类型
AR_b *= AR_LIKE_f  # E: 不兼容的类型
AR_b *= AR_LIKE_c  # E: 不兼容的类型
AR_b *= AR_LIKE_m  # E: 不兼容的类型

AR_u *= AR_LIKE_i  # E: 不兼容的类型
AR_u *= AR_LIKE_f  # E: 不兼容的类型
AR_u *= AR_LIKE_c  # E: 不兼容的类型
AR_u *= AR_LIKE_m  # E: 不兼容的类型

AR_i *= AR_LIKE_f  # E: 不兼容的类型
AR_i *= AR_LIKE_c  # E: 不兼容的类型
AR_i *= AR_LIKE_m  # E: 不兼容的类型

AR_f *= AR_LIKE_c  # E: 不兼容的类型
AR_f *= AR_LIKE_m  # E: 不兼容的类型

# 数组幂操作
# 对 AR_b 进行自乘赋值操作，使用 AR_LIKE_b 作为指数。可能出现的错误是“Invalid self argument”。
AR_b **= AR_LIKE_b  # E: Invalid self argument

# 对 AR_b 进行自乘赋值操作，使用 AR_LIKE_u 作为指数。可能出现的错误是“Invalid self argument”。
AR_b **= AR_LIKE_u  # E: Invalid self argument

# 对 AR_b 进行自乘赋值操作，使用 AR_LIKE_i 作为指数。可能出现的错误是“Invalid self argument”。
AR_b **= AR_LIKE_i  # E: Invalid self argument

# 对 AR_b 进行自乘赋值操作，使用 AR_LIKE_f 作为指数。可能出现的错误是“Invalid self argument”。
AR_b **= AR_LIKE_f  # E: Invalid self argument

# 对 AR_b 进行自乘赋值操作，使用 AR_LIKE_c 作为指数。可能出现的错误是“Invalid self argument”。
AR_b **= AR_LIKE_c  # E: Invalid self argument

# 对 AR_u 进行自乘赋值操作，使用 AR_LIKE_i 作为指数。可能出现的错误是“incompatible type”。
AR_u **= AR_LIKE_i  # E: incompatible type

# 对 AR_u 进行自乘赋值操作，使用 AR_LIKE_f 作为指数。可能出现的错误是“incompatible type”。
AR_u **= AR_LIKE_f  # E: incompatible type

# 对 AR_u 进行自乘赋值操作，使用 AR_LIKE_c 作为指数。可能出现的错误是“incompatible type”。
AR_u **= AR_LIKE_c  # E: incompatible type

# 对 AR_i 进行自乘赋值操作，使用 AR_LIKE_f 作为指数。可能出现的错误是“incompatible type”。
AR_i **= AR_LIKE_f  # E: incompatible type

# 对 AR_i 进行自乘赋值操作，使用 AR_LIKE_c 作为指数。可能出现的错误是“incompatible type”。
AR_i **= AR_LIKE_c  # E: incompatible type

# 对 AR_f 进行自乘赋值操作，使用 AR_LIKE_c 作为指数。可能出现的错误是“incompatible type”。
AR_f **= AR_LIKE_c  # E: incompatible type

# 对 b_ 进行减法，但没有给定操作数。可能出现的错误是“No overload variant”。
b_ - b_  # E: No overload variant

# 对 dt 变量进行加法，但加法操作不支持给定的操作数类型。可能出现的错误是“Unsupported operand types”。
dt + dt  # E: Unsupported operand types

# 对 td 变量进行减法，但减法操作不支持给定的操作数类型。可能出现的错误是“Unsupported operand types”。
td - dt  # E: Unsupported operand types

# 对 td 变量进行取模操作，但取模操作不支持给定的操作数类型（例如，1 是整数，td 可能是时间间隔类型）。可能出现的错误是“Unsupported operand types”。
td % 1  # E: Unsupported operand types

# 对 td 变量进行除法，但除法操作没有重载支持给定的操作数类型。可能出现的错误是“No overload”。
td / dt  # E: No overload

# 对 td 变量进行取模操作，但取模操作不支持给定的操作数类型。可能出现的错误是“Unsupported operand types”。
td % dt  # E: Unsupported operand types

# 对 b_ 变量进行一元负号操作，但操作的类型不支持给定的操作数类型。可能出现的错误是“Unsupported operand type”。
-b_  # E: Unsupported operand type

# 对 b_ 变量进行一元正号操作，但操作的类型不支持给定的操作数类型。可能出现的错误是“Unsupported operand type”。
+b_  # E: Unsupported operand type
```