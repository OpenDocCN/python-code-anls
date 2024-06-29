# `.\numpy\numpy\typing\tests\data\pass\comparisons.py`

```py
from __future__ import annotations  # 导入 future 模块中的 annotations 特性，用于支持注解的类型提示

from typing import Any  # 导入 Any 类型，表示可以是任意类型
import numpy as np  # 导入 NumPy 库并重命名为 np

c16 = np.complex128()  # 创建一个复数类型的变量 c16，精度为 complex128
f8 = np.float64()  # 创建一个浮点数类型的变量 f8，精度为 float64
i8 = np.int64()  # 创建一个整数类型的变量 i8，精度为 int64
u8 = np.uint64()  # 创建一个无符号整数类型的变量 u8，精度为 uint64

c8 = np.complex64()  # 创建一个复数类型的变量 c8，精度为 complex64
f4 = np.float32()  # 创建一个浮点数类型的变量 f4，精度为 float32
i4 = np.int32()  # 创建一个整数类型的变量 i4，精度为 int32
u4 = np.uint32()  # 创建一个无符号整数类型的变量 u4，精度为 uint32

dt = np.datetime64(0, "D")  # 创建一个日期时间类型的变量 dt，初始化为 0 天
td = np.timedelta64(0, "D")  # 创建一个时间差类型的变量 td，初始化为 0 天

b_ = np.bool()  # 创建一个布尔类型的变量 b_，默认初始化为 False

b = bool()  # 创建一个布尔类型的变量 b，Python 内置类型，默认初始化为 False
c = complex()  # 创建一个复数类型的变量 c，Python 内置类型，默认初始化为 0+0j
f = float()  # 创建一个浮点数类型的变量 f，Python 内置类型，默认初始化为 0.0
i = int()  # 创建一个整数类型的变量 i，Python 内置类型，默认初始化为 0

SEQ = (0, 1, 2, 3, 4)  # 创建一个元组 SEQ，包含整数 0 到 4

AR_b: np.ndarray[Any, np.dtype[np.bool]] = np.array([True])  # 创建一个 NumPy 布尔数组 AR_b，包含一个 True 值
AR_u: np.ndarray[Any, np.dtype[np.uint32]] = np.array([1], dtype=np.uint32)  # 创建一个 NumPy 无符号整数数组 AR_u，包含一个值为 1 的 uint32
AR_i: np.ndarray[Any, np.dtype[np.int_]] = np.array([1])  # 创建一个 NumPy 整数数组 AR_i，包含一个值为 1 的 int_
AR_f: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0])  # 创建一个 NumPy 浮点数数组 AR_f，包含一个值为 1.0 的 float64
AR_c: np.ndarray[Any, np.dtype[np.complex128]] = np.array([1.0j])  # 创建一个 NumPy 复数数组 AR_c，包含一个值为 1.0j 的 complex128
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]] = np.array([np.timedelta64("1")])  # 创建一个 NumPy 时间差数组 AR_m，包含一个值为 1 的 timedelta64
AR_M: np.ndarray[Any, np.dtype[np.datetime64]] = np.array([np.datetime64("1")])  # 创建一个 NumPy 日期时间数组 AR_M，包含一个值为 1 的 datetime64
AR_O: np.ndarray[Any, np.dtype[np.object_]] = np.array([1], dtype=object)  # 创建一个 NumPy 对象数组 AR_O，包含一个值为 1 的 object

# Arrays

AR_b > AR_b  # 比较两个布尔数组 AR_b 的元素是否逐个大于
AR_b > AR_u  # 比较布尔数组 AR_b 和无符号整数数组 AR_u 的元素是否逐个大于
AR_b > AR_i  # 比较布尔数组 AR_b 和整数数组 AR_i 的元素是否逐个大于
AR_b > AR_f  # 比较布尔数组 AR_b 和浮点数数组 AR_f 的元素是否逐个大于
AR_b > AR_c  # 比较布尔数组 AR_b 和复数数组 AR_c 的元素是否逐个大于

AR_u > AR_b  # 比较无符号整数数组 AR_u 和布尔数组 AR_b 的元素是否逐个大于
AR_u > AR_u  # 比较两个无符号整数数组 AR_u 的元素是否逐个大于
AR_u > AR_i  # 比较无符号整数数组 AR_u 和整数数组 AR_i 的元素是否逐个大于
AR_u > AR_f  # 比较无符号整数数组 AR_u 和浮点数数组 AR_f 的元素是否逐个大于
AR_u > AR_c  # 比较无符号整数数组 AR_u 和复数数组 AR_c 的元素是否逐个大于

AR_i > AR_b  # 比较整数数组 AR_i 和布尔数组 AR_b 的元素是否逐个大于
AR_i > AR_u  # 比较整数数组 AR_i 和无符号整数数组 AR_u 的元素是否逐个大于
AR_i > AR_i  # 比较两个整数数组 AR_i 的元素是否逐个大于
AR_i > AR_f  # 比较整数数组 AR_i 和浮点数数组 AR_f 的元素是否逐个大于
AR_i > AR_c  # 比较整数数组 AR_i 和复数数组 AR_c 的元素是否逐个大于

AR_f > AR_b  # 比较浮点数数组 AR_f 和布尔数组 AR_b 的元素是否逐个大于
AR_f > AR_u  # 比较浮点数数组 AR_f 和无符号整数数组 AR_u 的元素是否逐个大于
AR_f > AR_i  # 比较浮点数数组 AR_f 和整数数组 AR_i 的元素是否逐个大于
AR_f > AR_f  # 比较两个浮点数数组 AR_f 的元素是否逐个大于
AR_f > AR_c  # 比较浮点数数组 AR_f 和复数数组 AR_c 的元素是否逐个大于

AR_c > AR_b  # 比较复数数组 AR_c 和布尔数组 AR_b 的元素是否逐个大于
AR_c > AR_u  # 比较复数数组 AR_c 和无符号整数数组 AR_u 的元素是否逐个大于
AR_c > AR_i  # 比较复数数组 AR_c 和整数数组 AR_i 的元素是否逐个大于
AR_c > AR_f  # 比较复数数组 AR_c 和浮点数数组 AR_f 的元素是否逐个大于
AR_c > AR_c  # 比较两个复数数组 AR_c 的元素是否逐个大于

AR_m > AR_b  # 比较时间差数组 AR_m 和布尔数组 AR_b 的元素是否逐个大于
AR_m > AR_u  # 比较时间差数组 AR_m 和无符号整数数组 AR_u 的元素是否逐个大于
AR_m > AR_i  # 比较时间差数组 AR_m 和整数数组 AR_i 的元素是否逐个大于
AR_b > AR_m  # 比较布尔数组 AR_b 和时间差数组 AR_m 的元素是否逐个大于
AR_u > AR_m  # 比较无符号整数数组 AR_u 和时间差数组 AR_m 的元素是否逐个大于
AR_i > AR_m  # 比较整数数组 AR_i 和时间差数组 AR_m 的元素是否逐个大于

AR_M > AR_M  # 比较两个日期时间数组 AR_M 的元素是否逐个大于

AR_O > AR_O  # 比较两个对象数组 AR_O 的元素是否逐个大于
1 > AR_O  # 比较整数 1 和
```