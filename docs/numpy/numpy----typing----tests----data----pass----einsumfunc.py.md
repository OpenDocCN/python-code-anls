# `.\numpy\numpy\typing\tests\data\pass\einsumfunc.py`

```py
from __future__ import annotations

from typing import Any

import numpy as np

# 声明一些类似数组的对象
AR_LIKE_b = [True, True, True]       # 布尔类型的数组样式对象
AR_LIKE_u = [np.uint32(1), np.uint32(2), np.uint32(3)]   # 无符号整数类型的数组样式对象
AR_LIKE_i = [1, 2, 3]                # 整数类型的数组样式对象
AR_LIKE_f = [1.0, 2.0, 3.0]           # 浮点数类型的数组样式对象
AR_LIKE_c = [1j, 2j, 3j]             # 复数类型的数组样式对象
AR_LIKE_U = ["1", "2", "3"]          # Unicode 字符串类型的数组样式对象

# 创建一个形状为 (3,)、数据类型为 np.float64 的空数组对象
OUT_f: np.ndarray[Any, np.dtype[np.float64]] = np.empty(3, dtype=np.float64)
# 创建一个形状为 (3,)、数据类型为 np.complex128 的空数组对象
OUT_c: np.ndarray[Any, np.dtype[np.complex128]] = np.empty(3, dtype=np.complex128)

# 使用 einsum 函数进行张量运算，以下是一系列示例
np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b)      # 对布尔类型数组样式对象进行元素对应乘法
np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u)      # 对无符号整数类型数组样式对象进行元素对应乘法
np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i)      # 对整数类型数组样式对象进行元素对应乘法
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f)      # 对浮点数类型数组样式对象进行元素对应乘法
np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c)      # 对复数类型数组样式对象进行元素对应乘法
np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i)      # 对布尔类型和整数类型数组样式对象进行元素对应乘法
np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)   # 复杂的 einsum 操作

# 使用 einsum 函数指定输出的数据类型或者输出数组对象
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16")   # 指定输出的数据类型为复数类型
np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe")   # 指定输出的数据类型为布尔类型，不进行安全检查
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, out=OUT_c)     # 将结果存储到预先分配的复数类型数组对象中
np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=int, casting="unsafe", out=OUT_f)   # 将结果存储到预先分配的浮点数类型数组对象中

# 返回与指定的 einsum 操作相关的优化路径
np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b)      # 返回布尔类型数组样式对象的优化路径
np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u)      # 返回无符号整数类型数组样式对象的优化路径
np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i)      # 返回整数类型数组样式对象的优化路径
np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f)      # 返回浮点数类型数组样式对象的优化路径
np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c)      # 返回复数类型数组样式对象的优化路径
np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i)      # 返回布尔类型和整数类型数组样式对象的优化路径
np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)   # 返回复杂 einsum 操作的优化路径
```