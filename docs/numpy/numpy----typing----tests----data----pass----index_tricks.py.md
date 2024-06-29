# `.\numpy\numpy\typing\tests\data\pass\index_tricks.py`

```py
# 导入未来的注释，允许使用类型注释的特性
from __future__ import annotations
# 导入 Any 类型，表示可以是任何类型
from typing import Any
# 导入 NumPy 库，并简写为 np
import numpy as np

# 定义一个布尔类型的二维列表
AR_LIKE_b = [[True, True], [True, True]]
# 定义一个整数类型的二维列表
AR_LIKE_i = [[1, 2], [3, 4]]
# 定义一个浮点数类型的二维列表
AR_LIKE_f = [[1.0, 2.0], [3.0, 4.0]]
# 定义一个字符串类型的二维列表
AR_LIKE_U = [["1", "2"], ["3", "4"]]

# 创建一个整数类型的 NumPy 数组 AR_i8，使用 AR_LIKE_i 作为数据源，指定数据类型为 np.int64
AR_i8: np.ndarray[Any, np.dtype[np.int64]] = np.array(AR_LIKE_i, dtype=np.int64)

# 调用 np.ndenumerate 函数，返回 AR_i8 的枚举对象
np.ndenumerate(AR_i8)
# 调用 np.ndenumerate 函数，返回 AR_LIKE_f 的枚举对象
np.ndenumerate(AR_LIKE_f)
# 调用 np.ndenumerate 函数，返回 AR_LIKE_U 的枚举对象
np.ndenumerate(AR_LIKE_U)

# 获取 AR_i8 的枚举对象并访问其 iter 属性
np.ndenumerate(AR_i8).iter
# 获取 AR_LIKE_f 的枚举对象并访问其 iter 属性
np.ndenumerate(AR_LIKE_f).iter
# 获取 AR_LIKE_U 的枚举对象并访问其 iter 属性
np.ndenumerate(AR_LIKE_U).iter

# 调用 next 函数获取 AR_i8 的枚举对象的下一个元素
next(np.ndenumerate(AR_i8))
# 调用 next 函数获取 AR_LIKE_f 的枚举对象的下一个元素
next(np.ndenumerate(AR_LIKE_f))
# 调用 next 函数获取 AR_LIKE_U 的枚举对象的下一个元素

next(np.ndenumerate(AR_LIKE_U))

# 调用 iter 函数获取 np.ndenumerate(AR_i8) 的可迭代对象
iter(np.ndenumerate(AR_i8))
# 调用 iter 函数获取 np.ndenumerate(AR_LIKE_f) 的可迭代对象
iter(np.ndenumerate(AR_LIKE_f))
# 调用 iter 函数获取 np.ndenumerate(AR_LIKE_U) 的可迭代对象
iter(np.ndenumerate(AR_LIKE_U))

# 调用 iter 函数生成一个 np.ndindex 对象，表示多维数组的索引
iter(np.ndindex(1, 2, 3))
# 调用 next 函数获取 np.ndindex(1, 2, 3) 的下一个元素
next(np.ndindex(1, 2, 3))

# 使用 np.unravel_index 函数根据索引值和形状获取多维数组的坐标
np.unravel_index([22, 41, 37], (7, 6))
# 使用 np.unravel_index 函数根据索引值和形状以及指定的顺序获取多维数组的坐标
np.unravel_index([31, 41, 13], (7, 6), order='F')
# 使用 np.unravel_index 函数根据线性索引和形状获取多维数组的坐标
np.unravel_index(1621, (6, 7, 8, 9))

# 使用 np.ravel_multi_index 函数根据多维数组的坐标和形状获取线性索引
np.ravel_multi_index(AR_LIKE_i, (7, 6))
# 使用 np.ravel_multi_index 函数根据多维数组的坐标、形状和指定的顺序获取线性索引
np.ravel_multi_index(AR_LIKE_i, (7, 6), order='F')
# 使用 np.ravel_multi_index 函数根据多维数组的坐标、形状和指定的模式获取线性索引
np.ravel_multi_index(AR_LIKE_i, (4, 6), mode='clip')
# 使用 np.ravel_multi_index 函数根据多维数组的坐标、形状和指定的模式获取线性索引
np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=('clip', 'wrap'))
# 使用 np.ravel_multi_index 函数根据指定的坐标和形状获取线性索引
np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9))

# 使用 np.mgrid 函数创建多维网格
np.mgrid[1:1:2]
np.mgrid[1:1:2, None:10]

# 使用 np.ogrid 函数创建多维开放网格
np.ogrid[1:1:2]
np.ogrid[1:1:2, None:10]

# 使用 np.index_exp 函数创建扩展的索引
np.index_exp[0:1]
np.index_exp[0:1, None:3]
np.index_exp[0, 0:1, ..., [0, 1, 3]]

# 使用 np.s_ 函数创建扩展的切片对象
np.s_[0:1]
np.s_[0:1, None:3]
np.s_[0, 0:1, ..., [0, 1, 3]]

# 使用 np.ix_ 函数根据指定的数组创建广播后的索引器
np.ix_(AR_LIKE_b[0])
np.ix_(AR_LIKE_i[0], AR_LIKE_f[0])
np.ix_(AR_i8[0])

# 使用 np.fill_diagonal 函数将数组的对角线填充为指定的值
np.fill_diagonal(AR_i8, 5)

# 使用 np.diag_indices 函数获取指定大小的对角线索引
np.diag_indices(4)
# 使用 np.diag_indices 函数获取指定大小和偏移的对角线索引
np.diag_indices(2, 3)

# 使用 np.diag_indices_from 函数根据指定的数组获取对角线索引
np.diag_indices_from(AR_i8)
```