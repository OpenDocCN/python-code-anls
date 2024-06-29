# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\index_tricks.pyi`

```
import numpy as np  # 导入 NumPy 库，用于科学计算

AR_LIKE_i: list[int]  # 定义 AR_LIKE_i 为整数类型的列表
AR_LIKE_f: list[float]  # 定义 AR_LIKE_f 为浮点数类型的列表

np.ndindex([1, 2, 3])  # E: No overload variant
# 使用 np.ndindex() 函数创建一个迭代器，遍历给定维度下的所有索引组合

np.unravel_index(AR_LIKE_f, (1, 2, 3))  # E: incompatible type
# 使用 np.unravel_index() 函数将平坦索引或数组转换为多维索引，但 AR_LIKE_f 类型不兼容

np.ravel_multi_index(AR_LIKE_i, (1, 2, 3), mode="bob")  # E: No overload variant
# 使用 np.ravel_multi_index() 函数将多维索引转换为平坦索引，但参数或模式不匹配

np.mgrid[1]  # E: Invalid index type
# 使用 np.mgrid[] 生成一个多维网格，但索引类型无效

np.mgrid[...]  # E: Invalid index type
# 使用 np.mgrid[] 生成一个多维网格，但索引类型无效

np.ogrid[1]  # E: Invalid index type
# 使用 np.ogrid[] 生成一个单轴网格，但索引类型无效

np.ogrid[...]  # E: Invalid index type
# 使用 np.ogrid[] 生成一个单轴网格，但索引类型无效

np.fill_diagonal(AR_LIKE_f, 2)  # E: incompatible type
# 使用 np.fill_diagonal() 函数将对角线元素填充为指定值，但 AR_LIKE_f 类型不兼容

np.diag_indices(1.0)  # E: incompatible type
# 使用 np.diag_indices() 函数返回指定形状数组的对角线索引，但参数类型不兼容
```