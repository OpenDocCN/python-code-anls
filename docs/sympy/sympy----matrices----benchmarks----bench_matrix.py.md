# `D:\src\scipysrc\sympy\sympy\matrices\benchmarks\bench_matrix.py`

```
# 从 sympy 库中导入 Integer 类型
from sympy.core.numbers import Integer
# 从 sympy 库中导入 eye 和 zeros 函数
from sympy.matrices.dense import (eye, zeros)

# 创建一个整数对象 i3，其值为 3
i3 = Integer(3)
# 创建一个 100x100 的单位矩阵 M
M = eye(100)

# 定义一个函数用于测试矩阵索引操作 M[3, 3]
def timeit_Matrix__getitem_ii():
    M[3, 3]

# 定义一个函数用于测试矩阵索引操作 M[i3, i3]，其中 i3 是之前创建的整数对象
def timeit_Matrix__getitem_II():
    M[i3, i3]

# 定义一个函数用于测试矩阵切片操作 M[:, :]
def timeit_Matrix__getslice():
    M[:, :]

# 定义一个函数用于创建一个 100x100 的零矩阵
def timeit_Matrix_zeronm():
    zeros(100, 100)
```