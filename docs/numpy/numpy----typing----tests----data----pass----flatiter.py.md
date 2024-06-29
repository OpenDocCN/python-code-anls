# `.\numpy\numpy\typing\tests\data\pass\flatiter.py`

```py
# 导入 NumPy 库，简写为 np
import numpy as np

# 创建一个形状为 (2, 2) 的空数组，并获取其扁平化的迭代器对象
a = np.empty((2, 2)).flat

# 返回数组的基础对象（如果有的话，否则返回 None）
a.base

# 返回数组的一个副本
a.copy()

# 返回一个迭代器，用于扁平化数组的坐标
a.coords

# 返回当前迭代位置的索引
a.index

# 返回数组的迭代器对象
iter(a)

# 返回迭代器的下一个元素
next(a)

# 获取扁平化数组中索引为 0 的元素
a[0]

# 获取扁平化数组中索引为 0, 1, 2 的元素（索引超出范围的部分将抛出 IndexError）
a[[0, 1, 2]]

# 获取整个扁平化数组的切片（类似于获取所有元素）
a[...]

# 获取整个扁平化数组的切片（类似于获取所有元素）
a[:]

# 返回一个数组的视图（如果需要的话），或者返回一个具有指定数据类型的新数组
a.__array__()

# 返回一个数组的视图，并指定返回数组的数据类型为 np.float64
a.__array__(np.dtype(np.float64))
```