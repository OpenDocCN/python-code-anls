# `.\numpy\numpy\typing\tests\data\pass\ufuncs.py`

```
# 导入 numpy 库，约定使用 np 作为别名
import numpy as np

# 调用 np.sin 函数计算给定参数 1 的正弦值
np.sin(1)

# 调用 np.sin 函数计算给定数组 [1, 2, 3] 中每个元素的正弦值，返回一个数组
np.sin([1, 2, 3])

# 使用 np.sin 函数计算给定参数 1 的正弦值，并将结果存储到预先分配的 np.empty(1) 的数组中
np.sin(1, out=np.empty(1))

# 使用 np.matmul 函数计算两个 2x2x2 的全一数组的矩阵乘积，
# 指定轴的组合为 [(0, 1), (0, 1), (0, 1)]
np.matmul(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axes=[(0, 1), (0, 1), (0, 1)])

# 调用 np.sin 函数，指定 signature 参数为 "D->D"
np.sin(1, signature="D->D")

# 注意: `np.generic` 的子类不能保证支持加法操作；
# 当我们可以推断出 `np.sin(...)` 的确切返回类型时，可以重新启用这行代码。
#
# np.sin(1) + np.sin(1)

# 访问 np.sin 函数返回类型的第一个元素的类型信息
np.sin.types[0]

# 访问 np.sin 函数的名称
np.sin.__name__

# 访问 np.sin 函数的文档字符串
np.sin.__doc__

# 对 np.array([1]) 应用 np.abs 函数，返回其绝对值的数组
np.abs(np.array([1]))
```