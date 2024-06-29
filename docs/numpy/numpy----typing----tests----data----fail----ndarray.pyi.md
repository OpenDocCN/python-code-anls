# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\ndarray.pyi`

```py
# 导入 NumPy 库
import numpy as np

# 禁止设置 dtype，因为直接在原地改变数组的类型会使得 ndarray 无法泛化到不同的 dtype。
# 通常情况下，用户应该在这种情况下使用 `ndarray.view`。详细信息请参见：
#
# https://github.com/numpy/numpy-stubs/issues/7
#
# 获取一个包含单个浮点数的 NumPy 数组
float_array = np.array([1.0])
# 尝试设置数组的 dtype 为布尔型，但是因为 ndarray 的 dtype 是只读属性，所以会引发错误。
float_array.dtype = np.bool  # E: Property "dtype" defined in "ndarray" is read-only
```