# `.\numpy\numpy\typing\tests\data\pass\ndarray_shape_manipulation.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算

nd1 = np.array([[1, 2], [3, 4]])

# reshape方法示例
nd1.reshape(4)  # 将数组重塑为一维数组，形状为 (4,)
nd1.reshape(2, 2)  # 将数组重塑为 2x2 的二维数组
nd1.reshape((2, 2))  # 同样将数组重塑为 2x2 的二维数组

nd1.reshape((2, 2), order="C")  # 使用C顺序按行填充重塑数组
nd1.reshape(4, order="C")  # 使用C顺序按行填充重塑数组

# resize方法示例
nd1.resize()  # 将数组就地重塑为新形状
nd1.resize(4)  # 将数组就地重塑为形状为 (4,) 的新数组
nd1.resize(2, 2)  # 将数组就地重塑为 2x2 的新数组
nd1.resize((2, 2))  # 同样将数组就地重塑为 2x2 的新数组

nd1.resize((2, 2), refcheck=True)  # 将数组就地重塑为 2x2 的新数组，检查引用是否有效
nd1.resize(4, refcheck=True)  # 将数组就地重塑为形状为 (4,) 的新数组，检查引用是否有效

nd2 = np.array([[1, 2], [3, 4]])

# transpose方法示例
nd2.transpose()  # 返回转置的数组
nd2.transpose(1, 0)  # 返回轴的转置，等同于 nd2.T
nd2.transpose((1, 0))  # 同样返回轴的转置

# swapaxes方法示例
nd2.swapaxes(0, 1)  # 交换数组的两个轴

# flatten方法示例
nd2.flatten()  # 返回数组的一维副本，按行顺序
nd2.flatten("C")  # 同样返回按行顺序的一维副本

# ravel方法示例
nd2.ravel()  # 返回数组的扁平化视图，默认按行顺序
nd2.ravel("C")  # 同样返回按行顺序的扁平化视图

# squeeze方法示例
nd2.squeeze()  # 删除数组的单维度条目

nd3 = np.array([[1, 2]])
nd3.squeeze(0)  # 删除指定轴上的单维度条目，这里删除第0轴上的单维度

nd4 = np.array([[[1, 2]]])
nd4.squeeze((0, 1))  # 删除指定轴上的单维度条目，这里删除第0轴和第1轴上的单维度
```