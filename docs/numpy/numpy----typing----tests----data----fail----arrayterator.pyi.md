# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\arrayterator.pyi`

```py
import numpy as np  # 导入NumPy库
import numpy.typing as npt  # 导入NumPy类型定义模块

AR_i8: npt.NDArray[np.int64]  # 声明一个名为AR_i8的NumPy数组，其元素类型为np.int64

# 使用AR_i8创建一个Arrayterator对象，这是NumPy中的一个迭代器
ar_iter = np.lib.Arrayterator(AR_i8)

np.lib.Arrayterator(np.int64())  # E: 不兼容的类型
ar_iter.shape = (10, 5)  # E: 只读属性无法更改形状
ar_iter[None]  # E: 无效的索引类型
ar_iter[None, 1]  # E: 无效的索引类型
ar_iter[np.intp()]  # E: 无效的索引类型
ar_iter[np.intp(), ...]  # E: 无效的索引类型
ar_iter[AR_i8]  # E: 无效的索引类型
ar_iter[AR_i8, :]  # E: 无效的索引类型
```