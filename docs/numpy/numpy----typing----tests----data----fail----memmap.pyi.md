# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\memmap.pyi`

```
import numpy as np  # 导入 NumPy 库，用于科学计算和数组操作

with open("file.txt", "r") as f:
    np.memmap(f)  # 尝试使用 NumPy 创建一个内存映射，但是参数不足，缺少必要的内存映射对象大小或者 dtype

np.memmap("test.txt", shape=[10, 5])  # 尝试创建一个新的内存映射对象，映射到文件 "test.txt"，但是缺少必要的 dtype 参数，因此报错 "No overload variant"
```