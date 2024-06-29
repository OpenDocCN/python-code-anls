# `D:\src\scipysrc\pandas\pandas\_libs\hashing.pyi`

```
# 导入 NumPy 库，简写为 np
import numpy as np

# 导入 pandas 库中的类型提示 npt（用于声明数组类型）
from pandas._typing import npt

# 定义名为 hash_object_array 的函数，用于处理对象数组的哈希操作
def hash_object_array(
    arr: npt.NDArray[np.object_],  # 接收一个 NumPy 对象数组作为参数 arr，元素类型为 np.object_
    key: str,                       # 接收一个字符串类型的参数 key，作为哈希操作的关键字
    encoding: str = ...,            # 可选的参数 encoding，默认为省略号，未提供具体值
) -> npt.NDArray[np.uint64]:       # 函数返回一个 NumPy 无符号 64 位整数类型的数组

    # 函数体未提供具体实现，使用省略号表示
    ...
```