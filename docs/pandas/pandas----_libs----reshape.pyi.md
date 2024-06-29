# `D:\src\scipysrc\pandas\pandas\_libs\reshape.pyi`

```
# 导入 NumPy 库，用于处理数组和数值运算
import numpy as np

# 导入 pandas 库的类型提示模块，用于类型注解
from pandas._typing import npt

# 定义函数 unstack，接受以下参数并不返回任何结果
def unstack(
    values: np.ndarray,      # 要重塑的值数组，类型为 NumPy 数组
    mask: np.ndarray,        # 表示掩码的数组，类型为 NumPy 数组
    stride: int,             # 步长，用于确定数据的间隔
    length: int,             # 数据的长度
    width: int,              # 数据的宽度
    new_values: np.ndarray,  # 新值数组，类型为 NumPy 数组
    new_mask: np.ndarray,    # 新的掩码数组，类型为 NumPy 数组
) -> None:                   # 函数不返回任何值，返回类型为 None
    ...

# 定义函数 explode，接受一个参数并返回两个元素的元组
def explode(
    values: npt.NDArray[np.object_],  # 要拆解的对象数组，类型为 NumPy 对象数组
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int64]]:  # 返回一个包含两个 NumPy 数组的元组
    ...
```