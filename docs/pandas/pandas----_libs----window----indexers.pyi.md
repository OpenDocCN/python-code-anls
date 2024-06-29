# `D:\src\scipysrc\pandas\pandas\_libs\window\indexers.pyi`

```
# 导入 NumPy 库，简称为 np
import numpy as np

# 导入 pandas 库中的 _typing 模块中的 npt 类型
from pandas._typing import npt

# 定义一个函数 calculate_variable_window_bounds，用于计算可变窗口的边界
def calculate_variable_window_bounds(
    num_values: int,  # 输入参数：值的数量，类型为 int64_t
    window_size: int,  # 输入参数：窗口大小，类型为 int64_t
    min_periods,  # 输入参数：最小周期
    center: bool,  # 输入参数：是否居中
    closed: str | None,  # 输入参数：闭合方式，可以是字符串或 None
    index: np.ndarray,  # 输入参数：索引数组，类型为 int64_t 的 NumPy 数组
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:  # 函数返回一个元组，包含两个 int64_t 类型的 NumPy 数组
    ...
```