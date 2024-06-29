# `D:\src\scipysrc\pandas\pandas\core\array_algos\transforms.py`

```
"""
transforms.py is for shape-preserving functions.
"""

# 从未来导入注解以支持类型提示
from __future__ import annotations

# 导入类型检查所需的模块
from typing import TYPE_CHECKING

# 导入 NumPy 库，用于数组操作
import numpy as np

# 如果正在进行类型检查，则导入 Pandas 所需的类型
if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        Scalar,
    )

# 定义一个函数，用于将数组沿指定轴向移动指定周期的位置，并保持形状不变
def shift(
    values: np.ndarray, periods: int, axis: AxisInt, fill_value: Scalar
) -> np.ndarray:
    # 将新数组赋值为传入的数组
    new_values = values

    # 如果周期为0或数组大小为0，则直接返回传入数组的副本
    if periods == 0 or values.size == 0:
        return new_values.copy()

    # 检查数组是否按列（Fortran顺序）存储，如果是，则转置数组以便后续操作
    f_ordered = values.flags.f_contiguous
    if f_ordered:
        new_values = new_values.T
        axis = new_values.ndim - axis - 1

    # 如果新数组非空，则调用 NumPy 的 roll 函数进行数组元素的滚动操作
    if new_values.size:
        new_values = np.roll(
            new_values,
            np.intp(periods),
            axis=axis,
        )

    # 创建一个索引器列表，用于在指定轴上填充指定值
    axis_indexer = [slice(None)] * values.ndim
    if periods > 0:
        axis_indexer[axis] = slice(None, periods)
    else:
        axis_indexer[axis] = slice(periods, None)
    new_values[tuple(axis_indexer)] = fill_value

    # 如果最初数组按列存储，则恢复其原始顺序
    if f_ordered:
        new_values = new_values.T

    # 返回修改后的数组
    return new_values
```