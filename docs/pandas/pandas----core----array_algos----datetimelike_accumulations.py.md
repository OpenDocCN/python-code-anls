# `D:\src\scipysrc\pandas\pandas\core\array_algos\datetimelike_accumulations.py`

```
"""
datetimelke_accumulations.py is for accumulations of datetimelike extension arrays
"""

# 引入将来版本的特性，以便在类型检查时可以引用类型
from __future__ import annotations

# 引入类型提示中可能用到的模块
from typing import TYPE_CHECKING

# 引入 NumPy 库
import numpy as np

# 引入 pandas 库中的一些核心数据类型模块
from pandas._libs import iNaT

# 从 pandas 核心数据类型模块中引入 isna 函数
from pandas.core.dtypes.missing import isna

# 如果是类型检查阶段，引入 Callable 类型
if TYPE_CHECKING:
    from collections.abc import Callable


# 定义一个内部函数 _cum_func，用于累积处理 1 维的日期时间类数组
def _cum_func(
    func: Callable,
    values: np.ndarray,
    *,
    skipna: bool = True,
) -> np.ndarray:
    """
    Accumulations for 1D datetimelike arrays.

    Parameters
    ----------
    func : np.cumsum, np.maximum.accumulate, np.minimum.accumulate
        累积函数，可以是 np.cumsum、np.maximum.accumulate 或 np.minimum.accumulate 中的一个
    values : np.ndarray
        包含数值的 NumPy 数组（可以是支持该操作的任何数据类型）。此数组会在原地修改。
    skipna : bool, default True
        是否跳过 NA 值。
    """
    # 尝试根据传入的函数选择适当的填充值
    try:
        fill_value = {
            np.maximum.accumulate: np.iinfo(np.int64).min,
            np.cumsum: 0,
            np.minimum.accumulate: np.iinfo(np.int64).max,
        }[func]
    except KeyError as err:
        # 如果未找到匹配的累积函数，抛出 ValueError 异常
        raise ValueError(
            f"No accumulation for {func} implemented on BaseMaskedArray"
        ) from err

    # 根据 NA 值创建掩码
    mask = isna(values)
    # 将 values 视图转换为 int64 类型的数组 y
    y = values.view("i8")
    # 将 NA 值位置填充为预定义的填充值
    y[mask] = fill_value

    # 如果不跳过 NA 值，则对掩码进行最大累积操作
    if not skipna:
        mask = np.maximum.accumulate(mask)

    # 对数组应用累积函数 func，存储结果到 result
    result = func(y, axis=0)
    # 将原始 NA 值位置的结果值设置为 iNaT（pandas 中的 NA 时间）
    result[mask] = iNaT

    # 如果 values 的数据类型属于日期时间类或时间间隔类，则将结果视图转换回原始数据类型
    if values.dtype.kind in "mM":
        return result.view(values.dtype.base)
    return result


# 定义 cumsum 函数，对给定的数组进行累加操作
def cumsum(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    return _cum_func(np.cumsum, values, skipna=skipna)


# 定义 cummin 函数，对给定的数组进行累计最小值操作
def cummin(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    return _cum_func(np.minimum.accumulate, values, skipna=skipna)


# 定义 cummax 函数，对给定的数组进行累计最大值操作
def cummax(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    return _cum_func(np.maximum.accumulate, values, skipna=skipna)
```