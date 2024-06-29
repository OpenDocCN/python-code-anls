# `D:\src\scipysrc\pandas\pandas\core\sample.py`

```
"""
Module containing utilities for NDFrame.sample() and .GroupBy.sample()
"""

# 引入未来的注解功能，用于类型检查
from __future__ import annotations

# 引入类型检查相关模块
from typing import TYPE_CHECKING

# 引入 NumPy 库
import numpy as np

# 引入 pandas 库中的底层 C 函数库
from pandas._libs import lib

# 引入 pandas 库中的数据类型模块
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

# 如果类型检查为真，则引入额外的类型
if TYPE_CHECKING:
    from pandas._typing import AxisInt

    # 引入 pandas 核心模块中的通用数据结构 NDFrame
    from pandas.core.generic import NDFrame


def preprocess_weights(obj: NDFrame, weights, axis: AxisInt) -> np.ndarray:
    """
    Process and validate the `weights` argument to `NDFrame.sample` and
    `.GroupBy.sample`.

    Returns `weights` as an ndarray[np.float64], validated except for normalizing
    weights (because that must be done groupwise in groupby sampling).
    """
    # 如果 weights 是 Series 类型，则根据轴向重新索引
    if isinstance(weights, ABCSeries):
        weights = weights.reindex(obj.axes[axis])

    # 如果 weights 是字符串类型
    if isinstance(weights, str):
        # 如果 obj 是 DataFrame 并且 axis = 0
        if isinstance(obj, ABCDataFrame):
            if axis == 0:
                try:
                    # 尝试从 DataFrame 中选择指定列
                    weights = obj[weights]
                except KeyError as err:
                    raise KeyError(
                        "String passed to weights not a valid column"
                    ) from err
            else:
                raise ValueError(
                    "Strings can only be passed to "
                    "weights when sampling from rows on "
                    "a DataFrame"
                )
        else:
            raise ValueError(
                "Strings cannot be passed as weights when sampling from a Series."
            )

    # 根据 obj 的类型选择合适的构造函数
    if isinstance(obj, ABCSeries):
        func = obj._constructor
    else:
        func = obj._constructor_sliced

    # 使用 float64 数据类型构造 weights 数组
    weights = func(weights, dtype="float64")._values

    # 检查 weights 和轴的长度是否相同
    if len(weights) != obj.shape[axis]:
        raise ValueError("Weights and axis to be sampled must be of same length")

    # 检查 weights 是否包含无穷大的值
    if lib.has_infs(weights):
        raise ValueError("weight vector may not include `inf` values")

    # 检查 weights 是否包含负数
    if (weights < 0).any():
        raise ValueError("weight vector many not include negative values")

    # 检查 weights 是否包含 NaN 值，如果是则将其替换为 0
    missing = np.isnan(weights)
    if missing.any():
        # 不在原地修改 weights
        weights = weights.copy()
        weights[missing] = 0
    return weights


def process_sampling_size(
    n: int | None, frac: float | None, replace: bool
) -> int | None:
    """
    Process and validate the `n` and `frac` arguments to `NDFrame.sample` and
    `.GroupBy.sample`.

    Returns None if `frac` should be used (variable sampling sizes), otherwise returns
    the constant sampling size.
    """
    # 如果 n 和 frac 都为 None，则默认 n = 1
    if n is None and frac is None:
        n = 1
    # 如果 n 和 frac 同时存在，则抛出错误
    elif n is not None and frac is not None:
        raise ValueError("Please enter a value for `frac` OR `n`, not both")
    # 如果 n 不为 None，则进行以下判断和操作
    elif n is not None:
        # 如果 n 小于 0，则抛出值错误异常
        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide `n` >= 0."
            )
        # 如果 n 不是整数，则抛出值错误异常
        if n % 1 != 0:
            raise ValueError("Only integers accepted as `n` values")
    
    # 如果 n 为 None，则执行以下代码块
    else:
        # 确保 frac 不为 None（针对类型检查）
        assert frac is not None  # for mypy
        # 如果 frac 大于 1 且 replace 不为 True，则抛出值错误异常
        if frac > 1 and not replace:
            raise ValueError(
                "Replace has to be set to `True` when "
                "upsampling the population `frac` > 1."
            )
        # 如果 frac 小于 0，则抛出值错误异常
        if frac < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide `frac` >= 0."
            )
    
    # 返回 n 变量的值
    return n
# 定义函数 sample，用于从 np.arange(obj_len) 中随机抽取指定数量的索引

def sample(
    obj_len: int,
    size: int,
    replace: bool,
    weights: np.ndarray | None,
    random_state: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    """
    Randomly sample `size` indices in `np.arange(obj_len)`

    Parameters
    ----------
    obj_len : int
        考虑索引的长度
    size : int
        要选择的值的数量
    replace : bool
        是否允许重复抽样同一行
    weights : np.ndarray[np.float64] or None
        如果为 None，则权重相等；否则按照规范化后的向量进行加权
    random_state: np.random.RandomState or np.random.Generator
        用于随机抽样的状态

    Returns
    -------
    np.ndarray[np.intp]
        包含抽样结果索引的数组
    """
    # 如果 weights 不为 None，则进行权重归一化处理
    if weights is not None:
        weight_sum = weights.sum()
        # 如果权重和不为零，则进行归一化
        if weight_sum != 0:
            weights = weights / weight_sum
        else:
            # 如果权重和为零，则抛出异常
            raise ValueError("Invalid weights: weights sum to zero")

    # 使用 random_state 对象进行随机抽样，返回抽样结果的整数数组
    return random_state.choice(obj_len, size=size, replace=replace, p=weights).astype(
        np.intp, copy=False
    )
```