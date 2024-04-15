# `.\pandas-ta\pandas_ta\volatility\true_range.py`

```py
# -*- coding: utf-8 -*-
# 导入 numpy 库中的 nan，并将其重命名为 npNaN
from numpy import nan as npNaN
# 导入 pandas 库中的 concat 函数
from pandas import concat
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta 库中导入 get_drift, get_offset, non_zero_range, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series

# 定义一个名为 true_range 的函数，用于计算真实波动范围
def true_range(high, low, close, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: True Range"""
    # 验证参数
   high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 计算结果
   if Imports["talib"] and mode_tal:
        # 如果导入了 talib 库并且 mode_tal 为 True，则使用 talib 库中的 TRANGE 函数计算真实波动范围
        from talib import TRANGE
        true_range = TRANGE(high, low, close)
    else:
        # 否则，计算高低范围、前一日收盘价、以及真实波动范围
        high_low_range = non_zero_range(high, low)
        prev_close = close.shift(drift)
        ranges = [high_low_range, high - prev_close, prev_close - low]
        true_range = concat(ranges, axis=1)
        true_range = true_range.abs().max(axis=1)
        true_range.iloc[:drift] = npNaN

    # 偏移
    if offset != 0:
        true_range = true_range.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        true_range.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        true_range.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    true_range.name = f"TRUERANGE_{drift}"
    true_range.category = "volatility"

    return true_range

# 设置 true_range 函数的文档字符串
true_range.__doc__ = \
"""True Range

An method to expand a classical range (high minus low) to include
possible gap scenarios.

Sources:
    https://www.macroption.com/true-range/

Calculation:
    Default Inputs:
        drift=1
    ABS = Absolute Value
    prev_close = close.shift(drift)
    TRUE_RANGE = ABS([high - low, high - prev_close, low - prev_close])

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The shift period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""
```