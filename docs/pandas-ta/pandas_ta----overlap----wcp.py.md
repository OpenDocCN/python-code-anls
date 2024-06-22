# `.\pandas-ta\pandas_ta\overlap\wcp.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义函数 wcp，计算加权收盘价（WCP）
def wcp(high, low, close, talib=None, offset=None, **kwargs):
    """Indicator: Weighted Closing Price (WCP)"""
    # 验证参数
   high = verify_series(high)
   low = verify_series(low)
   close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 计算结果
   if Imports["talib"] and mode_tal:
        from talib import WCLPRICE
        wcp = WCLPRICE(high, low, close)
    else:
        wcp = (high + low + 2 * close) / 4

    # 偏移
   if offset != 0:
        wcp = wcp.shift(offset)

    # 处理填充
   if "fillna" in kwargs:
        wcp.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        wcp.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    wcp.name = "WCP"
    wcp.category = "overlap"

    return wcp


# 设置函数 wcp 的文档字符串
wcp.__doc__ = \
"""Weighted Closing Price (WCP)

Weighted Closing Price is the weighted price given: high, low
and double the close.

Sources:
    https://www.fmlabs.com/reference/default.htm?url=WeightedCloses.htm

Calculation:
    WCP = (2 * close + high + low) / 4

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```