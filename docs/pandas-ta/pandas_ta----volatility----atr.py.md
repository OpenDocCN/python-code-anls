# `.\pandas-ta\pandas_ta\volatility\atr.py`

```
# -*- coding: utf-8 -*-
# 导入 true_range 模块
from .true_range import true_range
# 导入 Imports 模块
from pandas_ta import Imports
# 导入 ma 模块
from pandas_ta.overlap import ma
# 导入 get_drift, get_offset, verify_series 模块
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义 ATR 函数，计算平均真实范围
def atr(high, low, close, length=None, mamode=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: Average True Range (ATR)"""
    # 验证参数
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "rma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        from talib import ATR
        atr = ATR(high, low, close, length)
    else:
        tr = true_range(high=high, low=low, close=close, drift=drift)
        atr = ma(mamode, tr, length=length)

    percentage = kwargs.pop("percent", False)
    if percentage:
        atr *= 100 / close

    # 偏移
    if offset != 0:
        atr = atr.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        atr.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        atr.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    atr.name = f"ATR{mamode[0]}_{length}{'p' if percentage else ''}"
    atr.category = "volatility"

    return atr

# 设置 ATR 函数的文档字符串
atr.__doc__ = \
"""Average True Range (ATR)

Averge True Range is used to measure volatility, especially volatility caused by
gaps or limit moves.

Sources:
    https://www.tradingview.com/wiki/Average_True_Range_(ATR)

Calculation:
    Default Inputs:
        length=14, drift=1, percent=False
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    WMA = Weighted Moving Average
    RMA = WildeR's Moving Average
    TR = True Range

    tr = TR(high, low, close, drift)
    if 'ema':
        ATR = EMA(tr, length)
    elif 'sma':
        ATR = SMA(tr, length)
    elif 'wma':
        ATR = WMA(tr, length)
    else:
        ATR = RMA(tr, length)

    if percent:
        ATR *= 100 / close

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    mamode (str): See ```help(ta.ma)```. Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    percent (bool, optional): Return as percentage. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```