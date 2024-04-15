# `.\pandas-ta\pandas_ta\overlap\trima.py`

```
# -*- coding: utf-8 -*-
# 从sma模块导入sma函数
from .sma import sma
# 从pandas_ta模块导入Imports
from pandas_ta import Imports
# 从pandas_ta.utils模块导入get_offset和verify_series函数
from pandas_ta.utils import get_offset, verify_series

# 定义Triangular Moving Average (TRIMA)指标函数
def trima(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Triangular Moving Average (TRIMA)"""
    # 验证参数
    # 如果length存在且大于0，则将其转换为整数，否则设为10
    length = int(length) if length and length > 0 else 10
    # 验证close序列，确保长度为length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 判断是否使用talib，默认为True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果close为空，则返回空
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果导入了talib并且mode_tal为True，则使用talib库计算TRIMA
        from talib import TRIMA
        trima = TRIMA(close, length)
    else:
        # 否则，计算TRIMA的一半长度
        half_length = round(0.5 * (length + 1))
        # 计算SMA1
        sma1 = sma(close, length=half_length)
        # 计算TRIMA
        trima = sma(sma1, length=half_length)

    # 偏移结果
    if offset != 0:
        trima = trima.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        trima.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        trima.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    trima.name = f"TRIMA_{length}"
    trima.category = "overlap"

    return trima


# 设置TRIMA函数的文档字符串
trima.__doc__ = \
"""Triangular Moving Average (TRIMA)

A weighted moving average where the shape of the weights are triangular and the
greatest weight is in the middle of the period.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triangular-moving-average-trima/
    tma = sma(sma(src, ceil(length / 2)), floor(length / 2) + 1)  # Tradingview
    trima = sma(sma(x, n), n)  # Tradingview

Calculation:
    Default Inputs:
        length=10
    SMA = Simple Moving Average
    half_length = round(0.5 * (length + 1))
    SMA1 = SMA(close, half_length)
    TRIMA = SMA(SMA1, half_length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```