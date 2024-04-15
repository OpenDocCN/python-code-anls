# `.\pandas-ta\pandas_ta\overlap\wma.py`

```py
# -*- coding: utf-8 -*-
# 导入需要的模块和函数
from pandas import Series
from pandas_ta import Imports
from pandas_ta.utils import get_offset, verify_series


def wma(close, length=None, asc=None, talib=None, offset=None, **kwargs):
    """Indicator: Weighted Moving Average (WMA)"""
    # 验证参数
    length = int(length) if length and length > 0 else 10  # 确定长度为正整数，默认为10
    asc = asc if asc else True  # 默认为升序
    close = verify_series(close, length)  # 确保close为Series，长度为length
    offset = get_offset(offset)  # 获取偏移量
    mode_tal = bool(talib) if isinstance(talib, bool) else True  # 是否使用TA Lib，默认为True

    if close is None: return  # 如果close为空，则返回

    # 计算结果
    if Imports["talib"] and mode_tal:  # 如果安装了TA Lib且使用TA Lib模式
        from talib import WMA
        wma = WMA(close, length)  # 使用TA Lib中的WMA函数计算WMA
    else:
        from numpy import arange as npArange
        from numpy import dot as npDot

        total_weight = 0.5 * length * (length + 1)  # 计算总权重
        weights_ = Series(npArange(1, length + 1))  # 创建1到length的Series
        weights = weights_ if asc else weights_[::-1]  # 如果升序，则不变；否则倒序

        def linear(w):
            def _compute(x):
                return npDot(x, w) / total_weight  # 线性权重计算WMA
            return _compute

        close_ = close.rolling(length, min_periods=length)  # 创建长度为length的rolling对象
        wma = close_.apply(linear(weights), raw=True)  # 应用线性权重计算WMA

    # 偏移
    if offset != 0:
        wma = wma.shift(offset)  # 偏移结果

    # 处理填充
    if "fillna" in kwargs:
        wma.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充空值
    if "fill_method" in kwargs:
        wma.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定填充方法填充空值

    # 名称和类别
    wma.name = f"WMA_{length}"  # 设置名称
    wma.category = "overlap"  # 设置类别

    return wma  # 返回WMA


wma.__doc__ = \
"""Weighted Moving Average (WMA)

The Weighted Moving Average where the weights are linearly increasing and
the most recent data has the heaviest weight.

Sources:
    https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

Calculation:
    Default Inputs:
        length=10, asc=True
    total_weight = 0.5 * length * (length + 1)
    weights_ = [1, 2, ..., length + 1]  # Ascending
    weights = weights if asc else weights[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    WMA = close.rolling(length)_.apply(linear_weights(weights), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    asc (bool): Recent values weigh more. Default: True
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