# `.\pandas-ta\pandas_ta\volatility\bbands.py`

```
# -*- coding: utf-8 -*-
# 导入所需的库和模块
from pandas import DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ma
from pandas_ta.statistics import stdev
from pandas_ta.utils import get_offset, non_zero_range, tal_ma, verify_series

# 定义函数，计算布林带指标
def bbands(close, length=None, std=None, ddof=0, mamode=None, talib=None, offset=None, **kwargs):
    """Indicator: Bollinger Bands (BBANDS)"""
    # 验证参数
    length = int(length) if length and length > 0 else 5
    std = float(std) if std and std > 0 else 2.0
    mamode = mamode if isinstance(mamode, str) else "sma"
    ddof = int(ddof) if ddof >= 0 and ddof < length else 1
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        from talib import BBANDS
        upper, mid, lower = BBANDS(close, length, std, std, tal_ma(mamode))
    else:
        standard_deviation = stdev(close=close, length=length, ddof=ddof)
        deviations = std * standard_deviation
        # deviations = std * standard_deviation.loc[standard_deviation.first_valid_index():,]

        mid = ma(mamode, close, length=length, **kwargs)
        lower = mid - deviations
        upper = mid + deviations

    ulr = non_zero_range(upper, lower)
    bandwidth = 100 * ulr / mid
    percent = non_zero_range(close, lower) / ulr

    # 偏移
    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)
        bandwidth = bandwidth.shift(offset)
        percent = bandwidth.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        mid.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
        bandwidth.fillna(kwargs["fillna"], inplace=True)
        percent.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        mid.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)
        bandwidth.fillna(method=kwargs["fill_method"], inplace=True)
        percent.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    lower.name = f"BBL_{length}_{std}"
    mid.name = f"BBM_{length}_{std}"
    upper.name = f"BBU_{length}_{std}"
    bandwidth.name = f"BBB_{length}_{std}"
    percent.name = f"BBP_{length}_{std}"
    upper.category = lower.category = "volatility"
    mid.category = bandwidth.category = upper.category

    # 准备返回的 DataFrame
    data = {
        lower.name: lower, mid.name: mid, upper.name: upper,
        bandwidth.name: bandwidth, percent.name: percent
    }
    bbandsdf = DataFrame(data)
    bbandsdf.name = f"BBANDS_{length}_{std}"
    bbandsdf.category = mid.category

    return bbandsdf

# 设置函数文档字符串
bbands.__doc__ = \
"""
Bollinger Bands (BBANDS)

John Bollinger 的一种流行的波动率指标。

Sources:
    https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

Calculation:
    计算方法：
        默认参数:
            length=5, std=2, mamode="sma", ddof=0
        EMA = 指数移动平均
        SMA = 简单移动平均
        STDEV = 标准差
        计算标准差 stdev = STDEV(close, length, ddof)
        如果使用 EMA：
            计算 MID = EMA(close, length)
        否则：
            计算 MID = SMA(close, length)

        LOWER = MID - std * stdev
        UPPER = MID + std * stdev

        BANDWIDTH = 100 * (UPPER - LOWER) / MID
        PERCENT = (close - LOWER) / (UPPER - LOWER)

Args:
    close (pd.Series): 'close' 的序列
    length (int): 短周期。默认值：5
    std (int): 长周期。默认值：2
    ddof (int): 使用的自由度。默认值：0
    mamode (str): 参见 ```help(ta.ma)```。默认值：'sma'
    talib (bool): 如果安装了 TA Lib 并且 talib 为 True，则返回 TA Lib 版本。默认值：True
    offset (int): 结果偏移了多少周期。默认值：0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): 填充方法的类型

Returns:
    pd.DataFrame: lower、mid、upper、bandwidth 和 percent 列。
"""
```