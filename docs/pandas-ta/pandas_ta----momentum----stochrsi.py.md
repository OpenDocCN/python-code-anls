# `.\pandas-ta\pandas_ta\momentum\stochrsi.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 .rsi 模块中导入 rsi 函数
from .rsi import rsi
# 从 pandas_ta.overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta.utils 模块中导入 get_offset, non_zero_range, verify_series 函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series


# 定义函数 stochrsi，计算 Stochastic RSI Oscillator (STOCHRSI)
def stochrsi(close, length=None, rsi_length=None, k=None, d=None, mamode=None, offset=None, **kwargs):
    """Indicator: Stochastic RSI Oscillator (STOCHRSI)"""
    # 校验参数
    length = length if length and length > 0 else 14
    rsi_length = rsi_length if rsi_length and rsi_length > 0 else 14
    k = k if k and k > 0 else 3
    d = d if d and d > 0 else 3
    # 校验 close 序列
    close = verify_series(close, max(length, rsi_length, k, d))
    offset = get_offset(offset)
    # 确定 mamode 默认为 "sma"，如果 mamode 不是字符串则设为 "sma"
    mamode = mamode if isinstance(mamode, str) else "sma"

    # 如果 close 为 None，返回空值
    if close is None: return

    # 计算结果
    # 计算 RSI
    rsi_ = rsi(close, length=rsi_length)
    # 计算最低 RSI
    lowest_rsi = rsi_.rolling(length).min()
    # 计算最高 RSI
    highest_rsi = rsi_.rolling(length).max()

    # 计算 stoch 值
    stoch = 100 * (rsi_ - lowest_rsi)
    stoch /= non_zero_range(highest_rsi, lowest_rsi)

    # 计算 STOCHRSI 的 %K 线和 %D 线
    stochrsi_k = ma(mamode, stoch, length=k)
    stochrsi_d = ma(mamode, stochrsi_k, length=d)

    # 偏移
    if offset != 0:
        stochrsi_k = stochrsi_k.shift(offset)
        stochrsi_d = stochrsi_d.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        stochrsi_k.fillna(kwargs["fillna"], inplace=True)
        stochrsi_d.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stochrsi_k.fillna(method=kwargs["fill_method"], inplace=True)
        stochrsi_d.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名并分类
    _name = "STOCHRSI"
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    stochrsi_k.name = f"{_name}k{_props}"
    stochrsi_d.name = f"{_name}d{_props}"
    stochrsi_k.category = stochrsi_d.category = "momentum"

    # 准备返回的 DataFrame
    data = {stochrsi_k.name: stochrsi_k, stochrsi_d.name: stochrsi_d}
    df = DataFrame(data)
    df.name = f"{_name}{_props}"
    df.category = stochrsi_k.category

    return df


# 设置 stochrsi 函数的文档字符串
stochrsi.__doc__ = \
"""Stochastic (STOCHRSI)

"Stochastic RSI and Dynamic Momentum Index" was created by Tushar Chande and Stanley Kroll and published in Stock & Commodities V.11:5 (189-199)

It is a range-bound oscillator with two lines moving between 0 and 100.
The first line (%K) displays the current RSI in relation to the period's
high/low range. The second line (%D) is a Simple Moving Average of the %K line.
The most common choices are a 14 period %K and a 3 period SMA for %D.

Sources:
    https://www.tradingview.com/wiki/Stochastic_(STOCH)

Calculation:
    Default Inputs:
        length=14, rsi_length=14, k=3, d=3
    RSI = Relative Strength Index
    SMA = Simple Moving Average

    RSI = RSI(high, low, close, rsi_length)
    LL  = lowest RSI for last rsi_length periods
    HH  = highest RSI for last rsi_length periods

    STOCHRSI  = 100 * (RSI - LL) / (HH - LL)
    STOCHRSIk = SMA(STOCHRSI, k)
    STOCHRSId = SMA(STOCHRSIk, d)

Args:
    high (pd.Series): Series of 'high's

"""
    low (pd.Series): 存储股价的最低价序列
    close (pd.Series): 存储股价的收盘价序列
    length (int): STOCHRSI 的周期。默认为 14
    rsi_length (int): RSI 的周期。默认为 14
    k (int): 快速 %K 的周期。默认为 3
    d (int): 慢速 %K 的周期。默认为 3
    mamode (str): 查看 ```help(ta.ma)```py。默认为 'sma'（简单移动平均）
    offset (int): 结果偏移的周期数。默认为 0
# 参数说明部分，描述函数的参数和返回值
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

# 返回值说明部分，描述函数返回的数据类型和列名
Returns:
    pd.DataFrame: RSI %K, RSI %D columns.
```