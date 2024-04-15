# `.\pandas-ta\pandas_ta\overlap\supertrend.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 库中的 nan 函数并重命名为 npNaN
from numpy import nan as npNaN
# 导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 overlap 模块中的 hl2 函数
from pandas_ta.overlap import hl2
# 从 pandas_ta 库中导入 volatility 模块中的 atr 函数
from pandas_ta.volatility import atr
# 从 pandas_ta 库中导入 utils 模块中的 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 supertrend，计算 Supertrend 指标
def supertrend(high, low, close, length=None, multiplier=None, offset=None, **kwargs):
    """Indicator: Supertrend"""
    # 验证参数
    length = int(length) if length and length > 0 else 7
    multiplier = float(multiplier) if multiplier and multiplier > 0 else 3.0
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None: return

    # 计算结果
    m = close.size
    dir_, trend = [1] * m, [0] * m
    long, short = [npNaN] * m, [npNaN] * m

    hl2_ = hl2(high, low)
    matr = multiplier * atr(high, low, close, length)
    upperband = hl2_ + matr
    lowerband = hl2_ - matr

    for i in range(1, m):
        if close.iloc[i] > upperband.iloc[i - 1]:
            dir_[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lowerband.iloc[i]
        else:
            trend[i] = short[i] = upperband.iloc[i]

    # 准备要返回的 DataFrame
    _props = f"_{length}_{multiplier}"
    df = DataFrame({
            f"SUPERT{_props}": trend,
            f"SUPERTd{_props}": dir_,
            f"SUPERTl{_props}": long,
            f"SUPERTs{_props}": short,
        }, index=close.index)

    df.name = f"SUPERT{_props}"
    df.category = "overlap"

    # 如果需要，应用偏移量
    if offset != 0:
        df = df.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    return df

# 设置 supertrend 函数的文档字符串
supertrend.__doc__ = \
"""Supertrend (supertrend)

Supertrend is an overlap indicator. It is used to help identify trend
direction, setting stop loss, identify support and resistance, and/or
generate buy & sell signals.

Sources:
    http://www.freebsensetips.com/blog/detail/7/What-is-supertrend-indicator-its-calculation

Calculation:
    Default Inputs:
        length=7, multiplier=3.0
    Default Direction:
    Set to +1 or bullish trend at start

    MID = multiplier * ATR
    LOWERBAND = HL2 - MID
    UPPERBAND = HL2 + MID

    if UPPERBAND[i] < FINAL_UPPERBAND[i-1] and close[i-1] > FINAL_UPPERBAND[i-1]:
        FINAL_UPPERBAND[i] = UPPERBAND[i]
    else:
        FINAL_UPPERBAND[i] = FINAL_UPPERBAND[i-1])
"""
    # 如果当前下轨大于前一天的最终下轨，并且前一天的收盘价小于前一天的最终下轨
    if LOWERBAND[i] > FINAL_LOWERBAND[i-1] and close[i-1] < FINAL_LOWERBAND[i-1]:
        # 将当前下轨作为最终下轨
        FINAL_LOWERBAND[i] = LOWERBAND[i]
    else:
        # 否则将前一天的最终下轨作为最终下轨
        FINAL_LOWERBAND[i] = FINAL_LOWERBAND[i-1]

    # 如果当前收盘价小于等于最终上轨
    if close[i] <= FINAL_UPPERBAND[i]:
        # 将最终上轨作为超级趋势值
        SUPERTREND[i] = FINAL_UPPERBAND[i]
    else:
        # 否则将最终下轨作为超级趋势值
        SUPERTREND[i] = FINAL_LOWERBAND[i]
# 定义函数参数
Args:
    high (pd.Series): 'high' 数据序列
    low (pd.Series): 'low' 数据序列
    close (pd.Series): 'close' 数据序列
    length (int) : ATR 计算的长度。默认值为 7
    multiplier (float): 上下轨距离中间范围的系数。默认值为 3.0
    offset (int): 结果的偏移周期数。默认值为 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value) 的填充值
    fill_method (value, optional): 填充方法的类型

Returns:
    pd.DataFrame: 包含 SUPERT (趋势), SUPERTd (方向), SUPERTl (长), SUPERTs (短) 列的数据框
```