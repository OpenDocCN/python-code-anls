# `.\pandas-ta\pandas_ta\momentum\qqe.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 库中导入 maximum 函数并重命名为 npMaximum
from numpy import maximum as npMaximum
# 从 numpy 库中导入 minimum 函数并重命名为 npMinimum
from numpy import minimum as npMinimum
# 从 numpy 库中导入 nan 常量并重命名为 npNaN
from numpy import nan as npNaN
# 从 pandas 库中导入 DataFrame 和 Series 类
from pandas import DataFrame, Series
# 从 rsi 模块中导入 rsi 函数
from .rsi import rsi
# 从 pandas_ta.overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta.utils 模块中导入 get_drift, get_offset, verify_series 函数

# 定义函数 qqe，计算 Quantitative Qualitative Estimation (QQE) 指标
def qqe(close, length=None, smooth=None, factor=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Quantitative Qualitative Estimation (QQE)"""
    # 验证参数
    length = int(length) if length and length > 0 else 14
    smooth = int(smooth) if smooth and smooth > 0 else 5
    factor = float(factor) if factor else 4.236
    wilders_length = 2 * length - 1
    mamode = mamode if isinstance(mamode, str) else "ema"
    # 验证 close 序列，并设置最大长度为 length、smooth 和 wilders_length 中的最大值
    close = verify_series(close, max(length, smooth, wilders_length))
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    rsi_ = rsi(close, length)
    _mode = mamode.lower()[0] if mamode != "ema" else ""
    rsi_ma = ma(mamode, rsi_, length=smooth)

    # 计算 RSI MA True Range
    rsi_ma_tr = rsi_ma.diff(drift).abs()

    # 使用 Wilder's Length 和默认宽度 4.236 双重平滑 RSI MA True Range
    smoothed_rsi_tr_ma = ma("ema", rsi_ma_tr, length=wilders_length)
    dar = factor * ma("ema", smoothed_rsi_tr_ma, length=wilders_length)

    # 创建围绕 RSI MA 的上下轨带
    upperband = rsi_ma + dar
    lowerband = rsi_ma - dar

    m = close.size
    # 创建 Series 对象 long、short、trend、qqe、qqe_long 和 qqe_short
    long = Series(0, index=close.index)
    short = Series(0, index=close.index)
    trend = Series(1, index=close.index)
    qqe = Series(rsi_ma.iloc[0], index=close.index)
    qqe_long = Series(npNaN, index=close.index)
    qqe_short = Series(npNaN, index=close.index)
    # 遍历范围为1到m-1的整数
    for i in range(1, m):
        # 获取当前和前一个 RSI_MA 值
        c_rsi, p_rsi = rsi_ma.iloc[i], rsi_ma.iloc[i - 1]
        # 获取当前和前一个 Long Line 值
        c_long, p_long = long.iloc[i - 1], long.iloc[i - 2]
        # 获取当前和前一个 Short Line 值
        c_short, p_short = short.iloc[i - 1], short.iloc[i - 2]

        # Long Line
        # 如果前一个 RSI_MA 大于当前 Long Line 并且当前 RSI_MA 大于当前 Long Line
        if p_rsi > c_long and c_rsi > c_long:
            long.iloc[i] = npMaximum(c_long, lowerband.iloc[i])
        else:
            long.iloc[i] = lowerband.iloc[i]

        # Short Line
        # 如果前一个 RSI_MA 小于当前 Short Line 并且当前 RSI_MA 小于当前 Short Line
        if p_rsi < c_short and c_rsi < c_short:
            short.iloc[i] = npMinimum(c_short, upperband.iloc[i])
        else:
            short.iloc[i] = upperband.iloc[i]

        # Trend & QQE Calculation
        # Long: 当前 RSI_MA 值穿过前一个 Short Line 值
        # Short: 当前 RSI_MA 值穿过前一个 Long Line 值
        if (c_rsi > c_short and p_rsi < p_short) or (c_rsi <= c_short and p_rsi >= p_short):
            trend.iloc[i] = 1
            qqe.iloc[i] = qqe_long.iloc[i] = long.iloc[i]
        elif (c_rsi > c_long and p_rsi < p_long) or (c_rsi <= c_long and p_rsi >= p_long):
            trend.iloc[i] = -1
            qqe.iloc[i] = qqe_short.iloc[i] = short.iloc[i]
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            if trend.iloc[i] == 1:
                qqe.iloc[i] = qqe_long.iloc[i] = long.iloc[i]
            else:
                qqe.iloc[i] = qqe_short.iloc[i]  = short.iloc[i]

    # Offset
    # 如果偏移量不为0，则对数据进行偏移
    if offset != 0:
        rsi_ma = rsi_ma.shift(offset)
        qqe = qqe.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)

    # Handle fills
    # 处理填充值
    if "fillna" in kwargs:
        rsi_ma.fillna(kwargs["fillna"], inplace=True)
        qqe.fillna(kwargs["fillna"], inplace=True)
        qqe_long.fillna(kwargs["fillna"], inplace=True)
        qqe_short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rsi_ma.fillna(method=kwargs["fill_method"], inplace=True)
        qqe.fillna(method=kwargs["fill_method"], inplace=True)
        qqe_long.fillna(method=kwargs["fill_method"], inplace=True)
        qqe_short.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    # 设置名称和分类
    _props = f"{_mode}_{length}_{smooth}_{factor}"
    qqe.name = f"QQE{_props}"
    rsi_ma.name = f"QQE{_props}_RSI{_mode.upper()}MA"
    qqe_long.name = f"QQEl{_props}"
    qqe_short.name = f"QQEs{_props}"
    qqe.category = rsi_ma.category = "momentum"
    qqe_long.category = qqe_short.category = qqe.category

    # Prepare DataFrame to return
    # 准备要返回的 DataFrame
    data = {
        qqe.name: qqe, rsi_ma.name: rsi_ma,
        # long.name: long, short.name: short
        qqe_long.name: qqe_long, qqe_short.name: qqe_short
    }
    df = DataFrame(data)
    df.name = f"QQE{_props}"
    df.category = qqe.category

    return df
# 设置 qqe 对象的文档字符串，描述了 Quantitative Qualitative Estimation (QQE) 指标的计算方法和用法
qqe.__doc__ = \
"""Quantitative Qualitative Estimation (QQE)

The Quantitative Qualitative Estimation (QQE) is similar to SuperTrend but uses a Smoothed RSI with an upper and lower bands. The band width is a combination of a one period True Range of the Smoothed RSI which is double smoothed using Wilder's smoothing length (2 * rsiLength - 1) and multiplied by the default factor of 4.236. A Long trend is determined when the Smoothed RSI crosses the previous upperband and a Short trend when the Smoothed RSI crosses the previous lowerband.

Based on QQE.mq5 by EarnForex Copyright © 2010, based on version by Tim Hyder (2008), based on version by Roman Ignatov (2006)

Sources:
    https://www.tradingview.com/script/IYfA9R2k-QQE-MT4/
    https://www.tradingpedia.com/forex-trading-indicators/quantitative-qualitative-estimation
    https://www.prorealcode.com/prorealtime-indicators/qqe-quantitative-qualitative-estimation/

Calculation:
    Default Inputs:
        length=14, smooth=5, factor=4.236, mamode="ema", drift=1

Args:
    close (pd.Series): Series of 'close's
    length (int): RSI period. Default: 14
    smooth (int): RSI smoothing period. Default: 5
    factor (float): QQE Factor. Default: 4.236
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: QQE, RSI_MA (basis), QQEl (long), and QQEs (short) columns.
"""
```