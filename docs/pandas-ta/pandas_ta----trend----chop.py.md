# `.\pandas-ta\pandas_ta\trend\chop.py`

```
# -*- coding: utf-8 -*-

# 从 numpy 库中导入 log10 函数并重命名为 npLog10
from numpy import log10 as npLog10
# 从 numpy 库中导入 log 函数并重命名为 npLn
from numpy import log as npLn
# 从 pandas_ta 库中导入 volatility 模块中的 atr 函数
from pandas_ta.volatility import atr
# 从 pandas_ta 库中导入 utils 模块中的 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义函数 chop，计算 Choppiness Index (CHOP)
def chop(high, low, close, length=None, atr_length=None, ln=None, scalar=None, drift=None, offset=None, **kwargs):
    """Indicator: Choppiness Index (CHOP)"""
    # 验证参数
    length = int(length) if length and length > 0 else 14
    atr_length = int(atr_length) if atr_length is not None and atr_length > 0 else 1
    ln = bool(ln) if isinstance(ln, bool) else False
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 如果 high、low、close 中有任何一个为 None，则返回
    if high is None or low is None or close is None: return

    # 计算结果
    diff = high.rolling(length).max() - low.rolling(length).min()

    atr_ = atr(high=high, low=low, close=close, length=atr_length)
    atr_sum = atr_.rolling(length).sum()

    chop = scalar
    if ln:
        chop *= (npLn(atr_sum) - npLn(diff)) / npLn(length)
    else:
        chop *= (npLog10(atr_sum) - npLog10(diff)) / npLog10(length)

    # 偏移结果
    if offset != 0:
        chop = chop.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        chop.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        chop.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    chop.name = f"CHOP{'ln' if ln else ''}_{length}_{atr_length}_{scalar}"
    chop.category = "trend"

    return chop

# 设置 chop 函数的文档字符串
chop.__doc__ = \
"""Choppiness Index (CHOP)

The Choppiness Index was created by Australian commodity trader
E.W. Dreiss and is designed to determine if the market is choppy
(trading sideways) or not choppy (trading within a trend in either
direction). Values closer to 100 implies the underlying is choppier
whereas values closer to 0 implies the underlying is trending.

Sources:
    https://www.tradingview.com/scripts/choppinessindex/
    https://www.motivewave.com/studies/choppiness_index.htm

Calculation:
    Default Inputs:
        length=14, scalar=100, drift=1
    HH = high.rolling(length).max()
    LL = low.rolling(length).min()

    ATR_SUM = SUM(ATR(drift), length)
    CHOP = scalar * (LOG10(ATR_SUM) - LOG10(HH - LL))
    CHOP /= LOG10(length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    atr_length (int): Length for ATR. Default: 1
    ln (bool): If True, uses ln otherwise log10. Default: False
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
"""
    # 创建一个 Pandas Series 对象，表示生成了一个新特征
    pd.Series: New feature generated.
# 这是一个空的字符串，通常用作多行注释的起始
```