# `.\pandas-ta\pandas_ta\momentum\eri.py`

```
# -*- coding: utf-8 -*-

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 ema 函数
from pandas_ta.overlap import ema
# 从 pandas_ta 库中导入 get_offset, verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 eri，计算 Elder Ray Index (ERI)
def eri(high, low, close, length=None, offset=None, **kwargs):
    """Indicator: Elder Ray Index (ERI)"""
    # 验证参数
   length = int(length) if length and length > 0 else 13
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    # 如果 high、low、close 有任何一个为 None，则返回空
    if high is None or low is None or close is None: return

    # 计算结果
    ema_ = ema(close, length)
    bull = high - ema_
    bear = low - ema_

    # 偏移结果
    if offset != 0:
        bull = bull.shift(offset)
        bear = bear.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        bull.fillna(kwargs["fillna"], inplace=True)
        bear.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        bull.fillna(method=kwargs["fill_method"], inplace=True)
        bear.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    bull.name = f"BULLP_{length}"
    bear.name = f"BEARP_{length}"
    bull.category = bear.category = "momentum"

    # 准备返回的 DataFrame
    data = {bull.name: bull, bear.name: bear}
    df = DataFrame(data)
    df.name = f"ERI_{length}"
    df.category = bull.category

    return df

# 设置函数 eri 的文档字符串
eri.__doc__ = \
"""Elder Ray Index (ERI)

Elder's Bulls Ray Index contains his Bull and Bear Powers. Which are useful ways
to look at the price and see the strength behind the market. Bull Power
measures the capability of buyers in the market, to lift prices above an average
consensus of value.

Bears Power measures the capability of sellers, to drag prices below an average
consensus of value. Using them in tandem with a measure of trend allows you to
identify favourable entry points. We hope you've found this to be a useful
discussion of the Bulls and Bears Power indicators.

Sources:
    https://admiralmarkets.com/education/articles/forex-indicators/bears-and-bulls-power-indicator

Calculation:
    Default Inputs:
        length=13
    EMA = Exponential Moving Average

    BULLPOWER = high - EMA(close, length)
    BEARPOWER = low - EMA(close, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: bull power and bear power columns.
"""
```