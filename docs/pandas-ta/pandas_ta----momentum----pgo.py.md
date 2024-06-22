# `.\pandas-ta\pandas_ta\momentum\pgo.py`

```py
# 设置文件编码格式为 UTF-8
# -*- coding: utf-8 -*-

# 导入所需模块和函数
from pandas_ta.overlap import ema, sma
from pandas_ta.volatility import atr
from pandas_ta.utils import get_offset, verify_series

# 定义 Pretty Good Oscillator (PGO) 指标函数
def pgo(high, low, close, length=None, offset=None, **kwargs):
    """Indicator: Pretty Good Oscillator (PGO)"""
    # 验证参数
    # 如果 length 有值并且大于 0，将其转换为整数，否则设为默认值 14
    length = int(length) if length and length > 0 else 14
    # 验证 high、low、close 参数并限制长度为 length
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 high、low、close 有任何一个为空，返回空值
    if high is None or low is None or close is None: return

    # 计算 PGO 指标
    # PGO = (close - SMA(close, length)) / EMA(ATR(high, low, close, length), length)
    pgo = close - sma(close, length)
    pgo /= ema(atr(high, low, close, length), length)

    # 偏移结果
    if offset != 0:
        pgo = pgo.shift(offset)

    # 处理填充
    # 如果 kwargs 中含有 "fillna"，使用指定的填充值进行填充
    if "fillna" in kwargs:
        pgo.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中含有 "fill_method"，使用指定的填充方法进行填充
    if "fill_method" in kwargs:
        pgo.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    pgo.name = f"PGO_{length}"
    pgo.category = "momentum"

    return pgo

# 为 PGO 函数添加文档字符串
pgo.__doc__ = \
"""Pretty Good Oscillator (PGO)

The Pretty Good Oscillator indicator was created by Mark Johnson to measure the distance of the current close from its N-day Simple Moving Average, expressed in terms of an average true range over a similar period. Johnson's approach was to
use it as a breakout system for longer term trades. Long if greater than 3.0 and
short if less than -3.0.

Sources:
    https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html

Calculation:
    Default Inputs:
        length=14
    ATR = Average True Range
    SMA = Simple Moving Average
    EMA = Exponential Moving Average

    PGO = (close - SMA(close, length)) / EMA(ATR(high, low, close, length), length)

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
    pd.Series: New feature generated.
"""
```