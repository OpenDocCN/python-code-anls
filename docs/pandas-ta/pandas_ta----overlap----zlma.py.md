# `.\pandas-ta\pandas_ta\overlap\zlma.py`

```py
# -*- coding: utf-8 -*-
# 导入所需的函数库
from . import (
    dema, ema, hma, linreg, rma, sma, swma, t3, tema, trima, vidya, wma
)
# 导入辅助函数
from pandas_ta.utils import get_offset, verify_series

# 定义 Zero Lag Moving Average (ZLMA) 函数
def zlma(close, length=None, mamode=None, offset=None, **kwargs):
    """Indicator: Zero Lag Moving Average (ZLMA)"""
    # 验证参数
    length = int(length) if length and length > 0 else 10
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    # 计算结果
    lag = int(0.5 * (length - 1))
    close_ = 2 * close - close.shift(lag)
    # 根据不同的 mamode 选择不同的移动平均方法
    if   mamode == "dema":   zlma = dema(close_, length=length, **kwargs)
    elif mamode == "hma":    zlma = hma(close_, length=length, **kwargs)
    elif mamode == "linreg": zlma = linreg(close_, length=length, **kwargs)
    elif mamode == "rma":    zlma = rma(close_, length=length, **kwargs)
    elif mamode == "sma":    zlma = sma(close_, length=length, **kwargs)
    elif mamode == "swma":   zlma = swma(close_, length=length, **kwargs)
    elif mamode == "t3":     zlma = t3(close_, length=length, **kwargs)
    elif mamode == "tema":   zlma = tema(close_, length=length, **kwargs)
    elif mamode == "trima":  zlma = trima(close_, length=length, **kwargs)
    elif mamode == "vidya":  zlma = vidya(close_, length=length, **kwargs)
    elif mamode == "wma":    zlma = wma(close_, length=length, **kwargs)
    else:                    zlma = ema(close_, length=length, **kwargs) # "ema"

    # 偏移结果
    if offset != 0:
        zlma = zlma.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        zlma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        zlma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    zlma.name = f"ZL_{zlma.name}"
    zlma.category = "overlap"

    return zlma

# 设置 ZLMA 函数的文档字符串
zlma.__doc__ = \
"""Zero Lag Moving Average (ZLMA)

The Zero Lag Moving Average attempts to eliminate the lag associated
with moving averages.  This is an adaption created by John Ehler and Ric Way.

Sources:
    https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average

Calculation:
    Default Inputs:
        length=10, mamode=EMA
    EMA = Exponential Moving Average
    lag = int(0.5 * (length - 1))

    SOURCE = 2 * close - close.shift(lag)
    ZLMA = MA(kind=mamode, SOURCE, length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    mamode (str): Options: 'ema', 'hma', 'sma', 'wma'. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```