# `.\pandas-ta\pandas_ta\momentum\trix.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 overlap 模块下的 ema 函数
from pandas_ta.overlap.ema import ema
# 从 pandas_ta 库中导入 utils 模块
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义 Trix 指标函数，用于计算 Trix (TRIX) 指标
def trix(close, length=None, signal=None, scalar=None, drift=None, offset=None, **kwargs):
    """Indicator: Trix (TRIX)"""
    # 验证参数
   length = int(length) if length and length > 0 else 30
    signal = int(signal) if signal and signal > 0 else 9
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, max(length, signal))
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    ema1 = ema(close=close, length=length, **kwargs)
    ema2 = ema(close=ema1, length=length, **kwargs)
    ema3 = ema(close=ema2, length=length, **kwargs)
    trix = scalar * ema3.pct_change(drift)

    trix_signal = trix.rolling(signal).mean()

    # 偏移
    if offset != 0:
        trix = trix.shift(offset)
        trix_signal = trix_signal.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        trix.fillna(kwargs["fillna"], inplace=True)
        trix_signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        trix.fillna(method=kwargs["fill_method"], inplace=True)
        trix_signal.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    trix.name = f"TRIX_{length}_{signal}"
    trix_signal.name = f"TRIXs_{length}_{signal}"
    trix.category = trix_signal.category = "momentum"

    # 准备返回的 DataFrame
    df = DataFrame({trix.name: trix, trix_signal.name: trix_signal})
    df.name = f"TRIX_{length}_{signal}"
    df.category = "momentum"

    return df

# 设置 trix 函数的文档字符串
trix.__doc__ = \
"""Trix (TRIX)

TRIX is a momentum oscillator to identify divergences.

Sources:
    https://www.tradingview.com/wiki/TRIX

Calculation:
    Default Inputs:
        length=18, drift=1
    EMA = Exponential Moving Average
    ROC = Rate of Change
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    TRIX = 100 * ROC(ema3, drift)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 18
    signal (int): It's period. Default: 9
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```