# `.\pandas-ta\pandas_ta\momentum\macd.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 concat 和 DataFrame 函数
from pandas import concat, DataFrame
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.overlap 模块中导入 ema 函数
from pandas_ta.overlap import ema
# 从 pandas_ta.utils 模块中导入 get_offset, verify_series, signals 函数
from pandas_ta.utils import get_offset, verify_series, signals

# 定义 MACD 指标函数，参数包括 close（收盘价）、fast（快速线周期）、slow（慢速线周期）、signal（信号线周期）、talib（是否使用 talib 库计算）、offset（偏移量）等
def macd(close, fast=None, slow=None, signal=None, talib=None, offset=None, **kwargs):
    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    # 验证参数
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    as_mode = kwargs.setdefault("asmode", False)

    # 计算结果
    if Imports["talib"] and mode_tal:
        from talib import MACD
        macd, signalma, histogram = MACD(close, fast, slow, signal)
    else:
        fastma = ema(close, length=fast)
        slowma = ema(close, length=slow)

        macd = fastma - slowma
        signalma = ema(close=macd.loc[macd.first_valid_index():,], length=signal)
        histogram = macd - signalma

    if as_mode:
        macd = macd - signalma
        signalma = ema(close=macd.loc[macd.first_valid_index():,], length=signal)
        histogram = macd - signalma

    # 偏移
    if offset != 0:
        macd = macd.shift(offset)
        histogram = histogram.shift(offset)
        signalma = signalma.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        macd.fillna(kwargs["fillna"], inplace=True)
        histogram.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        macd.fillna(method=kwargs["fill_method"], inplace=True)
        histogram.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    _asmode = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"
    macd.name = f"MACD{_asmode}{_props}"
    histogram.name = f"MACD{_asmode}h{_props}"
    signalma.name = f"MACD{_asmode}s{_props}"
    macd.category = histogram.category = signalma.category = "momentum"

    # 准备返回的 DataFrame
    data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
    df = DataFrame(data)
    df.name = f"MACD{_asmode}{_props}"
    df.category = macd.category

    signal_indicators = kwargs.pop("signal_indicators", False)
    # 如果信号指标存在
    if signal_indicators:
        # 将数据框与直方图信号和 MACD 信号合并
        signalsdf = concat(
            [
                df,
                signals(
                    indicator=histogram,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", True),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
                signals(
                    indicator=macd,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )
        # 返回合并后的数据框
        return signalsdf
    else:
        # 如果信号指标不存在，直接返回原始数据框
        return df
macd.__doc__ = \
"""Moving Average Convergence Divergence (MACD)

The MACD is a popular indicator to that is used to identify a security's trend.
While APO and MACD are the same calculation, MACD also returns two more series
called Signal and Histogram. The Signal is an EMA of MACD and the Histogram is
the difference of MACD and Signal.

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
    AS Mode: https://tr.tradingview.com/script/YFlKXHnP/

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9
    EMA = Exponential Moving Average
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

    if asmode:
        MACD = MACD - Signal
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    signal (int): The signal period. Default: 9
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    asmode (value, optional): When True, enables AS version of MACD.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: macd, histogram, signal columns.
"""
```