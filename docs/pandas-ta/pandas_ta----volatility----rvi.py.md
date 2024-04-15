# `.\pandas-ta\pandas_ta\volatility\rvi.py`

```
# -*- coding: utf-8 -*-
# 从pandas_ta.overlap模块导入ma函数
from pandas_ta.overlap import ma
# 从pandas_ta.statistics模块导入stdev函数
from pandas_ta.statistics import stdev
# 从pandas_ta.utils模块导入get_drift和get_offset函数
from pandas_ta.utils import get_drift, get_offset
# 从pandas_ta.utils模块导入unsigned_differences和verify_series函数
from pandas_ta.utils import unsigned_differences, verify_series


# 定义Relative Volatility Index (RVI)指标函数
def rvi(close, high=None, low=None, length=None, scalar=None, refined=None, thirds=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Relative Volatility Index (RVI)"""
    # 验证参数
    length = int(length) if length and length > 0 else 14  # 如果length为正整数则保留，否则默认为14
    scalar = float(scalar) if scalar and scalar > 0 else 100  # 如果scalar为正浮点数则保留，否则默认为100
    refined = False if refined is None else refined  # 如果refined为None则默认为False
    thirds = False if thirds is None else thirds  # 如果thirds为None则默认为False
    mamode = mamode if isinstance(mamode, str) else "ema"  # 如果mamode为字符串则保留，否则默认为"ema"
    close = verify_series(close, length)  # 验证close是否为有效序列
    drift = get_drift(drift)  # 获取漂移参数
    offset = get_offset(offset)  # 获取偏移参数

    if close is None: return

    if refined or thirds:
        high = verify_series(high)  # 验证high是否为有效序列
        low = verify_series(low)  # 验证low是否为有效序列

    # 计算结果
    def _rvi(source, length, scalar, mode, drift):
        """RVI"""
        # 计算标准差
        std = stdev(source, length)
        # 获取正差值和负差值
        pos, neg = unsigned_differences(source, amount=drift)

        # 计算正差值的标准差加权平均
        pos_std = pos * std
        pos_avg = ma(mode, pos_std, length=length)
        # 计算负差值的标准差加权平均
        neg_std = neg * std
        neg_avg = ma(mode, neg_std, length=length)

        # 计算RVI指标
        result = scalar * pos_avg
        result /= pos_avg + neg_avg
        return result

    _mode = ""
    if refined:  # 如果使用了refined模式
        # 计算高价RVI
        high_rvi = _rvi(high, length, scalar, mamode, drift)
        # 计算低价RVI
        low_rvi = _rvi(low, length, scalar, mamode, drift)
        # 计算RVI
        rvi = 0.5 * (high_rvi + low_rvi)
        _mode = "r"  # 设置模式为"r"
    elif thirds:  # 如果使用了thirds模式
        # 计算高价RVI
        high_rvi = _rvi(high, length, scalar, mamode, drift)
        # 计算低价RVI
        low_rvi = _rvi(low, length, scalar, mamode, drift)
        # 计算收盘价RVI
        close_rvi = _rvi(close, length, scalar, mamode, drift)
        # 计算RVI
        rvi = (high_rvi + low_rvi + close_rvi) / 3.0
        _mode = "t"  # 设置模式为"t"
    else:  # 如果未使用refined和thirds模式
        # 计算RVI
        rvi = _rvi(close, length, scalar, mamode, drift)

    # 偏移
    if offset != 0:
        rvi = rvi.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        rvi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rvi.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和归类
    rvi.name = f"RVI{_mode}_{length}"
    rvi.category = "volatility"

    return rvi


# 设置RVI指标的文档字符串
rvi.__doc__ = \
"""Relative Volatility Index (RVI)

The Relative Volatility Index (RVI) was created in 1993 and revised in 1995.
Instead of adding up price changes like RSI based on price direction, the RVI
adds up standard deviations based on price direction.

Sources:
    https://www.tradingview.com/wiki/Keltner_Channels_(KC)

Calculation:
    Default Inputs:
        length=14, scalar=100, refined=None, thirds=None
    EMA = Exponential Moving Average
    STDEV = Standard Deviation

    UP = STDEV(src, length) IF src.diff() > 0 ELSE 0
    DOWN = STDEV(src, length) IF src.diff() <= 0 ELSE 0

    UPSUM = EMA(UP, length)
    DOWNSUM = EMA(DOWN, length
"""
    # 计算相对强度指数（RSI），其计算公式为 RVI = scalar * (UPSUM / (UPSUM + DOWNSUM))
    RVI = scalar * (UPSUM / (UPSUM + DOWNSUM))
# 定义一个函数，用于计算布林带指标（Bollinger Bands）
def bollinger_hband_indicator(high, low, close, length=14, scalar=100, refined=False, thirds=False, mamode='ema', offset=0, fillna=None, fill_method=None):
    """
    Args:
        high (pd.Series): 最高价的序列
        low (pd.Series): 最低价的序列
        close (pd.Series): 收盘价的序列
        length (int): 短周期。默认值为14
        scalar (float): 缩放带的正浮点数。默认值为100
        refined (bool): 使用“精炼”计算，即 RVI(high) 和 RVI(low) 的平均值，而不是 RVI(close)。默认值为False
        thirds (bool): 使用最高价、最低价和收盘价的平均值。默认值为False
        mamode (str): 参见 ```help(ta.ma)```。默认值为'ema'
        offset (int): 结果的偏移周期数。默认值为0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value) 的参数
        fill_method (value, optional): 填充方法的类型

    Returns:
        pd.DataFrame: 包含 lower、basis、upper 列的数据框
    """
```