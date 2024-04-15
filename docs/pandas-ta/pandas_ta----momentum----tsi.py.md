# `.\pandas-ta\pandas_ta\momentum\tsi.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.overlap 模块导入 ema, ma 函数
from pandas_ta.overlap import ema, ma
# 从 pandas_ta.utils 模块导入 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series


# 定义 True Strength Index (TSI) 指标函数
def tsi(close, fast=None, slow=None, signal=None, scalar=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: True Strength Index (TSI)"""
    # 验证参数有效性
    # 如果 fast 参数存在且大于 0，则将其转换为整数，否则设为默认值 13
    fast = int(fast) if fast and fast > 0 else 13
    # 如果 slow 参数存在且大于 0，则将其转换为整数，否则设为默认值 25
    slow = int(slow) if slow and slow > 0 else 25
    # 如果 signal 参数存在且大于 0，则将其转换为整数，否则设为默认值 13
    signal = int(signal) if signal and signal > 0 else 13
    # 如果 close 序列为 None，则返回 None
    if close is None: return
    # 如果 scalar 存在，则将其转换为浮点数，否则设为默认值 100
    scalar = float(scalar) if scalar else 100
    # 获取漂移值，用于处理偏移
    drift = get_drift(drift)
    # 获取偏移量，用于处理偏移
    offset = get_offset(offset)
    # 如果 mamode 不是字符串类型，则将其设为默认值 "ema"
    mamode = mamode if isinstance(mamode, str) else "ema"
    # 如果 kwargs 中包含 "length" 键，则将其移除
    if "length" in kwargs: kwargs.pop("length")

    # 计算结果
    # 计算 close 序列的一阶差分
    diff = close.diff(drift)
    # 计算 slow 期 EMA
    slow_ema = ema(close=diff, length=slow, **kwargs)
    # 计算 fast 期 EMA
    fast_slow_ema = ema(close=slow_ema, length=fast, **kwargs)

    # 计算绝对差分
    abs_diff = diff.abs()
    # 计算 slow 期绝对差分的 EMA
    abs_slow_ema = ema(close=abs_diff, length=slow, **kwargs)
    # 计算 fast 期绝对差分的 EMA
    abs_fast_slow_ema = ema(close=abs_slow_ema, length=fast, **kwargs)

    # 计算 TSI
    tsi = scalar * fast_slow_ema / abs_fast_slow_ema
    # 计算 TSI 的信号线
    tsi_signal = ma(mamode, tsi, length=signal)

    # 处理偏移
    if offset != 0:
        tsi = tsi.shift(offset)
        tsi_signal = tsi_signal.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        tsi.fillna(kwargs["fillna"], inplace=True)
        tsi_signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        tsi.fillna(method=kwargs["fill_method"], inplace=True)
        tsi_signal.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名并分类化指标
    tsi.name = f"TSI_{fast}_{slow}_{signal}"
    tsi_signal.name = f"TSIs_{fast}_{slow}_{signal}"
    tsi.category = tsi_signal.category =  "momentum"

    # 准备返回的 DataFrame
    df = DataFrame({tsi.name: tsi, tsi_signal.name: tsi_signal})
    df.name = f"TSI_{fast}_{slow}_{signal}"
    df.category = "momentum"

    return df


# 设置 tsi 函数的文档字符串
tsi.__doc__ = \
"""True Strength Index (TSI)

The True Strength Index is a momentum indicator used to identify short-term
swings while in the direction of the trend as well as determining overbought
and oversold conditions.

Sources:
    https://www.investopedia.com/terms/t/tsi.asp

Calculation:
    Default Inputs:
        fast=13, slow=25, signal=13, scalar=100, drift=1
    EMA = Exponential Moving Average
    diff = close.diff(drift)

    slow_ema = EMA(diff, slow)
    fast_slow_ema = EMA(slow_ema, slow)

    abs_diff_slow_ema = absolute_diff_ema = EMA(ABS(diff), slow)
    abema = abs_diff_fast_slow_ema = EMA(abs_diff_slow_ema, fast)

    TSI = scalar * fast_slow_ema / abema
    Signal = EMA(TSI, signal)

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 13
    slow (int): The long period. Default: 25
    signal (int): The signal period. Default: 13
"""
    scalar (float): How much to magnify. Default: 100
    # 定义一个浮点数变量 scalar，表示放大倍数，默认值为 100

    mamode (str): Moving Average of TSI Signal Line.
        See ```help(ta.ma)```. Default: 'ema'
    # 定义一个字符串变量 mamode，表示 TSI 信号线的移动平均方式，默认值为 'ema'，可查看 ta.ma 的帮助文档

    drift (int): The difference period. Default: 1
    # 定义一个整数变量 drift，表示差分周期，默认值为 1

    offset (int): How many periods to offset the result. Default: 0
    # 定义一个整数变量 offset，表示结果的偏移周期数，默认值为 0
# 函数参数说明部分，kwargs表示可变关键字参数
Kwargs:
    # fillna参数，用于填充缺失值的值，类型为任意
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method参数，填充方法的类型
    fill_method (value, optional): Type of fill method

# 返回值说明部分，返回一个pandas DataFrame对象，包含tsi和signal两列
Returns:
    # 返回的pandas DataFrame对象，包含tsi和signal两列数据
    pd.DataFrame: tsi, signal.
```