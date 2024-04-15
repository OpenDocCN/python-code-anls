# `.\pandas-ta\pandas_ta\momentum\ppo.py`

```py
# -*- coding: utf-8 -*-
# 导入所需的库和模块
from pandas import DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ma
from pandas_ta.utils import get_offset, tal_ma, verify_series

# 定义 PPO 函数，计算百分比价格振荡器（PPO）
def ppo(close, fast=None, slow=None, signal=None, scalar=None, mamode=None, talib=None, offset=None, **kwargs):
    """Indicator: Percentage Price Oscillator (PPO)"""
    # 验证参数
    # 设置默认值并确保参数为整数或浮点数
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    scalar = float(scalar) if scalar else 100
    mamode = mamode if isinstance(mamode, str) else "sma"
    # 如果 slow 小于 fast，则交换它们的值
    if slow < fast:
        fast, slow = slow, fast
    # 验证 close 数据，并获取偏移量
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    # 判断是否使用 talib 库
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为空，则返回
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 使用 talib 库计算 PPO
        from talib import PPO
        ppo = PPO(close, fast, slow, tal_ma(mamode))
    else:
        # 使用自定义函数计算 PPO
        fastma = ma(mamode, close, length=fast)
        slowma = ma(mamode, close, length=slow)
        ppo = scalar * (fastma - slowma)
        ppo /= slowma

    # 计算信号线和直方图
    signalma = ma("ema", ppo, length=signal)
    histogram = ppo - signalma

    # 处理偏移
    if offset != 0:
        ppo = ppo.shift(offset)
        histogram = histogram.shift(offset)
        signalma = signalma.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        ppo.fillna(kwargs["fillna"], inplace=True)
        histogram.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ppo.fillna(method=kwargs["fill_method"], inplace=True)
        histogram.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    _props = f"_{fast}_{slow}_{signal}"
    ppo.name = f"PPO{_props}"
    histogram.name = f"PPOh{_props}"
    signalma.name = f"PPOs{_props}"
    ppo.category = histogram.category = signalma.category = "momentum"

    # 准备返回的 DataFrame
    data = {ppo.name: ppo, histogram.name: histogram, signalma.name: signalma}
    df = DataFrame(data)
    df.name = f"PPO{_props}"
    df.category = ppo.category

    return df

# 设置 PPO 函数的文档字符串
ppo.__doc__ = \
"""Percentage Price Oscillator (PPO)

The Percentage Price Oscillator is similar to MACD in measuring momentum.

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

Calculation:
    Default Inputs:
        fast=12, slow=26
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    fast_sma = SMA(close, fast)
    slow_sma = SMA(close, slow)
    PPO = 100 * (fast_sma - slow_sma) / slow_sma
    Signal = EMA(PPO, signal)
    Histogram = PPO - Signal

Args:
    close(pandas.Series): Series of 'close's
    fast(int): The short period. Default: 12
    slow(int): The long period. Default: 26

"""
    # 定义一个函数参数 signal，表示信号周期，默认值为 9
    signal(int): The signal period. Default: 9
    # 定义一个函数参数 scalar，表示放大倍数，默认值为 100
    scalar (float): How much to magnify. Default: 100
    # 定义一个函数参数 mamode，表示移动平均模式，查看可用模式的说明可调用 help(ta.ma)，默认值为 'sma'
    mamode (str): See ```help(ta.ma)```py. Default: 'sma'
    # 定义一个函数参数 talib，表示是否使用 TA Lib，如果 TA Lib 已安装且 talib 为 True，则返回 TA Lib 版本，默认值为 True
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib version. Default: True
    # 定义一个函数参数 offset，表示结果偏移多少周期，默认值为 0
    offset(int): How many periods to offset the result. Default: 0
# 定义函数的参数及其作用
Kwargs:
    # 填充缺失值的数值或方法，用于填充 DataFrame 的缺失值
    fillna (value, optional): pd.DataFrame.fillna(value)
    # 填充方法的类型，用于指定填充缺失值的方法
    fill_method (value, optional): Type of fill method

# 返回值说明
Returns:
    # 返回一个包含 ppo、histogram 和 signal 列的 pandas DataFrame
    pd.DataFrame: ppo, histogram, signal columns
```