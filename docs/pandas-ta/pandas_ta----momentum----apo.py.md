# `.\pandas-ta\pandas_ta\momentum\apo.py`

```py
# 设置文件编码为 UTF-8
# 导入所需模块
from pandas_ta import Imports
# 导入移动平均函数
from pandas_ta.overlap import ma
# 导入辅助函数
from pandas_ta.utils import get_offset, tal_ma, verify_series


def apo(close, fast=None, slow=None, mamode=None, talib=None, offset=None, **kwargs):
    """Indicator: Absolute Price Oscillator (APO)"""
    # 验证参数有效性，如果未指定则使用默认值
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    # 如果慢周期小于快周期，交换它们
    if slow < fast:
        fast, slow = slow, fast
    # 验证并准备输入序列
    close = verify_series(close, max(fast, slow))
    # 确定移动平均的模式，默认为简单移动平均
    mamode = mamode if isinstance(mamode, str) else "sma"
    # 获取偏移量
    offset = get_offset(offset)
    # 确定是否使用 TA-Lib 库进行计算，默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果输入序列为空，则返回 None
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果 TA-Lib 可用且需要使用它，则调用 TA-Lib 库计算 APO
        from talib import APO
        # 使用 TA-Lib 计算 APO
        apo = APO(close, fast, slow, tal_ma(mamode))
    else:
        # 否则使用自定义移动平均函数计算 APO
        # 计算快速和慢速移动平均线
        fastma = ma(mamode, close, length=fast)
        slowma = ma(mamode, close, length=slow)
        # 计算 APO
        apo = fastma - slowma

    # 根据偏移量调整结果
    if offset != 0:
        apo = apo.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        apo.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        apo.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    apo.name = f"APO_{fast}_{slow}"
    apo.category = "momentum"

    # 返回计算结果
    return apo


# 设置 APO 函数的文档字符串
apo.__doc__ = \
"""Absolute Price Oscillator (APO)

The Absolute Price Oscillator is an indicator used to measure a security's
momentum.  It is simply the difference of two Exponential Moving Averages
(EMA) of two different periods. Note: APO and MACD lines are equivalent.

Sources:
    https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/absolute-price-oscillator-apo/

Calculation:
    Default Inputs:
        fast=12, slow=26
    SMA = Simple Moving Average
    APO = SMA(close, fast) - SMA(close, slow)

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    mamode (str): See ```help(ta.ma)```py. Default: 'sma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```