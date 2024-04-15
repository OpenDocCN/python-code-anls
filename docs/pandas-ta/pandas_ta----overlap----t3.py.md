# `.\pandas-ta\pandas_ta\overlap\t3.py`

```
# -*- coding: utf-8 -*-
# 导入 ema 函数
from .ema import ema
# 导入 Imports 模块
from pandas_ta import Imports
# 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 T3 函数，计算 T3 指标
def t3(close, length=None, a=None, talib=None, offset=None, **kwargs):
    """Indicator: T3"""
    # 验证参数
   # 将 length 转换为整数，如果 length 大于 0 则使用 length，否则使用默认值 10
    length = int(length) if length and length > 0 else 10
    # 将 a 转换为浮点数，如果 a 在 0 和 1 之间则使用 a，否则使用默认值 0.7
    a = float(a) if a and a > 0 and a < 1 else 0.7
    # 验证 close 数据类型为 Series，长度为 length
    close = verify_series(close, length)
    # 获取 offset
    offset = get_offset(offset)
    # 判断是否使用 talib，默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为 None，则返回
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果导入了 talib 并且 mode_tal 为 True，则使用 talib 中的 T3 函数
        from talib import T3
        t3 = T3(close, length, a)
    else:
        # 计算 T3 指标的系数
        c1 = -a * a**2
        c2 = 3 * a**2 + 3 * a**3
        c3 = -6 * a**2 - 3 * a - 3 * a**3
        c4 = a**3 + 3 * a**2 + 3 * a + 1

        # 计算 ema1 到 ema6
        e1 = ema(close=close, length=length, **kwargs)
        e2 = ema(close=e1, length=length, **kwargs)
        e3 = ema(close=e2, length=length, **kwargs)
        e4 = ema(close=e3, length=length, **kwargs)
        e5 = ema(close=e4, length=length, **kwargs)
        e6 = ema(close=e5, length=length, **kwargs)
        # 计算 T3
        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # 偏移结果
    if offset != 0:
        t3 = t3.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        t3.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        t3.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    t3.name = f"T3_{length}_{a}"
    t3.category = "overlap"

    return t3

# 设置 T3 函数的文档字符串
t3.__doc__ = """Tim Tillson's T3 Moving Average (T3)

Tim Tillson's T3 Moving Average is considered a smoother and more responsive
moving average relative to other moving averages.

Sources:
    http://www.binarytribune.com/forex-trading-indicators/t3-moving-average-indicator/

Calculation:
    Default Inputs:
        length=10, a=0.7
    c1 = -a^3
    c2 = 3a^2 + 3a^3 = 3a^2 * (1 + a)
    c3 = -6a^2 - 3a - 3a^3
    c4 = a^3 + 3a^2 + 3a + 1

    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    ema4 = EMA(ema3, length)
    ema5 = EMA(ema4, length)
    ema6 = EMA(ema5, length)
    T3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    a (float): 0 < a < 1. Default: 0.7
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```