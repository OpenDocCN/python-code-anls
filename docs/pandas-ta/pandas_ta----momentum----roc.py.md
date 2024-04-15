# `.\pandas-ta\pandas_ta\momentum\roc.py`

```py
# -*- coding: utf-8 -*-

# 从 mom 模块中导入 mom 函数
from .mom import mom
# 从 pandas_ta 模块中导入 Imports
from pandas_ta import Imports
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 Rate of Change (ROC) 函数
def roc(close, length=None, scalar=None, talib=None, offset=None, **kwargs):
    """Indicator: Rate of Change (ROC)"""
    
    # 验证参数
    length = int(length) if length and length > 0 else 10
    scalar = float(scalar) if scalar and scalar > 0 else 100
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        from talib import ROC
        roc = ROC(close, length)
    else:
        roc = scalar * mom(close=close, length=length) / close.shift(length)

    # 偏移
    if offset != 0:
        roc = roc.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        roc.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        roc.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    roc.name = f"ROC_{length}"
    roc.category = "momentum"

    return roc

# 设置 ROC 函数的文档字符串
roc.__doc__ = \
"""Rate of Change (ROC)

Rate of Change is an indicator is also referred to as Momentum (yeah, confusingly).
It is a pure momentum oscillator that measures the percent change in price with the
previous price 'n' (or length) periods ago.

Sources:
    https://www.tradingview.com/wiki/Rate_of_Change_(ROC)

Calculation:
    Default Inputs:
        length=1
    MOM = Momentum
    ROC = 100 * MOM(close, length) / close.shift(length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    scalar (float): How much to magnify. Default: 100
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