# `.\pandas-ta\pandas_ta\overlap\tema.py`

```py
# -*- coding: utf-8 -*-

# 从 ema 模块中导入 ema 函数
from .ema import ema
# 从 pandas_ta 模块中导入 Imports 类
from pandas_ta import Imports
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 Triple Exponential Moving Average (TEMA) 指标函数
def tema(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Triple Exponential Moving Average (TEMA)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 验证 close 数据，并设定长度
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 判断是否使用 talib 模式
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果导入了 talib 模块并且使用 talib 模式，则调用 talib 中的 TEMA 函数
        from talib import TEMA
        tema = TEMA(close, length)
    else:
        # 否则，使用 ema 函数计算三次指数移动平均
        ema1 = ema(close=close, length=length, **kwargs)
        ema2 = ema(close=ema1, length=length, **kwargs)
        ema3 = ema(close=ema2, length=length, **kwargs)
        tema = 3 * (ema1 - ema2) + ema3

    # 偏移结果
    if offset != 0:
        tema = tema.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        tema.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        tema.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    tema.name = f"TEMA_{length}"
    tema.category = "overlap"

    return tema

# 设置 TEMA 函数的文档字符串
tema.__doc__ = \
"""Triple Exponential Moving Average (TEMA)

A less laggy Exponential Moving Average.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    TEMA = 3 * (ema1 - ema2) + ema3

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
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