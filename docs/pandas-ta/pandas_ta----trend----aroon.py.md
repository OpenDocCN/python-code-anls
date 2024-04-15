# `.\pandas-ta\pandas_ta\trend\aroon.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 模块中导入 get_offset, verify_series 函数
from pandas_ta.utils import get_offset, verify_series
# 从 pandas_ta.utils 模块中导入 recent_maximum_index, recent_minimum_index 函数
from pandas_ta.utils import recent_maximum_index, recent_minimum_index

# 定义函数 aroon，计算 Aroon 和 Aroon Oscillator 指标
def aroon(high, low, length=None, scalar=None, talib=None, offset=None, **kwargs):
    """Indicator: Aroon & Aroon Oscillator"""
    # 验证参数
    length = length if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        from talib import AROON, AROONOSC
        aroon_down, aroon_up = AROON(high, low, length)
        aroon_osc = AROONOSC(high, low, length)
    else:
        periods_from_hh = high.rolling(length + 1).apply(recent_maximum_index, raw=True)
        periods_from_ll = low.rolling(length + 1).apply(recent_minimum_index, raw=True)

        aroon_up = aroon_down = scalar
        aroon_up *= 1 - (periods_from_hh / length)
        aroon_down *= 1 - (periods_from_ll / length)
        aroon_osc = aroon_up - aroon_down

    # 处理填充
    if "fillna" in kwargs:
        aroon_up.fillna(kwargs["fillna"], inplace=True)
        aroon_down.fillna(kwargs["fillna"], inplace=True)
        aroon_osc.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        aroon_up.fillna(method=kwargs["fill_method"], inplace=True)
        aroon_down.fillna(method=kwargs["fill_method"], inplace=True)
        aroon_osc.fillna(method=kwargs["fill_method"], inplace=True)

    # 偏移
    if offset != 0:
        aroon_up = aroon_up.shift(offset)
        aroon_down = aroon_down.shift(offset)
        aroon_osc = aroon_osc.shift(offset)

    # 命名和分类
    aroon_up.name = f"AROONU_{length}"
    aroon_down.name = f"AROOND_{length}"
    aroon_osc.name = f"AROONOSC_{length}"

    aroon_down.category = aroon_up.category = aroon_osc.category = "trend"

    # 准备要返回的 DataFrame
    data = {
        aroon_down.name: aroon_down,
        aroon_up.name: aroon_up,
        aroon_osc.name: aroon_osc,
    }
    aroondf = DataFrame(data)
    aroondf.name = f"AROON_{length}"
    aroondf.category = aroon_down.category

    return aroondf


# 设置函数 aroon 的文档字符串
aroon.__doc__ = \
"""Aroon & Aroon Oscillator (AROON)

Aroon attempts to identify if a security is trending and how strong.

Sources:
    https://www.tradingview.com/wiki/Aroon
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/aroon-ar/

Calculation:
    Default Inputs:
        length=1, scalar=100

    recent_maximum_index(x): return int(np.argmax(x[::-1]))
    recent_minimum_index(x): return int(np.argmin(x[::-1]))

    periods_from_hh = high.rolling(length + 1).apply(recent_maximum_index, raw=True)

"""
    # 计算 Aroon 指标中的上升线，使用公式：scalar * (1 - (periods_from_hh / length))
    AROON_UP = scalar * (1 - (periods_from_hh / length))

    # 计算 Aroon 指标中的下降线，使用公式：scalar * (1 - (periods_from_ll / length))
    periods_from_ll = low.rolling(length + 1).apply(recent_minimum_index, raw=True)
    AROON_DN = scalar * (1 - (periods_from_ll / length))

    # 计算 Aroon 指标的震荡值，使用公式：AROON_UP - AROON_DN
    AROON_OSC = AROON_UP - AROON_DN
# 定义函数参数
Args:
    close (pd.Series): 包含'close'价格数据的Series
    length (int): 计算指标的周期，默认为14
    scalar (float): 放大倍数，默认为100
    talib (bool): 如果安装了TA Lib并且talib为True，则返回TA Lib版本，默认为True
    offset (int): 结果的偏移周期数，默认为0

# 定义函数关键字参数
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)的填充值
    fill_method (value, optional): 填充方法的类型

# 返回值
Returns:
    pd.DataFrame: 包含aroon_up、aroon_down、aroon_osc列的DataFrame
```