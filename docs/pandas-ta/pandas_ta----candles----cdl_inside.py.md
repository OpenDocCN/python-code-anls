# `.\pandas-ta\pandas_ta\candles\cdl_inside.py`

```
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 中导入 candle_color 和 get_offset 函数
from pandas_ta.utils import candle_color, get_offset
# 从 pandas_ta.utils 中导入 verify_series 函数
from pandas_ta.utils import verify_series

# 定义函数 cdl_inside，用于识别 Inside Bar 蜡烛形态
def cdl_inside(open_, high, low, close, asbool=False, offset=None, **kwargs):
    """Candle Type: Inside Bar"""
    # 验证参数是否为 Series 类型
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    # 获取偏移量
    offset = get_offset(offset)

    # 计算结果
    inside = (high.diff() < 0) & (low.diff() > 0)

    # 如果 asbool 为 False，则将结果乘以蜡烛颜色
    if not asbool:
        inside *= candle_color(open_, close)

    # 偏移结果
    if offset != 0:
        inside = inside.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        inside.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        inside.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置结果的名称和类别
    inside.name = f"CDL_INSIDE"
    inside.category = "candles"

    return inside

# 设置 cdl_inside 函数的文档字符串
cdl_inside.__doc__ = \
"""Candle Type: Inside Bar

An Inside Bar is a bar that is engulfed by the prior highs and lows of it's
previous bar. In other words, the current bar is smaller than it's previous bar.
Set asbool=True if you want to know if it is an Inside Bar. Note by default
asbool=False so this returns a 0 if it is not an Inside Bar, 1 if it is an
Inside Bar and close > open, and -1 if it is an Inside Bar but close < open.

Sources:
    https://www.tradingview.com/script/IyIGN1WO-Inside-Bar/

Calculation:
    Default Inputs:
        asbool=False
    inside = (high.diff() < 0) & (low.diff() > 0)

    if not asbool:
        inside *= candle_color(open_, close)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    asbool (bool): Returns the boolean result. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""
```