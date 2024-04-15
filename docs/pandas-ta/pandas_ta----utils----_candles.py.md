# `.\pandas-ta\pandas_ta\utils\_candles.py`

```
# -*- coding: utf-8 -*-
# 导入 Series 类
from pandas import Series
# 导入 non_zero_range 函数
from ._core import non_zero_range

# 计算蜡烛图的颜色
def candle_color(open_: Series, close: Series) -> Series:
    # 复制收盘价 Series，并将其类型转换为整数
    color = close.copy().astype(int)
    # 当收盘价大于等于开盘价时，将颜色设置为1
    color[close >= open_] = 1
    # 当收盘价小于开盘价时，将颜色设置为-1
    color[close < open_] = -1
    # 返回颜色 Series
    return color

# 计算最高价和最低价的范围
def high_low_range(high: Series, low: Series) -> Series:
    # 调用 non_zero_range 函数计算高低价的范围
    return non_zero_range(high, low)

# 计算实体部分（实体部分指收盘价与开盘价之间的绝对值）
def real_body(open_: Series, close: Series) -> Series:
    # 调用 non_zero_range 函数计算实体部分
    return non_zero_range(close, open_)
```