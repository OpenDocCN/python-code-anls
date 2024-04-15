# `.\pandas-ta\pandas_ta\trend\qstick.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta.overlap 模块导入 dema, ema, hma, rma, sma 函数
from pandas_ta.overlap import dema, ema, hma, rma, sma
# 从 pandas_ta.utils 模块导入 get_offset, non_zero_range, verify_series 函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series

# 定义 Q Stick 指标函数
def qstick(open_, close, length=None, offset=None, **kwargs):
    """Indicator: Q Stick"""
    # 验证参数
    length = int(length) if length and length > 0 else 10
    ma = kwargs.pop("ma", "sma")
    open_ = verify_series(open_, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if open_ is None or close is None: return

    # 计算结果
    diff = non_zero_range(close, open_)

    if ma == "dema":
        qstick = dema(diff, length=length, **kwargs)
    elif ma == "ema":
        qstick = ema(diff, length=length, **kwargs)
    elif ma == "hma":
        qstick = hma(diff, length=length)
    elif ma == "rma":
        qstick = rma(diff, length=length)
    else: # "sma"
        qstick = sma(diff, length=length)

    # 偏移
    if offset != 0:
        qstick = qstick.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        qstick.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        qstick.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    qstick.name = f"QS_{length}"
    qstick.category = "trend"

    return qstick

# 设置 Q Stick 函数的文档字符串
qstick.__doc__ = \
"""Q Stick

The Q Stick indicator, developed by Tushar Chande, attempts to quantify and
identify trends in candlestick charts.

Sources:
    https://library.tradingtechnologies.com/trade/chrt-ti-qstick.html

Calculation:
    Default Inputs:
        length=10
    xMA is one of: sma (default), dema, ema, hma, rma
    qstick = xMA(close - open, length)

Args:
    open (pd.Series): Series of 'open's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    ma (str): The type of moving average to use. Default: None, which is 'sma'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```