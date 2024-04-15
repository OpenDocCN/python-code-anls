# `.\pandas-ta\pandas_ta\candles\cdl_doji.py`

```
# -*- coding: utf-8 -*-

# 从 pandas_ta.overlap 模块导入 sma 函数
# 从 pandas_ta.utils 模块导入 get_offset, high_low_range, is_percent, real_body, verify_series 函数
from pandas_ta.overlap import sma
from pandas_ta.utils import get_offset, high_low_range, is_percent
from pandas_ta.utils import real_body, verify_series

# 定义一个名为 cdl_doji 的函数，用于识别 Doji 蜡烛
def cdl_doji(open_, high, low, close, length=None, factor=None, scalar=None, asint=True, offset=None, **kwargs):
    """Candle Type: Doji"""
    # 验证参数
   length = int(length) if length and length > 0 else 10
    factor = float(factor) if is_percent(factor) else 10
    scalar = float(scalar) if scalar else 100
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    naive = kwargs.pop("naive", False)

    # 如果输入数据有缺失，则返回空
    if open_ is None or high is None or low is None or close is None: return

    # 计算结果
    body = real_body(open_, close).abs()
    hl_range = high_low_range(high, low).abs()
    hl_range_avg = sma(hl_range, length)
    doji = body < 0.01 * factor * hl_range_avg

    # 如果 naive 为 True，则处理前 length 个数据
    if naive:
        doji.iloc[:length] = body < 0.01 * factor * hl_range
    # 如果 asint 为 True，则将结果转换为整数
    if asint:
        doji = scalar * doji.astype(int)

    # 偏移结果
    if offset != 0:
        doji = doji.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        doji.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        doji.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置结果的名称和类别
    doji.name = f"CDL_DOJI_{length}_{0.01 * factor}"
    doji.category = "candles"

    return doji

# 设置 cdl_doji 函数的文档字符串
cdl_doji.__doc__ = \
"""Candle Type: Doji

A candle body is Doji, when it's shorter than 10% of the
average of the 10 previous candles' high-low range.

Sources:
    TA-Lib: 96.56% Correlation

Calculation:
    Default values:
        length=10, percent=10 (0.1), scalar=100
    ABS = Absolute Value
    SMA = Simple Moving Average

    BODY = ABS(close - open)
    HL_RANGE = ABS(high - low)

    DOJI = scalar IF BODY < 0.01 * percent * SMA(HL_RANGE, length) ELSE 0

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 10
    factor (float): Doji value. Default: 100
    scalar (float): How much to magnify. Default: 100
    asint (bool): Keep results numerical instead of boolean. Default: True

Kwargs:
    naive (bool, optional): If True, prefills potential Doji less than
        the length if less than a percentage of it's high-low range.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: CDL_DOJI column.
"""
```