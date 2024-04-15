# `.\pandas-ta\pandas_ta\candles\cdl_z.py`

```
# -*- coding: utf-8 -*-

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.statistics 模块中导入 zscore 函数
from pandas_ta.statistics import zscore
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义一个名为 cdl_z 的函数，用于计算 Candle Type: Z Score
def cdl_z(open_, high, low, close, length=None, full=None, ddof=None, offset=None, **kwargs):
    """Candle Type: Z Score"""
    # 验证参数
    length = int(length) if length and length > 0 else 30
    ddof = int(ddof) if ddof and ddof >= 0 and ddof < length else 1
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    full = bool(full) if full is not None and full else False

    # 如果输入的数据有缺失，则返回空
    if open_ is None or high is None or low is None or close is None: return

    # 计算结果
    if full:
        length = close.size

    z_open = zscore(open_, length=length, ddof=ddof)
    z_high = zscore(high, length=length, ddof=ddof)
    z_low = zscore(low, length=length, ddof=ddof)
    z_close = zscore(close, length=length, ddof=ddof)

    _full = "a" if full else ""
    _props = _full if full else f"_{length}_{ddof}"
    # 创建一个 DataFrame 对象，包含计算得到的 Z Score 结果
    df = DataFrame({
        f"open_Z{_props}": z_open,
        f"high_Z{_props}": z_high,
        f"low_Z{_props}": z_low,
        f"close_Z{_props}": z_close,
    })

    # 如果 full 为 True，则使用 backfill 方法填充缺失值
    if full:
        df.fillna(method="backfill", axis=0, inplace=True)

    # 如果 offset 不为 0，则对 DataFrame 进行偏移
    if offset != 0:
        df = df.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置 DataFrame 的名称和类别
    df.name = f"CDL_Z{_props}"
    df.category = "candles"

    return df

# 设置 cdl_z 函数的文档字符串
cdl_z.__doc__ = \
"""Candle Type: Z

Normalizes OHLC Candles with a rolling Z Score.

Source: Kevin Johnson

Calculation:
    Default values:
        length=30, full=False, ddof=1
    Z = ZSCORE

    open  = Z( open, length, ddof)
    high  = Z( high, length, ddof)
    low   = Z(  low, length, ddof)
    close = Z(close, length, ddof)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 10

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