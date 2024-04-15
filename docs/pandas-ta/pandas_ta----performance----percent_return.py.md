# `.\pandas-ta\pandas_ta\performance\percent_return.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义一个名为 percent_return 的函数，用于计算百分比收益率
def percent_return(close, length=None, cumulative=None, offset=None, **kwargs):
    """Indicator: Percent Return"""
    
    # 验证参数
    length = int(length) if length and length > 0 else 1
    cumulative = bool(cumulative) if cumulative is not None and cumulative else False
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    # 计算结果
    if cumulative:
        pct_return = (close / close.iloc[0]) - 1
    else:
        pct_return = close.pct_change(length) # (close / close.shift(length)) - 1

    # 偏移
    if offset != 0:
        pct_return = pct_return.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        pct_return.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        pct_return.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    pct_return.name = f"{'CUM' if cumulative else ''}PCTRET_{length}"
    pct_return.category = "performance"

    return pct_return

# 设置 percent_return 函数的文档字符串
percent_return.__doc__ = \
"""Percent Return

Calculates the percent return of a Series.
See also: help(df.ta.percent_return) for additional **kwargs a valid 'df'.

Sources:
    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

Calculation:
    Default Inputs:
        length=1, cumulative=False
    PCTRET = close.pct_change(length)
    CUMPCTRET = PCTRET.cumsum() if cumulative

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    cumulative (bool): If True, returns the cumulative returns. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```