# `.\pandas-ta\pandas_ta\statistics\median.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义一个名为 median 的函数，用于计算中位数指标
def median(close, length=None, offset=None, **kwargs):
    """Indicator: Median"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 30
    length = int(length) if length and length > 0 else 30
    # 如果 kwargs 中存在 "min_periods"，且其值不为 None，则将其转换为整数，否则设为 length 的值
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证 close 是否为有效的 Series，并设定最小长度为 length 和 min_periods 中的较大值
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为 None，则返回空值
    if close is None: return

    # 计算结果
    # 计算 close 的滚动中位数，窗口长度为 length，最小观测数为 min_periods
    median = close.rolling(length, min_periods=min_periods).median()

    # 偏移结果
    if offset != 0:
        median = median.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        median.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        median.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    median.name = f"MEDIAN_{length}"
    median.category = "statistics"

    return median

# 设置 median 函数的文档字符串
median.__doc__ = \
"""Rolling Median

Rolling Median of over 'n' periods. Sibling of a Simple Moving Average.

Sources:
    https://www.incrediblecharts.com/indicators/median_price.php

Calculation:
    Default Inputs:
        length=30
    MEDIAN = close.rolling(length).median()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```