# `.\pandas-ta\pandas_ta\overlap\sinwma.py`

```py
# -*- coding: utf-8 -*-
# 导入必要的库
from numpy import pi as npPi  # 导入 numpy 库中的 pi 并重命名为 npPi
from numpy import sin as npSin  # 导入 numpy 库中的 sin 函数并重命名为 npSin
from pandas import Series  # 导入 pandas 库中的 Series 类
from pandas_ta.utils import get_offset, verify_series, weights  # 从 pandas_ta.utils 模块导入 get_offset, verify_series, weights 函数


def sinwma(close, length=None, offset=None, **kwargs):
    """Indicator: Sine Weighted Moving Average (SINWMA) by Everget of TradingView"""
    # Validate Arguments
    # 验证参数
    length = int(length) if length and length > 0 else 14  # 将 length 转换为整数，如果未提供或小于等于 0，则设置为默认值 14
    close = verify_series(close, length)  # 验证 close 参数是否为 Series 类型，并确保长度为 length
    offset = get_offset(offset)  # 获取偏移量

    if close is None: return

    # Calculate Result
    # 计算结果
    sines = Series([npSin((i + 1) * npPi / (length + 1)) for i in range(0, length)])  # 生成长度为 length 的正弦周期权重序列
    w = sines / sines.sum()  # 将权重序列标准化

    sinwma = close.rolling(length, min_periods=length).apply(weights(w), raw=True)  # 使用权重计算 SINWMA

    # Offset
    # 偏移结果
    if offset != 0:
        sinwma = sinwma.shift(offset)  # 将结果向前或向后偏移 offset 个周期

    # Handle fills
    # 处理填充
    if "fillna" in kwargs:
        sinwma.fillna(kwargs["fillna"], inplace=True)  # 使用指定的值填充缺失值
    if "fill_method" in kwargs:
        sinwma.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充缺失值

    # Name & Category
    # 设置指标名称和类别
    sinwma.name = f"SINWMA_{length}"  # 设置指标名称
    sinwma.category = "overlap"  # 设置指标类别为 overlap

    return sinwma  # 返回计算结果


sinwma.__doc__ = \
"""Sine Weighted Moving Average (SWMA)

A weighted average using sine cycles. The middle term(s) of the average have the
highest weight(s).

Source:
    https://www.tradingview.com/script/6MWFvnPO-Sine-Weighted-Moving-Average/
    Author: Everget (https://www.tradingview.com/u/everget/)

Calculation:
    Default Inputs:
        length=10

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    sines = Series([sin((i + 1) * pi / (length + 1)) for i in range(0, length)])
    w = sines / sines.sum()
    SINWMA = close.rolling(length, min_periods=length).apply(weights(w), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```