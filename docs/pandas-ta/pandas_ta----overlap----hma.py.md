# `.\pandas-ta\pandas_ta\overlap\hma.py`

```py
# -*- coding: utf-8 -*-

# 从 numpy 库中导入 sqrt 函数并重命名为 npSqrt
from numpy import sqrt as npSqrt
# 从当前目录下的 wma 模块中导入 wma 函数
from .wma import wma
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 Hull Moving Average (HMA) 指标函数
def hma(close, length=None, offset=None, **kwargs):
    """Indicator: Hull Moving Average (HMA)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 验证 close 数据为 pd.Series 类型，并且长度符合要求
    close = verify_series(close, length)
    # 获取 offset 值
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    half_length = int(length / 2)
    sqrt_length = int(npSqrt(length))

    # 计算 wmaf 和 wmas
    wmaf = wma(close=close, length=half_length)
    wmas = wma(close=close, length=length)
    # 计算 HMA
    hma = wma(close=2 * wmaf - wmas, length=sqrt_length)

    # 调整偏移量
    if offset != 0:
        hma = hma.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        hma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        hma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    hma.name = f"HMA_{length}"
    hma.category = "overlap"

    return hma

# 设置 HMA 函数的文档字符串
hma.__doc__ = \
"""Hull Moving Average (HMA)

The Hull Exponential Moving Average attempts to reduce or remove lag in moving
averages.

Sources:
    https://alanhull.com/hull-moving-average

Calculation:
    Default Inputs:
        length=10
    WMA = Weighted Moving Average
    half_length = int(0.5 * length)
    sqrt_length = int(sqrt(length))

    wmaf = WMA(close, half_length)
    wmas = WMA(close, length)
    HMA = WMA(2 * wmaf - wmas, sqrt_length)

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