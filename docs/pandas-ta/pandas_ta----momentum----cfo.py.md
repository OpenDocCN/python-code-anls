# `.\pandas-ta\pandas_ta\momentum\cfo.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta 库中导入 overlap 模块中的 linreg 函数
from pandas_ta.overlap import linreg
# 从 pandas_ta 库中导入 utils 模块中的 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series


# 定义 Chande Forcast Oscillator (CFO) 函数
def cfo(close, length=None, scalar=None, drift=None, offset=None, **kwargs):
    """Indicator: Chande Forcast Oscillator (CFO)"""
    # 验证参数
    length = int(length) if length and length > 0 else 9
    scalar = float(scalar) if scalar else 100
    # 验证 close 参数，确保其为有效的 pd.Series 对象，并应用 length 长度验证
    close = verify_series(close, length)
    # 获取 drift 参数的值
    drift = get_drift(drift)
    # 获取 offset 参数的值
    offset = get_offset(offset)

    # 如果 close 为 None，则返回 None
    if close is None: return

    # 计算 Series 的线性回归
    cfo = scalar * (close - linreg(close, length=length, tsf=True))
    cfo /= close

    # 偏移
    if offset != 0:
        cfo = cfo.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        cfo.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cfo.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    cfo.name = f"CFO_{length}"
    cfo.category = "momentum"

    return cfo


# 设置 CFO 函数的文档字符串
cfo.__doc__ = \
"""Chande Forcast Oscillator (CFO)

The Forecast Oscillator calculates the percentage difference between the actual
price and the Time Series Forecast (the endpoint of a linear regression line).

Sources:
    https://www.fmlabs.com/reference/default.htm?url=ForecastOscillator.htm

Calculation:
    Default Inputs:
        length=9, drift=1, scalar=100
    LINREG = Linear Regression

    CFO = scalar * (close - LINERREG(length, tdf=True)) / close

Args:
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 9
    scalar (float): How much to magnify. Default: 100
    drift (int): The short period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```