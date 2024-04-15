# `.\pandas-ta\pandas_ta\overlap\rma.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 rma，计算 wildeR 的移动平均值（RMA）
def rma(close, length=None, offset=None, **kwargs):
    """Indicator: wildeR's Moving Average (RMA)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为 10
    length = int(length) if length and length > 0 else 10
    # 计算 alpha 值，如果 length 大于 0 则为 1/length，否则为 0.5
    alpha = (1.0 / length) if length > 0 else 0.5
    # 验证 close 参数是否为有效序列，并指定长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    # 使用指数加权移动平均（Exponential Moving Average，EMA）计算 RMA
    rma = close.ewm(alpha=alpha, min_periods=length).mean()

    # 偏移结果
    if offset != 0:
        rma = rma.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        rma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    rma.name = f"RMA_{length}"
    rma.category = "overlap"

    return rma


# 设置 rma 函数的文档字符串
rma.__doc__ = \
"""wildeR's Moving Average (RMA)

The WildeR's Moving Average is simply an Exponential Moving Average (EMA) with
a modified alpha = 1 / length.

Sources:
    https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/WildersSmoothing
    https://www.incrediblecharts.com/indicators/wilder_moving_average.php

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    alpha = 1 / length
    RMA = EMA(close, alpha=alpha)

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