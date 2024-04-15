# `.\pandas-ta\pandas_ta\trend\vhf.py`

```py
# -*- coding: utf-8 -*-
# 导入 fabs 函数并重命名为 npFabs
from numpy import fabs as npFabs
# 从 pandas_ta.utils 模块中导入 get_drift、get_offset、non_zero_range、verify_series 函数
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series


# 定义垂直水平过滤器（VHF）指标函数
def vhf(close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Vertical Horizontal Filter (VHF)"""
    # 验证参数
    # 将长度转换为整数，如果长度大于0，则为指定的长度，否则为默认值28
    length = int(length) if length and length > 0 else 28
    # 验证收盘价序列，长度为指定的长度
    close = verify_series(close, length)
    # 获取漂移值
    drift = get_drift(drift)
    # 获取偏移值
    offset = get_offset(offset)

    # 如果收盘价为空，则返回空值
    if close is None: return

    # 计算结果
    # 最高收盘价
    hcp = close.rolling(length).max()
    # 最低收盘价
    lcp = close.rolling(length).min()
    # 收盘价变化的绝对值
    diff = npFabs(close.diff(drift))
    # 垂直水平过滤器值
    vhf = npFabs(non_zero_range(hcp, lcp)) / diff.rolling(length).sum()

    # 偏移
    if offset != 0:
        vhf = vhf.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        vhf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vhf.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    vhf.name = f"VHF_{length}"
    vhf.category = "trend"

    return vhf


# 设置 VHF 函数的文档字符串
vhf.__doc__ = \
"""Vertical Horizontal Filter (VHF)

VHF was created by Adam White to identify trending and ranging markets.

Sources:
    https://www.incrediblecharts.com/indicators/vertical_horizontal_filter.php

Calculation:
    Default Inputs:
        length = 28
    HCP = Highest Close Price in Period
    LCP = Lowest Close Price in Period
    Change = abs(Ct - Ct-1)
    VHF = (HCP - LCP) / RollingSum[length] of Change

Args:
    source (pd.Series): Series of prices (usually close).
    length (int): The period length. Default: 28
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```