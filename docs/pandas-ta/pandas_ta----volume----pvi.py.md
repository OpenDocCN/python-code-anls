# `.\pandas-ta\pandas_ta\volume\pvi.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta 库中导入动量模块中的 roc 函数
from pandas_ta.momentum import roc
# 从 pandas_ta 库中导入工具模块中的 get_offset, signed_series, verify_series 函数
from pandas_ta.utils import get_offset, signed_series, verify_series


# 定义 Positive Volume Index (PVI) 指标函数
def pvi(close, volume, length=None, initial=None, offset=None, **kwargs):
    """Indicator: Positive Volume Index (PVI)"""
    # 验证参数
    length = int(length) if length and length > 0 else 1
    # initial 默认为 1000
    initial = int(initial) if initial and initial > 0 else 1000
    # 验证 close 和 volume 参数是否为 Series 类型，并且长度符合要求
    close = verify_series(close, length)
    volume = verify_series(volume, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 或 volume 为 None，则返回空
    if close is None or volume is None: return

    # 计算结果
    # 将 volume 序列转换为带符号的序列
    signed_volume = signed_series(volume, 1)
    # 计算 PVI
    pvi = roc(close=close, length=length) * signed_volume[signed_volume > 0].abs()
    # 将 NaN 值填充为 0
    pvi.fillna(0, inplace=True)
    # 将第一个值设置为 initial
    pvi.iloc[0] = initial
    # 对 PVI 序列进行累积求和
    pvi = pvi.cumsum()

    # 偏移
    if offset != 0:
        pvi = pvi.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        pvi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        pvi.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    pvi.name = f"PVI_{length}"
    pvi.category = "volume"

    return pvi


# 设置 PVI 函数的文档字符串
pvi.__doc__ = \
"""Positive Volume Index (PVI)

The Positive Volume Index is a cumulative indicator that uses volume change in
an attempt to identify where smart money is active.
Used in conjunction with NVI.

Sources:
    https://www.investopedia.com/terms/p/pvi.asp

Calculation:
    Default Inputs:
        length=1, initial=1000
    ROC = Rate of Change

    roc = ROC(close, length)
    signed_volume = signed_series(volume, initial=1)
    pvi = signed_volume[signed_volume > 0].abs() * roc_
    pvi.fillna(0, inplace=True)
    pvi.iloc[0]= initial
    pvi = pvi.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period. Default: 13
    initial (int): The short period. Default: 1000
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```