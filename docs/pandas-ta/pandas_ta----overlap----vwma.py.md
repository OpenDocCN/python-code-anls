# `.\pandas-ta\pandas_ta\overlap\vwma.py`

```
# -*- coding: utf-8 -*-
# 从sma模块中导入sma函数
from .sma import sma
# 从pandas_ta.utils中导入get_offset和verify_series函数
from pandas_ta.utils import get_offset, verify_series

# 定义一个名为vwma的函数，计算Volume Weighted Moving Average（VWMA）
def vwma(close, volume, length=None, offset=None, **kwargs):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # 验证参数
    # 将length转换为整数，如果length存在且大于0，否则设为10
    length = int(length) if length and length > 0 else 10
    # 验证close序列，长度为length
    close = verify_series(close, length)
    # 验证volume序列，长度为length
    volume = verify_series(volume, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果close或volume为None，则返回空
    if close is None or volume is None: return

    # 计算结果
    # 计算pv（close * volume）
    pv = close * volume
    # 计算vwma（pv的SMA / volume的SMA）
    vwma = sma(close=pv, length=length) / sma(close=volume, length=length)

    # 偏移结果
    if offset != 0:
        # 偏移vwma
        vwma = vwma.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        # 使用指定值填充缺失值
        vwma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        # 使用指定的填充方法填充缺失值
        vwma.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和类别
    # 设置vwma的名称为"VWMA_长度"
    vwma.name = f"VWMA_{length}"
    # 设置vwma的类别为"overlap"
    vwma.category = "overlap"

    return vwma

# 设置vwma函数的文档字符串
vwma.__doc__ = \
"""Volume Weighted Moving Average (VWMA)

Volume Weighted Moving Average.

Sources:
    https://www.motivewave.com/studies/volume_weighted_moving_average.htm

Calculation:
    Default Inputs:
        length=10
    SMA = Simple Moving Average
    pv = close * volume
    VWMA = SMA(pv, length) / SMA(volume, length)

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```