# `.\pandas-ta\pandas_ta\momentum\ao.py`

```
# -*- coding: utf-8 -*-
# 从pandas_ta.overlap模块导入简单移动平均函数sma
from pandas_ta.overlap import sma
# 从pandas_ta.utils模块导入get_offset和verify_series函数
from pandas_ta.utils import get_offset, verify_series


def ao(high, low, fast=None, slow=None, offset=None, **kwargs):
    """Indicator: Awesome Oscillator (AO)"""
    # 验证参数
    # 如果fast存在且大于0，则转换为整数，否则默认为5
    fast = int(fast) if fast and fast > 0 else 5
    # 如果slow存在且大于0，则转换为整数，否则默认为34
    slow = int(slow) if slow and slow > 0 else 34
    # 如果slow小于fast，则交换它们的值
    if slow < fast:
        fast, slow = slow, fast
    # 计算_length为fast和slow中的最大值
    _length = max(fast, slow)
    # 验证high和low的Series长度为_length
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    # 获取offset的偏移量
    offset = get_offset(offset)

    # 如果high或low为None，则返回空
    if high is None or low is None: return

    # 计算结果
    # 计算中间价，即(high + low) / 2
    median_price = 0.5 * (high + low)
    # 计算fast期间的简单移动平均
    fast_sma = sma(median_price, fast)
    # 计算slow期间的简单移动平均
    slow_sma = sma(median_price, slow)
    # 计算AO指标，即fast期间的SMA减去slow期间的SMA
    ao = fast_sma - slow_sma

    # 偏移结果
    if offset != 0:
        # 将结果向前偏移offset个周期
        ao = ao.shift(offset)

    # 处理填充
    # 如果kwargs中有"fillna"参数，则用指定值填充空值
    if "fillna" in kwargs:
        ao.fillna(kwargs["fillna"], inplace=True)
    # 如果kwargs中有"fill_method"参数，则使用指定的填充方法
    if "fill_method" in kwargs:
        ao.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    # 设置AO指标的名称，格式为"AO_{fast}_{slow}"
    ao.name = f"AO_{fast}_{slow}"
    # 设置AO指标的类别为动量
    ao.category = "momentum"

    # 返回AO指标
    return ao


# 更新函数文档字符串
ao.__doc__ = \
"""Awesome Oscillator (AO)

The Awesome Oscillator is an indicator used to measure a security's momentum.
AO is generally used to affirm trends or to anticipate possible reversals.

Sources:
    https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
    https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

Calculation:
    Default Inputs:
        fast=5, slow=34
    SMA = Simple Moving Average
    median = (high + low) / 2
    AO = SMA(median, fast) - SMA(median, slow)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    fast (int): The short period. Default: 5
    slow (int): The long period. Default: 34
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```