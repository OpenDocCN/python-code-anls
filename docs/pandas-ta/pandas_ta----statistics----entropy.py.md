# `.\pandas-ta\pandas_ta\statistics\entropy.py`

```py
# -*- coding: utf-8 -*-
# 导入 log 函数并将其命名为 npLog
from numpy import log as npLog
# 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义熵指标函数，接受收盘价、周期、对数的基数、偏移量和其他参数
def entropy(close, length=None, base=None, offset=None, **kwargs):
    """Indicator: Entropy (ENTP)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 如果 base 存在且大于 0，则将其转换为浮点数，否则设为默认值 2.0
    base = float(base) if base and base > 0 else 2.0
    # 验证收盘价是否是一个有效的 Series，如果不是则返回 None
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果收盘价为 None，则返回 None
    if close is None: return

    # 计算结果
    # 计算每个价格占总和的比例
    p = close / close.rolling(length).sum()
    # 计算熵
    entropy = (-p * npLog(p) / npLog(base)).rolling(length).sum()

    # 偏移结果
    if offset != 0:
        entropy = entropy.shift(offset)

    # 处理填充
    # 如果参数中包含 "fillna"，则使用指定值填充空值
    if "fillna" in kwargs:
        entropy.fillna(kwargs["fillna"], inplace=True)
    # 如果参数中包含 "fill_method"，则使用指定的填充方法填充空值
    if "fill_method" in kwargs:
        entropy.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    entropy.name = f"ENTP_{length}"
    entropy.category = "statistics"

    return entropy


# 设置熵指标函数的文档字符串
entropy.__doc__ = \
"""Entropy (ENTP)

Introduced by Claude Shannon in 1948, entropy measures the unpredictability
of the data, or equivalently, of its average information. A die has higher
entropy (p=1/6) versus a coin (p=1/2).

Sources:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)

Calculation:
    Default Inputs:
        length=10, base=2

    P = close / SUM(close, length)
    E = SUM(-P * npLog(P) / npLog(base), length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    base (float): Logarithmic Base. Default: 2
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```