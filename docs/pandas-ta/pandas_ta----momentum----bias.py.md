# `.\pandas-ta\pandas_ta\momentum\bias.py`

```
# -*- coding: utf-8 -*-
# 导入 pandas_ta 库中的 ma 函数
from pandas_ta.overlap import ma
# 导入 pandas_ta 库中的 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义 Bias 指标函数，接受 close、length、mamode、offset 等参数
def bias(close, length=None, mamode=None, offset=None, **kwargs):
    """Indicator: Bias (BIAS)"""
    # 验证参数合法性
    # 如果未指定 length 或 length 小于等于 0，则默认为 26
    length = int(length) if length and length > 0 else 26
    # 如果未指定 mamode 或 mamode 不是字符串，则默认为 "sma"
    mamode = mamode if isinstance(mamode, str) else "sma"
    # 验证 close 是否为 Series，并指定长度为 length
    close = verify_series(close, length)
    # 获取 offset
    offset = get_offset(offset)

    # 如果 close 为 None，则返回 None
    if close is None: return

    # 计算结果
    # 计算移动平均线，参数为 mamode、close 和 length
    bma = ma(mamode, close, length=length, **kwargs)
    # 计算 Bias，即 (close / bma) - 1
    bias = (close / bma) - 1

    # 偏移
    # 如果 offset 不为 0，则对 Bias 进行偏移
    if offset != 0:
        bias = bias.shift(offset)

    # 处理填充
    # 如果 kwargs 中包含 "fillna"，则使用该值填充 NaN
    if "fillna" in kwargs:
        bias.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中包含 "fill_method"，则使用指定的填充方法
    if "fill_method" in kwargs:
        bias.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    # 指标名称为 "BIAS_移动平均线名称"，类别为 "momentum"
    bias.name = f"BIAS_{bma.name}"
    bias.category = "momentum"

    # 返回 Bias
    return bias


# 设置 Bias 函数的文档字符串
bias.__doc__ = \
"""Bias (BIAS)

Rate of change between the source and a moving average.

Sources:
    Few internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=26, MA='sma'

    BIAS = (close - MA(close, length)) / MA(close, length)
         = (close / MA(close, length)) - 1

Args:
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 26
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    drift (int): The short period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```