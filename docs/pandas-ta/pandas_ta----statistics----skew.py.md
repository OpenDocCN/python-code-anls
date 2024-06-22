# `.\pandas-ta\pandas_ta\statistics\skew.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def skew(close, length=None, offset=None, **kwargs):
    """Indicator: Skew"""
    # 验证参数
    # 如果 length 存在且大于0，则将其转换为整数，否则默认为30
    length = int(length) if length and length > 0 else 30
    # 如果 kwargs 中包含 "min_periods" 并且其值不为 None，则将其转换为整数，否则默认为 length
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证 close 是否为 pd.Series 类型，并保证其长度不小于 length 和 min_periods 中的较大值
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为 None，则返回空值
    if close is None: return

    # 计算结果
    # 计算 close 的滚动 skewness（偏度）
    skew = close.rolling(length, min_periods=min_periods).skew()

    # 偏移结果
    # 如果偏移量不为零，则对 skew 进行偏移
    if offset != 0:
        skew = skew.shift(offset)

    # 处理填充值
    # 如果 kwargs 中包含 "fillna"，则使用指定值填充缺失值
    if "fillna" in kwargs:
        skew.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中包含 "fill_method"，则使用指定的填充方法填充缺失值
    if "fill_method" in kwargs:
        skew.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和类别
    # 设置 skew 的名称为 "SKEW_长度"
    skew.name = f"SKEW_{length}"
    # 设置 skew 的类别为 "statistics"
    skew.category = "statistics"

    # 返回结果
    return skew


# 更新 skew 函数的文档字符串
skew.__doc__ = \
"""Rolling Skew

Sources:

Calculation:
    Default Inputs:
        length=30
    SKEW = close.rolling(length).skew()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```