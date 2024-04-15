# `.\pandas-ta\pandas_ta\statistics\variance.py`

```py
# 设置文件编码为 UTF-8
# -*- coding: utf-8 -*-

# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def variance(close, length=None, ddof=None, talib=None, offset=None, **kwargs):
    """Indicator: Variance"""
    # 验证参数
    # 将长度转换为整数，如果长度存在且大于1，则为长度，否则为30
    length = int(length) if length and length > 1 else 30
    # 将 ddof 转换为整数，如果 ddof 是整数且大于等于0且小于长度，则为 ddof，否则为1
    ddof = int(ddof) if isinstance(ddof, int) and ddof >= 0 and ddof < length else 1
    # 如果kwargs中存在"min_periods"，则将其转换为整数，否则为长度
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证close是否为有效Series，长度为最大值（长度，min_periods）
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)
    # 确定是否使用 talib
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果close为None，则返回
    if close is None: return

    # 计算结果
    # 如果 pandas_ta 已导入 talib 并且 mode_tal 为真，则使用 talib 中的 VAR 函数计算方差
    if Imports["talib"] and mode_tal:
        from talib import VAR
        variance = VAR(close, length)
    # 否则，使用 rolling 方法计算滚动方差
    else:
        variance = close.rolling(length, min_periods=min_periods).var(ddof)

    # 偏移结果
    if offset != 0:
        variance = variance.shift(offset)

    # 处理填充
    # 如果kwargs中存在"fillna"，则填充NaN值
    if "fillna" in kwargs:
        variance.fillna(kwargs["fillna"], inplace=True)
    # 如果kwargs中存在"fill_method"，则使用指定的填充方法
    if "fill_method" in kwargs:
        variance.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    variance.name = f"VAR_{length}"
    variance.category = "statistics"

    return variance


# 设置 variance 函数的文档字符串
variance.__doc__ = \
"""Rolling Variance

Sources:

Calculation:
    Default Inputs:
        length=30
    VARIANCE = close.rolling(length).var()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    ddof (int): Delta Degrees of Freedom.
                The divisor used in calculations is N - ddof,
                where N represents the number of elements. Default: 0
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```