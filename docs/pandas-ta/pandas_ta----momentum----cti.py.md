# `.\pandas-ta\pandas_ta\momentum\cti.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas 库导入 Series 类
from pandas import Series
# 从 pandas_ta.overlap 模块导入 linreg 函数
from pandas_ta.overlap import linreg
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 cti，计算 Correlation Trend Indicator（相关趋势指标）
def cti(close, length=None, offset=None, **kwargs) -> Series:
    """Indicator: Correlation Trend Indicator"""
    # 如果 length 存在且大于 0，则将其转换为整数类型；否则使用默认值 12
    length = int(length) if length and length > 0 else 12
    # 验证并确保 close 是一个 pd.Series 类型的数据，并且长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为 None，则返回空值
    if close is None: return

    # 计算相关趋势指标
    cti = linreg(close, length=length, r=True)

    # 偏移结果
    if offset != 0:
        cti = cti.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        cti.fillna(method=kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cti.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称
    cti.name = f"CTI_{length}"
    # 设置指标类别
    cti.category = "momentum"
    # 返回相关趋势指标的 Series
    return cti


# 设置 cti 函数的文档字符串
cti.__doc__ = \
"""Correlation Trend Indicator (CTI)

The Correlation Trend Indicator is an oscillator created by John Ehler in 2020.
It assigns a value depending on how close prices in that range are to following
a positively- or negatively-sloping straight line. Values range from -1 to 1.
This is a wrapper for ta.linreg(close, r=True).

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 12
    offset (int): How many periods to offset the result. Default: 0

Returns:
    pd.Series: Series of the CTI values for the given period.
"""
```