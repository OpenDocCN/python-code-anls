# `.\pandas-ta\pandas_ta\statistics\stdev.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 导入 sqrt 函数，并将其命名为 npsqrt
from numpy import sqrt as npsqrt
# 从 variance 模块导入 variance 函数
from .variance import variance
# 从 pandas_ta 模块导入 Imports 类
from pandas_ta import Imports
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义 stdev 函数，用于计算标准差
def stdev(close, length=None, ddof=None, talib=None, offset=None, **kwargs):
    """Indicator: Standard Deviation"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为 30
    length = int(length) if length and length > 0 else 30
    # 如果 ddof 是整数且大于等于 0 且小于 length，则将其转换为整数，否则设为 1
    ddof = int(ddof) if isinstance(ddof, int) and ddof >= 0 and ddof < length else 1
    # 验证 close 参数是否为有效的 Series，并根据 length 进行截断
    close = verify_series(close, length)
    # 获取 offset 参数
    offset = get_offset(offset)
    # 如果 talib 存在且为布尔值，则使用 talib 参数值，否则默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为 None，则返回
    if close is None: return

    # 计算结果
    # 如果 Imports 中有 "talib" 并且 mode_tal 为 True
    if Imports["talib"] and mode_tal:
        # 从 talib 中导入 STDDEV 函数，并计算标准差
        from talib import STDDEV
        stdev = STDDEV(close, length)
    else:
        # 否则使用自定义的 variance 函数计算方差，然后对结果应用平方根
        stdev = variance(close=close, length=length, ddof=ddof).apply(npsqrt)

    # 偏移结果
    if offset != 0:
        stdev = stdev.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        stdev.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stdev.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    stdev.name = f"STDEV_{length}"
    stdev.category = "statistics"

    # 返回结果
    return stdev


# 设置 stdev 函数的文档字符串
stdev.__doc__ = \
"""Rolling Standard Deviation

Sources:

Calculation:
    Default Inputs:
        length=30
    VAR = Variance
    STDEV = variance(close, length).apply(np.sqrt)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    ddof (int): Delta Degrees of Freedom.
                The divisor used in calculations is N - ddof,
                where N represents the number of elements. Default: 1
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