# `.\pandas-ta\pandas_ta\momentum\mom.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta 模块导入 Imports
from pandas_ta import Imports
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义动量指标（Momentum）函数
def mom(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Momentum (MOM)"""
    # 验证参数
    # 如果 length 参数存在且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 验证 close 是否为有效的数据序列，并设置长度
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 设置是否使用 talib 模式
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    # 如果 Imports["talib"] 为真且 mode_tal 为真，则使用 TA Lib 中的 MOM 函数计算动量
    if Imports["talib"] and mode_tal:
        from talib import MOM
        mom = MOM(close, length)
    # 否则，使用差分计算动量
    else:
        mom = close.diff(length)

    # 偏移结果
    if offset != 0:
        mom = mom.shift(offset)

    # 处理填充值
    # 如果 kwargs 中有 "fillna" 参数，则使用指定值填充空值
    if "fillna" in kwargs:
        mom.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中有 "fill_method" 参数，则使用指定填充方法填充空值
    if "fill_method" in kwargs:
        mom.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    # 设置动量指标的名称为 "MOM_长度"，设置分类为 "momentum"
    mom.name = f"MOM_{length}"
    mom.category = "momentum"

    # 返回动量指标结果
    return mom


# 设置动量指标函数的文档字符串
mom.__doc__ = \
"""Momentum (MOM)

Momentum is an indicator used to measure a security's speed (or strength) of
movement.  Or simply the change in price.

Sources:
    http://www.onlinetradingconcepts.com/TechnicalAnalysis/Momentum.html

Calculation:
    Default Inputs:
        length=1
    MOM = close.diff(length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
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