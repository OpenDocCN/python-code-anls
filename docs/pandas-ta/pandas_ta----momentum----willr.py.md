# `.\pandas-ta\pandas_ta\momentum\willr.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta 库中导入 Imports 类
from pandas_ta import Imports
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def willr(high, low, close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: William's Percent R (WILLR)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则默认为 14
    length = int(length) if length and length > 0 else 14
    # 如果 kwargs 中存在 "min_periods"，并且其值不为 None，则将其转换为整数，否则使用 length 的值
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 确定最终使用的长度为 length 和 min_periods 中的较大值
    _length = max(length, min_periods)
    # 验证 high、low、close 系列，并设置它们的长度为 _length
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    # 获取偏移量
    offset = get_offset(offset)
    # 确定是否使用 TA-Lib
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 high、low、close 中有任何一个为 None，则返回空
    if high is None or low is None or close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果 TA-Lib 可用且 mode_tal 为 True，则使用 TA-Lib 中的 WILLR 函数计算
        from talib import WILLR
        willr = WILLR(high, low, close, length)
    else:
        # 否则，使用自定义方法计算 WILLR
        # 计算长度为 length 的最低低点
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        # 计算长度为 length 的最高高点
        highest_high = high.rolling(length, min_periods=min_periods).max()
        # 计算 WILLR
        willr = 100 * ((close - lowest_low) / (highest_high - lowest_low) - 1)

    # 根据偏移量对结果进行偏移
    if offset != 0:
        willr = willr.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        willr.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        willr.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    willr.name = f"WILLR_{length}"
    willr.category = "momentum"

    # 返回计算结果
    return willr


# 设置函数文档字符串
willr.__doc__ = \
"""William's Percent R (WILLR)

William's Percent R is a momentum oscillator similar to the RSI that
attempts to identify overbought and oversold conditions.

Sources:
    https://www.tradingview.com/wiki/Williams_%25R_(%25R)

Calculation:
    Default Inputs:
        length=20
    LL = low.rolling(length).min()
    HH = high.rolling(length).max()

    WILLR = 100 * ((close - LL) / (HH - LL) - 1)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
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