# `.\pandas-ta\pandas_ta\overlap\sma.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas_ta 库中导入 Imports 对象
from pandas_ta import Imports
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义简单移动平均（Simple Moving Average，SMA）指标函数
def sma(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Simple Moving Average (SMA)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则默认为 10
    length = int(length) if length and length > 0 else 10
    # 如果 kwargs 中存在 "min_periods" 并且不为 None，则将其转换为整数，否则默认为 length
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证 close 序列，确保其长度至少为 length 或 min_periods
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)
    # 如果 talib 存在且为布尔类型，则将其转换为布尔值，否则默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为 None，则返回
    if close is None: return

    # 计算结果
    # 如果 Imports["talib"] 为 True 且 mode_tal 为 True，则使用 TA Lib 库中的 SMA 函数计算移动平均值
    if Imports["talib"] and mode_tal:
        from talib import SMA
        sma = SMA(close, length)
    else:
        # 否则，使用 pandas 库中的 rolling 和 mean 方法计算移动平均值
        sma = close.rolling(length, min_periods=min_periods).mean()

    # 偏移
    # 如果偏移量不为 0，则对移动平均值进行偏移
    if offset != 0:
        sma = sma.shift(offset)

    # 处理填充
    # 如果 kwargs 中存在 "fillna"，则使用指定的值填充 NaN 值
    if "fillna" in kwargs:
        sma.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中存在 "fill_method"，则使用指定的填充方法填充 NaN 值
    if "fill_method" in kwargs:
        sma.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和类别
    # 设置移动平均值序列的名称为 "SMA_长度"
    sma.name = f"SMA_{length}"
    # 设置移动平均值序列的类别为 "overlap"
    sma.category = "overlap"

    # 返回移动平均值序列
    return sma

# 设置 sma 函数的文档字符串
sma.__doc__ = \
"""Simple Moving Average (SMA)

The Simple Moving Average is the classic moving average that is the equally
weighted average over n periods.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

Calculation:
    Default Inputs:
        length=10
    SMA = SUM(close, length) / length

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```