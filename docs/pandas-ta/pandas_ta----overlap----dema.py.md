# `.\pandas-ta\pandas_ta\overlap\dema.py`

```
# -*- coding: utf-8 -*-
# 导入 ema 函数
from .ema import ema
# 导入 Imports 模块
from pandas_ta import Imports
# 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 dema 函数，计算 Double Exponential Moving Average (DEMA)
def dema(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Double Exponential Moving Average (DEMA)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 验证 close 数据类型为 Series，并且长度符合要求
    close = verify_series(close, length)
    # 获取 offset 值
    offset = get_offset(offset)
    # 判断是否使用 talib 模式
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果导入了 talib 并且使用 talib 模式，则调用 talib 中的 DEMA 函数
        from talib import DEMA
        dema = DEMA(close, length)
    else:
        # 否则，分别计算两个 EMA 值
        ema1 = ema(close=close, length=length)
        ema2 = ema(close=ema1, length=length)
        # 计算 DEMA 值
        dema = 2 * ema1 - ema2

    # 对结果进行偏移
    if offset != 0:
        dema = dema.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        dema.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        dema.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    dema.name = f"DEMA_{length}"
    dema.category = "overlap"

    return dema

# 设置 dema 函数的文档字符串
dema.__doc__ = \
"""Double Exponential Moving Average (DEMA)

The Double Exponential Moving Average attempts to a smoother average with less
lag than the normal Exponential Moving Average (EMA).

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)

    DEMA = 2 * ema1 - ema2

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
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