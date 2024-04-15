# `.\pandas-ta\pandas_ta\volatility\massi.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.overlap 模块导入 ema 函数
from pandas_ta.overlap import ema
# 从 pandas_ta.utils 模块导入 get_offset、non_zero_range、verify_series 函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series


# 定义一个函数 massi，用于计算 Mass Index（MASSI）
def massi(high, low, fast=None, slow=None, offset=None, **kwargs):
    """Indicator: Mass Index (MASSI)"""
    # 验证参数的有效性
    # 如果 fast 有值且大于 0，则将其转换为整数，否则默认为 9
    fast = int(fast) if fast and fast > 0 else 9
    # 如果 slow 有值且大于 0，则将其转换为整数，否则默认为 25
    slow = int(slow) if slow and slow > 0 else 25
    # 如果 slow 小于 fast，则交换它们的值
    if slow < fast:
        fast, slow = slow, fast
    # 计算参数的最大值
    _length = max(fast, slow)
    # 验证 high 和 low 是否为有效序列
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    # 获取偏移量
    offset = get_offset(offset)
    # 移除 kwargs 中的 "length" 键
    if "length" in kwargs: kwargs.pop("length")

    # 如果 high 或 low 为 None，则返回
    if high is None or low is None: return

    # 计算结果
    # 计算高低价范围
    high_low_range = non_zero_range(high, low)
    # 计算高低价范围的 EMA
    hl_ema1 = ema(close=high_low_range, length=fast, **kwargs)
    # 计算高低价范围的 EMA 的 EMA
    hl_ema2 = ema(close=hl_ema1, length=fast, **kwargs)

    # 计算 hl_ratio
    hl_ratio = hl_ema1 / hl_ema2
    # 计算 MASSI
    massi = hl_ratio.rolling(slow, min_periods=slow).sum()

    # 调整偏移量
    if offset != 0:
        massi = massi.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        massi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        massi.fillna(method=kwargs["fill_method"], inplace=True)

    # 给结果命名并归类
    massi.name = f"MASSI_{fast}_{slow}"
    massi.category = "volatility"

    return massi


# 将 massi 函数的文档字符串重新赋值，用于说明该函数的功能、计算方法以及参数等信息
massi.__doc__ = \
"""Mass Index (MASSI)

The Mass Index is a non-directional volatility indicator that utilitizes the
High-Low Range to identify trend reversals based on range expansions.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
    mi = sum(ema(high - low, 9) / ema(ema(high - low, 9), 9), length)

Calculation:
    Default Inputs:
        fast: 9, slow: 25
    EMA = Exponential Moving Average
    hl = high - low
    hl_ema1 = EMA(hl, fast)
    hl_ema2 = EMA(hl_ema1, fast)
    hl_ratio = hl_ema1 / hl_ema2
    MASSI = SUM(hl_ratio, slow)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    fast (int): The short period. Default: 9
    slow (int): The long period. Default: 25
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```