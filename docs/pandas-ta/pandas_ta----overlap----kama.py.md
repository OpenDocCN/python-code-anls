# `.\pandas-ta\pandas_ta\overlap\kama.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 库中的 nan 别名 npNaN
from numpy import nan as npNaN
# 导入 pandas 库中的 Series 类
from pandas import Series
# 从 pandas_ta 库中的 utils 模块中导入 get_drift、get_offset、non_zero_range、verify_series 函数
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series


# 定义 Kaufman's Adaptive Moving Average (KAMA) 函数
def kama(close, length=None, fast=None, slow=None, drift=None, offset=None, **kwargs):
    """Indicator: Kaufman's Adaptive Moving Average (KAMA)"""
    # 验证参数
    # 如果 length 不为空且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 如果 fast 不为空且大于 0，则将其转换为整数，否则设为默认值 2
    fast = int(fast) if fast and fast > 0 else 2
    # 如果 slow 不为空且大于 0，则将其转换为整数，否则设为默认值 30
    slow = int(slow) if slow and slow > 0 else 30
    # 验证 close 序列，长度为 fast、slow、length 中的最大值
    close = verify_series(close, max(fast, slow, length))
    # 获取 drift 参数
    drift = get_drift(drift)
    # 获取 offset 参数
    offset = get_offset(offset)

    # 如果 close 为空，则返回空
    if close is None: return

    # 计算结果
    # 定义 weight 函数，用于计算权重
    def weight(length: int) -> float:
        return 2 / (length + 1)

    # 计算 fast 和 slow 的权重
    fr = weight(fast)
    sr = weight(slow)

    # 计算绝对差和同侧差
    abs_diff = non_zero_range(close, close.shift(length)).abs()
    peer_diff = non_zero_range(close, close.shift(drift)).abs()
    peer_diff_sum = peer_diff.rolling(length).sum()
    er = abs_diff / peer_diff_sum
    x = er * (fr - sr) + sr
    sc = x * x

    # 获取 close 序列的长度
    m = close.size
    # 初始化结果列表，前 length-1 个值为 npNaN，最后一个值为 0
    result = [npNaN for _ in range(0, length - 1)] + [0]
    # 遍历计算 KAMA
    for i in range(length, m):
        result.append(sc.iloc[i] * close.iloc[i] + (1 - sc.iloc[i]) * result[i - 1])

    # 将结果转换为 Series 类型，索引为 close 序列的索引
    kama = Series(result, index=close.index)

    # 偏移结果
    if offset != 0:
        kama = kama.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        kama.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        kama.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    kama.name = f"KAMA_{length}_{fast}_{slow}"
    kama.category = "overlap"

    # 返回 KAMA 序列
    return kama


# 设置 KAMA 函数的文档字符串
kama.__doc__ = \
"""Kaufman's Adaptive Moving Average (KAMA)

Developed by Perry Kaufman, Kaufman's Adaptive Moving Average (KAMA) is a moving average
designed to account for market noise or volatility. KAMA will closely follow prices when
the price swings are relatively small and the noise is low. KAMA will adjust when the
price swings widen and follow prices from a greater distance. This trend-following indicator
can be used to identify the overall trend, time turning points and filter price movements.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:kaufman_s_adaptive_moving_average
    https://www.tradingview.com/script/wZGOIz9r-REPOST-Indicators-3-Different-Adaptive-Moving-Averages/

Calculation:
    Default Inputs:
        length=10

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    fast (int): Fast MA period. Default: 2
    slow (int): Slow MA period. Default: 30
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```