# `.\pandas-ta\pandas_ta\trend\ttm_trend.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas 库导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.overlap 模块导入 hl2 函数
from pandas_ta.overlap import hl2
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义 ttm_trend 函数，用于计算 TTM Trend 指标
def ttm_trend(high, low, close, length=None, offset=None, **kwargs):
    """Indicator: TTM Trend (TTM_TRND)"""

    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则将其设置为默认值 6
    length = int(length) if length and length > 0 else 6
    # 验证 high、low、close 数据是否为 Series，并确保它们的长度为 length
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    # 获取 offset 的偏移量
    offset = get_offset(offset)

    # 如果 high、low、close 有任何一个为 None，则返回空
    if high is None or low is None or close is None: return

    # 计算结果
    # 计算中间价（hl2）的平均值
    trend_avg = hl2(high, low)
    for i in range(1, length):
        trend_avg = trend_avg + hl2(high.shift(i), low.shift(i))

    trend_avg = trend_avg / length

    # 计算 ttm_trend，如果 close 大于 trend_avg，则设置为 1，否则设置为 -1
    tm_trend = (close > trend_avg).astype(int)
    tm_trend.replace(0, -1, inplace=True)

    # 调整结果的偏移量
    if offset != 0:
        tm_trend = tm_trend.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        tm_trend.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        tm_trend.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    tm_trend.name = f"TTM_TRND_{length}"
    tm_trend.category = "momentum"

    # 准备返回的 DataFrame
    data = {tm_trend.name: tm_trend}
    df = DataFrame(data)
    df.name = f"TTMTREND_{length}"
    df.category = tm_trend.category

    return df


# 设置 ttm_trend 函数的文档字符串
ttm_trend.__doc__ = \
"""TTM Trend (TTM_TRND)

This indicator is from John Carters book “Mastering the Trade” and plots the
bars green or red. It checks if the price is above or under the average price of
the previous 5 bars. The indicator should hep you stay in a trade until the
colors chance. Two bars of the opposite color is the signal to get in or out.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/ttm-trend-price/

Calculation:
    Default Inputs:
        length=6
    averageprice = (((high[5]+low[5])/2)+((high[4]+low[4])/2)+((high[3]+low[3])/2)+
    ((high[2]+low[2])/2)+((high[1]+low[1])/2)+((high[6]+low[6])/2)) / 6

    if close > averageprice:
        drawcandle(open,high,low,close) coloured(0,255,0)

    if close < averageprice:
        drawcandle(open,high,low,close) coloured(255,0,0)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 6
    offset (int): How many periods to offset the result. Default: 0
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
Returns:
    pd.DataFrame: ttm_trend.
"""
```