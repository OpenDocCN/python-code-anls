# `.\pandas-ta\pandas_ta\overlap\mcgd.py`

```py
# -*- coding: utf-8 -*-
# 导入所需模块和函数
from pandas_ta.utils import get_offset, verify_series


def mcgd(close, length=None, offset=None, c=None, **kwargs):
    """Indicator: McGinley Dynamic Indicator"""
    # 验证参数有效性
    # 如果 length 存在且大于 0，则将其转换为整数；否则，默认为 10
    length = int(length) if length and length > 0 else 10
    # 如果 c 存在且在 0 到 1 之间，则将其转换为浮点数；否则，默认为 1
    c = float(c) if c and 0 < c <= 1 else 1
    # 验证 close 是否为有效的 Series，并将其长度限制为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为 None，则返回空值
    if close is None: return

    # 计算结果
    # 复制 close Series，避免直接修改原始数据
    close = close.copy()

    # 定义 McGinley Dynamic Indicator 计算函数
    def mcg_(series):
        # 计算分母
        denom = (c * length * (series.iloc[1] / series.iloc[0]) ** 4)
        # 计算 McGinley Dynamic Indicator
        series.iloc[1] = (series.iloc[0] + ((series.iloc[1] - series.iloc[0]) / denom))
        return series.iloc[1]

    # 应用 mcg_ 函数到 rolling window 上，计算 McGinley Dynamic Indicator
    mcg_cell = close[0:].rolling(2, min_periods=2).apply(mcg_, raw=False)
    # 将第一个值添加回结果 Series 中
    mcg_ds = close[:1].append(mcg_cell[1:])

    # 偏移结果 Series
    if offset != 0:
        mcg_ds = mcg_ds.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        mcg_ds.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        mcg_ds.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置结果 Series 的名称和类别
    mcg_ds.name = f"MCGD_{length}"
    mcg_ds.category = "overlap"

    return mcg_ds


# 设置 McGinley Dynamic Indicator 的文档字符串
mcgd.__doc__ = \
"""McGinley Dynamic Indicator

The McGinley Dynamic looks like a moving average line, yet it is actually a
smoothing mechanism for prices that minimizes price separation, price whipsaws,
and hugs prices much more closely. Because of the calculation, the Dynamic Line
speeds up in down markets as it follows prices yet moves more slowly in up
markets. The indicator was designed by John R. McGinley, a Certified Market
Technician and former editor of the Market Technicians Association's Journal
of Technical Analysis.

Sources:
    https://www.investopedia.com/articles/forex/09/mcginley-dynamic-indicator.asp

Calculation:
    Default Inputs:
        length=10
        offset=0
        c=1

    def mcg_(series):
        denom = (constant * length * (series.iloc[1] / series.iloc[0]) ** 4)
        series.iloc[1] = (series.iloc[0] + ((series.iloc[1] - series.iloc[0]) / denom))
        return series.iloc[1]
    mcg_cell = close[0:].rolling(2, min_periods=2).apply(mcg_, raw=False)
    mcg_ds = close[:1].append(mcg_cell[1:])

Args:
    close (pd.Series): Series of 'close's
    length (int): Indicator's period. Default: 10
    offset (int): Number of periods to offset the result. Default: 0
    c (float): Multiplier for the denominator, sometimes set to 0.6. Default: 1

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```