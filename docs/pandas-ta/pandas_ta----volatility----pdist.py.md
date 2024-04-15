# `.\pandas-ta\pandas_ta\volatility\pdist.py`

```
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 模块导入 get_drift, get_offset, non_zero_range, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series

# 定义函数 pdist，用于计算价格距离（PDIST）
def pdist(open_, high, low, close, drift=None, offset=None, **kwargs):
    """Indicator: Price Distance (PDIST)"""
    # 验证参数的有效性，确保它们都是 pd.Series 类型
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    # 获取漂移和偏移值，如果未提供，则使用默认值
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 计算结果
    # PDIST = 2 * (high - low) - |close - open| + |open - close[drift]|
    pdist = 2 * non_zero_range(high, low)
    pdist += non_zero_range(open_, close.shift(drift)).abs()
    pdist -= non_zero_range(close, open_).abs()

    # 对结果进行偏移处理
    if offset != 0:
        pdist = pdist.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        pdist.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        pdist.fillna(method=kwargs["fill_method"], inplace=True)

    # 指定结果的名称和分类
    pdist.name = "PDIST"
    pdist.category = "volatility"

    return pdist

# 为 pdist 函数添加文档字符串
pdist.__doc__ = \
"""Price Distance (PDIST)

Measures the "distance" covered by price movements.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/pricedistance/

Calculation:
    Default Inputs:
        drift=1

    PDIST = 2(high - low) - ABS(close - open) + ABS(open - close[drift])

Args:
    open_ (pd.Series): Series of 'opens's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```