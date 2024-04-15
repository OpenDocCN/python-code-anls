# `.\pandas-ta\pandas_ta\volume\cmf.py`

```
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 导入一些必要的函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series

# 定义 CMF 函数，计算 Chaikin Money Flow 指标
def cmf(high, low, close, volume, open_=None, length=None, offset=None, **kwargs):
    """Indicator: Chaikin Money Flow (CMF)"""
    # 验证参数
    # 确保 length 是整数且大于 0，若未指定则默认为 20
    length = int(length) if length and length > 0 else 20
    # 最小期数为 length 或者 kwargs 中 min_periods 的值，若未指定则为 length
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 取 length 和 min_periods 中的较大值作为真正的 length
    _length = max(length, min_periods)
    # 确保传入的数据是合法的 Series 类型，长度为 _length
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    # 获取偏移量
    offset = get_offset(offset)

    # 若传入的数据有缺失，则返回 None
    if high is None or low is None or close is None or volume is None: return

    # 计算结果
    if open_ is not None:
        # 若存在开盘价数据，则使用开盘价计算 AD
        open_ = verify_series(open_)
        ad = non_zero_range(close, open_)  # 使用开盘价计算 AD
    else:
        # 若不存在开盘价数据，则使用高、低、收盘价计算 AD
        ad = 2 * close - (high + low)  # 使用高、低、收盘价计算 AD

    # 计算 CMF
    ad *= volume / non_zero_range(high, low)
    cmf = ad.rolling(length, min_periods=min_periods).sum()
    cmf /= volume.rolling(length, min_periods=min_periods).sum()

    # 偏移结果
    if offset != 0:
        cmf = cmf.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        cmf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cmf.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名并分类
    cmf.name = f"CMF_{length}"
    cmf.category = "volume"

    # 返回结果
    return cmf

# 设置 CMF 函数的文档字符串
cmf.__doc__ = \
"""Chaikin Money Flow (CMF)

Chailin Money Flow measures the amount of money flow volume over a specific
period in conjunction with Accumulation/Distribution.

Sources:
    https://www.tradingview.com/wiki/Chaikin_Money_Flow_(CMF)
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

Calculation:
    Default Inputs:
        length=20
    if 'open':
        ad = close - open
    else:
        ad = 2 * close - high - low

    hl_range = high - low
    ad = ad * volume / hl_range
    CMF = SUM(ad, length) / SUM(volume, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    open_ (pd.Series): Series of 'open's. Default: None
    length (int): The short period. Default: 20
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```