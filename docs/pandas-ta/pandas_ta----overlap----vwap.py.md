# `.\pandas-ta\pandas_ta\overlap\vwap.py`

```
# -*- coding: utf-8 -*-
# 导入依赖库中的hlc3函数
from .hlc3 import hlc3
# 导入辅助函数
from pandas_ta.utils import get_offset, is_datetime_ordered, verify_series

# VWAP函数定义
def vwap(high, low, close, volume, anchor=None, offset=None, **kwargs):
    """Indicator: Volume Weighted Average Price (VWAP)"""
    # 验证参数
    high = verify_series(high)  # 验证high序列
    low = verify_series(low)  # 验证low序列
    close = verify_series(close)  # 验证close序列
    volume = verify_series(volume)  # 验证volume序列
    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"  # 将anchor转换为大写字母，如果为空则默认为"D"
    offset = get_offset(offset)  # 获取offset值

    # 计算typical_price
    typical_price = hlc3(high=high, low=low, close=close)
    # 检查volume序列是否按时间排序
    if not is_datetime_ordered(volume):
        print(f"[!] VWAP volume series is not datetime ordered. Results may not be as expected.")
    # 检查typical_price序列是否按时间排序
    if not is_datetime_ordered(typical_price):
        print(f"[!] VWAP price series is not datetime ordered. Results may not be as expected.")

    # 计算结果
    wp = typical_price * volume  # 计算加权价格
    vwap  = wp.groupby(wp.index.to_period(anchor)).cumsum()  # 计算累积加权价格
    vwap /= volume.groupby(volume.index.to_period(anchor)).cumsum()  # 计算累积成交量

    # 偏移
    if offset != 0:
        vwap = vwap.shift(offset)  # 偏移vwap序列

    # 处理填充值
    if "fillna" in kwargs:
        vwap.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充NaN
    if "fill_method" in kwargs:
        vwap.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充NaN

    # 设置名称和类别
    vwap.name = f"VWAP_{anchor}"  # 设置VWAP的名称
    vwap.category = "overlap"  # 设置VWAP的类别为overlap（重叠型指标）

    return vwap  # 返回VWAP序列


# VWAP文档字符串
vwap.__doc__ = \
"""Volume Weighted Average Price (VWAP)

The Volume Weighted Average Price that measures the average typical price
by volume.  It is typically used with intraday charts to identify general
direction.

Sources:
    https://www.tradingview.com/wiki/Volume_Weighted_Average_Price_(VWAP)
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/volume-weighted-average-price-vwap/
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vwap_intraday

Calculation:
    tp = typical_price = hlc3(high, low, close)
    tpv = tp * volume
    VWAP = tpv.cumsum() / volume.cumsum()

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    anchor (str): How to anchor VWAP. Depending on the index values, it will
        implement various Timeseries Offset Aliases as listed here:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        Default: "D".
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```