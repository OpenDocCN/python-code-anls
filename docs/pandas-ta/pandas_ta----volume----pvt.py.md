# `.\pandas-ta\pandas_ta\volume\pvt.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas_ta.momentum 模块中导入 roc 函数
from pandas_ta.momentum import roc
# 从 pandas_ta.utils 模块中导入 get_drift、get_offset、verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义函数 pvt，计算价格-成交量趋势指标（Price-Volume Trend，PVT）
def pvt(close, volume, drift=None, offset=None, **kwargs):
    """Indicator: Price-Volume Trend (PVT)"""
    # 验证参数
    close = verify_series(close)  # 验证并确保 close 是有效的 Series 类型
    volume = verify_series(volume)  # 验证并确保 volume 是有效的 Series 类型
    drift = get_drift(drift)  # 获取漂移参数的值
    offset = get_offset(offset)  # 获取偏移参数的值

    # 计算结果
    # 计算 ROC（收盘价的变化率）并乘以成交量
    pv = roc(close=close, length=drift) * volume
    # 计算 PVT 的累积值
    pvt = pv.cumsum()

    # 调整偏移
    if offset != 0:
        pvt = pvt.shift(offset)  # 将结果向前或向后偏移指定的周期数

    # 处理填充值
    if "fillna" in kwargs:
        pvt.fillna(kwargs["fillna"], inplace=True)  # 使用指定的值填充缺失值
    if "fill_method" in kwargs:
        pvt.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充缺失值

    # 命名并分类化结果
    pvt.name = f"PVT"  # 设置结果 Series 的名称
    pvt.category = "volume"  # 设置结果 Series 的分类

    return pvt  # 返回 PVT 结果的 Series


# 设置函数 pvt 的文档字符串
pvt.__doc__ = \
"""Price-Volume Trend (PVT)

The Price-Volume Trend utilizes the Rate of Change with volume to
and it's cumulative values to determine money flow.

Sources:
    https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)

Calculation:
    Default Inputs:
        drift=1
    ROC = Rate of Change
    pv = ROC(close, drift) * volume
    PVT = pv.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    drift (int): The diff period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```