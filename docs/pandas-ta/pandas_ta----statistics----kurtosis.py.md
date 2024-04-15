# `.\pandas-ta\pandas_ta\statistics\kurtosis.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 kurtosis，计算某个时间段内的峰度
def kurtosis(close, length=None, offset=None, **kwargs):
    """Indicator: Kurtosis"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数；否则将 length 设为默认值 30
    length = int(length) if length and length > 0 else 30
    # 如果 kwargs 中有 "min_periods" 参数且不为 None，则将其转换为整数；否则将 min_periods 设为 length 的值
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证 close 参数，并确保其长度不小于 length 和 min_periods 的最大值
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为 None，则返回空值
    if close is None: return

    # 计算结果
    # 计算 close 的滚动窗口长度为 length 的峰度，并使用 min_periods 参数指定最小期数
    kurtosis = close.rolling(length, min_periods=min_periods).kurt()

    # 处理偏移
    if offset != 0:
        # 对计算的峰度结果进行偏移
        kurtosis = kurtosis.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        # 使用指定的值填充缺失值
        kurtosis.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        # 使用指定的填充方法填充缺失值
        kurtosis.fillna(method=kwargs["fill_method"], inplace=True)

    # 设定指标的名称和类别
    kurtosis.name = f"KURT_{length}"
    kurtosis.category = "statistics"

    # 返回计算结果
    return kurtosis


# 为函数 kurtosis 添加文档字符串
kurtosis.__doc__ = \
"""Rolling Kurtosis

Sources:

Calculation:
    Default Inputs:
        length=30
    KURTOSIS = close.rolling(length).kurt()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```