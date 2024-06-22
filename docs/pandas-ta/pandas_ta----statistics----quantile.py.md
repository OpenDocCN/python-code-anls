# `.\pandas-ta\pandas_ta\statistics\quantile.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义一个名为 quantile 的函数，用于计算滚动分位数
def quantile(close, length=None, q=None, offset=None, **kwargs):
    """Indicator: Quantile"""  # 函数文档字符串，指示 quantile 函数的作用
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 30
    length = int(length) if length and length > 0 else 30
    # 如果 kwargs 中存在 "min_periods" 键且其值不为 None，则将其转换为整数，否则设为 length 的值
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 如果 q 存在且大于 0 且小于 1，则将其转换为浮点数，否则设为默认值 0.5
    q = float(q) if q and q > 0 and q < 1 else 0.5
    # 验证 close 序列，确保其长度不小于 length 和 min_periods 中的较大值
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为 None，则返回 None
    if close is None: return

    # 计算结果
    # 使用 close 序列进行滚动窗口计算分位数，窗口长度为 length，最小周期数为 min_periods
    quantile = close.rolling(length, min_periods=min_periods).quantile(q)

    # 偏移结果
    # 如果偏移量不为 0，则对 quantile 序列进行偏移
    if offset != 0:
        quantile = quantile.shift(offset)

    # 处理填充值
    # 如果 kwargs 中存在 "fillna" 键，则使用给定值填充 quantile 序列的缺失值
    if "fillna" in kwargs:
        quantile.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中存在 "fill_method" 键，则使用指定的填充方法填充 quantile 序列的缺失值
    if "fill_method" in kwargs:
        quantile.fillna(method=kwargs["fill_method"], inplace=True)

    # 序列命名和分类
    # 设置 quantile 序列的名称为 "QTL_{length}_{q}"
    quantile.name = f"QTL_{length}_{q}"
    # 设置 quantile 序列的分类为 "statistics"
    quantile.category = "statistics"

    return quantile  # 返回 quantile 序列


# 为 quantile 函数添加文档字符串，说明其作用、源以及参数等信息
quantile.__doc__ = \
"""Rolling Quantile

Sources:

Calculation:
    Default Inputs:
        length=30, q=0.5
    QUANTILE = close.rolling(length).quantile(q)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    q (float): The quantile. Default: 0.5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```