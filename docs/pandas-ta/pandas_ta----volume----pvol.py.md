# `.\pandas-ta\pandas_ta\volume\pvol.py`

```py
# -*- coding: utf-8 -*- 
# 导入所需的库和函数
from pandas_ta.utils import get_offset, signed_series, verify_series

# 定义函数 pvol，计算价格和成交量的乘积
def pvol(close, volume, offset=None, **kwargs):
    """Indicator: Price-Volume (PVOL)"""
    # 验证参数的有效性
    close = verify_series(close)  # 验证价格序列
    volume = verify_series(volume)  # 验证成交量序列
    offset = get_offset(offset)  # 获取偏移量
    signed = kwargs.pop("signed", False)  # 获取 signed 参数，默认为 False

    # 计算结果
    pvol = close * volume  # 计算价格和成交量的乘积
    if signed:
         pvol *= signed_series(close, 1)  # 如果 signed 为 True，则乘以 close 的符号系列

    # 偏移
    if offset != 0:
        pvol = pvol.shift(offset)  # 对结果进行偏移

    # 处理填充
    if "fillna" in kwargs:
        pvol.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充缺失值
    if "fill_method" in kwargs:
        pvol.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充缺失值

    # 设置名称和分类
    pvol.name = f"PVOL"  # 设置结果的名称
    pvol.category = "volume"  # 设置结果的分类

    return pvol  # 返回计算结果


pvol.__doc__ = \
"""Price-Volume (PVOL)

Returns a series of the product of price and volume.

Calculation:
    if signed:
        pvol = signed_series(close, 1) * close * volume
    else:
        pvol = close * volume

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    signed (bool): Keeps the sign of the difference in 'close's. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```