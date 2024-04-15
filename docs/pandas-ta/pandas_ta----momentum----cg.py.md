# `.\pandas-ta\pandas_ta\momentum\cg.py`

```
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 中导入 get_offset, verify_series, weights 函数
from pandas_ta.utils import get_offset, verify_series, weights

# 定义 Center of Gravity (CG) 指标函数
def cg(close, length=None, offset=None, **kwargs):
    """Indicator: Center of Gravity (CG)"""
    # 验证参数
    # 将长度转换为整数，如果长度存在且大于0，则取其值，否则默认为10
    length = int(length) if length and length > 0 else 10
    # 验证 close 数据，并将长度设为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回空
    if close is None: return

    # 计算结果
    # 计算系数列表，范围为 [length-0, length-1, ..., length-length]
    coefficients = [length - i for i in range(0, length)]
    # 计算分子，使用 close 的滚动窗口应用权重函数，raw=True 表示原始数据传递给权重函数
    numerator = -close.rolling(length).apply(weights(coefficients), raw=True)
    # 计算 Center of Gravity (CG)，即分子除以 close 滚动窗口的和
    cg = numerator / close.rolling(length).sum()

    # 偏移结果
    if offset != 0:
        # 对 CG 应用偏移
        cg = cg.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        # 使用指定值填充缺失值
        cg.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        # 使用指定方法填充缺失值
        cg.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    cg.name = f"CG_{length}"
    cg.category = "momentum"

    return cg

# 设置 Center of Gravity (CG) 函数的文档字符串
cg.__doc__ = \
"""Center of Gravity (CG)

The Center of Gravity Indicator by John Ehlers attempts to identify turning
points while exhibiting zero lag and smoothing.

Sources:
    http://www.mesasoftware.com/papers/TheCGOscillator.pdf

Calculation:
    Default Inputs:
        length=10

Args:
    close (pd.Series): Series of 'close's
    length (int): The length of the period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```