# `.\pandas-ta\pandas_ta\overlap\fwma.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 模块导入 fibonacci、get_offset、verify_series、weights 函数
from pandas_ta.utils import fibonacci, get_offset, verify_series, weights


# 定义 Fibonacci's Weighted Moving Average (FWMA) 函数
def fwma(close, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Fibonacci's Weighted Moving Average (FWMA)"""
    # 验证参数
    # 如果 length 参数存在且大于 0，则将其转换为整数，否则设为默认值 10
    length = int(length) if length and length > 0 else 10
    # 如果 asc 参数存在且为真，则保持其值，否则设为默认值 True
    asc = asc if asc else True
    # 验证 close 参数，并设定长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    # 根据长度生成 Fibonacci 数列，使用加权方法
    fibs = fibonacci(n=length, weighted=True)
    # 计算 FWMA
    fwma = close.rolling(length, min_periods=length).apply(weights(fibs), raw=True)

    # 偏移
    # 如果偏移量不为零，则对 FWMA 进行偏移
    if offset != 0:
        fwma = fwma.shift(offset)

    # 处理填充
    # 如果 kwargs 中包含 fillna 键，则使用指定值进行填充
    if "fillna" in kwargs:
        fwma.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中包含 fill_method 键，则使用指定的填充方法
    if "fill_method" in kwargs:
        fwma.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称与类别
    # 设置 FWMA 的名称为 FWMA_length，类别为 overlap
    fwma.name = f"FWMA_{length}"
    fwma.category = "overlap"

    # 返回 FWMA 结果
    return fwma


# 设置 FWMA 函数的文档字符串
fwma.__doc__ = \
"""Fibonacci's Weighted Moving Average (FWMA)

Fibonacci's Weighted Moving Average is similar to a Weighted Moving Average
(WMA) where the weights are based on the Fibonacci Sequence.

Source: Kevin Johnson

Calculation:
    Default Inputs:
        length=10,

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    fibs = utils.fibonacci(length - 1)
    FWMA = close.rolling(length)_.apply(weights(fibs), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    asc (bool): Recent values weigh more. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```