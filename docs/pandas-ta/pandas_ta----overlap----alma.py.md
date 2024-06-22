# `.\pandas-ta\pandas_ta\overlap\alma.py`

```py
# -*- coding: utf-8 -*-
# 从 numpy 库中导入 exp 函数并重命名为 npExp
from numpy import exp as npExp
# 从 numpy 库中导入 nan 常量并重命名为 npNaN
from numpy import nan as npNaN
# 从 pandas 库中导入 Series 类
from pandas import Series
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def alma(close, length=None, sigma=None, distribution_offset=None, offset=None, **kwargs):
    """Indicator: Arnaud Legoux Moving Average (ALMA)"""
    # 验证参数
    # 将长度转换为整数，如果长度存在且大于 0；否则默认为 10
    length = int(length) if length and length > 0 else 10
    # 将 sigma 转换为浮点数，如果 sigma 存在且大于 0；否则默认为 6.0
    sigma = float(sigma) if sigma and sigma > 0 else 6.0
    # 将 distribution_offset 转换为浮点数，如果 distribution_offset 存在且大于 0；否则默认为 0.85
    distribution_offset = float(distribution_offset) if distribution_offset and distribution_offset > 0 else 0.85
    # 验证 close 序列，长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，返回空
    if close is None: return

    # 预先计算
    m = distribution_offset * (length - 1)
    s = length / sigma
    wtd = list(range(length))
    for i in range(0, length):
        # 计算权重（窗口）
        wtd[i] = npExp(-1 * ((i - m) * (i - m)) / (2 * s * s))

    # 计算结果
    # 初始化结果为长度-1个 NaN 和 1 个 0 组成的列表
    result = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, close.size):
        window_sum = 0
        cum_sum = 0
        for j in range(0, length):
            # 计算窗口和
            window_sum = window_sum + wtd[j] * close.iloc[i - j]
            # 计算累积和
            cum_sum = cum_sum + wtd[j]

        # 计算 ALMA
        almean = window_sum / cum_sum
        # 如果 i 等于长度，则将结果列表追加 NaN，否则追加 almean
        result.append(npNaN) if i == length else result.append(almean)

    # 创建 ALMA Series 对象
    alma = Series(result, index=close.index)

    # 处理偏移
    if offset != 0:
        alma = alma.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        alma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        alma.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和分类
    alma.name = f"ALMA_{length}_{sigma}_{distribution_offset}"
    alma.category = "overlap"

    return alma


# 为 alma 函数添加文档字符串
alma.__doc__ = \
"""Arnaud Legoux Moving Average (ALMA)

The ALMA moving average uses the curve of the Normal (Gauss) distribution, which
can be shifted from 0 to 1. This allows regulating the smoothness and high
sensitivity of the indicator. Sigma is another parameter that is responsible for
the shape of the curve coefficients. This moving average reduces lag of the data
in conjunction with smoothing to reduce noise.

Implemented for Pandas TA by rengel8 based on the source provided below.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/

Calculation:
    refer to provided source

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period, window size. Default: 10
    sigma (float): Smoothing value. Default 6.0
    distribution_offset (float): Value to offset the distribution min 0
        (smoother), max 1 (more responsive). Default 0.85
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:

"""
    # 创建一个 Pandas Series 对象，表示生成了一个新的特征
# 这是一个空的字符串，通常用作多行注释的起始
```