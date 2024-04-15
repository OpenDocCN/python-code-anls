# `.\pandas-ta\pandas_ta\overlap\hwma.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 Series 类
from pandas import Series
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义函数 hwma，计算 Holt-Winter 移动平均值
def hwma(close, na=None, nb=None, nc=None, offset=None, **kwargs):
    """Indicator: Holt-Winter Moving Average"""
    # 验证参数有效性并设置默认值
    na = float(na) if na and na > 0 and na < 1 else 0.2
    nb = float(nb) if nb and nb > 0 and nb < 1 else 0.1
    nc = float(nc) if nc and nc > 0 and nc < 1 else 0.1
    # 验证 close 参数并转换为 pd.Series 类型
    close = verify_series(close)
    # 获取偏移量
    offset = get_offset(offset)

    # 计算结果
    last_a = last_v = 0
    last_f = close.iloc[0]

    result = []
    m = close.size
    for i in range(m):
        # 计算 F
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close.iloc[i]
        # 计算 V
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        # 计算 A
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result.append((F + V + 0.5 * A))
        # 更新变量值
        last_a, last_f, last_v = A, F, V

    # 创建 Series 对象
    hwma = Series(result, index=close.index)

    # 处理偏移
    if offset != 0:
        hwma = hwma.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        hwma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        hwma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    suffix = f"{na}_{nb}_{nc}"
    hwma.name = f"HWMA_{suffix}"
    hwma.category = "overlap"

    return hwma



# 为 hwma 函数添加文档字符串
hwma.__doc__ = \
"""HWMA (Holt-Winter Moving Average)

Indicator HWMA (Holt-Winter Moving Average) is a three-parameter moving average
by the Holt-Winter method; the three parameters should be selected to obtain a
forecast.

This version has been implemented for Pandas TA by rengel8 based
on a publication for MetaTrader 5.

Sources:
    https://www.mql5.com/en/code/20856

Calculation:
    HWMA[i] = F[i] + V[i] + 0.5 * A[i]
    where..
    F[i] = (1-na) * (F[i-1] + V[i-1] + 0.5 * A[i-1]) + na * Price[i]
    V[i] = (1-nb) * (V[i-1] + A[i-1]) + nb * (F[i] - F[i-1])
    A[i] = (1-nc) * A[i-1] + nc * (V[i] - V[i-1])

Args:
    close (pd.Series): Series of 'close's
    na (float): Smoothed series parameter (from 0 to 1). Default: 0.2
    nb (float): Trend parameter (from 0 to 1). Default: 0.1
    nc (float): Seasonality parameter (from 0 to 1). Default: 0.1
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: hwma
"""
```