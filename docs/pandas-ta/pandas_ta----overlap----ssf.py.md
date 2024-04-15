# `.\pandas-ta\pandas_ta\overlap\ssf.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 库，并将其中的 cos、exp、pi、sqrt 函数别名为 npCos、npExp、npPi、npSqrt
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sqrt as npSqrt
# 从 pandas_ta.utils 模块中导入 get_offset、verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义函数 ssf，实现 Ehler 的超平滑滤波器（SSF）
def ssf(close, length=None, poles=None, offset=None, **kwargs):
    """Indicator: Ehler's Super Smoother Filter (SSF)"""
    # 验证参数
    # 如果 length 不为空且大于 0，则将其转换为整数，否则设为 10
    length = int(length) if length and length > 0 else 10
    # 如果 poles 不为空且在 [2, 3] 中，则将其转换为整数，否则设为 2
    poles = int(poles) if poles in [2, 3] else 2
    # 验证 close 是否为有效的 Series，长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    m = close.size
    # 复制 close 到 ssf
    ssf = close.copy()

    # 根据 poles 的值进行不同的计算
    if poles == 3:
        # 计算参数
        x = npPi / length # x = PI / n
        a0 = npExp(-x) # e^(-x)
        b0 = 2 * a0 * npCos(npSqrt(3) * x) # 2e^(-x)*cos(3^(.5) * x)
        c0 = a0 * a0 # e^(-2x)

        c4 = c0 * c0 # e^(-4x)
        c3 = -c0 * (1 + b0) # -e^(-2x) * (1 + 2e^(-x)*cos(3^(.5) * x))
        c2 = c0 + b0 # e^(-2x) + 2e^(-x)*cos(3^(.5) * x)
        c1 = 1 - c2 - c3 - c4

        # 循环计算 ssf
        for i in range(0, m):
            ssf.iloc[i] = c1 * close.iloc[i] + c2 * ssf.iloc[i - 1] + c3 * ssf.iloc[i - 2] + c4 * ssf.iloc[i - 3]

    else: # poles == 2
        # 计算参数
        x = npPi * npSqrt(2) / length # x = PI * 2^(.5) / n
        a0 = npExp(-x) # e^(-x)
        a1 = -a0 * a0 # -e^(-2x)
        b1 = 2 * a0 * npCos(x) # 2e^(-x)*cos(x)
        c1 = 1 - a1 - b1 # e^(-2x) - 2e^(-x)*cos(x) + 1

        # 循环计算 ssf
        for i in range(0, m):
            ssf.iloc[i] = c1 * close.iloc[i] + b1 * ssf.iloc[i - 1] + a1 * ssf.iloc[i - 2]

    # 偏移结果
    if offset != 0:
        ssf = ssf.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        ssf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ssf.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    ssf.name = f"SSF_{length}_{poles}"
    ssf.category = "overlap"

    return ssf


# 设置函数 ssf 的文档字符串
ssf.__doc__ = \
"""Ehler's Super Smoother Filter (SSF) © 2013

John F. Ehlers's solution to reduce lag and remove aliasing noise with his
research in aerospace analog filter design. This indicator comes with two
versions determined by the keyword poles. By default, it uses two poles but
there is an option for three poles. Since SSF is a (Resursive) Digital Filter,
the number of poles determine how many prior recursive SSF bars to include in
the design of the filter. So two poles uses two prior SSF bars and three poles
uses three prior SSF bars for their filter calculations.

Sources:
    http://www.stockspotter.com/files/PredictiveIndicators.pdf
    https://www.tradingview.com/script/VdJy0yBJ-Ehlers-Super-Smoother-Filter/
    https://www.mql5.com/en/code/588
    https://www.mql5.com/en/code/589

Calculation:
    Default Inputs:
        length=10, poles=[2, 3]

    See the source code or Sources listed above.

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    poles (int): The number of poles to use, either 2 or 3. Default: 2

"""
    offset (int): How many periods to offset the result. Default: 0
# 定义函数参数
Kwargs:
    # fillna 参数：用于指定 pd.DataFrame.fillna 方法中的填充值
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method 参数：用于指定填充方法的类型
    fill_method (value, optional): Type of fill method

# 返回值说明
Returns:
    # 返回一个 Pandas Series 对象，表示生成的新特征
    pd.Series: New feature generated.
```