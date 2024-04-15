# `.\pandas-ta\pandas_ta\overlap\pwma.py`

```
# -*- coding: utf-8 -*-

# 从 pandas_ta.utils 模块中导入 get_offset, pascals_triangle, verify_series, weights 函数
from pandas_ta.utils import get_offset, pascals_triangle, verify_series, weights

# 定义函数：Pascals Weighted Moving Average (PWMA)
def pwma(close, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Pascals Weighted Moving Average (PWMA)"""
    
    # 验证参数
    length = int(length) if length and length > 0 else 10
    asc = asc if asc else True
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    # 计算结果
    triangle = pascals_triangle(n=length - 1, weighted=True)
    pwma = close.rolling(length, min_periods=length).apply(weights(triangle), raw=True)

    # 偏移
    if offset != 0:
        pwma = pwma.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        pwma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        pwma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    pwma.name = f"PWMA_{length}"
    pwma.category = "overlap"

    return pwma


# 设置函数文档字符串
pwma.__doc__ = \
"""Pascal's Weighted Moving Average (PWMA)

Pascal's Weighted Moving Average is similar to a symmetric triangular window
except PWMA's weights are based on Pascal's Triangle.

Source: Kevin Johnson

Calculation:
    Default Inputs:
        length=10

    def weights(w):
        def _compute(x):
            return np.dot(w * x)
        return _compute

    triangle = utils.pascals_triangle(length + 1)
    PWMA = close.rolling(length)_.apply(weights(triangle), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    asc (bool): Recent values weigh more. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```