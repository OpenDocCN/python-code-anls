# `.\pandas-ta\pandas_ta\trend\decreasing.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 模块中导入所需函数和类
from pandas_ta.utils import get_drift, get_offset, is_percent, verify_series

# 定义一个名为 decreasing 的函数，用于计算序列是否递减
def decreasing(close, length=None, strict=None, asint=None, percent=None, drift=None, offset=None, **kwargs):
    """Indicator: Decreasing"""
    # 验证参数
    # 如果 length 参数存在且大于 0，则将其转换为整数，否则设为 1
    length = int(length) if length and length > 0 else 1
    # 如果 strict 参数是布尔类型，则保持不变，否则设为 False
    strict = strict if isinstance(strict, bool) else False
    # 如果 asint 参数是布尔类型，则保持不变，否则设为 True
    asint = asint if isinstance(asint, bool) else True
    # 对 close 序列进行验证，并设定长度为 length
    close = verify_series(close, length)
    # 获取 drift 和 offset 参数的值
    drift = get_drift(drift)
    offset = get_offset(offset)
    # 如果 percent 参数是百分比，则将其转换为浮点数，否则设为 False
    percent = float(percent) if is_percent(percent) else False

    # 如果 close 为 None，则返回 None
    if close is None: return

    # 计算结果
    # 如果 percent 存在，则对 close 序列进行缩放
    close_ = (1 - 0.01 * percent) * close if percent else close
    # 如果 strict 为 True，则进行严格递减的计算
    if strict:
        # 使用循环检查连续递减的情况
        decreasing = close < close_.shift(drift)
        for x in range(3, length + 1):
            decreasing = decreasing & (close.shift(x - (drift + 1)) < close_.shift(x - drift))

        # 填充缺失值为 0，并将结果转换为布尔类型
        decreasing.fillna(0, inplace=True)
        decreasing = decreasing.astype(bool)
    else:
        # 否则，使用简单的递减计算
        decreasing = close_.diff(length) < 0

    # 如果 asint 为 True，则将结果转换为整数
    if asint:
        decreasing = decreasing.astype(int)

    # 对结果进行偏移
    if offset != 0:
        decreasing = decreasing.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        decreasing.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        decreasing.fillna(method=kwargs["fill_method"], inplace=True)

    # 设定结果的名称和类别
    _percent = f"_{0.01 * percent}" if percent else ''
    _props = f"{'S' if strict else ''}DEC{'p' if percent else ''}"
    decreasing.name = f"{_props}_{length}{_percent}"
    decreasing.category = "trend"

    return decreasing

# 为 decreasing 函数添加文档字符串
decreasing.__doc__ = \
"""Decreasing

Returns True if the series is decreasing over a period, False otherwise.
If the kwarg 'strict' is True, it returns True if it is continuously decreasing
over the period. When using the kwarg 'asint', then it returns 1 for True
or 0 for False.

Calculation:
    if strict:
        decreasing = all(i > j for i, j in zip(close[-length:], close[1:]))
    else:
        decreasing = close.diff(length) < 0

    if asint:
        decreasing = decreasing.astype(int)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    strict (bool): If True, checks if the series is continuously decreasing over the period. Default: False
    percent (float): Percent as an integer. Default: None
    asint (bool): Returns as binary. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```