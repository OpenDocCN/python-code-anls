# `.\pandas-ta\pandas_ta\trend\increasing.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 中导入所需的函数和模块
from pandas_ta.utils import get_drift, get_offset, is_percent, verify_series

# 定义名为 increasing 的函数，用于计算序列是否递增
def increasing(close, length=None, strict=None, asint=None, percent=None, drift=None, offset=None, **kwargs):
    """Indicator: Increasing"""
    # 验证参数的有效性
    length = int(length) if length and length > 0 else 1  # 将长度转换为整数，如果未提供或小于等于0，则设为1
    strict = strict if isinstance(strict, bool) else False  # 如果 strict 不是布尔值，则设为 False
    asint = asint if isinstance(asint, bool) else True  # 如果 asint 不是布尔值，则设为 True
    close = verify_series(close, length)  # 验证并处理输入的序列数据
    drift = get_drift(drift)  # 获取漂移值
    offset = get_offset(offset)  # 获取偏移值
    percent = float(percent) if is_percent(percent) else False  # 将百分比转换为浮点数，如果不是百分比，则设为 False

    if close is None: return  # 如果序列为空，则返回空值

    # 计算结果
    close_ = (1 + 0.01 * percent) * close if percent else close  # 如果有百分比参数，则对序列进行调整
    if strict:
        # 返回值是否为 float64？必须转换为布尔值
        increasing = close > close_.shift(drift)  # 检查当前值是否大于移动后的值
        for x in range(3, length + 1):
            increasing = increasing & (close.shift(x - (drift + 1)) > close_.shift(x - drift))  # 检查连续多个值是否递增

        increasing.fillna(0, inplace=True)  # 填充缺失值为0
        increasing = increasing.astype(bool)  # 将结果转换为布尔值
    else:
        increasing = close_.diff(length) > 0  # 检查序列是否在给定周期内递增

    if asint:
        increasing = increasing.astype(int)  # 将结果转换为整数类型

    # 偏移结果
    if offset != 0:
        increasing = increasing.shift(offset)  # 对结果进行偏移

    # 处理填充值
    if "fillna" in kwargs:
        increasing.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充缺失值
    if "fill_method" in kwargs:
        increasing.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定方法填充缺失值

    # 命名并分类结果
    _percent = f"_{0.01 * percent}" if percent else ''  # 根据是否存在百分比参数构建后缀
    _props = f"{'S' if strict else ''}INC{'p' if percent else ''}"  # 根据参数构建特性标识
    increasing.name = f"{_props}_{length}{_percent}"  # 构建结果的名称
    increasing.category = "trend"  # 将结果分类为趋势

    return increasing  # 返回计算结果


increasing.__doc__ = \
"""Increasing

Returns True if the series is increasing over a period, False otherwise.
If the kwarg 'strict' is True, it returns True if it is continuously increasing
over the period. When using the kwarg 'asint', then it returns 1 for True
or 0 for False.

Calculation:
    if strict:
        increasing = all(i < j for i, j in zip(close[-length:], close[1:]))
    else:
        increasing = close.diff(length) > 0

    if asint:
        increasing = increasing.astype(int)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    strict (bool): If True, checks if the series is continuously increasing over the period. Default: False
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