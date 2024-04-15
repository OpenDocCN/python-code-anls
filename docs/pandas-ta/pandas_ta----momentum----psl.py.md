# `.\pandas-ta\pandas_ta\momentum\psl.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 导入 sign 函数并命名为 npSign
from numpy import sign as npSign
# 从 pandas_ta.utils 模块导入 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义函数 psl，计算心理线指标
def psl(close, open_=None, length=None, scalar=None, drift=None, offset=None, **kwargs):
    """Indicator: Psychological Line (PSL)"""
    # 验证参数
    # 将长度转换为整数，如果长度大于 0 则取参数值，否则默认为 12
    length = int(length) if length and length > 0 else 12
    # 将标量转换为浮点数，如果标量大于 0 则取参数值，否则默认为 100
    scalar = float(scalar) if scalar and scalar > 0 else 100
    # 验证 close 数据类型，并将其调整为指定长度
    close = verify_series(close, length)
    # 获取漂移值
    drift = get_drift(drift)
    # 获取偏移值
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    # 如果存在 open_ 参数
    if open_ is not None:
        # 验证 open_ 数据类型
        open_ = verify_series(open_)
        # 计算 close 与 open_ 之间的差异，并取其符号
        diff = npSign(close - open_)
    else:
        # 计算 close 在漂移期内的差异，并取其符号
        diff = npSign(close.diff(drift))

    # 将缺失值填充为 0
    diff.fillna(0, inplace=True)
    # 将小于等于 0 的值设置为 0
    diff[diff <= 0] = 0  # Zero negative values

    # 计算心理线值
    psl = scalar * diff.rolling(length).sum()
    psl /= length

    # 偏移
    if offset != 0:
        psl = psl.shift(offset)

    # 填充缺失值
    if "fillna" in kwargs:
        psl.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        psl.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    _props = f"_{length}"
    psl.name = f"PSL{_props}"
    psl.category = "momentum"

    return psl


# 设置函数 psl 的文档字符串
psl.__doc__ = \
"""Psychological Line (PSL)

The Psychological Line is an oscillator-type indicator that compares the
number of the rising periods to the total number of periods. In other
words, it is the percentage of bars that close above the previous
bar over a given period.

Sources:
    https://www.quantshare.com/item-851-psychological-line

Calculation:
    Default Inputs:
        length=12, scalar=100, drift=1

    IF NOT open:
        DIFF = SIGN(close - close[drift])
    ELSE:
        DIFF = SIGN(close - open)

    DIFF.fillna(0)
    DIFF[DIFF <= 0] = 0

    PSL = scalar * SUM(DIFF, length) / length

Args:
    close (pd.Series): Series of 'close's
    open_ (pd.Series, optional): Series of 'open's
    length (int): It's period. Default: 12
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```