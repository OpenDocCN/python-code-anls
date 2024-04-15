# `.\pandas-ta\pandas_ta\statistics\tos_stdevall.py`

```py
# -*- coding: utf-8 -*-

# 从 numpy 库中导入 array 别名为 npArray
from numpy import array as npArray
# 从 numpy 库中导入 arange 别名为 npArange
from numpy import arange as npArange
# 从 numpy 库中导入 polyfit 别名为 npPolyfit
from numpy import polyfit as npPolyfit
# 从 numpy 库中导入 std 别名为 npStd
from numpy import std as npStd
# 从 pandas 库中导入 DataFrame、DatetimeIndex、Series
from pandas import DataFrame, DatetimeIndex, Series
# 从 .stdev 模块中导入 stdev 别名为 stdev
from .stdev import stdev as stdev
# 从 pandas_ta.utils 模块中导入 get_offset、verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 tos_stdevall，计算 Think or Swim 标准偏差
def tos_stdevall(close, length=None, stds=None, ddof=None, offset=None, **kwargs):
    """Indicator: TD Ameritrade's Think or Swim Standard Deviation All"""
    # 验证参数
    # 如果 stds 是非空列表，则使用 stds，否则默认为 [1, 2, 3]
    stds = stds if isinstance(stds, list) and len(stds) > 0 else [1, 2, 3]
    # 如果 stds 中有小于等于 0 的数，则返回空
    if min(stds) <= 0: return
    # 如果 stds 中存在逆序排列，则将其反转为升序排列
    if not all(i < j for i, j in zip(stds, stds[1:])):
        stds = stds[::-1]
    # 将 ddof 转换为整数，确保在合理范围内，默认为 1
    ddof = int(ddof) if ddof and ddof >= 0 and ddof < length else 1
    # 获取偏移量
    offset = get_offset(offset)

    # 属性名称
    _props = f"TOS_STDEVALL"
    # 如果 length 为 None，则使用全部数据；否则，使用指定长度的数据
    if length is None:
        length = close.size
    else:
        # 将 length 转换为整数，确保大于 2，默认为 30
        length = int(length) if isinstance(length, int) and length > 2 else 30
        # 仅保留最近 length 个数据
        close = close.iloc[-length:]
        _props = f"{_props}_{length}"

    # 确保 close 是一个 Series，并且长度为 length
    close = verify_series(close, length)

    # 如果 close 为空，则返回空
    if close is None: return

    # 计算结果
    X = src_index = close.index
    # 如果 close 的索引是 DatetimeIndex 类型，则创建等差数组 X，并将 close 转换为数组
    if isinstance(close.index, DatetimeIndex):
        X = npArange(length)
        close = npArray(close)

    # 使用线性回归拟合得到斜率 m 和截距 b
    m, b = npPolyfit(X, close, 1)
    # 计算线性回归线 lr，索引与 close 保持一致
    lr = Series(m * X + b, index=src_index)
    # 计算标准差 stdev
    stdev = npStd(close, ddof=ddof)

    # 组装结果 DataFrame
    df = DataFrame({f"{_props}_LR": lr}, index=src_index)
    # 对于每个标准偏差值，计算上下界，并设置名称和分类
    for i in stds:
        df[f"{_props}_L_{i}"] = lr - i * stdev
        df[f"{_props}_U_{i}"] = lr + i * stdev
        df[f"{_props}_L_{i}"].name = df[f"{_props}_U_{i}"].name = f"{_props}"
        df[f"{_props}_L_{i}"].category = df[f"{_props}_U_{i}"].category = "statistics"

    # 对结果进行偏移
    if offset != 0:
        df = df.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # 准备返回的 DataFrame
    df.name = f"{_props}"
    df.category = "statistics"

    return df

# 设定函数的文档字符串
tos_stdevall.__doc__ = \
"""TD Ameritrade's Think or Swim Standard Deviation All (TOS_STDEV)

A port of TD Ameritrade's Think or Swim Standard Deviation All indicator which
returns the standard deviation of data for the entire plot or for the interval
of the last bars defined by the length parameter.

Sources:
    https://tlc.thinkorswim.com/center/reference/thinkScript/Functions/Statistical/StDevAll

Calculation:
    Default Inputs:
        length=None (All), stds=[1, 2, 3], ddof=1
    LR = Linear Regression
    STDEV = Standard Deviation

    LR = LR(close, length)
    STDEV = STDEV(close, length, ddof)
    for level in stds:
        LOWER = LR - level * STDEV
        UPPER = LR + level * STDEV

Args:
    close (pd.Series): Series of 'close's
    length (int): Bars from current bar. Default: None

"""
    stds (list): 存储标准偏差的列表，按照从中心线性回归线开始增加的顺序排列。默认值为 [1,2,3]
    ddof (int): Delta 自由度。在计算中使用的除数是 N - ddof，其中 N 表示元素的数量。默认值为 1
    offset (int): 结果的偏移周期数。默认值为 0
# 函数参数说明，接受关键字参数
Kwargs:
    # fillna 参数，用于填充缺失值的数值
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method 参数，指定填充方法的类型
    fill_method (value, optional): Type of fill method

# 返回值说明，返回一个 pandas DataFrame 对象
Returns:
    # 返回一个 pandas DataFrame 对象，包含中心 LR 和基于标准差倍数的上下 LR 线对
    pd.DataFrame: Central LR, Pairs of Lower and Upper LR Lines based on
        mulitples of the standard deviation. Default: returns 7 columns.
```