# `.\pandas-ta\pandas_ta\candles\cdl_pattern.py`

```
# -*- coding: utf-8 -*-
# 导入必要的类型和模块
from typing import Sequence, Union
from pandas import Series, DataFrame

# 从当前目录下的文件中导入指定函数
from . import cdl_doji, cdl_inside
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series
# 从 pandas_ta 模块中导入 Imports 对象
from pandas_ta import Imports

# 定义所有的蜡烛图形式样
ALL_PATTERNS = [
    "2crows", "3blackcrows", "3inside", "3linestrike", "3outside", "3starsinsouth",
    "3whitesoldiers", "abandonedbaby", "advanceblock", "belthold", "breakaway",
    "closingmarubozu", "concealbabyswall", "counterattack", "darkcloudcover", "doji",
    "dojistar", "dragonflydoji", "engulfing", "eveningdojistar", "eveningstar",
    "gapsidesidewhite", "gravestonedoji", "hammer", "hangingman", "harami",
    "haramicross", "highwave", "hikkake", "hikkakemod", "homingpigeon",
    "identical3crows", "inneck", "inside", "invertedhammer", "kicking", "kickingbylength",
    "ladderbottom", "longleggeddoji", "longline", "marubozu", "matchinglow", "mathold",
    "morningdojistar", "morningstar", "onneck", "piercing", "rickshawman",
    "risefall3methods", "separatinglines", "shootingstar", "shortline", "spinningtop",
    "stalledpattern", "sticksandwich", "takuri", "tasukigap", "thrusting", "tristar",
    "unique3river", "upsidegap2crows", "xsidegap3methods"
]

# 定义函数 cdl_pattern，接收开盘价、最高价、最低价、收盘价等参数，返回 DataFrame 类型
def cdl_pattern(open_, high, low, close, name: Union[str, Sequence[str]]="all", scalar=None, offset=None, **kwargs) -> DataFrame:
    """Candle Pattern"""
    # 验证参数
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)
    scalar = float(scalar) if scalar else 100

    # pandas-ta 中已实现的蜡烛图形式样
    pta_patterns = {
        "doji": cdl_doji, "inside": cdl_inside,
    }

    # 如果 name 参数为 "all"，则将其替换为所有蜡烛图形式样
    if name == "all":
        name = ALL_PATTERNS
    # 如果 name 参数为字符串类型，则转换为列表
    if type(name) is str:
        name = [name]

    # 如果导入了 talib 模块
    if Imports["talib"]:
        import talib.abstract as tala

    # 初始化结果字典
    result = {}
    # 对于给定的每个图案名称进行迭代
    for n in name:
        # 检查图案名称是否在 ALL_PATTERNS 列表中
        if n not in ALL_PATTERNS:
            # 如果不在，打印错误消息，并跳过当前迭代
            print(f"[X] There is no candle pattern named {n} available!")
            continue

        # 检查图案是否已在 pta_patterns 字典中定义
        if n in pta_patterns:
            # 如果已定义，调用对应的函数计算图案结果
            pattern_result = pta_patterns[n](open_, high, low, close, offset=offset, scalar=scalar, **kwargs)
            # 将图案结果添加到结果字典中
            result[pattern_result.name] = pattern_result
        else:
            # 如果图案未在 pta_patterns 中定义

            # 检查是否已导入 TA-Lib 模块
            if not Imports["talib"]:
                # 如果未导入，打印错误消息，并跳过当前迭代
                print(f"[X] Please install TA-Lib to use {n}. (pip install TA-Lib)")
                continue

            # 根据图案名称创建对应的 TA-Lib 函数对象
            pattern_func = tala.Function(f"CDL{n.upper()}")
            # 调用 TA-Lib 函数计算图案结果
            pattern_result = Series(pattern_func(open_, high, low, close, **kwargs) / 100 * scalar)
            # 设置图案结果的索引与 close 的索引一致
            pattern_result.index = close.index

            # 处理偏移
            if offset != 0:
                # 将图案结果进行偏移
                pattern_result = pattern_result.shift(offset)

            # 处理填充
            if "fillna" in kwargs:
                # 如果指定了填充值，使用指定值填充缺失值
                pattern_result.fillna(kwargs["fillna"], inplace=True)
            if "fill_method" in kwargs:
                # 如果指定了填充方法，使用指定方法填充缺失值
                pattern_result.fillna(method=kwargs["fill_method"], inplace=True)

            # 将图案结果添加到结果字典中，以"CDL_"加大写的图案名称作为键
            result[f"CDL_{n.upper()}"] = pattern_result

    # 如果结果字典为空，则返回
    if len(result) == 0: return

    # 准备要返回的 DataFrame
    df = DataFrame(result)
    # 设置 DataFrame 的名称属性
    df.name = "CDL_PATTERN"
    # 设置 DataFrame 的 category 属性
    df.category = "candles"
    # 返回 DataFrame
    return df
# 设置 cdl_pattern 的文档字符串，描述蜡烛图模式的使用方法和参数说明
cdl_pattern.__doc__ = \
"""Candle Pattern

A wrapper around all candle patterns.

Examples:

Get all candle patterns (This is the default behaviour)
>>> df = df.ta.cdl_pattern(name="all")
Or
>>> df.ta.cdl("all", append=True) # = df.ta.cdl_pattern("all", append=True)

Get only one pattern
>>> df = df.ta.cdl_pattern(name="doji")
Or
>>> df.ta.cdl("doji", append=True)

Get some patterns
>>> df = df.ta.cdl_pattern(name=["doji", "inside"])
Or
>>> df.ta.cdl(["doji", "inside"], append=True)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    name: (Union[str, Sequence[str]]): name of the patterns
    scalar (float): How much to magnify. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: one column for each pattern.
"""

# 将 cdl_pattern 函数的引用赋值给 cdl 变量，用于简化调用
cdl = cdl_pattern
```