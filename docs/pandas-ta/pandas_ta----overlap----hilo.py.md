# `.\pandas-ta\pandas_ta\overlap\hilo.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 中的 nan 作为 npNaN
from numpy import nan as npNaN
# 从 pandas 中导入 DataFrame 和 Series
from pandas import DataFrame, Series
# 从当前包中导入 ma 模块
from .ma import ma
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def hilo(high, low, close, high_length=None, low_length=None, mamode=None, offset=None, **kwargs):
    """Indicator: Gann HiLo (HiLo)"""
    # 验证参数
    # 如果 high_length 存在且大于 0，则转换为整数；否则设为默认值 13
    high_length = int(high_length) if high_length and high_length > 0 else 13
    # 如果 low_length 存在且大于 0，则转换为整数；否则设为默认值 21
    low_length = int(low_length) if low_length and low_length > 0 else 21
    # 如果 mamode 是字符串，则转换为小写；否则设为默认值 "sma"
    mamode = mamode.lower() if isinstance(mamode, str) else "sma"
    # 计算 high 和 low 的最大长度
    _length = max(high_length, low_length)
    # 验证 high、low 和 close 的数据，并取长度为 _length 的数据
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 high、low 或 close 为空，则返回空值
    if high is None or low is None or close is None: return

    # 计算结果
    m = close.size
    # 初始化 hilo、long 和 short 为全 NaN 的 Series
    hilo = Series(npNaN, index=close.index)
    long = Series(npNaN, index=close.index)
    short = Series(npNaN, index=close.index)

    # 计算 high 和 low 的移动平均值
    high_ma = ma(mamode, high, length=high_length)
    low_ma = ma(mamode, low, length=low_length)

    # 循环计算 hilo、long 和 short
    for i in range(1, m):
        if close.iloc[i] > high_ma.iloc[i - 1]:
            hilo.iloc[i] = long.iloc[i] = low_ma.iloc[i]
        elif close.iloc[i] < low_ma.iloc[i - 1]:
            hilo.iloc[i] = short.iloc[i] = high_ma.iloc[i]
        else:
            hilo.iloc[i] = hilo.iloc[i - 1]
            long.iloc[i] = short.iloc[i] = hilo.iloc[i - 1]

    # 偏移结果
    if offset != 0:
        hilo = hilo.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        hilo.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        hilo.fillna(method=kwargs["fill_method"], inplace=True)
        long.fillna(method=kwargs["fill_method"], inplace=True)
        short.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和类别
    _props = f"_{high_length}_{low_length}"
    # 创建包含 hilo、long 和 short 数据的 DataFrame
    data = {f"HILO{_props}": hilo, f"HILOl{_props}": long, f"HILOs{_props}": short}
    df = DataFrame(data, index=close.index)

    # 设置 DataFrame 的名称和类别
    df.name = f"HILO{_props}"
    df.category = "overlap"

    # 返回 DataFrame
    return df


# 设置 hilo 函数的文档字符串
hilo.__doc__ = \
"""Gann HiLo Activator(HiLo)

The Gann High Low Activator Indicator was created by Robert Krausz in a 1998
issue of Stocks & Commodities Magazine. It is a moving average based trend
indicator consisting of two different simple moving averages.

The indicator tracks both curves (of the highs and the lows). The close of the
bar defines which of the two gets plotted.

Increasing high_length and decreasing low_length better for short trades,
vice versa for long positions.

Sources:
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=447&Name=Gann_HiLo_Activator
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/
"""
    # 通过指定的 URL 访问 Gann High Low 脚本
    https://www.tradingview.com/script/XNQSLIYb-Gann-High-Low/
# 计算函数，根据所选的移动平均模式计算高低移动平均线
Calculation:
    # 默认输入参数：高期限、低期限、移动平均模式（默认为简单移动平均）
    Default Inputs:
        high_length=13, low_length=21, mamode="sma"
    # EMA = 指数移动平均
    EMA = Exponential Moving Average
    # HMA = 哈尔移动平均
    HMA = Hull Moving Average
    # SMA = 简单移动平均 # 默认

    # 根据所选的移动平均模式计算高期限和低期限移动平均值
    if "ema":
        high_ma = EMA(high, high_length)
        low_ma = EMA(low, low_length)
    elif "hma":
        high_ma = HMA(high, high_length)
        low_ma = HMA(low, low_length)
    else: # "sma"
        high_ma = SMA(high, high_length)
        low_ma = SMA(low, low_length)

    # 类似于Supertrend MA选择
    # 创建一个Series对象，用于存储高低移动平均线
    hilo = Series(npNaN, index=close.index)
    # 循环计算
    for i in range(1, m):
        # 如果当前收盘价大于上一个周期的高期限移动平均值，则将当前位置的低期限移动平均值存入hilo
        if close.iloc[i] > high_ma.iloc[i - 1]:
            hilo.iloc[i] = low_ma.iloc[i]
        # 如果当前收盘价小于上一个周期的低期限移动平均值，则将当前位置的高期限移动平均值存入hilo
        elif close.iloc[i] < low_ma.iloc[i - 1]:
            hilo.iloc[i] = high_ma.iloc[i]
        # 否则，维持前一个周期的值
        else:
            hilo.iloc[i] = hilo.iloc[i - 1]

Args:
    # 高价的Series
    high (pd.Series): Series of 'high's
    # 低价的Series
    low (pd.Series): Series of 'low's
    # 收盘价的Series
    close (pd.Series): Series of 'close's
    # 高期限的长度，即移动平均线的周期。默认值为13
    high_length (int): It's period. Default: 13
    # 低期限的长度，即移动平均线的周期。默认值为21
    low_length (int): It's period. Default: 21
    # 移动平均模式，参见```help(ta.ma)```。默认为'sma'
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    # 结果的偏移量，即将结果向前或向后移动的周期数。默认为0
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    # 是否调整结果
    adjust (bool): Default: True
    # 是否使用SMA作为初始值
    presma (bool, optional): If True, uses SMA for initial value.
    # 对DataFrame进行fillna填充
    fillna (value, optional): pd.DataFrame.fillna(value)
    # 填充方法的类型
    fill_method (value, optional): Type of fill method

Returns:
    # 返回一个DataFrame，包含HILO（线）、HILOl（长）、HILOs（短）列。
    pd.DataFrame: HILO (line), HILOl (long), HILOs (short) columns.
```