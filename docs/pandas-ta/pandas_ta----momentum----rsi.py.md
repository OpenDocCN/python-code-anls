# `.\pandas-ta\pandas_ta\momentum\rsi.py`

```py
# -*- coding: utf-8 -*-
# 导入所需模块和函数
from pandas import DataFrame, concat
from pandas_ta import Imports
from pandas_ta.overlap import rma
from pandas_ta.utils import get_drift, get_offset, verify_series, signals


def rsi(close, length=None, scalar=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: Relative Strength Index (RSI)"""
    # 验证参数
    # 如果指定了长度且长度大于0，则将长度转换为整数；否则默认为14
    length = int(length) if length and length > 0 else 14
    # 如果指定了标量，则将其转换为浮点数；否则默认为100
    scalar = float(scalar) if scalar else 100
    # 确保收盘价是有效的数据序列，且长度符合要求
    close = verify_series(close, length)
    # 获取漂移参数，默认为1
    drift = get_drift(drift)
    # 获取偏移参数，默认为0
    offset = get_offset(offset)
    # 如果指定了 talib 参数且为 True，则使用 talib 模式；否则默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果启用了 talib 并且处于 talib 模式，则使用 talib 库计算 RSI
        from talib import RSI
        rsi = RSI(close, length)
    else:
        # 否则，按照公式手动计算 RSI
        # 计算价格变化的正值和负值
        negative = close.diff(drift)
        positive = negative.copy()

        # 将正值序列中小于0的值置为0，以保证正值序列只包含正的价格变化
        positive[positive < 0] = 0  
        # 将负值序列中大于0的值置为0，以保证负值序列只包含负的价格变化
        negative[negative > 0] = 0  

        # 计算正值序列和负值序列的移动平均值
        positive_avg = rma(positive, length=length)
        negative_avg = rma(negative, length=length)

        # 计算 RSI
        rsi = scalar * positive_avg / (positive_avg + negative_avg.abs())

    # 偏移结果
    if offset != 0:
        rsi = rsi.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        rsi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rsi.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    rsi.name = f"RSI_{length}"
    rsi.category = "momentum"

    # 如果指定了信号指示器参数为 True，则返回包含信号指示器的数据框；否则返回 RSI 序列
    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        # 如果启用了信号指示器，则生成包含信号指示器的数据框
        signalsdf = concat(
            [
                DataFrame({rsi.name: rsi}),
                signals(
                    indicator=rsi,
                    xa=kwargs.pop("xa", 80),
                    xb=kwargs.pop("xb", 20),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        # 否则，返回 RSI 序列
        return rsi


# 设置 RSI 函数的文档字符串
rsi.__doc__ = \
"""Relative Strength Index (RSI)

The Relative Strength Index is popular momentum oscillator used to measure the
velocity as well as the magnitude of directional price movements.

Sources:
    https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)

Calculation:
    Default Inputs:
        length=14, scalar=100, drift=1
    ABS = Absolute Value
    RMA = Rolling Moving Average

    diff = close.diff(drift)
    positive = diff if diff > 0 else 0
    negative = diff if diff < 0 else 0

    pos_avg = RMA(positive, length)
    neg_avg = ABS(RMA(negative, length))

    RSI = scalar * pos_avg / (pos_avg + neg_avg)

Args:

"""
    # close (pd.Series): 'close' 是一个 Pandas Series 对象，存储了收盘价数据
    # length (int): 计算指标所用的周期长度，默认为 14
    # scalar (float): 放大倍数，默认为 100
    # talib (bool): 如果 TA Lib 被安装并且 talib 参数为 True，则返回 TA Lib 的版本信息，默认为 True
    # drift (int): 差分的周期长度，默认为 1
    # offset (int): 结果向后偏移的周期数，默认为 0
# 参数说明部分，描述函数的参数和返回值
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
# 返回值说明部分，描述函数返回的数据类型
Returns:
    pd.Series: New feature generated.
```