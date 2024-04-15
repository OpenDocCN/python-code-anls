# `.\pandas-ta\pandas_ta\momentum\fisher.py`

```py
# -*- coding: utf-8 -*-
# 导入所需的库和函数
from numpy import log as nplog
from numpy import nan as npNaN
from pandas import DataFrame, Series
from pandas_ta.overlap import hl2
from pandas_ta.utils import get_offset, high_low_range, verify_series

# 定义 Fisher Transform (FISHT) 指标函数
def fisher(high, low, length=None, signal=None, offset=None, **kwargs):
    """Indicator: Fisher Transform (FISHT)"""
    # 验证参数
    length = int(length) if length and length > 0 else 9
    signal = int(signal) if signal and signal > 0 else 1
    _length = max(length, signal)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)

    if high is None or low is None: return

    # 计算结果
    hl2_ = hl2(high, low)
    highest_hl2 = hl2_.rolling(length).max()
    lowest_hl2 = hl2_.rolling(length).min()

    hlr = high_low_range(highest_hl2, lowest_hl2)
    hlr[hlr < 0.001] = 0.001

    position = ((hl2_ - lowest_hl2) / hlr) - 0.5

    v = 0
    m = high.size
    result = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        v = 0.66 * position.iloc[i] + 0.67 * v
        if v < -0.99: v = -0.999
        if v > 0.99: v = 0.999
        result.append(0.5 * (nplog((1 + v) / (1 - v)) + result[i - 1]))
    fisher = Series(result, index=high.index)
    signalma = fisher.shift(signal)

    # 调整偏移量
    if offset != 0:
        fisher = fisher.shift(offset)
        signalma = signalma.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        fisher.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        fisher.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    _props = f"_{length}_{signal}"
    fisher.name = f"FISHERT{_props}"
    signalma.name = f"FISHERTs{_props}"
    fisher.category = signalma.category = "momentum"

    # 准备返回的 DataFrame
    data = {fisher.name: fisher, signalma.name: signalma}
    df = DataFrame(data)
    df.name = f"FISHERT{_props}"
    df.category = fisher.category

    return df

# 设置 Fisher Transform (FISHT) 指标的计算说明
fisher.__doc__ = \
"""Fisher Transform (FISHT)

Attempts to identify significant price reversals by normalizing prices over a
user-specified number of periods. A reversal signal is suggested when the the
two lines cross.

Sources:
    TradingView (Correlation >99%)

Calculation:
    Default Inputs:
        length=9, signal=1
    HL2 = hl2(high, low)
    HHL2 = HL2.rolling(length).max()
    LHL2 = HL2.rolling(length).min()

    HLR = HHL2 - LHL2
    HLR[HLR < 0.001] = 0.001

    position = ((HL2 - LHL2) / HLR) - 0.5

    v = 0
    m = high.size
    FISHER = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        v = 0.66 * position[i] + 0.67 * v
        if v < -0.99: v = -0.999
        if v >  0.99: v =  0.999
        FISHER.append(0.5 * (nplog((1 + v) / (1 - v)) + FISHER[i - 1]))
"""
    # 使用 FISHER 的 shift() 方法来对信号进行移位处理，并将结果赋值给 SIGNAL
    SIGNAL = FISHER.shift(signal)
# 定义一个函数，用于计算 Fisher 变换后的特征
def fisher(high, low, length=9, signal=1, offset=0, **kwargs):
    # 在参数中传入的高价序列
    high
    # 在参数中传入的低价序列
    low
    # Fisher 变换的周期，默认为9
    length
    # Fisher 信号的周期，默认为1
    signal
    # 结果的偏移周期数，默认为0
    offset

    # 可选参数
    # 填充缺失值的数值或填充方法
    fillna = kwargs.get('fillna')
    fill_method = kwargs.get('fill_method')

    # 返回新生成的特征序列
    return pd.Series
```  
```