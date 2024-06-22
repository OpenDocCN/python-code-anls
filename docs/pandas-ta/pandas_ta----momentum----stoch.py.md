# `.\pandas-ta\pandas_ta\momentum\stoch.py`

```py
# -*- coding: utf-8 -*-
# 导入DataFrame类
from pandas import DataFrame
# 从pandas_ta.overlap模块中导入ma函数
from pandas_ta.overlap import ma
# 从pandas_ta.utils模块中导入get_offset、non_zero_range和verify_series函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series


# 定义Stochastic Oscillator (STOCH)函数
def stoch(high, low, close, k=None, d=None, smooth_k=None, mamode=None, offset=None, **kwargs):
    """Indicator: Stochastic Oscillator (STOCH)"""
    # 校验参数
    # 如果k为正数则使用k，否则默认为14
    k = k if k and k > 0 else 14
    # 如果d为正数则使用d，否则默认为3
    d = d if d and d > 0 else 3
    # 如果smooth_k为正数则使用smooth_k，否则默认为3
    smooth_k = smooth_k if smooth_k and smooth_k > 0 else 3
    # 计算_max(k, d, smooth_k)
    _length = max(k, d, smooth_k)
    # 校验high、low和close的长度是否为_length
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    # 获取offset值
    offset = get_offset(offset)
    # 如果mamode不是字符串则设为"sma"
    mamode = mamode if isinstance(mamode, str) else "sma"

    # 如果high、low或close有任何一个为None，则返回空值
    if high is None or low is None or close is None: return

    # 计算结果
    # 计算过去k个周期的最低值
    lowest_low = low.rolling(k).min()
    # 计算过去k个周期的最高值
    highest_high = high.rolling(k).max()

    # 计算stoch值
    stoch = 100 * (close - lowest_low)
    stoch /= non_zero_range(highest_high, lowest_low)

    # 计算stoch_k和stoch_d
    stoch_k = ma(mamode, stoch.loc[stoch.first_valid_index():,], length=smooth_k)
    stoch_d = ma(mamode, stoch_k.loc[stoch_k.first_valid_index():,], length=d)

    # 偏移处理
    if offset != 0:
        stoch_k = stoch_k.shift(offset)
        stoch_d = stoch_d.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        stoch_k.fillna(kwargs["fillna"], inplace=True)
        stoch_d.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stoch_k.fillna(method=kwargs["fill_method"], inplace=True)
        stoch_d.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    _name = "STOCH"
    _props = f"_{k}_{d}_{smooth_k}"
    stoch_k.name = f"{_name}k{_props}"
    stoch_d.name = f"{_name}d{_props}"
    stoch_k.category = stoch_d.category = "momentum"

    # 准备要返回的DataFrame
    data = {stoch_k.name: stoch_k, stoch_d.name: stoch_d}
    df = DataFrame(data)
    df.name = f"{_name}{_props}"
    df.category = stoch_k.category
    return df


# 设置stoch函数的文档字符串
stoch.__doc__ = \
"""Stochastic (STOCH)

The Stochastic Oscillator (STOCH) was developed by George Lane in the 1950's.
He believed this indicator was a good way to measure momentum because changes in
momentum precede changes in price.

It is a range-bound oscillator with two lines moving between 0 and 100.
The first line (%K) displays the current close in relation to the period's
high/low range. The second line (%D) is a Simple Moving Average of the %K line.
The most common choices are a 14 period %K and a 3 period SMA for %D.

Sources:
    https://www.tradingview.com/wiki/Stochastic_(STOCH)
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=332&Name=KD_-_Slow

Calculation:
    Default Inputs:
        k=14, d=3, smooth_k=3
    SMA = Simple Moving Average
    LL  = low for last k periods
    HH  = high for last k periods

    STOCH = 100 * (close - LL) / (HH - LL)
    STOCHk = SMA(STOCH, smooth_k)
    STOCHd = SMA(FASTK, d)

Args:
    high (pd.Series): Series of 'high's

"""
    # 表示传入函数的参数，分别为低价序列
    low (pd.Series): Series of 'low's
    # 表示传入函数的参数，分别为收盘价序列
    close (pd.Series): Series of 'close's
    # 表示传入函数的参数，表示快速 %K 的周期，默认为 14
    k (int): The Fast %K period. Default: 14
    # 表示传入函数的参数，表示慢速 %K 的周期，默认为 3
    d (int): The Slow %K period. Default: 3
    # 表示传入函数的参数，表示慢速 %D 的周期，默认为 3
    smooth_k (int): The Slow %D period. Default: 3
    # 表示传入函数的参数，参见 ta.ma 的帮助文档。默认为 'sma'
    mamode (str): See ```help(ta.ma)```py. Default: 'sma'
    # 表示传入函数的参数，表示结果的偏移周期数，默认为 0
    offset (int): How many periods to offset the result. Default: 0
# 参数说明：
# - fillna (value, optional): 使用 value 对 pd.DataFrame 进行填充
# - fill_method (value, optional): 填充方法的类型

# 返回值：
# - 返回一个 pd.DataFrame，包含 %K 和 %D 列
```