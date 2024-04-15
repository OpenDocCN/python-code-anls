# `.\pandas-ta\pandas_ta\volatility\kc.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 .true_range 模块中导入 true_range 函数
from .true_range import true_range
# 从 pandas_ta.overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta.utils 模块中导入 get_offset, high_low_range, verify_series 函数
from pandas_ta.utils import get_offset, high_low_range, verify_series

# 定义函数 kc，用于计算 Keltner 通道（KC）指标
def kc(high, low, close, length=None, scalar=None, mamode=None, offset=None, **kwargs):
    """Indicator: Keltner Channels (KC)"""
    # 验证参数
    # 如果 length 存在且大于 0，则转换为整数，否则设置为默认值 20
    length = int(length) if length and length > 0 else 20
    # 如果 scalar 存在且大于 0，则转换为浮点数，否则设置为默认值 2
    scalar = float(scalar) if scalar and scalar > 0 else 2
    # 如果 mamode 是字符串类型，则保持不变，否则设置为默认值 "ema"
    mamode = mamode if isinstance(mamode, str) else "ema"
    # 验证 high、low、close 是否为有效的 Series，长度为 length
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 high、low、close 存在空值，则返回空值
    if high is None or low is None or close is None: return

    # 计算结果
    # 判断是否使用 True Range（TR），默认为 True
    use_tr = kwargs.pop("tr", True)
    if use_tr:
        range_ = true_range(high, low, close)
    else:
        range_ = high_low_range(high, low)

    # 计算基准线和波动范围
    basis = ma(mamode, close, length=length)
    band = ma(mamode, range_, length=length)

    lower = basis - scalar * band
    upper = basis + scalar * band

    # 处理偏移量
    if offset != 0:
        lower = lower.shift(offset)
        basis = basis.shift(offset)
        upper = upper.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        basis.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        basis.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名并分类化
    _props = f"{mamode.lower()[0] if len(mamode) else ''}_{length}_{scalar}"
    lower.name = f"KCL{_props}"
    basis.name = f"KCB{_props}"
    upper.name = f"KCU{_props}"
    basis.category = upper.category = lower.category = "volatility"

    # 准备返回的 DataFrame
    data = {lower.name: lower, basis.name: basis, upper.name: upper}
    kcdf = DataFrame(data)
    kcdf.name = f"KC{_props}"
    kcdf.category = basis.category

    return kcdf


# 设置 kc 函数的文档字符串
kc.__doc__ = \
"""Keltner Channels (KC)

A popular volatility indicator similar to Bollinger Bands and
Donchian Channels.

Sources:
    https://www.tradingview.com/wiki/Keltner_Channels_(KC)

Calculation:
    Default Inputs:
        length=20, scalar=2, mamode=None, tr=True
    TR = True Range
    SMA = Simple Moving Average
    EMA = Exponential Moving Average

    if tr:
        RANGE = TR(high, low, close)
    else:
        RANGE = high - low

    if mamode == "ema":
        BASIS = sma(close, length)
        BAND = sma(RANGE, length)
    elif mamode == "sma":
        BASIS = sma(close, length)
        BAND = sma(RANGE, length)

    LOWER = BASIS - scalar * BAND
    UPPER = BASIS + scalar * BAND

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's

"""
    length (int): The short period.  Default: 20
    scalar (float): A positive float to scale the bands. Default: 2
    mamode (str): See ```help(ta.ma)```py. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0
# 函数参数：
#   tr (bool): 如果为 True，则使用 True Range 进行计算；如果为 False，则使用高 - 低作为范围计算。默认值为 True
#   fillna (value, optional): pd.DataFrame.fillna(value) 的可选参数，用于指定填充缺失值的值
#   fill_method (value, optional): 填充方法的类型

# 返回值：
#   返回一个 pandas DataFrame，包含 lower、basis、upper 列。
```