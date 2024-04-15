# `.\pandas-ta\pandas_ta\volatility\accbands.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中的 overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta 库中的 utils 模块中导入 get_drift, get_offset, non_zero_range, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series


# 定义函数 accbands，用于计算加速带指标
def accbands(high, low, close, length=None, c=None, drift=None, mamode=None, offset=None, **kwargs):
    """Indicator: Acceleration Bands (ACCBANDS)"""
    # 验证参数
    # 若 length 存在且大于 0，则将其转换为整数类型，否则设为 20
    length = int(length) if length and length > 0 else 20
    # 若 c 存在且大于 0，则将其转换为浮点数类型，否则设为 4
    c = float(c) if c and c > 0 else 4
    # 若 mamode 不为字符串类型，则设为 "sma"
    mamode = mamode if isinstance(mamode, str) else "sma"
    # 验证 high、low、close 系列，使其长度为 length
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    # 获取 drift 和 offset
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 若 high、low、close 存在空值，则返回空值
    if high is None or low is None or close is None: return

    # 计算结果
    # 计算 high 和 low 的非零范围
    high_low_range = non_zero_range(high, low)
    # 计算 high_low_range 与 (high + low) 的比值
    hl_ratio = high_low_range / (high + low)
    # 将 hl_ratio 乘以 c
    hl_ratio *= c
    # 计算下轨线 _lower
    _lower = low * (1 - hl_ratio)
    # 计算上轨线 _upper
    _upper = high * (1 + hl_ratio)

    # 计算移动平均值
    lower = ma(mamode, _lower, length=length)
    mid = ma(mamode, close, length=length)
    upper = ma(mamode, _upper, length=length)

    # 对结果进行位移
    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        mid.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        mid.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    lower.name = f"ACCBL_{length}"
    mid.name = f"ACCBM_{length}"
    upper.name = f"ACCBU_{length}"
    mid.category = upper.category = lower.category = "volatility"

    # 准备返回的 DataFrame
    data = {lower.name: lower, mid.name: mid, upper.name: upper}
    accbandsdf = DataFrame(data)
    accbandsdf.name = f"ACCBANDS_{length}"
    accbandsdf.category = mid.category

    return accbandsdf


# 设置函数文档字符串
accbands.__doc__ = \
"""Acceleration Bands (ACCBANDS)

Acceleration Bands created by Price Headley plots upper and lower envelope
bands around a simple moving average.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/

Calculation:
    Default Inputs:
        length=10, c=4
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    HL_RATIO = c * (high - low) / (high + low)
    LOW = low * (1 - HL_RATIO)
    HIGH = high * (1 + HL_RATIO)

    if 'ema':
        LOWER = EMA(LOW, length)
        MID = EMA(close, length)
        UPPER = EMA(HIGH, length)
    else:
        LOWER = SMA(LOW, length)
        MID = SMA(close, length)
        UPPER = SMA(HIGH, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's

"""
    # 表示参数 `length` 是一个整数，代表周期。默认值为 10
    length (int): It's period. Default: 10
    # 表示参数 `c` 是一个整数，代表乘数。默认值为 4
    c (int): Multiplier. Default: 4
    # 表示参数 `mamode` 是一个字符串，参见 `ta.ma` 的帮助文档。默认值为 'sma'
    mamode (str): See ```help(ta.ma)```py. Default: 'sma'
    # 表示参数 `drift` 是一个整数，代表差异周期。默认值为 1
    drift (int): The difference period. Default: 1
    # 表示参数 `offset` 是一个整数，代表结果的偏移周期数。默认值为 0
    offset (int): How many periods to offset the result. Default: 0
# 函数参数，用于指定填充缺失值的值，可选参数
fillna (value, optional): pd.DataFrame.fillna(value)
# 函数参数，指定填充方法的类型，可选参数
fill_method (value, optional): Type of fill method

# 返回值，返回一个 DataFrame，包含 lower、mid、upper 列
pd.DataFrame: lower, mid, upper columns.
```