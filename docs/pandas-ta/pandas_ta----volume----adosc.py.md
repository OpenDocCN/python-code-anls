# `.\pandas-ta\pandas_ta\volume\adosc.py`

```
# -*- coding: utf-8 -*-
# 导入 ad 模块
from .ad import ad
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.overlap 模块中导入 ema 函数
from pandas_ta.overlap import ema
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 adosc 函数，用于计算累积/分布振荡器指标
def adosc(high, low, close, volume, open_=None, fast=None, slow=None, talib=None, offset=None, **kwargs):
    """Indicator: Accumulation/Distribution Oscillator"""
    # 验证参数有效性
    # 如果 fast 参数存在且大于 0，则将其转换为整数类型，否则设为默认值 3
    fast = int(fast) if fast and fast > 0 else 3
    # 如果 slow 参数存在且大于 0，则将其转换为整数类型，否则设为默认值 10
    slow = int(slow) if slow and slow > 0 else 10
    # 计算 _length，即 fast 和 slow 中的较大值
    _length = max(fast, slow)
    # 验证 high、low、close、volume 系列数据的长度，使其与 _length 相同
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    # 获取偏移量，根据 offset 参数
    offset = get_offset(offset)
    # 如果 kwargs 中存在 "length" 键，则将其移除
    if "length" in kwargs: kwargs.pop("length")
    # 如果 talib 参数为布尔类型且为真，则将 mode_tal 设为 True，否则设为 False
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 high、low、close、volume 中存在空值，则返回空值
    if high is None or low is None or close is None or volume is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果导入了 TA Lib 并且 mode_tal 为 True，则使用 TA Lib 计算 adosc
        from talib import ADOSC
        adosc = ADOSC(high, low, close, volume, fast, slow)
    else:
        # 否则，使用自定义的 ad 函数计算 ad_，然后分别计算其快速和慢速移动平均线
        ad_ = ad(high=high, low=low, close=close, volume=volume, open_=open_)
        fast_ad = ema(close=ad_, length=fast, **kwargs)
        slow_ad = ema(close=ad_, length=slow, **kwargs)
        adosc = fast_ad - slow_ad

    # 根据偏移量对结果进行偏移
    if offset != 0:
        adosc = adosc.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        adosc.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        adosc.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    adosc.name = f"ADOSC_{fast}_{slow}"
    adosc.category = "volume"

    # 返回计算结果
    return adosc


# 设置 adosc 函数的文档字符串
adosc.__doc__ = \
"""Accumulation/Distribution Oscillator or Chaikin Oscillator

Accumulation/Distribution Oscillator indicator utilizes
Accumulation/Distribution and treats it similarily to MACD
or APO.

Sources:
    https://www.investopedia.com/articles/active-trading/031914/understanding-chaikin-oscillator.asp

Calculation:
    Default Inputs:
        fast=12, slow=26
    AD = Accum/Dist
    ad = AD(high, low, close, open)
    fast_ad = EMA(ad, fast)
    slow_ad = EMA(ad, slow)
    ADOSC = fast_ad - slow_ad

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    open (pd.Series): Series of 'open's
    volume (pd.Series): Series of 'volume's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```