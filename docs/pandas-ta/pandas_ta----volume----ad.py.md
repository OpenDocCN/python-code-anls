# `.\pandas-ta\pandas_ta\volume\ad.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta 库导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 模块导入 get_offset、non_zero_range、verify_series 函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series

# 定义累积/分布指标函数
def ad(high, low, close, volume, open_=None, talib=None, offset=None, **kwargs):
    """Indicator: Accumulation/Distribution (AD)"""
    # 验证参数
    # 将 high、low、close、volume 分别验证为 pandas.Series 类型
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    # 将 offset 转换为偏移量
    offset = get_offset(offset)
    # 判断是否使用 talib，若 talib 参数为布尔类型则以其值为准，否则默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果导入了 talib 并且 mode_tal 为真，则使用 talib 计算 AD 指标
        from talib import AD
        # 调用 talib 库中的 AD 函数
        ad = AD(high, low, close, volume)
    else:
        # 如果没有使用 talib 或者 mode_tal 为假，则根据情况计算 AD 指标
        if open_ is not None:
            # 如果提供了 open_ 参数，则使用 close 和 open_ 计算 AD
            open_ = verify_series(open_)
            # 计算 AD 指标，使用 close 和 open_
            ad = non_zero_range(close, open_)  # AD with Open
        else:
            # 如果未提供 open_ 参数，则使用 high、low 和 close 计算 AD
            ad = 2 * close - (high + low)  # AD with High, Low, Close

        # 计算 high-low 范围
        high_low_range = non_zero_range(high, low)
        # 根据 high-low 范围和交易量计算 AD
        ad *= volume / high_low_range
        # 对 AD 进行累积求和
        ad = ad.cumsum()

    # 偏移结果
    if offset != 0:
        # 如果偏移量不为零，则对 AD 进行偏移
        ad = ad.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        # 如果提供了 fillna 参数，则使用提供的值填充缺失值
        ad.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        # 如果提供了 fill_method 参数，则使用提供的填充方法填充缺失值
        ad.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    # 根据是否提供了 open_ 参数命名 AD 指标
    ad.name = "AD" if open_ is None else "ADo"
    # 将 AD 指标分类为“volume”
    ad.category = "volume"

    # 返回 AD 指标
    return ad


# 设置 AD 函数的文档字符串
ad.__doc__ = \
"""Accumulation/Distribution (AD)

Accumulation/Distribution indicator utilizes the relative position
of the close to it's High-Low range with volume.  Then it is cumulated.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/accumulationdistribution-ad/

Calculation:
    CUM = Cumulative Sum
    if 'open':
        AD = close - open
    else:
        AD = 2 * close - high - low

    hl_range = high - low
    AD = AD * volume / hl_range
    AD = CUM(AD)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    open (pd.Series): Series of 'open's
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