# `.\pandas-ta\pandas_ta\volatility\donchian.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数：唐奇安通道（Donchian Channels，DC）
def donchian(high, low, lower_length=None, upper_length=None, offset=None, **kwargs):
    """Indicator: Donchian Channels (DC)"""
    # 验证参数
    # 如果 lower_length 存在且大于 0，则将其转换为整数，否则设为默认值 20
    lower_length = int(lower_length) if lower_length and lower_length > 0 else 20
    # 如果 upper_length 存在且大于 0，则将其转换为整数，否则设为默认值 20
    upper_length = int(upper_length) if upper_length and upper_length > 0 else 20
    # 如果 kwargs 中存在 "lower_min_periods"，则将其转换为整数，否则设为 lower_length 的值
    lower_min_periods = int(kwargs["lower_min_periods"]) if "lower_min_periods" in kwargs and kwargs["lower_min_periods"] is not None else lower_length
    # 如果 kwargs 中存在 "upper_min_periods"，则将其转换为整数，否则设为 upper_length 的值
    upper_min_periods = int(kwargs["upper_min_periods"]) if "upper_min_periods" in kwargs and kwargs["upper_min_periods"] is not None else upper_length
    # 计算有效期最大值
    _length = max(lower_length, lower_min_periods, upper_length, upper_min_periods)
    # 验证 high 和 low Series
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 high 或 low 为 None，则返回空
    if high is None or low is None: return

    # 计算结果
    # 计算下界，使用滚动窗口计算最小值
    lower = low.rolling(lower_length, min_periods=lower_min_periods).min()
    # 计算上界，使用滚动窗口计算最大值
    upper = high.rolling(upper_length, min_periods=upper_min_periods).max()
    # 计算中位数
    mid = 0.5 * (lower + upper)

    # 填充缺失值
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        mid.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        mid.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)

    # 偏移结果
    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)

    # 设置名称和类别
    lower.name = f"DCL_{lower_length}_{upper_length}"
    mid.name = f"DCM_{lower_length}_{upper_length}"
    upper.name = f"DCU_{lower_length}_{upper_length}"
    mid.category = upper.category = lower.category = "volatility"

    # 准备返回的 DataFrame
    data = {lower.name: lower, mid.name: mid, upper.name: upper}
    dcdf = DataFrame(data)
    dcdf.name = f"DC_{lower_length}_{upper_length}"
    dcdf.category = mid.category

    return dcdf

# 将函数的文档字符串设置为说明唐奇安通道的计算方法和参数含义
donchian.__doc__ = \
"""Donchian Channels (DC)

Donchian Channels are used to measure volatility, similar to
Bollinger Bands and Keltner Channels.

Sources:
    https://www.tradingview.com/wiki/Donchian_Channels_(DC)

Calculation:
    Default Inputs:
        lower_length=upper_length=20
    LOWER = low.rolling(lower_length).min()
    UPPER = high.rolling(upper_length).max()
    MID = 0.5 * (LOWER + UPPER)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    lower_length (int): The short period. Default: 20
    upper_length (int): The short period. Default: 20
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)

"""
    fill_method (value, optional): Type of fill method

# 填充方法（可选参数）：填充方法的类型
# 返回一个 pandas DataFrame 对象，包含 lower、mid 和 upper 列。
```