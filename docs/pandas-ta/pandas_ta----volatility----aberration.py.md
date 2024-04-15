# `.\pandas-ta\pandas_ta\volatility\aberration.py`

```
# -*- coding: utf-8 -*-
# from numpy import sqrt as npsqrt  # 导入 numpy 中的 sqrt 函数并重命名为 npsqrt（已注释掉）
from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
from .atr import atr  # 从当前包中的 atr 模块中导入 atr 函数
from pandas_ta.overlap import hlc3, sma  # 从 pandas_ta.overlap 模块中导入 hlc3 和 sma 函数
from pandas_ta.utils import get_offset, verify_series  # 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数


def aberration(high, low, close, length=None, atr_length=None, offset=None, **kwargs):
    """Indicator: Aberration (ABER)"""
    # Validate arguments
    # 确认参数合法性，若参数未指定或非正整数，则使用默认值
    length = int(length) if length and length > 0 else 5
    atr_length = int(atr_length) if atr_length and atr_length > 0 else 15
    _length = max(atr_length, length)  # 选择最大长度作为计算时使用的长度
    high = verify_series(high, _length)  # 确认 high Series 的合法性和长度
    low = verify_series(low, _length)  # 确认 low Series 的合法性和长度
    close = verify_series(close, _length)  # 确认 close Series 的合法性和长度
    offset = get_offset(offset)  # 获取偏移量

    if high is None or low is None or close is None: return  # 如果输入数据有缺失，则返回空值

    # Calculate Result
    # 计算结果
    atr_ = atr(high=high, low=low, close=close, length=atr_length)  # 计算 ATR 指标
    jg = hlc3(high=high, low=low, close=close)  # 计算 JG（typical price，即三价均价）

    zg = sma(jg, length)  # 计算 ZG（SMA of JG）
    sg = zg + atr_  # 计算 SG（ZG + ATR）
    xg = zg - atr_  # 计算 XG（ZG - ATR）

    # Offset
    # 偏移结果
    if offset != 0:
        zg = zg.shift(offset)  # 对 ZG 进行偏移
        sg = sg.shift(offset)  # 对 SG 进行偏移
        xg = xg.shift(offset)  # 对 XG 进行偏移
        atr_ = atr_.shift(offset)  # 对 ATR 进行偏移

    # Handle fills
    # 处理填充缺失值
    if "fillna" in kwargs:
        zg.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充 ZG 中的缺失值
        sg.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充 SG 中的缺失值
        xg.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充 XG 中的缺失值
        atr_.fillna(kwargs["fillna"], inplace=True)  # 使用指定值填充 ATR 中的缺失值
    if "fill_method" in kwargs:
        zg.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充 ZG 中的缺失值
        sg.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充 SG 中的缺失值
        xg.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充 XG 中的缺失值
        atr_.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充 ATR 中的缺失值

    # Name and Categorize it
    # 命名和分类
    _props = f"_{length}_{atr_length}"  # 用于生成属性名称的后缀
    zg.name = f"ABER_ZG{_props}"  # 设置 ZG Series 的名称
    sg.name = f"ABER_SG{_props}"  # 设置 SG Series 的名称
    xg.name = f"ABER_XG{_props}"  # 设置 XG Series 的名称
    atr_.name = f"ABER_ATR{_props}"  # 设置 ATR Series 的名称
    zg.category = sg.category = "volatility"  # 设置 ZG 和 SG Series 的分类为波动性
    xg.category = atr_.category = zg.category  # 设置 XG 和 ATR Series 的分类与 ZG 相同

    # Prepare DataFrame to return
    # 准备要返回的 DataFrame
    data = {zg.name: zg, sg.name: sg, xg.name: xg, atr_.name: atr_}  # 构建数据字典
    aberdf = DataFrame(data)  # 使用数据字典创建 DataFrame
    aberdf.name = f"ABER{_props}"  # 设置 DataFrame 的名称
    aberdf.category = zg.category  # 设置 DataFrame 的分类与 ZG 相同

    return aberdf  # 返回计算结果的 DataFrame


aberration.__doc__ = \
"""Aberration

A volatility indicator similar to Keltner Channels.

Sources:
    Few internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=5, atr_length=15
    ATR = Average True Range
    SMA = Simple Moving Average

    ATR = ATR(length=atr_length)
    JG = TP = HLC3(high, low, close)
    ZG = SMA(JG, length)
    SG = ZG + ATR
    XG = ZG - ATR

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period. Default: 5
    atr_length (int): The short period. Default: 15
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
"""  # 设置 aberration 函数的文档字符串
    fill_method (value, optional): 填充方法的类型
# 返回一个 pandas DataFrame，包含 zg、sg、xg、atr 列
```