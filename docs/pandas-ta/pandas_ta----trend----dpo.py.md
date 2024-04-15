# `.\pandas-ta\pandas_ta\trend\dpo.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta.overlap 模块导入 sma 函数
from pandas_ta.overlap import sma
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def dpo(close, length=None, centered=True, offset=None, **kwargs):
    """Indicator: Detrend Price Oscillator (DPO)"""
    # 验证参数
    # 将长度参数转换为整数，如果长度参数存在且大于0，则使用，否则默认为20
    length = int(length) if length and length > 0 else 20
    # 验证 close 是否为有效的时间序列，长度为指定的长度
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 如果 kwargs 中的 "lookahead" 为 False，则将 centered 设置为 False
    if not kwargs.get("lookahead", True):
        centered = False

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    # 计算 t，即 int(0.5 * length) + 1
    t = int(0.5 * length) + 1
    # 计算 close 的简单移动平均
    ma = sma(close, length)

    # 计算 DPO，close 减去 ma 后向前位移 t 个周期
    dpo = close - ma.shift(t)
    # 如果 centered 为 True，则再将 DPO 向后位移 t 个周期
    if centered:
        dpo = (close.shift(t) - ma).shift(-t)

    # 偏移
    if offset != 0:
        dpo = dpo.shift(offset)

    # 处理填充
    # 如果 kwargs 中有 "fillna"，则使用该值填充 NaN
    if "fillna" in kwargs:
        dpo.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中有 "fill_method"，则使用指定的填充方法
    if "fill_method" in kwargs:
        dpo.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    dpo.name = f"DPO_{length}"
    dpo.category = "trend"

    return dpo


# 更新文档字符串
dpo.__doc__ = \
"""Detrend Price Oscillator (DPO)

Is an indicator designed to remove trend from price and make it easier to
identify cycles.

Sources:
    https://www.tradingview.com/scripts/detrendedpriceoscillator/
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/dpo
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

Calculation:
    Default Inputs:
        length=20, centered=True
    SMA = Simple Moving Average
    t = int(0.5 * length) + 1

    DPO = close.shift(t) - SMA(close, length)
    if centered:
        DPO = DPO.shift(-t)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    centered (bool): Shift the dpo back by int(0.5 * length) + 1. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```