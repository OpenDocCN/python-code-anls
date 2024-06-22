# `.\pandas-ta\pandas_ta\momentum\cmo.py`

```py
# -*- coding: utf-8 -*-

# 导入必要的模块和函数
from pandas_ta import Imports
from pandas_ta.overlap import rma
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义 Chande Momentum Oscillator (CMO) 指标函数
def cmo(close, length=None, scalar=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: Chande Momentum Oscillator (CMO)"""
    # 验证参数
    length = int(length) if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为空，返回空值
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        from talib import CMO
        # 使用 TA-Lib 计算 CMO
        cmo = CMO(close, length)
    else:
        # 计算动量
        mom = close.diff(drift)
        positive = mom.copy().clip(lower=0)
        negative = mom.copy().clip(upper=0).abs()

        if mode_tal:
            # 使用 RMA 函数计算动量的指数移动平均值
            pos_ = rma(positive, length)
            neg_ = rma(negative, length)
        else:
            # 使用滚动窗口计算动量的总和
            pos_ = positive.rolling(length).sum()
            neg_ = negative.rolling(length).sum()

        # 计算 CMO 指标值
        cmo = scalar * (pos_ - neg_) / (pos_ + neg_)

    # 对结果进行偏移处理
    if offset != 0:
        cmo = cmo.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        cmo.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cmo.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    cmo.name = f"CMO_{length}"
    cmo.category = "momentum"

    return cmo

# 设置 CMO 函数的文档字符串
cmo.__doc__ = \
"""Chande Momentum Oscillator (CMO)

Attempts to capture the momentum of an asset with overbought at 50 and
oversold at -50.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/chande-momentum-oscillator-cmo/
    https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/

Calculation:
    Default Inputs:
        drift=1, scalar=100

    # Same Calculation as RSI except for this step
    CMO = scalar * (PSUM - NSUM) / (PSUM + NSUM)

Args:
    close (pd.Series): Series of 'close's
    scalar (float): How much to magnify. Default: 100
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. If TA Lib is not installed but talib is True, it runs the Python
        version TA Lib. Default: True
    drift (int): The short period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    talib (bool): If True, uses TA-Libs implementation. Otherwise uses EMA version. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```