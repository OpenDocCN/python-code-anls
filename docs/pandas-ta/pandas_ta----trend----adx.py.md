# `.\pandas-ta\pandas_ta\trend\adx.py`

```py
# -*- coding: utf-8 -*-
# 导入所需的库和模块
from pandas import DataFrame
from pandas_ta.overlap import ma
from pandas_ta.volatility import atr
from pandas_ta.utils import get_drift, get_offset, verify_series, zero

# 定义 ADX 指标函数
def adx(high, low, close, length=None, lensig=None, scalar=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: ADX"""
    # 验证参数
    length = length if length and length > 0 else 14
    lensig = lensig if lensig and lensig > 0 else length
    mamode = mamode if isinstance(mamode, str) else "rma"
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None: return

    # 计算 ATR 指标
    atr_ = atr(high=high, low=low, close=close, length=length)

    # 计算上升和下降动向线
    up = high - high.shift(drift)  # high.diff(drift)
    dn = low.shift(drift) - low    # low.diff(-drift).shift(drift)

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn

    pos = pos.apply(zero)
    neg = neg.apply(zero)

    k = scalar / atr_
    dmp = k * ma(mamode, pos, length=length)
    dmn = k * ma(mamode, neg, length=length)

    dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
    adx = ma(mamode, dx, length=lensig)

    # 偏移
    if offset != 0:
        dmp = dmp.shift(offset)
        dmn = dmn.shift(offset)
        adx = adx.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        adx.fillna(kwargs["fillna"], inplace=True)
        dmp.fillna(kwargs["fillna"], inplace=True)
        dmn.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        adx.fillna(method=kwargs["fill_method"], inplace=True)
        dmp.fillna(method=kwargs["fill_method"], inplace=True)
        dmn.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    adx.name = f"ADX_{lensig}"
    dmp.name = f"DMP_{length}"
    dmn.name = f"DMN_{length}"

    adx.category = dmp.category = dmn.category = "trend"

    # 准备返回的 DataFrame
    data = {adx.name: adx, dmp.name: dmp, dmn.name: dmn}
    adxdf = DataFrame(data)
    adxdf.name = f"ADX_{lensig}"
    adxdf.category = "trend"

    return adxdf


# 设置 ADX 函数的文档字符串
adx.__doc__ = \
"""Average Directional Movement (ADX)

Average Directional Movement is meant to quantify trend strength by measuring
the amount of movement in a single direction.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/average-directional-movement-adx/
    TA Lib Correlation: >99%

Calculation:
        DMI ADX TREND 2.0 by @TraderR0BERT, NETWORTHIE.COM
            // 由 @TraderR0BERT, NETWORTHIE.COM 创建，最后更新日期为 01/26/2016
            // DMI 指标
            // 分辨率输入选项，用于更高/更低的时间框架
            study(title="DMI ADX TREND 2.0", shorttitle="ADX TREND 2.0")

        adxlen = input(14, title="ADX Smoothing")
        dilen = input(14, title="DI Length")
        thold = input(20, title="Threshold")

        threshold = thold

        //Script for Indicator
        dirmov(len) =>
            up = change(high)
            down = -change(low)
            truerange = rma(tr, len)
            plus = fixnan(100 * rma(up > down and up > 0 ? up : 0, len) / truerange)
            minus = fixnan(100 * rma(down > up and down > 0 ? down : 0, len) / truerange)
            [plus, minus]

        adx(dilen, adxlen) =>
            [plus, minus] = dirmov(dilen)
            sum = plus + minus
            adx = 100 * rma(abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)
            [adx, plus, minus]

        [sig, up, down] = adx(dilen, adxlen)
        osob=input(40,title="Exhaustion Level for ADX, default = 40")
        col = sig >= sig[1] ? green : sig <= sig[1] ? red : gray

        //Plot Definitions Current Timeframe
        p1 = plot(sig, color=col, linewidth = 3, title="ADX")
        p2 = plot(sig, color=col, style=circles, linewidth=3, title="ADX")
        p3 = plot(up, color=blue, linewidth = 3, title="+DI")
        p4 = plot(up, color=blue, style=circles, linewidth=3, title="+DI")
        p5 = plot(down, color=fuchsia, linewidth = 3, title="-DI")
        p6 = plot(down, color=fuchsia, style=circles, linewidth=3, title="-DI")
        h1 = plot(threshold, color=black, linewidth =3, title="Threshold")

        trender = (sig >= up or sig >= down) ? 1 : 0
        bgcolor(trender>0?black:gray, transp=85)

        //Alert Function for ADX crossing Threshold
        Up_Cross = crossover(up, threshold)
        alertcondition(Up_Cross, title="DMI+ cross", message="DMI+ Crossing Threshold")
        Down_Cross = crossover(down, threshold)
        alertcondition(Down_Cross, title="DMI- cross", message="DMI- Crossing Threshold")
# 定义函数参数
Args:
    high (pd.Series): 'high' 数据序列
    low (pd.Series): 'low' 数据序列
    close (pd.Series): 'close' 数据序列
    length (int): 周期长度，默认为14
    lensig (int): 信号长度，类似于 TradingView 的默认 ADX 长度，默认为 length
    scalar (float): 放大倍数，默认为100
    mamode (str): 参考 ```help(ta.ma)```py，默认为 'rma'
    drift (int): 差异周期，默认为1
    offset (int): 结果偏移周期数，默认为0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value) 的填充值
    fill_method (value, optional): 填充方法类型

Returns:
    pd.DataFrame: 包含 adx、dmp、dmn 列的数据框
```