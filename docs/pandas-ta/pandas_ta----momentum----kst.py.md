# `.\pandas-ta\pandas_ta\momentum\kst.py`

```py
# -*- coding: utf-8 -*-
# 导入DataFrame类
from pandas import DataFrame
# 导入roc函数
from .roc import roc
# 导入验证序列函数、获取漂移和偏移的函数
from pandas_ta.utils import get_drift, get_offset, verify_series


# 定义函数：'Know Sure Thing' (KST)
def kst(close, roc1=None, roc2=None, roc3=None, roc4=None, sma1=None, sma2=None, sma3=None, sma4=None, signal=None, drift=None, offset=None, **kwargs):
    """Indicator: 'Know Sure Thing' (KST)"""
    # 验证参数
    # 若roc1参数存在且大于0，则将其转换为整数，否则使用默认值10
    roc1 = int(roc1) if roc1 and roc1 > 0 else 10
    # 若roc2参数存在且大于0，则将其转换为整数，否则使用默认值15
    roc2 = int(roc2) if roc2 and roc2 > 0 else 15
    # 若roc3参数存在且大于0，则将其转换为整数，否则使用默认值20
    roc3 = int(roc3) if roc3 and roc3 > 0 else 20
    # 若roc4参数存在且大于0，则将其转换为整数，否则使用默认值30
    roc4 = int(roc4) if roc4 and roc4 > 0 else 30

    # 若sma1参数存在且大于0，则将其转换为整数，否则使用默认值10
    sma1 = int(sma1) if sma1 and sma1 > 0 else 10
    # 若sma2参数存在且大于0，则将其转换为整数，否则使用默认值10
    sma2 = int(sma2) if sma2 and sma2 > 0 else 10
    # 若sma3参数存在且大于0，则将其转换为整数，否则使用默认值10
    sma3 = int(sma3) if sma3 and sma3 > 0 else 10
    # 若sma4参数存在且大于0，则将其转换为整数，否则使用默认值15
    sma4 = int(sma4) if sma4 and sma4 > 0 else 15

    # 若signal参数存在且大于0，则将其转换为整数，否则使用默认值9
    signal = int(signal) if signal and signal > 0 else 9
    # 计算参数的最大值
    _length = max(roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signal)
    # 验证序列
    close = verify_series(close, _length)
    # 获取漂移和偏移
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 若close为空，则返回空值
    if close is None: return

    # 计算结果
    # 计算第一个ROC的移动平均值
    rocma1 = roc(close, roc1).rolling(sma1).mean()
    # 计算第二个ROC的移动平均值
    rocma2 = roc(close, roc2).rolling(sma2).mean()
    # 计算第三个ROC的移动平均值
    rocma3 = roc(close, roc3).rolling(sma3).mean()
    # 计算第四个ROC的移动平均值
    rocma4 = roc(close, roc4).rolling(sma4).mean()

    # 计算KST值
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    # 计算KST信号线
    kst_signal = kst.rolling(signal).mean()

    # 偏移
    if offset != 0:
        kst = kst.shift(offset)
        kst_signal = kst_signal.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        kst.fillna(kwargs["fillna"], inplace=True)
        kst_signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        kst.fillna(method=kwargs["fill_method"], inplace=True)
        kst_signal.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    # 命名KST指标
    kst.name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}"
    # 命名KST信号线
    kst_signal.name = f"KSTs_{signal}"
    # 分类为动量类指标
    kst.category = kst_signal.category = "momentum"

    # 准备返回的DataFrame
    data = {kst.name: kst, kst_signal.name: kst_signal}
    kstdf = DataFrame(data)
    # 命名DataFrame
    kstdf.name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}_{signal}"
    # 分类为动量类指标
    kstdf.category = "momentum"

    return kstdf


# 设置函数文档字符串
kst.__doc__ = \
"""'Know Sure Thing' (KST)

The 'Know Sure Thing' is a momentum based oscillator and based on ROC.

Sources:
    https://www.tradingview.com/wiki/Know_Sure_Thing_(KST)
    https://www.incrediblecharts.com/indicators/kst.php

Calculation:
    Default Inputs:
        roc1=10, roc2=15, roc3=20, roc4=30,
        sma1=10, sma2=10, sma3=10, sma4=15, signal=9, drift=1
    ROC = Rate of Change
    SMA = Simple Moving Average
    rocsma1 = SMA(ROC(close, roc1), sma1)
    rocsma2 = SMA(ROC(close, roc2), sma2)
    rocsma3 = SMA(ROC(close, roc3), sma3)
    rocsma4 = SMA(ROC(close, roc4), sma4)

    KST = 100 * (rocsma1 + 2 * rocsma2 + 3 * rocsma3 + 
    # 'close'是一个 pd.Series 对象，代表了股票的收盘价数据
    close (pd.Series): Series of 'close's
    # roc1 是 ROC 指标的第一个周期，缺省值为 10
    roc1 (int): ROC 1 period. Default: 10
    # roc2 是 ROC 指标的第二个周期，缺省值为 15
    roc2 (int): ROC 2 period. Default: 15
    # roc3 是 ROC 指标的第三个周期，缺省值为 20
    roc3 (int): ROC 3 period. Default: 20
    # roc4 是 ROC 指标的第四个周期，缺省值为 30
    roc4 (int): ROC 4 period. Default: 30
    # sma1 是简单移动平均线（SMA）的第一个周期，缺省值为 10
    sma1 (int): SMA 1 period. Default: 10
    # sma2 是简单移动平均线（SMA）的第二个周期，缺省值为 10
    sma2 (int): SMA 2 period. Default: 10
    # sma3 是简单移动平均线（SMA）的第三个周期，缺省值为 10
    sma3 (int): SMA 3 period. Default: 10
    # sma4 是简单移动平均线（SMA）的第四个周期，缺省值为 15
    sma4 (int): SMA 4 period. Default: 15
    # signal 是信号线的周期，缺省值为 9
    signal (int): It's period. Default: 9
    # drift 是差分周期，缺省值为 1
    drift (int): The difference period. Default: 1
    # offset 是结果的偏移周期数，缺省值为 0
    offset (int): How many periods to offset the result. Default: 0
Kwargs:
    # fillna参数用于填充缺失值，其值将传递给pd.DataFrame.fillna()方法
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method参数用于指定填充方法的类型
    fill_method (value, optional): Type of fill method

Returns:
    # 返回一个Pandas DataFrame对象，包含kst和kst_signal两列
    pd.DataFrame: kst and kst_signal columns
```