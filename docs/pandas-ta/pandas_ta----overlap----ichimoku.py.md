# `.\pandas-ta\pandas_ta\overlap\ichimoku.py`

```
# -*- coding: utf-8 -*-
# 导入所需的库
from pandas import date_range, DataFrame, RangeIndex, Timedelta
from .midprice import midprice
from pandas_ta.utils import get_offset, verify_series

# 定义 Ichimoku 函数，计算 Ichimoku Kinkō Hyō 指标
def ichimoku(high, low, close, tenkan=None, kijun=None, senkou=None, include_chikou=True, offset=None, **kwargs):
    """Indicator: Ichimoku Kinkō Hyō (Ichimoku)"""
    # 设置默认的参数值
    tenkan = int(tenkan) if tenkan and tenkan > 0 else 9
    kijun = int(kijun) if kijun and kijun > 0 else 26
    senkou = int(senkou) if senkou and senkou > 0 else 52
    _length = max(tenkan, kijun, senkou)
    # 验证输入的数据
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    # 根据参数设置是否包含未来数据
    if not kwargs.get("lookahead", True):
        include_chikou = False

    # 如果输入数据有缺失，则返回空值
    if high is None or low is None or close is None: return None, None

    # 计算 Ichimoku 指标的各个线
    tenkan_sen = midprice(high=high, low=low, length=tenkan)
    kijun_sen = midprice(high=high, low=low, length=kijun)
    span_a = 0.5 * (tenkan_sen + kijun_sen)
    span_b = midprice(high=high, low=low, length=senkou)

    # 复制 Span A 和 Span B 在移动之前的值
    _span_a = span_a[-kijun:].copy()
    _span_b = span_b[-kijun:].copy()

    # 移动 Span A 和 Span B 的值
    span_a = span_a.shift(kijun)
    span_b = span_b.shift(kijun)
    chikou_span = close.shift(-kijun)

    # 根据偏移量对数据进行偏移
    if offset != 0:
        tenkan_sen = tenkan_sen.shift(offset)
        kijun_sen = kijun_sen.shift(offset)
        span_a = span_a.shift(offset)
        span_b = span_b.shift(offset)
        chikou_span = chikou_span.shift(offset)

    # 处理缺失值
    if "fillna" in kwargs:
        span_a.fillna(kwargs["fillna"], inplace=True)
        span_b.fillna(kwargs["fillna"], inplace=True)
        chikou_span.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        span_a.fillna(method=kwargs["fill_method"], inplace=True)
        span_b.fillna(method=kwargs["fill_method"], inplace=True)
        chikou_span.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置各个线的名称和类别
    span_a.name = f"ISA_{tenkan}"
    span_b.name = f"ISB_{kijun}"
    tenkan_sen.name = f"ITS_{tenkan}"
    kijun_sen.name = f"IKS_{kijun}"
    chikou_span.name = f"ICS_{kijun}"

    chikou_span.category = kijun_sen.category = tenkan_sen.category = "trend"
    span_b.category = span_a.category = chikou_span

    # 准备 Ichimoku DataFrame
    data = {
        span_a.name: span_a,
        span_b.name: span_b,
        tenkan_sen.name: tenkan_sen,
        kijun_sen.name: kijun_sen,
    }
    if include_chikou:
        data[chikou_span.name] = chikou_span

    ichimokudf = DataFrame(data)
    ichimokudf.name = f"ICHIMOKU_{tenkan}_{kijun}_{senkou}"
    ichimokudf.category = "overlap"

    # 准备 Span DataFrame
    last = close.index[-1]
    # 如果收盘价索引的数据类型为 "int64"，执行以下操作
    if close.index.dtype == "int64":
        # 创建一个新的范围索引，起始于 last + 1，结束于 last + kijun + 1
        ext_index = RangeIndex(start=last + 1, stop=last + kijun + 1)
        # 创建一个空的 DataFrame，索引为 ext_index，列为 span_a.name 和 span_b.name
        spandf = DataFrame(index=ext_index, columns=[span_a.name, span_b.name])
        # 将 _span_a 和 _span_b 的索引设置为 ext_index
        _span_a.index = _span_b.index = ext_index
    # 如果收盘价索引的数据类型不为 "int64"，执行以下操作
    else:
        # 统计收盘价索引中各值的频次，并取出出现频次最多的值
        df_freq = close.index.value_counts().mode()[0]
        # 创建一个时间增量对象，时间间隔为 df_freq 天
        tdelta = Timedelta(df_freq, unit="d")
        # 创建一个新的日期范围，起始日期为 last + tdelta，包含 kijun 个工作日
        new_dt = date_range(start=last + tdelta, periods=kijun, freq="B")
        # 创建一个空的 DataFrame，索引为 new_dt，列为 span_a.name 和 span_b.name
        spandf = DataFrame(index=new_dt, columns=[span_a.name, span_b.name])
        # 将 _span_a 和 _span_b 的索引设置为 new_dt
    
    spandf[span_a.name] = _span_a
    spandf[span_b.name] = _span_b
    # 设置 spandf 的名称为特定字符串，包含 tenkan 和 kijun 的值
    spandf.name = f"ICHISPAN_{tenkan}_{kijun}"
    # 设置 spandf 的类别为 "overlap"
    spandf.category = "overlap"
    
    # 返回 ichimokudf 和 spandf
    return ichimokudf, spandf
# 将 ichimoku.__doc__ 的值设为字符串，用于描述 Ichimoku Kinkō Hyō（一种用于金融市场预测的模型）的计算方法和参数
ichimoku.__doc__ = \
"""Ichimoku Kinkō Hyō (ichimoku)

Developed Pre WWII as a forecasting model for financial markets.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/ichimoku-ich/

Calculation:
    Default Inputs:
        tenkan=9, kijun=26, senkou=52
    MIDPRICE = Midprice
    TENKAN_SEN = MIDPRICE(high, low, close, length=tenkan)
    KIJUN_SEN = MIDPRICE(high, low, close, length=kijun)
    CHIKOU_SPAN = close.shift(-kijun)

    SPAN_A = 0.5 * (TENKAN_SEN + KIJUN_SEN)
    SPAN_A = SPAN_A.shift(kijun)

    SPAN_B = MIDPRICE(high, low, close, length=senkou)
    SPAN_B = SPAN_B.shift(kijun)

Args:
    high (pd.Series): Series of 'high's  # high 数据序列
    low (pd.Series): Series of 'low's  # low 数据序列
    close (pd.Series): Series of 'close's  # close 数据序列
    tenkan (int): Tenkan period. Default: 9  # Tenkan 周期，默认为 9
    kijun (int): Kijun period. Default: 26  # Kijun 周期，默认为 26
    senkou (int): Senkou period. Default: 52  # Senkou 周期，默认为 52
    include_chikou (bool): Whether to include chikou component. Default: True  # 是否包含 chikou 组件，默认为 True
    offset (int): How many periods to offset the result. Default: 0  # 结果偏移的周期数，默认为 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)  # fillna 方法的参数，用于填充缺失值
    fill_method (value, optional): Type of fill method  # 填充方法的类型

Returns:
    pd.DataFrame: Two DataFrames.  # 返回两个 DataFrame
        For the visible period: spanA, spanB, tenkan_sen, kijun_sen,  # 可见期间的 DataFrame，包含 spanA、spanB、tenkan_sen、kijun_sen
            and chikou_span columns  # 以及 chikou_span 列
        For the forward looking period: spanA and spanB columns  # 未来观察期间的 DataFrame，包含 spanA 和 spanB 列
"""
```