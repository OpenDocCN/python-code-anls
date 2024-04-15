# `.\pandas-ta\pandas_ta\momentum\stc.py`

```
# -*- coding: utf-8 -*-
从 pandas 库中导入 DataFrame 和 Series 类
从 pandas_ta.overlap 模块中导入 ema 函数
从 pandas_ta.utils 模块中导入 get_offset、non_zero_range 和 verify_series 函数


# 定义函数：Schaff Trend Cycle (STC)
def stc(close, tclength=None, fast=None, slow=None, factor=None, offset=None, **kwargs):
    # 验证参数
    tclength = int(tclength) if tclength and tclength > 0 else 10
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    factor = float(factor) if factor and factor > 0 else 0.5
    # 如果慢线小于快线，则交换它们的值
    if slow < fast:                
        fast, slow = slow, fast
    # 计算所需数据的长度，取最大值
    _length = max(tclength, fast, slow)
    # 验证收盘价数据，返回验证后的 Series 对象
    close = verify_series(close, _length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果收盘价为 None，则返回
    if close is None: return

    # kwargs 允许传递三个更多的 Series（ma1、ma2 和 osc），这些可以在这里传递，
    # ma1 和 ma2 输入会抵消内部的 ema 计算，osc 替代了两个 ma。
    ma1 = kwargs.pop("ma1", False)
    ma2 = kwargs.pop("ma2", False)
    osc = kwargs.pop("osc", False)

    # 3 种不同的计算模式..
    if isinstance(ma1, Series) and isinstance(ma2, Series) and not osc:
        # 验证输入的两个外部 Series 对象
        ma1 = verify_series(ma1, _length)
        ma2 = verify_series(ma2, _length)

        # 如果其中一个为 None，则返回
        if ma1 is None or ma2 is None: return
        # 根据外部提供的 Series 计算结果
        xmacd = ma1 - ma2
        # 调用共享计算函数
        pff, pf = schaff_tc(close, xmacd, tclength, factor)

    elif isinstance(osc, Series):
        # 验证输入的振荡器 Series 对象
        osc = verify_series(osc, _length)
        # 如果为 None，则返回
        if osc is None: return
        # 根据提供的振荡器计算结果（应在 0 轴附近）
        xmacd = osc
        # 调用共享计算函数
        pff, pf = schaff_tc(close, xmacd, tclength, factor)

    else:
        # 计算结果..（传统/完整）
        # MACD 线
        fastma = ema(close, length=fast)
        slowma = ema(close, length=slow)
        xmacd = fastma - slowma
        # 调用共享计算函数
        pff, pf = schaff_tc(close, xmacd, tclength, factor)

    # 结果 Series
    stc = Series(pff, index=close.index)
    macd = Series(xmacd, index=close.index)
    stoch = Series(pf, index=close.index)

    # 偏移
    if offset != 0:
        stc = stc.shift(offset)
        macd = macd.shift(offset)
        stoch = stoch.shift(offset)

    # 填充缺失值
    if "fillna" in kwargs:
        stc.fillna(kwargs["fillna"], inplace=True)
        macd.fillna(kwargs["fillna"], inplace=True)
        stoch.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stc.fillna(method=kwargs["fill_method"], inplace=True)
        macd.fillna(method=kwargs["fill_method"], inplace=True)
        stoch.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    _props = f"_{tclength}_{fast}_{slow}_{factor}"
    stc.name = f"STC{_props}"
    macd.name = f"STCmacd{_props}"
    # 设置 stoch 对象的名称属性为包含 _props 的字符串
    stoch.name = f"STCstoch{_props}"
    # 设置 stc 和 macd 对象的 category 属性为 "momentum"
    stc.category = macd.category = stoch.category ="momentum"

    # 准备要返回的 DataFrame
    # 创建一个字典，包含 stc.name、macd.name 和 stoch.name 作为键，对应的对象为值
    data = {stc.name: stc, macd.name: macd, stoch.name: stoch}
    # 用 data 字典创建 DataFrame 对象
    df = DataFrame(data)
    # 设置 DataFrame 对象的名称属性为包含 _props 的字符串
    df.name = f"STC{_props}"
    # 设置 DataFrame 对象的 category 属性为 stc 对象的 category 属性
    df.category = stc.category

    # 返回 DataFrame 对象
    return df
# 设置 stc 的文档字符串，描述 Schaff Trend Cycle（STC）指标的计算方法和用法
stc.__doc__ = \
"""Schaff Trend Cycle (STC)

The Schaff Trend Cycle is an evolution of the popular MACD incorportating two
cascaded stochastic calculations with additional smoothing.

The STC returns also the beginning MACD result as well as the result after the
first stochastic including its smoothing. This implementation has been extended
for Pandas TA to also allow for separatly feeding any other two moving Averages
(as ma1 and ma2) or to skip this to feed an oscillator (osc), based on which the
Schaff Trend Cycle should be calculated.

Feed external moving averages:
Internally calculation..
    stc = ta.stc(close=df["close"], tclen=stc_tclen, fast=ma1_interval, slow=ma2_interval, factor=stc_factor)
becomes..
    extMa1 = df.ta.zlma(close=df["close"], length=ma1_interval, append=True)
    extMa2 = df.ta.ema(close=df["close"], length=ma2_interval, append=True)
    stc = ta.stc(close=df["close"], tclen=stc_tclen, ma1=extMa1, ma2=extMa2, factor=stc_factor)

The same goes for osc=, which allows the input of an externally calculated oscillator, overriding ma1 & ma2.


Sources:
    Implemented by rengel8 based on work found here:
    https://www.prorealcode.com/prorealtime-indicators/schaff-trend-cycle2/

Calculation:
    STCmacd = Moving Average Convergance/Divergance or Oscillator
    STCstoch = Intermediate Stochastic of MACD/Osc.
    2nd Stochastic including filtering with results in the
    STC = Schaff Trend Cycle

Args:
    close (pd.Series): Series of 'close's, used for indexing Series, mandatory
    tclen (int): SchaffTC Signal-Line length.  Default: 10 (adjust to the half of cycle)
    fast (int): The short period.   Default: 12
    slow (int): The long period.   Default: 26
    factor (float): smoothing factor for last stoch. calculation.   Default: 0.5
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    ma1: 1st moving average provided externally (mandatory in conjuction with ma2)
    ma2: 2nd moving average provided externally (mandatory in conjuction with ma1)
    osc: an externally feeded osillator
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: stc, macd, stoch
"""

# 定义 Schaff Trend Cycle（STC）计算函数
def schaff_tc(close, xmacd, tclength, factor):
    # 实际计算部分，这部分计算适用于不同的操作模式
    # 1. MACD 的 Stochastic
    # 计算区间 tclen 内 xmacd 的最小值
    lowest_xmacd = xmacd.rolling(tclength).min()  # min value in interval tclen
    # 计算区间 tclen 内 xmacd 的范围（最大值 - 最小值）
    xmacd_range = non_zero_range(xmacd.rolling(tclength).max(), lowest_xmacd)
    # 获取 xmacd 的长度
    m = len(xmacd)

    # 计算 MACD 的快速 %K
    # 初始化 stoch1 和 pf 列表
    stoch1, pf = list(xmacd), list(xmacd)
    # 第一个元素的值为 0
    stoch1[0], pf[0] = 0, 0
    # 循环计算 stoch1 和 pf 列表中的值
    for i in range(1, m):
        # 如果 lowest_xmacd[i] 大于 0，则计算快速 %K
        if lowest_xmacd[i] > 0:
            stoch1[i] = 100 * ((xmacd[i] - lowest_xmacd[i]) / xmacd_range[i])
        else:
            # 否则保持前一个值不变
            stoch1[i] = stoch1[i - 1]
        # 计算平滑后的 %D
        pf[i] = round(pf[i - 1] + (factor * (stoch1[i] - pf[i - 1])), 8)
    # 将 pf 转换为 Series 类型，并以 close 的索引为索引
    pf = Series(pf, index=close.index)

    # 计算平滑后的 Percent Fast D, 'PF' 的随机指标
    # 计算滚动窗口为 tclength 的最小值
    lowest_pf = pf.rolling(tclength).min()
    # 计算 pf 在滚动窗口为 tclength 的范围内的非零范围
    pf_range = non_zero_range(pf.rolling(tclength).max(), lowest_pf)

    # 计算 % Fast K of PF
    stoch2, pff = list(xmacd), list(xmacd)
    stoch2[0], pff[0] = 0, 0
    for i in range(1, m):
        if pf_range[i] > 0:
            # 计算 % Fast K of PF
            stoch2[i] = 100 * ((pf[i] - lowest_pf[i]) / pf_range[i])
        else:
            stoch2[i] = stoch2[i - 1]
        # 计算平滑后的 % Fast D of PF
        # 使用平滑因子 factor 进行平滑计算
        pff[i] = round(pff[i - 1] + (factor * (stoch2[i] - pff[i - 1])), 8)

    # 返回平滑后的 % Fast D of PF 和原始的 PF
    return [pff, pf]
```  
```