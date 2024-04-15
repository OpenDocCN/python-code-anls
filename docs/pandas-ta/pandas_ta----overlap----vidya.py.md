# `.\pandas-ta\pandas_ta\overlap\vidya.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 库中的 nan 并重命名为 npNaN
from numpy import nan as npNaN
# 从 pandas 库中导入 Series 类
from pandas import Series
# 从 pandas_ta.utils 模块中导入 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义变量指数动态平均值（VIDYA）指标函数
def vidya(close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Variable Index Dynamic Average (VIDYA)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 14
    length = int(length) if length and length > 0 else 14
    # 验证 close 是否为 Series 类型，并且长度符合要求
    close = verify_series(close, length)
    # 获取漂移值
    drift = get_drift(drift)
    # 获取偏移值
    offset = get_offset(offset)

    # 如果 close 为 None，则返回空
    if close is None: return

    # 定义 Chande 动量振荡器（CMO）函数
    def _cmo(source: Series, n:int , d: int):
        """Chande Momentum Oscillator (CMO) Patch
        For some reason: from pandas_ta.momentum import cmo causes
        pandas_ta.momentum.coppock to not be able to import it's
        wma like from pandas_ta.overlap import wma?
        Weird Circular TypeError!?!
        """
        # 计算动量
        mom = source.diff(d)
        # 获取正值
        positive = mom.copy().clip(lower=0)
        # 获取负值
        negative = mom.copy().clip(upper=0).abs()
        # 计算正值的滚动和
        pos_sum = positive.rolling(n).sum()
        # 计算负值的滚动和
        neg_sum = negative.rolling(n).sum()
        # 返回 CMO 值
        return (pos_sum - neg_sum) / (pos_sum + neg_sum)

    # 计算结果
    m = close.size
    alpha = 2 / (length + 1)
    abs_cmo = _cmo(close, length, drift).abs()
    vidya = Series(0, index=close.index)
    for i in range(length, m):
        vidya.iloc[i] = alpha * abs_cmo.iloc[i] * close.iloc[i] + vidya.iloc[i - 1] * (1 - alpha * abs_cmo.iloc[i])
    vidya.replace({0: npNaN}, inplace=True)

    # 偏移
    if offset != 0:
        vidya = vidya.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        vidya.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vidya.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    vidya.name = f"VIDYA_{length}"
    vidya.category = "overlap"

    return vidya


# 设置函数文档字符串
vidya.__doc__ = \
"""Variable Index Dynamic Average (VIDYA)

Variable Index Dynamic Average (VIDYA) was developed by Tushar Chande. It is
similar to an Exponential Moving Average but it has a dynamically adjusted
lookback period dependent on relative price volatility as measured by Chande
Momentum Oscillator (CMO). When volatility is high, VIDYA reacts faster to
price changes. It is often used as moving average or trend identifier.

Sources:
    https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/
    https://www.perfecttrendsystem.com/blog_mt4_2/en/vidya-indicator-for-mt4

Calculation:
    Default Inputs:
        length=10, adjust=False, sma=True
    if sma:
        sma_nth = close[0:length].sum() / length
        close[:length - 1] = np.NaN
        close.iloc[length - 1] = sma_nth
    EMA = close.ewm(span=length, adjust=adjust).mean()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool, optional): Use adjust option for EMA calculation. Default: False

"""
    # sma (bool, optional): 如果为 True，使用简单移动平均（SMA）作为 EMA 计算的初始值。默认为 True
    sma (bool, optional): If True, uses SMA for initial value for EMA calculation. Default: True
    
    # talib (bool): 如果为 True，则使用 TA-Lib 的实现来计算 CMO。否则使用 EMA 版本。默认为 True
    talib (bool): If True, uses TA-Libs implementation for CMO. Otherwise uses EMA version. Default: True
    
    # fillna (value, optional): pd.DataFrame.fillna(value) 的参数，用于指定填充缺失值的值
    fillna (value, optional): Parameter for pd.DataFrame.fillna(value) to specify the value to fill missing values with.
    
    # fill_method (value, optional): 填充缺失值的方法类型
    fill_method (value, optional): Type of fill method.
# 返回类型声明：pd.Series，表示生成的新特征是一个 Pandas 的 Series 对象
pd.Series: New feature generated.
```