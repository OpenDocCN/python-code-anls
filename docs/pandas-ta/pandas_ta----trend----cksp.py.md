# `.\pandas-ta\pandas_ta\trend\cksp.py`

```
# -*- coding: utf-8 -*-

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 atr 函数
from pandas_ta.volatility import atr
# 从 pandas_ta 库中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 cksp，计算 Chande Kroll Stop (CKSP) 指标
def cksp(high, low, close, p=None, x=None, q=None, tvmode=None, offset=None, **kwargs):
    """Indicator: Chande Kroll Stop (CKSP)"""
    
    # 验证参数
    p = int(p) if p and p > 0 else 10
    x = float(x) if x and x > 0 else 1 if tvmode is True else 3
    q = int(q) if q and q > 0 else 9 if tvmode is True else 20
    _length = max(p, q, x)

    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    if high is None or low is None or close is None: return

    offset = get_offset(offset)
    tvmode = tvmode if isinstance(tvmode, bool) else True
    mamode = "rma" if tvmode is True else "sma"

    # 计算结果
    atr_ = atr(high=high, low=low, close=close, length=p, mamode=mamode)

    long_stop_ = high.rolling(p).max() - x * atr_
    long_stop = long_stop_.rolling(q).max()

    short_stop_ = low.rolling(p).min() + x * atr_
    short_stop = short_stop_.rolling(q).min()

    # 偏移
    if offset != 0:
        long_stop = long_stop.shift(offset)
        short_stop = short_stop.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        long_stop.fillna(kwargs["fillna"], inplace=True)
        short_stop.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        long_stop.fillna(method=kwargs["fill_method"], inplace=True)
        short_stop.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    _props = f"_{p}_{x}_{q}"
    long_stop.name = f"CKSPl{_props}"
    short_stop.name = f"CKSPs{_props}"
    long_stop.category = short_stop.category = "trend"

    # 准备返回的 DataFrame
    ckspdf = DataFrame({long_stop.name: long_stop, short_stop.name: short_stop})
    ckspdf.name = f"CKSP{_props}"
    ckspdf.category = long_stop.category

    return ckspdf

# 设置函数 cksp 的文档字符串
cksp.__doc__ = \
"""Chande Kroll Stop (CKSP)

The Tushar Chande and Stanley Kroll in their book
“The New Technical Trader”. It is a trend-following indicator,
identifying your stop by calculating the average true range of
the recent market volatility. The indicator defaults to the implementation
found on tradingview but it provides the original book implementation as well,
which differs by the default periods and moving average mode. While the trading
view implementation uses the Welles Wilder moving average, the book uses a
simple moving average.

Sources:
    https://www.multicharts.com/discussion/viewtopic.php?t=48914
    "The New Technical Trader", Wikey 1st ed. ISBN 9780471597803, page 95

Calculation:
    Default Inputs:
        p=10, x=1, q=9, tvmode=True
    ATR = Average True Range

    LS0 = high.rolling(p).max() - x * ATR(length=p)
    LS = LS0.rolling(q).max()

    SS0 = high.rolling(p).min() + x * ATR(length=p)
    SS = SS0.rolling(q).min()

Args:
"""
    # 'close'是一个包含收盘价的Series对象
    # p是ATR和第一个停止期的值，以整数表示。在两种模式下默认值均为10
    # x是ATR的标量值，在Trading View模式下默认值为1，在其他模式下默认值为3
    # q是第二个停止期的值，以整数表示。在Trading View模式下默认值为9，在其他模式下默认值为20
    # tvmode是一个布尔值，表示是否使用Trading View模式或书中实现模式。默认为True表示使用Trading View模式
    # offset是结果的偏移周期数。默认值为0
# 定义函数的参数列表，这里使用了可选参数
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)  # 填充缺失值的数值
    fill_method (value, optional): Type of fill method  # 填充方法的类型

# 返回值说明
Returns:
    pd.DataFrame: long and short columns.  # 返回一个包含长列和短列的 Pandas 数据帧
```