# `.\pandas-ta\pandas_ta\trend\xsignals.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 中导入 nan 并重命名为 npNaN
from numpy import nan as npNaN
# 从 pandas 中导入 DataFrame
from pandas import DataFrame
# 从当前包中导入 tsignals 模块
from .tsignals import tsignals
# 从 pandas_ta.utils._signals 中导入 cross_value 函数
from pandas_ta.utils._signals import cross_value
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 xsignals，用于计算交叉信号
def xsignals(signal, xa, xb, above:bool=True, long:bool=True, asbool:bool=None, trend_reset:int=0, trade_offset:int=None, offset:int=None, **kwargs):
    """Indicator: Cross Signals"""
    # 验证参数
    signal = verify_series(signal)
    offset = get_offset(offset)

    # 计算结果
    if above:
        # 如果 above 为 True，计算 signal 与 xa 交叉的位置
        entries = cross_value(signal, xa)
        # 计算 signal 与 xb 交叉的位置，注意指定 above=False
        exits = -cross_value(signal, xb, above=False)
    else:
        # 如果 above 为 False，计算 signal 与 xa 交叉的位置，注意指定 above=False
        entries = cross_value(signal, xa, above=False)
        # 计算 signal 与 xb 交叉的位置
        exits = -cross_value(signal, xb)
    # 计算交叉信号
    trades = entries + exits

    # 修改交叉信号以填充趋势间的间隙
    trades.replace({0: npNaN}, inplace=True)
    trades.interpolate(method="pad", inplace=True)
    trades.fillna(0, inplace=True)

    # 将交叉信号转换为趋势
    trends = (trades > 0).astype(int)
    if not long:
        trends = 1 - trends

    # 构建传递给 tsignals 函数的关键字参数字典
    tskwargs = {
        "asbool":asbool,
        "trade_offset":trade_offset,
        "trend_reset":trend_reset,
        "offset":offset
    }
    # 调用 tsignals 函数计算趋势信号
    df = tsignals(trends, **tskwargs)

    # 处理偏移，由 tsignals 函数处理
    DataFrame({
        f"XS_LONG": df.TS_Trends,
        f"XS_SHORT": 1 - df.TS_Trends
    })

    # 处理填充
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # 设定名称和类别
    df.name = f"XS"
    df.category = "trend"

    return df

# 设定函数文档字符串
xsignals.__doc__ = \
"""Cross Signals (XSIGNALS)

Cross Signals returns Trend Signal (TSIGNALS) results for Signal Crossings. This
is useful for indicators like RSI, ZSCORE, et al where one wants trade Entries
and Exits (and Trends).

Cross Signals has two kinds of modes: above and long.

The first mode 'above', default True, xsignals determines if the signal first
crosses above 'xa' and then below 'xb'. If 'above' is False, xsignals determines
if the signal first crosses below 'xa' and then above 'xb'.

The second mode 'long', default True, passes the long trend result into
tsignals so it can determine the appropriate Entries and Exits. When 'long' is
False, it does the same but for the short side.

Example:
# These are two different outcomes and depends on the indicator and it's
# characteristics. Please check BOTH outcomes BEFORE making an Issue.
rsi = df.ta.rsi()
# Returns tsignal DataFrame when RSI crosses above 20 and then below 80
ta.xsignals(rsi, 20, 80, above=True)
# Returns tsignal DataFrame when RSI crosses below 20 and then above 80
ta.xsignals(rsi, 20, 80, above=False)

Source: Kevin Johnson

Calculation:
    Default Inputs:
        asbool=False, trend_reset=0, trade_offset=0, drift=1

    trades = trends.diff().shift(trade_offset).fillna(0).astype(int)
    entries = (trades > 0).astype(int)
    exits = (trades < 0).abs().astype(int)

Args:
"""
    # 定义一个布尔值，表示信号是在'xa'之上首次穿越，然后再穿越'xb'，还是在'xa'之下首次穿越，然后再穿越'xb'
    above (bool): When the signal crosses above 'xa' first and then 'xb'. When
        False, then when the signal crosses below 'xa' first and then 'xb'.
        Default: True
    # 将长期趋势传递给tsignals的趋势参数。当为False时，将短期趋势传递给tsignals的趋势参数
    long (bool): Passes the long trend into tsignals' trend argument. When
        False, it passes the short trend into tsignals trend argument.
        Default: True
    # 差异期。默认值为1
    drift (int): The difference period. Default: 1
    # 结果的偏移量。默认值为0
    offset (int): How many periods to offset the result. Default: 0

    # TSIGNAL传递参数
    # 如果为True，则将Trends、Entries和Exits列转换为布尔值。当为布尔值时，也可用于使用vectorbt的Portfolio.from_signal(close, entries, exits)进行回测
    asbool (bool): If True, it converts the Trends, Entries and Exits columns to
        booleans. When boolean, it is also useful for backtesting with
        vectorbt's Portfolio.from_signal(close, entries, exits) Default: False
    # 用于识别趋势是否结束的值。默认值为0
    trend_reset (value): Value used to identify if a trend has ended. Default: 0
    # 用于移动交易进出的值。使用1进行回测，使用0进行实时交易。默认值为0
    trade_offset (value): Value used shift the trade entries/exits Use 1 for
        backtesting and 0 for live. Default: 0
# 函数参数说明，使用关键字参数传递给函数的参数列表
Kwargs:
    # fillna参数，用于填充缺失值的值，采用pd.DataFrame.fillna(value)方式
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method参数，填充缺失值的方法类型
    fill_method (value, optional): Type of fill method

# 返回值说明，返回一个pd.DataFrame对象，其包含以下列：
Returns:
    # Trends列，趋势（有趋势: 1，无趋势: 0）
    Trends (trend: 1, no trend: 0),
    # Trades列，交易（进入: 1，退出: -1，其他: 0）
    Trades (Enter: 1, Exit: -1, Otherwise: 0),
    # Entries列，入口（入口: 1，无: 0）
    Entries (entry: 1, nothing: 0),
    # Exits列，出口（出口: 1，无: 0）
    Exits (exit: 1, nothing: 0)
```