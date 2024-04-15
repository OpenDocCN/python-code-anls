# `.\pandas-ta\pandas_ta\trend\tsignals.py`

```
# -*- coding: utf-8 -*-

# 从 pandas 库导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.utils 模块导入 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义 tsignals 函数，用于计算趋势信号
def tsignals(trend, asbool=None, trend_reset=0, trade_offset=None, drift=None, offset=None, **kwargs):
    """Indicator: Trend Signals"""
    # 验证参数
    # 将 trend 参数转换为 pandas Series 对象
    trend = verify_series(trend)
    # 将 asbool 参数转换为布尔值，默认为 False
    asbool = bool(asbool) if isinstance(asbool, bool) else False
    # 将 trend_reset 参数转换为整数，默认为 0
    trend_reset = int(trend_reset) if trend_reset and isinstance(trend_reset, int) else 0
    # 如果 trade_offset 不为 0，则将其转换为整数，默认为 0
    if trade_offset != 0:
        trade_offset = int(trade_offset) if trade_offset and isinstance(trade_offset, int) else 0
    # 获取 drift 和 offset 参数的值
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 计算结果
    # 将趋势值转换为整数类型
    trends = trend.astype(int)
    # 计算交易信号
    trades = trends.diff(drift).shift(trade_offset).fillna(0).astype(int)
    # 计算进入交易的信号
    entries = (trades > 0).astype(int)
    # 计算退出交易的信号
    exits = (trades < 0).abs().astype(int)

    # 如果 asbool 为 True，则将结果转换为布尔值
    if asbool:
        trends = trends.astype(bool)
        entries = entries.astype(bool)
        exits = exits.astype(bool)

    # 构建结果数据
    data = {
        f"TS_Trends": trends,
        f"TS_Trades": trades,
        f"TS_Entries": entries,
        f"TS_Exits": exits,
    }
    # 创建 DataFrame 对象
    df = DataFrame(data, index=trends.index)

    # 处理偏移
    if offset != 0:
        df = df.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    df.name = f"TS"
    df.category = "trend"

    return df

# 设置 tsignals 函数的文档字符串
tsignals.__doc__ = \
"""Trend Signals

Given a Trend, Trend Signals returns the Trend, Trades, Entries and Exits as
boolean integers. When 'asbool=True', it returns Trends, Entries and Exits as
boolean values which is helpful when combined with the vectorbt backtesting
package.

A Trend can be a simple as: 'close' > 'moving average' or something more complex
whose values are boolean or integers (0 or 1).

Examples:
ta.tsignals(close > ta.sma(close, 50), asbool=False)
ta.tsignals(ta.ema(close, 8) > ta.ema(close, 21), asbool=True)

Source: Kevin Johnson

Calculation:
    Default Inputs:
        asbool=False, trend_reset=0, trade_offset=0, drift=1

    trades = trends.diff().shift(trade_offset).fillna(0).astype(int)
    entries = (trades > 0).astype(int)
    exits = (trades < 0).abs().astype(int)

Args:
    trend (pd.Series): Series of 'trend's. The trend can be either a boolean or
        integer series of '0's and '1's
    asbool (bool): If True, it converts the Trends, Entries and Exits columns to
        booleans. When boolean, it is also useful for backtesting with
        vectorbt's Portfolio.from_signal(close, entries, exits) Default: False
    trend_reset (value): Value used to identify if a trend has ended. Default: 0
    trade_offset (value): Value used shift the trade entries/exits Use 1 for
        backtesting and 0 for live. Default: 0
    drift (int): The difference period. Default: 1

"""
    offset (int): How many periods to offset the result. Default: 0
# 函数参数说明部分，描述了函数的参数和返回值
Kwargs:
    # fillna 参数，用于填充缺失值，使用 pd.DataFrame.fillna 函数
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method 参数，填充方法的类型说明
    fill_method (value, optional): Type of fill method

# 返回值说明部分，描述了函数返回的 DataFrame 的列
Returns:
    # 返回一个 pandas DataFrame，包含以下列：
    pd.DataFrame with columns:
    # Trends 列，表示趋势，有趋势为 1，无趋势为 0
    Trends (trend: 1, no trend: 0),
    # Trades 列，表示交易，进入为 1，退出为 -1，其他情况为 0
    Trades (Enter: 1, Exit: -1, Otherwise: 0),
    # Entries 列，表示进入，进入为 1，无操作为 0
    Entries (entry: 1, nothing: 0),
    # Exits 列，表示退出，退出为 1，无操作为 0
    Exits (exit: 1, nothing: 0)
```