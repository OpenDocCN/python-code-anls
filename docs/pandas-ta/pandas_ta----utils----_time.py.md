# `.\pandas-ta\pandas_ta\utils\_time.py`

```py
# -*- coding: utf-8 -*-

# 从 datetime 模块中导入 datetime 类
from datetime import datetime
# 从 time 模块中导入 localtime 和 perf_counter 函数
from time import localtime, perf_counter
# 从 typing 模块中导入 Tuple 类型
from typing import Tuple
# 从 pandas 模块中导入 DataFrame 和 Timestamp 类
from pandas import DataFrame, Timestamp
# 从 pandas_ta 模块中导入 EXCHANGE_TZ 和 RATE 变量
from pandas_ta import EXCHANGE_TZ, RATE

# 定义函数 df_dates，接受一个 DataFrame 和日期元组作为参数，返回过滤后的 DataFrame
def df_dates(df: DataFrame, dates: Tuple[str, list] = None) -> DataFrame:
    """Yields the DataFrame with the given dates"""
    # 若日期元组为空，则返回 None
    if dates is None: return None
    # 如果日期元组不是列表类型，则将其转换为列表
    if not isinstance(dates, list):
        dates = [dates]
    # 返回过滤后的 DataFrame，只包含日期元组中指定的日期
    return df[df.index.isin(dates)]

# 定义函数 df_month_to_date，接受一个 DataFrame 作为参数，返回当月的 DataFrame
def df_month_to_date(df: DataFrame) -> DataFrame:
    """Yields the Month-to-Date (MTD) DataFrame"""
    # 获取当前日期的月初日期，并判断 DataFrame 的索引是否大于等于该日期
    in_mtd = df.index >= Timestamp.now().strftime("%Y-%m-01")
    # 如果有数据在当月，则返回当月的 DataFrame
    if any(in_mtd): return df[in_mtd]
    # 否则返回原始 DataFrame
    return df

# 定义函数 df_quarter_to_date，接受一个 DataFrame 作为参数，返回当季的 DataFrame
def df_quarter_to_date(df: DataFrame) -> DataFrame:
    """Yields the Quarter-to-Date (QTD) DataFrame"""
    # 获取当前日期，并遍历季度开始的月份
    now = Timestamp.now()
    for m in [1, 4, 7, 10]:
        # 如果当前月份小于等于遍历到的月份
        if now.month <= m:
                # 获取季度开始日期，并判断 DataFrame 的索引是否大于等于该日期
                in_qtr = df.index >= datetime(now.year, m, 1).strftime("%Y-%m-01")
                # 如果有数据在当季，则返回当季的 DataFrame
                if any(in_qtr): return df[in_qtr]
    # 否则返回从当前月份开始的 DataFrame
    return df[df.index >= now.strftime("%Y-%m-01")]

# 定义函数 df_year_to_date，接受一个 DataFrame 作为参数，返回当年的 DataFrame
def df_year_to_date(df: DataFrame) -> DataFrame:
    """Yields the Year-to-Date (YTD) DataFrame"""
    # 获取当前日期的年初日期，并判断 DataFrame 的索引是否大于等于该日期
    in_ytd = df.index >= Timestamp.now().strftime("%Y-01-01")
    # 如果有数据在当年，则返回当年的 DataFrame
    if any(in_ytd): return df[in_ytd]
    # 否则返回原始 DataFrame
    return df

# 定义函数 final_time，接受一个起始时间参数，返回从起始时间到当前时间的耗时
def final_time(stime: float) -> str:
    """Human readable elapsed time. Calculates the final time elasped since
    stime and returns a string with microseconds and seconds."""
    # 计算当前时间与起始时间的差值
    time_diff = perf_counter() - stime
    # 返回耗时的字符串，包含毫秒和秒
    return f"{time_diff * 1000:2.4f} ms ({time_diff:2.4f} s)"

# 定义函数 get_time，接受交易所名称、是否显示全信息和是否返回字符串作为参数，返回当前时间及交易所时间信息
def get_time(exchange: str = "NYSE", full:bool = True, to_string:bool = False) -> Tuple[None, str]:
    """Returns Current Time, Day of the Year and Percentage, and the current
    time of the selected Exchange."""
    # 默认交易所时区为东部时间（NYSE）
    tz = EXCHANGE_TZ["NYSE"] 
    # 如果传入的交易所名称为字符串类型
    if isinstance(exchange, str):
        # 将交易所名称转换为大写
        exchange = exchange.upper()
        # 获取对应交易所的时区信息
        tz = EXCHANGE_TZ[exchange]

    # 获取当前时间
    today = Timestamp.now()
    # 格式化日期字符串
    date = f"{today.day_name()} {today.month_name()} {today.day}, {today.year}"

    # 获取当前时间在交易所时区的时间
    _today = today.timetuple()
    exchange_time = f"{(_today.tm_hour + tz) % 24}:{_today.tm_min:02d}:{_today.tm_sec:02d}"

    # 如果需要显示全信息
    if full:
        # 获取本地时间信息
        lt = localtime()
        local_ = f"Local: {lt.tm_hour}:{lt.tm_min:02d}:{lt.tm_sec:02d} {lt.tm_zone}"
        # 计算当天在一年中的日期和百分比
        doy = f"Day {today.dayofyear}/365 ({100 * round(today.dayofyear/365, 2):.2f}%)"
        exchange_ = f"{exchange}: {exchange_time}"

        # 构建包含完整信息的字符串
        s = f"{date}, {exchange_}, {local_}, {doy}"
    else:
        # 构建简略信息的字符串
        s = f"{date}, {exchange}: {exchange_time}"

    # 如果需要返回字符串，则返回构建的字符串，否则打印字符串并返回 None
    return s if to_string else print(s)

# 定义函数 total_time，接受一个 DataFrame 和时间间隔类型参数作为输入，返回 DataFrame 的总时间间隔
def total_time(df: DataFrame, tf: str = "years") -> float:
    """Calculates the total time of a DataFrame. Difference of the Last and
    First index. Options: 'months', 'weeks', 'days', 'hours', 'minutes'
    and 'seconds'. Default: 'years'.
    Useful for annualization."""
    # 计算 DataFrame 的总时间间
    TimeFrame = {
        "years": time_diff.days / RATE["TRADING_DAYS_PER_YEAR"],  # 计算时间差对应的年数
        "months": time_diff.days / 30.417,  # 计算时间差对应的月数
        "weeks": time_diff.days / 7,  # 计算时间差对应的周数
        "days": time_diff.days,  # 计算时间差对应的天数
        "hours": time_diff.days * 24,  # 计算时间差对应的小时数
        "minutes": time_diff.total_seconds() / 60,  # 计算时间差对应的分钟数
        "seconds": time_diff.total_seconds()  # 计算时间差对应的秒数
    }

    if isinstance(tf, str) and tf in TimeFrame.keys():  # 检查 tf 是否为字符串且在 TimeFrame 字典的键中
        return TimeFrame[tf]  # 返回对应 tf 的时间差
    return TimeFrame["years"]  # 如果 tf 不在 TimeFrame 字典的键中，则返回默认的年数时间差
# 将 DataFrame 的索引转换为 UTC 时区，或者使用 tz_convert 将索引设置为 UTC 时区
def to_utc(df: DataFrame) -> DataFrame:
    # 检查 DataFrame 是否为空
    if not df.empty:
        try:
            # 尝试将索引本地化为 UTC 时区
            df.index = df.index.tz_localize("UTC")
        except TypeError:
            # 如果出现 TypeError，则使用 tz_convert 将索引转换为 UTC 时区
            df.index = df.index.tz_convert("UTC")
    # 返回处理后的 DataFrame
    return df

# 别名
mtd = df_month_to_date
qtd = df_quarter_to_date
ytd = df_year_to_date
```