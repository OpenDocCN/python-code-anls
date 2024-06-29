# `D:\src\scipysrc\pandas\pandas\_testing\_hypothesis.py`

```
"""
Hypothesis data generator helpers.
"""

# 从 datetime 模块导入 datetime 类
from datetime import datetime

# 导入 hypothesis 库中的 strategies 模块，并重命名为 st
from hypothesis import strategies as st

# 导入 hypothesis 库中的 extra.dateutil 模块中的 timezones，并重命名为 dateutil_timezones
from hypothesis.extra.dateutil import timezones as dateutil_timezones

# 从 pandas.compat 模块导入 is_platform_windows 函数
from pandas.compat import is_platform_windows

# 导入 pandas 库，并重命名为 pd
import pandas as pd

# 从 pandas.tseries.offsets 模块导入以下类
from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)

# OPTIONAL_INTS 是一个 Hypothesis 的策略，生成一个包含可选整数的列表，长度在3到10之间
OPTIONAL_INTS = st.lists(st.one_of(st.integers(), st.none()), max_size=10, min_size=3)

# OPTIONAL_FLOATS 是一个 Hypothesis 的策略，生成一个包含可选浮点数的列表，长度在3到10之间
OPTIONAL_FLOATS = st.lists(st.one_of(st.floats(), st.none()), max_size=10, min_size=3)

# OPTIONAL_TEXT 是一个 Hypothesis 的策略，生成一个包含可选文本字符串的列表，长度在3到10之间
OPTIONAL_TEXT = st.lists(st.one_of(st.none(), st.text()), max_size=10, min_size=3)

# OPTIONAL_DICTS 是一个 Hypothesis 的策略，生成一个包含可选字典的列表，字典的键是文本字符串，值是整数，长度在3到10之间
OPTIONAL_DICTS = st.lists(
    st.one_of(st.none(), st.dictionaries(st.text(), st.integers())),
    max_size=10,
    min_size=3,
)

# OPTIONAL_LISTS 是一个 Hypothesis 的策略，生成一个包含可选列表的列表，列表中包含文本字符串，长度在3到10之间
OPTIONAL_LISTS = st.lists(
    st.one_of(st.none(), st.lists(st.text(), max_size=10, min_size=3)),
    max_size=10,
    min_size=3,
)

# OPTIONAL_ONE_OF_ALL 是一个 Hypothesis 的策略，生成一个可选类型的列表，包括字典、浮点数、整数、列表和文本字符串
OPTIONAL_ONE_OF_ALL = st.one_of(
    OPTIONAL_DICTS, OPTIONAL_FLOATS, OPTIONAL_INTS, OPTIONAL_LISTS, OPTIONAL_TEXT
)

# 根据平台判断是否为 Windows，选择不同的 datetime 策略
if is_platform_windows():
    DATETIME_NO_TZ = st.datetimes(min_value=datetime(1900, 1, 1))
else:
    DATETIME_NO_TZ = st.datetimes()

# DATETIME_JAN_1_1900_OPTIONAL_TZ 是一个 Hypothesis 的策略，生成一个日期时间，包含时区信息，最小值和最大值为 1900 年 1 月 1 日的 pandas Timestamp
DATETIME_JAN_1_1900_OPTIONAL_TZ = st.datetimes(
    min_value=pd.Timestamp(1900, 1, 1).to_pydatetime(),
    max_value=pd.Timestamp(1900, 1, 1).to_pydatetime(),
    timezones=st.one_of(st.none(), dateutil_timezones(), st.timezones()),
)

# DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ 是一个 Hypothesis 的策略，生成一个不包含时区信息的日期时间，最小值和最大值由 pandas Timestamp 的最小和最大日期时间确定
DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ = st.datetimes(
    min_value=pd.Timestamp.min.to_pydatetime(warn=False),
    max_value=pd.Timestamp.max.to_pydatetime(warn=False),
)

# INT_NEG_999_TO_POS_999 是一个 Hypothesis 的策略，生成一个整数，范围在 -999 到 999 之间
INT_NEG_999_TO_POS_999 = st.integers(-999, 999)

# YQM_OFFSET 是一个 Hypothesis 的策略，生成一个月、季度或年的偏移量对象
YQM_OFFSET = st.one_of(
    *map(
        st.from_type,
        [
            MonthBegin,
            MonthEnd,
            BMonthBegin,
            BMonthEnd,
            QuarterBegin,
            QuarterEnd,
            BQuarterBegin,
            BQuarterEnd,
            YearBegin,
            YearEnd,
            BYearBegin,
            BYearEnd,
        ],
    )
)
```