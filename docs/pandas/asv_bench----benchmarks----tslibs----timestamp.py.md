# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\timestamp.py`

```
from datetime import (
    datetime,
    timezone,
)
import zoneinfo

import numpy as np

from pandas import Timestamp

from .tslib import _tzs


class TimestampConstruction:
    def setup(self):
        # 初始化一个 numpy 的 datetime64 对象，表示2020年1月1日 00:00:00
        self.npdatetime64 = np.datetime64("2020-01-01 00:00:00")
        # 初始化一个没有时区信息的 datetime 对象，表示2020年1月1日 00:00:00
        self.dttime_unaware = datetime(2020, 1, 1, 0, 0, 0)
        # 初始化一个带有时区信息的 datetime 对象，表示2020年1月1日 00:00:00 UTC时区
        self.dttime_aware = datetime(2020, 1, 1, 0, 0, 0, 0, timezone.utc)
        # 初始化一个 pandas Timestamp 对象，表示2020年1月1日 00:00:00
        self.ts = Timestamp("2020-01-01 00:00:00")

    def time_parse_iso8601_no_tz(self):
        # 解析 ISO8601 格式的时间字符串，没有时区信息
        Timestamp("2017-08-25 08:16:14")

    def time_parse_iso8601_tz(self):
        # 解析 ISO8601 格式的时间字符串，带有时区信息 (-0500 表示美国东部时间)
        Timestamp("2017-08-25 08:16:14-0500")

    def time_parse_dateutil(self):
        # 使用 dateutil 解析时间字符串，支持多种格式，这里是以 AM/PM 格式示例
        Timestamp("2017/08/25 08:16:14 AM")

    def time_parse_today(self):
        # 解析字符串 "today"，返回今天的日期
        Timestamp("today")

    def time_parse_now(self):
        # 解析字符串 "now"，返回当前的日期和时间
        Timestamp("now")

    def time_fromordinal(self):
        # 从序数创建 Timestamp 对象，730120 表示从 0001-01-01 起的天数
        Timestamp.fromordinal(730120)

    def time_fromtimestamp(self):
        # 从时间戳创建 Timestamp 对象，1515448538 表示从1970-01-01 00:00:00 UTC开始的秒数
        Timestamp.fromtimestamp(1515448538)

    def time_from_npdatetime64(self):
        # 从 numpy datetime64 对象创建 Timestamp 对象
        Timestamp(self.npdatetime64)

    def time_from_datetime_unaware(self):
        # 从没有时区信息的 datetime 对象创建 Timestamp 对象
        Timestamp(self.dttime_unaware)

    def time_from_datetime_aware(self):
        # 从带有时区信息的 datetime 对象创建 Timestamp 对象
        Timestamp(self.dttime_aware)

    def time_from_pd_timestamp(self):
        # 从 pandas Timestamp 对象创建 Timestamp 对象
        Timestamp(self.ts)


class TimestampProperties:
    params = [_tzs]
    param_names = ["tz"]

    def setup(self, tz):
        # 初始化一个带有时区信息的 Timestamp 对象，使用给定的时区 tzinfo
        self.ts = Timestamp("2017-08-25 08:16:14", tzinfo=tz)

    def time_tz(self, tz):
        # 获取 Timestamp 对象的时区信息
        self.ts.tz

    def time_dayofweek(self, tz):
        # 获取 Timestamp 对象表示的星期几（0-6，0 表示星期一）
        self.ts.dayofweek

    def time_dayofyear(self, tz):
        # 获取 Timestamp 对象在年份中的第几天（1-365）
        self.ts.dayofyear

    def time_week(self, tz):
        # 获取 Timestamp 对象所在的一年中的第几周（1-52）
        self.ts.week

    def time_quarter(self, tz):
        # 获取 Timestamp 对象所在的季度（1-4）
        self.ts.quarter

    def time_days_in_month(self, tz):
        # 获取 Timestamp 对象所在月份的天数
        self.ts.days_in_month

    def time_is_month_start(self, tz):
        # 检查 Timestamp 对象是否是所在月份的第一天
        self.ts.is_month_start

    def time_is_month_end(self, tz):
        # 检查 Timestamp 对象是否是所在月份的最后一天
        self.ts.is_month_end

    def time_is_quarter_start(self, tz):
        # 检查 Timestamp 对象是否是所在季度的第一天
        self.ts.is_quarter_start

    def time_is_quarter_end(self, tz):
        # 检查 Timestamp 对象是否是所在季度的最后一天
        self.ts.is_quarter_end

    def time_is_year_start(self, tz):
        # 检查 Timestamp 对象是否是所在年份的第一天
        self.ts.is_year_start

    def time_is_year_end(self, tz):
        # 检查 Timestamp 对象是否是所在年份的最后一天
        self.ts.is_year_end

    def time_is_leap_year(self, tz):
        # 检查 Timestamp 对象所在年份是否是闰年
        self.ts.is_leap_year

    def time_microsecond(self, tz):
        # 获取 Timestamp 对象的微秒数部分
        self.ts.microsecond

    def time_month_name(self, tz):
        # 获取 Timestamp 对象所在月份的英文名称
        self.ts.month_name()

    def time_weekday_name(self, tz):
        # 获取 Timestamp 对象所在星期几的英文名称
        self.ts.day_name()


class TimestampOps:
    params = _tzs
    param_names = ["tz"]

    def setup(self, tz):
        # 初始化一个带有时区信息的 Timestamp 对象，使用给定的时区 tz
        self.ts = Timestamp("2017-08-25 08:16:14", tz=tz)

    def time_replace_tz(self, tz):
        # 替换 Timestamp 对象的时区信息为指定的时区 "US/Eastern"
        self.ts.replace(tzinfo=zoneinfo.ZoneInfo("US/Eastern"))

    def time_replace_None(self, tz):
        # 移除 Timestamp 对象的时区信息
        self.ts.replace(tzinfo=None)

    def time_to_pydatetime(self, tz):
        # 将 Timestamp 对象转换为 Python 的 datetime 对象
        self.ts.to_pydatetime()

    def time_normalize(self, tz):
        # 标准化 Timestamp 对象，确保其时区和日期时间表示符合一致的规范
        self.ts.normalize()
    # 如果时间戳对象已经有时区信息，则进行时区转换
    def time_tz_convert(self, tz):
        if self.ts.tz is not None:
            self.ts.tz_convert(tz)

    # 如果时间戳对象没有时区信息，则进行时区本地化
    def time_tz_localize(self, tz):
        if self.ts.tz is None:
            self.ts.tz_localize(tz)

    # 将时间戳对象转换为儒略日
    def time_to_julian_date(self, tz):
        self.ts.to_julian_date()

    # 对时间戳对象进行向下取整，精度为5分钟
    def time_floor(self, tz):
        self.ts.floor("5min")

    # 对时间戳对象进行向上取整，精度为5分钟
    def time_ceil(self, tz):
        self.ts.ceil("5min")
class TimestampAcrossDst:
    # 初始化方法，设置时区信息和时间戳对象
    def setup(self):
        # 创建一个指定日期时间的 datetime 对象，fold=0 表示不使用重复的小时
        dt = datetime(2016, 3, 27, 1, fold=0)
        # 将 datetime 对象转换为 Europe/Berlin 时区的 tzinfo 对象，并保存到实例变量中
        self.tzinfo = dt.astimezone(zoneinfo.ZoneInfo("Europe/Berlin")).tzinfo
        # 使用转换后的时区信息初始化 Timestamp 对象，并保存到实例变量中
        self.ts2 = Timestamp(dt)

    # 替换时间戳对象的时区信息
    def time_replace_across_dst(self):
        # 调用 Timestamp 对象的 replace 方法，替换时区信息为 setup 方法中保存的时区信息
        self.ts2.replace(tzinfo=self.tzinfo)
```