# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\timedelta.py`

```
"""
Timedelta benchmarks that rely only on tslibs. See benchmarks.timedeltas for
Timedelta benchmarks that rely on other parts of pandas.
"""

import datetime  # 导入 datetime 库，用于处理日期时间相关操作

import numpy as np  # 导入 numpy 库，用于科学计算

from pandas import Timedelta  # 从 pandas 库中导入 Timedelta 类，用于处理时间差

class TimedeltaConstructor:
    def setup(self):
        self.nptimedelta64 = np.timedelta64(3600)  # 创建一个 numpy timedelta64 对象，表示3600秒
        self.dttimedelta = datetime.timedelta(seconds=3600)  # 创建一个 datetime timedelta 对象，表示3600秒
        self.td = Timedelta(3600, unit="s")  # 创建一个 pandas Timedelta 对象，表示3600秒

    def time_from_int(self):
        Timedelta(123456789)  # 从整数创建 Timedelta 对象

    def time_from_unit(self):
        Timedelta(1, unit="D")  # 从单位为天的整数创建 Timedelta 对象

    def time_from_components(self):
        Timedelta(
            days=1,
            hours=2,
            minutes=3,
            seconds=4,
            milliseconds=5,
            microseconds=6,
            nanoseconds=7,
        )  # 从各个时间单元创建 Timedelta 对象

    def time_from_datetime_timedelta(self):
        Timedelta(self.dttimedelta)  # 从 datetime.timedelta 对象创建 Timedelta 对象

    def time_from_np_timedelta(self):
        Timedelta(self.nptimedelta64)  # 从 numpy.timedelta64 对象创建 Timedelta 对象

    def time_from_string(self):
        Timedelta("1 days")  # 从字符串创建 Timedelta 对象

    def time_from_iso_format(self):
        Timedelta("P4DT12H30M5S")  # 从 ISO 8601 格式字符串创建 Timedelta 对象

    def time_from_missing(self):
        Timedelta("nat")  # 创建一个表示 'nat'（Not a Time）的 Timedelta 对象

    def time_from_pd_timedelta(self):
        Timedelta(self.td)  # 从 pandas Timedelta 对象创建 Timedelta 对象


class TimedeltaProperties:
    def setup_cache(self):
        td = Timedelta(days=365, minutes=35, seconds=25, milliseconds=35)
        return td  # 创建一个 Timedelta 对象并返回

    def time_timedelta_days(self, td):
        td.days  # 获取 Timedelta 对象的天数属性

    def time_timedelta_seconds(self, td):
        td.seconds  # 获取 Timedelta 对象的秒数属性

    def time_timedelta_microseconds(self, td):
        td.microseconds  # 获取 Timedelta 对象的微秒数属性

    def time_timedelta_nanoseconds(self, td):
        td.nanoseconds  # 获取 Timedelta 对象的纳秒数属性
```