# `D:\src\scipysrc\pandas\asv_bench\benchmarks\tslibs\offsets.py`

```
"""
offsets benchmarks that rely only on tslibs.  See benchmarks.offset for
offsets benchmarks that rely on other parts of pandas.
"""

# 导入 datetime 模块，用于处理日期时间
from datetime import datetime

# 导入 numpy 模块，并使用 np 作为别名
import numpy as np

# 从 pandas 中导入 offsets 模块
from pandas import offsets

# 尝试导入 pandas.tseries.holiday 模块，如果导入失败则忽略
try:
    import pandas.tseries.holiday
except ImportError:
    pass

# 创建一个美国联邦节假日日历对象
hcal = pandas.tseries.holiday.USFederalHolidayCalendar()

# 这些偏移量目前在调用 .apply_index() 时会引发 NotImplementedError
non_apply = [
    offsets.Day(),  # 每天的偏移量
    offsets.BYearEnd(),  # 业务年度结束的偏移量
    offsets.BYearBegin(),  # 业务年度开始的偏移量
    offsets.BQuarterEnd(),  # 业务季度结束的偏移量
    offsets.BQuarterBegin(),  # 业务季度开始的偏移量
    offsets.BMonthEnd(),  # 业务月结束的偏移量
    offsets.BMonthBegin(),  # 业务月开始的偏移量
    offsets.CustomBusinessDay(),  # 自定义工作日偏移量
    offsets.CustomBusinessDay(calendar=hcal),  # 自定义工作日偏移量，指定使用 hcal 日历
    offsets.CustomBusinessMonthBegin(calendar=hcal),  # 自定义业务月开始偏移量，指定使用 hcal 日历
    offsets.CustomBusinessMonthEnd(calendar=hcal),  # 自定义业务月结束偏移量，指定使用 hcal 日历
]

# 其他类型的偏移量
other_offsets = [
    offsets.YearEnd(),  # 年度结束的偏移量
    offsets.YearBegin(),  # 年度开始的偏移量
    offsets.QuarterEnd(),  # 季度结束的偏移量
    offsets.QuarterBegin(),  # 季度开始的偏移量
    offsets.MonthEnd(),  # 月结束的偏移量
    offsets.MonthBegin(),  # 月开始的偏移量
    offsets.DateOffset(months=2, days=2),  # 自定义日期偏移量，2个月2天
    offsets.BusinessDay(),  # 工作日偏移量
    offsets.SemiMonthEnd(),  # 半月结束的偏移量
    offsets.SemiMonthBegin(),  # 半月开始的偏移量
]

# 组合所有的偏移量对象
offset_objs = non_apply + other_offsets


# 定义一个类 OnOffset
class OnOffset:
    # 参数为偏移量对象列表
    params = offset_objs
    # 参数名称为 "offset"
    param_names = ["offset"]

    # 初始化方法，设置日期列表 self.dates
    def setup(self, offset):
        # 使用列表推导式生成日期列表，不包括11月31日
        self.dates = [
            datetime(2016, m, d)
            for m in [10, 11, 12]
            for d in [1, 2, 3, 28, 29, 30, 31]
            if not (m == 11 and d == 31)
        ]

    # 计时方法，检查日期是否在偏移量上
    def time_on_offset(self, offset):
        for date in self.dates:
            offset.is_on_offset(date)


# 定义一个类 OffestDatetimeArithmetic
class OffestDatetimeArithmetic:
    # 参数为偏移量对象列表
    params = offset_objs
    # 参数名称为 "offset"
    param_names = ["offset"]

    # 初始化方法，设置日期和 np.datetime64 对象
    def setup(self, offset):
        self.date = datetime(2011, 1, 1)
        self.dt64 = np.datetime64("2011-01-01 09:00Z")

    # 计时方法，使用 np.datetime64 对象进行日期偏移操作
    def time_add_np_dt64(self, offset):
        offset + self.dt64

    # 计时方法，使用 datetime 对象进行日期偏移操作
    def time_add(self, offset):
        self.date + offset

    # 计时方法，使用 datetime 对象进行10倍偏移操作
    def time_add_10(self, offset):
        self.date + (10 * offset)

    # 计时方法，使用 datetime 对象进行日期减法操作
    def time_subtract(self, offset):
        self.date - offset

    # 计时方法，使用 datetime 对象进行10倍日期减法操作
    def time_subtract_10(self, offset):
        self.date - (10 * offset)
```