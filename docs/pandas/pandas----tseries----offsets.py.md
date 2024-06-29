# `D:\src\scipysrc\pandas\pandas\tseries\offsets.py`

```
# 导入未来的注释功能模块，使得在类型注解中可以引用当前定义的类
from __future__ import annotations

# 从pandas库的时间序列偏移模块中导入多个类和函数
from pandas._libs.tslibs.offsets import (
    FY5253,                 # 财年偏移
    BaseOffset,             # 基本偏移类
    BDay,                   # 工作日偏移
    BMonthBegin,            # 月初工作日偏移
    BMonthEnd,              # 月末工作日偏移
    BQuarterBegin,          # 季初工作日偏移
    BQuarterEnd,            # 季末工作日偏移
    BusinessDay,            # 工作日偏移（同BDay）
    BusinessHour,           # 工作小时偏移
    BusinessMonthBegin,     # 工作月初偏移
    BusinessMonthEnd,       # 工作月末偏移
    BYearBegin,             # 年初工作日偏移
    BYearEnd,               # 年末工作日偏移
    CBMonthBegin,           # 自定义工作月初偏移
    CBMonthEnd,             # 自定义工作月末偏移
    CDay,                   # 自定义工作日偏移
    CustomBusinessDay,      # 自定义工作日偏移
    CustomBusinessHour,     # 自定义工作小时偏移
    CustomBusinessMonthBegin,  # 自定义工作月初偏移
    CustomBusinessMonthEnd,    # 自定义工作月末偏移
    DateOffset,             # 日期偏移
    Day,                    # 日偏移
    Easter,                 # 复活节偏移
    FY5253Quarter,          # 财年季度偏移
    Hour,                   # 小时偏移
    LastWeekOfMonth,        # 月最后一周偏移
    Micro,                  # 微秒偏移
    Milli,                  # 毫秒偏移
    Minute,                 # 分钟偏移
    MonthBegin,             # 月初偏移
    MonthEnd,               # 月末偏移
    Nano,                   # 纳秒偏移
    QuarterBegin,           # 季初偏移
    QuarterEnd,             # 季末偏移
    Second,                 # 秒偏移
    SemiMonthBegin,         # 半月初偏移
    SemiMonthEnd,           # 半月末偏移
    Tick,                   # tick偏移
    Week,                   # 周偏移
    WeekOfMonth,            # 周内偏移
    YearBegin,              # 年初偏移
    YearEnd,                # 年末偏移
)

# 声明所有需要导出的类和函数的名称列表，以便在模块外部使用
__all__ = [
    "Day",
    "BaseOffset",
    "BusinessDay",
    "BusinessMonthBegin",
    "BusinessMonthEnd",
    "BDay",
    "CustomBusinessDay",
    "CustomBusinessMonthBegin",
    "CustomBusinessMonthEnd",
    "CDay",
    "CBMonthEnd",
    "CBMonthBegin",
    "MonthBegin",
    "BMonthBegin",
    "MonthEnd",
    "BMonthEnd",
    "SemiMonthEnd",
    "SemiMonthBegin",
    "BusinessHour",
    "CustomBusinessHour",
    "YearBegin",
    "BYearBegin",
    "YearEnd",
    "BYearEnd",
    "QuarterBegin",
    "BQuarterBegin",
    "QuarterEnd",
    "BQuarterEnd",
    "LastWeekOfMonth",
    "FY5253Quarter",
    "FY5253",
    "Week",
    "WeekOfMonth",
    "Easter",
    "Tick",
    "Hour",
    "Minute",
    "Second",
    "Milli",
    "Micro",
    "Nano",
    "DateOffset",
]
```