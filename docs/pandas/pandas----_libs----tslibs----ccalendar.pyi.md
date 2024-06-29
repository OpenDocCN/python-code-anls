# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\ccalendar.pyi`

```
# 一周中的每一天，作为字符串列表
DAYS: list[str]

# 月份别名的映射，将整数月份映射到对应的字符串名称
MONTH_ALIASES: dict[int, str]

# 字符串月份到整数月份的映射
MONTH_NUMBERS: dict[str, int]

# 月份名称列表
MONTHS: list[str]

# 整数表示的星期几到字符串的映射
int_to_weekday: dict[int, str]

# 给定年份和月份，返回该月份的第一个工作日的日期
def get_firstbday(year: int, month: int) -> int:
    ...

# 给定年份和月份，返回该月份的最后一个工作日的日期
def get_lastbday(year: int, month: int) -> int:
    ...

# 给定年份、月份和日期，返回该日期在年份中的第几天（从1开始计数）
def get_day_of_year(year: int, month: int, day: int) -> int:
    ...

# 给定年份、月份和日期，返回ISO 8601日历格式的元组（年、周数、周几）
def get_iso_calendar(year: int, month: int, day: int) -> tuple[int, int, int]:
    ...

# 给定年份、月份和日期，返回该日期所在年份的第几周（ISO 8601标准）
def get_week_of_year(year: int, month: int, day: int) -> int:
    ...

# 给定年份和月份，返回该月份有多少天
def get_days_in_month(year: int, month: int) -> int:
    ...
```