# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\ccalendar.pxd`

```
# 导入 Cython 中的 Py_ssize_t 类型
from cython cimport Py_ssize_t
# 导入 Cython 中的 int32_t 和 int64_t 类型
from numpy cimport (
    int32_t,
    int64_t,
)

# 定义一个类型别名 iso_calendar_t，表示一个由三个 int32_t 组成的元组
ctypedef (int32_t, int32_t, int32_t) iso_calendar_t

# 声明一个 C 函数 dayofweek，计算给定日期的星期几，不抛出异常，不使用全局解锁
cdef int dayofweek(int y, int m, int d) noexcept nogil
# 声明一个 C 函数 is_leapyear，判断给定年份是否是闰年，不抛出异常，不使用全局解锁
cdef bint is_leapyear(int64_t year) noexcept nogil
# 声明一个 Cython 可调用函数 get_days_in_month，返回指定年份和月份的天数，不抛出异常，不使用全局解锁
cpdef int32_t get_days_in_month(int year, Py_ssize_t month) noexcept nogil
# 声明一个 Cython 可调用函数 get_week_of_year，返回指定日期的年份周数，不抛出异常，不使用全局解锁
cpdef int32_t get_week_of_year(int year, int month, int day) noexcept nogil
# 声明一个 Cython 可调用函数 get_iso_calendar，返回 ISO 格式的日历元组，不抛出异常，不使用全局解锁
cpdef iso_calendar_t get_iso_calendar(int year, int month, int day) noexcept nogil
# 声明一个 Cython 可调用函数 get_day_of_year，返回指定日期在年份中的第几天，不抛出异常，不使用全局解锁
cpdef int32_t get_day_of_year(int year, int month, int day) noexcept nogil
# 声明一个 Cython 可调用函数 get_lastbday，返回指定年份和月份的最后一个工作日，不抛出异常，不使用全局解锁
cpdef int get_lastbday(int year, int month) noexcept nogil
# 声明一个 Cython 可调用函数 get_firstbday，返回指定年份和月份的第一个工作日，不抛出异常，不使用全局解锁
cpdef int get_firstbday(int year, int month) noexcept nogil

# 定义两个 C 字典，用于月份数字和对应的日历数字之间的映射关系
cdef dict c_MONTH_NUMBERS, MONTH_TO_CAL_NUM

# 声明一个指向 int32_t 的指针 month_offset
cdef int32_t* month_offset
```