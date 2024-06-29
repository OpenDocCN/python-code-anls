# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\ccalendar.pyx`

```
# 使用 Cython 语法声明 boundscheck=False，以提高性能
# """
# Cython implementations of functions resembling the stdlib calendar module
# """
cimport cython
# 导入 numpy 库中的 int32_t 和 int64_t 类型
from numpy cimport (
    int32_t,
    int64_t,
)

# ----------------------------------------------------------------------
# 常量定义

# 通过数组定义每个月份的天数，前12个为非闰年的月份天数，后12个为闰年的月份天数
cdef int32_t* days_per_month_array = [
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# 定义每个月份起始日期到年初的天数偏移量数组，前13个为非闰年，后13个为闰年
cdef int* em = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

# 定义月份偏移量数组，前13个为非闰年，后13个为闰年
cdef int32_t* month_offset = [
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
    0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

# 定义全局常量，存储月份的缩写和全名
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
          "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTHS_FULL = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]
# 创建字典，将月份缩写映射为其对应的数字
MONTH_NUMBERS = {name: num for num, name in enumerate(MONTHS)}
cdef dict c_MONTH_NUMBERS = MONTH_NUMBERS
# 创建字典，将月份数字映射为其缩写的别名
MONTH_ALIASES = {(num + 1): name for num, name in enumerate(MONTHS)}
# 创建字典，将月份缩写映射为其对应的日历数字
cdef dict MONTH_TO_CAL_NUM = {name: num + 1 for num, name in enumerate(MONTHS)}

# 定义星期的名称和全名列表
DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DAYS_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
# 创建字典，将星期数字映射为其缩写的别名
int_to_weekday = {num: name for num, name in enumerate(DAYS)}
# 创建字典，将星期缩写映射为其对应的数字
weekday_to_int = {int_to_weekday[key]: key for key in int_to_weekday}


# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义函数，返回给定年份和月份的天数
cpdef int32_t get_days_in_month(int year, Py_ssize_t month) noexcept nogil:
    """
    Return the number of days in the given month of the given year.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    days_in_month : int

    Notes
    -----
    Assumes that the arguments are valid.  Passing a month not between 1 and 12
    risks a segfault.
    """
    return days_per_month_array[12 * is_leapyear(year) + month - 1]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义整数除法的函数
cdef long quot(long a , long b) noexcept nogil:
    cdef long x
    x = a/b
    if (a < 0):
        x -= (a % b != 0)
    return x


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义计算给定日期是星期几的函数
cdef int dayofweek(int y, int m, int d) noexcept nogil:
    """
    Calculate the day of the week for a given date.

    Parameters
    ----------
    y : int
        Year.
    m : int
        Month.
    d : int
        Day.

    Returns
    -------
    int
        Day of the week (0-6, Monday to Sunday).
    """
    # 实现内容在此省略
    Find the day of week for the date described by the Y/M/D triple y, m, d
    using Gauss' method, from wikipedia.

    0 represents Monday.  See [1]_.

    Parameters
    ----------
    y : int
        Year of the date (input parameter).
    m : int
        Month of the date (input parameter).
    d : int
        Day of the date (input parameter).

    Returns
    -------
    weekday : int
        An integer representing the day of the week (0 for Monday, 1 for Tuesday, etc.).

    Notes
    -----
    Assumes that y, m, d, represents a valid date.

    See Also
    --------
    [1] https://docs.python.org/3/library/calendar.html#calendar.weekday

    [2] https://en.wikipedia.org/wiki/\
    Determination_of_the_day_of_the_week#Gauss's_algorithm
    """
    # Note: this particular implementation comes from
    # http://berndt-schwerdtfeger.de/wp-content/uploads/pdf/cal.pdf

    # Define variables for use in the algorithm
    cdef:
        long c   # Century part of the year
        int g    # Year within the century
        int f    # Calculation factor based on the century
        int e    # Month adjustment factor

    # Adjust the year and month if the month is January or February
    if (m < 3):
        y -= 1

    # Compute values based on Gauss' algorithm
    c = quot(y, 100)         # Century part of the year
    g = y - c * 100          # Year within the century
    f = 5 * (c - quot(c, 4) * 4)  # Calculation factor based on the century
    e = em[m]                # Month adjustment factor from a predefined array

    if (m > 2):
        e -= 1

    # Return the day of the week using Gauss' formula
    return (-1 + d + e + f + g + g/4) % 7
cdef bint is_leapyear(int64_t year) noexcept nogil:
    """
    Returns 1 if the given year is a leap year, 0 otherwise.

    Parameters
    ----------
    year : int

    Returns
    -------
    is_leap : bool
    """
    return ((year & 0x3) == 0 and  # 检查 year 是否为 4 的倍数
            ((year % 100) != 0 or (year % 400) == 0))


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_week_of_year(int year, int month, int day) noexcept nogil:
    """
    Return the ordinal week-of-year for the given day.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    week_of_year : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    return get_iso_calendar(year, month, day)[1]


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef iso_calendar_t get_iso_calendar(int year, int month, int day) noexcept nogil:
    """
    Return the year, week, and day of year corresponding to ISO 8601

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    year : int32_t
    week : int32_t
    day : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    cdef:
        int32_t doy, dow
        int32_t iso_year, iso_week

    doy = get_day_of_year(year, month, day)
    dow = dayofweek(year, month, day)

    # 估算 ISO 周数
    iso_week = (doy - 1) - dow + 3
    if iso_week >= 0:
        iso_week = iso_week // 7 + 1

    # 验证 ISO 周数的正确性
    if iso_week < 0:
        if (iso_week > -2) or (iso_week == -2 and is_leapyear(year - 1)):
            iso_week = 53
        else:
            iso_week = 52
    elif iso_week == 53:
        if 31 - day + dow < 3:
            iso_week = 1

    iso_year = year
    if iso_week == 1 and month == 12:
        iso_year += 1

    elif iso_week >= 52 and month == 1:
        iso_year -= 1

    return iso_year, iso_week, dow + 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_day_of_year(int year, int month, int day) noexcept nogil:
    """
    Return the ordinal day-of-year for the given day.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    day_of_year : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    cdef:
        bint isleap
        int32_t mo_off
        int day_of_year

    isleap = is_leapyear(year)

    # 计算当前月份的偏移量
    mo_off = month_offset[isleap * 13 + month - 1]

    day_of_year = mo_off + day
    return day_of_year


# ---------------------------------------------------------------------
# Business Helpers

cpdef int get_lastbday(int year, int month) noexcept nogil:
    """
    Find the last day of the month that is a business day.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    last_bday : int
    """
    cdef:
        int wkday, days_in_month

    wkday = dayofweek(year, month, 1)
    # 调用函数获取指定年份和月份的天数
    days_in_month = get_days_in_month(year, month)
    # 计算给定月份最后一个星期五的日期
    # 计算方式为：从该月最后一天往前推，找到最接近的星期五
    return days_in_month - max(((wkday + days_in_month - 1) % 7) - 4, 0)
# 定义一个 Cython 函数，获取指定年份和月份的第一个工作日的日期

cpdef int get_firstbday(int year, int month) noexcept nogil:
    """
    Find the first day of the month that is a business day.

    Parameters
    ----------
    year : int
        年份参数，用于指定需要查询的年份
    month : int
        月份参数，用于指定需要查询的月份

    Returns
    -------
    first_bday : int
        第一个工作日的日期

    Notes
    -----
    This function uses Cython types (`cpdef`, `cdef`) for performance optimization.
    It calls `dayofweek` to determine the weekday of the first day of the given month,
    then adjusts the return value based on whether the first day is a Saturday or Sunday.
    """
    cdef:
        int first, wkday  # 定义两个整数变量，first 用于存储第一个工作日的日期，wkday 用于存储月份第一天的星期几编号

    # 调用 dayofweek 函数获取指定年份和月份第一天的星期几编号
    wkday = dayofweek(year, month, 1)
    first = 1  # 默认第一个工作日为月份的第一天

    # 根据月份第一天的星期几编号进行判断和调整
    if wkday == 5:  # 如果是星期六
        first = 3  # 第一个工作日是第三天
    elif wkday == 6:  # 如果是星期日
        first = 2  # 第一个工作日是第二天

    # 返回计算得到的第一个工作日的日期
    return first
```