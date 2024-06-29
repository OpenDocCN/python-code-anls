# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\fields.pyx`

```
"""
Functions for accessing attributes of Timestamp/datetime64/datetime-like
objects and arrays
"""
# 从 locale 模块导入 LC_TIME 常量
from locale import LC_TIME

# 从 _strptime 模块导入 LocaleTime 类
from _strptime import LocaleTime

# 导入 Cython 相关声明
cimport cython
from cython cimport Py_ssize_t

# 导入 NumPy 库
import numpy as np

# 使用 NumPy 的 C 接口
cimport numpy as cnp
from numpy cimport (
    int8_t,
    int32_t,
    int64_t,
    ndarray,
    uint32_t,
)

# 导入 NumPy 的 C 接口的数组方法
cnp.import_array()

# 从 pandas 库中导入设置地区信息的函数
from pandas._config.localization import set_locale

# 从 pandas 库的 tslibs.ccalendar 模块中导入相关常量和函数
from pandas._libs.tslibs.ccalendar import (
    DAYS_FULL,
    MONTHS_FULL,
)

# 从 pandas 库的 tslibs.ccalendar 模块的 C 接口中导入相关函数和结构体
from pandas._libs.tslibs.ccalendar cimport (
    dayofweek,
    get_day_of_year,
    get_days_in_month,
    get_firstbday,
    get_iso_calendar,
    get_lastbday,
    get_week_of_year,
    iso_calendar_t,
)

# 从 pandas 库的 tslibs.nattype 模块中导入 NPY_NAT 常量
from pandas._libs.tslibs.nattype cimport NPY_NAT

# 从 pandas 库的 tslibs.np_datetime 模块中导入相关类型和函数
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pandas_timedelta_to_timedeltastruct,
    pandas_timedeltastruct,
)

# 调用 import_pandas_datetime 函数以导入 pandas 的 datetime 相关函数

import_pandas_datetime()

# 定义一个 Cython 函数，禁用数组越界检查和负索引处理
@cython.wraparound(False)
@cython.boundscheck(False)
def build_field_sarray(const int64_t[:] dtindex, NPY_DATETIMEUNIT reso):
    """
    Datetime as int64 representation to a structured array of fields
    """
    # 声明变量和数据结构
    cdef:
        Py_ssize_t i, count = len(dtindex)
        npy_datetimestruct dts
        ndarray[int32_t] years, months, days, hours, minutes, seconds, mus

    # 结构化数组的 dtype
    sa_dtype = [
        ("Y", "i4"),  # year
        ("M", "i4"),  # month
        ("D", "i4"),  # day
        ("h", "i4"),  # hour
        ("m", "i4"),  # min
        ("s", "i4"),  # second
        ("u", "i4"),  # microsecond
    ]

    # 创建一个空的结构化数组
    out = np.empty(count, dtype=sa_dtype)

    # 分配结构化数组中的字段到对应的数组变量
    years = out["Y"]
    months = out["M"]
    days = out["D"]
    hours = out["h"]
    minutes = out["m"]
    seconds = out["s"]
    mus = out["u"]

    # 遍历日期索引数组，将日期转换为结构体并分配到对应字段数组中
    for i in range(count):
        pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
        years[i] = dts.year
        months[i] = dts.month
        days[i] = dts.day
        hours[i] = dts.hour
        minutes[i] = dts.min
        seconds[i] = dts.sec
        mus[i] = dts.us

    # 返回填充好的结构化数组
    return out


# 定义一个函数，检查月份在年度中的位置
def month_position_check(fields, weekdays) -> str | None:
    # 声明变量
    cdef:
        int32_t daysinmonth, y, m, d
        bint calendar_end = True
        bint business_end = True
        bint calendar_start = True
        bint business_start = True
        bint cal
        int32_t[:] years = fields["Y"]
        int32_t[:] months = fields["M"]
        int32_t[:] days = fields["D"]
    # 遍历给定的年份、月份、日期和星期几的列表
    for y, m, d, wd in zip(years, months, days, weekdays):
        # 如果是计算日历的起始日期，则检查当前日期是否为每月的第一天
        if calendar_start:
            calendar_start &= d == 1
        # 如果是计算工作日的起始日期，则检查当前日期是否为每月的第一天或者在前三天内并且是星期一
        if business_start:
            business_start &= d == 1 or (d <= 3 and wd == 0)

        # 如果已经开始计算日历结束或工作日结束日期
        if calendar_end or business_end:
            # 获取当前月份的总天数
            daysinmonth = get_days_in_month(y, m)
            # 判断当前日期是否为当月最后一天
            cal = d == daysinmonth
            # 如果是计算日历结束日期，则检查当前日期是否为当月最后一天
            if calendar_end:
                calendar_end &= cal
            # 如果是计算工作日结束日期，则检查当前日期是否为当月最后一天或者在最后三天内并且是星期五
            if business_end:
                business_end &= cal or (daysinmonth - d < 3 and wd == 4)
        # 如果既不是计算日历起始日期也不是计算工作日起始日期，则退出循环
        elif not calendar_start and not business_start:
            break

    # 返回相应的标识，指示日历结束、工作日结束、日历起始、工作日起始或者没有特定标识
    if calendar_end:
        return "ce"  # 日历结束
    elif business_end:
        return "be"  # 工作日结束
    elif calendar_start:
        return "cs"  # 日历起始
    elif business_start:
        return "bs"  # 工作日起始
    else:
        return None  # 没有特定标识
# 设置 Cython 的优化选项，禁用数组索引检查和负数索引包装
@cython.wraparound(False)
@cython.boundscheck(False)
def get_date_name_field(
    # 接受 int64 类型的 datetime 索引数组和一个字符串字段名
    const int64_t[:] dtindex,
    str field,
    # 可选参数：地区设置，默认为 None
    object locale=None,
    # 结果的时间分辨率，默认为纳秒
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    """
    给定基于 int64 的 datetime 索引数组，返回基于请求字段的日期名称字符串数组
    """
    cdef:
        # 循环计数器
        Py_ssize_t i
        # 索引数组的长度
        cnp.npy_intp count = dtindex.shape[0]
        # 输出的字符串数组，以及名称数组
        ndarray[object] out, names
        # datetime 结构体
        npy_datetimestruct dts
        # 星期几的索引
        int dow

    # 创建一个空的对象数组，用于存放输出结果
    out = cnp.PyArray_EMPTY(1, &count, cnp.NPY_OBJECT, 0)

    # 如果请求的字段是 "day_name"
    if field == "day_name":
        # 如果地区设置为 None，则使用全名星期数组
        if locale is None:
            names = np.array(DAYS_FULL, dtype=np.object_)
        else:
            # 否则，根据地区设置获取星期名称数组
            names = np.array(_get_locale_names("f_weekday", locale),
                             dtype=np.object_)
        # 遍历 datetime 索引数组
        for i in range(count):
            # 如果索引值为 NPY_NAT（Not a Time），将输出值设置为 NaN
            if dtindex[i] == NPY_NAT:
                out[i] = np.nan
                continue

            # 将 pandas 的 datetime 转换为 datetime 结构体
            pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
            # 获取日期对应的星期几索引
            dow = dayofweek(dts.year, dts.month, dts.day)
            # 将输出设为对应星期名称的首字母大写字符串
            out[i] = names[dow].capitalize()

    # 如果请求的字段是 "month_name"
    elif field == "month_name":
        # 如果地区设置为 None，则使用全名月份数组
        if locale is None:
            names = np.array(MONTHS_FULL, dtype=np.object_)
        else:
            # 否则，根据地区设置获取月份名称数组
            names = np.array(_get_locale_names("f_month", locale),
                             dtype=np.object_)
        # 遍历 datetime 索引数组
        for i in range(count):
            # 如果索引值为 NPY_NAT（Not a Time），将输出值设置为 NaN
            if dtindex[i] == NPY_NAT:
                out[i] = np.nan
                continue

            # 将 pandas 的 datetime 转换为 datetime 结构体
            pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
            # 将输出设为对应月份名称的首字母大写字符串
            out[i] = names[dts.month].capitalize()

    # 如果字段不是 "day_name" 或 "month_name"，抛出值错误
    else:
        raise ValueError(f"Field {field} not supported")

    # 返回结果数组
    return out


# 定义一个内部函数，检查日期是否处于给定月份
cdef bint _is_on_month(int month, int compare_month, int modby) noexcept nogil:
    """
    类似于 DateOffset.is_on_offset，检查日期是否处于给定月份的部分。
    """
    # 如果 modby 为 1，始终返回 True
    if modby == 1:
        return True
    # 如果 modby 为 3，检查月份是否是 compare_month 的倍数
    elif modby == 3:
        return (month - compare_month) % 3 == 0
    # 否则，检查月份是否与 compare_month 相等
    else:
        return month == compare_month


# 设置 Cython 的优化选项，禁用数组索引检查和负数索引包装
@cython.wraparound(False)
@cython.boundscheck(False)
def get_start_end_field(
    # 接受 int64 类型的 datetime 索引数组和一个字符串字段名
    const int64_t[:] dtindex,
    str field,
    # 频率名称字符串，可选，默认为 None
    str freq_name=None,
    # 月份关键字参数，默认为 12
    int month_kw=12,
    # 结果的时间分辨率，默认为纳秒
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    """
    给定基于 int64 的 datetime 索引数组，返回指示时间戳是否在月/季/年的开始/结束位置的布尔数组
    （根据给定的频率定义）。

    参数
    ----------
    dtindex : ndarray[int64]
    field : str
    freq_name : str 或 None，默认为 None
    month_kw : int，默认为 12
    reso : NPY_DATETIMEUNIT，默认为 NPY_FR_ns

    返回
    -------
    ndarray[bool]
    """
    cdef:
        # 循环计数器
        Py_ssize_t i
        # 索引数组的长度
        int count = dtindex.shape[0]
        # 是否为工作日的布尔标志
        bint is_business = 0
        # 年的结束月份，默认为 12
        int end_month = 12
        # 年的开始月份，默认为 1
        int start_month = 1
        # 输出的布尔数组
        ndarray[int8_t] out
        # datetime 结构体
        npy_datetimestruct dts
        # 比较的月份，以及用于比较的模数
        int compare_month, modby

    # 创建一个全零数组，用于存放输出结果
    out = np.zeros(count, dtype="int8")
    # 如果频率名称存在，则执行以下逻辑
    if freq_name:
        # 如果频率名称为"C"，则抛出值错误异常，指明不支持自定义工作日
        if freq_name == "C":
            raise ValueError(f"Custom business days is not supported by {field}")
        
        # 判断是否为工作日频率，以频率名称的首字符是否为"B"来判断
        is_business = freq_name[0] == "B"

        # 对于不同的频率名称进行处理
        # 如果频率名称以"B"开头，且后续两个字符为"QS"或"YS"
        if freq_name.lstrip("B")[0:2] in ["QS", "YS"]:
            # 根据月份关键字确定结束月份和起始月份
            end_month = 12 if month_kw == 1 else month_kw - 1
            start_month = month_kw

        else:
            # 否则，结束月份为指定的月份关键字，起始月份为结束月份的下一个月
            end_month = month_kw
            start_month = (end_month % 12) + 1

    else:
        # 如果频率名称不存在，默认结束月份为12，起始月份为1
        end_month = 12
        start_month = 1

    # 根据字段名称确定比较的月份，如果字段名称包含"start"，则使用起始月份，否则使用结束月份
    compare_month = start_month if "start" in field else end_month
    
    # 根据字段类型确定模数值
    if "month" in field:
        modby = 1
    elif "quarter" in field:
        modby = 3
    else:
        modby = 12

    # 根据字段类型进行不同的逻辑判断和处理
    if field in ["is_month_start", "is_quarter_start", "is_year_start"]:
        # 如果是工作日频率，依次处理每个索引值
        if is_business:
            for i in range(count):
                # 如果索引值为无效日期，则将输出数组对应位置置为0并继续下一次循环
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                # 将 Pandas 的日期时间转换为日期时间结构体
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                # 如果当前日期时间结构体的月份符合比较月份条件，并且是该月份的第一个工作日
                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_firstbday(dts.year, dts.month)):
                    # 将输出数组对应位置置为1
                    out[i] = 1

        else:
            # 如果不是工作日频率，依次处理每个索引值
            for i in range(count):
                # 如果索引值为无效日期，则将输出数组对应位置置为0并继续下一次循环
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                # 将 Pandas 的日期时间转换为日期时间结构体
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                # 如果当前日期时间结构体的月份符合比较月份条件，并且是该月份的第一天
                if _is_on_month(dts.month, compare_month, modby) and dts.day == 1:
                    # 将输出数组对应位置置为1
                    out[i] = 1

    elif field in ["is_month_end", "is_quarter_end", "is_year_end"]:
        # 如果是工作日频率，依次处理每个索引值
        if is_business:
            for i in range(count):
                # 如果索引值为无效日期，则将输出数组对应位置置为0并继续下一次循环
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                # 将 Pandas 的日期时间转换为日期时间结构体
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                # 如果当前日期时间结构体的月份符合比较月份条件，并且是该月份的最后一个工作日
                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_lastbday(dts.year, dts.month)):
                    # 将输出数组对应位置置为1
                    out[i] = 1

        else:
            # 如果不是工作日频率，依次处理每个索引值
            for i in range(count):
                # 如果索引值为无效日期，则将输出数组对应位置置为0并继续下一次循环
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                # 将 Pandas 的日期时间转换为日期时间结构体
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)

                # 如果当前日期时间结构体的月份符合比较月份条件，并且是该月份的最后一天
                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_days_in_month(dts.year, dts.month)):
                    # 将输出数组对应位置置为1
                    out[i] = 1

    else:
        # 如果字段名称不支持，则抛出值错误异常
        raise ValueError(f"Field {field} not supported")

    # 返回输出数组的布尔视图
    return out.view(bool)
# 禁用 Cython 的边界检查和索引包装，以优化性能
@cython.wraparound(False)
@cython.boundscheck(False)
def get_date_field(
    const int64_t[:] dtindex,  # 接受一个 int64 类型的不可变数组作为日期时间索引
    str field,  # 表示要提取的日期时间字段的字符串标识符
    NPY_DATETIMEUNIT reso=NPY_FR_ns,  # 指定日期时间的分辨率，默认为纳秒
):
    """
    Given a int64-based datetime index, extract the year, month, etc.,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)  # 定义循环计数器和数组长度
        ndarray[int32_t] out  # 定义一个 int32 类型的 NumPy 数组来存储结果
        npy_datetimestruct dts  # 定义一个 npy_datetimestruct 结构体变量，用于存储日期时间信息

    out = np.empty(count, dtype="i4")  # 使用 i4 类型（32 位整数）初始化输出数组

    if field == "Y":  # 如果要提取年份
        with nogil:  # 进入无全局解释器锁定（GIL）的上下文环境
            for i in range(count):  # 遍历日期时间索引数组
                if dtindex[i] == NPY_NAT:  # 如果索引值为 NaT（非法的日期时间）
                    out[i] = -1  # 将结果数组中对应位置设为 -1
                    continue  # 继续下一次循环

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)  # 调用函数将 Pandas 的日期时间转换为结构体
                out[i] = dts.year  # 将结构体中的年份赋值给结果数组
        return out  # 返回结果数组

    elif field == "M":  # 如果要提取月份，以下各分支逻辑类似
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.month
        return out

    elif field == "D":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.day
        return out

    elif field == "h":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.hour
                # TODO: can we de-dup with period.pyx <accessor>s?
        return out

    elif field == "m":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.min
        return out

    elif field == "s":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.sec
        return out

    elif field == "us":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.us
        return out

    elif field == "ns":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dts.ps // 1000
        return out
    # 如果字段为 "doy"，计算每个日期的一年中的第几天并存储在输出数组中
    elif field == "doy":
        # 使用 nogil 上下文以释放全局解释器锁，提高性能
        with nogil:
            # 遍历日期索引数组
            for i in range(count):
                # 如果日期索引为 NPY_NAT，表示日期不可用，设置输出为 -1 并继续下一轮循环
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                # 将 Pandas 的日期时间转换为 C 语言的 datetimestruct 结构体
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                # 获取日期的一年中的第几天并存储在输出数组中
                out[i] = get_day_of_year(dts.year, dts.month, dts.day)
        # 返回结果数组
        return out

    # 如果字段为 "dow"，计算每个日期的星期几并存储在输出数组中
    elif field == "dow":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = dayofweek(dts.year, dts.month, dts.day)
        return out

    # 如果字段为 "woy"，计算每个日期的一年中的第几周并存储在输出数组中
    elif field == "woy":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = get_week_of_year(dts.year, dts.month, dts.day)
        return out

    # 如果字段为 "q"，计算每个日期的季度并存储在输出数组中
    elif field == "q":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                # 获取日期的月份并转换为季度数，存储在输出数组中
                out[i] = dts.month
                out[i] = ((out[i] - 1) // 3) + 1
        return out

    # 如果字段为 "dim"，获取每个日期对应月份的天数并存储在输出数组中
    elif field == "dim":
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                out[i] = get_days_in_month(dts.year, dts.month)
        return out

    # 如果字段为 "is_leap_year"，返回每个日期是否为闰年的布尔数组
    elif field == "is_leap_year":
        # 调用函数获取日期数组中每个日期对应的年份，并检查是否为闰年
        return isleapyear_arr(get_date_field(dtindex, "Y", reso=reso))

    # 如果字段不支持，抛出 ValueError 异常
    raise ValueError(f"Field {field} not supported")
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数，接收 int64_t 类型的 timedelta 索引数组 tdindex，一个字符串字段 field，
# 以及一个默认为 NPY_FR_ns 的 NPY_DATETIMEUNIT 类型参数 reso，返回一个包含提取的时间差值字段的数组。
def get_timedelta_field(
    const int64_t[:] tdindex,  # 接收一个不可变的 int64_t 数组 tdindex
    str field,  # 字符串类型的字段名称
    NPY_DATETIMEUNIT reso=NPY_FR_ns,  # 默认为 NPY_FR_ns 的日期时间单位参数 reso
):
    """
    Given a int64-based timedelta index, extract the days, hrs, sec.,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(tdindex)  # 使用 Cython 定义变量 i 和 count，count 为 tdindex 的长度
        ndarray[int32_t] out  # 定义一个 int32_t 类型的 NumPy 数组 out
        pandas_timedeltastruct tds  # 定义一个 pandas_timedeltastruct 结构体变量 tds

    out = np.empty(count, dtype="i4")  # 使用 NumPy 创建一个长度为 count 的 int32 类型数组 out

    if field == "seconds":  # 如果字段为 "seconds"
        with nogil:  # 使用 nogil 上下文进行 GIL 释放的并行操作
            for i in range(count):  # 遍历 count 次数
                if tdindex[i] == NPY_NAT:  # 如果 tdindex[i] 等于 NPY_NAT
                    out[i] = -1  # 将 out[i] 设置为 -1 并继续下一个循环
                    continue

                pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)  # 调用函数将 tdindex 转换为 timedeltastruct 结构体
                out[i] = tds.seconds  # 将 tds 的秒字段赋值给 out[i]
        return out  # 返回数组 out

    elif field == "microseconds":  # 如果字段为 "microseconds"
        with nogil:  # 使用 nogil 上下文进行 GIL 释放的并行操作
            for i in range(count):  # 遍历 count 次数
                if tdindex[i] == NPY_NAT:  # 如果 tdindex[i] 等于 NPY_NAT
                    out[i] = -1  # 将 out[i] 设置为 -1 并继续下一个循环
                    continue

                pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)  # 调用函数将 tdindex 转换为 timedeltastruct 结构体
                out[i] = tds.microseconds  # 将 tds 的微秒字段赋值给 out[i]
        return out  # 返回数组 out

    elif field == "nanoseconds":  # 如果字段为 "nanoseconds"
        with nogil:  # 使用 nogil 上下文进行 GIL 释放的并行操作
            for i in range(count):  # 遍历 count 次数
                if tdindex[i] == NPY_NAT:  # 如果 tdindex[i] 等于 NPY_NAT
                    out[i] = -1  # 将 out[i] 设置为 -1 并继续下一个循环
                    continue

                pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)  # 调用函数将 tdindex 转换为 timedeltastruct 结构体
                out[i] = tds.nanoseconds  # 将 tds 的纳秒字段赋值给 out[i]
        return out  # 返回数组 out

    raise ValueError(f"Field {field} not supported")  # 如果字段不在 "seconds", "microseconds", "nanoseconds" 中则抛出 ValueError 异常


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数，接收 int64_t 类型的 timedelta 索引数组 tdindex，以及一个默认为 NPY_FR_ns 的 NPY_DATETIMEUNIT 类型参数 reso，
# 返回一个包含提取的天数字段的数组。
def get_timedelta_days(
    const int64_t[:] tdindex,  # 接收一个不可变的 int64_t 数组 tdindex
    NPY_DATETIMEUNIT reso=NPY_FR_ns,  # 默认为 NPY_FR_ns 的日期时间单位参数 reso
):
    """
    Given a int64-based timedelta index, extract the days,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(tdindex)  # 使用 Cython 定义变量 i 和 count，count 为 tdindex 的长度
        ndarray[int64_t] out  # 定义一个 int64_t 类型的 NumPy 数组 out
        pandas_timedeltastruct tds  # 定义一个 pandas_timedeltastruct 结构体变量 tds

    out = np.empty(count, dtype="i8")  # 使用 NumPy 创建一个长度为 count 的 int64 类型数组 out

    with nogil:  # 使用 nogil 上下文进行 GIL 释放的并行操作
        for i in range(count):  # 遍历 count 次数
            if tdindex[i] == NPY_NAT:  # 如果 tdindex[i] 等于 NPY_NAT
                out[i] = -1  # 将 out[i] 设置为 -1 并继续下一个循环
                continue

            pandas_timedelta_to_timedeltastruct(tdindex[i], reso, &tds)  # 调用函数将 tdindex 转换为 timedeltastruct 结构体
            out[i] = tds.days  # 将 tds 的天数字段赋值给 out[i]
    return out  # 返回数组 out


# 定义一个函数，接收一个 NumPy 数组 years，返回一个布尔类型的数组，表示每个年份是否是闰年。
cpdef isleapyear_arr(ndarray years):
    """vectorized version of isleapyear; NaT evaluates as False"""
    cdef:
        ndarray[int8_t] out  # 定义一个 int8_t 类型的 NumPy 数组 out

    out = np.zeros(len(years), dtype="int8")  # 使用 NumPy 创建一个长度为 years 的 int8 类型数组，初始化为 0
    out[np.logical_or(years % 400 == 0,
                      np.logical_and(years % 4 == 0,
                                     years % 100 > 0))] = 1  # 将是闰年的位置置为 1
    return out.view(bool)  # 返回转换为布尔类型的数组 out


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数，接收 int64_t 类型的 datetime 索引数组 dtindex，以及一个 NPY_DATETIMEUNIT 类型参数 reso，
# 返回一个包含 ISO 8601 年份、周数和日期的结构化数组。
def build_isocalendar_sarray(const int64_t[:] dtindex, NPY_DATETIMEUNIT reso):
    """
    Given a int64-based datetime array, return the ISO 8601 year, week, and day
    as a structured array.
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)  # 使用 Cython 定义变量 i 和 count，count 为 dtindex 的长度
        npy_datetimestruct dts  # 定义一个 npy_datetimestruct 结构体变量 dts
        ndarray[uint32_t] iso_years, iso_weeks, days  # 定义三个 uint32_t 类型的 NumPy 数组 iso_years, iso_weeks, days
        iso_calendar_t ret_val  # 定义一个 iso_calendar_t 类型的变量 ret_val
    # 定义结构化数组的数据类型，包含三个字段：year（年份，无符号整数），week（周数，无符号整数），day（天数，无符号整数）
    sa_dtype = [
        ("year", "u4"),
        ("week", "u4"),
        ("day", "u4"),
    ]

    # 创建一个空的结构化数组，用于存储结果，数组长度为 count，数据类型为 sa_dtype 定义的结构化类型
    out = np.empty(count, dtype=sa_dtype)

    # 分别取出结构化数组 out 中的年份、周数和天数字段，赋值给对应的变量
    iso_years = out["year"]
    iso_weeks = out["week"]
    days = out["day"]

    # 使用 nogil 上下文，进行 GIL（全局解释器锁）释放的并行计算
    with nogil:
        # 遍历 count 次，进行以下操作
        for i in range(count):
            # 如果 dtindex[i] 是 NPY_NAT（NumPy 中的 Not a Time 值）
            if dtindex[i] == NPY_NAT:
                # 设置返回值为 0, 0, 0
                ret_val = 0, 0, 0
            else:
                # 将 pandas 的日期时间对象转换为 C 语言风格的日期时间结构体，并存储在 dts 变量中
                pandas_datetime_to_datetimestruct(dtindex[i], reso, &dts)
                # 调用函数获取 ISO 日历的年、周、日信息，并将结果存储在 ret_val 变量中
                ret_val = get_iso_calendar(dts.year, dts.month, dts.day)

            # 将 ret_val 中的年份、周数、天数分别赋值给结构化数组 out 中的 iso_years[i]、iso_weeks[i] 和 days[i] 字段
            iso_years[i] = ret_val[0]
            iso_weeks[i] = ret_val[1]
            days[i] = ret_val[2]

    # 返回填充好数据的结构化数组 out
    return out
# 返回一个包含本地化的星期或月份名称数组
def _get_locale_names(name_type: str, locale: object = None):
    """
    Returns an array of localized day or month names.

    Parameters
    ----------
    name_type : str
        Attribute of LocaleTime() in which to return localized names.
    locale : str

    Returns
    -------
    list of locale names
    """
    # 设置指定的区域设置并返回对应的本地化名称数组
    with set_locale(locale, LC_TIME):
        return getattr(LocaleTime(), name_type)


# ---------------------------------------------------------------------
# Rounding


class RoundTo:
    """
    enumeration defining the available rounding modes

    Attributes
    ----------
    MINUS_INFTY
        round towards -∞, or floor [2]_
    PLUS_INFTY
        round towards +∞, or ceil [3]_
    NEAREST_HALF_EVEN
        round to nearest, tie-break half to even [6]_
    NEAREST_HALF_MINUS_INFTY
        round to nearest, tie-break half to -∞ [5]_
    NEAREST_HALF_PLUS_INFTY
        round to nearest, tie-break half to +∞ [4]_


    References
    ----------
    .. [1] "Rounding - Wikipedia"
           https://en.wikipedia.org/wiki/Rounding
    .. [2] "Rounding down"
           https://en.wikipedia.org/wiki/Rounding#Rounding_down
    .. [3] "Rounding up"
           https://en.wikipedia.org/wiki/Rounding#Rounding_up
    .. [4] "Round half up"
           https://en.wikipedia.org/wiki/Rounding#Round_half_up
    .. [5] "Round half down"
           https://en.wikipedia.org/wiki/Rounding#Round_half_down
    .. [6] "Round half to even"
           https://en.wikipedia.org/wiki/Rounding#Round_half_to_even
    """
    @property
    def MINUS_INFTY(self) -> int:
        return 0

    @property
    def PLUS_INFTY(self) -> int:
        return 1

    @property
    def NEAREST_HALF_EVEN(self) -> int:
        return 2

    @property
    def NEAREST_HALF_PLUS_INFTY(self) -> int:
        return 3

    @property
    def NEAREST_HALF_MINUS_INFTY(self) -> int:
        return 4


# 定义一个 C 扩展函数，将 int64_t 数组向下取整到指定单位
cdef ndarray[int64_t] _floor_int64(const int64_t[:] values, int64_t unit):
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]
            # 如果值为 NPY_NAT，结果也为 NPY_NAT
            if value == NPY_NAT:
                res = NPY_NAT
            else:
                # 向下取整到指定单位
                res = value - value % unit
            result[i] = res

    return result


# 定义一个 C 扩展函数，将 int64_t 数组向上取整到指定单位
cdef ndarray[int64_t] _ceil_int64(const int64_t[:] values, int64_t unit):
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value, remainder

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]

            # 如果值为 NPY_NAT，结果也为 NPY_NAT
            if value == NPY_NAT:
                res = NPY_NAT
            else:
                remainder = value % unit
                if remainder == 0:
                    res = value
                else:
                    # 向上取整到指定单位
                    res = value + (unit - remainder)

            result[i] = res
    返回函数内部变量 result 的值作为函数的返回结果
cdef ndarray[int64_t] _rounddown_int64(values, int64_t unit):
    # 定义函数 _rounddown_int64，接受一个整数数组 values 和一个整数单位 unit
    cdef:
        Py_ssize_t i, n = len(values)
        # 声明整数变量 i 和 n，分别表示 values 数组的长度
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        # 创建一个长度为 n 的 int64 类型的 NumPy 数组 result，用于存储结果
        int64_t res, value, remainder, half
        # 声明整数变量 res, value, remainder 和 half

    half = unit // 2
    # 计算 unit 的一半并赋值给 half

    with cython.overflowcheck(True):
        # 启用溢出检查
        for i in range(n):
            # 遍历 values 数组
            value = values[i]
            # 获取当前索引 i 处的值

            if value == NPY_NAT:
                # 如果 value 等于 NPY_NAT
                res = NPY_NAT
                # 直接将 res 设为 NPY_NAT
            else:
                # 否则
                # 进行向下取整的调整，是 rounddown_int64 和 _ceil_int64 唯一的区别
                value = value - half
                remainder = value % unit
                if remainder == 0:
                    res = value
                else:
                    res = value + (unit - remainder)

            result[i] = res
            # 将计算得到的 res 存储到结果数组 result 中的索引 i 处

    return result
    # 返回存储结果的数组


cdef ndarray[int64_t] _roundup_int64(values, int64_t unit):
    # 定义函数 _roundup_int64，接受一个整数数组 values 和一个整数单位 unit
    return _floor_int64(values + unit // 2, unit)
    # 调用 _floor_int64 函数，向上取整，并返回结果


cdef ndarray[int64_t] _round_nearest_int64(const int64_t[:] values, int64_t unit):
    # 定义函数 _round_nearest_int64，接受一个常量整数数组 values 和一个整数单位 unit
    cdef:
        Py_ssize_t i, n = len(values)
        # 声明整数变量 i 和 n，分别表示 values 数组的长度
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        # 创建一个长度为 n 的 int64 类型的 NumPy 数组 result，用于存储结果
        int64_t res, value, half, remainder, quotient
        # 声明整数变量 res, value, half, remainder 和 quotient

    half = unit // 2
    # 计算 unit 的一半并赋值给 half

    with cython.overflowcheck(True):
        # 启用溢出检查
        for i in range(n):
            # 遍历 values 数组
            value = values[i]
            # 获取当前索引 i 处的值

            if value == NPY_NAT:
                # 如果 value 等于 NPY_NAT
                res = NPY_NAT
                # 直接将 res 设为 NPY_NAT
            else:
                quotient, remainder = divmod(value, unit)
                if remainder > half:
                    res = value + (unit - remainder)
                elif remainder == half and quotient % 2:
                    res = value + (unit - remainder)
                else:
                    res = value - remainder

            result[i] = res
            # 将计算得到的 res 存储到结果数组 result 中的索引 i 处

    return result
    # 返回存储结果的数组


def round_nsint64(values: np.ndarray, mode: RoundTo, nanos: int) -> np.ndarray:
    """
    Applies rounding mode at given frequency

    Parameters
    ----------
    values : np.ndarray[int64_t]
        Input array of int64 values to be rounded
    mode : instance of `RoundTo` enumeration
        Rounding mode to apply (e.g., MINUS_INFTY, PLUS_INFTY, NEAREST_HALF_MINUS_INFTY)
    nanos : np.int64
        Frequency to round to, expressed in nanoseconds

    Returns
    -------
    np.ndarray[int64_t]
        Array of int64 values after applying the specified rounding mode
    """
    cdef:
        int64_t unit = nanos
        # 声明整数变量 unit，并将 nanos 赋值给它

    if mode == RoundTo.MINUS_INFTY:
        # 如果 mode 是 MINUS_INFTY
        return _floor_int64(values, unit)
        # 调用 _floor_int64 函数，向下取整，并返回结果
    elif mode == RoundTo.PLUS_INFTY:
        # 如果 mode 是 PLUS_INFTY
        return _ceil_int64(values, unit)
        # 调用 _ceil_int64 函数，向上取整，并返回结果
    elif mode == RoundTo.NEAREST_HALF_MINUS_INFTY:
        # 如果 mode 是 NEAREST_HALF_MINUS_INFTY
        return _rounddown_int64(values, unit)
        # 调用 _rounddown_int64 函数，向下取整，并返回结果
    elif mode == RoundTo.NEAREST_HALF_PLUS_INFTY:
        # 如果 mode 是 NEAREST_HALF_PLUS_INFTY
        return _roundup_int64(values, unit)
        # 调用 _roundup_int64 函数，向上取整，并返回结果
    elif mode == RoundTo.NEAREST_HALF_EVEN:
        # 如果 mode 是 NEAREST_HALF_EVEN
        # 对于奇数单位，不需要进行 tie break
        if unit % 2:
            return _rounddown_int64(values, unit)
        return _round_nearest_int64(values, unit)
        # 如果单位是偶数，则调用 _round_nearest_int64 函数进行最近舍入，并返回结果

    # 如果上述条件都不满足，说明是一个未识别的 rounding mode
    raise ValueError("round_nsint64 called with an unrecognized rounding mode")
    # 抛出 ValueError 异常，指示调用了未识别的 rounding mode
```