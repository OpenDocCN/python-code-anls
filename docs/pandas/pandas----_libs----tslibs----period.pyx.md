# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\period.pyx`

```
# 导入正则表达式模块
import re

# 导入 C 接口的 numpy 库，并使用别名 cnp
cimport numpy as cnp

# 从 CPython 的对象模块中导入指定的符号
from cpython.object cimport (
    Py_EQ,  # 等于比较的宏
    Py_NE,  # 不等于比较的宏
    PyObject,  # Python 对象结构体
    PyObject_RichCompare,  # 富比较函数
    PyObject_RichCompareBool,  # 富比较函数返回布尔值
)

# 从 C 接口的 numpy 模块中导入指定的符号
from numpy cimport (
    int32_t,  # 32 位整数类型
    int64_t,  # 64 位整数类型
    ndarray,  # NumPy 数组类型
)

# 导入 numpy 并使用别名 np
import numpy as np

# 调用 cnp 的 import_array() 函数
cnp.import_array()

# 导入 Cython 模块
cimport cython

# 从 CPython 的 datetime 模块中导入指定的符号
from cpython.datetime cimport (
    PyDate_Check,  # 检查是否为日期对象的宏
    PyDateTime_Check,  # 检查是否为日期时间对象的宏
    datetime,  # Python datetime 结构体
    import_datetime,  # 导入 datetime C API
)

# 从 C 标准库的 stdlib 中导入指定的符号
from libc.stdlib cimport (
    free,  # 释放内存的函数
    malloc,  # 分配内存的函数
)

# 从 C 标准库的 string 中导入指定的符号
from libc.string cimport (
    memset,  # 内存清零的函数
    strlen,  # 字符串长度的函数
)

# 从 C 标准库的 time 中导入指定的符号
from libc.time cimport (
    strftime,  # 格式化时间为字符串的函数
    tm,  # C 标准库的时间结构体
)

# 从 pandas._libs.tslibs.dtypes 模块中导入 c_OFFSET_TO_PERIOD_FREQSTR 符号
from pandas._libs.tslibs.dtypes cimport c_OFFSET_TO_PERIOD_FREQSTR

# 从 pandas._libs.tslibs.np_datetime 模块中导入 OutOfBoundsDatetime 异常类
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

# 导入 datetime C API
import_datetime()

# 从 pandas._libs.tslibs.util 模块中导入 util 符号
cimport pandas._libs.tslibs.util as util

# 从 pandas._libs.missing 模块中导入 C_NA 符号
from pandas._libs.missing cimport C_NA

# 从 pandas._libs.tslibs.np_datetime 模块中导入指定的符号
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,  # NumPy 日期时间单位
    NPY_FR_D,  # NumPy 中的 'D' 频率
    astype_overflowsafe,  # 安全转换为指定类型的函数
    dts_to_iso_string,  # 将 datetime64 转换为 ISO 格式字符串的函数
    import_pandas_datetime,  # 导入 pandas datetime C API
    npy_datetimestruct,  # NumPy 日期时间结构体
    npy_datetimestruct_to_datetime,  # 将 npy 日期时间结构体转换为 datetime 对象的函数
    pandas_datetime_to_datetimestruct,  # 将 pandas datetime 对象转换为 npy 日期时间结构体的函数
)

# 导入 pandas datetime C API
import_pandas_datetime()

# 从 pandas._libs.tslibs.timestamps 模块中导入 Timestamp 类
from pandas._libs.tslibs.timestamps import Timestamp

# 从 pandas._libs.tslibs.ccalendar 模块中导入指定的符号
from pandas._libs.tslibs.ccalendar cimport (
    dayofweek,  # 获取日期是星期几的函数
    get_day_of_year,  # 获取年中的第几天的函数
    get_days_in_month,  # 获取指定月份的天数的函数
    get_week_of_year,  # 获取年中的第几周的函数
    is_leapyear,  # 判断是否为闰年的函数
)

# 从 pandas._libs.tslibs.timedeltas 模块中导入指定的符号
from pandas._libs.tslibs.timedeltas cimport (
    delta_to_nanoseconds,  # 将 timedelta 转换为纳秒数的函数
    is_any_td_scalar,  # 检查是否为任意 timedelta 标量的函数
)

# 从 pandas._libs.tslibs.conversion 模块中导入 DT64NS_DTYPE 符号
from pandas._libs.tslibs.conversion import DT64NS_DTYPE

# 从 pandas._libs.tslibs.dtypes 模块中导入指定的符号
from pandas._libs.tslibs.dtypes cimport (
    FR_ANN,  # 年度频率码
    FR_BUS,  # 工作日频率码
    FR_DAY,  # 日频率码
    FR_HR,  # 小时频率码
    FR_MIN,  # 分钟频率码
    FR_MS,  # 毫秒频率码
    FR_MTH,  # 月频率码
    FR_NS,  # 纳秒频率码
    FR_QTR,  # 季度频率码
    FR_SEC,  # 秒频率码
    FR_UND,  # 未知频率码
    FR_US,  # 微秒频率码
    FR_WK,  # 周频率码
    PeriodDtypeBase,  # PeriodDtype 基类
    attrname_to_abbrevs,  # 属性名称到缩写的映射字典
    freq_group_code_to_npy_unit,  # 频率组代码到 NumPy 单位的映射字典
)

# 从 pandas._libs.tslibs.parsing 模块中导入 quarter_to_myear 函数
from pandas._libs.tslibs.parsing cimport quarter_to_myear

# 从 pandas._libs.tslibs.parsing 模块中导入 parse_datetime_string_with_reso 函数
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso

# 从 pandas._libs.tslibs.nattype 模块中导入指定的符号
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,  # NumPy 中的 NaT 值
    c_NaT as NaT,  # NaT 的 C 别名
    c_nat_strings as nat_strings,  # NaT 字符串的集合
    checknull_with_nat,  # 检查空值并处理 NaT 的函数
)

# 从 pandas._libs.tslibs.offsets 模块中导入指定的符号
from pandas._libs.tslibs.offsets cimport (
    BaseOffset,  # 基础偏移量类
    is_offset_object,  # 检查是否为偏移量对象的函数
    to_offset,  # 转换为偏移量对象的函数
)

# 从 pandas._libs.tslibs.offsets 模块中导入 BDay 符号
from pandas._libs.tslibs.offsets import (
    INVALID_FREQ_ERR_MSG,  # 无效频率错误消息
    BDay,  # 工作日偏移量
)

# 定义一个枚举类型 INT32_MIN，并赋值为 -2_147_483_648LL
cdef:
    enum:
        INT32_MIN = -2_147_483_648LL

# 定义一个 C 结构体 asfreq_info，包含四个整型成员变量
ctypedef struct asfreq_info:
    int64_t intraday_conversion_factor  # 一天内的转换因子
    int is_end  # 结束标志
    int to_end  # 转向结束标志
    int from_end  # 从结束标志

# 定义一个指向 freq_conv_func 函数指针类型的别名，接受两个参数并返回 int64_t 类型
ctypedef int64_t (*freq_conv_func)(int64_t, asfreq_info*) noexcept nogil

# 从外部源导入 C 代码
    # 定义一个静态的二维数组，表示不同时间单位之间的转换因子矩阵
    static npy_int64 daytime_conversion_factor_matrix[7][7] = {
        {1, 24, 1440, 86400, 86400000, 86400000000, 86400000000000},   # 从秒到不同时间单位的转换因子：分钟、小时、天、月、年、世纪
        {0LL,  1LL,   60LL,  3600LL,  3600000LL,  3600000000LL,  3600000000000LL},   # 从分钟到不同时间单位的转换因子：小时、天、月、年、世纪
        {0,  0,   1,     60,    60000,    60000000,    60000000000},   # 从小时到不同时间单位的转换因子：天、月、年、世纪
        {0,  0,   0,      1,     1000,     1000000,     1000000000},   # 从天到不同时间单位的转换因子：秒、毫秒、微秒、纳秒
        {0,  0,   0,      0,        1,        1000,        1000000},   # 从月到不同时间单位的转换因子：天、小时、分钟、秒、毫秒
        {0,  0,   0,      0,        0,           1,           1000},   # 从年到不同时间单位的转换因子：小时、分钟、秒、毫秒、微秒
        {0,  0,   0,      0,        0,           0,              1}    # 从世纪到不同时间单位的转换因子：年、月、天、小时、分钟、秒
    };
# 定义一个内联函数 max_value，返回两个整数中较大的那个数
cdef int max_value(int left, int right) noexcept nogil:
    # 如果 left 大于 right，则返回 left
    if left > right:
        return left
    # 否则返回 right
    return right


# 定义一个内联函数 min_value，返回两个整数中较小的那个数
cdef int min_value(int left, int right) noexcept nogil:
    # 如果 left 小于 right，则返回 left
    if left < right:
        return left
    # 否则返回 right
    return right


# 定义一个内联函数 get_daytime_conversion_factor，返回日间转换系数
cdef int64_t get_daytime_conversion_factor(int from_index, int to_index) noexcept nogil:
    # 定义并初始化变量 row 和 col，分别表示 from_index 和 to_index 的最小值和最大值
    cdef:
        int row = min_value(from_index, to_index)
        int col = max_value(from_index, to_index)
    
    # 如果 row 小于 6，或者 col 小于 6，则返回 0，表示频率严格低于每日，不使用日间转换系数
    if row < 6:
        return 0
    elif col < 6:
        return 0
    
    # 否则返回日间转换系数矩阵中指定位置的值
    return daytime_conversion_factor_matrix[row - 6][col - 6]


# 定义一个内联函数 nofunc，返回 INT32_MIN，表示未定义的功能
cdef int64_t nofunc(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    return INT32_MIN


# 定义一个内联函数 no_op，直接返回传入的 ordinal，表示不进行任何操作
cdef int64_t no_op(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    return ordinal


# 定义一个内联函数 get_asfreq_func，返回根据给定频率组获取的频率转换函数
cdef freq_conv_func get_asfreq_func(int from_freq, int to_freq) noexcept nogil:
    # 获取 from_freq 和 to_freq 对应的频率组
    cdef:
        int from_group = get_freq_group(from_freq)
        int to_group = get_freq_group(to_freq)

    # 如果 from_group 为未定义（FR_UND），则将其设为每日（FR_DAY）
    if from_group == FR_UND:
        from_group = FR_DAY

    # 根据 from_group 和 to_group 的组合返回相应的频率转换函数
    if from_group == FR_BUS:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_BtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_BtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_BtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_BtoW
        elif to_group == FR_BUS:
            return <freq_conv_func>no_op
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_BtoDT
        else:
            return <freq_conv_func>nofunc

    elif to_group == FR_BUS:
        if from_group == FR_ANN:
            return <freq_conv_func>asfreq_AtoB
        elif from_group == FR_QTR:
            return <freq_conv_func>asfreq_QtoB
        elif from_group == FR_MTH:
            return <freq_conv_func>asfreq_MtoB
        elif from_group == FR_WK:
            return <freq_conv_func>asfreq_WtoB
        elif from_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_DTtoB
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_ANN:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_AtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_AtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_AtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_AtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_AtoDT
        else:
            return <freq_conv_func>nofunc
    # 如果起始频率为季度（Quarterly）
    elif from_group == FR_QTR:
        # 如果目标频率为年度（Annual）
        if to_group == FR_ANN:
            # 返回季度到年度的频率转换函数
            return <freq_conv_func>asfreq_QtoA
        # 如果目标频率为季度（Quarterly）
        elif to_group == FR_QTR:
            # 返回季度到季度的频率转换函数
            return <freq_conv_func>asfreq_QtoQ
        # 如果目标频率为月度（Monthly）
        elif to_group == FR_MTH:
            # 返回季度到月度的频率转换函数
            return <freq_conv_func>asfreq_QtoM
        # 如果目标频率为周度（Weekly）
        elif to_group == FR_WK:
            # 返回季度到周度的频率转换函数
            return <freq_conv_func>asfreq_QtoW
        # 如果目标频率为日度、小时、分钟、秒、毫秒、微秒或纳秒
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            # 返回季度到日间时间单位的频率转换函数
            return <freq_conv_func>asfreq_QtoDT
        else:
            # 返回无效转换函数（应该不会到达这里）
            return <freq_conv_func>nofunc

    # 如果起始频率为月度（Monthly）
    elif from_group == FR_MTH:
        # 如果目标频率为年度（Annual）
        if to_group == FR_ANN:
            # 返回月度到年度的频率转换函数
            return <freq_conv_func>asfreq_MtoA
        # 如果目标频率为季度（Quarterly）
        elif to_group == FR_QTR:
            # 返回月度到季度的频率转换函数
            return <freq_conv_func>asfreq_MtoQ
        # 如果目标频率为月度（Monthly）
        elif to_group == FR_MTH:
            # 返回月度到月度的无操作函数
            return <freq_conv_func>no_op
        # 如果目标频率为周度（Weekly）
        elif to_group == FR_WK:
            # 返回月度到周度的频率转换函数
            return <freq_conv_func>asfreq_MtoW
        # 如果目标频率为日度、小时、分钟、秒、毫秒、微秒或纳秒
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            # 返回月度到日间时间单位的频率转换函数
            return <freq_conv_func>asfreq_MtoDT
        else:
            # 返回无效转换函数（应该不会到达这里）
            return <freq_conv_func>nofunc

    # 如果起始频率为周度（Weekly）
    elif from_group == FR_WK:
        # 如果目标频率为年度（Annual）
        if to_group == FR_ANN:
            # 返回周度到年度的频率转换函数
            return <freq_conv_func>asfreq_WtoA
        # 如果目标频率为季度（Quarterly）
        elif to_group == FR_QTR:
            # 返回周度到季度的频率转换函数
            return <freq_conv_func>asfreq_WtoQ
        # 如果目标频率为月度（Monthly）
        elif to_group == FR_MTH:
            # 返回周度到月度的频率转换函数
            return <freq_conv_func>asfreq_WtoM
        # 如果目标频率为周度（Weekly）
        elif to_group == FR_WK:
            # 返回周度到周度的频率转换函数
            return <freq_conv_func>asfreq_WtoW
        # 如果目标频率为日度、小时、分钟、秒、毫秒、微秒或纳秒
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            # 返回周度到日间时间单位的频率转换函数
            return <freq_conv_func>asfreq_WtoDT
        else:
            # 返回无效转换函数（应该不会到达这里）
            return <freq_conv_func>nofunc

    # 如果起始频率为日度、小时、分钟、秒、毫秒、微秒或纳秒
    elif from_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
        # 如果目标频率为年度（Annual）
        if to_group == FR_ANN:
            # 返回日间时间单位到年度的频率转换函数
            return <freq_conv_func>asfreq_DTtoA
        # 如果目标频率为季度（Quarterly）
        elif to_group == FR_QTR:
            # 返回日间时间单位到季度的频率转换函数
            return <freq_conv_func>asfreq_DTtoQ
        # 如果目标频率为月度（Monthly）
        elif to_group == FR_MTH:
            # 返回日间时间单位到月度的频率转换函数
            return <freq_conv_func>asfreq_DTtoM
        # 如果目标频率为周度（Weekly）
        elif to_group == FR_WK:
            # 返回日间时间单位到周度的频率转换函数
            return <freq_conv_func>asfreq_DTtoW
        # 如果目标频率为日度、小时、分钟、秒、毫秒、微秒或纳秒
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            # 如果起始频率高于目标频率，返回降采样的频率转换函数；否则返回升采样的频率转换函数
            if from_group > to_group:
                return <freq_conv_func>downsample_daytime
            else:
                return <freq_conv_func>upsample_daytime
        else:
            # 返回无效转换函数（应该不会到达这里）
            return <freq_conv_func>nofunc

    # 如果无法识别起始频率或目标频率
    else:
        # 返回无效转换函数（应该不会到达这里）
        return <freq_conv_func>nofunc
# --------------------------------------------------------------------
# Frequency Conversion Helpers

# 将 Unix 时间转换为工作日计数（从 1970-01-01 开始），以周为单位，周日算作一周的最后一天
cdef int64_t DtoB_weekday(int64_t unix_date) noexcept nogil:
    return ((unix_date + 4) // 7) * 5 + ((unix_date + 4) % 7) - 4

# 将给定日期结构体中的日期转换为工作日计数
cdef int64_t DtoB(npy_datetimestruct *dts, int roll_back,
                  int64_t unix_date) noexcept nogil:
    # 计算当前日期是一周的第几天
    cdef:
        int day_of_week = dayofweek(dts.year, dts.month, dts.day)

    if roll_back == 1:
        if day_of_week > 4:
            # 将日期回滚至周末前的周五
            unix_date -= (day_of_week - 4)
    else:
        if day_of_week > 4:
            # 将日期调整至周末后的周一
            unix_date += (7 - day_of_week)

    return DtoB_weekday(unix_date)

# 将日内时间序数向上采样到更高频率
cdef int64_t upsample_daytime(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    if af_info.is_end:
        return (ordinal + 1) * af_info.intraday_conversion_factor - 1
    else:
        return ordinal * af_info.intraday_conversion_factor

# 将日内时间序数向下采样到更低频率
cdef int64_t downsample_daytime(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    return ordinal // af_info.intraday_conversion_factor

# 通过每日转换函数链依次转换时间序数
cdef int64_t transform_via_day(int64_t ordinal,
                               asfreq_info *af_info,
                               freq_conv_func first_func,
                               freq_conv_func second_func) noexcept nogil:
    cdef:
        int64_t result

    result = first_func(ordinal, af_info)
    result = second_func(result, af_info)
    return result

# --------------------------------------------------------------------
# Conversion _to_ Daily Freq

# 将年度频率转换为每日频率（年度到每日）
cdef int64_t asfreq_AtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int64_t unix_date
        npy_datetimestruct dts

    ordinal += af_info.is_end

    dts.year = ordinal + 1970
    dts.month = 1
    adjust_dts_for_month(&dts, af_info.from_end)

    unix_date = unix_date_from_ymd(dts.year, dts.month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)

# 将季度频率转换为每日频率（季度到每日）
cdef int64_t asfreq_QtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int64_t unix_date
        npy_datetimestruct dts

    ordinal += af_info.is_end

    dts.year = ordinal // 4 + 1970
    dts.month = (ordinal % 4) * 3 + 1
    adjust_dts_for_month(&dts, af_info.from_end)

    unix_date = unix_date_from_ymd(dts.year, dts.month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)

# 将月度频率转换为每日频率（月度到每日）
cdef int64_t asfreq_MtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int64_t unix_date
        int year, month

    ordinal += af_info.is_end

    year = ordinal // 12 + 1970
    month = ordinal % 12 + 1

    unix_date = unix_date_from_ymd(year, month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)
# 定义函数 asfreq_WtoDT，将周频率转换为天时间单位，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_WtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 计算新的序数值，使用指定的偏移量和布尔值计算
    ordinal = (ordinal * 7 + af_info.from_end - 4 +
               (7 - 1) * (af_info.is_end - 1))
    # 调用 upsample_daytime 函数处理序数和 af_info 结构体，返回处理后的结果
    return upsample_daytime(ordinal, af_info)


# --------------------------------------------------------------------
# 转换为工作日频率的函数定义

# 定义函数 asfreq_AtoB，将年频率转换为工作日频率，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_AtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int roll_back  # 定义整型变量 roll_back
        npy_datetimestruct dts  # 定义 npy_datetimestruct 结构体变量 dts
        int64_t unix_date = asfreq_AtoDT(ordinal, af_info)  # 调用 asfreq_AtoDT 函数，将结果赋给 unix_date

    # 调用 pandas_datetime_to_datetimestruct 函数，将 unix_date 转换为 NPY_FR_D 格式的日期时间结构体 dts
    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # 设置 roll_back 变量为 af_info.is_end 的值
    roll_back = af_info.is_end
    # 调用 DtoB 函数，将 dts 和 roll_back 作为参数，返回转换后的工作日频率序数
    return DtoB(&dts, roll_back, unix_date)


# 定义函数 asfreq_QtoB，将季度频率转换为工作日频率，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_QtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int roll_back  # 定义整型变量 roll_back
        npy_datetimestruct dts  # 定义 npy_datetimestruct 结构体变量 dts
        int64_t unix_date = asfreq_QtoDT(ordinal, af_info)  # 调用 asfreq_QtoDT 函数，将结果赋给 unix_date

    # 调用 pandas_datetime_to_datetimestruct 函数，将 unix_date 转换为 NPY_FR_D 格式的日期时间结构体 dts
    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # 设置 roll_back 变量为 af_info.is_end 的值
    roll_back = af_info.is_end
    # 调用 DtoB 函数，将 dts 和 roll_back 作为参数，返回转换后的工作日频率序数
    return DtoB(&dts, roll_back, unix_date)


# 定义函数 asfreq_MtoB，将月频率转换为工作日频率，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_MtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int roll_back  # 定义整型变量 roll_back
        npy_datetimestruct dts  # 定义 npy_datetimestruct 结构体变量 dts
        int64_t unix_date = asfreq_MtoDT(ordinal, af_info)  # 调用 asfreq_MtoDT 函数，将结果赋给 unix_date

    # 调用 pandas_datetime_to_datetimestruct 函数，将 unix_date 转换为 NPY_FR_D 格式的日期时间结构体 dts
    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # 设置 roll_back 变量为 af_info.is_end 的值
    roll_back = af_info.is_end
    # 调用 DtoB 函数，将 dts 和 roll_back 作为参数，返回转换后的工作日频率序数
    return DtoB(&dts, roll_back, unix_date)


# 定义函数 asfreq_WtoB，将周频率转换为工作日频率，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_WtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int roll_back  # 定义整型变量 roll_back
        npy_datetimestruct dts  # 定义 npy_datetimestruct 结构体变量 dts
        int64_t unix_date = asfreq_WtoDT(ordinal, af_info)  # 调用 asfreq_WtoDT 函数，将结果赋给 unix_date

    # 调用 pandas_datetime_to_datetimestruct 函数，将 unix_date 转换为 NPY_FR_D 格式的日期时间结构体 dts
    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # 设置 roll_back 变量为 af_info.is_end 的值
    roll_back = af_info.is_end
    # 调用 DtoB 函数，将 dts 和 roll_back 作为参数，返回转换后的工作日频率序数
    return DtoB(&dts, roll_back, unix_date)


# 定义函数 asfreq_DTtoB，将天频率转换为工作日频率，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_DTtoB(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        int roll_back  # 定义整型变量 roll_back
        npy_datetimestruct dts  # 定义 npy_datetimestruct 结构体变量 dts
        int64_t unix_date = downsample_daytime(ordinal, af_info)  # 调用 downsample_daytime 函数，将结果赋给 unix_date

    # 调用 pandas_datetime_to_datetimestruct 函数，将 unix_date 转换为 NPY_FR_D 格式的日期时间结构体 dts
    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # 设置 roll_back 变量为 af_info.is_end 取反后的值
    # 这里的使用方式与其他函数定义中的 roll_back 方向相反
    roll_back = 1 - af_info.is_end
    # 调用 DtoB 函数，将 dts 和 roll_back 作为参数，返回转换后的工作日频率序数
    return DtoB(&dts, roll_back, unix_date)


# ----------------------------------------------------------------------
# 从每日频率转换的函数定义

# 定义函数 asfreq_DTtoA，将每日频率转换为年频率，接受一个整数参数和 asfreq_info 结构体指针
cdef int64_t asfreq_DTtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        npy_datetimestruct dts  # 定义 npy_datetimestruct 结构体变量 dts

    # 调用 downsample_daytime 函数，将结果赋给 ordinal
    ordinal = downsample_daytime(ordinal, af_info)
    # 调用 pandas_datetime_to_datetimestruct 函数，将 ordinal 转换为 NPY_FR_D 格式的日期时间结构体 dts
    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, &dts)
    # 调用 dts_to_year_ordinal 函数，将 dts 和 af_info.to_end 作为参数，返回年序数
    return dts_to_year_ordinal(&dts, af_info.to_end)


# 定义函数 DtoQ_yq，将日期时间结构体转换为季度信息，接受一个整数参数、asfreq_info 结构体指针和 npy_datetimestruct 结构体指针
cdef int DtoQ_yq(int64_t ordinal, asfreq_info *af_info,
                 npy_datetimestruct* d
    # 使用函数 downsample_daytime 对变量 ordinal 进行白天下采样处理
    ordinal = downsample_daytime(ordinal, af_info)

    # 使用函数 DtoQ_yq 将变量 ordinal 转换为季度，同时更新 dts 结构体
    quarter = DtoQ_yq(ordinal, af_info, &dts)
    
    # 返回计算得到的年份和季度对应的整数值
    return <int64_t>((dts.year - 1970) * 4 + quarter - 1)
# 转换函数：将日频率的日期转换为月份的序数
cdef int64_t asfreq_DTtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    cdef:
        npy_datetimestruct dts

    # 调用 downsample_daytime 函数，将日期降采样到日频率
    ordinal = downsample_daytime(ordinal, af_info)
    # 将 pandas 的日期时间转换为 npy_datetimestruct 结构体
    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, &dts)
    # 返回日期时间结构体的月份序数
    return dts_to_month_ordinal(&dts)


# 转换函数：将日频率的日期转换为周数
cdef int64_t asfreq_DTtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 downsample_daytime 函数，将日期降采样到日频率
    ordinal = downsample_daytime(ordinal, af_info)
    # 调用 unix_date_to_week 函数，将 UNIX 日期转换为周数
    return unix_date_to_week(ordinal, af_info.to_end)


# 计算函数：将 UNIX 日期转换为周数
cdef int64_t unix_date_to_week(int64_t unix_date, int to_end) noexcept nogil:
    # 计算给定 UNIX 日期所在的周数
    return (unix_date + 3 - to_end) // 7 + 1


# --------------------------------------------------------------------
# 从工作日频率转换

# 转换函数：将工作日频率的日期转换为日频率
cdef int64_t asfreq_BtoDT(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 根据工作日频率的日期计算对应的日频率日期
    ordinal = ((ordinal + 3) // 5) * 7 + (ordinal + 3) % 5 - 3
    # 调用 upsample_daytime 函数，将日期升采样到日频率
    return upsample_daytime(ordinal, af_info)


# 转换函数：将工作日频率的日期转换为年度频率
cdef int64_t asfreq_BtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将工作日频率日期转换为年度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoA)


# 转换函数：将工作日频率的日期转换为季度频率
cdef int64_t asfreq_BtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将工作日频率日期转换为季度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


# 转换函数：将工作日频率的日期转换为月度频率
cdef int64_t asfreq_BtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将工作日频率日期转换为月度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoM)


# 转换函数：将工作日频率的日期转换为周度频率
cdef int64_t asfreq_BtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将工作日频率日期转换为周度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# 从年度频率转换

# 转换函数：将年度频率的日期转换为年度频率（保持不变）
cdef int64_t asfreq_AtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将年度频率日期转换为年度频率日期（保持不变）
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoA)


# 转换函数：将年度频率的日期转换为季度频率
cdef int64_t asfreq_AtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将年度频率日期转换为季度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


# 转换函数：将年度频率的日期转换为月度频率
cdef int64_t asfreq_AtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将年度频率日期转换为月度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoM)


# 转换函数：将年度频率的日期转换为周度频率
cdef int64_t asfreq_AtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用 transform_via_day 函数，通过日频率转换函数将年度频率日期转换为周度频率日期
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoW)
# ----------------------------------------------------------------------
# Conversion _from_ Quarterly Freq

# 将季度频率转换为季度频率的函数
cdef int64_t asfreq_QtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_QtoDT 和 asfreq_DTtoQ 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


# 将季度频率转换为年度频率的函数
cdef int64_t asfreq_QtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_QtoDT 和 asfreq_DTtoA 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoA)


# 将季度频率转换为月度频率的函数
cdef int64_t asfreq_QtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_QtoDT 和 asfreq_DTtoM 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoM)


# 将季度频率转换为周度频率的函数
cdef int64_t asfreq_QtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_QtoDT 和 asfreq_DTtoW 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Monthly Freq

# 将月度频率转换为年度频率的函数
cdef int64_t asfreq_MtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_MtoDT 和 asfreq_DTtoA 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoA)


# 将月度频率转换为季度频率的函数
cdef int64_t asfreq_MtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_MtoDT 和 asfreq_DTtoQ 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


# 将月度频率转换为周度频率的函数
cdef int64_t asfreq_MtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_MtoDT 和 asfreq_DTtoW 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Weekly Freq

# 将周度频率转换为年度频率的函数
cdef int64_t asfreq_WtoA(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_WtoDT 和 asfreq_DTtoA 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoA)


# 将周度频率转换为季度频率的函数
cdef int64_t asfreq_WtoQ(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_WtoDT 和 asfreq_DTtoQ 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


# 将周度频率转换为月度频率的函数
cdef int64_t asfreq_WtoM(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 通过 transform_via_day 函数转换日期，使用 asfreq_WtoDT 和 asfreq_DTtoM 两个函数
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoM)


# 将周度频率转换为周度频率的函数
cdef int64_t asfreq_WtoW(int64_t ordinal, asfreq_info *af_info) noexcept nogil:
    # 调用一个函数，通过日期序数和一些信息进行转换，返回转换后的结果
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoW)
# ----------------------------------------------------------------------

@cython.cdivision
# 使用 Cython 的 cdivision 插件，优化整数除法运算的性能

cdef char* c_strftime(npy_datetimestruct *dts, char *fmt):
    """
    Generate a nice string representation of the period
    object, originally from DateObject_strftime

    Parameters
    ----------
    dts : npy_datetimestruct*
        包含日期时间信息的结构体指针
    fmt : char*
        格式化字符串

    Returns
    -------
    result : char*
        格式化后的字符串
    """
    cdef:
        tm c_date
        char *result
        int result_len = strlen(fmt) + 50

    c_date.tm_sec = dts.sec
    c_date.tm_min = dts.min
    c_date.tm_hour = dts.hour
    c_date.tm_mday = dts.day
    c_date.tm_mon = dts.month - 1
    c_date.tm_year = dts.year - 1900
    c_date.tm_wday = (dayofweek(dts.year, dts.month, dts.day) + 1) % 7
    c_date.tm_yday = get_day_of_year(dts.year, dts.month, dts.day) - 1
    c_date.tm_isdst = -1

    # 分配内存来存储格式化后的字符串
    result = <char*>malloc(result_len * sizeof(char))
    if result is NULL:
        raise MemoryError()

    # 使用 strftime 函数将日期时间格式化为字符串
    strftime(result, result_len, fmt, &c_date)

    return result


# ----------------------------------------------------------------------
# Conversion between date_info and npy_datetimestruct

cdef int get_freq_group(int freq) noexcept nogil:
    # 参考 FreqGroup.get_freq_group 方法实现
    # 返回频率的分组
    return (freq // 1000) * 1000


cdef int get_freq_group_index(int freq) noexcept nogil:
    # 返回频率的分组索引
    return freq // 1000


cdef void adjust_dts_for_month(npy_datetimestruct* dts, int from_end) noexcept nogil:
    # 调整日期时间结构体以处理月份偏移
    if from_end != 12:
        dts.month += from_end
        if dts.month > 12:
            dts.month -= 12
        else:
            dts.year -= 1


cdef void adjust_dts_for_qtr(npy_datetimestruct* dts, int to_end) noexcept nogil:
    # 调整日期时间结构体以处理季度偏移
    if to_end != 12:
        dts.month -= to_end
        if dts.month <= 0:
            dts.month += 12
        else:
            dts.year += 1


# Find the unix_date (days elapsed since datetime(1970, 1, 1)
# for the given year/month/day.
# Assumes GREGORIAN_CALENDAR */
cdef int64_t unix_date_from_ymd(int year, int month, int day) noexcept nogil:
    # 计算给定年月日对应的 UNIX 时间戳
    cdef:
        npy_datetimestruct dts
        int64_t unix_date

    memset(&dts, 0, sizeof(npy_datetimestruct))
    dts.year = year
    dts.month = month
    dts.day = day
    # 调用 npy_datetimestruct_to_datetime 函数转换为 UNIX 时间戳
    unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, &dts)
    return unix_date


cdef int64_t dts_to_month_ordinal(npy_datetimestruct* dts) noexcept nogil:
    # 将日期时间结构体转换为月份序数
    # 等同于 npy_datetimestruct_to_datetime(NPY_FR_M, &dts)
    return <int64_t>((dts.year - 1970) * 12 + dts.month - 1)


cdef int64_t dts_to_year_ordinal(npy_datetimestruct *dts, int to_end) noexcept nogil:
    # 将日期时间结构体转换为年份序数
    cdef:
        int64_t result

    result = npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT.NPY_FR_Y, dts)
    if dts.month > to_end:
        return result + 1
    else:
        return result


cdef int64_t dts_to_qtr_ordinal(npy_datetimestruct* dts, int to_end) noexcept nogil:
    # 将日期时间结构体转换为季度序数
    cdef:
        int quarter

    adjust_dts_for_qtr(dts, to_end)
    quarter = month_to_quarter(dts.month)
    # 计算给定日期的季度对应的序数
    return <int64_t>((dts.year - 1970) * 4 + quarter - 1)
    # 根据年份和季度计算季度序数的算法，1970年为起始年，每年四个季度，减一是为了从零开始索引季度
cdef int get_anchor_month(int freq, int freq_group) noexcept nogil:
    """
    Calculate the anchor month based on frequency and frequency group.

    Parameters
    ----------
    freq : int
        Frequency identifier.
    freq_group : int
        Frequency group identifier.

    Returns
    -------
    fmonth : int
        Calculated anchor month.
    """
    cdef:
        int fmonth

    # Calculate anchor month based on frequency and its group
    fmonth = freq - freq_group
    if fmonth == 0:
        fmonth = 12
    return fmonth


# specifically _dont_ use cdvision or else ordinals near -1 are assigned to
# incorrect dates GH#19643
@cython.cdivision(False)
cdef int64_t get_period_ordinal(npy_datetimestruct *dts, int freq) noexcept nogil:
    """
    Generate an ordinal in period space.

    Parameters
    ----------
    dts : npy_datetimestruct*
        Datetime structure pointer.
    freq : int
        Frequency identifier.

    Returns
    -------
    period_ordinal : int64_t
        Generated period ordinal.
    """
    cdef:
        int64_t unix_date
        int freq_group, fmonth
        NPY_DATETIMEUNIT unit

    freq_group = get_freq_group(freq)

    # Determine action based on frequency group
    if freq_group == FR_ANN:
        fmonth = get_anchor_month(freq, freq_group)
        return dts_to_year_ordinal(dts, fmonth)

    elif freq_group == FR_QTR:
        fmonth = get_anchor_month(freq, freq_group)
        return dts_to_qtr_ordinal(dts, fmonth)

    elif freq_group == FR_WK:
        unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, dts)
        return unix_date_to_week(unix_date, freq - FR_WK)

    elif freq == FR_BUS:
        unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, dts)
        return DtoB(dts, 0, unix_date)

    unit = freq_group_code_to_npy_unit(freq)
    return npy_datetimestruct_to_datetime(unit, dts)


cdef void get_date_info(int64_t ordinal,
                        int freq, npy_datetimestruct *dts) noexcept nogil:
    """
    Populate datetime structure `dts` with information based on `ordinal` and `freq`.

    Parameters
    ----------
    ordinal : int64_t
        Period ordinal.
    freq : int
        Frequency identifier.
    dts : npy_datetimestruct*
        Datetime structure pointer.
    """
    cdef:
        int64_t unix_date, nanos
        npy_datetimestruct dts2

    # Retrieve UNIX date and nanoseconds
    unix_date = get_unix_date(ordinal, freq)
    nanos = get_time_nanos(freq, unix_date, ordinal)

    # Populate `dts` with date information
    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, dts)

    pandas_datetime_to_datetimestruct(nanos, NPY_DATETIMEUNIT.NPY_FR_ns, &dts2)
    dts.hour = dts2.hour
    dts.min = dts2.min
    dts.sec = dts2.sec
    dts.us = dts2.us
    dts.ps = dts2.ps


cdef int64_t get_unix_date(int64_t period_ordinal, int freq) noexcept nogil:
    """
    Calculate the proleptic Gregorian ordinal date corresponding to `period_ordinal`.

    Parameters
    ----------
    period_ordinal : int64_t
        Period ordinal.
    freq : int
        Frequency identifier.

    Returns
    -------
    unix_date : int64_t
        Number of days since datetime(1970, 1, 1).
    """
    cdef:
        asfreq_info af_info
        freq_conv_func toDaily = NULL

    if freq == FR_DAY:
        return period_ordinal

    # Convert `period_ordinal` to daily frequency
    toDaily = get_asfreq_func(freq, FR_DAY)
    get_asfreq_info(freq, FR_DAY, True, &af_info)
    return toDaily(period_ordinal, &af_info)


@cython.cdivision
cdef int64_t get_time_nanos(int freq, int64_t unix_date,
                            int64_t ordinal) noexcept nogil:
    """
    Calculate nanoseconds after midnight for a given `ordinal` and `unix_date` in `freq`.

    Parameters
    ----------
    freq : int
        Frequency identifier.
    unix_date : int64_t
        Number of days since datetime(1970, 1, 1).
    ordinal : int64_t
        Period ordinal.

    Returns
    -------
    nanoseconds : int64_t
        Nanoseconds after midnight.
    """
    # Calculate nanoseconds for the given `ordinal` and `unix_date` in `freq`
    # 定义函数的参数和返回值的数据类型和意义
    Parameters
    ----------
    freq : int
        表示时间频率的整数值
    unix_date : int64_t
        表示 UNIX 时间的整数值
    ordinal : int64_t
        表示要转换的时间戳的整数值

    Returns
    -------
    int64_t
        返回计算得到的时间戳之间的差异，单位为纳秒
    """
    # 定义本地变量 sub 和 factor 作为 int64_t 类型
    cdef:
        int64_t sub, factor
        int64_t nanos_in_day = 24 * 3600 * 10**9

    # 调用 get_freq_group 函数获取时间频率的分组
    freq = get_freq_group(freq)

    # 如果频率小于等于 FR_DAY，直接返回 0
    if freq <= FR_DAY:
        return 0

    # 根据频率不同，设置不同的因子 factor
    elif freq == FR_NS:
        factor = 1

    elif freq == FR_US:
        factor = 10**3

    elif freq == FR_MS:
        factor = 10**6

    elif freq == FR_SEC:
        factor = 10**9

    elif freq == FR_MIN:
        factor = 10**9 * 60

    else:
        # 如果 freq == FR_HR，设置因子为 10^9 * 3600
        factor = 10**9 * 3600

    # 计算时间戳的差异 sub，单位为纳秒
    sub = ordinal - unix_date * (nanos_in_day / factor)
    # 返回计算结果乘以因子 factor
    return sub * factor
cdef int get_yq(int64_t ordinal, int freq, npy_datetimestruct* dts):
    """
    Find the year and quarter of a Period with the given ordinal and frequency

    Parameters
    ----------
    ordinal : int64_t
        Ordinal representing the period
    freq : int
        Frequency of the period
    dts : *npy_datetimestruct
        Pointer to a struct holding date and time information

    Returns
    -------
    quarter : int
        Implied quarterly frequency associated with `freq`

    Notes
    -----
    Sets dts.year in-place.
    """
    cdef:
        asfreq_info af_info   # Structure for conversion information
        int qtr_freq          # Frequency representing quarters
        int64_t unix_date     # Unix timestamp corresponding to ordinal
        int quarter           # Quarter derived from the date

    unix_date = get_unix_date(ordinal, freq)

    if get_freq_group(freq) == FR_QTR:  # Check if freq is quarterly
        qtr_freq = freq
    else:
        qtr_freq = FR_QTR  # Default to quarterly frequency if not

    get_asfreq_info(FR_DAY, qtr_freq, True, &af_info)  # Fetch conversion info

    quarter = DtoQ_yq(unix_date, &af_info, dts)  # Calculate quarter from date
    return quarter


cdef int month_to_quarter(int month) noexcept nogil:
    """
    Convert a month index to its corresponding quarter.

    Parameters
    ----------
    month : int
        Month index (1-12)

    Returns
    -------
    quarter : int
        Corresponding quarter (1-4)
    """
    return (month - 1) // 3 + 1  # Calculate quarter from month index


# ----------------------------------------------------------------------
# Period logic

@cython.wraparound(False)
@cython.boundscheck(False)
def periodarr_to_dt64arr(const int64_t[:] periodarr, int freq):
    """
    Convert array to datetime64 values from a set of ordinals corresponding to
    periods per period convention.

    Parameters
    ----------
    periodarr : const int64_t[:]
        Array of period ordinals
    freq : int
        Frequency of the periods

    Returns
    -------
    out.base : np.ndarray
        Array of datetime64 values corresponding to the periods
    """
    cdef:
        int64_t[::1] out   # Array for storing output
        Py_ssize_t i, N    # Index and length variables

    if freq < 6000:  # Handle frequencies not directly corresponding to datetime64 units
        N = len(periodarr)
        out = np.empty(N, dtype="i8")  # Initialize output array

        # Convert each period ordinal to datetime64
        for i in range(N):
            out[i] = period_ordinal_to_dt64(periodarr[i], freq)

        return out.base  # Return underlying np.ndarray for efficiency

    else:
        # Direct mapping for high-frequency cases
        if freq == FR_NS:
            return periodarr.base  # Return base array for nanoseconds

        # Map frequencies to datetime64 units
        if freq == FR_US:
            dta = periodarr.base.view("M8[us]")
        elif freq == FR_MS:
            dta = periodarr.base.view("M8[ms]")
        elif freq == FR_SEC:
            dta = periodarr.base.view("M8[s]")
        elif freq == FR_MIN:
            dta = periodarr.base.view("M8[m]")
        elif freq == FR_HR:
            dta = periodarr.base.view("M8[h]")
        elif freq == FR_DAY:
            dta = periodarr.base.view("M8[D]")
        return astype_overflowsafe(dta, dtype=DT64NS_DTYPE)


cdef void get_asfreq_info(int from_freq, int to_freq,
                          bint is_end, asfreq_info *af_info) noexcept nogil:
    """
    Construct the `asfreq_info` object used to convert an ordinal from
    `from_freq` to `to_freq`.

    Parameters
    ----------
    from_freq : int
        Starting frequency of the conversion
    to_freq : int
        Target frequency of the conversion
    is_end : bool
        Flag indicating if it's an end conversion
    af_info : *asfreq_info
        Pointer to the structure holding conversion information
    """
    cdef:
        int from_group = get_freq_group(from_freq)  # Group of starting frequency
        int to_group = get_freq_group(to_freq)      # Group of target frequency

    af_info.is_end = is_end  # Set end flag in af_info structure

    # Calculate daytime conversion factor between the two frequency groups
    af_info.intraday_conversion_factor = get_daytime_conversion_factor(
        get_freq_group_index(max_value(from_group, FR_DAY)),
        get_freq_group_index(max_value(to_group, FR_DAY)))
    # 如果起始频率组为周频率
    if from_group == FR_WK:
        # 计算起始频率组的周末日期，并赋值给af_info的from_end属性
        af_info.from_end = calc_week_end(from_freq, from_group)
    # 如果起始频率组为年度频率
    elif from_group == FR_ANN:
        # 计算起始频率组的年末日期，并赋值给af_info的from_end属性
        af_info.from_end = calc_a_year_end(from_freq, from_group)
    # 如果起始频率组为季度频率
    elif from_group == FR_QTR:
        # 计算起始频率组的年末日期，并赋值给af_info的from_end属性
        af_info.from_end = calc_a_year_end(from_freq, from_group)

    # 如果结束频率组为周频率
    if to_group == FR_WK:
        # 计算结束频率组的周末日期，并赋值给af_info的to_end属性
        af_info.to_end = calc_week_end(to_freq, to_group)
    # 如果结束频率组为年度频率
    elif to_group == FR_ANN:
        # 计算结束频率组的年末日期，并赋值给af_info的to_end属性
        af_info.to_end = calc_a_year_end(to_freq, to_group)
    # 如果结束频率组为季度频率
    elif to_group == FR_QTR:
        # 计算结束频率组的年末日期，并赋值给af_info的to_end属性
        af_info.to_end = calc_a_year_end(to_freq, to_group)
@cython.cdivision
cdef int calc_a_year_end(int freq, int group) noexcept nogil:
    """
    Calculate the end of the year based on frequency and group.
    """
    cdef:
        int result = (freq - group) % 12
    if result == 0:
        return 12
    else:
        return result


cdef int calc_week_end(int freq, int group) noexcept nogil:
    """
    Calculate the end of the week based on frequency and group.
    """
    return freq - group


cpdef int64_t period_asfreq(int64_t ordinal, int freq1, int freq2, bint end):
    """
    Convert period ordinal from one frequency to another, and if upsampling,
    choose to use start ('S') or end ('E') of period.
    """
    cdef:
        int64_t retval

    _period_asfreq(&ordinal, &retval, 1, freq1, freq2, end)
    return retval


@cython.wraparound(False)
@cython.boundscheck(False)
def period_asfreq_arr(ndarray[int64_t] arr, int freq1, int freq2, bint end):
    """
    Convert int64-array of period ordinals from one frequency to another, and
    if upsampling, choose to use start ('S') or end ('E') of period.
    """
    cdef:
        Py_ssize_t n = len(arr)
        Py_ssize_t increment = arr.strides[0] // 8
        ndarray[int64_t] result = cnp.PyArray_EMPTY(
            arr.ndim, arr.shape, cnp.NPY_INT64, 0
        )

    _period_asfreq(
        <int64_t*>cnp.PyArray_DATA(arr),
        <int64_t*>cnp.PyArray_DATA(result),
        n,
        freq1,
        freq2,
        end,
        increment,
    )
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _period_asfreq(
    int64_t* ordinals,
    int64_t* out,
    Py_ssize_t length,
    int freq1,
    int freq2,
    bint end,
    Py_ssize_t increment=1,
) noexcept:
    """
    Internal function to convert period ordinals from one frequency to another.
    """
    cdef:
        Py_ssize_t i
        freq_conv_func func
        asfreq_info af_info
        int64_t val

    if length == 1 and ordinals[0] == NPY_NAT:
        # fastpath avoid calling get_asfreq_func
        out[0] = NPY_NAT
        return

    func = get_asfreq_func(freq1, freq2)
    get_asfreq_info(freq1, freq2, end, &af_info)

    for i in range(length):
        val = ordinals[i * increment]
        if val != NPY_NAT:
            val = func(val, &af_info)
        out[i] = val


cpdef int64_t period_ordinal(int y, int m, int d, int h, int min,
                             int s, int us, int ps, int freq):
    """
    Find the ordinal representation of the given datetime components at the
    frequency `freq`.
    """
    cdef:
        npy_datetimestruct dts
    dts.year = y
    dts.month = m
    dts.day = d
    dts.hour = h
    dts.min = min
    dts.sec = s
    dts.us = us
    dts.ps = ps
    return get_period_ordinal(&dts, freq)


cdef int64_t period_ordinal_to_dt64(int64_t ordinal, int freq) except? -1:
    """
    Convert period ordinal to datetime64 at the given frequency.
    """
    cdef:
        npy_datetimestruct dts
        int64_t result

    if ordinal == NPY_NAT:
        return NPY_NAT

    get_date_info(ordinal, freq, &dts)
    try:
        # 尝试将 NPY_DATETIMEUNIT.NPY_FR_ns 格式的日期时间结构转换为 datetime 对象
        result = npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT.NPY_FR_ns, &dts)
    except OverflowError as err:
        # 如果发生 OverflowError，则将日期时间结构转换为 ISO 格式字符串
        fmt = dts_to_iso_string(&dts)
        # 抛出 OutOfBoundsDatetime 异常，包含具体的错误信息和原始异常对象
        raise OutOfBoundsDatetime(f"Out of bounds nanosecond timestamp: {fmt}") from err

    # 返回转换后的 datetime 对象或 ISO 格式字符串
    return result
cdef str period_format(int64_t value, int freq, object fmt=None):
    # 定义变量和类型声明
    cdef:
        int freq_group, quarter  # 频率组和季度数
        npy_datetimestruct dts  # numpy 的日期时间结构体
        bint is_fmt_none  # 标记是否未指定格式

    if value == NPY_NAT:
        return "NaT"  # 如果值为 NPY_NAT，返回字符串 "NaT"

    # 填充 dts 和获取频率组信息
    get_date_info(value, freq, &dts)
    freq_group = get_freq_group(freq)

    # 根据频率组选择适当的默认格式
    is_fmt_none = fmt is None
    if freq_group == FR_ANN and (is_fmt_none or fmt == "%Y"):
        return f"{dts.year}"  # 年度频率，返回年份

    elif freq_group == FR_QTR and (is_fmt_none or fmt == "%FQ%q"):
        # 季度频率，获取季度并调整 dts.year 为财年
        quarter = get_yq(value, freq, &dts)
        return f"{dts.year}Q{quarter}"  # 返回年份和季度

    elif freq_group == FR_MTH and (is_fmt_none or fmt == "%Y-%m"):
        return f"{dts.year}-{dts.month:02d}"  # 月度频率，返回年份和月份

    elif freq_group == FR_WK and is_fmt_none:
        # 周频率，特殊处理：开始日期/结束日期。递归调用 period_asfreq 函数
        left = period_asfreq(value, freq, FR_DAY, 0)
        right = period_asfreq(value, freq, FR_DAY, 1)
        return f"{period_format(left, FR_DAY)}/{period_format(right, FR_DAY)}"

    elif (freq_group == FR_BUS or freq_group == FR_DAY) and (is_fmt_none or fmt == "%Y-%m-%d"):
        return f"{dts.year}-{dts.month:02d}-{dts.day:02d}"  # 工作日或每日频率，返回年月日格式

    elif freq_group == FR_HR and (is_fmt_none or fmt == "%Y-%m-%d %H:00"):
        return f"{dts.year}-{dts.month:02d}-{dts.day:02d} {dts.hour:02d}:00"  # 小时频率，返回年月日时格式

    elif freq_group == FR_MIN and (is_fmt_none or fmt == "%Y-%m-%d %H:%M"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}")  # 分钟频率，返回年月日时分格式

    elif freq_group == FR_SEC and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")  # 秒频率，返回年月日时分秒格式

    elif freq_group == FR_MS and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S.%l"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}"
                f".{(dts.us // 1_000):03d}")  # 毫秒频率，返回年月日时分秒毫秒格式

    elif freq_group == FR_US and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S.%u"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}"
                f".{(dts.us):06d}")  # 微秒频率，返回年月日时分秒微秒格式

    elif freq_group == FR_NS and (is_fmt_none or fmt == "%Y-%m-%d %H:%M:%S.%n"):
        return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}"
                f".{((dts.us * 1000) + (dts.ps // 1000)):09d}")  # 纳秒频率，返回年月日时分秒纳秒格式

    elif is_fmt_none:
        # 若未指定格式且频率组无效，抛出异常
        raise ValueError(f"Unknown freq: {freq}")

    else:
        # 如果需要自定义格式，则使用 _period_strftime 函数处理
        if isinstance(fmt, str):
            # 在当前区域设置下编码，以处理 fmt 包含非 UTF-8 字符
            fmt = <bytes>util.string_encode_locale(fmt)

        return _period_strftime(value, freq, fmt, dts)
cdef list extra_fmts = [(b"%q", b"^`AB`^"),  # 定义额外的格式化指令列表，包含'%q'到'^`AB`^'的映射
                        (b"%f", b"^`CD`^"),  # '%f'到'^`CD`^'的映射
                        (b"%F", b"^`EF`^"),  # '%F'到'^`EF`^'的映射
                        (b"%l", b"^`GH`^"),  # '%l'到'^`GH`^'的映射
                        (b"%u", b"^`IJ`^"),  # '%u'到'^`IJ`^'的映射
                        (b"%n", b"^`KL`^")]  # '%n'到'^`KL`^'的映射

cdef list str_extra_fmts = ["^`AB`^", "^`CD`^", "^`EF`^",  # 定义字符串形式的额外格式化指令列表
                            "^`GH`^", "^`IJ`^", "^`KL`^"]

cdef str _period_strftime(int64_t value, int freq, bytes fmt, npy_datetimestruct dts):
    cdef:
        Py_ssize_t i  # 循环变量的定义
        char *formatted  # C 字符串格式化后的结果
        bytes pat, brepl  # 模式和替换字符串的定义
        list found_pat = [False] * len(extra_fmts)  # 记录找到的额外指令的列表
        int quarter  # 季度变量
        int32_t us, ps  # 微秒和纳秒的整数变量
        str result, repl  # 结果字符串和替换字符串

    # 在格式化字符串中查找额外的指令，并用不会被c_strftime处理的占位符替换它们
    for i in range(len(extra_fmts)):
        pat = extra_fmts[i][0]
        brepl = extra_fmts[i][1]
        if pat in fmt:
            fmt = fmt.replace(pat, brepl)
            found_pat[i] = True

    # 调用c_strftime处理常规的日期时间指令
    formatted = c_strftime(&dts, <char*>fmt)

    # 根据当前区域设置解码结果
    result = util.char_to_string_locale(formatted)
    free(formatted)

    # 填充额外指令对应的占位符
    us = dts.us
    ps = dts.ps
    if any(found_pat[0:3]):
        quarter = get_yq(value, freq, &dts)  # 如果找到'%q'、'%f'或'%F'，调用get_yq修改dts以获取季度信息
    else:
        quarter = 0

    for i in range(len(extra_fmts)):
        if found_pat[i]:
            if i == 0:  # 如果是'%q'，表示一位数季度
                repl = f"{quarter}"
            elif i == 1:  # 如果是'%f'，表示两位数的'Fiscal'年份
                repl = f"{(dts.year % 100):02d}"
            elif i == 2:  # 如果是'%F'，表示完整的'Fiscal'年份
                repl = str(dts.year)
            elif i == 3:  # 如果是'%l'，表示毫秒
                repl = f"{(us // 1_000):03d}"
            elif i == 4:  # 如果是'%u'，表示微秒
                repl = f"{(us):06d}"
            elif i == 5:  # 如果是'%n'，表示纳秒
                repl = f"{((us * 1000) + (ps // 1000)):09d}"

            result = result.replace(str_extra_fmts[i], repl)  # 将结果中的占位符替换为实际值

    return result


def period_array_strftime(
    ndarray values, int dtype_code, object na_rep, str date_format
):
    """
    Vectorized Period.strftime used for PeriodArray._format_native_types.

    Parameters
    ----------
    values : ndarray[int64_t], ndim unrestricted  # 传入的值数组，可以是任意维度的int64_t类型数组
    dtype_code : int  # 对应于PeriodDtype._dtype_code的数据类型代码
    na_rep : any  # 缺失值的表示形式，可以是任意类型
    date_format : str or None  # 日期格式字符串或None
    """
    cdef:
        Py_ssize_t i, n = values.size      # 定义变量 i 和 n，其中 n 是 values 的大小
        int64_t ordinal                    # 定义 ordinal 变量为 int64_t 类型
        object item_repr                   # 定义 item_repr 变量为 object 类型
        ndarray out = cnp.PyArray_EMPTY(   # 创建一个空的 ndarray 对象 out
            values.ndim, values.shape, cnp.NPY_OBJECT, 0
        )
        object[::1] out_flat = out.ravel() # 将 out 展平为一维数组，并赋给 out_flat
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, values)  # 创建多迭代器 mi，用于操作 out 和 values

    for i in range(n):
        # 类似于：ordinal = values[i]
        ordinal = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # 从 mi 中获取当前位置的 ordinal 值

        if ordinal == NPY_NAT:
            item_repr = na_rep            # 如果 ordinal 为 NPY_NAT，则使用 na_rep
        else:
            # 这相当于：
            # freq = frequency_corresponding_to_dtype_code(dtype_code)
            # per = Period(ordinal, freq=freq)
            # if date_format:
            #     item_repr = per.strftime(date_format)
            # else:
            #     item_repr = str(per)
            item_repr = period_format(ordinal, dtype_code, date_format)  # 使用 period_format 处理 ordinal

        # 类似于：ordinals[i] = ordinal
        out_flat[i] = item_repr           # 将 item_repr 存入 out_flat 的第 i 个位置

        cnp.PyArray_MultiIter_NEXT(mi)    # 移动 mi 到下一个位置

    return out                           # 返回最终的 out 数组
# ----------------------------------------------------------------------
# period accessors

# 定义一个 C 语言风格的函数指针类型 accessor，接受两个 int 参数并返回 int 结果，可能会抛出 INT32_MIN 异常
ctypedef int (*accessor)(int64_t ordinal, int freq) except INT32_MIN


# 定义一个 Cython 函数 pyear，用于获取给定 ordinal 和 freq 的年份信息
cdef int pyear(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return dts.year  # 返回日期结构体中的年份信息


# 定义一个 Cython 函数 pqyear，用于获取给定 ordinal 和 freq 的四分之一年信息
cdef int pqyear(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_yq(ordinal, freq, &dts)  # 调用 C 函数获取年份和四分之一年信息并填充 dts
    return dts.year  # 返回日期结构体中的年份信息


# 定义一个 Cython 函数 pquarter，用于获取给定 ordinal 和 freq 的季度信息
cdef int pquarter(int64_t ordinal, int freq):
    cdef:
        int quarter  # 声明一个 int 类型变量 quarter
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    quarter = get_yq(ordinal, freq, &dts)  # 调用 C 函数获取年份和四分之一年信息并填充 dts，并将季度信息赋值给 quarter
    return quarter  # 返回季度信息


# 定义一个 Cython 函数 pmonth，用于获取给定 ordinal 和 freq 的月份信息
cdef int pmonth(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return dts.month  # 返回日期结构体中的月份信息


# 定义一个 Cython 函数 pday，用于获取给定 ordinal 和 freq 的日信息
cdef int pday(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return dts.day  # 返回日期结构体中的日信息


# 定义一个 Cython 函数 pweekday，用于获取给定 ordinal 和 freq 的星期几信息
cdef int pweekday(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return dayofweek(dts.year, dts.month, dts.day)  # 返回日期结构体中的星期几信息


# 定义一个 Cython 函数 pday_of_year，用于获取给定 ordinal 和 freq 的一年中的第几天信息
cdef int pday_of_year(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return get_day_of_year(dts.year, dts.month, dts.day)  # 返回日期结构体中的一年中的第几天信息


# 定义一个 Cython 函数 pweek，用于获取给定 ordinal 和 freq 的一年中的第几周信息
cdef int pweek(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return get_week_of_year(dts.year, dts.month, dts.day)  # 返回日期结构体中的一年中的第几周信息


# 定义一个 Cython 函数 phour，用于获取给定 ordinal 和 freq 的小时信息
cdef int phour(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return dts.hour  # 返回日期结构体中的小时信息


# 定义一个 Cython 函数 pminute，用于获取给定 ordinal 和 freq 的分钟信息
cdef int pminute(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return dts.min  # 返回日期结构体中的分钟信息


# 定义一个 Cython 函数 psecond，用于获取给定 ordinal 和 freq 的秒钟信息
cdef int psecond(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return <int>dts.sec  # 返回日期结构体中的秒钟信息，将其转换为整数


# 定义一个 Cython 函数 pdays_in_month，用于获取给定 ordinal 和 freq 的月份中的天数信息
cdef int pdays_in_month(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts  # 声明一个 npy_datetimestruct 结构体变量 dts
    get_date_info(ordinal, freq, &dts)  # 调用 C 函数获取日期信息并填充 dts
    return get_days_in_month(dts.year, dts.month)  # 返回日期结构体中指定月份的天数信息


# 定义一个 Cython 函数 get_period_field_arr，用于根据给定的字段名、数组和频率获取对应的时间段字段数组
@cython.wraparound(False)  # 关闭负数索引的包装检查
@cython.boundscheck(False)  # 关闭数组边界检查
def get_period_field_arr(str field, const int64_t[:] arr, int freq):
    cdef:
        Py_ssize_t i, sz  # 声明 Py_ssize_t 类型的变量 i 和 sz
        int64_t[::1] out  # 声明一个一维连续数组 out，存储 int64_t 类型数据

    func = _get_accessor_func(field)  # 调用 _get_accessor_func 函数获取指定字段名的访问函数
    if func is NULL:
        raise ValueError(f"Unrecognized field name: {field}")  # 如果获取的函数指针为空，则抛出 ValueError 异常

    sz = len(arr)  # 获取数组 arr 的长度
    out = np.empty(sz, dtype=np.int64)  # 创建一个长度为 sz 的空数组 out，数据类型为 np.int64

    for i in range(sz):  # 遍历数组 arr
        if arr[i] == NPY_NAT:  # 如果数组元素为 NPY_NAT（表示不可用的时间戳）
            out[i] = -1  # 将 out 数组中对应位置置为 -1
    # 如果字段为 "hour"，返回对应的小时访问器函数
    elif field == "hour":
        return <accessor>phour
    # 如果字段为 "minute"，返回对应的分钟访问器函数
    elif field == "minute":
        return <accessor>pminute
    # 如果字段为 "second"，返回对应的秒钟访问器函数
    elif field == "second":
        return <accessor>psecond
    # 如果字段为 "week"，返回对应的周访问器函数
    elif field == "week":
        return <accessor>pweek
    # 如果字段为 "day_of_year"，返回对应的年内第几天访问器函数
    elif field == "day_of_year":
        return <accessor>pday_of_year
    # 如果字段为 "weekday" 或 "day_of_week"，返回对应的周几访问器函数
    elif field == "weekday" or field == "day_of_week":
        return <accessor>pweekday
    # 如果字段为 "days_in_month"，返回对应的月份天数访问器函数
    elif field == "days_in_month":
        return <accessor>pdays_in_month
    # 如果以上条件都不满足，返回空值
    return NULL
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义从整数数组提取序数的函数，接受整数数组和频率作为参数
def from_ordinals(const int64_t[:] values, freq):
    cdef:
        Py_ssize_t i, n = len(values)  # 声明并初始化循环索引和数组长度
        int64_t[::1] result = np.empty(len(values), dtype="i8")  # 初始化结果数组
        int64_t val  # 声明变量用于存储当前值

    freq = to_offset(freq, is_period=True)  # 转换频率参数为BaseOffset对象
    if not isinstance(freq, BaseOffset):  # 检查频率是否为BaseOffset的实例，如果不是则抛出异常
        raise ValueError("freq not specified and cannot be inferred")

    for i in range(n):  # 循环遍历输入的整数数组
        val = values[i]  # 获取当前索引处的值
        if val == NPY_NAT:  # 如果值为NPY_NAT
            result[i] = NPY_NAT  # 结果数组对应位置也设为NPY_NAT
        else:
            result[i] = Period(val, freq=freq).ordinal  # 否则将值转换为Period对象并获取其ordinal属性

    return result.base  # 返回结果数组的底层数组


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义从对象数组中提取序数的函数，接受对象数组和频率作为参数，返回一个整数数组
def extract_ordinals(ndarray values, freq) -> np.ndarray:
    # values 是对象类型的ndarray，可能是二维数组

    cdef:
        Py_ssize_t i, n = values.size  # 声明并初始化循环索引和数组长度
        int64_t ordinal  # 声明变量用于存储序数
        ndarray ordinals = cnp.PyArray_EMPTY(  # 初始化存储序数的数组
            values.ndim, values.shape, cnp.NPY_INT64, 0
        )
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(ordinals, values)  # 创建多迭代器对象
        object p  # 声明变量用于存储当前对象

    if values.descr.type_num != cnp.NPY_OBJECT:  # 检查输入数组的数据类型是否为对象类型，如果不是则抛出类型错误
        raise TypeError("extract_ordinals values must be object-dtype")

    freqstr = PeriodDtypeBase(freq._period_dtype_code, freq.n)._freqstr  # 获取频率的字符串表示

    for i in range(n):  # 循环遍历输入数组
        # 类似于: p = values[i]
        p = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # 获取当前对象

        ordinal = _extract_ordinal(p, freqstr, freq)  # 调用辅助函数提取当前对象的序数

        # 类似于: ordinals[i] = ordinal
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ordinal  # 将序数存入结果数组中

        cnp.PyArray_MultiIter_NEXT(mi)  # 移动多迭代器到下一个元素

    return ordinals  # 返回存储序数的数组


cdef int64_t _extract_ordinal(object item, str freqstr, freq) except? -1:
    """
    See extract_ordinals.
    """
    cdef:
        int64_t ordinal  # 声明变量用于存储序数

    if checknull_with_nat(item) or item is C_NA:  # 检查对象是否为特殊值或C_NA
        ordinal = NPY_NAT  # 如果是，则序数为NPY_NAT
    elif util.is_integer_object(item):  # 检查对象是否为整数对象
        if item == NPY_NAT:  # 如果对象为NPY_NAT
            ordinal = NPY_NAT  # 序数为NPY_NAT
        else:
            raise TypeError(item)  # 否则抛出类型错误
    else:
        try:
            ordinal = item.ordinal  # 尝试获取对象的ordinal属性

            if item.freqstr != freqstr:  # 检查对象的频率字符串是否与给定频率字符串不同
                msg = DIFFERENT_FREQ.format(cls="PeriodIndex",
                                            own_freq=freqstr,
                                            other_freq=item.freqstr)
                raise IncompatibleFrequency(msg)  # 如果不同，抛出不兼容频率的异常

        except AttributeError:  # 如果对象没有ordinal属性
            item = Period(item, freq=freq)  # 创建Period对象
            if item is NaT:  # 如果对象为NaT
                # 输入可能包含类似NaT的字符串
                ordinal = NPY_NAT  # 序数为NPY_NAT
            else:
                ordinal = item.ordinal  # 否则获取Period对象的序数

    return ordinal  # 返回序数


# 定义从对象数组中提取频率的函数，接受对象数组作为参数，返回BaseOffset对象
def extract_freq(ndarray[object] values) -> BaseOffset:
    # TODO: Change type to const object[:] when Cython supports that.

    cdef:
        Py_ssize_t i, n = len(values)  # 声明并初始化循环索引和数组长度
        object value  # 声明变量用于存储当前对象

    for i in range(n):  # 循环遍历输入数组
        value = values[i]  # 获取当前索引处的对象

        if is_period_object(value):  # 检查对象是否为Period对象
            return value.freq  # 返回对象的频率

    raise ValueError("freq not specified and cannot be inferred")  # 如果未找到频率信息，抛出值错误异常
# -----------------------------------------------------------------------
# period helpers
# 这部分是关于周期的辅助功能

DIFFERENT_FREQ = ("Input has different freq={other_freq} "
                  "from {cls}(freq={own_freq})")
# 当输入频率与预期不匹配时的错误消息模板，包含外部频率和当前对象频率的占位符

class IncompatibleFrequency(ValueError):
    # 自定义异常类，用于表示频率不兼容的错误
    pass

cdef class PeriodMixin:
    # Methods shared between Period and PeriodArray
    # Period 和 PeriodArray 共享的方法集合

    @property
    def start_time(self) -> Timestamp:
        """
        Get the Timestamp for the start of the period.

        Returns
        -------
        Timestamp
        返回周期开始的时间戳

        See Also
        --------
        Period.end_time : Return the end Timestamp.
        Period.dayofyear : Return the day of year.
        Period.daysinmonth : Return the days in that month.
        Period.dayofweek : Return the day of the week.

        Examples
        --------
        >>> period = pd.Period('2012-1-1', freq='D')
        >>> period
        Period('2012-01-01', 'D')

        >>> period.start_time
        Timestamp('2012-01-01 00:00:00')

        >>> period.end_time
        Timestamp('2012-01-01 23:59:59.999999999')
        """
        return self.to_timestamp(how="start")
        # 调用自身的 to_timestamp 方法，返回周期开始时间戳

    @property
    def end_time(self) -> Timestamp:
        """
        Get the Timestamp for the end of the period.

        Returns
        -------
        Timestamp
        返回周期结束的时间戳

        See Also
        --------
        Period.start_time : Return the start Timestamp.
        Period.dayofyear : Return the day of year.
        Period.daysinmonth : Return the days in that month.
        Period.dayofweek : Return the day of the week.

        Examples
        --------
        For Period:

        >>> pd.Period('2020-01', 'D').end_time
        Timestamp('2020-01-01 23:59:59.999999999')

        For Series:

        >>> period_index = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')
        >>> s = pd.Series(period_index)
        >>> s
        0   2020-01
        1   2020-02
        2   2020-03
        dtype: period[M]
        >>> s.dt.end_time
        0   2020-01-31 23:59:59.999999999
        1   2020-02-29 23:59:59.999999999
        2   2020-03-31 23:59:59.999999999
        dtype: datetime64[ns]

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.end_time
        DatetimeIndex(['2023-01-31 23:59:59.999999999',
                       '2023-02-28 23:59:59.999999999',
                       '2023-03-31 23:59:59.999999999'],
                       dtype='datetime64[ns]', freq=None)
        """
        return self.to_timestamp(how="end")
        # 调用自身的 to_timestamp 方法，返回周期结束时间戳
    # 要求当前对象和另一个对象具有相同的频率
    def _require_matching_freq(self, other: BaseOffset, bint base=False):
        # 查看 arrays.period.raise_on_incompatible 以获取更多信息
        # 根据 base 参数确定是否比较基础频率
        if base:
            condition = self.freq.base != other.base
        else:
            condition = self.freq != other

        # 如果条件不满足，则抛出异常
        if condition:
            # 获取当前对象的频率字符串
            freqstr = PeriodDtypeBase(
                self.freq._period_dtype_code, self.freq.n
            )._freqstr
            # 获取另一个对象的频率字符串
            if hasattr(other, "_period_dtype_code"):
                other_freqstr = PeriodDtypeBase(
                    other._period_dtype_code, other.n
                )._freqstr
            else:
                other_freqstr = other.freqstr
            # 构建异常消息
            msg = DIFFERENT_FREQ.format(
                cls=type(self).__name__,
                own_freq=freqstr,
                other_freq=other_freqstr,
            )
            # 抛出不兼容频率异常
            raise IncompatibleFrequency(msg)
    # 定义名为 _Period 的 Cython 类，继承自 PeriodMixin
    cdef class _Period(PeriodMixin):

        # 定义只读属性 ordinal 和 _dtype，以及 freq 属性
        cdef readonly:
            int64_t ordinal
            PeriodDtypeBase _dtype
            BaseOffset freq

        # 设置此类的数组优先级高于 np.ndarray、np.matrix 和 np.timedelta64
        __array_priority__ = 100

        # 将 _Period 的 day_of_week 方法赋值给 dayofweek
        dayofweek = _Period.day_of_week
        # 将 _Period 的 day_of_year 方法赋值给 dayofyear
        dayofyear = _Period.day_of_year

        # 构造函数，初始化 ordinal 和 freq 属性
        def __cinit__(self, int64_t ordinal, BaseOffset freq):
            self.ordinal = ordinal
            self.freq = freq
            # 注意：这种方法比 PeriodDtype.from_date_offset(freq) 更高效，
            # 因为 from_date_offset 无法成为 cdef 方法（直到 Cython 支持 cdef classmethods）
            self._dtype = PeriodDtypeBase(freq._period_dtype_code, freq.n)

        @classmethod
        def _maybe_convert_freq(cls, object freq) -> BaseOffset:
            """
            A Period's freq attribute must have `freq.n > 0`, which we check for here.

            Returns
            -------
            DateOffset
            """
            # 将 freq 转换为偏移量对象，并确保 freq.n 大于 0
            freq = to_offset(freq, is_period=True)

            if freq.n <= 0:
                raise ValueError("Frequency must be positive, because it "
                                 f"represents span: {freq.freqstr}")

            return freq

        @classmethod
        def _from_ordinal(cls, ordinal: int64_t, freq: BaseOffset) -> "Period":
            """
            Fast creation from an ordinal and freq that are already validated!
            """
            # 如果 ordinal 等于 NPY_NAT，则返回 NaT
            if ordinal == NPY_NAT:
                return NaT
            else:
                # 调用 _maybe_convert_freq 方法确保 freq 符合要求
                freq = cls._maybe_convert_freq(freq)
                # 使用提供的 ordinal 和 freq 创建并返回 _Period 实例
                self = _Period.__new__(cls, ordinal, freq)
                return self

        # 富比较方法，用于比较两个 Period 对象
        def __richcmp__(self, other, op):
            if is_period_object(other):
                # 如果 other 是 Period 对象
                if other._dtype != self._dtype:
                    # 如果 other 的 _dtype 不等于 self 的 _dtype
                    if op == Py_EQ:
                        return False
                    elif op == Py_NE:
                        return True
                    # 确保 other 的 freq 与 self 的 freq 匹配
                    self._require_matching_freq(other.freq)
                # 使用 PyObject_RichCompareBool 比较两个 ordinal 属性
                return PyObject_RichCompareBool(self.ordinal, other.ordinal, op)
            elif other is NaT:
                # 如果 other 是 NaT，则根据操作符返回比较结果
                return op == Py_NE
            elif util.is_array(other):
                # 如果 other 是数组
                # GH#44285
                if cnp.PyArray_IsZeroDim(other):
                    # 如果 other 是零维数组，比较 self 和 other.item()
                    return PyObject_RichCompare(self, other.item(), op)
                else:
                    # 对于 ndarray[object]，比较 self 和 other 中的每个元素
                    return np.array([PyObject_RichCompare(self, x, op) for x in other])
            # 返回未实现的结果
            return NotImplemented

        # 返回对象的哈希值，用于集合等需要哈希的数据结构
        def __hash__(self):
            return hash((self.ordinal, self.freqstr))
    # 定义一个方法用于将类中的时间增量与标量进行加法操作，并返回一个新的 Period 对象
    def _add_timedeltalike_scalar(self, other) -> "Period":
        cdef:
            int64_t inc, ordinal  # 定义两个 C 基本类型的变量

        # 如果当前 Period 对象的频率不是 tick-like 类型，则抛出异常
        if not self._dtype._is_tick_like():
            raise IncompatibleFrequency("Input cannot be converted to "
                                        f"Period(freq={self.freqstr})")

        # 如果 other 是 np.timedelta64 类型且其值为 NPY_NAT（即 "nat"），则返回 NaT
        if (
            cnp.is_timedelta64_object(other) and
            cnp.get_timedelta64_value(other) == NPY_NAT
        ):
            # 即 np.timedelta64("nat")
            return NaT

        try:
            # 尝试将 other 转换为纳秒增量，使用当前 Period 对象的分辨率
            inc = delta_to_nanoseconds(other, reso=self._dtype._creso, round_ok=False)
        except ValueError as err:
            # 若转换失败，则抛出异常
            raise IncompatibleFrequency("Input cannot be converted to "
                                        f"Period(freq={self.freqstr})") from err
        with cython.overflowcheck(True):
            # 计算新的 ordinal 值，即当前 Period 对象的 ordinal 加上增量 inc
            ordinal = self.ordinal + inc
        # 返回一个新的 Period 对象，其 ordinal 为计算后的 ordinal，频率为当前对象的频率
        return Period(ordinal=ordinal, freq=self.freq)

    # 定义一个方法用于将类中的时间偏移对象与标量进行加法操作，并返回一个新的 Period 对象
    def _add_offset(self, other) -> "Period":
        # 要求其他偏移对象的频率与当前 Period 对象的频率匹配
        self._require_matching_freq(other, base=True)

        # 计算新的 ordinal 值，即当前 Period 对象的 ordinal 加上偏移对象的值
        ordinal = self.ordinal + other.n
        # 返回一个新的 Period 对象，其 ordinal 为计算后的 ordinal，频率为当前对象的频率
        return Period(ordinal=ordinal, freq=self.freq)

    # 定义一个方法重载加法操作符，支持 Period 对象与不同类型的对象相加
    @cython.overflowcheck(True)
    def __add__(self, other):
        # 如果 other 是任何类型的时间增量标量，则调用 _add_timedeltalike_scalar 方法处理
        if is_any_td_scalar(other):
            return self._add_timedeltalike_scalar(other)
        # 如果 other 是 DateOffset 类型的对象，则调用 _add_offset 方法处理
        elif is_offset_object(other):
            return self._add_offset(other)
        # 如果 other 是 NaT（Not a Time）类型，则直接返回 NaT
        elif other is NaT:
            return NaT
        # 如果 other 是整数对象，则计算新的 ordinal 值并返回一个新的 Period 对象
        elif util.is_integer_object(other):
            ordinal = self.ordinal + other * self._dtype._n
            return Period(ordinal=ordinal, freq=self.freq)

        # 如果 other 是 Period 对象，则抛出类型错误异常，不支持直接相加
        elif is_period_object(other):
            # 不能将 datetime-like 对象直接添加到 Period 对象上
            # GH#17983; 不能简单返回 NotImplemented，因为会导致递归错误
            # 当通过 np.add.reduce 调用时，请参见 npdev 构建中的 TestNumpyReductions.test_add
            sname = type(self).__name__
            oname = type(other).__name__
            raise TypeError(f"unsupported operand type(s) for +: '{sname}' "
                            f"and '{oname}'")

        # 如果 other 是数组对象，则逐个元素递归调用加法操作，并返回一个数组对象
        elif util.is_array(other):
            if other.dtype == object:
                # GH#50162
                return np.array([self + x for x in other], dtype=object)

        # 如果以上条件都不满足，则返回 NotImplemented
        return NotImplemented

    # 定义一个方法用于支持反向加法操作，直接调用 __add__ 方法进行处理
    def __radd__(self, other):
        return self.__add__(other)
    # 定义减法运算符的重载方法，用于处理 Period 对象与其他对象的减法操作
    def __sub__(self, other):
        # 检查 other 是否为任意时间差标量、偏移对象或整数对象
        if (
            is_any_td_scalar(other)
            or is_offset_object(other)
            or util.is_integer_object(other)
        ):
            # 返回 self 与 -other 的加法结果
            return self + (-other)
        # 如果 other 是 Period 对象
        elif is_period_object(other):
            # 确保 self 和 other 的频率匹配
            self._require_matching_freq(other.freq)
            # GH 23915 - 由于 __add__ 方法不关心 n，因此乘以基础频率
            return (self.ordinal - other.ordinal) * self.freq.base
        # 如果 other 是 NaT（Not a Time），返回 NaT
        elif other is NaT:
            return NaT

        # 如果 other 是数组
        elif util.is_array(other):
            # 如果数组元素的数据类型是对象类型
            if other.dtype == object:
                # GH#50162
                # 返回一个由每个元素与 self 相减的结果组成的对象数组
                return np.array([self - x for x in other], dtype=object)

        # 如果无法处理，返回 Not Implemented
        return NotImplemented

    # 定义右向减法运算符的重载方法
    def __rsub__(self, other):
        # 如果 other 是 NaT，返回 NaT
        if other is NaT:
            return NaT

        # 如果 other 是数组
        elif util.is_array(other):
            # 如果数组元素的数据类型是对象类型
            if other.dtype == object:
                # GH#50162
                # 返回一个由每个元素减去 self 的结果组成的对象数组
                return np.array([x - self for x in other], dtype=object)

        # 如果无法处理，返回 Not Implemented
        return NotImplemented

    # 将 Period 对象转换为指定频率的 Period 对象
    def asfreq(self, freq, how="E") -> "Period":
        """
        Convert Period to desired frequency, at the start or end of the interval.

        Parameters
        ----------
        freq : str, BaseOffset
            The desired frequency. If passing a `str`, it needs to be a
            valid :ref:`period alias <timeseries.period_aliases>`.
        how : {'E', 'S', 'end', 'start'}, default 'end'
            Start or end of the timespan.

        Returns
        -------
        resampled : Period

        Examples
        --------
        >>> period = pd.Period('2023-1-1', freq='D')
        >>> period.asfreq('h')
        Period('2023-01-01 23:00', 'h')
        """
        # 可能将 freq 转换为频率对象
        freq = self._maybe_convert_freq(freq)
        # 确定计算的时段结束位置
        how = validate_end_alias(how)
        # 获取 self 的基础数据类型代码
        base1 = self._dtype._dtype_code
        # 获取目标频率的数据类型代码
        base2 = freq_to_dtype_code(freq)

        # 如果 how 是 "E"（end），计算结束位置的序数
        end = how == "E"
        if end:
            ordinal = self.ordinal + self._dtype._n - 1
        else:
            ordinal = self.ordinal
        # 将序数转换为指定频率的序数
        ordinal = period_asfreq(ordinal, base1, base2, end)

        # 返回一个新的 Period 对象，表示转换后的时间段
        return Period(ordinal=ordinal, freq=freq)
    def to_timestamp(self, freq=None, how="start") -> Timestamp:
        """
        Return the Timestamp representation of the Period.

        Uses the target frequency specified at the part of the period specified
        by `how`, which is either `Start` or `Finish`.

        Parameters
        ----------
        freq : str or DateOffset
            Target frequency. Default is 'D' if self.freq is week or
            longer and 'S' otherwise.
        how : str, default 'S' (start)
            One of 'S', 'E'. Can be aliased as case insensitive
            'Start', 'Finish', 'Begin', 'End'.

        Returns
        -------
        Timestamp

        Examples
        --------
        >>> period = pd.Period('2023-1-1', freq='D')
        >>> timestamp = period.to_timestamp()
        >>> timestamp
        Timestamp('2023-01-01 00:00:00')
        """
        how = validate_end_alias(how)  # 调用函数，确保how参数是有效的结束别名

        end = how == "E"  # 判断是否要求结束日期的时间戳
        if end:
            if freq == "B" or self.freq == "B":
                # 如果频率是工作日或者指定的周期是工作日，调整日期到最近的工作日
                adjust = np.timedelta64(1, "D") - np.timedelta64(1, "ns")
                return self.to_timestamp(how="start") + adjust
            # 计算期间结束的时间戳，并减去1纳秒以确保精确到期间的最后时刻
            endpoint = (self + self.freq).to_timestamp(how="start")
            return endpoint - np.timedelta64(1, "ns")

        if freq is None:
            # 如果频率未指定，默认使用基于self.freq的'D'或者更短的频率
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            # 尝试将频率转换为适当的格式
            freq = self._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        # 将期间转换为指定频率和类型的时间戳
        val = self.asfreq(freq, how)

        # 将期间的序数转换为datetime64类型，并返回对应的Timestamp对象
        dt64 = period_ordinal_to_dt64(val.ordinal, base)
        return Timestamp(dt64)

    @property
    def year(self) -> int:
        """
        Return the year this Period falls on.

        Examples
        --------
        >>> period = pd.Period('2022-01', 'M')
        >>> period.year
        2022
        """
        base = self._dtype._dtype_code
        return pyear(self.ordinal, base)  # 获取此期间所在的年份

    @property
    def month(self) -> int:
        """
        Return the month this Period falls on.

        Examples
        --------
        >>> period = pd.Period('2022-01', 'M')
        >>> period.month
        1
        """
        base = self._dtype._dtype_code
        return pmonth(self.ordinal, base)  # 获取此期间所在的月份

    @property
    def day(self) -> int:
        """
        Get day of the month that a Period falls on.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day of the week.
        Period.dayofyear : Get the day of the year.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", freq='h')
        >>> p.day
        11
        """
        base = self._dtype._dtype_code
        return pday(self.ordinal, base)  # 获取此期间所在的日期（月份中的第几天）
    def hour(self) -> int:
        """
        Get the hour of the day component of the Period.

        Returns
        -------
        int
            The hour as an integer, between 0 and 23.

        See Also
        --------
        Period.second : Get the second component of the Period.
        Period.minute : Get the minute component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.hour
        13

        Period longer than a day

        >>> p = pd.Period("2018-03-11", freq="M")
        >>> p.hour
        0
        """
        # 使用当前 Period 对象的 ordinal 值和基础值来计算小时数
        base = self._dtype._dtype_code
        return phour(self.ordinal, base)

    @property
    def minute(self) -> int:
        """
        Get minute of the hour component of the Period.

        Returns
        -------
        int
            The minute as an integer, between 0 and 59.

        See Also
        --------
        Period.hour : Get the hour component of the Period.
        Period.second : Get the second component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.minute
        3
        """
        # 使用当前 Period 对象的 ordinal 值和基础值来计算分钟数
        base = self._dtype._dtype_code
        return pminute(self.ordinal, base)

    @property
    def second(self) -> int:
        """
        Get the second component of the Period.

        Returns
        -------
        int
            The second of the Period (ranges from 0 to 59).

        See Also
        --------
        Period.hour : Get the hour component of the Period.
        Period.minute : Get the minute component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.second
        12
        """
        # 使用当前 Period 对象的 ordinal 值和基础值来计算秒数
        base = self._dtype._dtype_code
        return psecond(self.ordinal, base)

    @property
    def weekofyear(self) -> int:
        """
        Get the week of the year on the given Period.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day component of the Period.
        Period.weekday : Get the day component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", "h")
        >>> p.weekofyear
        10

        >>> p = pd.Period("2018-02-01", "D")
        >>> p.weekofyear
        5

        >>> p = pd.Period("2018-01-06", "D")
        >>> p.weekofyear
        1
        """
        # 使用当前 Period 对象的 ordinal 值和基础值来计算一年中的周数
        base = self._dtype._dtype_code
        return pweek(self.ordinal, base)
    @property
    def week(self) -> int:
        """
        Get the week of the year on the given Period.

        Returns
        -------
        int
            The week of the year.

        See Also
        --------
        Period.dayofweek : Get the day component of the Period.
        Period.weekday : Get the day component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", "h")
        >>> p.week
        10

        >>> p = pd.Period("2018-02-01", "D")
        >>> p.week
        5

        >>> p = pd.Period("2018-01-06", "D")
        >>> p.week
        1
        """
        return self.weekofyear

    @property
    def day_of_week(self) -> int:
        """
        Day of the week the period lies in, with Monday=0 and Sunday=6.

        If the period frequency is lower than daily (e.g. hourly), and the
        period spans over multiple days, the day at the start of the period is
        used.

        If the frequency is higher than daily (e.g. monthly), the last day
        of the period is used.

        Returns
        -------
        int
            Day of the week.

        See Also
        --------
        Period.day_of_week : Day of the week the period lies in.
        Period.weekday : Alias of Period.day_of_week.
        Period.day : Day of the month.
        Period.dayofyear : Day of the year.

        Examples
        --------
        >>> per = pd.Period('2017-12-31 22:00', 'h')
        >>> per.day_of_week
        6

        For periods that span over multiple days, the day at the beginning of
        the period is returned.

        >>> per = pd.Period('2017-12-31 22:00', '4h')
        >>> per.day_of_week
        6
        >>> per.start_time.day_of_week
        6

        For periods with a frequency higher than days, the last day of the
        period is returned.

        >>> per = pd.Period('2018-01', 'M')
        >>> per.day_of_week
        2
        >>> per.end_time.day_of_week
        2
        """
        base = self._dtype._dtype_code
        return pweekday(self.ordinal, base)
    def weekday(self) -> int:
        """
        返回周期所在的星期几，星期一为0，星期日为6。

        如果周期频率低于日频率（例如每小时），并且周期跨越多天，则使用周期开始时的日期。

        如果周期频率高于日频率（例如每月），则使用周期结束时的日期。

        Returns
        -------
        int
            星期几的数字表示。

        See Also
        --------
        Period.dayofweek : 周期所在的星期几。
        Period.weekday : Period.dayofweek 的别名。
        Period.day : 月份中的具体日期。
        Period.dayofyear : 年份中的具体日期。

        Examples
        --------
        >>> per = pd.Period('2017-12-31 22:00', 'h')
        >>> per.dayofweek
        6

        对于跨越多天的周期，返回周期开始时的星期几。

        >>> per = pd.Period('2017-12-31 22:00', '4h')
        >>> per.dayofweek
        6
        >>> per.start_time.dayofweek
        6

        对于频率高于天的周期，返回周期结束时的星期几。

        >>> per = pd.Period('2018-01', 'M')
        >>> per.dayofweek
        2
        >>> per.end_time.dayofweek
        2
        """
        # 文档字符串与 dayofweek 中的文档字符串重复。在 Cython 文件中无法使用
        # Appender 重复使用文档字符串，同时也无法设置 __doc__ 属性。
        return self.dayofweek

    @property
    def day_of_year(self) -> int:
        """
        返回一年中的第几天。

        该属性返回特定日期在一年中的第几天。返回值范围为1到365（常规年份）或1到366（闰年）。

        Returns
        -------
        int
            一年中的第几天。

        See Also
        --------
        Period.day : 返回月份中的具体日期。
        Period.day_of_week : 返回星期几。
        PeriodIndex.day_of_year : 返回所有索引的一年中的第几天。

        Examples
        --------
        >>> period = pd.Period("2015-10-23", freq='h')
        >>> period.day_of_year
        296
        >>> period = pd.Period("2012-12-31", freq='D')
        >>> period.day_of_year
        366
        >>> period = pd.Period("2013-01-01", freq='D')
        >>> period.day_of_year
        1
        """
        base = self._dtype._dtype_code
        return pday_of_year(self.ordinal, base)
    def quarter(self) -> int:
        """
        Return the quarter this Period falls on.

        See Also
        --------
        Timestamp.quarter : Return the quarter of the Timestamp.
        Period.year : Return the year of the period.
        Period.month : Return the month of the period.

        Examples
        --------
        >>> period = pd.Period('2022-04', 'M')
        >>> period.quarter
        2
        """
        # 获取基础数据类型的代码
        base = self._dtype._dtype_code
        # 调用底层函数 pquarter，返回当前 Period 所在的季度
        return pquarter(self.ordinal, base)

    @property
    def qyear(self) -> int:
        """
        Fiscal year the Period lies in according to its starting-quarter.

        The `year` and the `qyear` of the period will be the same if the fiscal
        and calendar years are the same. When they are not, the fiscal year
        can be different from the calendar year of the period.

        Returns
        -------
        int
            The fiscal year of the period.

        See Also
        --------
        Period.year : Return the calendar year of the period.

        Examples
        --------
        If the natural and fiscal year are the same, `qyear` and `year` will
        be the same.

        >>> per = pd.Period('2018Q1', freq='Q')
        >>> per.qyear
        2018
        >>> per.year
        2018

        If the fiscal year starts in April (`Q-MAR`), the first quarter of
        2018 will start in April 2017. `year` will then be 2017, but `qyear`
        will be the fiscal year, 2018.

        >>> per = pd.Period('2018Q1', freq='Q-MAR')
        >>> per.start_time
        Timestamp('2017-04-01 00:00:00')
        >>> per.qyear
        2018
        >>> per.year
        2017
        """
        # 获取基础数据类型的代码
        base = self._dtype._dtype_code
        # 调用底层函数 pqyear，返回当前 Period 对应的财政年度
        return pqyear(self.ordinal, base)

    @property
    def days_in_month(self) -> int:
        """
        Get the total number of days in the month that this period falls on.

        Returns
        -------
        int

        See Also
        --------
        Period.daysinmonth : Gets the number of days in the month.
        DatetimeIndex.daysinmonth : Gets the number of days in the month.
        calendar.monthrange : Returns a tuple containing weekday
            (0-6 ~ Mon-Sun) and number of days (28-31).

        Examples
        --------
        >>> p = pd.Period('2018-2-17')
        >>> p.days_in_month
        28

        >>> pd.Period('2018-03-01').days_in_month
        31

        Handles the leap year case as well:

        >>> p = pd.Period('2016-2-17')
        >>> p.days_in_month
        29
        """
        # 获取基础数据类型的代码
        base = self._dtype._dtype_code
        # 调用底层函数 pdays_in_month，返回当前 Period 所在月份的天数
        return pdays_in_month(self.ordinal, base)

    @property
    def daysinmonth(self) -> int:
        """
        Get the total number of days of the month that this period falls on.

        Returns
        -------
        int
            The number of days in the month of the period.

        See Also
        --------
        Period.days_in_month : Return the days of the month.
        Period.dayofyear : Return the day of the year.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", freq='h')
        >>> p.daysinmonth
        31
        """
        return self.days_in_month


```        
    @property
    def is_leap_year(self) -> bool:
        """
        Return True if the period's year is in a leap year.

        Returns
        -------
        bool
            True if the period's year is a leap year; False otherwise.

        See Also
        --------
        Timestamp.is_leap_year : Check if the year in a Timestamp is a leap year.
        DatetimeIndex.is_leap_year : Boolean indicator if the date belongs to a
            leap year.

        Examples
        --------
        >>> period = pd.Period('2022-01', 'M')
        >>> period.is_leap_year
        False

        >>> period = pd.Period('2020-01', 'M')
        >>> period.is_leap_year
        True
        """
        return bool(is_leapyear(self.year))



    @classmethod
    def now(cls, freq):
        """
        Return the period of now's date.

        Parameters
        ----------
        freq : str, BaseOffset
            Frequency to use for the returned period.

        Returns
        -------
        Period
            A Period object representing the current date.

        Examples
        --------
        >>> pd.Period.now('h')  # doctest: +SKIP
        Period('2023-06-12 11:00', 'h')
        """
        return Period(datetime.now(), freq=freq)



    @property
    def freqstr(self) -> str:
        """
        Return a string representation of the frequency.

        Returns
        -------
        str
            A string representing the frequency of the period.

        Examples
        --------
        >>> pd.Period('2020-01', 'D').freqstr
        'D'
        """
        freqstr = PeriodDtypeBase(self.freq._period_dtype_code, self.freq.n)._freqstr
        return freqstr



    def __repr__(self) -> str:
        """
        Return a string representation of the Period object.

        Returns
        -------
        str
            A string representation of the Period object.

        """
        base = self._dtype._dtype_code
        formatted = period_format(self.ordinal, base)
        return f"Period('{formatted}', '{self.freqstr}')"



    def __str__(self) -> str:
        """
        Return a string representation for a particular Period object.

        Returns
        -------
        str
            A string representation of the Period object.

        """
        base = self._dtype._dtype_code
        formatted = period_format(self.ordinal, base)
        value = str(formatted)
        return value



    def __setstate__(self, state):
        """
        Set the state of the Period object from the given state.

        Parameters
        ----------
        state : tuple
            A tuple containing the state information.

        """
        self.freq = state[1]
        self.ordinal = state[2]



    def __reduce__(self):
        """
        Reduce the Period object to a state that can be serialized.

        Returns
        -------
        tuple
            A tuple containing the necessary information to reconstruct the Period object.

        """
        object_state = None, self.freq, self.ordinal
        return (Period, object_state)
class Period(_Period):
    """
    Represents a period of time.

    Parameters
    ----------
    value : Period, str, datetime, date or pandas.Timestamp, default None
        The time period represented (e.g., '4Q2005'). This represents neither
        the start or the end of the period, but rather the entire period itself.
    freq : str, default None
        One of pandas period strings or corresponding objects. Accepted
        strings are listed in the
        :ref:`period alias section <timeseries.period_aliases>` in the user docs.
        If value is datetime, freq is required.
    ordinal : int, default None
        The period offset from the proleptic Gregorian epoch.
    year : int, default None
        Year value of the period.
    month : int, default 1
        Month value of the period.
    quarter : int, default None
        Quarter value of the period.
    day : int, default 1
        Day value of the period.
    hour : int, default 0
        Hour value of the period.
    minute : int, default 0
        Minute value of the period.
    second : int, default 0
        Second value of the period.

    See Also
    --------
    Timestamp : Pandas replacement for python datetime.datetime object.
    date_range : Return a fixed frequency DatetimeIndex.
    timedelta_range : Generates a fixed frequency range of timedeltas.

    Examples
    --------
    >>> period = pd.Period('2012-1-1', freq='D')
    >>> period
    Period('2012-01-01', 'D')
    """

cdef bint is_period_object(object obj):
    """
    Check if the given object is an instance of _Period.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bint
        True if obj is an instance of _Period, False otherwise.
    """
    return isinstance(obj, _Period)


cpdef int freq_to_dtype_code(BaseOffset freq) except? -1:
    """
    Convert a frequency object into its corresponding dtype code.

    Parameters
    ----------
    freq : BaseOffset
        The frequency object to convert.

    Returns
    -------
    int
        The dtype code corresponding to the frequency.

    Raises
    ------
    ValueError
        If the frequency object does not have a _period_dtype_code attribute.
    """
    try:
        return freq._period_dtype_code
    except AttributeError as err:
        raise ValueError(INVALID_FREQ_ERR_MSG.format(freq)) from err


cdef int64_t _ordinal_from_fields(int year, int month, quarter, int day,
                                  int hour, int minute, int second,
                                  BaseOffset freq):
    """
    Calculate the ordinal value from individual time period fields.

    Parameters
    ----------
    year : int
        Year value.
    month : int
        Month value.
    quarter : int
        Quarter value (optional).
    day : int
        Day value.
    hour : int
        Hour value.
    minute : int
        Minute value.
    second : int
        Second value.
    freq : BaseOffset
        The frequency object for period calculation.

    Returns
    -------
    int64_t
        The ordinal value representing the time period.
    """
    base = freq_to_dtype_code(freq)
    if quarter is not None:
        year, month = quarter_to_myear(year, quarter, freq.freqstr)

    return period_ordinal(year, month, day, hour,
                          minute, second, 0, 0, base)


def validate_end_alias(how: str) -> str:  # Literal["E", "S"]
    """
    Validate and normalize the 'how' parameter for end alias.

    Parameters
    ----------
    how : str
        The alias indicating the end ('E') or start ('S').

    Returns
    -------
    str
        The normalized alias ('S' for start or 'E' for end).

    Raises
    ------
    ValueError
        If 'how' is not one of {'S', 'E'}.
    """
    how_dict = {"S": "S", "E": "E",
                "START": "S", "FINISH": "E",
                "BEGIN": "S", "END": "E"}
    how = how_dict.get(str(how).upper())
    if how not in {"S", "E"}:
        raise ValueError("How must be one of S or E")
    return how


cdef _parse_weekly_str(value, BaseOffset freq):
    """
    Parse a weekly period string of the format "2017-01-23/2017-01-29".

    Parameters
    ----------
    value : str
        The weekly period string to parse.
    freq : BaseOffset
        The frequency object for period parsing.

    Notes
    -----
    This function handles parsing of weekly period strings that cannot be
    parsed by general datetime logic.

    Raises
    ------
    ValueError
        If 'value' is not in the expected format.
    """
    # GH#50803
    start, end = value.split("/")
    start = Timestamp(start)
    end = Timestamp(end)
    # 如果日期范围不是6天（一周的长度），则抛出数值错误异常
    if (end - start).days != 6:
        # 我们只关注周期为一周的情况
        raise ValueError("Could not parse as weekly-freq Period")

    # 如果频率参数为None，则根据结束日期获取其对应的周几缩写作为频率字符串
    if freq is None:
        # 获取结束日期的周几名称的前三个字母，并转换为大写
        day_name = end.day_name()[:3].upper()
        # 构建以周几开头的频率字符串，如"W-MON"
        freqstr = f"W-{day_name}"
        # 使用频率字符串创建频率对象，并指定其为周期
        freq = to_offset(freqstr, is_period=True)
        # 应确保 freq 对象能够对应到结束日期上

    # 返回结束日期和频率对象作为结果
    return end, freq
```