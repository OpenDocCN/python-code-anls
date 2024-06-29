# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\parsing.pyx`

```
"""
Parsing functions for datetime and datetime-like strings.
"""
# 导入所需的模块和库
import re  # 导入正则表达式模块
import time  # 导入时间模块
import warnings  # 导入警告模块

from pandas.util._exceptions import find_stack_level  # 从 pandas 库中导入异常处理函数

# 从 Cython 编译模块中导入日期时间相关的类和函数
from cpython.datetime cimport (
    datetime,
    datetime_new,
    import_datetime,
    timedelta,
    tzinfo,
)

from datetime import timezone  # 从标准库 datetime 中导入时区相关模块

# 从 Cython 编译模块中导入 Unicode 处理函数和数据类型
from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from cython cimport Py_ssize_t
from libc.string cimport strchr

import_datetime()  # 调用 Cython 编译模块中的日期时间导入函数

import numpy as np  # 导入 NumPy 库

cimport numpy as cnp  # 使用 Cython 加速导入 NumPy 模块
from numpy cimport int64_t  # 从 NumPy 中导入特定整数类型

cnp.import_array()  # 调用 Cython 加速 NumPy 数组导入函数

# dateutil 兼容性

# 从 decimal 模块中导入异常处理类
from decimal import InvalidOperation

# 从 dateutil 库中导入日期时间解析器和时区相关模块
from dateutil.parser import DEFAULTPARSER
from dateutil.tz import (
    tzoffset,
    tzutc as _dateutil_tzutc,
)

# 从 pandas 库中导入配置管理函数
from pandas._config import get_option

# 从 pandas 库中导入时间序列日历相关的 Cython 编译模块
from pandas._libs.tslibs.ccalendar cimport MONTH_TO_CAL_NUM
# 从 pandas 库中导入时间序列数据类型相关的 Cython 编译模块
from pandas._libs.tslibs.dtypes cimport (
    attrname_to_npy_unit,
    npy_unit_to_attrname,
)
# 从 pandas 库中导入时间序列缺失值类型相关的 Cython 编译模块
from pandas._libs.tslibs.nattype cimport (
    c_NaT as NaT,
    c_nat_strings as nat_strings,
)

# 从 pandas 库中导入时间序列日期时间相关的 Cython 编译模块
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

# 从 pandas 库中导入时间序列日期时间相关的 Cython 编译模块
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    import_pandas_datetime,
    npy_datetimestruct,
    string_to_dts,
)

import_pandas_datetime()  # 调用 Cython 编译模块中的 Pandas 日期时间导入函数

# 从 pandas 库中导入时间序列格式化解析函数
from pandas._libs.tslibs.strptime import array_strptime

# 从 pandas 的 C 扩展头文件中导入函数声明
cdef extern from "pandas/portable.h":
    int getdigit_ascii(char c, int default) nogil

cdef extern from "pandas/parser/tokenizer.h":
    double xstrtod(const char *p, char **q, char decimal, char sci, char tsep,
                   int skip_trailing, int *error, int *maybe_int)


# ----------------------------------------------------------------------
# 常量定义

# 自定义异常类，用于日期解析错误
class DateParseError(ValueError):
    pass

# 默认的 datetime 对象，表示最小日期时间
_DEFAULT_DATETIME = datetime(1, 1, 1).replace(hour=0, minute=0,
                                              second=0, microsecond=0)

cdef:
    # 非日期时间类字符串的集合，用于解析过程中的排除
    set _not_datelike_strings = {"a", "A", "m", "M", "p", "P", "t", "T"}

    # 被视为时间戳单位的集合，将其四舍五入到纳秒
    set _timestamp_units = {
        NPY_DATETIMEUNIT.NPY_FR_ns,
        NPY_DATETIMEUNIT.NPY_FR_ps,
        NPY_DATETIMEUNIT.NPY_FR_fs,
        NPY_DATETIMEUNIT.NPY_FR_as,
    }

# ----------------------------------------------------------------------
cdef:
    # 日期分隔符的字符串常量
    const char* delimiters = " /-."
    # 每月最大天数和最大月份常量
    int MAX_DAYS_IN_MONTH = 31, MAX_MONTH = 12


cdef bint _is_delimiter(const char ch):
    # 函数用于判断给定字符是否为日期分隔符
    return strchr(delimiters, ch) != NULL

# 解析单个数字字符的函数，返回整数结果
cdef int _parse_1digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 1
    return result

# 解析两位数字字符的函数，返回整数结果
cdef int _parse_2digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 10
    result += getdigit_ascii(s[1], -100) * 1
    return result

# 解析四位数字字符的函数，返回整数结果
cdef int _parse_4digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 1000
    result += getdigit_ascii(s[1], -100) * 100
    result += getdigit_ascii(s[2], -1000) * 10
    result += getdigit_ascii(s[3], -10000) * 1
    return result
# 解析特殊日期格式的函数：MM/DD/YYYY、DD/MM/YYYY、MM/YYYY。
# 函数首先尝试解析 MM/DD/YYYY 格式的日期，但如果月份大于12则尝试解析 DD/MM/YYYY（当 `dayfirst == False` 时）。
# 如果 `dayfirst == True`，函数会尝试解析 DD/MM/YYYY 格式的日期，如果解析错误则尝试解析 DD/MM/YYYY 格式的日期。

# 对于 MM/DD/YYYY、DD/MM/YYYY 格式：分隔符可以是空格或者 /- 中的一个。
# 对于 MM/YYYY 格式：分隔符可以是空格或者 /- 中的一个。
# 如果无法将 `date_string` 转换为日期，则函数返回 None, None。

cdef datetime _parse_delimited_date(
    str date_string, bint dayfirst, NPY_DATETIMEUNIT* out_bestunit
):
    """
    Parse special cases of dates: MM/DD/YYYY, DD/MM/YYYY, MM/YYYY.

    At the beginning function tries to parse date in MM/DD/YYYY format, but
    if month > 12 - in DD/MM/YYYY (`dayfirst == False`).
    With `dayfirst == True` function makes an attempt to parse date in
    DD/MM/YYYY, if an attempt is wrong - in DD/MM/YYYY

    For MM/DD/YYYY, DD/MM/YYYY: delimiter can be a space or one of /-.
    For MM/YYYY: delimiter can be a space or one of /-
    If `date_string` can't be converted to date, then function returns
    None, None

    Parameters
    ----------
    date_string : str
    dayfirst : bool
    out_bestunit : NPY_DATETIMEUNIT*
        For specifying identified resolution.

    Returns:
    --------
    datetime or None
    """
    
    cdef:
        const char* buf
        Py_ssize_t length
        int day = 1, month = 1, year
        bint can_swap = 0

    buf = PyUnicode_AsUTF8AndSize(date_string, &length)
    if length == 10 and _is_delimiter(buf[2]) and _is_delimiter(buf[5]):
        # 解析 MM?DD?YYYY 和 DD?MM?YYYY 日期
        month = _parse_2digit(buf)
        day = _parse_2digit(buf + 3)
        year = _parse_4digit(buf + 6)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 9 and _is_delimiter(buf[1]) and _is_delimiter(buf[4]):
        # 解析 M?DD?YYYY 和 D?MM?YYYY 日期
        month = _parse_1digit(buf)
        day = _parse_2digit(buf + 2)
        year = _parse_4digit(buf + 5)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 9 and _is_delimiter(buf[2]) and _is_delimiter(buf[4]):
        # 解析 MM?D?YYYY 和 DD?M?YYYY 日期
        month = _parse_2digit(buf)
        day = _parse_1digit(buf + 3)
        year = _parse_4digit(buf + 5)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 8 and _is_delimiter(buf[1]) and _is_delimiter(buf[3]):
        # 解析 M?D?YYYY 和 D?M?YYYY 日期
        month = _parse_1digit(buf)
        day = _parse_1digit(buf + 2)
        year = _parse_4digit(buf + 4)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 7 and _is_delimiter(buf[2]):
        # 解析 MM?YYYY 日期
        if buf[2] == b".":
            # 我们无法确定例如 10.2010 是浮点数还是日期，因此拒绝在此处解析它
            return None
        month = _parse_2digit(buf)
        year = _parse_4digit(buf + 3)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
    else:
        return None

    if month < 0 or day < 0 or year < 1000:
        # 某些部分不是整数，因此无法将 `date_string` 转换为日期，以上述格式
        return None
    # 检查日期的有效性：月份和日期必须在有效范围内，并且在特定条件下允许月份和日期互换
    if 1 <= month <= MAX_DAYS_IN_MONTH and 1 <= day <= MAX_DAYS_IN_MONTH \
            and (month <= MAX_MONTH or day <= MAX_MONTH):
        # 如果月份大于最大月份或者在 dayfirst 为真且可以交换日期时，交换月份和日期
        if (month > MAX_MONTH or (day <= MAX_MONTH and dayfirst)) and can_swap:
            day, month = month, day
        # 在 Python <= 3.6.0 中，C API 没有对无效日期范围进行检查，因此在 3.6.1 或更新版本中调用更快的 C 版本
        return datetime_new(year, month, day, 0, 0, 0, 0, None)

    # 抛出日期解析错误，指出指定的日期无效
    raise DateParseError(f"Invalid date specified ({month}/{day})")
cdef bint _does_string_look_like_time(str parse_string):
    """
    Checks whether given string is a time: it has to start either from
    H:MM or from HH:MM, and hour and minute values must be valid.

    Parameters
    ----------
    parse_string : str
        The string to be checked for time format.

    Returns
    -------
    bool
        True if the string is potentially a time, False otherwise.
    """
    cdef:
        const char* buf  # C-style string pointer
        Py_ssize_t length  # Length of the string
        int hour = -1, minute = -1  # Initialized hour and minute values

    buf = PyUnicode_AsUTF8AndSize(parse_string, &length)  # Convert Python string to UTF-8 C string
    if length >= 4:
        if buf[1] == b":":
            # h:MM format
            hour = getdigit_ascii(buf[0], -1)  # Extract hour from the first character
            minute = _parse_2digit(buf + 2)  # Extract minute from characters after ':'
        elif buf[2] == b":":
            # HH:MM format
            hour = _parse_2digit(buf)  # Extract hour from the first two characters
            minute = _parse_2digit(buf + 3)  # Extract minute from characters after ':'

    return 0 <= hour <= 23 and 0 <= minute <= 59  # Return True if hour and minute are valid


def py_parse_datetime_string(
    str date_string, bint dayfirst=False, bint yearfirst=False
):
    """
    Parse a datetime string using Python-accessible function for testing purposes.

    Parameters
    ----------
    date_string : str
        The datetime string to parse.
    dayfirst : bool, optional
        Whether the day comes first in the date string (default is False).
    yearfirst : bool, optional
        Whether the year comes first in the date string (default is False).

    Returns
    -------
    datetime
        Parsed datetime object.

    Notes
    -----
    This function is used for testing purposes and delegates parsing to the C function.
    """
    cdef:
        NPY_DATETIMEUNIT out_bestunit  # Unit of the best resolution of parsed datetime
        int64_t nanos  # Nanoseconds component of parsed datetime

    return parse_datetime_string(
        date_string, dayfirst, yearfirst, &out_bestunit, &nanos
    )


cdef datetime parse_datetime_string(
    # NB: This will break with np.str_ (GH#32264) even though
    #  isinstance(npstrobj, str) evaluates to True, so caller must ensure
    #  the argument is *exactly* 'str'
    str date_string,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit,
    int64_t* nanos,
):
    """
    Parse datetime string and return a datetime object.

    Parameters
    ----------
    date_string : str
        The datetime string to parse.
    dayfirst : bool
        Whether the day comes first in the date string.
    yearfirst : bool
        Whether the year comes first in the date string.
    out_bestunit : pointer to NPY_DATETIMEUNIT
        Pointer to store the best resolution unit of parsed datetime.
    nanos : pointer to int64_t
        Pointer to store the nanoseconds component of parsed datetime.

    Returns
    -------
    datetime
        Parsed datetime object.

    Raises
    ------
    ValueError
        If the provided date string does not appear to be a datetime.

    Notes
    -----
    Does not handle "today" or "now", which caller must handle separately.
    """
    cdef:
        datetime dt  # Datetime object to return
        bint is_quarter = 0  # Flag indicating if the parsed date is a quarter

    if not _does_string_look_like_datetime(date_string):
        raise ValueError(f'Given date string "{date_string}" not likely a datetime')

    if _does_string_look_like_time(date_string):
        # time without date e.g. "01:01:01.111"
        # use current datetime as default, not pass _DEFAULT_DATETIME
        default = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dt = dateutil_parse(date_string, default=default,
                            dayfirst=dayfirst, yearfirst=yearfirst,
                            ignoretz=False, out_bestunit=out_bestunit, nanos=nanos)
        return dt

    dt = _parse_delimited_date(date_string, dayfirst, out_bestunit)
    if dt is not None:
        return dt

    try:
        dt = _parse_dateabbr_string(
            date_string, _DEFAULT_DATETIME, None, out_bestunit, &is_quarter
        )
        return dt
    except DateParseError:
        raise
    except ValueError:
        pass  # Fall through to return None (implicit in Python)
    # 使用 dateutil_parse 函数解析日期字符串并返回对应的 datetime 对象
    dt = dateutil_parse(date_string, default=_DEFAULT_DATETIME,
                        dayfirst=dayfirst, yearfirst=yearfirst,
                        ignoretz=False, out_bestunit=out_bestunit, nanos=nanos)
    # 返回解析后的 datetime 对象
    return dt
def parse_datetime_string_with_reso(
    str date_string, str freq=None, dayfirst=None, yearfirst=None
):
    # NB: This will break with np.str_ (GH#45580) even though
    #  isinstance(npstrobj, str) evaluates to True, so caller must ensure
    #  the argument is *exactly* 'str'
    """
    Try hard to parse datetime string, leveraging dateutil plus some extra
    goodies like quarter recognition.

    Parameters
    ----------
    date_string : str
        待解析的日期时间字符串
    freq : str or None, default None
        如果提供，帮助解释时间字符串，对应于 `offset.rule_code`
    dayfirst : bool, default None
        如果为 None，则使用 print_config 中的默认设置
    yearfirst : bool, default None
        如果为 None，则使用 print_config 中的默认设置

    Returns
    -------
    datetime
        解析后的日期时间对象
    str
        描述解析后字符串的分辨率

    Raises
    ------
    ValueError
        当预检查表明字符串不是日期时间时
    DateParseError
        在 dateutil 中出现错误
    """

    if dayfirst is None:
        dayfirst = get_option("display.date_dayfirst")
    if yearfirst is None:
        yearfirst = get_option("display.date_yearfirst")

    cdef:
        datetime parsed
        str reso
        bint string_to_dts_failed
        npy_datetimestruct dts
        NPY_DATETIMEUNIT out_bestunit
        int out_local = 0
        int out_tzoffset
        tzinfo tz
        bint is_quarter = 0

    if not _does_string_look_like_datetime(date_string):
        raise ValueError(f'Given date string "{date_string}" not likely a datetime')

    # Try iso8601 first, as it handles nanoseconds
    string_to_dts_failed = string_to_dts(
        date_string, &dts, &out_bestunit, &out_local,
        &out_tzoffset, False
    )
    if not string_to_dts_failed:
        # Match Timestamp and drop picoseconds, femtoseconds, attoseconds
        # The new resolution will just be nano
        # GH#50417
        if out_bestunit in _timestamp_units:
            out_bestunit = NPY_DATETIMEUNIT.NPY_FR_ns

        if out_bestunit == NPY_DATETIMEUNIT.NPY_FR_ns:
            # TODO: avoid circular import
            from pandas import Timestamp
            parsed = Timestamp(date_string)
        else:
            if out_local:
                tz = timezone(timedelta(minutes=out_tzoffset))
            else:
                tz = None
            parsed = datetime_new(
                dts.year, dts.month, dts.day, dts.hour, dts.min, dts.sec, dts.us, tz
            )

        reso = npy_unit_to_attrname[out_bestunit]
        return parsed, reso

    parsed = _parse_delimited_date(date_string, dayfirst, &out_bestunit)
    if parsed is not None:
        reso = npy_unit_to_attrname[out_bestunit]
        return parsed, reso

    try:
        parsed = _parse_dateabbr_string(
            date_string, _DEFAULT_DATETIME, freq, &out_bestunit, &is_quarter
        )
    except DateParseError:
        raise
    except ValueError:
        pass
    else:
        # 如果不满足第一个条件，则执行以下代码块
        if is_quarter:
            # 如果 is_quarter 为真，则将 reso 设置为 "quarter"
            reso = "quarter"
        else:
            # 否则，根据 out_bestunit 查找对应的属性名称赋给 reso
            reso = npy_unit_to_attrname[out_bestunit]
        # 返回解析后的日期时间对象和 reso 变量
        return parsed, reso

    # 调用 dateutil_parse 函数解析日期字符串，返回解析后的日期时间对象
    parsed = dateutil_parse(date_string, _DEFAULT_DATETIME,
                            dayfirst=dayfirst, yearfirst=yearfirst,
                            ignoretz=False, out_bestunit=&out_bestunit, nanos=NULL)
    # 根据 out_bestunit 查找对应的属性名称赋给 reso
    reso = npy_unit_to_attrname[out_bestunit]
    # 返回解析后的日期时间对象和 reso 变量
    return parsed, reso
# 检查给定的字符串是否可能表示日期时间：字符串必须以'0'开头或大于等于1000
cpdef bint _does_string_look_like_datetime(str py_string):
    """
    Checks whether given string is a datetime: it has to start with '0' or
    be greater than 1000.

    Parameters
    ----------
    py_string: str
        The string to check.

    Returns
    -------
    bool
        Whether given string is potentially a datetime.
    """
    cdef:
        const char *buf         # 指向字符串的UTF-8表示及其长度的缓冲区
        char *endptr = NULL     # 用于xstrtod函数的结束指针
        Py_ssize_t length = -1  # 字符串长度
        double converted_date   # 转换后的日期数值
        char first              # 字符串的第一个字符
        int error = 0           # xstrtod函数的错误指示符

    buf = PyUnicode_AsUTF8AndSize(py_string, &length)  # 将Python字符串转换为UTF-8表示
    if length >= 1:
        first = buf[0]
        if first == b"0":
            # 以'0'开头的字符串更符合日期格式而不是数值
            return True
        elif py_string in _not_datelike_strings:
            return False
        else:
            # 调用xstrtod函数以模拟Python的float转换行为
            # 例如，" 35.e-1 " 是有效的字符串
            # 设置正确的xstrtod调用参数：b'.' - 小数点作为分隔符, b'e' - 指数形式的浮点数, b'\0' - 不使用千位分隔符, 1 - 跳过前后的额外空格
            converted_date = xstrtod(buf, &endptr,
                                     b".", b"e", b"\0", 1, &error, NULL)
            # 如果没有错误并且整行解析完成，则判断是否大于等于1000
            if error == 0 and endptr == buf + length:
                return converted_date >= 1000

    return True
    # 检查日期字符串长度是否在4到7之间
    if 4 <= date_len <= 7:
        # 将date_string转换为UTF-8编码的缓冲区buf，并返回buf的长度到date_len
        buf = PyUnicode_AsUTF8AndSize(date_string, &date_len)
        try:
            # 在字符串date_string中查找第一个"Q"，位置范围在索引1到6之间
            i = date_string.index("Q", 1, 6)
            if i == 1:
                # 如果"Q"在索引1处，解析季度部分，如int(date_string[0])
                quarter = _parse_1digit(buf)  # 例如 int(date_string[0])
                # 根据date_len的长度判断年份的解析方式
                if date_len == 4 or (date_len == 5
                                     and date_string[i + 1] == "-"):
                    # 匹配形如r'(\d)Q-?(\d\d')的年份格式
                    year = 2000 + int(date_string[-2:])
                elif date_len == 6 or (date_len == 7
                                       and date_string[i + 1] == "-"):
                    # 匹配形如r'(\d)Q-?(\d\d\d\d')的年份格式
                    year = int(date_string[-4:])
                else:
                    raise ValueError
            elif i == 2 or i == 3:
                # 匹配形如r'(\d\d)-?Q(\d)'的年份格式
                if date_len == 4 or (date_len == 5
                                     and date_string[i - 1] == "-"):
                    # 解析季度部分，如int(date_string[-1])
                    quarter = _parse_1digit(buf + date_len - 1)
                    # 解析年份，如2000 + int(date_string[:2])
                    year = 2000 + int(date_string[:2])
                else:
                    raise ValueError
            elif i == 4 or i == 5:
                if date_len == 6 or (date_len == 7
                                     and date_string[i - 1] == "-"):
                    # 匹配形如r'(\d\d\d\d)-?Q(\d)'的年份格式
                    # 解析季度部分，如int(date_string[-1])
                    quarter = _parse_1digit(buf + date_len - 1)
                    # 解析年份，如int(date_string[:4])
                    year = int(date_string[:4])
                else:
                    raise ValueError

            # 检查解析得到的季度是否在1到4之间
            if not (1 <= quarter <= 4):
                raise DateParseError(f"Incorrect quarterly string is given, "
                                     f"quarter must be "
                                     f"between 1 and 4: {date_string}")

            try:
                # 使用给定的年份、季度和频率freq计算年份和月份
                year, month = quarter_to_myear(year, quarter, freq)
            except KeyError:
                raise DateParseError("Unable to retrieve month "
                                     "information from given "
                                     f"freq: {freq}")

            # 使用计算得到的年份和月份构造并返回日期对象ret
            ret = default.replace(year=year, month=month)
            # 将输出最佳单位设置为月份，因为不能完全匹配NPY_FR_Q
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
            # 将is_quarter设置为1，表示这是一个季度数据
            is_quarter[0] = 1
            return ret

        except DateParseError:
            # 如果解析日期时遇到DateParseError，则向上抛出
            raise
        except ValueError:
            # 如果解析日期时遇到其他ValueError，如"Q"不在date_string中，则不做处理
            pass
    # 如果日期字符串长度为6且频率为"ME"（月度），则执行以下操作
    if date_len == 6 and freq == "ME":
        # 提取年份和月份
        year = int(date_string[:4])
        month = int(date_string[4:6])
        try:
            # 用提取的年份和月份替换默认日期
            ret = default.replace(year=year, month=month)
            # 将时间单位设置为月
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
            return ret
        except ValueError as err:
            # 如果替换失败，则抛出值错误，并附带错误信息
            raise ValueError(f"Unable to parse {date_string}") from err

    # 尝试使用不同的日期格式进行解析
    for pat in ["%Y-%m", "%b %Y", "%b-%Y"]:
        try:
            # 使用当前格式尝试解析日期字符串
            ret = datetime.strptime(date_string, pat)
            # 将时间单位设置为月
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
            return ret
        except ValueError:
            # 如果解析失败，则继续尝试下一个格式
            pass

    # 如果所有尝试都失败，则抛出值错误，指示无法解析日期字符串
    raise ValueError(f"Unable to parse {date_string}")
# 定义一个 Cython 函数，将季度和频率转换为对应的日历年和月份
cpdef quarter_to_myear(int year, int quarter, str freq):
    """
    A quarterly frequency defines a "year" which may not coincide with
    the calendar-year.  Find the calendar-year and calendar-month associated
    with the given year and quarter under the `freq`-derived calendar.

    Parameters
    ----------
    year : int
        年份
    quarter : int
        季度，范围为 1 到 4
    freq : str or None
        频率字符串或 None

    Returns
    -------
    year : int
        日历年份
    month : int
        日历月份

    See Also
    --------
    Period.qyear
    """
    if quarter <= 0 or quarter > 4:
        raise ValueError("Quarter must be 1 <= q <= 4")

    if freq is not None:
        # 根据频率确定的月份，计算对应的月数
        mnum = MONTH_TO_CAL_NUM[get_rule_month(freq)]
        month = (mnum + (quarter - 1) * 3) % 12 + 1
        if month > mnum:
            year -= 1
    else:
        # 没有指定频率时，按照季度计算月份
        month = (quarter - 1) * 3 + 1

    return year, month


# 定义一个 Cython 函数，用于解析日期时间字符串
cdef datetime dateutil_parse(
    str timestr,
    datetime default,
    bint ignoretz,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit,
    int64_t* nanos,
):
    """ lifted from dateutil to get resolution"""

    cdef:
        str attr  # 属性名称
        datetime ret  # 返回的日期时间对象
        object res  # 解析的结果对象
        str reso = None  # 解析出的日期时间分辨率，默认为 None
        dict repl = {}  # 存储要替换的日期时间属性字典

    try:
        # 使用 dateutil 的默认解析器解析时间字符串
        res, _ = DEFAULTPARSER._parse(timestr, dayfirst=dayfirst, yearfirst=yearfirst)
    except InvalidOperation:
        # dateutil 可能会引发 decimal.InvalidOperation 异常
        res = None

    if res is None:
        # 如果解析结果为空，抛出解析错误异常
        raise DateParseError(
            f"Unknown datetime string format, unable to parse: {timestr}"
        )

    # 遍历解析结果的各个属性，构建替换字典和确定解析的最精确的日期时间属性
    for attr in ["year", "month", "day", "hour",
                 "minute", "second", "microsecond"]:
        value = getattr(res, attr)
        if value is not None:
            repl[attr] = value
            reso = attr

    if reso is None:
        # 如果未能解析出有效的日期时间属性，抛出解析错误异常
        raise DateParseError(f"Unable to parse datetime string: {timestr}")

    # 如果解析出的是微秒级别且微秒数能被 1000 整除，则尝试确定更精确的日期时间分辨率
    if reso == "microsecond" and repl["microsecond"] % 1000 == 0:
        reso = _find_subsecond_reso(timestr, nanos=nanos)

    try:
        # 使用解析出的属性替换默认的日期时间对象属性，构建返回的日期时间对象
        ret = default.replace(**repl)
    except ValueError as err:
        # 如果替换属性时出现 ValueError 异常，例如 "day is out of range for month"，则抛出与 dateutil 相匹配的异常消息
        raise DateParseError(str(err) + ": " + timestr) from err
    except OverflowError as err:
        # 如果在尝试转换日期时间时发生 OverflowError 异常，例如尝试将年份转换为 datetime 时发生溢出，则抛出溢出日期时间异常
        raise OutOfBoundsDatetime(
            f'Parsing "{timestr}" to datetime overflows'
        ) from err

    if res.weekday is not None and not res.day:
        # GH#52659
        # 如果解析出的日期时间包含了星期信息但未包含日期信息，则抛出不支持的值异常
        raise ValueError(
            "Parsing datetimes with weekday but no day information is "
            "not supported"
        )
    `
        # 如果不忽略时区信息
        if not ignoretz:
            # 如果结果对象包含时区名称，并且该名称在标准时间名称列表中
            if res.tzname and res.tzname in time.tzname:
                # GH#50791
                # 如果时区名称不是"UTC"
                if res.tzname != "UTC":
                    # 抛出值错误异常，说明不再支持将'{res.tzname}'解析为tzlocal（依赖于系统时区）。
                    # 建议在构造时使用'tz'关键字参数或在构造后调用tz_localize。
                    raise ValueError(
                        f"Parsing '{res.tzname}' as tzlocal (dependent on system timezone) "
                        "is no longer supported. Pass the 'tz' "
                        "keyword or call tz_localize after construction instead",
                    )
                # 将结果对象中的时区信息替换为UTC
                ret = ret.replace(tzinfo=timezone.utc)
            # 如果结果对象的时区偏移为0
            elif res.tzoffset == 0:
                # 将结果对象中的时区信息替换为_dateutil_tzutc()返回的时区信息
                ret = ret.replace(tzinfo=_dateutil_tzutc())
            # 如果结果对象的时区偏移不为0
            elif res.tzoffset:
                # 将结果对象中的时区信息替换为使用res.tzname和res.tzoffset创建的时区信息
                ret = ret.replace(tzinfo=tzoffset(res.tzname, res.tzoffset))
    
                # dateutil可能会返回具有超出(-24h, 24h)范围的tzoffset的datetime对象，这是无效的
                # （可以构造，但如果调用str(ret)会引发异常）。检查并在必要时引发异常。
                try:
                    ret.utcoffset()
                except ValueError as err:
                    # 时区偏移必须严格在-timedelta(hours=24)和timedelta(hours=24)之间
                    raise ValueError(
                        f'Parsed string "{timestr}" gives an invalid tzoffset, '
                        "which must be between -timedelta(hours=24) and timedelta(hours=24)"
                    )
            # 如果结果对象的时区名称不为None
            elif res.tzname is not None:
                # 例如"1994 Jan 15 05:16 FOO"中的FOO未被识别为有效时区名
                # GH#18702, # GH 50235 enforced in 3.0
                raise ValueError(
                    f'Parsed string "{timestr}" included an un-recognized timezone '
                    f'"{res.tzname}".'
                )
    
        # 将out_bestunit列表的第一个元素设置为attrname_to_npy_unit[reso]的值
        out_bestunit[0] = attrname_to_npy_unit[reso]
        # 返回处理后的结果对象ret
        return ret
# 编译正则表达式，用于匹配时间字符串中的子秒精度部分
cdef object _reso_pattern = re.compile(r"\d:\d{2}:\d{2}\.(?P<frac>\d+)")

# 在时间字符串中查找子秒精度，更新传入的纳秒指针
cdef _find_subsecond_reso(str timestr, int64_t* nanos):
    # GH#55737
    # 检查 H:M:S.f 模式中的尾随零
    match = _reso_pattern.search(timestr)
    if not match:
        reso = "second"  # 如果没有匹配到子秒精度，则为秒级精度
    else:
        frac = match.groupdict()["frac"]
        if len(frac) <= 3:
            reso = "millisecond"  # 如果小数部分长度不超过3，则为毫秒级精度
        elif len(frac) > 6:
            if frac[6:] == "0" * len(frac[6:]):
                # 特殊情况，未丢失数据时，为纳秒级精度
                reso = "nanosecond"
            elif len(frac) <= 9:
                reso = "nanosecond"  # 小数部分长度不超过9，则为纳秒级精度
                if nanos is not NULL:
                    if len(frac) < 9:
                        frac = frac + "0" * (9 - len(frac))
                    nanos[0] = int(frac[6:])  # 更新纳秒部分
            else:
                # TODO: 在高于纳秒级的情况下，是否应该发出警告/抛出异常？
                reso = "nanosecond"
                if nanos is not NULL:
                    nanos[0] = int(frac[6:9])  # 更新纳秒部分
        else:
            reso = "microsecond"  # 其他情况为微秒级精度
    return reso


# ----------------------------------------------------------------------
# 用于类型推断的解析


def try_parse_dates(object[:] values, parser) -> np.ndarray:
    cdef:
        Py_ssize_t i, n
        object[::1] result

    n = len(values)
    result = np.empty(n, dtype="O")

    for i in range(n):
        if values[i] == "":
            result[i] = np.nan
        else:
            result[i] = parser(values[i])

    return result.base  # .base 用于访问底层的 ndarray


# ----------------------------------------------------------------------
# 杂项


# 从 https://github.com/dateutil/dateutil/pull/732 完全复制的类
#
# 该类用于解析和标记日期字符串。然而，由于它是 dateutil 库中的私有类，
# 依赖于向后兼容性并不切实际。事实上，使用此类会发出警告 (参见 gh-21322)。
# 因此，我们将该类移植过来以解决这两个问题。
#
# 许可证详见 LICENSES/DATEUTIL_LICENSE
class _timelex:
    def __init__(self, instream):
        if getattr(instream, "decode", None) is not None:
            instream = instream.decode()

        if isinstance(instream, str):
            self.stream = instream
        elif getattr(instream, "read", None) is None:
            raise TypeError(
                "Parser must be a string or character stream, not "
                f"{type(instream).__name__}")
        else:
            self.stream = instream.read()
    def get_tokens(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.
        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        # 声明一个变量 n，类型为 Py_ssize_t
        cdef:
            Py_ssize_t n

        # 将字符串流中的空字节替换为空字符串
        stream = self.stream.replace("\x00", "")

        # 使用正则表达式查找所有的 tokens
        tokens = re.findall(r"\s|"
                            r"(?<![\.\d])\d+\.\d+(?![\.\d])"
                            r"|\d+"
                            r"|[a-zA-Z]+"
                            r"|[\./:]+"
                            r"|[^\da-zA-Z\./:\s]+", stream)

        # 对于形如 ["59", ",", "456"] 的 token 元组重新组合，因为在这种上下文中 "," 被视为小数点
        # (例如在 Python 默认的日志格式中)
        for n, token in enumerate(tokens[:-2]):
            # 为了匹配 ,-decimal 行为而采用的把戏；最好在后续处理中完成这一步骤，简化标记化过程
            if (token is not None and token.isdigit() and
                    tokens[n + 1] == "," and tokens[n + 2].isdigit()):
                # 由于在循环过程中 tokens[n + 1] 和 tokens[n + 2] 可能被替换，所以需要检查是否为 None
                # TODO: 我真的不喜欢在这里编造值
                tokens[n] = token + "." + tokens[n + 2]
                tokens[n + 1] = None
                tokens[n + 2] = None

        # 去除 tokens 中的 None 值
        tokens = [x for x in tokens if x is not None]
        # 返回最终的 tokens 列表
        return tokens

    @classmethod
    def split(cls, s):
        # 使用类方法 cls 的实例化对象，并调用 get_tokens 方法返回 tokens 列表
        return cls(s).get_tokens()
# 将 _timelex.split 赋值给 _DATEUTIL_LEXER_SPLIT，以备后用
_DATEUTIL_LEXER_SPLIT = _timelex.split

# 猜测给定日期时间字符串的日期时间格式
def guess_datetime_format(dt_str: str, bint dayfirst=False) -> str | None:
    """
    猜测给定日期时间字符串的日期时间格式。

    这个函数尝试推断给定日期时间字符串的格式。在日期时间格式未知且需要正确解析的情况下非常有用。
    函数不能保证总是返回一个格式。

    Parameters
    ----------
    dt_str : str
        要猜测格式的日期时间字符串。
    dayfirst : bool, 默认 False
        如果为 True，则解析日期时将日排在前面，例如 20/01/2005。

        .. warning::
            dayfirst=True 不是严格的，但会倾向于首选日排在前面（这是已知的 bug）。

    Returns
    -------
    str or None : 返回 datetime 格式字符串（用于 `strftime` 或 `strptime`），
                  如果无法猜测则返回 None。

    See Also
    --------
    to_datetime : 将参数转换为 datetime。
    Timestamp : Pandas 替代 Python datetime.datetime 对象。
    DatetimeIndex : 不可变的类似 ndarray 的 datetime64 数据。

    Examples
    --------
    >>> from pandas.tseries.api import guess_datetime_format
    >>> guess_datetime_format('09/13/2023')
    '%m/%d/%Y'

    >>> guess_datetime_format('2023|September|13')
    """
    # 定义 NPY_DATETIMEUNIT 类型的变量 out_bestunit
    cdef:
        NPY_DATETIMEUNIT out_bestunit

    # 表示包含 "day" 属性的元组，以及其对应的格式字符串 "%d"，没有填充
    day_attribute_and_format = (("day",), "%d", 2)

    # 日期时间属性到格式字符串的映射列表
    datetime_attrs_to_format = [
        (("year", "month", "day", "hour", "minute", "second"), "%Y%m%d%H%M%S", 0),
        (("year", "month", "day", "hour", "minute"), "%Y%m%d%H%M", 0),
        (("year", "month", "day", "hour"), "%Y%m%d%H", 0),
        (("year", "month", "day"), "%Y%m%d", 0),
        (("hour", "minute", "second"), "%H%M%S", 0),
        (("year",), "%Y", 0),
        (("hour", "minute"), "%H%M", 0),
        (("month",), "%B", 0),
        (("month",), "%b", 0),
        (("month",), "%m", 2),
        day_attribute_and_format,
        (("hour",), "%H", 2),
        (("minute",), "%M", 2),
        (("second",), "%S", 2),
        (("second", "microsecond"), "%S.%f", 0),
        (("tzinfo",), "%z", 0),
        (("tzinfo",), "%Z", 0),
        (("day_of_week",), "%a", 0),
        (("day_of_week",), "%A", 0),
        (("meridiem",), "%p", 0),
    ]

    # 如果 dayfirst 为 True，则从列表中移除原有的 day_attribute_and_format，并将其插入到第一个位置
    if dayfirst:
        datetime_attrs_to_format.remove(day_attribute_and_format)
        datetime_attrs_to_format.insert(0, day_attribute_and_format)

    # 使用此默认值而不是 dateutil 的默认值 `datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)`
    # 因为后者在二月二十九日会导致问题。
    default = datetime(1970, 1, 1)

    # 尝试解析 dt_str 为 datetime 对象，使用 dateutil_parse 函数
    parsed_datetime = dateutil_parse(
        dt_str,
        default=default,
        dayfirst=dayfirst,
        yearfirst=False,
        ignoretz=False,
        out_bestunit=&out_bestunit,
        nanos=NULL,
    )
    except (ValueError, OverflowError, InvalidOperation):
        # 如果无法解析日期时间，返回 None
        # 无法猜测其格式
        return None

    if parsed_datetime is None:
        # 如果解析后的日期时间为 None，则返回 None
        return None

    # 使用 dateutil 中的 _DATEUTIL_LEXER_SPLIT 函数将日期时间字符串拆分为 tokens
    tokens = _DATEUTIL_LEXER_SPLIT(dt_str)

    # 规范化 tokens 中的偏移部分
    # 时区偏移有多种格式
    # 为了使最终步骤中 `strftime` 的输出能与 tokens 合并后的格式匹配，
    # tokens 中的偏移部分必须与 '%z' 格式（如 '+0900'）匹配，而不是 '+09:00'。
    if parsed_datetime.tzinfo is not None:
        offset_index = None
        if len(tokens) > 0 and tokens[-1] == "Z":
            # 最后一个 'Z' 表示零时区偏移
            offset_index = -1
        elif len(tokens) > 1 and tokens[-2] in ("+", "-"):
            # 例如 [..., '+', '0900']
            offset_index = -2
        elif len(tokens) > 3 and tokens[-4] in ("+", "-"):
            # 例如 [..., '+', '09', ':', '00']
            offset_index = -4

        if offset_index is not None:
            # 如果输入字符串具有像 '+0900' 这样的时区偏移，
            # 则偏移被分成两个 tokens，例如 ['+', '0900’]。
            # 这种分割会阻止后续处理正确解析时区格式。
            # 因此，除了格式规范化外，在这里我们重新连接它们。
            try:
                tokens[offset_index] = parsed_datetime.strftime("%z")
            except ValueError:
                # 在 du_parse 中可能不会引发无效的偏移
                # https://github.com/dateutil/dateutil/issues/188
                return None
            tokens = tokens[:offset_index + 1 or None]

    format_guess = [None] * len(tokens)
    found_attrs = set()

    for attrs, attr_format, padding in datetime_attrs_to_format:
        # 如果某个给定属性已放置在格式字符串中，则跳过
        # 其他该属性的格式（例如，月份可以以多种不同方式表示）
        if set(attrs) & found_attrs:
            continue

        if parsed_datetime.tzinfo is None and attr_format in ("%Z", "%z"):
            continue

        # 将解析后的日期时间转换为指定格式 attr_format
        parsed_formatted = parsed_datetime.strftime(attr_format)
        for i, token_format in enumerate(format_guess):
            # 填充 token 并与解析后的格式进行比较
            token_filled = _fill_token(tokens[i], padding)
            if token_format is None and token_filled == parsed_formatted:
                # 如果匹配，则更新 format_guess 和 tokens
                format_guess[i] = attr_format
                tokens[i] = token_filled
                found_attrs.update(attrs)
                break

    # 只有当我们有年、月和日时才考虑它是有效的猜测
    # 对于 %Y 和 %Y-%m（仅限 `-` 分隔符），它们符合 ISO8601 标准，我们可以做出例外。
    # 检查是否找到了 "year", "month", "day" 这三个属性，并且猜测的日期格式不是 ["%Y"]，且不是 ("%Y", None, "%m") 并且 tokens[1] 是 "-"
    if (
        len({"year", "month", "day"} & found_attrs) != 3
        and format_guess != ["%Y"]
        and not (
            format_guess == ["%Y", None, "%m"] and tokens[1] == "-"
        )
    ):
        return None

    # 初始化一个空的输出格式列表
    output_format = []
    # 遍历猜测的格式列表
    for i, guess in enumerate(format_guess):
        if guess is not None:
            # 如果猜测的格式不为 None，则将其添加到输出格式列表中
            output_format.append(guess)
        else:
            # 如果猜测的格式为 None，则处理对应的 tokens 元素
            try:
                # 如果 tokens 元素可以转换为浮点数，则说明我们的猜测可能错误，返回 None
                float(tokens[i])
                return None
            except ValueError:
                pass

            # 将 tokens 元素添加到输出格式列表中
            output_format.append(tokens[i])

    # 如果输出格式列表中同时包含 "%p" 和 "%H"，则将 "%H" 替换为 "%I"
    if "%p" in output_format and "%H" in output_format:
        i = output_format.index("%H")
        output_format[i] = "%I"

    # 将输出格式列表转换为字符串，形成最终猜测的日期时间格式
    guessed_format = "".join(output_format)

    try:
        # 尝试使用猜测的格式解析日期时间字符串 dt_str
        array_strptime(np.asarray([dt_str], dtype=object), guessed_format)
    except ValueError:
        # 如果解析失败，则返回 None
        return None

    # 重新构建字符串 dt_str，捕获任何推断的填充
    dt_str = "".join(tokens)
    # 如果解析后的日期时间字符串与原始字符串 dt_str 相等，则进行一些可能的警告操作，并返回猜测的格式
    if parsed_datetime.strftime(guessed_format) == dt_str:
        _maybe_warn_about_dayfirst(guessed_format, dayfirst)
        return guessed_format
    else:
        # 如果不相等，则返回 None
        return None
cdef str _fill_token(token: str, padding: int):
    # 如果 token 不包含小数点，例如：98
    cdef str token_filled
    if re.search(r"\d+\.\d+", token) is None:
        # 使用 zfill 方法填充 token，确保长度为 padding
        token_filled = token.zfill(padding)
    else:
        # 如果 token 包含小数点，例如：00.123
        seconds, nanoseconds = token.split(".")
        seconds = f"{int(seconds):02d}"
        # 右侧填充以获得纳秒，然后只取前6位数字（微秒），因为标准库 datetime 不支持纳秒
        nanoseconds = nanoseconds.ljust(9, "0")[:6]
        token_filled = f"{seconds}.{nanoseconds}"
    return token_filled


cdef void _maybe_warn_about_dayfirst(format: str, bint dayfirst) noexcept:
    """如果推测的日期时间格式不符合 dayfirst 参数，则发出警告。"""
    cdef:
        int day_index = format.find("%d")
        int month_index = format.find("%m")

    if (day_index != -1) and (month_index != -1):
        if (day_index > month_index) and dayfirst:
            # 如果 dayfirst 为 True，且格式中的 day_index 大于 month_index，则发出警告
            warnings.warn(
                f"Parsing dates in {format} format when dayfirst=True was specified. "
                "Pass `dayfirst=False` or specify a format to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        if (day_index < month_index) and not dayfirst:
            # 如果 dayfirst 为 False，且格式中的 day_index 小于 month_index，则发出警告
            warnings.warn(
                f"Parsing dates in {format} format when dayfirst=False (the default) "
                "was specified. "
                "Pass `dayfirst=True` or specify a format to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )


cpdef str get_rule_month(str source):
    """
    根据给定频率字符串返回起始月份，默认为 December。

    Parameters
    ----------
    source : str
        派生自 `freq.rule_code` 或 `freq.freqstr`。

    Returns
    -------
    rule_month: str

    Examples
    --------
    >>> get_rule_month('D')
    'DEC'

    >>> get_rule_month('A-JAN')
    'JAN'
    """
    source = source.upper()
    if "-" not in source:
        # 如果 source 中不包含 '-'，返回默认月份 'DEC'
        return "DEC"
    else:
        # 如果 source 中包含 '-'，返回分割后的第二部分作为月份
        return source.split("-")[1]
```