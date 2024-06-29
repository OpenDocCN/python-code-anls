# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\strptime.pyx`

```
# 引入标准库中的部分功能和类
"""Strptime-related classes and functions.

TimeRE, _calc_julian_from_U_or_W are vendored
from the standard library, see
https://github.com/python/cpython/blob/main/Lib/_strptime.py
Licence at LICENSES/PSF_LICENSE
The original module-level docstring follows.

Strptime-related classes and functions.
CLASSES:
    LocaleTime -- Discovers and stores locale-specific time information
    TimeRE -- Creates regexes for pattern matching a string of text containing
                time information
FUNCTIONS:
    _getlang -- Figure out what language is being used for the locale
    strptime -- Calculates the time struct represented by the passed-in string
"""
# 从 datetime 模块中引入 timezone 类
from datetime import timezone

# 从 cpython.datetime 模块中引入特定的 C 语言扩展功能
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    date,
    import_datetime,
    timedelta,
    tzinfo,
)

# 从 _strptime 模块中引入 TimeRE 和 _getlang 类
from _strptime import (
    TimeRE as _TimeRE,
    _getlang,
)
# 引入 _strptime 模块中的 LocaleTime 类，不进行 Cython 语法检查
from _strptime import LocaleTime  # no-cython-lint

# 调用 import_datetime 函数，执行初始化操作
import_datetime()

# 从 _thread 模块中引入 allocate_lock 函数，并重命名为 _thread_allocate_lock
from _thread import allocate_lock as _thread_allocate_lock
# 引入 re 模块，用于正则表达式操作
import re

# 引入 numpy 库，并重命名为 np
import numpy as np
# 引入 pytz 模块，用于处理时区相关操作
import pytz

# 引入 numpy 库的 C 语言扩展部分
cimport numpy as cnp
# 从 numpy 库的 C 语言扩展部分引入特定类型和数组
from numpy cimport (
    int64_t,
    ndarray,
)

# 从 pandas._libs.missing 模块的 C 语言扩展部分引入 checknull_with_nat_and_na 函数
from pandas._libs.missing cimport checknull_with_nat_and_na
# 从 pandas._libs.tslibs.conversion 模块的 C 语言扩展部分引入多个函数
from pandas._libs.tslibs.conversion cimport (
    get_datetime64_nanos,
    parse_pydatetime,
)
# 从 pandas._libs.tslibs.dtypes 模块的 C 语言扩展部分引入多个函数
from pandas._libs.tslibs.dtypes cimport (
    get_supported_reso,
    npy_unit_to_abbrev,
    npy_unit_to_attrname,
)
# 从 pandas._libs.tslibs.nattype 模块的 C 语言扩展部分引入 NPY_NAT 和 nat_strings
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_nat_strings as nat_strings,
)
# 从 pandas._libs.tslibs.np_datetime 模块的 C 语言扩展部分引入多个函数和类型
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    get_datetime64_unit,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pydate_to_dt64,
    string_to_dts,
)
# 调用 import_pandas_datetime 函数，执行初始化操作
import_pandas_datetime()

# 从 pandas._libs.tslibs.np_datetime 模块的 Python 部分引入 OutOfBoundsDatetime 异常类
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

# 从 pandas._libs.tslibs.timestamps 模块的 C 语言扩展部分引入 _Timestamp 类
from pandas._libs.tslibs.timestamps cimport _Timestamp
# 从 pandas._libs.tslibs.timezones 模块的 C 语言扩展部分引入 tz_compare 函数
from pandas._libs.tslibs.timezones cimport tz_compare
# 从 pandas._libs.util 模块的 C 语言扩展部分引入多个函数
from pandas._libs.util cimport (
    is_float_object,
    is_integer_object,
)

# 从 pandas._libs.tslibs.timestamps 模块中引入 Timestamp 类
from pandas._libs.tslibs.timestamps import Timestamp

# 从 pandas._libs.tslibs.tzconversion 模块的 C 语言扩展部分引入 tz_localize_to_utc_single 函数
from pandas._libs.tslibs.tzconversion cimport tz_localize_to_utc_single

# 调用 numpy 库的 import_array 函数，执行初始化操作
cnp.import_array()

# 定义一个 C 语言函数 format_is_iso，用于检查给定的格式字符串是否符合 ISO8601 标准
cdef bint format_is_iso(f: str):
    """
    Does format match the iso8601 set that can be handled by the C parser?
    Generally of form YYYY-MM-DDTHH:MM:SS - date separator can be different
    but must be consistent.  Leading 0s in dates and times are optional.
    """
    # 编译 ISO 8601 格式的日期时间正则表达式模式
    iso_regex = re.compile(
        r"""
        ^                     # 字符串的开始
        %Y                    # 年份
        (?:([-/ \\.]?)%m      # 月份，可带或不带分隔符
        (?: \1%d              # 日，使用与年月相同的分隔符
        (?:[ T]%H             # 小时，带分隔符
        (?:\:%M               # 分钟，带分隔符
        (?:\:%S               # 秒，带分隔符
        (?:%z|\.%f(?:%z)?     # 时区或小数秒
        )?)?)?)?)?)?          # 可选部分
        $                     # 字符串的结束
        """,
        re.VERBOSE,            # 使用 VERBOSE 模式，允许多行和注释
    )
    # 排除列表中的日期时间格式
    excluded_formats = ["%Y%m"]
    # 返回是否匹配 ISO 8601 格式，并且不在排除列表中
    return re.match(iso_regex, f) is not None and f not in excluded_formats
# 仅在测试中使用，调用 format_is_iso 函数进行 ISO 格式检查并返回结果
def _test_format_is_iso(f: str) -> bool:
    return format_is_iso(f)


# parse_today_now 函数的 Cython 版本，解析时间字符串并返回结果
cdef bint parse_today_now(
    str val, int64_t* iresult, bint utc, NPY_DATETIMEUNIT creso, bint infer_reso = False
):
    # 定义一个 Cython 结构体变量 _Timestamp ts
    cdef:
        _Timestamp ts

    if val == "now":
        # 如果输入值为 "now"
        if infer_reso:
            # 如果需要推断时间分辨率，则将 creso 设置为微秒级
            creso = NPY_DATETIMEUNIT.NPY_FR_us
        if utc:
            # 如果需要 UTC 时间，则创建当前 UTC 时间戳
            ts = <_Timestamp>Timestamp.now(timezone.utc)
            # 将时间戳按给定分辨率转换并存入 iresult 数组中
            iresult[0] = ts._as_creso(creso)._value
        else:
            # 否则，创建当前本地时间戳
            ts = <_Timestamp>Timestamp.now()
            # 将时间戳按给定分辨率转换并存入 iresult 数组中
            iresult[0] = ts._as_creso(creso)._value
        # 返回 True 表示成功解析
        return True
    elif val == "today":
        # 如果输入值为 "today"
        if infer_reso:
            # 如果需要推断时间分辨率，则将 creso 设置为微秒级
            creso = NPY_DATETIMEUNIT.NPY_FR_us
        # 创建当前本地日期时间戳
        ts = <_Timestamp>Timestamp.today()
        # 将日期时间戳按给定分辨率转换并存入 iresult 数组中
        iresult[0] = ts._as_creso(creso)._value
        # 返回 True 表示成功解析
        return True
    # 如果输入值不是 "now" 或 "today"，返回 False 表示解析失败
    return False

# 日期时间格式指令解析字典，映射字符到其在数组中的索引
cdef dict _parse_code_table = {"y": 0,
                               "Y": 1,
                               "m": 2,
                               "B": 3,
                               "b": 4,
                               "d": 5,
                               "H": 6,
                               "I": 7,
                               "M": 8,
                               "S": 9,
                               "f": 10,
                               "A": 11,
                               "a": 12,
                               "w": 13,
                               "j": 14,
                               "U": 15,
                               "W": 16,
                               "Z": 17,
                               "p": 18,  # 一个额外的键，仅在 I 中存在
                               "z": 19,
                               "G": 20,
                               "V": 21,
                               "u": 22}

# 验证日期时间格式字符串的有效性，抛出相应的 ValueError 异常
cdef _validate_fmt(str fmt):
    if "%W" in fmt or "%U" in fmt:
        if "%Y" not in fmt and "%y" not in fmt:
            raise ValueError("Cannot use '%W' or '%U' without day and year")
        if "%A" not in fmt and "%a" not in fmt and "%w" not in fmt:
            raise ValueError("Cannot use '%W' or '%U' without day and year")
    elif "%Z" in fmt and "%z" in fmt:
        raise ValueError("Cannot parse both %Z and %z")
    elif "%j" in fmt and "%G" in fmt:
        raise ValueError("Day of the year directive '%j' is not "
                         "compatible with ISO year directive '%G'. "
                         "Use '%Y' instead.")
    elif "%G" in fmt and (
        "%V" not in fmt
        or not (
            "%A" in fmt
            or "%a" in fmt
            or "%w" in fmt
            or "%u" in fmt
        )
    ):
        raise ValueError("Cannot use ISO year directive '%G' without '%V', '%A', '%a', '%w', or '%u'")
    ):
        # 如果格式字符串中包含 '%G'（ISO 年份指令），但未包含 '%V'（ISO 周指令）或者没有包含任何星期几指令（'%A', '%a', '%w', '%u'），则抛出数值错误异常
        raise ValueError("ISO year directive '%G' must be used with "
                         "the ISO week directive '%V' and a weekday "
                         "directive '%A', '%a', '%w', or '%u'.")
    elif "%V" in fmt and "%Y" in fmt:
        # 如果格式字符串中同时包含 '%V'（ISO 周指令）和 '%Y'（年份指令），则抛出数值错误异常，因为它们是不兼容的。应该使用 ISO 年份指令 '%G' 替代 '%Y'
        raise ValueError("ISO week directive '%V' is incompatible with "
                         "the year directive '%Y'. Use the ISO year "
                         "'%G' instead.")
    elif "%V" in fmt and (
        "%G" not in fmt
        or not (
            "%A" in fmt
            or "%a" in fmt
            or "%w" in fmt
            or "%u" in fmt
        )
    ):
        # 如果格式字符串中包含 '%V'（ISO 周指令），但未包含 '%G'（ISO 年份指令）或者没有包含任何星期几指令（'%A', '%a', '%w', '%u'），则抛出数值错误异常
        raise ValueError("ISO week directive '%V' must be used with "
                         "the ISO year directive '%G' and a weekday "
                         "directive '%A', '%a', '%w', or '%u'.")
# 定义一个 C 语言级别的函数，用于生成指定格式的正则表达式对象
cdef _get_format_regex(str fmt):
    # 声明两个全局变量，用于缓存时间正则表达式和普通正则表达式
    global _TimeRE_cache, _regex_cache
    # 使用线程锁确保线程安全地访问缓存
    with _cache_lock:
        # 如果当前语言环境与缓存的时间正则表达式的语言环境不一致，则重新创建 TimeRE 对象并清空正则表达式缓存
        if _getlang() != _TimeRE_cache.locale_time.lang:
            _TimeRE_cache = TimeRE()
            _regex_cache.clear()
        # 如果普通正则表达式缓存的大小超过预设的最大值，则清空缓存
        if len(_regex_cache) > _CACHE_MAX_SIZE:
            _regex_cache.clear()
        
        # 获取当前缓存的 TimeRE 对象中的语言环境
        locale_time = _TimeRE_cache.locale_time
        # 根据指定的格式 fmt 从正则表达式缓存中获取对应的正则表达式对象
        format_regex = _regex_cache.get(fmt)
        # 如果未找到对应的正则表达式对象，则尝试根据 fmt 编译新的正则表达式对象
        if not format_regex:
            try:
                format_regex = _TimeRE_cache.compile(fmt)
            except KeyError as err:
                # 当发现错误的格式指令时会抛出 KeyError，例如当格式中有 "%%" 时，转义为 "%" 后抛出
                bad_directive = err.args[0]
                if bad_directive == "\\":
                    bad_directive = "%"
                del err
                # 抛出 ValueError 异常，说明格式中包含了错误的指令
                raise ValueError(f"'{bad_directive}' is a bad directive "
                                 f"in format '{fmt}'")
            except IndexError:
                # 当格式字符串为 "%" 时会抛出 IndexError
                # 抛出 ValueError 异常，说明格式中有多余的 "%" 字符
                raise ValueError(f"stray % in format '{fmt}'")
            # 将新编译的正则表达式对象存入缓存
            _regex_cache[fmt] = format_regex
    # 返回获取到的正则表达式对象和语言环境对象
    return format_regex, locale_time


# 定义一个 C 语言级别的类，用于管理日期时间解析的状态
cdef class DatetimeParseState:
    def __cinit__(self, NPY_DATETIMEUNIT creso):
        # 初始化日期时间解析状态对象的属性
        # found_tz 和 found_naive 是关于带有时区信息和不带时区信息的 datetime/Timestamp 对象的状态标记
        self.found_tz = False
        self.found_naive = False
        # found_naive_str 指示是否找到了解析为时区无关的 datetime 字符串
        self.found_naive_str = False
        # found_aware_str 指示是否找到了解析为带时区的 datetime 字符串
        self.found_aware_str = False
        # found_other 指示是否找到了其他类型的日期时间字符串
        self.found_other = False
        
        # out_tzoffset_vals 是一个集合，用于存储解析过程中获取的时区偏移值
        self.out_tzoffset_vals = set()
        
        # creso 是一个日期时间单元类型的枚举值，用于指示解析的精度
        self.creso = creso
        # creso_ever_changed 标记是否曾经改变过解析的精度
        self.creso_ever_changed = False

    # 定义一个 C 语言级别的方法，用于更新解析的精度
    cdef bint update_creso(self, NPY_DATETIMEUNIT item_reso) noexcept:
        # 返回一个布尔值，指示是否将解析精度提升到更高的级别
        if self.creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
            self.creso = item_reso
        elif item_reso > self.creso:
            self.creso = item_reso
            self.creso_ever_changed = True
            return True
        return False
    # 处理日期时间的方法，接受一个日期时间对象 dt、时区对象 tz、以及一个布尔值 utc_convert
    cdef tzinfo process_datetime(self, datetime dt, tzinfo tz, bint utc_convert):
        # 如果输入的日期时间对象 dt 是时区感知的，则设置 found_tz 为 True
        if dt.tzinfo is not None:
            self.found_tz = True
        # 否则设置 found_naive 为 True（即 dt 是时区无关的）
        else:
            self.found_naive = True

        # 如果日期时间对象 dt 是时区感知的
        if dt.tzinfo is not None:
            # 如果需要将时区转换为 UTC，则不做处理
            if utc_convert:
                pass
            # 如果未设置将时区转换为 UTC，但输入包含了时区无关的日期时间对象，抛出 ValueError
            elif self.found_naive:
                raise ValueError("Tz-aware datetime.datetime "
                                 "cannot be converted to "
                                 "datetime64 unless utc=True")
            # 如果提供了目标时区 tz，并且输入的日期时间对象 dt 的时区与目标时区 tz 不同，抛出 ValueError
            elif tz is not None and not tz_compare(tz, dt.tzinfo):
                raise ValueError("Tz-aware datetime.datetime "
                                 "cannot be converted to "
                                 "datetime64 unless utc=True")
            # 否则，将目标时区 tz 设置为输入日期时间对象 dt 的时区
            else:
                tz = dt.tzinfo
        # 如果日期时间对象 dt 是时区无关的
        else:
            # 如果已经发现了时区感知的输入并且不打算将其转换为 UTC，抛出 ValueError
            if self.found_tz and not utc_convert:
                raise ValueError("Cannot mix tz-aware with "
                                 "tz-naive values")
        
        # 返回处理后的时区对象 tz
        return tz

    # 检查混合输入的方法，接受一个目标时区对象 tz_out 和一个布尔值 utc
    cdef tzinfo check_for_mixed_inputs(
        self,
        tzinfo tz_out,
        bint utc,
        ):
        cdef:
            bint is_same_offsets  # 定义变量，用于标识是否所有偏移量相同
            float tz_offset  # 定义变量，存储时区偏移量

        if self.found_aware_str and not utc:
            # 如果找到了带时区的字符串并且不是 UTC 模式：
            # GH#17697, GH#57275
            # 1) 如果所有的偏移量相同，则返回一个偏移量以便于传递给 DatetimeIndex
            # 2) 如果偏移量不同，则不强制解析，并引发 ValueError 异常：
            #    "cannot parse datetimes with mixed time zones unless `utc=True`"
            is_same_offsets = len(self.out_tzoffset_vals) == 1
            if not is_same_offsets or (self.found_naive or self.found_other):
                # 例如：test_to_datetime_mixed_awareness_mixed_types (array_to_datetime)
                raise ValueError(
                    "Mixed timezones detected. Pass utc=True in to_datetime "
                    "or tz='UTC' in DatetimeIndex to convert to a common timezone."
                )
            elif tz_out is not None:
                # GH#55693
                tz_offset = self.out_tzoffset_vals.pop()
                tz_out2 = timezone(timedelta(seconds=tz_offset))
                if not tz_compare(tz_out, tz_out2):
                    # 例如：(array_strptime)
                    #  test_to_datetime_mixed_offsets_with_utc_false_removed
                    # 例如：test_to_datetime_mixed_tzs_mixed_types (array_to_datetime)
                    raise ValueError(
                        "Mixed timezones detected. Pass utc=True in to_datetime "
                        "or tz='UTC' in DatetimeIndex to convert to a common timezone."
                    )
                # 例如：(array_strptime)
                #  test_guess_datetime_format_with_parseable_formats
                # 例如：test_to_datetime_mixed_types_matching_tzs (array_to_datetime)
            else:
                # 例如：test_to_datetime_iso8601_with_timezone_valid (array_strptime)
                tz_offset = self.out_tzoffset_vals.pop()
                tz_out = timezone(timedelta(seconds=tz_offset))
        elif not utc:
            if tz_out and (self.found_other or self.found_naive_str):
                # found_other 表示一个无时区的整数、浮点数、dt64 或日期
                # 例如：test_to_datetime_mixed_awareness_mixed_types (array_to_datetime)
                raise ValueError(
                    "Mixed timezones detected. Pass utc=True in to_datetime "
                    "or tz='UTC' in DatetimeIndex to convert to a common timezone."
                )
        return tz_out
# 定义一个函数用于解析包含日期时间信息的字符串数组为日期时间结构体数组
def array_strptime(
    # 参数values: 包含日期时间信息的字符串数组
    ndarray[object] values,
    # 参数fmt: 日期时间格式的字符串，支持正则表达式
    str fmt,
    # 参数exact: 如果为True，则匹配必须精确；如果为False，则可以模糊匹配
    bint exact=True,
    # 参数errors: 指定错误处理方式，{'raise', 'coerce'}之一
    errors="raise",
    # 参数utc: 如果为True，则解析为UTC时间
    bint utc=False,
    # 参数creso: NPY_DATETIMEUNIT类型，用于指定日期时间的分辨率，默认为NPY_FR_GENERIC
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_GENERIC,
):
    """
    Calculates the datetime structs represented by the passed array of strings

    Parameters
    ----------
    values : ndarray of string-like objects
        Contains strings representing dates and times.
    fmt : string-like regex
        String format, possibly containing regular expressions.
    exact : bool, default True
        If True, matches must be exact; if False, searching is allowed.
    errors : str, default 'raise'
        String specifying error handling: {'raise', 'coerce'}.
    creso : NPY_DATETIMEUNIT, default NPY_FR_GENERIC
        Set to NPY_FR_GENERIC to infer a resolution.
    """

    # 声明变量和类型
    cdef:
        # 使用Cython声明变量类型
        Py_ssize_t i, n = len(values)
        npy_datetimestruct dts
        int64_t[::1] iresult
        object val
        bint is_raise = errors=="raise"
        bint is_coerce = errors=="coerce"
        tzinfo tz, tz_out = None
        bint iso_format = format_is_iso(fmt)
        NPY_DATETIMEUNIT out_bestunit, item_reso
        int out_local = 0, out_tzoffset = 0
        bint string_to_dts_succeeded = 0
        bint infer_reso = creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        DatetimeParseState state = DatetimeParseState(creso)

    # 断言，确保错误处理方式只能是'raise'或'coerce'
    assert is_raise or is_coerce

    # 验证日期时间格式的有效性
    _validate_fmt(fmt)
    # 获取日期时间格式的正则表达式和本地时间
    format_regex, locale_time = _get_format_regex(fmt)

    # 如果需要推断分辨率
    if infer_reso:
        abbrev = "ns"
    else:
        # 否则根据指定的NPY_DATETIMEUNIT转换为对应的时间单位缩写
        abbrev = npy_unit_to_abbrev(creso)
    # 创建一个空的结果数组，根据推断的分辨率创建对应dtype的numpy数组
    result = np.empty(n, dtype=f"M8[{abbrev}]")
    iresult = result.view("i8")

    # 初始化日期时间结构体的微秒、皮秒和飞秒字段为0
    dts.us = dts.ps = dts.as = 0

    # 检查是否需要考虑时区混合输入，并进行相应的处理
    tz_out = state.check_for_mixed_inputs(tz_out, utc)

    # 如果需要推断分辨率
    if infer_reso:
        if state.creso_ever_changed:
            # 遇到不匹配的分辨率，需要重新使用正确的分辨率重新解析
            return array_strptime(
                values,
                fmt=fmt,
                exact=exact,
                errors=errors,
                utc=utc,
                creso=state.creso,
            )
        elif state.creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
            # 即我们从未遇到任何非NaT值，默认使用"s"作为分辨率，确保NaT插入和连接操作不会提升单位
            result = iresult.base.view("M8[s]")
        else:
            # 否则，我们可以使用遇到的单一分辨率，避免进行第二次解析
            abbrev = npy_unit_to_abbrev(state.creso)
            result = iresult.base.view(f"M8[{abbrev}]")
    # 返回结果数组和时区信息
    return result, tz_out


# 声明一个Cython函数，用于根据指定格式解析日期时间字符串
cdef tzinfo _parse_with_format(
    # 参数val: 待解析的日期时间字符串
    str val,
    # 参数fmt: 日期时间格式的字符串
    str fmt,
    # 参数exact: 是否需要精确匹配
    bint exact,
    # 参数format_regex: 日期时间格式的正则表达式
    format_regex,
    # 参数locale_time: 本地时间
    locale_time,
    # 参数dts: 日期时间结构体指针
    npy_datetimestruct* dts,
    # 参数item_reso: 返回的日期时间分辨率
    NPY_DATETIMEUNIT* item_reso,
):
    # 基于https://github.com/python/cpython/blob/main/Lib/_strptime.py#L293的实现
    cdef:
        int year, month, day, minute, hour, second, weekday, julian
        int week_of_year, week_of_year_start, parse_code, ordinal
        int iso_week, iso_year
        int64_t us, ns
        object found
        tzinfo tz
        dict found_dict
        str group_key, ampm

    if exact:
        # 如果需要精确匹配，则使用正则表达式的match方法匹配val和格式fmt
        found = format_regex.match(val)
        if not found:
            # 如果匹配失败，则抛出ValueError异常，提示时间数据与指定格式不匹配
            raise ValueError(
                f"time data \"{val}\" doesn't match format \"{fmt}\""
            )
        if len(val) != found.end():
            # 如果匹配成功但未完全转换整个val，则抛出ValueError异常，提示未转换的剩余数据
            raise ValueError(
                "unconverted data remains when parsing with "
                f"format \"{fmt}\": \"{val[found.end():]}\""
            )

    else:
        # 如果不需要精确匹配，则使用正则表达式的search方法查找val和格式fmt的匹配项
        found = format_regex.search(val)
        if not found:
            # 如果查找失败，则抛出ValueError异常，提示时间数据与指定格式不匹配
            raise ValueError(
                f"time data \"{val}\" doesn't match format \"{fmt}\""
            )

    item_reso[0] = NPY_DATETIMEUNIT.NPY_FR_s

    iso_year = -1
    year = 1900
    month = day = 1
    hour = minute = second = ns = us = 0
    tz = None
    # 默认为-1表示这些值未知，不过这些值不是必需的
    iso_week = week_of_year = -1
    week_of_year_start = -1
    # weekday和julian默认为-1，表示需要计算这些值
    weekday = julian = -1
    found_dict = found.groupdict()
    # 如果已知周年和周的日期，可以计算一年中的Julian日
    if julian == -1 and weekday != -1:
        if week_of_year != -1:
            # 例如，val='2013020'; fmt='%Y%U%w'
            week_starts_Mon = week_of_year_start == 0
            julian = _calc_julian_from_U_or_W(year, week_of_year, weekday,
                                              week_starts_Mon)
        elif iso_year != -1 and iso_week != -1:
            # 例如，val='2015-1-7'; fmt='%G-%V-%u'
            year, julian = _calc_julian_from_V(iso_year, iso_week,
                                               weekday + 1)
        # else:
        #    # 例如，val='Thu Sep 25 2003'; fmt='%a %b %d %Y'
        #    pass

    # 不能预先计算date()，因为在Julian计算中可能会发生变化，从而导致星期几计算结果不同
    if julian == -1:
        # 需要加1，因为一年的第一天是1而不是0
        # 实际上不需要ordinal/julian，但需要在例如val='2015-04-31'; fmt='%Y-%m-%d'时引发异常
        ordinal = date(year, month, day).toordinal()
        julian = ordinal - date(year, 1, 1).toordinal() + 1
    else:
        # 假设如果提供了Julian日，它将是准确的
        datetime_result = date.fromordinal(
            (julian - 1) + date(year, 1, 1).toordinal())
        year = datetime_result.year
        month = datetime_result.month
        day = datetime_result.day
    # 如果 weekday 值为 -1，则说明这里实际上不会使用 weekday 变量，
    # 但是需要执行这段代码以便在年/月/日组合无效时引发异常。

    # 将传入的年、月、日、时、分、秒、微秒、纳秒分配给日期时间结构体 dts 的各个成员变量。
    dts.year = year
    dts.month = month
    dts.day = day
    dts.hour = hour
    dts.min = minute
    dts.sec = second
    dts.us = us
    dts.ps = ns * 1000

    # 返回时区 tz，表示日期时间结构体已经设置完成。
    return tz
class TimeRE(_TimeRE):
    """
    处理从格式指令到正则表达式的转换。

    创建用于匹配包含时间信息的文本字符串的正则表达式。
    """

    def __init__(self, locale_time=None):
        """
        创建键/值对。

        执行顺序对于依赖关系很重要。
        """
        self._Z = None
        super().__init__(locale_time=locale_time)
        # GH 48767: 重写 cpython 的 TimeRE
        #  1) 解析到纳秒而不是微秒
        self.update({"f": r"(?P<f>[0-9]{1,9})"}),

    def __getitem__(self, key):
        if key == "Z":
            # 懒计算
            if self._Z is None:
                self._Z = self.__seqToRE(pytz.all_timezones, "Z")
            # 注意：处理 Z 是与使用标准库 _strptime.TimeRE 的关键区别。
            # 使用标准库版本时，使用 fmt='%Y-%m-%d %H:%M:%S %Z' 的测试会失败。
            return self._Z
        return super().__getitem__(key)


_cache_lock = _thread_allocate_lock()
# 不要在没有获取缓存锁的情况下修改 _TimeRE_cache 或 _regex_cache！
_TimeRE_cache = TimeRE()
_CACHE_MAX_SIZE = 5  # _regex_cache 中存储的最大正则表达式数目
_regex_cache = {}


cdef int _calc_julian_from_U_or_W(int year, int week_of_year,
                                  int day_of_week, int week_starts_Mon):
    """
    根据年份、年的周数和星期几计算儒略日。

    week_starts_Mon 表示一周的开始是星期天还是星期一（6 或 0）。

    Parameters
    ----------
    year : int
        年份
    week_of_year : int
        从格式 U 或 W 获取的周数
    week_starts_Mon : int
        表示一周的开始是星期天还是星期一（6 或 0）

    Returns
    -------
    int
        转换后的儒略日
    """

    cdef:
        int first_weekday, week_0_length, days_to_week

    first_weekday = date(year, 1, 1).weekday()
    # 如果使用 %U 指令（一周从星期天开始），则将视图简单地转移到星期天作为周的第一天。
    if not week_starts_Mon:
        first_weekday = (first_weekday + 1) % 7
        day_of_week = (day_of_week + 1) % 7

    # 需要注意第 0 周（当年的第一天与 %U 或 %W 指定的不同）。
    week_0_length = (7 - first_weekday) % 7
    if week_of_year == 0:
        return 1 + day_of_week - first_weekday
    else:
        days_to_week = week_0_length + (7 * (week_of_year - 1))
        return 1 + days_to_week + day_of_week


cdef (int, int) _calc_julian_from_V(int iso_year, int iso_week, int iso_weekday):
    """
    根据 ISO 8601 年份、周数和星期几计算儒略日。

    ISO 周从星期一开始，第 01 周包含 1 月 4 日。
    # ISO周日期范围从1（星期一）到7（星期天）。

    # 参数
    # ----------
    # iso_year : int
    #     从格式 %G 中获取的年份
    # iso_week : int
    #     从格式 %V 中获取的周数
    # iso_weekday : int
    #     从格式 %u 中获取的星期几

    # 返回
    # -------
    # (int, int)
    #     ISO年份和格里高利历的序数日期/儒略日期
    """

    cdef:
        int correction, ordinal

    # 计算修正值，用于确定ISO周的第一天是一年的第几天
    correction = date(iso_year, 1, 4).isoweekday() + 3
    # 计算ISO日期的儒略日序数
    ordinal = (iso_week * 7) + iso_weekday - correction
    # 如果序数小于1，则表示日期在前一年
    if ordinal < 1:
        # 调整ISO年份和序数，将日期转换为上一年的对应序数日期
        ordinal += date(iso_year, 1, 1).toordinal()
        iso_year -= 1
        ordinal -= date(iso_year, 1, 1).toordinal()
    return iso_year, ordinal
# 定义一个 C 扩展函数，用于解析 '%z' 指令并返回 datetime.timezone 对象
cdef tzinfo parse_timezone_directive(str z):
    """
    Parse the '%z' directive and return a datetime.timezone object.

    Parameters
    ----------
    z : string of the UTC offset

    Returns
    -------
    datetime.timezone

    Notes
    -----
    This is essentially similar to the cpython implementation
    https://github.com/python/cpython/blob/546cab84448b892c92e68d9c1a3d3b58c13b3463/Lib/_strptime.py#L437-L454
    Licence at LICENSES/PSF_LICENSE
    """

    cdef:
        int hours, minutes, seconds, pad_number, microseconds  # 定义整数变量用于存储小时、分钟、秒数、填充数和微秒
        int total_minutes  # 用于存储总分钟数
        str gmtoff_remainder, gmtoff_remainder_padding  # 字符串变量存储剩余的偏移量和填充的字符串

    if z == "Z":
        return timezone(timedelta(0))  # 如果是 "Z" 表示 UTC 时间，返回一个零时差的 timezone 对象

    if z[3] == ":":
        z = z[:3] + z[4:]  # 如果偏移量中包含冒号，去掉冗余的冒号
        if len(z) > 5:
            if z[5] != ":":
                raise ValueError(f"Inconsistent use of : in {z}")  # 如果冒号使用不一致，抛出异常
            z = z[:5] + z[6:]  # 继续修正冒号位置

    hours = int(z[1:3])  # 提取小时部分并转换为整数
    minutes = int(z[3:5])  # 提取分钟部分并转换为整数
    seconds = int(z[5:7] or 0)  # 提取秒数部分并转换为整数，如果为空则默认为 0

    # 填充以确保总是返回微秒
    gmtoff_remainder = z[8:]  # 提取剩余的偏移量字符串
    pad_number = 6 - len(gmtoff_remainder)  # 计算需要填充的位数
    gmtoff_remainder_padding = "0" * pad_number  # 生成填充字符串
    microseconds = int(gmtoff_remainder + gmtoff_remainder_padding)  # 将填充后的字符串转换为整数微秒数

    total_minutes = ((hours * 60) + minutes + (seconds // 60) +
                     (microseconds // 60_000_000))  # 计算总分钟数，包括秒和微秒的贡献
    total_minutes = -total_minutes if z.startswith("-") else total_minutes  # 如果是负偏移，则取相反数
    return timezone(timedelta(minutes=total_minutes))  # 根据总分钟数创建并返回一个 timezone 对象
```