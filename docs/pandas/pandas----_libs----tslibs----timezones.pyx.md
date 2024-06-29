# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timezones.pyx`

```
from datetime import (
    timedelta,
    timezone,
)

from pandas.compat._optional import import_optional_dependency

try:
    # 尝试导入 zoneinfo 库及其 ZoneInfo 类（仅适用于 Python 3.9+）
    import zoneinfo
    from zoneinfo import ZoneInfo
except ImportError:
    # 如果导入失败，则设置 zoneinfo 和 ZoneInfo 为 None
    zoneinfo = None
    ZoneInfo = None

from cpython.datetime cimport (
    datetime,
    timedelta,
    tzinfo,
)

# dateutil 兼容性

from dateutil.tz import (
    gettz as dateutil_gettz,  # 导入 dateutil 库中的 gettz 函数
    tzfile as _dateutil_tzfile,  # 导入 dateutil 库中的 tzfile 类
    tzlocal as _dateutil_tzlocal,  # 导入 dateutil 库中的 tzlocal 类
    tzutc as _dateutil_tzutc,  # 导入 dateutil 库中的 tzutc 类
)
import numpy as np  # 导入 numpy 库并命名为 np
import pytz  # 导入 pytz 库
from pytz.tzinfo import BaseTzInfo as _pytz_BaseTzInfo  # 从 pytz 库导入 BaseTzInfo 类并命名为 _pytz_BaseTzInfo

cimport numpy as cnp  # 导入 C 扩展模块中的 numpy
from numpy cimport int64_t  # 从 numpy 中导入 int64_t 类型

cnp.import_array()  # 导入 numpy 数组模块

# ----------------------------------------------------------------------
from pandas._libs.tslibs.util cimport (
    get_nat,  # 从 pandas 库的 C 扩展模块中导入 get_nat 函数
    is_integer_object,  # 从 pandas 库的 C 扩展模块中导入 is_integer_object 函数
)

cdef int64_t NPY_NAT = get_nat()  # 使用 get_nat 函数获取 pandas 中的缺失值常量

cdef tzinfo utc_stdlib = timezone.utc  # 创建名为 utc_stdlib 的 tzinfo 对象，代表标准 UTC 时间
cdef tzinfo utc_pytz = pytz.utc  # 创建名为 utc_pytz 的 tzinfo 对象，代表 pytz 库中的 UTC 时间
cdef tzinfo utc_dateutil_str = dateutil_gettz("UTC")  # 使用 dateutil 库的 gettz 函数创建名为 utc_dateutil_str 的 tzinfo 对象，表示字符串 "UTC" 的时区信息

cdef tzinfo utc_zoneinfo = None  # 初始化 utc_zoneinfo 变量为 None


# ----------------------------------------------------------------------

cdef bint is_utc_zoneinfo(tzinfo tz):
    # 检查是否为 zoneinfo 库中的 UTC 时区
    # 解决缺少 tzdata 的情况
    if tz is None or zoneinfo is None:
        return False

    global utc_zoneinfo
    if utc_zoneinfo is None:
        try:
            # 尝试创建名为 "UTC" 的 ZoneInfo 对象
            utc_zoneinfo = ZoneInfo("UTC")
        except zoneinfo.ZoneInfoNotFoundError:
            return False
        # 如果系统的 tzdata 版本太旧，提出警告
        import_optional_dependency("tzdata", errors="warn", min_version="2022.7")

    return tz is utc_zoneinfo


cpdef inline bint is_utc(tzinfo tz):
    # 检查是否为 UTC 时间
    return (
        tz is utc_pytz
        or tz is utc_stdlib
        or isinstance(tz, _dateutil_tzutc)
        or tz is utc_dateutil_str
        or is_utc_zoneinfo(tz)
    )


cdef bint is_zoneinfo(tzinfo tz):
    # 检查是否为 zoneinfo 库中的时区对象
    if ZoneInfo is None:
        return False
    return isinstance(tz, ZoneInfo)


cdef bint is_tzlocal(tzinfo tz):
    # 检查是否为本地时间对象
    return isinstance(tz, _dateutil_tzlocal)


cdef bint treat_tz_as_pytz(tzinfo tz):
    # 检查是否应将时区对象视为 pytz 类型
    return (hasattr(tz, "_utc_transition_times") and
            hasattr(tz, "_transition_info"))


cdef bint treat_tz_as_dateutil(tzinfo tz):
    # 检查是否应将时区对象视为 dateutil 类型
    return hasattr(tz, "_trans_list") and hasattr(tz, "_trans_idx")


# 返回字符串或时区信息对象
cpdef inline object get_timezone(tzinfo tz):
    """
    We need to do several things here:
    1) Distinguish between pytz and dateutil timezones
    2) Not be over-specific (e.g. US/Eastern with/without DST is same *zone*
       but a different tz object)
    3) Provide something to serialize when we're storing a datetime object
       in pytables.

    We return a string prefaced with dateutil if it's a dateutil tz, else just
    the tz name. It needs to be a string so that we can serialize it with
    """
    # 在此处执行几个操作：
    # 1) 区分 pytz 和 dateutil 时区
    # 2) 不要过于具体（例如，带有/不带有夏令时的 US/Eastern 是同一 *区*，但是不同的 tz 对象）
    # 3) 在将 datetime 对象存储在 pytables 中时提供序列化内容

    # 如果是 dateutil 时区，则返回以 dateutil 开头的字符串，否则返回时区名称
    return (
        "dateutil " + tz.zone if isinstance(tz, tzinfo)
        else tz  # 返回时区名称的字符串表示形式
    )
    """
    处理时区信息，返回适合处理的时区名称或标识符。
    """
    if tz is None:
        # 如果时区参数为 None，则抛出类型错误异常
        raise TypeError("tz argument cannot be None")
    if is_utc(tz):
        # 如果时区为 UTC，则直接返回该时区
        return tz
    elif is_zoneinfo(tz):
        # 如果时区是 zoneinfo 类型，则返回其键值
        return tz.key
    elif treat_tz_as_pytz(tz):
        # 如果按照 pytz 处理时区，则获取其时区名称
        zone = tz.zone
        if zone is None:
            # 如果时区名称为 None，则返回时区对象本身
            return tz
        return zone
    elif treat_tz_as_dateutil(tz):
        # 如果按照 dateutil 处理时区，则检查其文件名，构造对应的时区标识符
        if ".tar.gz" in tz._filename:
            # 如果文件名包含 ".tar.gz"，则抛出数值错误异常，提示不良的时区文件名
            raise ValueError(
                "Bad tz filename. Dateutil on python 3 on windows has a "
                "bug which causes tzfile._filename to be the same for all "
                "timezone files. Please construct dateutil timezones "
                'implicitly by passing a string like "dateutil/Europe'
                '/London" when you construct your pandas objects instead '
                "of passing a timezone object. See "
                "https://github.com/pandas-dev/pandas/pull/7362")
        # 返回按照 dateutil 处理的时区标识符
        return "dateutil/" + tz._filename
    else:
        # 如果无法识别时区类型，则直接返回传入的时区对象
        return tz
# 构建一个函数，根据输入的参数 tz，可能构造一个时区对象。如果 tz 是字符串，则使用它来构造时区对象；否则直接返回 tz。
cpdef inline tzinfo maybe_get_tz(object tz):
    if isinstance(tz, str):  # 如果 tz 是字符串
        if tz == "tzlocal()":  # 如果 tz 是 "tzlocal()"
            tz = _dateutil_tzlocal()  # 调用 _dateutil_tzlocal() 函数获取本地时区
        elif tz.startswith("dateutil/"):  # 如果 tz 以 "dateutil/" 开头
            zone = tz[9:]  # 获取时区名
            tz = dateutil_gettz(zone)  # 调用 dateutil_gettz 函数获取对应的 dateutil 时区对象
            if isinstance(tz, _dateutil_tzfile) and ".tar.gz" in tz._filename:  # 如果是 dateutil 的 tzfile 对象且文件名不正确
                tz._filename = zone  # 修正文件名
        elif tz[0] in {"-", "+"}:  # 如果 tz 以 "-" 或 "+" 开头
            hours = int(tz[0:3])  # 解析小时
            minutes = int(tz[0] + tz[4:6])  # 解析分钟
            tz = timezone(timedelta(hours=hours, minutes=minutes))  # 创建时区对象
        elif tz[0:4] in {"UTC-", "UTC+"}:  # 如果 tz 以 "UTC-" 或 "UTC+" 开头
            hours = int(tz[3:6])  # 解析小时
            minutes = int(tz[3] + tz[7:9])  # 解析分钟
            tz = timezone(timedelta(hours=hours, minutes=minutes))  # 创建时区对象
        elif tz == "UTC" or tz == "utc":  # 如果 tz 是 "UTC" 或 "utc"
            tz = utc_stdlib  # 设置为标准库中的 UTC 时区对象
        else:
            tz = pytz.timezone(tz)  # 根据字符串创建 pytz 时区对象
    elif is_integer_object(tz):  # 如果 tz 是整数对象
        tz = timezone(timedelta(seconds=tz))  # 使用秒数创建时区对象
    elif isinstance(tz, tzinfo):  # 如果 tz 是 tzinfo 对象
        pass  # 直接跳过，保持 tz 不变
    elif tz is None:  # 如果 tz 是 None
        pass  # 直接跳过，保持 tz 不变
    else:
        raise TypeError(type(tz))  # 抛出类型错误异常，表示不支持的 tz 类型
    return tz  # 返回处理后的 tz 对象


# Python 接口的缓存函数，用于辅助测试
def _p_tz_cache_key(tz: tzinfo):
    return tz_cache_key(tz)  # 调用 tz_cache_key 函数返回时区对象的缓存键


# 时区数据缓存，键为 pytz 字符串或 dateutil 文件名
dst_cache = {}


# 定义一个 C 语言级别的函数，返回时区对象的缓存键或 None（如果未知）
cdef object tz_cache_key(tzinfo tz):
    if isinstance(tz, _pytz_BaseTzInfo):  # 如果 tz 是 pytz 的基本时区信息对象
        return tz.zone  # 返回时区名称作为缓存键
    elif isinstance(tz, _dateutil_tzfile):  # 如果 tz 是 dateutil 的 tzfile 对象
        if ".tar.gz" in tz._filename:  # 如果文件名中包含 ".tar.gz"
            raise ValueError("Bad tz filename. Dateutil on python 3 on "
                             "windows has a bug which causes tzfile._filename "
                             "to be the same for all timezone files. Please "
                             "construct dateutil timezones implicitly by "
                             'passing a string like "dateutil/Europe/London" '
                             "when you construct your pandas objects instead "
                             "of passing a timezone object. See "
                             "https://github.com/pandas-dev/pandas/pull/7362")
        return "dateutil" + tz._filename  # 返回以 "dateutil" 开头的文件名作为缓存键
    else:
        return None  # 其它情况返回 None，表示未知时区对象
# ----------------------------------------------------------------------
# UTC Offsets

# 获取时区对象的 UTC 偏移量
cdef timedelta get_utcoffset(tzinfo tz, datetime obj):
    try:
        # 尝试获取时区对象的内部 UTC 偏移量
        return tz._utcoffset
    except AttributeError:
        # 若不存在内部 UTC 偏移量，则计算给定日期时间对象的 UTC 偏移量
        return tz.utcoffset(obj)


# 判断时区对象是否为固定偏移量时区
cpdef inline bint is_fixed_offset(tzinfo tz):
    if treat_tz_as_dateutil(tz):
        # 如果是 dateutil 库的时区对象
        if len(tz._trans_idx) == 0 and len(tz._trans_list) == 0:
            return 1  # 是固定偏移量时区
        else:
            return 0  # 不是固定偏移量时区
    elif treat_tz_as_pytz(tz):
        # 如果是 pytz 库的时区对象
        if (len(tz._transition_info) == 0
                and len(tz._utc_transition_times) == 0):
            return 1  # 是固定偏移量时区
        else:
            return 0  # 不是固定偏移量时区
    elif is_zoneinfo(tz):
        # 如果是 zoneinfo 格式的时区对象
        return 0  # 不是固定偏移量时区
    # 对于 datetime.timezone 对象，也认为是固定偏移量时区
    return 1


# 从 dateutil 库的时区对象获取 UTC 过渡时间
cdef object _get_utc_trans_times_from_dateutil_tz(tzinfo tz):
    """
    Transition times in dateutil timezones are stored in local non-dst
    time.  This code converts them to UTC. It's the reverse of the code
    in dateutil.tz.tzfile.__init__.
    """
    new_trans = list(tz._trans_list)
    last_std_offset = 0
    for i, (trans, tti) in enumerate(zip(tz._trans_list, tz._trans_idx)):
        if not tti.isdst:
            last_std_offset = tti.offset
        new_trans[i] = trans - last_std_offset
    return new_trans


# 将 UTC 偏移信息转换为 int64_t 类型的数组
cdef int64_t[::1] unbox_utcoffsets(object transinfo):
    cdef:
        Py_ssize_t i
        cnp.npy_intp sz
        int64_t[::1] arr

    sz = len(transinfo)
    arr = cnp.PyArray_EMPTY(1, &sz, cnp.NPY_INT64, 0)

    for i in range(sz):
        # 将每个过渡时间对象转换为纳秒，并存入数组
        arr[i] = int(transinfo[i][0].total_seconds()) * 1_000_000_000

    return arr


# ----------------------------------------------------------------------
# Daylight Savings

# 获取时区对象的夏令时信息
cdef object get_dst_info(tzinfo tz):
    """
    Returns
    -------
    ndarray[int64_t]
        Nanosecond UTC times of DST transitions.
    ndarray[int64_t]
        Nanosecond UTC offsets corresponding to DST transitions.
    str
        Describing the type of tzinfo object.
    """
    cache_key = tz_cache_key(tz)
    if cache_key is None:
        # 若缓存键为空，说明无法确定时区信息，返回默认值
        # 例如：pytz.FixedOffset, matplotlib.dates._UTC, psycopg2.tz.FixedOffsetTimezone
        num = int(get_utcoffset(tz, None).total_seconds()) * 1_000_000_000
        return (np.array([NPY_NAT + 1], dtype=np.int64),
                np.array([num], dtype=np.int64),
                "unknown")
    # 检查缓存中是否存在指定的缓存键
    if cache_key not in dst_cache:
        # 如果时区被视为 pytz 类型，则处理时区转换信息
        if treat_tz_as_pytz(tz):
            # 将时区的 UTC 转换时间转换为 numpy 的 datetime64[ns] 类型数组
            trans = np.array(tz._utc_transition_times, dtype="M8[ns]")
            trans = trans.view("i8")
            # 如果第一个 UTC 转换时间的年份为 1，则进行特殊处理
            if tz._utc_transition_times[0].year == 1:
                trans[0] = NPY_NAT + 1
            # 获取并解析时区的偏移信息
            deltas = unbox_utcoffsets(tz._transition_info)
            typ = "pytz"

        # 如果时区被视为 dateutil 类型，则处理时区转换信息
        elif treat_tz_as_dateutil(tz):
            # 如果时区具有转换列表，则获取其 UTC 转换时间
            if len(tz._trans_list):
                # 从 dateutil 时区对象中获取 UTC 转换时间列表
                trans_list = _get_utc_trans_times_from_dateutil_tz(tz)
                # 将转换时间列表与首个占位符连接成 numpy datetime64[ns] 类型数组
                trans = np.hstack([
                    np.array([0], dtype="M8[s]"),  # 用于占位的第一项
                    np.array(trans_list, dtype="M8[s]")]).astype(
                    "M8[ns]")  # 所有列出的转换时间
                trans = trans.view("i8")
                trans[0] = NPY_NAT + 1

                # 获取并解析时区的偏移信息
                deltas = np.array([v.offset for v in (
                    tz._ttinfo_before,) + tz._trans_idx], dtype="i8")
                deltas *= 1_000_000_000
                typ = "dateutil"

            # 如果是固定偏移时区，则处理固定偏移信息
            elif is_fixed_offset(tz):
                trans = np.array([NPY_NAT + 1], dtype=np.int64)
                deltas = np.array([tz._ttinfo_std.offset],
                                  dtype="i8") * 1_000_000_000
                typ = "fixed"
            else:
                # 如果未处理的分支，在测试中未到达，且未在调用 get_dst_info 的任何函数中处理
                # 会导致调用函数假设 `deltas` 非空而引发 IndexError
                raise AssertionError("dateutil tzinfo is not a FixedOffset "
                                     "and has an empty `_trans_list`.", tz)
        else:
            # 静态时区信息处理，可能是 pytz.StaticTZInfo 类型
            trans = np.array([NPY_NAT + 1], dtype=np.int64)
            num = int(get_utcoffset(tz, None).total_seconds()) * 1_000_000_000
            deltas = np.array([num], dtype=np.int64)
            typ = "static"

        # 将计算得到的转换时间、偏移量和类型信息存入缓存
        dst_cache[cache_key] = (trans, deltas, typ)

    # 返回缓存中指定缓存键的值
    return dst_cache[cache_key]
def infer_tzinfo(datetime start, datetime end):
    # 如果 start 和 end 都不为 None，则获取 start 的时区信息
    if start is not None and end is not None:
        tz = start.tzinfo
        # 检查 start 和 end 的时区是否相同，如果不同则抛出异常
        if not tz_compare(tz, end.tzinfo):
            raise AssertionError(f"Inputs must both have the same timezone, "
                                 f"{tz} != {end.tzinfo}")
    # 如果只有 start 不为 None，则获取 start 的时区信息
    elif start is not None:
        tz = start.tzinfo
    # 如果只有 end 不为 None，则获取 end 的时区信息
    elif end is not None:
        tz = end.tzinfo
    # 如果 start 和 end 都为 None，则时区设为 None
    else:
        tz = None
    # 返回推断出的时区信息
    return tz


cpdef bint tz_compare(tzinfo start, tzinfo end):
    """
    Compare string representations of timezones

    The same timezone can be represented as different instances of
    timezones. For example
    `<DstTzInfo 'Europe/Paris' LMT+0:09:00 STD>` and
    `<DstTzInfo 'Europe/Paris' CET+1:00:00 STD>` are essentially same
    timezones but aren't evaluated such, but the string representation
    for both of these is `'Europe/Paris'`.

    This exists only to add a notion of equality to pytz-style zones
    that is compatible with the notion of equality expected of tzinfo
    subclasses.

    Parameters
    ----------
    start : tzinfo
    end : tzinfo

    Returns:
    -------
    bool
    """
    # GH 18523
    # 如果 start 是 UTC 时间，则与 end 比较是否也是 UTC 时间
    if is_utc(start):
        # GH#38851 consider pytz/dateutil/stdlib UTCs as equivalent
        return is_utc(end)
    # 如果 end 是 UTC 时间，则返回 False，不认为 tzlocal 等同于 UTC
    elif is_utc(end):
        # Ensure we don't treat tzlocal as equal to UTC when running in UTC
        return False
    # 如果 start 或 end 其中一个为 None，则只有当 start 和 end 都为 None 时返回 True
    elif start is None or end is None:
        return start is None and end is None
    # 否则比较 start 和 end 的时区是否相同
    return get_timezone(start) == get_timezone(end)


def tz_standardize(tz: tzinfo) -> tzinfo:
    """
    If the passed tz is a pytz timezone object, "normalize" it to the a
    consistent version

    Parameters
    ----------
    tz : tzinfo

    Returns
    -------
    tzinfo

    Examples
    --------
    >>> from datetime import datetime
    >>> from pytz import timezone
    >>> tz = timezone('US/Pacific').normalize(
    ...     datetime(2014, 1, 1, tzinfo=pytz.utc)
    ... ).tzinfo
    >>> tz
    <DstTzInfo 'US/Pacific' PST-1 day, 16:00:00 STD>
    >>> tz_standardize(tz)
    <DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>

    >>> tz = timezone('US/Pacific')
    >>> tz
    <DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>
    >>> tz_standardize(tz)
    <DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>
    """
    # 如果需要将 tz 视为 pytz 时区对象，则返回 pytz 中的该时区
    if treat_tz_as_pytz(tz):
        return pytz.timezone(str(tz))
    # 否则直接返回传入的时区对象
    return tz
```