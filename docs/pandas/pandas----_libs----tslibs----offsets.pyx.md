# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\offsets.pyx`

```
# 导入正则表达式模块
import re
# 导入时间模块
import time
# 导入警告模块
import warnings

# 从 pandas.util._exceptions 模块中导入 find_stack_level 函数
from pandas.util._exceptions import find_stack_level

# 使用 Cython 导入声明
cimport cython
# 从 cpython.datetime 模块中导入多个 C 类型和函数
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    date,
    datetime,
    import_datetime,
    time as dt_time,
    timedelta,
)

# 再次导入警告模块
import warnings

# 调用 import_datetime 函数
import_datetime()

# 导入 numpy 模块，并为其创建一个别名 np
import numpy as np

# 使用 Cython 导入声明
cimport numpy as cnp
# 从 numpy 模块中导入多个 C 类型
from numpy cimport (
    int64_t,
    ndarray,
)

# 调用 cnp.import_array() 函数
cnp.import_array()

# TODO: formalize having _libs.properties "above" tslibs in the dependency structure

# 从 pandas._libs.properties 模块中导入 cache_readonly 函数
from pandas._libs.properties import cache_readonly

# 从 pandas._libs.tslibs 模块中导入 util
from pandas._libs.tslibs cimport util
# 从 pandas._libs.tslibs.util 模块中导入多个 C 函数
from pandas._libs.tslibs.util cimport (
    is_float_object,
    is_integer_object,
)

# 从 pandas._libs.tslibs.ccalendar 模块中导入多个 C 函数和常量
from pandas._libs.tslibs.ccalendar import (
    MONTH_ALIASES,
    int_to_weekday,
    weekday_to_int,
)

# 从 pandas.util._exceptions 模块中导入 find_stack_level 函数
from pandas.util._exceptions import find_stack_level

# 从 pandas._libs.tslibs.ccalendar 模块中导入多个 C 函数
from pandas._libs.tslibs.ccalendar cimport (
    MONTH_TO_CAL_NUM,
    dayofweek,
    get_days_in_month,
    get_firstbday,
    get_lastbday,
)

# 从 pandas._libs.tslibs.conversion 模块中导入 localize_pydatetime 函数
from pandas._libs.tslibs.conversion cimport localize_pydatetime

# 从 pandas._libs.tslibs.dtypes 模块中导入多个 C 常量
from pandas._libs.tslibs.dtypes cimport (
    c_DEPR_ABBREVS,
    c_OFFSET_RENAMED_FREQSTR,
    c_OFFSET_TO_PERIOD_FREQSTR,
    c_PERIOD_AND_OFFSET_DEPR_FREQSTR,
    c_PERIOD_TO_OFFSET_FREQSTR,
    periods_per_day,
)

# 从 pandas._libs.tslibs.nattype 模块中导入多个 C 常量
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)

# 从 pandas._libs.tslibs.np_datetime 模块中导入多个 C 函数和常量
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydate_to_dtstruct,
)

# 调用 import_pandas_datetime() 函数
import_pandas_datetime()

# 从 .dtypes 模块中导入 PeriodDtypeCode 类型
from .dtypes cimport PeriodDtypeCode
# 从 .timedeltas 模块中导入多个 C 类型和函数
from .timedeltas cimport (
    _Timedelta,
    delta_to_nanoseconds,
    is_any_td_scalar,
)

# 从 .timedeltas 模块中导入 Timedelta 类
from .timedeltas import Timedelta

# 从 .timestamps 模块中导入 _Timestamp 类
from .timestamps cimport _Timestamp

# 从 .timestamps 模块中导入 Timestamp 类
from .timestamps import Timestamp

# ---------------------------------------------------------------------
# Misc Helpers

# 定义一个 Cython 函数 is_offset_object，检查对象是否为 BaseOffset 类型
cdef bint is_offset_object(object obj):
    return isinstance(obj, BaseOffset)

# 定义一个 Cython 函数 is_tick_object，检查对象是否为 Tick 类型
cdef bint is_tick_object(object obj):
    return isinstance(obj, Tick)

# 定义一个 Python 函数 apply_wraps，接受一个函数作为参数
def apply_wraps(func):
    # 注意：通常我们会使用 `@functools.wraps(func)`，但这与 Cython 类方法不兼容
    pass  # 此函数暂时没有实现内容，只有注释
    # 定义装饰器函数 wrapper，接受 self 和 other 两个参数
    def wrapper(self, other):
        # 检查 other 是否为 NaT（Not a Time，即不是时间类型）
        if other is NaT:
            return NaT
        elif (
            isinstance(other, BaseOffset)  # 检查 other 是否为 BaseOffset 的实例
            or PyDelta_Check(other)        # 检查 other 是否为 PyDelta 对象
            or cnp.is_timedelta64_object(other)  # 检查 other 是否为 pandas 的 timedelta64 对象
        ):
            # 如果 other 是时间增量类型，则调用 func(self, other) 处理
            return func(self, other)
        elif cnp.is_datetime64_object(other) or PyDate_Check(other):
            # 如果 other 是 datetime64 对象或者 Python 的日期对象
            other = Timestamp(other)  # 转换为 pandas 的 Timestamp 对象
        else:
            # 如果 other 不符合上述任何条件，抛出 ApplyTypeError 异常
            raise ApplyTypeError

        tz = other.tzinfo  # 获取 other 的时区信息
        nano = other.nanosecond  # 获取 other 的纳秒数

        if self._adjust_dst:
            other = other.tz_localize(None)  # 如果 self._adjust_dst 为真，则将 other 的时区设为 None

        result = func(self, other)  # 调用 func 处理 self 和处理过时区的 other

        result2 = Timestamp(result).as_unit(other.unit)  # 将处理结果转换为 other 的单位

        if result == result2:
            # 如果 result 等于 result2，即转换是非损失的情况，如 test_milliseconds_combination 不会发生此情况
            result = result2

        if self._adjust_dst:
            result = result.tz_localize(tz)  # 如果 self._adjust_dst 为真，则将 result 的时区设回 tz

        if self.normalize:
            result = result.normalize()  # 如果 self.normalize 为真，则对 result 进行标准化处理

        # 如果 offset 对象没有纳秒部分，则结果的纳秒部分可能会丢失
        if not self.normalize and nano != 0 and not hasattr(self, "nanoseconds"):
            if result.nanosecond != nano:
                if result.tz is not None:
                    # 将 result 转换为 UTC 时区
                    res = result.tz_localize(None)
                else:
                    res = result
                value = res.as_unit("ns")._value
                result = Timestamp(value + nano)  # 将 result 的值增加 nano 纳秒

        if tz is not None and result.tzinfo is None:
            result = result.tz_localize(tz)  # 如果 result 没有时区信息但 other 有，则将 result 的时区设为 tz

        return result

    # 手动设置 wrapper 函数的名称和文档字符串，因为 @functools.wraps(func) 在 cdef 函数上不起作用
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    return wrapper


这段代码定义了一个装饰器函数 `wrapper`，用于处理两个时间对象之间的操作。根据 `other` 的类型和属性，选择不同的处理路径，并确保最终结果的时区和纳秒精度符合预期。
# 根据 PyDelta_Check 函数检查 result 是否为 PyDelta 类型对象，如果是则进行处理
def _wrap_timedelta_result(result):
    """
    Tick operations dispatch to their Timedelta counterparts.  Wrap the result
    of these operations in a Tick if possible.

    Parameters
    ----------
    result : object

    Returns
    -------
    object
    """
    if PyDelta_Check(result):
        # 将 Timedelta 转换回 Tick
        return delta_to_tick(result)

    # 如果 result 不是 PyDelta 类型对象，则直接返回原始 result
    return result


# ---------------------------------------------------------------------
# Business Helpers


# 根据给定的 weekmask、holidays 和 calendar 生成 busdaycalendar
cdef _get_calendar(weekmask, holidays, calendar):
    """
    Generate busdaycalendar
    """
    # 如果 calendar 是 np.busdaycalendar 类型的实例
    if isinstance(calendar, np.busdaycalendar):
        # 如果 holidays 为空，则使用 calendar 的 holidays；否则转换为元组使用
        if not holidays:
            holidays = tuple(calendar.holidays)
        elif not isinstance(holidays, tuple):
            holidays = tuple(holidays)
        else:
            # 假设 calendar.holidays 和 holidays 是一致的
            pass
        return calendar, holidays

    # 如果 holidays 为 None，则设为一个空列表
    if holidays is None:
        holidays = []

    # 尝试从 calendar 获取 holidays 并转换为 np.datetime64[D] 类型
    try:
        holidays = holidays + calendar.holidays().tolist()
    except AttributeError:
        pass

    # 将 holidays 转换为排序后的元组，每个元素都转换为 np.datetime64[D] 类型
    holidays = tuple(sorted(_to_dt64D(dt) for dt in holidays))

    # 设置 kwargs 字典，用于创建 np.busdaycalendar 对象
    kwargs = {"weekmask": weekmask}
    if holidays:
        kwargs["holidays"] = holidays

    # 使用 kwargs 创建 busdaycalendar 对象
    busdaycalendar = np.busdaycalendar(**kwargs)
    return busdaycalendar, holidays


# 将给定的 dt 转换为 np.datetime64[D] 类型
cdef _to_dt64D(dt):
    # 如果 dt 包含时区信息，则转换为 naive datetime 后再转换为 np.datetime64[D]
    if getattr(dt, "tzinfo", None) is not None:
        naive = dt.replace(tzinfo=None)
        dt = np.datetime64(naive, "D")
    else:
        dt = np.datetime64(dt)

    # 如果 dt 的类型不是 datetime64[D]，则强制转换为 datetime64[D]
    if dt.dtype.name != "datetime64[D]":
        dt = dt.astype("datetime64[D]")
    return dt


# ---------------------------------------------------------------------
# Validation


# 验证 t_input 是否为有效的业务时间
cdef _validate_business_time(t_input):
    # 如果 t_input 是字符串，则尝试解析为时间格式 %H:%M
    if isinstance(t_input, str):
        try:
            t = time.strptime(t_input, "%H:%M")
            return dt_time(hour=t.tm_hour, minute=t.tm_min)
        except ValueError:
            raise ValueError("time data must match '%H:%M' format")
    # 如果 t_input 是 datetime.time 类型，则确保只包含小时和分钟
    elif isinstance(t_input, dt_time):
        if t_input.second != 0 or t_input.microsecond != 0:
            raise ValueError(
                "time data must be specified only with hour and minute")
        return t_input
    else:
        # 如果 t_input 类型不是字符串或 datetime.time，则抛出异常
        raise ValueError("time data must be string or datetime.time")


# ---------------------------------------------------------------------
# Constructor Helpers
# 定义了一个集合，包含了相对时间偏移的关键字参数，用于后续处理
_relativedelta_kwds = {"years", "months", "weeks", "days", "year", "month",
                       "day", "weekday", "hour", "minute", "second",
                       "microsecond", "millisecond", "nanosecond",
                       "nanoseconds", "hours", "minutes", "seconds",
                       "milliseconds", "microseconds"}

# 定义一个Cython函数，用于确定时间偏移量
cdef _determine_offset(kwds):
    # 如果没有传入关键字参数，按照历史约定，默认返回1天的时间偏移量
    if not kwds:
        return timedelta(days=1), False

    # 如果关键字参数包含 "millisecond"，则抛出未实现的错误，因为不支持毫秒级别的偏移
    if "millisecond" in kwds:
        raise NotImplementedError(
            "Using DateOffset to replace `millisecond` component in "
            "datetime object is not supported. Use "
            "`microsecond=timestamp.microsecond % 1000 + ms * 1000` "
            "instead."
        )

    # 定义一个集合，包含纳秒级别的关键字参数
    nanos = {"nanosecond", "nanoseconds"}

    # 如果所有关键字参数都是纳秒级别的，通过 apply_wraps 处理，返回0天的时间偏移量
    if all(k in nanos for k in kwds):
        return timedelta(days=0), False

    # 去除掉所有的纳秒级别的关键字参数后，剩下的关键字参数集合
    kwds_no_nanos = {k: v for k, v in kwds.items() if k not in nanos}

    # 用于相对时间处理的关键字参数集合
    kwds_use_relativedelta = {
        "year", "month", "day", "hour", "minute",
        "second", "microsecond", "weekday", "years", "months", "weeks", "days",
        "hours", "minutes", "seconds", "microseconds"
    }

    # 用于 timedelta 处理的关键字参数集合
    kwds_use_timedelta = {
        "seconds", "microseconds", "milliseconds", "minutes", "hours",
    }

    # 如果所有的关键字参数都是 timedelta 可用的，表示小于一天的时间偏移量，返回对应的 timedelta
    if all(k in kwds_use_timedelta for k in kwds_no_nanos):
        return timedelta(**kwds_no_nanos), False

    # 如果关键字参数包含 "milliseconds"，将其转换为微秒级别，以便 relativedelta 可以处理
    if "milliseconds" in kwds_no_nanos:
        micro = kwds_no_nanos.pop("milliseconds") * 1000
        kwds_no_nanos["microseconds"] = kwds_no_nanos.get("microseconds", 0) + micro

    # 如果所有的关键字参数都是 relativedelta 可用的，使用 relativedelta 处理
    if all(k in kwds_use_relativedelta for k in kwds_no_nanos):
        from dateutil.relativedelta import relativedelta
        return relativedelta(**kwds_no_nanos), True

    # 若以上条件都不满足，抛出值错误，表示传入的参数组合无效
    raise ValueError(
        f"Invalid argument/s or bad combination of arguments: {list(kwds.keys())}"
    )
    # 设置 numpy 标量的数组优先级，确保反向操作返回 NotImplemented
    __array_priority__ = 1000

    # 初始化类变量 _day_opt 为 None
    _day_opt = None

    # 初始化元组 _attributes 包含两个字符串元素 "n" 和 "normalize"
    _attributes = tuple(["n", "normalize"])

    # 初始化布尔变量 _use_relativedelta 为 False
    _use_relativedelta = False

    # 初始化布尔变量 _adjust_dst 为 True
    _adjust_dst = True

    # 定义类的构造函数，初始化实例变量 n 和 normalize
    def __init__(self, n=1, normalize=False):
        # 调用内部方法 _validate_n 验证并返回有效的 n 值
        n = self._validate_n(n)
        # 设置实例变量 self.n 为验证后的 n 值
        self.n = n
        # 设置实例变量 self.normalize 为传入的 normalize 值
        self.normalize = normalize
        # 初始化实例变量 _cache 为空字典
        self._cache = {}

    # 定义等于操作符的方法，返回布尔值
    def __eq__(self, other) -> bool:
        # 如果 other 是字符串类型，尝试将其转换为 DateOffset 对象
        if isinstance(other, str):
            try:
                # 调用 to_offset 函数尝试转换，若失败则处理为不可比较的类型，返回 False
                other = to_offset(other)
            except ValueError:
                # 如果转换失败，例如遇到 "infer"，直接返回 False
                return False
        try:
            # 比较两个 DateOffset 对象的 _params 属性是否相等
            return self._params == other._params
        except AttributeError:
            # 如果 other 不是 DateOffset 对象，返回 False
            return False

    # 定义不等于操作符的方法，返回布尔值
    def __ne__(self, other):
        # 利用 __eq__ 方法的结果取反，实现不等于操作
        return not self == other

    # 定义哈希方法，返回对象的哈希值
    def __hash__(self) -> int:
        # 返回 _params 属性的哈希值
        return hash(self._params)

    # 定义只读缓存装饰器方法 _params
    @cache_readonly
    def _params(self):
        """
        返回一个元组，包含评估两个 DateOffset 对象是否相等所需的所有属性。
        """
        # 获取当前实例的 __dict__ 属性
        d = getattr(self, "__dict__", {})
        # 复制所有参数到字典 all_paras
        all_paras = d.copy()
        # 设置 n 和 normalize 属性到 all_paras 字典
        all_paras["n"] = self.n
        all_paras["normalize"] = self.normalize
        # 遍历 _attributes 元组，将存在于对象中但不在 d 中的属性添加到 all_paras 中
        for attr in self._attributes:
            if hasattr(self, attr) and attr not in d:
                all_paras[attr] = getattr(self, attr)

        # 如果 all_paras 中包含 holidays 属性且其值为空，则移除该属性
        if "holidays" in all_paras and not all_paras["holidays"]:
            all_paras.pop("holidays")

        # 定义排除集合 exclude，将其中的属性从 all_paras 中移除
        exclude = {"kwds", "name", "calendar"}
        attrs = {(k, v) for k, v in all_paras.items()
                 if (k not in exclude) and (k[0] != "_")}
        # 将所有属性转换为元组并返回
        params = tuple([str(type(self))] + sorted(attrs))
        return params

    # 定义属性方法 kwds，返回偏移量的额外参数字典
    @property
    def kwds(self) -> dict:
        """
        返回偏移量的额外参数字典。

        参见
        --------
        tseries.offsets.DateOffset : 所有 pandas 日期偏移量的基类。
        tseries.offsets.WeekOfMonth : 代表月份中的某一周。
        tseries.offsets.LastWeekOfMonth : 代表月份中的最后一周。

        示例
        --------
        >>> pd.DateOffset(5).kwds
        {}

        >>> pd.offsets.FY5253Quarter().kwds
        {'weekday': 0,
         'startingMonth': 1,
         'qtr_with_extra_week': 1,
         'variation': 'nearest'}
        """
        # 为了向后兼容性，构建并返回 _attributes 中除去 "n" 和 "normalize" 的属性字典
        kwds = {name: getattr(self, name, None) for name in self._attributes
                if name not in ["n", "normalize"]}
        return {name: kwds[name] for name in kwds if kwds[name] is not None}
    def base(self):
        """
        Returns a copy of the calling offset object with n=1 and all other
        attributes equal.
        """
        # 返回一个调用偏移对象的副本，其中 n=1，其它属性保持不变
        return type(self)(n=1, normalize=self.normalize, **self.kwds)

    def __add__(self, other):
        if util.is_array(other) and other.dtype == object:
            # 如果 other 是数组且元素类型是对象，则返回一个包含所有元素与 self 相加结果的数组
            return np.array([self + x for x in other])

        try:
            # 否则尝试调用 _apply 方法进行操作
            return self._apply(other)
        except ApplyTypeError:
            # 如果操作失败，则返回 NotImplemented
            return NotImplemented

    def __radd__(self, other):
        # 右侧加法操作，委托给 __add__ 方法处理
        return self.__add__(other)

    def __sub__(self, other):
        if PyDateTime_Check(other):
            # 如果 other 是日期时间对象，则抛出类型错误
            raise TypeError("Cannot subtract datetime from offset.")
        elif type(other) is type(self):
            # 如果 other 类型与 self 类型相同，则返回一个新的偏移对象，n 为两者的差值
            return type(self)(self.n - other.n, normalize=self.normalize,
                              **self.kwds)
        else:
            # 否则返回 NotImplemented，例如 PeriodIndex
            return NotImplemented

    def __rsub__(self, other):
        # 右侧减法操作，等效于 -self + other
        return (-self).__add__(other)

    def __mul__(self, other):
        if util.is_array(other):
            # 如果 other 是数组，则返回一个包含所有元素与 self 相乘结果的数组
            return np.array([self * x for x in other])
        elif is_integer_object(other):
            # 如果 other 是整数对象，则返回一个新的偏移对象，n 为 other 与 self.n 的乘积
            return type(self)(n=other * self.n, normalize=self.normalize,
                              **self.kwds)
        # 否则返回 NotImplemented
        return NotImplemented

    def __rmul__(self, other):
        # 右侧乘法操作，委托给 __mul__ 方法处理
        return self.__mul__(other)

    def __neg__(self):
        # 返回当前偏移对象的相反数，使用 __mul__ 方法进行计算
        # 注意：直接使用 __mul__ 而非 __rmul__，以便支持 `cdef class` 中可用的方法
        return self * -1

    def copy(self):
        """
        Return a copy of the frequency.

        See Also
        --------
        tseries.offsets.Week.copy : Return a copy of Week offset.
        tseries.offsets.DateOffset.copy : Return a copy of date offset.
        tseries.offsets.MonthEnd.copy : Return a copy of MonthEnd offset.
        tseries.offsets.YearBegin.copy : Return a copy of YearBegin offset.

        Examples
        --------
        >>> freq = pd.DateOffset(1)
        >>> freq_copy = freq.copy()
        >>> freq is freq_copy
        False
        """
        # 返回当前频率的一个副本，使用 __mul__ 方法进行计算，即 self * 1
        return self * 1

    # ------------------------------------------------------------------
    # Name and Rendering Methods

    def __repr__(self) -> str:
        # _output_name 用于 B(Year|Quarter)(End|Begin) 扩展 "B" -> "Business"
        class_name = getattr(self, "_output_name", type(self).__name__)

        if abs(self.n) != 1:
            plural = "s"
        else:
            plural = ""

        n_str = ""
        if self.n != 1:
            n_str = f"{self.n} * "

        # 返回当前对象的字符串表示形式，包括偏移量 n、类名和其它属性
        out = f"<{n_str}{class_name}{plural}{self._repr_attrs()}>"
        return out
    # 返回描述对象属性的字符串表示，排除指定的属性集合
    def _repr_attrs(self) -> str:
        exclude = {"n", "inc", "normalize"}
        attrs = []
        for attr in sorted(self._attributes):
            # 因为 Cython 属性不在 __dict__ 中，所以使用 _attributes 而不是 __dict__
            if attr.startswith("_") or attr == "kwds" or not hasattr(self, attr):
                # DateOffset 可能没有一些这些属性
                continue
            elif attr not in exclude:
                # 获取属性值并添加到属性列表中
                value = getattr(self, attr)
                attrs.append(f"{attr}={value}")

        out = ""
        if attrs:
            # 如果有属性，则以冒号开头并用逗号分隔连接所有属性
            out += ": " + ", ".join(attrs)
        return out

    @property
    def name(self) -> str:
        """
        返回表示基础频率的字符串。

        参见
        --------
        tseries.offsets.Week : 表示每周的偏移量。
        DateOffset : 所有其他偏移量类的基类。
        tseries.offsets.Day : 表示单日偏移量。
        tseries.offsets.MonthEnd : 表示按月结束对齐的偏移量。

        示例
        --------
        >>> pd.offsets.Hour().name
        'h'

        >>> pd.offsets.Hour(5).name
        'h'
        """
        return self.rule_code

    @property
    def _prefix(self) -> str:
        # 抛出未实现错误，提示子类需要定义 _prefix 方法
        raise NotImplementedError("Prefix not defined")

    @property
    def rule_code(self) -> str:
        # 返回 _prefix 的值作为规则代码
        return self._prefix

    @cache_readonly
    def freqstr(self) -> str:
        """
        返回表示频率的字符串。

        示例
        --------
        >>> pd.DateOffset(5).freqstr
        '<5 * DateOffsets>'

        >>> pd.offsets.BusinessHour(2).freqstr
        '2bh'

        >>> pd.offsets.Nano().freqstr
        'ns'

        >>> pd.offsets.Nano(-3).freqstr
        '-3ns'
        """
        try:
            # 尝试获取规则代码
            code = self.rule_code
        except NotImplementedError:
            # 如果未实现则返回对象的字符串表示形式
            return str(repr(self))

        if self.n != 1:
            # 如果 n 不为 1，则格式化频率字符串
            fstr = f"{self.n}{code}"
        else:
            fstr = code

        try:
            if self._offset:
                # 如果存在 _offset，则追加其字符串表示形式
                fstr += self._offset_str()
        except AttributeError:
            # TODO: 统一 `_offset` 和 `offset` 命名约定
            pass

        return fstr

    def _offset_str(self) -> str:
        # 返回空字符串，子类可以实现该方法以提供偏移量的字符串表示形式
        return ""

    # ------------------------------------------------------------------

    def _apply(self, other):
        # 抛出未实现错误，提示子类需要实现 _apply 方法
        raise NotImplementedError("implemented by subclasses")

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # NB: _apply_array 不处理 `self.normalize` 的尊重，调用者 (DatetimeArray) 在后处理中处理这一点。
        # 抛出未实现错误，指示 DateOffset 子类没有向量化实现
        raise NotImplementedError(
            f"DateOffset subclass {type(self).__name__} "
            "does not have a vectorized implementation"
        )
    def rollback(self, dt) -> datetime:
        """
        Roll provided date backward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
        # 将输入的日期转换为时间戳对象
        dt = Timestamp(dt)
        # 如果输入的日期不在偏移量上
        if not self.is_on_offset(dt):
            # 向后滚动一个偏移量，并重新赋值给 dt
            dt = dt - type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def rollforward(self, dt) -> datetime:
        """
        Roll provided date forward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
        # 将输入的日期转换为时间戳对象
        dt = Timestamp(dt)
        # 如果输入的日期不在偏移量上
        if not self.is_on_offset(dt):
            # 向前滚动一个偏移量，并重新赋值给 dt
            dt = dt + type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def _get_offset_day(self, other: datetime) -> int:
        # 子类必须实现 `_day_opt`；从基类调用时会默认假定 `day_opt = "business_end"`，参见 `get_day_of_month`。
        cdef:
            npy_datetimestruct dts
        # 将 Python 的日期对象转换为 C 的日期结构体
        pydate_to_dtstruct(other, &dts)
        # 返回日期所在月份中的天数，根据 `_day_opt` 的设置不同返回不同结果
        return get_day_of_month(&dts, self._day_opt)

    def is_on_offset(self, dt: datetime) -> bool:
        """
        Return boolean whether a timestamp intersects with this frequency.

        Parameters
        ----------
        dt : datetime.datetime
            Timestamp to check intersections with frequency.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Day(1)
        >>> freq.is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 6)
        >>> ts.day_name()
        'Saturday'
        >>> freq = pd.offsets.BusinessDay(1)
        >>> freq.is_on_offset(ts)
        False
        """
        # 如果启用了标准化，并且日期未标准化，则返回 False
        if self.normalize and not _is_normalized(dt):
            return False

        # 默认（较慢）的方法用于确定某个日期是否是由该偏移量生成的日期范围的成员。
        # 子类可以以更好的方式重新实现这个方法。
        a = dt
        b = (dt + self) - self
        # 返回是否两个日期相等，判断日期是否在偏移量上
        return a == b

    # ------------------------------------------------------------------

    # 静态方法，使得可以从 Tick.__init__ 中调用，一旦 BaseOffset 是 cdef 类并被 Tick 继承，则此方法将不再需要
    @staticmethod
    # 确保 `n` 是一个整数的验证函数
    def _validate_n(n) -> int:
        """
        Require that `n` be an integer.

        Parameters
        ----------
        n : int
            要验证的整数值

        Returns
        -------
        nint : int
            转换后的整数值

        Raises
        ------
        TypeError if `int(n)` raises
            如果 `n` 不是整数类型，则抛出类型错误异常
        ValueError if n != int(n)
            如果 `n` 不等于其转换为整数后的值，则抛出值错误异常
        """
        if cnp.is_timedelta64_object(n):
            raise TypeError(f"`n` argument must be an integer, got {type(n)}")
        try:
            nint = int(n)
        except (ValueError, TypeError):
            raise TypeError(f"`n` argument must be an integer, got {type(n)}")
        if n != nint:
            raise ValueError(f"`n` argument must be an integer, got {n}")
        return nint

    # 从已序列化的状态中重建对象实例
    def __setstate__(self, state):
        """
        Reconstruct an instance from a pickled state
        """
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})
        # 此时我们期望 state 为空

    # 返回可被 pickle 序列化的状态
    def __getstate__(self):
        """
        Return a pickleable state
        """
        state = {}
        state["n"] = self.n
        state["normalize"] = self.normalize

        # 我们不希望将日历对象实际序列化
        # 因为它是一个 np.busyday; 我们在反序列化时重新创建它
        state.pop("calendar", None)
        if "kwds" in state:
            state["kwds"].pop("calendar", None)

        return state

    @property
    def nanos(self):
        raise ValueError(f"{self} is a non-fixed frequency")

    # ------------------------------------------------------------------

    # 检查时间戳是否在月初
    def is_month_start(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the month start.

        Parameters
        ----------
        ts : Timestamp
            要检查的时间戳对象

        See Also
        --------
        is_month_end : Return boolean whether a timestamp occurs on the month end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_start(ts)
        True
        """
        return ts._get_start_end_field("is_month_start", self)

    # 检查时间戳是否在月末
    def is_month_end(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the month end.

        Parameters
        ----------
        ts : Timestamp
            要检查的时间戳对象

        See Also
        --------
        is_month_start : Return boolean whether a timestamp occurs on the month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_end(ts)
        False
        """
        return ts._get_start_end_field("is_month_end", self)
    # 返回一个布尔值，指示时间戳是否处于季度的开始
    def is_quarter_start(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the quarter start.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_quarter_end : Return boolean whether a timestamp occurs on the quarter end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_start(ts)
        True
        """
        return ts._get_start_end_field("is_quarter_start", self)

    # 返回一个布尔值，指示时间戳是否处于季度的结束
    def is_quarter_end(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the quarter end.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_quarter_start : Return boolean whether a timestamp
            occurs on the quarter start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_end(ts)
        False
        """
        return ts._get_start_end_field("is_quarter_end", self)

    # 返回一个布尔值，指示时间戳是否处于年度的开始
    def is_year_start(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the year start.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_year_end : Return boolean whether a timestamp occurs on the year end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_start(ts)
        True
        """
        return ts._get_start_end_field("is_year_start", self)

    # 返回一个布尔值，指示时间戳是否处于年度的结束
    def is_year_end(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the year end.

        Parameters
        ----------
        ts : Timestamp
            The timestamp to check.

        See Also
        --------
        is_year_start : Return boolean whether a timestamp occurs on the year start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_end(ts)
        False
        """
        return ts._get_start_end_field("is_year_end", self)
# 定义一个 Cython 类 SingleConstructorOffset，继承自 BaseOffset
cdef class SingleConstructorOffset(BaseOffset):
    
    # 定义类方法 _from_name，用于根据名称创建对象实例
    @classmethod
    def _from_name(cls, suffix=None):
        # 如果 suffix 不为 None，则抛出 ValueError 异常
        if suffix:
            raise ValueError(f"Bad freq suffix {suffix}")
        # 返回当前类的实例，不带参数
        return cls()

    # 实现特殊方法 __reduce__，用于对象的序列化和反序列化
    def __reduce__(self):
        # 构造包含对象属性的元组，除了不序列化 "calendar" 属性外
        tup = tuple(
            getattr(self, attr) if attr != "calendar" else None
            for attr in self._attributes
        )
        # 返回对象类型和属性元组，用于反序列化
        return type(self), tup


# ---------------------------------------------------------------------
# Tick Offsets

# 定义 Cython 类 Tick，继承自 SingleConstructorOffset
cdef class Tick(SingleConstructorOffset):
    
    # 类属性 _adjust_dst 设为 False
    _adjust_dst = False
    # 类属性 _prefix 设为 "undefined"
    _prefix = "undefined"
    # 类属性 _attributes 包含 "n" 和 "normalize"
    _attributes = tuple(["n", "normalize"])

    # 初始化方法，接受参数 n 和 normalize，默认 normalize=False
    def __init__(self, n=1, normalize=False):
        # 调用 _validate_n 方法验证并设置 n 属性
        n = self._validate_n(n)
        self.n = n
        # 设置 normalize 属性为 False
        self.normalize = False
        # 初始化缓存属性 _cache 为空字典
        self._cache = {}
        # 如果 normalize 参数为 True，则抛出 ValueError 异常
        if normalize:
            raise ValueError(
                "Tick offset with `normalize=True` are not allowed."
            )

    # 注意：不将此方法声明为 cpdef 将导致在调用 from __mul__ 时出现 AttributeError
    # 定义 cpdef 方法 _next_higher_resolution，返回更高分辨率的 Tick 对象
    cpdef Tick _next_higher_resolution(Tick self):
        # 根据当前 Tick 对象的类型返回下一个更高分辨率的 Tick 对象
        if type(self) is Day:
            return Hour(self.n * 24)
        if type(self) is Hour:
            return Minute(self.n * 60)
        if type(self) is Minute:
            return Second(self.n * 60)
        if type(self) is Second:
            return Milli(self.n * 1000)
        if type(self) is Milli:
            return Micro(self.n * 1000)
        if type(self) is Micro:
            return Nano(self.n * 1000)
        # 如果无法转换为整数偏移量，则抛出 ValueError 异常
        raise ValueError("Could not convert to integer offset at any resolution")

    # --------------------------------------------------------------------

    # 定义 _repr_attrs 方法，返回空字符串，覆盖父类的实现
    def _repr_attrs(self) -> str:
        return ""

    # 定义属性装饰器 cache_readonly，返回 Timedelta 对象
    @cache_readonly
    def _as_pd_timedelta(self):
        return Timedelta(self)

    # 定义属性 nanos，返回当前 Tick 对象总纳秒数，为整数类型
    @property
    def nanos(self) -> int64_t:
        """
        Return an integer of the total number of nanoseconds.

        Raises
        ------
        ValueError
            If the frequency is non-fixed.

        Examples
        --------
        >>> pd.offsets.Hour(5).nanos
        18000000000000
        """
        return self.n * self._nanos_inc

    # 定义方法 is_on_offset，接受参数 dt(datetime 对象)，返回布尔值
    def is_on_offset(self, dt: datetime) -> bool:
        return True

    # 重新定义特殊方法 __hash__，返回对象的哈希值，用于对象的哈希映射
    def __hash__(self) -> int:
        return hash(self._params)

    # --------------------------------------------------------------------
    # Comparison and Arithmetic Methods
    def __eq__(self, other):
        # 如果 `other` 是字符串类型
        if isinstance(other, str):
            try:
                # 尝试将 `other` 转换为时间偏移量对象 `to_offset`
                other = to_offset(other)
            except ValueError:
                # 如果转换失败，例如 "infer"
                return False
        # 返回当前对象的时间偏移量是否等于 `other`
        return self._as_pd_timedelta == other

    def __ne__(self, other):
        # 返回当前对象的时间偏移量是否不等于 `other`
        return not (self == other)

    def __le__(self, other):
        # 返回当前对象的时间偏移量是否小于等于 `other`
        return self._as_pd_timedelta.__le__(other)

    def __lt__(self, other):
        # 返回当前对象的时间偏移量是否小于 `other`
        return self._as_pd_timedelta.__lt__(other)

    def __ge__(self, other):
        # 返回当前对象的时间偏移量是否大于等于 `other`
        return self._as_pd_timedelta.__ge__(other)

    def __gt__(self, other):
        # 返回当前对象的时间偏移量是否大于 `other`
        return self._as_pd_timedelta.__gt__(other)

    def __mul__(self, other):
        # 如果 `other` 是浮点数对象
        if is_float_object(other):
            # 计算乘积
            n = other * self.n
            # 如果乘积是整数，则返回相同类型的时间偏移量对象
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            # 否则，使用更高分辨率的子类重新计算乘积
            new_self = self._next_higher_resolution()
            return new_self * other
        # 如果 `other` 不是浮点数对象，则调用基类的乘法操作
        return BaseOffset.__mul__(self, other)

    def __rmul__(self, other):
        # 右乘法操作，委托给 `__mul__` 方法处理
        return self.__mul__(other)

    def __truediv__(self, other):
        # 如果 `self` 不是 Tick 类型
        if not isinstance(self, Tick):
            # 根据 Cython 语义，有时参数会被交换，执行反向的真除法操作
            result = other._as_pd_timedelta.__rtruediv__(self)
        else:
            # 否则，执行当前对象与 `other` 的真除法操作
            result = self._as_pd_timedelta.__truediv__(other)
        # 封装并返回时间增量的计算结果
        return _wrap_timedelta_result(result)

    def __rtruediv__(self, other):
        # 执行反向的真除法操作
        result = self._as_pd_timedelta.__rtruediv__(other)
        # 封装并返回时间增量的计算结果
        return _wrap_timedelta_result(result)

    def __add__(self, other):
        # 如果 `other` 是 Tick 类型的对象
        if isinstance(other, Tick):
            # 如果两者类型相同，直接相加并返回相同类型的 Tick 对象
            if type(self) is type(other):
                return type(self)(self.n + other.n)
            else:
                # 否则，将两个时间增量转换为 Tick 对象并相加
                return delta_to_tick(self._as_pd_timedelta + other._as_pd_timedelta)
        try:
            # 尝试将 `other` 应用于当前对象
            return self._apply(other)
        except ApplyTypeError:
            # 如果应用类型错误，则返回 NotImplemented
            return NotImplemented
        except OverflowError as err:
            # 如果发生溢出错误，则引发 OverflowError，并指明具体的溢出操作
            raise OverflowError(
                f"the add operation between {self} and {other} will overflow"
            ) from err

    def __radd__(self, other):
        # 右加法操作，委托给 `__add__` 方法处理
        return self.__add__(other)
    # 对另一个对象应用时间戳操作，这里不需要使用 apply_wraps
    def _apply(self, other):
        if isinstance(other, _Timestamp):  # 检查是否为 _Timestamp 类型
            # 返回另一个时间戳对象加上当前对象的 pandas 时间增量
            # GH#15126
            return other + self._as_pd_timedelta
        elif other is NaT:  # 如果 other 是 NaT（Not a Time）
            return NaT  # 返回 NaT
        elif cnp.is_datetime64_object(other) or PyDate_Check(other):
            # 如果 other 是 datetime64 对象或者是 Python 的日期对象
            # 使用 Timestamp 类构造函数创建一个新的时间戳对象，加上当前对象
            return Timestamp(other) + self

        if cnp.is_timedelta64_object(other) or PyDelta_Check(other):
            # 如果 other 是 timedelta64 对象或者是 Python 的 timedelta 对象
            # 返回 other 加上当前对象的 pandas 时间增量
            return other + self._as_pd_timedelta

        # 如果 other 类型未处理，抛出异常
        raise ApplyTypeError(f"Unhandled type: {type(other).__name__}")

    # --------------------------------------------------------------------
    # Pickle Methods

    def __setstate__(self, state):
        # 设置对象的状态，从传入的状态字典中恢复对象的属性
        self.n = state["n"]
        self.normalize = False
cdef class Day(Tick):
    """
    Offset ``n`` days.

    Attributes
    ----------
    n : int, default 1
        The number of days represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n days.

    >>> from pandas.tseries.offsets import Day
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Day()
    Timestamp('2022-12-10 15:00:00')
    >>> ts - Day(4)
    Timestamp('2022-12-05 15:00:00')

    >>> ts + Day(-4)
    Timestamp('2022-12-05 15:00:00')
    """
    # 纳秒增量表示一天的时间
    _nanos_inc = 24 * 3600 * 1_000_000_000
    # 前缀表示这是一天的偏移
    _prefix = "D"
    # 周期数据类型代码，表示这是一个天的周期
    _period_dtype_code = PeriodDtypeCode.D
    # C 语言表示，表示天的时间单位
    _creso = NPY_DATETIMEUNIT.NPY_FR_D


cdef class Hour(Tick):
    """
    Offset ``n`` hours.

    Parameters
    ----------
    n : int, default 1
        The number of hours represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n hours.

    >>> from pandas.tseries.offsets import Hour
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Hour()
    Timestamp('2022-12-09 16:00:00')
    >>> ts - Hour(4)
    Timestamp('2022-12-09 11:00:00')

    >>> ts + Hour(-4)
    Timestamp('2022-12-09 11:00:00')
    """
    # 纳秒增量表示一个小时的时间
    _nanos_inc = 3600 * 1_000_000_000
    # 前缀表示这是一个小时的偏移
    _prefix = "h"
    # 周期数据类型代码，表示这是一个小时的周期
    _period_dtype_code = PeriodDtypeCode.H
    # C 语言表示，表示小时的时间单位
    _creso = NPY_DATETIMEUNIT.NPY_FR_h


cdef class Minute(Tick):
    """
    Offset ``n`` minutes.

    Parameters
    ----------
    n : int, default 1
        The number of minutes represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n minutes.

    >>> from pandas.tseries.offsets import Minute
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Minute(n=10)
    Timestamp('2022-12-09 15:10:00')
    >>> ts - Minute(n=10)
    Timestamp('2022-12-09 14:50:00')

    >>> ts + Minute(n=-10)
    Timestamp('2022-12-09 14:50:00')
    """
    # 纳秒增量表示一分钟的时间
    _nanos_inc = 60 * 1_000_000_000
    # 前缀表示这是一分钟的偏移
    _prefix = "min"
    # 周期数据类型代码，表示这是一个分钟的周期
    _period_dtype_code = PeriodDtypeCode.T
    # C 语言表示，表示分钟的时间单位
    _creso = NPY_DATETIMEUNIT.NPY_FR_m


cdef class Second(Tick):
    """
    Offset ``n`` seconds.

    Parameters
    ----------
    n : int, default 1
        The number of seconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n seconds.

    >>> from pandas.tseries.offsets import Second
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')


    """
    # 这里没有实际代码，因此不需要注释
    # 在时间戳上添加指定秒数，返回新的时间戳对象
    >>> ts + Second(n=10)
    Timestamp('2022-12-09 15:00:10')
    # 在时间戳上减去指定秒数，返回新的时间戳对象
    >>> ts - Second(n=10)
    Timestamp('2022-12-09 14:59:50')

    # 负数秒数也能被处理，与上一个示例效果相同
    >>> ts + Second(n=-10)
    Timestamp('2022-12-09 14:59:50')
    """
    # 定义纳秒增量为 1_000_000_000
    _nanos_inc = 1_000_000_000
    # 定义时间增量的前缀为 "s"，表示秒
    _prefix = "s"
    # 定义周期数据类型代码为 PeriodDtypeCode.S，表示秒
    _period_dtype_code = PeriodDtypeCode.S
    # 定义 NPY 数据时间单元为 NPY_FR_s，表示秒
    _creso = NPY_DATETIMEUNIT.NPY_FR_s
cdef class Milli(Tick):
    """
    Offset ``n`` milliseconds.

    Parameters
    ----------
    n : int, default 1
        The number of milliseconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n milliseconds.

    >>> from pandas.tseries.offsets import Milli
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Milli(n=10)
    Timestamp('2022-12-09 15:00:00.010000')

    >>> ts - Milli(n=10)
    Timestamp('2022-12-09 14:59:59.990000')

    >>> ts + Milli(n=-10)
    Timestamp('2022-12-09 14:59:59.990000')
    """
    # 定义增量为1毫秒的时间间隔
    _nanos_inc = 1_000_000
    # 时间偏移的单位前缀为“ms”
    _prefix = "ms"
    # 周期数据类型码为L（长周期）
    _period_dtype_code = PeriodDtypeCode.L
    # 时间分辨率设置为毫秒级别
    _creso = NPY_DATETIMEUNIT.NPY_FR_ms


cdef class Micro(Tick):
    """
    Offset ``n`` microseconds.

    Parameters
    ----------
    n : int, default 1
        The number of microseconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n microseconds.

    >>> from pandas.tseries.offsets import Micro
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Micro(n=1000)
    Timestamp('2022-12-09 15:00:00.001000')

    >>> ts - Micro(n=1000)
    Timestamp('2022-12-09 14:59:59.999000')

    >>> ts + Micro(n=-1000)
    Timestamp('2022-12-09 14:59:59.999000')
    """
    # 定义增量为1微秒的时间间隔
    _nanos_inc = 1000
    # 时间偏移的单位前缀为“us”
    _prefix = "us"
    # 周期数据类型码为U（微秒）
    _period_dtype_code = PeriodDtypeCode.U
    # 时间分辨率设置为微秒级别
    _creso = NPY_DATETIMEUNIT.NPY_FR_us


cdef class Nano(Tick):
    """
    Offset ``n`` nanoseconds.

    Parameters
    ----------
    n : int, default 1
        The number of nanoseconds represented.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n nanoseconds.

    >>> from pandas.tseries.offsets import Nano
    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts
    Timestamp('2022-12-09 15:00:00')

    >>> ts + Nano(n=1000)
    Timestamp('2022-12-09 15:00:00.000001')

    >>> ts - Nano(n=1000)
    Timestamp('2022-12-09 14:59:59.999999')

    >>> ts + Nano(n=-1000)
    Timestamp('2022-12-09 14:59:59.999999')
    """
    # 定义增量为1纳秒的时间间隔
    _nanos_inc = 1
    # 时间偏移的单位前缀为“ns”
    _prefix = "ns"
    # 周期数据类型码为N（纳秒）
    _period_dtype_code = PeriodDtypeCode.N
    # 时间分辨率设置为纳秒级别
    _creso = NPY_DATETIMEUNIT.NPY_FR_ns


def delta_to_tick(delta: timedelta) -> Tick:
    # 函数定义：将 timedelta 转换为 Tick 类型
    # 检查时间增量对象是否精确到微秒且没有纳秒属性
    if delta.microseconds == 0 and getattr(delta, "nanoseconds", 0) == 0:
        # 如果仅支持 pandas 的 Timedelta 对象的纳秒属性
        if delta.seconds == 0:
            # 如果时间增量表示天数
            return Day(delta.days)
        else:
            # 计算总秒数，用于后续判断是小时、分钟还是秒
            seconds = delta.days * 86400 + delta.seconds
            if seconds % 3600 == 0:
                # 如果总秒数是3600的倍数，返回小时对象
                return Hour(seconds / 3600)
            elif seconds % 60 == 0:
                # 如果总秒数是60的倍数，返回分钟对象
                return Minute(seconds / 60)
            else:
                # 否则返回秒对象
                return Second(seconds)
    else:
        # 将时间增量转换为纳秒
        nanos = delta_to_nanoseconds(delta)
        if nanos % 1_000_000 == 0:
            # 如果纳秒数是1000000的倍数，返回毫秒对象
            return Milli(nanos // 1_000_000)
        elif nanos % 1000 == 0:
            # 如果纳秒数是1000的倍数，返回微秒对象
            return Micro(nanos // 1000)
        else:  # pragma: no cover
            # 否则返回纳秒对象（注：这段代码块的测试覆盖率应该忽略）
            return Nano(nanos)
# --------------------------------------------------------------------

# cdef class定义一个Cython扩展类型，RelativeDeltaOffset继承自BaseOffset，实现日期偏移量功能
cdef class RelativeDeltaOffset(BaseOffset):
    """
    DateOffset subclass backed by a dateutil relativedelta object.
    """
    # _attributes定义了类的属性元组，包括'n', 'normalize'以及_reladelta_kwds中的所有关键字参数
    _attributes = tuple(["n", "normalize"] + list(_relativedelta_kwds))
    # _adjust_dst设为False，不调整夏令时

    # 初始化方法，设置n、normalize和kwds等属性
    def __init__(self, n=1, normalize=False, **kwds):
        # 调用父类BaseOffset的初始化方法
        BaseOffset.__init__(self, n, normalize)
        # 调用_determine_offset函数确定偏移量和是否使用relativedelta对象
        off, use_rd = _determine_offset(kwds)
        # 将偏移量赋值给_offset属性
        object.__setattr__(self, "_offset", off)
        # 将是否使用relativedelta对象赋值给_use_relativedelta属性
        object.__setattr__(self, "_use_relativedelta", use_rd)
        # 遍历kwds中的键值对，设置实例的属性
        for key in kwds:
            val = kwds[key]
            object.__setattr__(self, key, val)

    # 返回可pickle状态的对象状态
    def __getstate__(self):
        """
        Return a pickleable state
        """
        # 复制实例的__dict__状态
        state = self.__dict__.copy()
        # 设置n和normalize状态
        state["n"] = self.n
        state["normalize"] = self.normalize
        return state

    # 从pickle状态中重建实例
    def __setstate__(self, state):
        """
        Reconstruct an instance from a pickled state
        """

        # 如果state中包含'offset'，则处理旧版本的属性
        if "offset" in state:
            # 旧版本(<0.22.0)使用offset属性而非_offset
            if "_offset" in state:  # pragma: no cover
                raise AssertionError("Unexpected key `_offset`")
            state["_offset"] = state.pop("offset")
            state["kwds"]["offset"] = state["_offset"]

        # 恢复n和normalize属性
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})

        # 更新实例的__dict__状态
        self.__dict__.update(state)

    # 应用装饰器apply_wraps，对应_apply方法
    @apply_wraps
    # _apply方法，接受一个datetime对象other，返回一个datetime对象
    def _apply(self, other: datetime) -> datetime:
        # 初始化other_nanos为0
        other_nanos = 0
        # 如果使用relativedelta对象
        if self._use_relativedelta:
            # 如果other是_Timestamp的实例
            if isinstance(other, _Timestamp):
                # 获取other的nanosecond属性
                other_nanos = other.nanosecond
                # 将other转换为Python的datetime对象
                other = other.to_pydatetime(warn=False)

        # 如果kwds不为空
        if len(self.kwds) > 0:
            # 获取other的tzinfo属性
            tzinfo = getattr(other, "tzinfo", None)
            # 如果tzinfo不为空且使用relativedelta对象
            if tzinfo is not None and self._use_relativedelta:
                # 在UTC时间中执行计算
                other = other.replace(tzinfo=None)

            # 对other进行偏移量计算
            other = other + (self._offset * self.n)

            # 如果存在nanoseconds属性
            if hasattr(self, "nanoseconds"):
                # 添加nanoseconds的时间增量
                other = self.n * Timedelta(nanoseconds=self.nanoseconds) + other
            # 如果other_nanos不为0
            if other_nanos != 0:
                # 添加nanoseconds的时间增量
                other = Timedelta(nanoseconds=other_nanos) + other

            # 如果tzinfo不为空且使用relativedelta对象
            if tzinfo is not None and self._use_relativedelta:
                # 从UTC计算中恢复时区信息
                other = localize_pydatetime(other, tzinfo)

            # 返回Timestamp对象
            return Timestamp(other)
        else:
            # 返回other加上时间增量的结果
            return other + timedelta(self.n)

    # 缓存只读属性
    @cache_readonly
    def _pd_timedelta(self) -> Timedelta:
        # 提取可以转换为 pd.Timedelta 的 _offset 组件

        kwds = self.kwds
        relativedelta_fast = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
            "milliseconds",
        }
        # 只有在使用相对增量（relativedelta）且关键字全部属于 relativedelta_fast 时才有效
        if self._use_relativedelta and set(kwds).issubset(relativedelta_fast):
            td_args = {
                "days",
                "hours",
                "minutes",
                "seconds",
                "microseconds",
                "milliseconds"
            }
            # 从 kwds 中筛选出可以作为 Timedelta 参数的关键字及其对应值
            td_kwds = {
                key: val
                for key, val in kwds.items()
                if key in td_args
            }
            # 如果包含 "weeks" 关键字，则将其转换为等效的天数
            if "weeks" in kwds:
                days = td_kwds.get("days", 0)
                td_kwds["days"] = days + 7 * kwds["weeks"]

            if td_kwds:
                # 根据筛选出的关键字及其值创建 Timedelta 对象
                delta = Timedelta(**td_kwds)
                # 根据关键字 "microseconds" 或 "milliseconds" 调整单位
                if "microseconds" in kwds:
                    delta = delta.as_unit("us")
                elif "milliseconds" in kwds:
                    delta = delta.as_unit("ms")
                else:
                    delta = delta.as_unit("s")
            else:
                delta = Timedelta(0).as_unit("s")

            return delta * self.n

        elif not self._use_relativedelta and hasattr(self, "_offset"):
            # timedelta
            num_nano = getattr(self, "nanoseconds", 0)
            if num_nano != 0:
                rem_nano = Timedelta(nanoseconds=num_nano)
                # 带有纳秒部分的 Timedelta 对象
                delta = Timedelta((self._offset + rem_nano) * self.n)
            else:
                delta = Timedelta(self._offset * self.n)
                # 根据关键字 "microseconds" 或 "milliseconds" 调整单位
                if "microseconds" in kwds:
                    delta = delta.as_unit("us")
                elif "milliseconds" in kwds:
                    delta = delta.as_unit("ms")
                else:
                    delta = delta.as_unit("s")
            return delta

        else:
            # 具有其他关键字的相对增量
            kwd = set(kwds) - relativedelta_fast
            # 抛出未实现的错误，无法对相对增量使用向量化操作
            raise NotImplementedError(
                "DateOffset with relativedelta "
                f"keyword(s) {kwd} not able to be "
                "applied vectorized"
            )

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # 从数据类型中获取时间单位并转换为 reso
        reso = get_unit_from_dtype(dtarr.dtype)
        # 将 dtarr 转换为 numpy 数组
        dt64other = np.asarray(dtarr)

        # 获取时间增量，可能会引发 NotImplementedError 异常
        delta = self._pd_timedelta  # may raise NotImplementedError

        kwds = self.kwds
        # 计算总月数，考虑年份和月份关键字，并乘以 self.n
        months = (kwds.get("years", 0) * 12 + kwds.get("months", 0)) * self.n
        # 如果存在月数，则对 dt64other 进行月份偏移
        if months:
            shifted = shift_months(dt64other.view("i8"), months, reso=reso)
            dt64other = shifted.view(dtarr.dtype)
        # 返回 dtarr 增加 delta 后的结果
        return dt64other + delta
    # 判断给定的日期时间对象是否处于偏移状态
    def is_on_offset(self, dt: datetime) -> bool:
        # 如果设置了归一化标志并且日期时间对象未被归一化，则返回假
        if self.normalize and not _is_normalized(dt):
            return False
        # 否则，返回真
        return True
# 定义一个元类，用于使所有 BaseOffset 的子类表现为从 DateOffset 继承的关系（这对向后兼容性很重要）。
class OffsetMeta(type):
    """
    Metaclass that allows us to pretend that all BaseOffset subclasses
    inherit from DateOffset (which is needed for backward-compatibility).
    """

    @classmethod
    # 检查给定对象是否是 BaseOffset 的实例
    def __instancecheck__(cls, obj) -> bool:
        return isinstance(obj, BaseOffset)

    @classmethod
    # 检查给定类是否是 BaseOffset 的子类
    def __subclasscheck__(cls, obj) -> bool:
        return issubclass(obj, BaseOffset)


# TODO: figure out a way to use a metaclass with a cdef class
# 定义一个类 DateOffset，它继承自 RelativeDeltaOffset，并且使用 OffsetMeta 作为元类
class DateOffset(RelativeDeltaOffset, metaclass=OffsetMeta):
    """
    Standard kind of date increment used for a date range.

    Works exactly like the keyword argument form of relativedelta.
    Note that the positional argument form of relativedelta is not
    supported. Use of the keyword n is discouraged-- you would be better
    off specifying n in the keywords you use, but regardless it is
    there for you. n is needed for DateOffset subclasses.

    DateOffset works as follows.  Each offset specify a set of dates
    that conform to the DateOffset.  For example, Bday defines this
    set to be the set of dates that are weekdays (M-F).  To test if a
    date is in the set of a DateOffset dateOffset we can use the
    is_on_offset method: dateOffset.is_on_offset(date).

    If a date is not on a valid date, the rollback and rollforward
    methods can be used to roll the date to the nearest valid date
    before/after the date.

    DateOffsets can be created to move dates forward a given number of
    valid dates.  For example, Bday(2) can be added to a date to move
    it two business days forward.  If the date does not start on a
    valid date, first it is moved to a valid date.  Thus pseudo code
    is::

        def __add__(date):
          date = rollback(date) # does nothing if date is valid
          return date + <n number of periods>

    When a date offset is created for a negative number of periods,
    the date is first rolled forward.  The pseudo code is::

        def __add__(date):
          date = rollforward(date) # does nothing if date is valid
          return date + <n number of periods>

    Zero presents a problem.  Should it roll forward or back?  We
    arbitrarily have it rollforward:

    date + BDay(0) == BDay.rollforward(date)

    Since 0 is a bit weird, we suggest avoiding its use.

    Besides, adding a DateOffsets specified by the singular form of the date
    component can be used to replace certain component of the timestamp.

    Parameters
    ----------
    n : int, default 1
        The number of time periods the offset represents.
        If specified without a temporal pattern, defaults to n days.
    normalize : bool, default False
        Whether to round the result of a DateOffset addition down to the
        previous midnight.
    """
    def __setattr__(self, name, value):
        # 覆盖默认的属性设置方法，阻止修改属性
        raise AttributeError("DateOffset objects are immutable.")
# --------------------------------------------------------------------

cdef class BusinessMixin(SingleConstructorOffset):
    """
    Mixin to business types to provide related functions.
    """

    cdef readonly:
        timedelta _offset
        # Only Custom subclasses use weekmask, holiday, calendar
        object weekmask, holidays, calendar

    def __init__(self, n=1, normalize=False, offset=timedelta(0)):
        # 调用父类的初始化方法，设置基本偏移量和是否归一化的参数
        BaseOffset.__init__(self, n, normalize)
        # 初始化自身的偏移量属性
        self._offset = offset

    cpdef _init_custom(self, weekmask, holidays, calendar):
        """
        Additional __init__ for Custom subclasses.
        """
        # 获取日历和假期信息，以及自定义的周掩码
        calendar, holidays = _get_calendar(
            weekmask=weekmask, holidays=holidays, calendar=calendar
        )
        # 设置周掩码、假期和日历属性
        self.weekmask = weekmask
        self.holidays = holidays
        self.calendar = calendar

    @property
    def offset(self):
        """
        Alias for self._offset.
        """
        # 返回偏移量的别名属性，用于向后兼容
        return self._offset

    def _repr_attrs(self) -> str:
        if self.offset:
            attrs = [f"offset={repr(self.offset)}"]
        else:
            attrs = []
        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    cpdef __setstate__(self, state):
        # 使用 cdef/cpdef 方法设置只读的 _offset 属性
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            # 旧版本 (<0.22.0) 中使用 offset 属性代替 _offset
            self._offset = state.pop("offset")

        if self._prefix.startswith(("C", "c")):
            # 如果是自定义类，则需要设置周掩码和假期信息
            weekmask = state.pop("weekmask")
            holidays = state.pop("holidays")
            calendar, holidays = _get_calendar(weekmask=weekmask,
                                               holidays=holidays,
                                               calendar=None)
            self.weekmask = weekmask
            self.calendar = calendar
            self.holidays = holidays

        # 调用基类的 __setstate__ 方法处理其余状态信息
        BaseOffset.__setstate__(self, state)


cdef class BusinessDay(BusinessMixin):
    """
    DateOffset subclass representing possibly n business days.

    Parameters
    ----------
    n : int, default 1
        The number of days represented.
    normalize : bool, default False
        Normalize start/end dates to midnight.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n business days.

    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts.strftime('%a %d %b %Y %H:%M')
    'Fri 09 Dec 2022 15:00'
    >>> (ts + pd.offsets.BusinessDay(n=5)).strftime('%a %d %b %Y %H:%M')
    'Fri 16 Dec 2022 15:00'
    """
    Passing the parameter ``normalize`` equal to True, you shift the start
    of the next business day to midnight.

    >>> ts = pd.Timestamp(2022, 12, 9, 15)
    >>> ts + pd.offsets.BusinessDay(normalize=True)
    Timestamp('2022-12-12 00:00:00')
    """
    # 定义类变量 _period_dtype_code，指定为 PeriodDtypeCode.B
    _period_dtype_code = PeriodDtypeCode.B
    # 定义类变量 _prefix，赋值为 "B"
    _prefix = "B"
    # 定义类变量 _attributes，包含元素 ["n", "normalize", "offset"]
    _attributes = tuple(["n", "normalize", "offset"])

    # 定义特殊方法 __setstate__，用于从状态中恢复对象的属性
    cpdef __setstate__(self, state):
        # 从状态中取出并设置属性 n
        self.n = state.pop("n")
        # 从状态中取出并设置属性 normalize
        self.normalize = state.pop("normalize")
        # 如果状态中包含 "_offset"，则设置为对象的 _offset 属性
        if "_offset" in state:
            self._offset = state.pop("_offset")
        # 否则，从状态中取出并设置属性 offset 为对象的 _offset
        elif "offset" in state:
            self._offset = state.pop("offset")
        # 从状态中取出并设置属性 _cache，如果不存在则设为空字典
        self._cache = state.pop("_cache", {})

    # 定义方法 _offset_str，返回偏移量的字符串表示形式
    def _offset_str(self) -> str:
        # 定义内部函数 get_str，用于将时间差转换为字符串表示形式
        def get_str(td):
            off_str = ""
            if td.days > 0:
                off_str += str(td.days) + "D"
            if td.seconds > 0:
                s = td.seconds
                hrs = int(s / 3600)
                if hrs != 0:
                    off_str += str(hrs) + "h"
                    s -= hrs * 3600
                mts = int(s / 60)
                if mts != 0:
                    off_str += str(mts) + "Min"
                    s -= mts * 60
                if s != 0:
                    off_str += str(s) + "s"
            if td.microseconds > 0:
                off_str += str(td.microseconds) + "us"
            return off_str
        
        # 检查 self.offset 是否为 PyDelta 类型的对象
        if PyDelta_Check(self.offset):
            zero = timedelta(0, 0, 0)
            # 如果偏移量大于等于零，返回正数形式的偏移量字符串
            if self.offset >= zero:
                off_str = "+" + get_str(self.offset)
            # 否则，返回负数形式的偏移量字符串
            else:
                off_str = "-" + get_str(-self.offset)
            return off_str
        else:
            # 如果不是 PyDelta 类型，直接返回 repr(self.offset)
            return "+" + repr(self.offset)

    # 定义装饰器应用方法 _apply，用于应用业务日和其他日期时间的组合操作
    @apply_wraps
    def _apply(self, other):
        # 如果 other 是 Python 的日期时间对象
        if PyDateTime_Check(other):
            # 取出 self.n 和 other 的星期几
            n = self.n
            wday = other.weekday()

            # 根据星期几和周数调整天数，计算结果
            weeks = n // 5
            days = self._adjust_ndays(wday, weeks)

            result = other + timedelta(days=7 * weeks + days)
            # 如果有偏移量，将偏移量加到结果中
            if self.offset:
                result = result + self.offset
            return result

        # 如果 other 是任意的 timedelta 标量
        elif is_any_td_scalar(other):
            # 将 self.offset 转换为 Timedelta 类型，加上 other
            td = Timedelta(self.offset) + other
            return BusinessDay(
                self.n, offset=td.to_pytimedelta(), normalize=self.normalize
            )
        else:
            # 抛出类型错误，仅支持业务日和日期时间或时间差的组合
            raise ApplyTypeError(
                "Only know how to combine business day with datetime or timedelta."
            )

    # 定义 Cython 方法 _shift_bdays，用于处理业务日的偏移量计算
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef ndarray _shift_bdays(
        self,
        ndarray i8other,
        NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
    ):
        """
        Implementation of BusinessDay.apply_offset.

        Parameters
        ----------
        i8other : const int64_t[:]
            An array of 64-bit integers representing dates or timestamps.
        reso : NPY_DATETIMEUNIT, default NPY_FR_ns
            Resolution of the datetime units to operate on.

        Returns
        -------
        ndarray[int64_t]
            Resulting array of 64-bit integers after applying offsets.
        """
        cdef:
            int periods = self.n  # Number of periods to shift
            Py_ssize_t i, n = i8other.size  # Size of the input array
            ndarray result = cnp.PyArray_EMPTY(
                i8other.ndim, i8other.shape, cnp.NPY_INT64, 0
            )  # Allocate an empty ndarray for the result
            int64_t val, res_val  # Variables to hold current and resulting values
            int wday, days  # Variables for weekday and adjusted days
            npy_datetimestruct dts  # Datetime structure for date manipulation
            int64_t DAY_PERIODS = periods_per_day(reso)  # Number of periods per day
            cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, i8other)  # Multi-iterator for broadcasting

        for i in range(n):
            # Analogous to: val = i8other[i]
            val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # Current value from input array

            if val == NPY_NAT:
                res_val = NPY_NAT  # If input value is NaT (Not a Time), result is NaT as well
            else:
                # Calculate adjusted dates and times similar to BusinessDay.apply
                weeks = periods // 5  # Number of weeks in the given periods
                pandas_datetime_to_datetimestruct(val, reso, &dts)  # Convert Pandas datetime to datetimestruct
                wday = dayofweek(dts.year, dts.month, dts.day)  # Determine weekday of the date

                days = self._adjust_ndays(wday, weeks)  # Adjust number of days based on weekdays and weeks
                res_val = val + (7 * weeks + days) * DAY_PERIODS  # Calculate the resulting value

            # Analogous to: out[i] = res_val
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val  # Assign resulting value to output array

            cnp.PyArray_MultiIter_NEXT(mi)  # Move to the next iteration in the multi-iterator

        return result  # Return the resulting ndarray

    cdef int _adjust_ndays(self, int wday, int weeks):
        cdef:
            int n = self.n  # Number of periods to shift
            int days  # Adjusted number of days

        if n <= 0 and wday > 4:
            # roll forward
            n += 1

        n -= 5 * weeks  # Adjust n by subtracting 5 * weeks

        # n is always >= 0 at this point
        if n == 0 and wday > 4:
            # roll back
            days = 4 - wday  # Calculate days to roll back
        elif wday > 4:
            # roll forward
            days = (7 - wday) + (n - 1)  # Calculate days to roll forward
        elif wday + n <= 4:
            # shift by n days without leaving the current week
            days = n  # Shift by n days within the current week
        else:
            # shift by n days plus 2 to get past the weekend
            days = n + 2  # Shift by n days plus 2 to move past the weekend
        return days  # Return the adjusted number of days

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        i8other = dtarr.view("i8")  # View input array as 64-bit integers
        reso = get_unit_from_dtype(dtarr.dtype)  # Get datetime resolution from input dtype
        res = self._shift_bdays(i8other, reso=reso)  # Shift business days in the input array
        if self.offset:
            res = res.view(dtarr.dtype) + Timedelta(self.offset)  # Add offset if specified
        return res  # Return the resulting array

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False  # Return False if normalization is required and not normalized
        return dt.weekday() < 5  # Return True if the day of the week is less than 5 (Monday to Friday)
    """
    DateOffset subclass representing possibly n business hours.

    Parameters
    ----------
    n : int, default 1
        The number of hours represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    start : str, time, or list of str/time, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, time, or list of str/time, default: "17:00"
        End time of your custom business hour in 24h format.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    You can use the parameter ``n`` to represent a shift of n hours.

    >>> ts = pd.Timestamp(2022, 12, 9, 8)
    >>> ts + pd.offsets.BusinessHour(n=5)
    Timestamp('2022-12-09 14:00:00')

    You can also change the start and the end of business hours.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.BusinessHour(start="11:00")
    Timestamp('2022-08-08 11:00:00')

    >>> from datetime import time as dt_time
    >>> ts = pd.Timestamp(2022, 8, 5, 22)
    >>> ts + pd.offsets.BusinessHour(end=dt_time(19, 0))
    Timestamp('2022-08-08 10:00:00')

    Passing the parameter ``normalize`` equal to True, you shift the start
    of the next business hour to midnight.

    >>> ts = pd.Timestamp(2022, 12, 9, 8)
    >>> ts + pd.offsets.BusinessHour(normalize=True)
    Timestamp('2022-12-09 00:00:00')

    You can divide your business day hours into several parts.

    >>> import datetime as dt
    >>> freq = pd.offsets.BusinessHour(start=["06:00", "10:00", "15:00"],
    ...                                end=["08:00", "12:00", "17:00"])
    >>> pd.date_range(dt.datetime(2022, 12, 9), dt.datetime(2022, 12, 13), freq=freq)
    DatetimeIndex(['2022-12-09 06:00:00', '2022-12-09 07:00:00',
                   '2022-12-09 10:00:00', '2022-12-09 11:00:00',
                   '2022-12-09 15:00:00', '2022-12-09 16:00:00',
                   '2022-12-12 06:00:00', '2022-12-12 07:00:00',
                   '2022-12-12 10:00:00', '2022-12-12 11:00:00',
                   '2022-12-12 15:00:00', '2022-12-12 16:00:00'],
                   dtype='datetime64[ns]', freq='bh')
    """

    _prefix = "bh"
    _anchor = 0
    _attributes = tuple(["n", "normalize", "start", "end", "offset"])
    _adjust_dst = False

    cdef readonly:
        tuple start, end

    def __init__(
            self, n=1, normalize=False, start="09:00", end="17:00", offset=timedelta(0)
    ):
        # 初始化 BusinessHour 对象，设定初始参数
        self.n = n  # 设定业务小时的数量
        self.normalize = normalize  # 是否归一化起始/结束时间到午夜以生成日期范围
        self.start = start  # 自定义业务小时的起始时间
        self.end = end  # 自定义业务小时的结束时间
        self.offset = offset  # 应用的时间偏移量
    ):
        # 使用 BusinessMixin 的构造函数初始化对象
        BusinessMixin.__init__(self, n, normalize, offset)

        # 如果 start 不是列表型态，将其转换为列表
        if np.ndim(start) == 0:
            start = [start]
        # 如果 start 为空列表，则抛出数值错误异常
        if not len(start):
            raise ValueError("Must include at least 1 start time")

        # 如果 end 不是列表型态，将其转换为列表
        if np.ndim(end) == 0:
            end = [end]
        # 如果 end 为空列表，则抛出数值错误异常
        if not len(end):
            raise ValueError("Must include at least 1 end time")

        # 验证 start 和 end 中的每个时间，确保它们是有效的业务时间
        start = np.array([_validate_business_time(x) for x in start])
        end = np.array([_validate_business_time(x) for x in end])

        # 验证输入的一致性，即 start 和 end 数量必须相同
        if len(start) != len(end):
            raise ValueError("number of starting time and ending time must be the same")
        # 记录开放时间段的数量
        num_openings = len(start)

        # 按照开始时间对开始和结束时间进行排序
        index = np.argsort(start)
        start = tuple(start[index])
        end = tuple(end[index])

        # 初始化总秒数为 0，计算每个开放时间段的业务小时总数
        total_secs = 0
        for i in range(num_openings):
            total_secs += self._get_business_hours_by_sec(start[i], end[i])
            # 计算相邻开放时间段之间的非业务时间总数
            total_secs += self._get_business_hours_by_sec(
                end[i], start[(i + 1) % num_openings]
            )
        
        # 如果总秒数不等于一天的秒数，抛出数值错误异常，表示时间段有重叠或相连
        if total_secs != 24 * 60 * 60:
            raise ValueError(
                "invalid starting and ending time(s): "
                "opening hours should not touch or overlap with "
                "one another"
            )

        # 将验证通过的 start 和 end 时间段赋值给对象属性
        self.start = start
        self.end = end

    cpdef __setstate__(self, state):
        # 从状态中获取并重设 start 和 end 属性
        start = state.pop("start")
        start = (start,) if np.ndim(start) == 0 else tuple(start)
        end = state.pop("end")
        end = (end,) if np.ndim(end) == 0 else tuple(end)
        self.start = start
        self.end = end

        # 清除状态中的其他不必要的属性
        state.pop("kwds", {})
        state.pop("next_bday", None)
        # 调用父类的 __setstate__ 方法，传入清理后的状态信息
        BusinessMixin.__setstate__(self, state)

    def _repr_attrs(self) -> str:
        # 调用父类方法获取对象属性的字符串表示
        out = super()._repr_attrs()
        # 使用字符串格式化生成时间段的表示形式
        hours = ",".join(
            f"{st.hour:02d}:{st.minute:02d}-{en.hour:02d}:{en.minute:02d}"
            for st, en in zip(self.start, self.end)
        )
        # 将时间段添加到属性列表中
        attrs = [f"{self._prefix}={hours}"]
        # 将属性列表添加到输出字符串中
        out += ": " + ", ".join(attrs)
        return out

    def _get_business_hours_by_sec(self, start, end):
        """
        Return business hours in a day by seconds.
        """
        # 创建虚拟的 datetime 对象来计算一天内的业务小时数
        dtstart = datetime(2014, 4, 1, start.hour, start.minute)
        # 如果结束时间在开始时间之前，将日期设置为第二天
        day = 1 if start < end else 2
        until = datetime(2014, 4, day, end.hour, end.minute)
        # 返回两个时间点之间的秒数差，即业务小时数
        return int((until - dtstart).total_seconds())
    def _get_closing_time(self, dt: datetime) -> datetime:
        """
        Get the closing time of a business hour interval by its opening time.

        Parameters
        ----------
        dt : datetime
            Opening time of a business hour interval.

        Returns
        -------
        result : datetime
            Corresponding closing time.
        """
        # 遍历 self.start 列表，查找与给定开放时间 dt 相匹配的项
        for i, st in enumerate(self.start):
            # 如果找到与 dt 的小时和分钟相匹配的开放时间
            if st.hour == dt.hour and st.minute == dt.minute:
                # 返回开放时间 dt 加上到关闭时间的时间增量
                return dt + timedelta(
                    seconds=self._get_business_hours_by_sec(st, self.end[i])
                )
        # 如果未找到匹配项，触发断言异常
        assert False

    @cache_readonly
    def next_bday(self):
        """
        Used for moving to next business day.
        """
        # 如果 self.n 大于等于 0
        if self.n >= 0:
            # 设置下一个工作日的偏移量为 1
            nb_offset = 1
        else:
            # 设置下一个工作日的偏移量为 -1
            nb_offset = -1
        # 如果 self._prefix 以 "c" 开头
        if self._prefix.startswith(("c")):
            # 返回一个 CustomBusinessDay 对象，使用指定参数
            return CustomBusinessDay(
                n=nb_offset,
                weekmask=self.weekmask,
                holidays=self.holidays,
                calendar=self.calendar,
            )
        else:
            # 返回一个 BusinessDay 对象，使用指定参数
            return BusinessDay(n=nb_offset)
    def _next_opening_time(self, other, sign=1):
        """
        If self.n and sign have the same sign, return the earliest opening time
        later than or equal to current time.
        Otherwise the latest opening time earlier than or equal to current
        time.

        Opening time always locates on BusinessDay.
        However, closing time may not if business hour extends over midnight.

        Parameters
        ----------
        other : datetime
            Current time.
        sign : int, default 1.
            Either 1 or -1. Going forward in time if it has the same sign as
            self.n. Going backward in time otherwise.

        Returns
        -------
        result : datetime
            Next opening time.
        """
        # 获取最早的开放时间
        earliest_start = self.start[0]
        # 获取最晚的开放时间
        latest_start = self.start[-1]

        # 判断是否与 self.n 的符号相同
        if self.n == 0:
            is_same_sign = sign > 0
        else:
            is_same_sign = self.n * sign >= 0

        # 如果不是营业日，则调整时间到最近的营业日
        if not self.next_bday.is_on_offset(other):
            # 如果今天不是营业日
            other = other + sign * self.next_bday
            if is_same_sign:
                hour, minute = earliest_start.hour, earliest_start.minute
            else:
                hour, minute = latest_start.hour, latest_start.minute
        else:
            # 如果今天是营业日
            if is_same_sign:
                if latest_start < other.time():
                    # 当前时间在今天最晚的开放时间之后
                    other = other + sign * self.next_bday
                    hour, minute = earliest_start.hour, earliest_start.minute
                else:
                    # 找到不早于当前时间的最早开放时间
                    for st in self.start:
                        if other.time() <= st:
                            hour, minute = st.hour, st.minute
                            break
            else:
                if other.time() < earliest_start:
                    # 当前时间在今天最早的开放时间之前
                    other = other + sign * self.next_bday
                    hour, minute = latest_start.hour, latest_start.minute
                else:
                    # 找到不晚于当前时间的最晚开放时间
                    for st in reversed(self.start):
                        if other.time() >= st:
                            hour, minute = st.hour, st.minute
                            break

        # 返回计算后的日期时间对象
        return datetime(other.year, other.month, other.day, hour, minute)
    def _prev_opening_time(self, other: datetime) -> datetime:
        """
        If n is positive, return the latest opening time earlier than or equal
        to current time.
        Otherwise the earliest opening time later than or equal to current
        time.

        Parameters
        ----------
        other : datetime
            Current time.

        Returns
        -------
        result : datetime
            Previous opening time.
        """
        # 调用 _next_opening_time 方法，传入负数参数来获取前一个开放时间
        return self._next_opening_time(other, sign=-1)

    @apply_wraps
    def rollback(self, dt: datetime) -> datetime:
        """
        Roll provided date backward to next offset only if not on offset.
        """
        # 如果提供的日期不在偏移量上
        if not self.is_on_offset(dt):
            # 如果 n 大于等于 0，则向前滚动到上一个开放时间
            if self.n >= 0:
                dt = self._prev_opening_time(dt)
            else:
                # 如果 n 小于 0，则向后滚动到下一个开放时间
                dt = self._next_opening_time(dt)
            # 返回相应的关闭时间
            return self._get_closing_time(dt)
        return dt

    @apply_wraps
    def rollforward(self, dt: datetime) -> datetime:
        """
        Roll provided date forward to next offset only if not on offset.
        """
        # 如果提供的日期不在偏移量上
        if not self.is_on_offset(dt):
            # 如果 n 大于等于 0，则向后滚动到下一个开放时间
            if self.n >= 0:
                return self._next_opening_time(dt)
            else:
                # 如果 n 小于 0，则向前滚动到上一个开放时间
                return self._prev_opening_time(dt)
        return dt

    @apply_wraps
    def is_on_offset(self, dt: datetime) -> bool:
        # 如果需要标准化并且日期时间不标准化，则返回 False
        if self.normalize and not _is_normalized(dt):
            return False

        # 如果日期时间带有时区信息，则将其截断到时区信息
        if dt.tzinfo is not None:
            dt = datetime(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
            )
        
        # 返回 _is_on_offset 方法的结果
        return self._is_on_offset(dt)

    def _is_on_offset(self, dt: datetime) -> bool:
        """
        Slight speedups using calculated values.
        """
        # 根据 n 的值选择前一个或后一个开放时间
        if self.n >= 0:
            op = self._prev_opening_time(dt)
        else:
            op = self._next_opening_time(dt)
        
        # 计算当前时间与前一个或后一个开放时间的时间差（秒数）
        span = (dt - op).total_seconds()
        businesshours = 0
        # 遍历起始时间和结束时间列表，获取当前开放时间段的秒数
        for i, st in enumerate(self.start):
            if op.hour == st.hour and op.minute == st.minute:
                businesshours = self._get_business_hours_by_sec(st, self.end[i])
        
        # 如果时间差小于等于当前开放时间段的秒数，则返回 True；否则返回 False
        if span <= businesshours:
            return True
        else:
            return False
cdef class WeekOfMonthMixin(SingleConstructorOffset):
    """
    Mixin for methods common to WeekOfMonth and LastWeekOfMonth.
    """

    cdef readonly:
        int weekday, week  # 声明只读属性：weekday（星期几）和week（第几周）

    def __init__(self, n=1, normalize=False, weekday=0):
        BaseOffset.__init__(self, n, normalize)  # 调用父类构造函数初始化基本偏移量
        self.weekday = weekday  # 设置对象的星期几属性

        if weekday < 0 or weekday > 6:
            raise ValueError(f"Day must be 0<=day<=6, got {weekday}")  # 如果weekday不在0到6之间，抛出数值错误异常

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        compare_day = self._get_offset_day(other)  # 获取与指定日期的偏移天数进行比较

        months = self.n  # 将月份数赋给变量months
        months = roll_convention(other.day, months, compare_day)  # 根据指定日期的天数、月份数和比较天数进行月份滚动计算

        shifted = shift_month(other, months, "start")  # 将指定日期按照月份滚动后的日期赋给shifted
        to_day = self._get_offset_day(shifted)  # 获取shifted日期的偏移天数
        return _shift_day(shifted, to_day - shifted.day)  # 返回按照偏移天数调整后的日期

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):  # 如果对象要求规范化并且指定日期不规范化，则返回False
            return False
        return dt.day == self._get_offset_day(dt)  # 返回指定日期是否处于偏移天数上

    @property
    def rule_code(self) -> str:
        weekday = int_to_weekday.get(self.weekday, "")  # 获取星期几对应的名称
        if self.week == -1:
            # LastWeekOfMonth
            return f"{self._prefix}-{weekday}"  # 如果是LastWeekOfMonth，返回格式化的规则代码
        return f"{self._prefix}-{self.week + 1}{weekday}"  # 返回格式化的规则代码，包含第几周和星期几的信息



# ----------------------------------------------------------------------
# Year-Based Offset Classes

cdef class YearOffset(SingleConstructorOffset):
    """
    DateOffset that just needs a month.
    """
    _attributes = tuple(["n", "normalize", "month"])  # 声明私有属性：n（偏移量）、normalize（是否规范化）、month（月份）

    # FIXME(cython#4446): python annotation here gives compile-time errors
    # _default_month: int

    cdef readonly:
        int month  # 声明只读属性：month（月份）

    def __init__(self, n=1, normalize=False, month=None):
        BaseOffset.__init__(self, n, normalize)  # 调用父类构造函数初始化基本偏移量

        month = month if month is not None else self._default_month  # 如果传入的月份为None，则使用默认月份
        self.month = month  # 设置对象的月份属性

        if month < 1 or month > 12:
            raise ValueError("Month must go from 1 to 12")  # 如果月份不在1到12之间，抛出数值错误异常

    cpdef __setstate__(self, state):
        self.month = state.pop("month")  # 从状态中恢复月份属性
        self.n = state.pop("n")  # 从状态中恢复偏移量属性
        self.normalize = state.pop("normalize")  # 从状态中恢复规范化属性
        self._cache = {}  # 初始化缓存字典

    @classmethod
    def _from_name(cls, suffix=None):
        kwargs = {}
        if suffix:
            kwargs["month"] = MONTH_TO_CAL_NUM[suffix]  # 如果有后缀，则根据后缀获取对应的月份数字
        return cls(**kwargs)  # 返回使用关键字参数初始化的类实例

    @property
    def rule_code(self) -> str:
        month = MONTH_ALIASES[self.month]  # 获取月份的别名
        return f"{self._prefix}-{month}"  # 返回格式化的规则代码，包含前缀和月份别名信息

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):  # 如果对象要求规范化并且指定日期不规范化，则返回False
            return False
        return dt.month == self.month and dt.day == self._get_offset_day(dt)  # 返回指定日期的月份与偏移天数是否匹配
    # 对日期进行偏移，返回偏移后的日期
    def _apply(self, other: datetime) -> datetime:
        # 使用 roll_qtrday 函数计算年份偏移量
        years = roll_qtrday(other, self.n, self.month, self._day_opt, modby=12)
        # 计算总月数偏移量，包括年份偏移和月份差异
        months = years * 12 + (self.month - other.month)
        # 调用 shift_month 函数，对日期进行月份偏移
        return shift_month(other, months, self._day_opt)

    # 对日期数组进行偏移，返回偏移后的日期数组
    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # 从数组数据类型中获取时间单位
        reso = get_unit_from_dtype(dtarr.dtype)
        # 使用 shift_quarters 函数对日期数组进行季度偏移
        shifted = shift_quarters(
            dtarr.view("i8"), self.n, self.month, self._day_opt, modby=12, reso=reso
        )
        # 返回偏移后的日期数组
        return shifted
cdef class BYearEnd(YearOffset):
    """
    DateOffset increments between the last business day of the year.

    Parameters
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 12
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BYearEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts - BYearEnd()
    Timestamp('2019-12-31 05:01:15')
    >>> ts + BYearEnd()
    Timestamp('2020-12-31 05:01:15')
    >>> ts + BYearEnd(3)
    Timestamp('2022-12-30 05:01:15')
    >>> ts + BYearEnd(-3)
    Timestamp('2017-12-29 05:01:15')
    >>> ts + BYearEnd(month=11)
    Timestamp('2020-11-30 05:01:15')
    """

    _outputName = "BusinessYearEnd"  # 输出名称为 "BusinessYearEnd"
    _default_month = 12  # 默认月份为 12
    _prefix = "BYE"  # 前缀为 "BYE"
    _day_opt = "business_end"  # 选项为业务年末


cdef class BYearBegin(YearOffset):
    """
    DateOffset increments between the first business day of the year.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 1
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BYearBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BYearBegin()
    Timestamp('2021-01-01 05:01:15')
    >>> ts - BYearBegin()
    Timestamp('2020-01-01 05:01:15')
    >>> ts + BYearBegin(-1)
    Timestamp('2020-01-01 05:01:15')
    >>> ts + BYearBegin(2)
    Timestamp('2022-01-03 05:01:15')
    >>> ts + BYearBegin(month=11)
    Timestamp('2020-11-02 05:01:15')
    """

    _outputName = "BusinessYearBegin"  # 输出名称为 "BusinessYearBegin"
    _default_month = 1  # 默认月份为 1
    _prefix = "BYS"  # 前缀为 "BYS"
    _day_opt = "business_start"  # 选项为业务年初


cdef class YearEnd(YearOffset):
    """
    DateOffset increments between calendar year end dates.

    YearEnd goes to the next date which is the end of the year.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 12
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.YearEnd()
    Timestamp('2022-12-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 12, 31)
    >>> ts + pd.offsets.YearEnd()
    Timestamp('2023-12-31 00:00:00')
    """
    # 默认的月份为12，表示年末偏移量
    _default_month = 12
    # 前缀用于标识这个偏移量对象，这里是"YE"
    _prefix = "YE"
    # 表示该偏移量的行为，这里是获取结束日期
    _day_opt = "end"

    # 声明一个只读的私有变量来存储周期数据类型代码
    cdef readonly:
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, month=None):
        # YearEnd 可以作为 Period 的频率，为了提高性能，
        # 在构造函数中定义 _period_dtype_code
        YearOffset.__init__(self, n, normalize, month)
        # 计算出周期数据类型代码，这里用于 PeriodDtypeCode.A 加上月份模 12 的结果
        self._period_dtype_code = PeriodDtypeCode.A + self.month % 12
cdef class YearBegin(YearOffset):
    """
    DateOffset increments between calendar year begin dates.

    YearBegin goes to the next date which is the start of the year.

    Attributes
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    month : int, default 1
        A specific integer for the month of the year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> ts + pd.offsets.YearBegin()
    Timestamp('2023-01-01 00:00:00')

    >>> ts = pd.Timestamp(2023, 1, 1)
    >>> ts + pd.offsets.YearBegin()
    Timestamp('2024-01-01 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.YearBegin(month=2)
    Timestamp('2022-02-01 00:00:00')

    If you want to get the start of the current year:

    >>> ts = pd.Timestamp(2023, 1, 1)
    >>> pd.offsets.YearBegin().rollback(ts)
    Timestamp('2023-01-01 00:00:00')
    """

    _default_month = 1
    _prefix = "YS"
    _day_opt = "start"


# ----------------------------------------------------------------------
# Quarter-Based Offset Classes

cdef class QuarterOffset(SingleConstructorOffset):
    _attributes = tuple(["n", "normalize", "startingMonth"])
    # TODO: Consider combining QuarterOffset and YearOffset __init__ at some
    #       point.  Also apply_index, is_on_offset, rule_code if
    #       startingMonth vs month attr names are resolved

    # FIXME(cython#4446): python annotation here gives compile-time errors
    # _default_starting_month: int
    # _from_name_starting_month: int

    cdef readonly:
        int startingMonth

    def __init__(self, n=1, normalize=False, startingMonth=None):
        # Initialize base class with number of quarters (n) and normalize flag
        BaseOffset.__init__(self, n, normalize)

        # Set starting month to default if not provided
        if startingMonth is None:
            startingMonth = self._default_starting_month
        self.startingMonth = startingMonth

    cpdef __setstate__(self, state):
        # Restore object state from dictionary
        self.startingMonth = state.pop("startingMonth")
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")

    @classmethod
    def _from_name(cls, suffix=None):
        # Create an instance of QuarterOffset from a suffix (e.g., 'Q1')
        kwargs = {}
        if suffix:
            kwargs["startingMonth"] = MONTH_TO_CAL_NUM[suffix]
        else:
            if cls._from_name_starting_month is not None:
                kwargs["startingMonth"] = cls._from_name_starting_month
        return cls(**kwargs)

    @property
    def rule_code(self) -> str:
        # Generate rule code based on starting month and prefix
        month = MONTH_ALIASES[self.startingMonth]
        return f"{self._prefix}-{month}"

    def is_on_offset(self, dt: datetime) -> bool:
        # Check if given datetime is on the quarter offset
        if self.normalize and not _is_normalized(dt):
            return False
        mod_month = (dt.month - self.startingMonth) % 3
        return mod_month == 0 and dt.day == self._get_offset_day(dt)

    @apply_wraps
    # 对日期进行偏移计算，返回偏移后的日期对象
    def _apply(self, other: datetime) -> datetime:
        # 计算月份的偏移量，找到包含 other.month 的日历季度，
        # 例如如果 other.month == 8，则日历季度是 [7月, 8月, 9月]。
        # 然后找到该季度中包含 self 的 on-offset 日期的月份。
        # `months_since` 是将 other.month 调整到该 on-offset 月份所需的月份数。
        months_since = other.month % 3 - self.startingMonth % 3
        # 使用 roll_qtrday 函数计算偏移后的季度数
        qtrs = roll_qtrday(
            other, self.n, self.startingMonth, day_opt=self._day_opt, modby=3
        )
        # 计算总的月份偏移量
        months = qtrs * 3 - months_since
        # 返回偏移后的日期对象
        return shift_month(other, months, self._day_opt)

    # 对日期数组进行偏移计算，返回偏移后的日期数组
    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # 从 dtarr 的数据类型获取单位信息
        reso = get_unit_from_dtype(dtarr.dtype)
        # 使用 shift_quarters 函数对日期数组进行季度偏移计算
        shifted = shift_quarters(
            dtarr.view("i8"),
            self.n,
            self.startingMonth,
            self._day_opt,
            modby=3,
            reso=reso,
        )
        # 返回偏移后的日期数组
        return shifted
# 定义一个名为 BQuarterEnd 的 Cython 类，继承自 QuarterOffset 类。
# 该类表示每个季度最后一个工作日之间的日期增量。

    """
    DateOffset increments between the last business day of each Quarter.

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/30/2007, 6/29/2007, ...

    Attributes
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BQuarterEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BQuarterEnd()
    Timestamp('2020-06-30 05:01:15')
    >>> ts + BQuarterEnd(2)
    Timestamp('2020-09-30 05:01:15')
    >>> ts + BQuarterEnd(1, startingMonth=2)
    Timestamp('2020-05-29 05:01:15')
    >>> ts + BQuarterEnd(startingMonth=2)
    Timestamp('2020-05-29 05:01:15')
    """

    _output_name = "BusinessQuarterEnd"
    _default_starting_month = 3
    _from_name_starting_month = 12
    _prefix = "BQE"
    _day_opt = "business_end"


```    
# 定义一个名为 BQuarterBegin 的 Cython 类，继承自 QuarterOffset 类。
# 该类表示每个季度第一个工作日之间的日期增量。

    """
    DateOffset increments between the first business day of each Quarter.

    startingMonth = 1 corresponds to dates like 1/01/2007, 4/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 5/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 6/01/2007, ...

    Parameters
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> from pandas.tseries.offsets import BQuarterBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BQuarterBegin()
    Timestamp('2020-06-01 05:01:15')
    >>> ts + BQuarterBegin(2)
    Timestamp('2020-09-01 05:01:15')
    >>> ts + BQuarterBegin(startingMonth=2)
    Timestamp('2020-08-03 05:01:15')
    >>> ts + BQuarterBegin(-1)
    Timestamp('2020-03-02 05:01:15')
    """

    _output_name = "BusinessQuarterBegin"
    _default_starting_month = 3
    _from_name_starting_month = 1
    _prefix = "BQS"
    _day_opt = "business_start"


```    
# 定义一个名为 QuarterEnd 的 Cython 类，继承自 QuarterOffset 类。
# 该类表示每个季度结束日期之间的日期增量。

    """
    DateOffset increments between Quarter end dates.

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    # startingMonth变量用于指定季度的起始月份，例如3表示从每年的3月开始算季度
    startingMonth = 3 corresponds to dates like 3/31/2007, 6/30/2007, ...

    Attributes
    ----------
    n : int, default 1
        表示季度的数量。
    normalize : bool, default False
        是否将开始/结束日期标准化到午夜以前再生成日期范围。
    startingMonth : int, default 3
        指定年度中作为季度起始的月份。

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : 标准的日期增量类。

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.QuarterEnd()
    Timestamp('2022-03-31 00:00:00')
    """
    # 默认的季度起始月份
    _default_starting_month = 3
    # 用于标识季度结束的字符串前缀
    _prefix = "QE"
    # 指示季度末尾的日期选项，默认为"end"
    _day_opt = "end"

    cdef readonly:
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, startingMonth=None):
        # 因为QuarterEnd可以用作Period的频率，为了提高性能，在构造时定义_period_dtype_code
        QuarterOffset.__init__(self, n, normalize, startingMonth)
        # 根据指定的startingMonth计算_period_dtype_code，用于表示季度的数据类型代码
        self._period_dtype_code = PeriodDtypeCode.Q_DEC + self.startingMonth % 12
# ----------------------------------------------------------------------
# Quarter-Based Offset Classes

cdef class QuarterBegin(QuarterOffset):
    """
    DateOffset increments between Quarter start dates.

    startingMonth = 1 corresponds to dates like 1/01/2007, 4/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 5/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 6/01/2007, ...

    Parameters
    ----------
    n : int, default 1
        The number of quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    startingMonth : int, default 3
        A specific integer for the month of the year from which we start quarters.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.QuarterBegin()
    Timestamp('2022-03-01 00:00:00')
    """

    _default_starting_month = 3  # 默认的季度开始月份为3
    _from_name_starting_month = 1  # 从名称开始的月份为1
    _prefix = "QS"  # 前缀为"QS"
    _day_opt = "start"  # 日期选项为"start"


# ----------------------------------------------------------------------
# Month-Based Offset Classes

cdef class MonthOffset(SingleConstructorOffset):
    """
    Base class representing a month-based date offset.

    Methods
    -------
    is_on_offset(dt: datetime) -> bool
        Checks if the given datetime is on the offset date.
    _apply(other: datetime) -> datetime
        Applies the offset to the given datetime.
    _apply_array(dtarr: np.ndarray) -> np.ndarray
        Applies the offset to an array of datetimes.

    Attributes
    ----------
    n : int
        The number of months represented by the offset.
    normalize : bool
        Whether to normalize start/end dates to midnight.
    _day_opt : str
        Option specifying the day convention for offsets.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.
    """

    def is_on_offset(self, dt: datetime) -> bool:
        """
        Checks if the given datetime is on the offset date.

        Parameters
        ----------
        dt : datetime
            The datetime to check.

        Returns
        -------
        bool
            True if the datetime is on the offset date, False otherwise.
        """
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day == self._get_offset_day(dt)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        """
        Applies the offset to the given datetime.

        Parameters
        ----------
        other : datetime
            The datetime to which the offset is applied.

        Returns
        -------
        datetime
            The resulting datetime after applying the offset.
        """
        compare_day = self._get_offset_day(other)
        n = roll_convention(other.day, self.n, compare_day)
        return shift_month(other, n, self._day_opt)

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        """
        Applies the offset to an array of datetimes.

        Parameters
        ----------
        dtarr : np.ndarray
            Array of datetimes to which the offset is applied.

        Returns
        -------
        np.ndarray
            Array of datetimes after applying the offset.
        """
        reso = get_unit_from_dtype(dtarr.dtype)
        shifted = shift_months(dtarr.view("i8"), self.n, self._day_opt, reso=reso)
        return shifted

    cpdef __setstate__(self, state):
        """
        Custom method to set object state during unpickling.

        Parameters
        ----------
        state : dict
            State dictionary containing object attributes.

        Notes
        -----
        Removes specific attributes from the state dictionary to ensure proper unpickling.
        """
        state.pop("_use_relativedelta", False)
        state.pop("offset", None)
        state.pop("_offset", None)
        state.pop("kwds", {})

        BaseOffset.__setstate__(self, state)


cdef class MonthEnd(MonthOffset):
    """
    DateOffset of one month end.

    MonthEnd goes to the next date which is an end of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 30)
    >>> ts + pd.offsets.MonthEnd()
    Timestamp('2022-01-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 31)
    >>> ts + pd.offsets.MonthEnd()
    Timestamp('2022-02-28 00:00:00')

    If you want to get the end of the current month:

    >>> ts = pd.Timestamp(2022, 1, 31)
    >>> pd.offsets.MonthEnd().rollforward(ts)
    Timestamp('2022-01-31 00:00:00')
    """

    _period_dtype_code = PeriodDtypeCode.M  # 期间数据类型代码为'M'
    _prefix = "ME"  # 前缀为"ME"
    # 设定一个变量 _day_opt，并赋值为字符串 "end"
    _day_opt = "end"
cdef class MonthBegin(MonthOffset):
    """
    DateOffset of one month at beginning.

    MonthBegin goes to the next date which is a start of the month.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> ts + pd.offsets.MonthBegin()
    Timestamp('2022-12-01 00:00:00')

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> ts + pd.offsets.MonthBegin()
    Timestamp('2023-01-01 00:00:00')

    If you want to get the start of the current month:

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> pd.offsets.MonthBegin().rollback(ts)
    Timestamp('2022-12-01 00:00:00')
    """
    _prefix = "MS"
    _day_opt = "start"


cdef class BusinessMonthEnd(MonthOffset):
    """
    DateOffset increments between the last business day of the month.

    BusinessMonthEnd goes to the next date which is the last business day of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 11, 29)
    >>> ts + pd.offsets.BMonthEnd()
    Timestamp('2022-11-30 00:00:00')

    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> ts + pd.offsets.BMonthEnd()
    Timestamp('2022-12-30 00:00:00')

    If you want to get the end of the current business month:

    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> pd.offsets.BMonthEnd().rollforward(ts)
    Timestamp('2022-11-30 00:00:00')
    """
    _prefix = "BME"
    _day_opt = "business_end"


cdef class BusinessMonthBegin(MonthOffset):
    """
    DateOffset of one month at the first business day.

    BusinessMonthBegin goes to the next date which is the first business day
    of the month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 11, 30)
    >>> ts + pd.offsets.BMonthBegin()
    Timestamp('2022-12-01 00:00:00')

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> ts + pd.offsets.BMonthBegin()
    Timestamp('2023-01-02 00:00:00')

    If you want to get the start of the current business month:

    >>> ts = pd.Timestamp(2022, 12, 1)
    >>> pd.offsets.BMonthBegin().rollback(ts)
    Timestamp('2022-12-01 00:00:00')
    """
    _prefix = "BMS"
    # 定义一个变量 _day_opt，并初始化为字符串 "business_start"
    _day_opt = "business_start"
# ---------------------------------------------------------------------
# Semi-Month Based Offsets

# 定义一个 Cython 类 `SemiMonthOffset`，继承自 `SingleConstructorOffset`
cdef class SemiMonthOffset(SingleConstructorOffset):
    # 默认的月中日期为 15 号
    _default_day_of_month = 15
    # 允许的最小月中日期为 2 号
    _min_day_of_month = 2
    # 定义只读属性列表 `_attributes`，包含了 `n`、`normalize`、`day_of_month`
    _attributes = tuple(["n", "normalize", "day_of_month"])

    cdef readonly:
        # 月中日期的整数表示
        int day_of_month

    # 初始化方法，接受参数 `n`、`normalize`、`day_of_month`
    def __init__(self, n=1, normalize=False, day_of_month=None):
        # 调用父类的初始化方法 `BaseOffset.__init__`，传递参数 `n` 和 `normalize`
        BaseOffset.__init__(self, n, normalize)

        # 如果未提供 `day_of_month`，则使用默认值 `_default_day_of_month`
        if day_of_month is None:
            day_of_month = self._default_day_of_month

        # 将 `day_of_month` 转换为整数类型，并检查其范围
        self.day_of_month = int(day_of_month)
        if not self._min_day_of_month <= self.day_of_month <= 27:
            # 如果 `day_of_month` 超出允许范围，则抛出 ValueError 异常
            raise ValueError(
                "day_of_month must be "
                f"{self._min_day_of_month}<=day_of_month<=27, "
                f"got {self.day_of_month}"
            )

    # 定义 `__setstate__` 方法，用于反序列化时恢复对象状态
    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.day_of_month = state.pop("day_of_month")

    # 类方法 `_from_name`，根据后缀创建对象
    @classmethod
    def _from_name(cls, suffix=None):
        return cls(day_of_month=suffix)

    # 属性方法 `rule_code`，返回规则代码的字符串表示
    @property
    def rule_code(self) -> str:
        suffix = f"-{self.day_of_month}"
        return self._prefix + suffix

    # 方法 `_apply`，应用偏移量到日期 `other`，返回新的日期 `datetime`
    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # 判断当前对象是否为 `SemiMonthBegin`
        is_start = isinstance(self, SemiMonthBegin)

        # 调整 `other` 到 `self.day_of_month`，根据需要递增 `n`
        n = roll_convention(other.day, self.n, self.day_of_month)

        # 获取 `other` 当前月份的天数
        days_in_month = get_days_in_month(other.year, other.month)

        # 如果是 `SemiMonthBegin` 并且 `self.n` 小于等于 0 并且 `other.day` 等于 1
        if is_start and (self.n <= 0 and other.day == 1):
            n -= 1
        # 如果是 `SemiMonthEnd` 并且 `self.n` 大于 0 并且 `other.day` 等于本月最后一天
        elif (not is_start) and (self.n > 0 and other.day == days_in_month):
            n += 1

        # 如果是 `SemiMonthBegin`，计算月份偏移量和目标日期 `to_day`
        if is_start:
            months = n // 2 + n % 2
            to_day = 1 if n % 2 else self.day_of_month
        # 如果是 `SemiMonthEnd`，计算月份偏移量和目标日期 `to_day`
        else:
            months = n // 2
            to_day = 31 if n % 2 else self.day_of_month

        # 调用 `shift_month` 方法，将 `other` 按计算出的月份偏移和目标日期 `to_day` 进行调整
        return shift_month(other, months, to_day)

    # 设置 Cython 的 `wraparound` 和 `boundscheck` 优化标记
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        cdef:
            # 将输入数组以int64视图存储为另一个数组
            ndarray i8other = dtarr.view("i8")
            # 初始化循环变量和计数器
            Py_ssize_t i, count = dtarr.size
            # 声明整数变量
            int64_t val, res_val
            # 创建一个空的输出数组，类型为int64
            ndarray out = cnp.PyArray_EMPTY(
                i8other.ndim, i8other.shape, cnp.NPY_INT64, 0
            )
            # 声明日期时间结构体变量
            npy_datetimestruct dts
            # 声明整数变量和一些初始值，从self对象获取
            int months, to_day, nadj, n = self.n
            int days_in_month, day, anchor_dom = self.day_of_month
            # 检查self对象是否为SemiMonthBegin类型
            bint is_start = isinstance(self, SemiMonthBegin)
            # 从输入数组的dtype获取时间单位
            NPY_DATETIMEUNIT reso = get_unit_from_dtype(dtarr.dtype)
            # 创建用于多重迭代的广播对象
            cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, i8other)

        with nogil:
            for i in range(count):
                # 类比操作：val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    # 如果值为NPY_NAT，则结果值也为NPY_NAT
                    res_val = NPY_NAT

                else:
                    # 将Pandas日期时间转换为日期时间结构体
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    day = dts.day

                    # 根据特定约定调整，确保始终参考self.day_of_month，可能会增加或减少n
                    nadj = roll_convention(day, n, anchor_dom)

                    # 获取当前月份的天数
                    days_in_month = get_days_in_month(dts.year, dts.month)
                    
                    # 对于SemiMonthBegin，当day == 1且n <= 0时，总是需要调整n
                    if is_start and (n <= 0 and day == 1):
                        nadj -= 1
                    # 对于SemiMonthEnd，当day == 当月天数且n > 0时，总是需要调整n
                    elif (not is_start) and (n > 0 and day == days_in_month):
                        nadj += 1

                    if is_start:
                        # 参考：SemiMonthBegin._apply
                        months = nadj // 2 + nadj % 2
                        to_day = 1 if nadj % 2 else anchor_dom

                    else:
                        # 参考：SemiMonthEnd._apply
                        months = nadj // 2
                        to_day = 31 if nadj % 2 else anchor_dom

                    # 增加指定月份数到年份和月份
                    dts.year = year_add_months(dts, months)
                    dts.month = month_add_months(dts, months)
                    # 更新当前月份的天数
                    days_in_month = get_days_in_month(dts.year, dts.month)
                    # 确保日期不超过当月最大天数
                    dts.day = min(to_day, days_in_month)

                    # 将日期时间结构体转换为datetime值
                    res_val = npy_datetimestruct_to_datetime(reso, &dts)

                # 类比操作：out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                # 移动到下一个迭代位置
                cnp.PyArray_MultiIter_NEXT(mi)

        # 返回处理后的输出数组
        return out
cdef class SemiMonthEnd(SemiMonthOffset):
    """
    Two DateOffset's per month repeating on the last day of the month & day_of_month.

    Attributes
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    day_of_month : int, {1, 3,...,27}, default 15
        A specific integer for the day of the month.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 14)
    >>> ts + pd.offsets.SemiMonthEnd()
    Timestamp('2022-01-15 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 15)
    >>> ts + pd.offsets.SemiMonthEnd()
    Timestamp('2022-01-31 00:00:00')

    >>> ts = pd.Timestamp(2022, 1, 31)
    >>> ts + pd.offsets.SemiMonthEnd()
    Timestamp('2022-02-15 00:00:00')

    If you want to get the result for the current month:

    >>> ts = pd.Timestamp(2022, 1, 15)
    >>> pd.offsets.SemiMonthEnd().rollforward(ts)
    Timestamp('2022-01-15 00:00:00')
    """
    _prefix = "SME"  # 定义类别名称前缀为"SME"
    _min_day_of_month = 1  # 定义最小的日期为月初的第一天

    def is_on_offset(self, dt: datetime) -> bool:
        """
        Determine if a given datetime is on this offset.

        Parameters
        ----------
        dt : datetime
            The datetime to check.

        Returns
        -------
        bool
            True if the datetime matches the offset, False otherwise.
        """
        if self.normalize and not _is_normalized(dt):
            return False  # 如果需要归一化且日期未被归一化，则返回False
        days_in_month = get_days_in_month(dt.year, dt.month)
        return dt.day in (self.day_of_month, days_in_month)  # 判断日期是否在该偏移量上


cdef class SemiMonthBegin(SemiMonthOffset):
    """
    Two DateOffset's per month repeating on the first day of the month & day_of_month.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    day_of_month : int, {1, 3,...,27}, default 15
        A specific integer for the day of the month.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.SemiMonthBegin()
    Timestamp('2022-01-15 00:00:00')
    """

    _prefix = "SMS"  # 定义类别名称前缀为"SMS"

    def is_on_offset(self, dt: datetime) -> bool:
        """
        Determine if a given datetime is on this offset.

        Parameters
        ----------
        dt : datetime
            The datetime to check.

        Returns
        -------
        bool
            True if the datetime matches the offset, False otherwise.
        """
        if self.normalize and not _is_normalized(dt):
            return False  # 如果需要归一化且日期未被归一化，则返回False
        return dt.day in (1, self.day_of_month)  # 判断日期是否在该偏移量上


# ---------------------------------------------------------------------
# Week-Based Offset Classes


cdef class Week(SingleConstructorOffset):
    """
    Weekly offset.

    Parameters
    ----------
    n : int, default 1
        The number of weeks represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int or None, default None
        Always generate specific day of week.
        0 for Monday and 6 for Sunday.

    See Also
    --------
    pd.tseries.offsets.WeekOfMonth :
     Describes monthly dates like, the Tuesday of the
     2nd week of each month.

    Examples
    --------

    >>> date_object = pd.Timestamp("2023-01-13")
    >>> date_object
    Timestamp('2023-01-13 00:00:00')

    >>> date_plus_one_week = date_object + pd.tseries.offsets.Week(n=1)
    >>> date_plus_one_week
    Timestamp('2023-01-20 00:00:00')
    """
    # 创建一个时间戳对象，表示特定日期和时间
    Timestamp('2023-01-20 00:00:00')

    # 使用 Pandas 的日期偏移功能，计算下一个周一的日期
    >>> date_next_monday = date_object + pd.tseries.offsets.Week(weekday=0)
    >>> date_next_monday
    # 打印结果为下一个周一的时间戳
    Timestamp('2023-01-16 00:00:00')

    # 使用 Pandas 的日期偏移功能，计算下一个周日的日期
    >>> date_next_sunday = date_object + pd.tseries.offsets.Week(weekday=6)
    >>> date_next_sunday
    # 打印结果为下一个周日的时间戳
    Timestamp('2023-01-15 00:00:00')
    """

    # 偏移量，表示一周
    _inc = timedelta(weeks=1)
    # 前缀，表示周偏移量的标识
    _prefix = "W"
    # 属性元组，包含偏移量的属性信息
    _attributes = tuple(["n", "normalize", "weekday"])

    # 定义只读变量和属性
    cdef readonly:
        object weekday  # int or None  # 周几，可以是整数或None
        int _period_dtype_code  # 周期数据类型代码

    # 初始化方法，接受参数 n、normalize 和 weekday
    def __init__(self, n=1, normalize=False, weekday=None):
        # 调用父类 BaseOffset 的初始化方法
        BaseOffset.__init__(self, n, normalize)
        # 设置 weekday 属性
        self.weekday = weekday

        # 如果指定了 weekday
        if self.weekday is not None:
            # 如果 weekday 超出 0-6 的范围，抛出值错误
            if self.weekday < 0 or self.weekday > 6:
                raise ValueError(f"Day must be 0<=day<=6, got {self.weekday}")

            # 计算周期数据类型代码
            self._period_dtype_code = PeriodDtypeCode.W_SUN + (weekday + 1) % 7

    # 设置对象状态的方法，从状态中恢复对象
    cpdef __setstate__(self, state):
        # 恢复 n、normalize 和 weekday 属性
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        # 恢复缓存属性
        self._cache = state.pop("_cache", {})

    # 应用函数修饰器，将函数应用到其他对象上
    @apply_wraps
    def _apply(self, other):
        # 如果未指定 weekday，则简单相加偏移量
        if self.weekday is None:
            return other + self.n * self._inc

        # 如果 other 不是日期时间对象，则引发类型错误
        if not PyDateTime_Check(other):
            raise TypeError(
                f"Cannot add {type(other).__name__} to {type(self).__name__}"
            )

        # 计算要添加的偏移量
        k = self.n
        otherDay = other.weekday()
        if otherDay != self.weekday:
            other = other + timedelta((self.weekday - otherDay) % 7)
            if k > 0:
                k -= 1

        return other + timedelta(weeks=k)

    # 应用函数到数组的方法，返回处理后的日期数组
    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray:
        # 如果未指定 weekday，则简单相加一周的偏移量
        if self.weekday is None:
            td = timedelta(days=7 * self.n)
            unit = np.datetime_data(dtarr.dtype)[0]
            td64 = np.timedelta64(td, unit)
            return dtarr + td64
        else:
            # 否则，根据数据类型获取单位，进行处理
            reso = get_unit_from_dtype(dtarr.dtype)
            i8other = dtarr.view("i8")
            return self._end_apply_index(i8other, reso=reso)

    # Cython 的性能优化装饰器，关闭数组索引的包装和边界检查
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef ndarray _end_apply_index(self, ndarray i8other, NPY_DATETIMEUNIT reso):
        """
        Add self to the given DatetimeIndex, specialized for case where
        self.weekday is non-null.

        Parameters
        ----------
        i8other : const int64_t[:]
            另一个整数数组，用于计算结果
        reso : NPY_DATETIMEUNIT
            表示时间分辨率的枚举类型

        Returns
        -------
        ndarray[int64_t]
            包含结果的整数数组
        """
        cdef:
            Py_ssize_t i, count = i8other.size
            int64_t val, res_val
            ndarray out = cnp.PyArray_EMPTY(
                i8other.ndim, i8other.shape, cnp.NPY_INT64, 0
            )
            npy_datetimestruct dts
            int wday, days, weeks, n = self.n
            int anchor_weekday = self.weekday
            int64_t DAY_PERIODS = periods_per_day(reso)
            cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, i8other)

        with nogil:
            for i in range(count):
                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    wday = dayofweek(dts.year, dts.month, dts.day)

                    days = 0
                    weeks = n
                    if wday != anchor_weekday:
                        days = (anchor_weekday - wday) % 7
                        if weeks > 0:
                            weeks -= 1

                    res_val = val + (7 * weeks + days) * DAY_PERIODS

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

        return out

    def is_on_offset(self, dt: datetime) -> bool:
        """
        判断给定的日期时间是否符合当前偏移规则。

        Parameters
        ----------
        dt : datetime
            要检查的日期时间对象

        Returns
        -------
        bool
            如果日期时间符合偏移规则，则返回 True；否则返回 False
        """
        if self.normalize and not _is_normalized(dt):
            return False
        elif self.weekday is None:
            return True
        return dt.weekday() == self.weekday

    @property
    def rule_code(self) -> str:
        """
        获取规则的代码表示形式。

        Returns
        -------
        str
            规则的代码表示形式，由前缀和可选的星期几后缀组成
        """
        suffix = ""
        if self.weekday is not None:
            weekday = int_to_weekday[self.weekday]
            suffix = f"-{weekday}"
        return self._prefix + suffix

    @classmethod
    def _from_name(cls, suffix=None):
        """
        从规则名称创建对象的类方法。

        Parameters
        ----------
        suffix : str or None
            规则名称的后缀，用于确定星期几

        Returns
        -------
        cls
            根据名称创建的对象实例
        """
        if not suffix:
            weekday = None
        else:
            weekday = weekday_to_int[suffix]
        return cls(weekday=weekday)
cdef class WeekOfMonth(WeekOfMonthMixin):
    """
    Describes monthly dates like "the Tuesday of the 2nd week of each month".

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    week : int {0, 1, 2, 3, ...}, default 0
        A specific integer for the week of the month.
        e.g. 0 is 1st week of month, 1 is the 2nd week, etc.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.WeekOfMonth()
    Timestamp('2022-01-03 00:00:00')
    """

    _prefix = "WOM"
    _attributes = tuple(["n", "normalize", "week", "weekday"])

    def __init__(self, n=1, normalize=False, week=0, weekday=0):
        """
        Initialize WeekOfMonth object.

        Parameters
        ----------
        n : int, optional
            Number of months represented, default is 1.
        normalize : bool, optional
            Whether to normalize start/end dates to midnight, default is False.
        week : int, optional
            Week index within the month (0 for 1st week, 1 for 2nd, etc.), default is 0.
        weekday : int, optional
            Day of the week index (0 for Monday, 1 for Tuesday, etc.), default is 0.
        """
        WeekOfMonthMixin.__init__(self, n, normalize, weekday)
        self.week = week

        if self.week < 0 or self.week > 3:
            raise ValueError(f"Week must be 0<=week<=3, got {self.week}")

    cpdef __setstate__(self, state):
        """
        Set the state of the WeekOfMonth object during deserialization.

        Parameters
        ----------
        state : dict
            Dictionary containing serialized state information.
        """
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.week = state.pop("week")

    def _get_offset_day(self, other: datetime) -> int:
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the self.week'th such day in the month.

        Parameters
        ----------
        other : datetime
            Reference date to calculate the offset from.

        Returns
        -------
        day : int
            Day of the month that matches the criteria.
        """
        mstart = datetime(other.year, other.month, 1)
        wday = mstart.weekday()
        shift_days = (self.weekday - wday) % 7
        return 1 + shift_days + self.week * 7

    @classmethod
    def _from_name(cls, suffix=None):
        """
        Create a WeekOfMonth object based on a name suffix.

        Parameters
        ----------
        suffix : str
            Suffix indicating week and weekday information (e.g., '1T' for 2nd week, Tuesday).

        Returns
        -------
        cls
            Initialized WeekOfMonth object.
        
        Raises
        ------
        ValueError
            If suffix is missing.
        """
        if not suffix:
            raise ValueError(f"Prefix {repr(cls._prefix)} requires a suffix.")
        # only one digit weeks (1 --> week 0, 2 --> week 1, etc.)
        week = int(suffix[0]) - 1
        weekday = weekday_to_int[suffix[1:]]
        return cls(week=week, weekday=weekday)


cdef class LastWeekOfMonth(WeekOfMonthMixin):
    """
    Describes monthly dates in last week of month.

    For example "the last Tuesday of each month".

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    # 创建一个 Pandas Timestamp 对象，表示 2022 年 1 月 1 日
    >>> ts + pd.offsets.LastWeekOfMonth()
    # 将 Timestamp 对象 ts 增加一个 LastWeekOfMonth 偏移量，得到最后一个月的最后一周的日期
    Timestamp('2022-01-31 00:00:00')
    # 返回结果为 Timestamp 对象，表示 2022 年 1 月的最后一天

    """

    _prefix = "LWOM"
    # 类属性，表示偏移类的前缀字符串
    _attributes = tuple(["n", "normalize", "weekday"])
    # 类属性，元组，包含初始化参数的名称列表

    def __init__(self, n=1, normalize=False, weekday=0):
        # 构造函数，初始化对象
        WeekOfMonthMixin.__init__(self, n, normalize, weekday)
        # 调用父类 WeekOfMonthMixin 的构造函数，传递初始化参数
        self.week = -1
        # 初始化对象的属性 self.week 为 -1

        if self.n == 0:
            raise ValueError("N cannot be 0")
        # 如果初始化参数 self.n 为 0，则引发值错误异常

    cpdef __setstate__(self, state):
        # Cython 特定的函数修饰符，用于设置对象状态
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.week = -1
        # 设置对象的状态属性，并初始化 self.week 为 -1

    def _get_offset_day(self, other: datetime) -> int:
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the last such day in the month.

        Parameters
        ----------
        other: datetime

        Returns
        -------
        day: int
        """
        dim = get_days_in_month(other.year, other.month)
        # 调用 get_days_in_month 函数获取给定年份和月份的天数
        mend = datetime(other.year, other.month, dim)
        # 创建一个日期对象，表示给定年份和月份的最后一天
        wday = mend.weekday()
        # 获取这一天是星期几的索引
        shift_days = (wday - self.weekday) % 7
        # 计算需要偏移的天数，使得日期与 self.weekday 相同，并且是本月的最后一个这样的日期
        return dim - shift_days
        # 返回本月中与 self.weekday 相同的最后一个日期的天数

    @classmethod
    def _from_name(cls, suffix=None):
        # 类方法，根据后缀创建对象实例
        if not suffix:
            raise ValueError(f"Prefix {repr(cls._prefix)} requires a suffix.")
        # 如果没有提供后缀参数，引发值错误异常，指明需要提供后缀
        weekday = weekday_to_int[suffix]
        # 根据后缀查找并获取对应的星期索引
        return cls(weekday=weekday)
        # 使用获取的星期索引创建并返回类的实例
# ---------------------------------------------------------------------
# Special Offset Classes

# FY5253Mixin 类继承自 SingleConstructorOffset，定义了一个特殊的偏移类

cdef class FY5253Mixin(SingleConstructorOffset):
    cdef readonly:
        int startingMonth  # 起始月份，整数类型
        int weekday  # 每周的某一天，整数类型
        str variation  # 变体，字符串类型，可选值为 "nearest" 或 "last"

    def __init__(
        self, n=1, normalize=False, weekday=0, startingMonth=1, variation="nearest"
    ):
        # 初始化方法，设置初始属性值
        BaseOffset.__init__(self, n, normalize)
        self.startingMonth = startingMonth
        self.weekday = weekday
        self.variation = variation

        if self.n == 0:
            raise ValueError("N cannot be 0")  # 如果 n 为 0，则引发值错误异常

        if self.variation not in ["nearest", "last"]:
            raise ValueError(f"{self.variation} is not a valid variation")  
            # 如果 variation 不是 "nearest" 或 "last"，则引发值错误异常

    cpdef __setstate__(self, state):
        # 设置状态方法，从状态中恢复对象的属性
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.variation = state.pop("variation")

    # --------------------------------------------------------------------
    # Name-related methods

    @property
    def rule_code(self) -> str:
        # 获取规则代码属性，返回规则代码的字符串表示
        prefix = self._prefix
        suffix = self.get_rule_code_suffix()
        return f"{prefix}-{suffix}"

    def _get_suffix_prefix(self) -> str:
        # 获取后缀前缀方法，根据 variation 返回相应的前缀
        if self.variation == "nearest":
            return "N"
        else:
            return "L"

    def get_rule_code_suffix(self) -> str:
        # 获取规则代码后缀方法，返回由前缀、起始月份和每周某天组成的字符串后缀
        prefix = self._get_suffix_prefix()
        month = MONTH_ALIASES[self.startingMonth]
        weekday = int_to_weekday[self.weekday]
        return f"{prefix}-{month}-{weekday}"


cdef class FY5253(FY5253Mixin):
    """
    Describes 52-53 week fiscal year. This is also known as a 4-4-5 calendar.

    It is used by companies that desire that their
    fiscal year always end on the same day of the week.

    It is a method of managing accounting periods.
    It is a common calendar structure for some industries,
    such as retail, manufacturing and parking industry.

    For more information see:
    https://en.wikipedia.org/wiki/4-4-5_calendar

    The year may either:

    - end on the last X day of the Y month.
    - end on the last X day closest to the last day of the Y month.

    X is a specific day of the week.
    Y is a certain month of the year

    Parameters
    ----------
    n : int
        The number of fiscal years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    startingMonth : int {1, 2, ... 12}, default 1
        The month in which the fiscal year ends.

    """

    # FY5253 类，描述了 52-53 周财政年度，也称为 4-4-5 日历
    """
    variation : str, default "nearest"
        Method of employing 4-4-5 calendar.

        There are two options:

        - "nearest" means year end is **weekday** closest to last day of month in year.
        - "last" means year end is final **weekday** of the final month in fiscal year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below the default parameters give the next 52-53 week fiscal year.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253()
    Timestamp('2022-01-31 00:00:00')

    By the parameter ``startingMonth`` we can specify
    the month in which fiscal years end.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253(startingMonth=3)
    Timestamp('2022-03-28 00:00:00')

    52-53 week fiscal year can be specified by
    ``weekday`` and ``variation`` parameters.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253(weekday=5, startingMonth=12, variation="last")
    Timestamp('2022-12-31 00:00:00')
    """
    
    # 定义私有属性前缀
    _prefix = "RE"
    # 定义只读属性列表
    _attributes = tuple(["n", "normalize", "weekday", "startingMonth", "variation"])

    # 判断给定的日期是否符合当前偏移量的要求
    def is_on_offset(self, dt: datetime) -> bool:
        # 如果开启了标准化选项并且日期未标准化，则返回 False
        if self.normalize and not _is_normalized(dt):
            return False
        # 将日期规范化到年月日
        dt = datetime(dt.year, dt.month, dt.day)
        # 获取当前偏移量下的年末日期
        year_end = self.get_year_end(dt)

        # 根据 variation 参数的值进行不同的判断逻辑
        if self.variation == "nearest":
            # 需要检查“本”财政年度和上一个月的年末日期是否等于给定日期
            return year_end == dt or self.get_year_end(shift_month(dt, -1, None)) == dt
        else:
            # 返回当前偏移量下的年末日期是否等于给定日期
            return year_end == dt

    # 应用装饰器
    @apply_wraps
    # 应用函数，将给定的日期时间规范化后应用处理
    def _apply(self, other: datetime) -> datetime:
        # 将输入的日期时间转换为 Timestamp 对象并规范化
        norm = Timestamp(other).normalize()

        # 获取当前对象的年份变量
        n = self.n
        # 计算前一年的年末日期
        prev_year = self.get_year_end(datetime(other.year - 1, self.startingMonth, 1))
        # 计算当前年的年末日期
        cur_year = self.get_year_end(datetime(other.year, self.startingMonth, 1))
        # 计算下一年的年末日期
        next_year = self.get_year_end(datetime(other.year + 1, self.startingMonth, 1))

        # 将年末日期转换为输入日期时区下的本地时间
        prev_year = localize_pydatetime(prev_year, other.tzinfo)
        cur_year = localize_pydatetime(cur_year, other.tzinfo)
        next_year = localize_pydatetime(next_year, other.tzinfo)

        # 注意：由于 next_year.year == other.year + 1，因此总是满足 other < next_year
        if norm == prev_year:
            # 如果规范化后的日期等于前一年的年末日期，则 n 减少 1
            n -= 1
        elif norm == cur_year:
            # 如果规范化后的日期等于当前年的年末日期，则保持不变
            pass
        elif n > 0:
            if norm < prev_year:
                # 如果规范化后的日期早于前一年的年末日期，则 n 减少 2
                n -= 2
            elif prev_year < norm < cur_year:
                # 如果规范化后的日期介于前一年和当前年的年末日期之间，则 n 减少 1
                n -= 1
            elif cur_year < norm < next_year:
                # 如果规范化后的日期介于当前年和下一年的年末日期之间，则保持不变
                pass
        else:
            if cur_year < norm < next_year:
                # 如果规范化后的日期介于当前年和下一年的年末日期之间，则 n 增加 1
                n += 1
            elif prev_year < norm < cur_year:
                # 如果规范化后的日期介于前一年和当前年的年末日期之间，则保持不变
                pass
            elif (
                norm.year == prev_year.year
                and norm < prev_year
                and prev_year - norm <= timedelta(6)
            ):
                # 特例处理：当 next_year.year == cur_year.year 时可能出错的情况
                # 例如：prev_year == datetime(2004, 1, 3)，other == datetime(2004, 1, 1)
                n -= 1
            else:
                # 如果不满足以上条件，则断言出错
                assert False

        # 计算偏移后的日期时间，生成新的日期时间对象
        shifted = datetime(other.year + n, self.startingMonth, 1)
        # 获取新日期时间对象所对应的年末日期
        result = self.get_year_end(shifted)
        # 将结果调整为与输入日期时间对象相同的时间
        result = datetime(
            result.year,
            result.month,
            result.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        # 返回最终处理后的日期时间对象
        return result

    # 获取给定日期时间所在年的年末日期
    def get_year_end(self, dt: datetime) -> datetime:
        # 断言输入日期时间对象不含时区信息

        assert dt.tzinfo is None

        # 获取指定年份和起始月份的天数
        dim = get_days_in_month(dt.year, self.startingMonth)
        # 构建目标日期时间对象为当年最后一天
        target_date = datetime(dt.year, self.startingMonth, dim)
        # 计算目标日期时间与当前对象的周几差异
        wkday_diff = self.weekday - target_date.weekday()

        if wkday_diff == 0:
            # 如果差异为 0，则年末日期即为目标日期
            return target_date

        if self.variation == "last":
            # 如果 variation 为 "last"，计算需要向前调整的天数
            days_forward = (wkday_diff % 7) - 7
            # 因为 days_forward 总是负数，所以结果仍然位于与 dt 同一年
            return target_date + timedelta(days=days_forward)
        else:
            # 如果 variation 不为 "last"，则为 "nearest"，计算需要向前调整的天数
            days_forward = wkday_diff % 7
            if days_forward <= 3:
                # 如果下一个 self.weekday 更接近于目标日期，则向前调整相应天数
                return target_date + timedelta(days=days_forward)
            else:
                # 如果上一个 self.weekday 更接近于目标日期，则向前调整相应天数
                return target_date + timedelta(days=days_forward - 7)
    # 解析后缀代码，根据不同的varion_code确定variation的值
    def _parse_suffix(cls, varion_code, startingMonth_code, weekday_code):
        if varion_code == "N":
            variation = "nearest"
        elif varion_code == "L":
            variation = "last"
        else:
            # 如果varion_code不是"N"或"L"，则引发数值错误异常
            raise ValueError(f"Unable to parse varion_code: {varion_code}")

        # 使用MONTH_TO_CAL_NUM字典将startingMonth_code转换为对应的月份数值
        startingMonth = MONTH_TO_CAL_NUM[startingMonth_code]
        # 使用weekday_to_int字典将weekday_code转换为对应的整数表示
        weekday = weekday_to_int[weekday_code]

        # 返回包含解析后信息的字典
        return {
            "weekday": weekday,
            "startingMonth": startingMonth,
            "variation": variation,
        }

    @classmethod
    # 从给定的参数调用_parse_suffix方法，返回相应的类对象
    def _from_name(cls, *args):
        return cls(**cls._parse_suffix(*args))
# 定义一个 Cython 类 FY5253Quarter，继承自 FY5253Mixin
cdef class FY5253Quarter(FY5253Mixin):
    """
    DateOffset increments between business quarter dates for 52-53 week fiscal year.

    Also known as a 4-4-5 calendar.

    It is used by companies that desire that their
    fiscal year always end on the same day of the week.

    It is a method of managing accounting periods.
    It is a common calendar structure for some industries,
    such as retail, manufacturing and parking industry.

    For more information see:
    https://en.wikipedia.org/wiki/4-4-5_calendar

    The year may either:

    - end on the last X day of the Y month.
    - end on the last X day closest to the last day of the Y month.

    X is a specific day of the week.
    Y is a certain month of the year

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/30/2007, 6/29/2007, ...

    Parameters
    ----------
    n : int
        The number of business quarters represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    startingMonth : int {1, 2, ..., 12}, default 1
        The month in which fiscal years end.

    qtr_with_extra_week : int {1, 2, 3, 4}, default 1
        The quarter number that has the leap or 14 week when needed.

    variation : str, default "nearest"
        Method of employing 4-4-5 calendar.

        There are two options:

        - "nearest" means year end is **weekday** closest to last day of month in year.
        - "last" means year end is final **weekday** of the final month in fiscal year.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below the default parameters give
    the next business quarter for 52-53 week fiscal year.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253Quarter()
    Timestamp('2022-01-31 00:00:00')

    By the parameter ``startingMonth`` we can specify
    the month in which fiscal years end.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253Quarter(startingMonth=3)
    Timestamp('2022-03-28 00:00:00')

    Business quarters for 52-53 week fiscal year can be specified by
    ``weekday`` and ``variation`` parameters.

    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.FY5253Quarter(weekday=5, startingMonth=12, variation="last")
    Timestamp('2022-04-02 00:00:00')
    """

    # 类变量，表示这个类所代表的对象前缀是 "REQ"
    _prefix = "REQ"
    # 定义一个元组，包含多个字符串属性，表示对象的固定属性列表
    _attributes = tuple(
        [
            "n",
            "normalize",
            "weekday",
            "startingMonth",
            "qtr_with_extra_week",
            "variation",
        ]
    )

    # 定义一个只读属性，表示对象的季度有额外周数
    cdef readonly:
        int qtr_with_extra_week

    # 初始化方法，设置对象的各种属性
    def __init__(
        self,
        n=1,
        normalize=False,
        weekday=0,
        startingMonth=1,
        qtr_with_extra_week=1,
        variation="nearest",
    ):
        # 调用父类的初始化方法，设置对象的基本属性
        FY5253Mixin.__init__(
            self, n, normalize, weekday, startingMonth, variation
        )
        # 设置对象特有的属性，表示季度有额外周数
        self.qtr_with_extra_week = qtr_with_extra_week

    # 定义对象状态恢复方法，从状态中恢复季度有额外周数的属性
    cpdef __setstate__(self, state):
        FY5253Mixin.__setstate__(self, state)
        self.qtr_with_extra_week = state.pop("qtr_with_extra_week")

    # 使用缓存装饰器定义一个只读属性，表示对象的偏移量
    @cache_readonly
    def _offset(self):
        return FY5253(
            startingMonth=self.startingMonth,
            weekday=self.weekday,
            variation=self.variation,
        )

    # 回滚方法，将日期回滚到最近的财政年度结束日期
    def _rollback_to_year(self, other: datetime):
        """
        Roll `other` back to the most recent date that was on a fiscal year
        end.

        Return the date of that year-end, the number of full quarters
        elapsed between that year-end and other, and the remaining Timedelta
        since the most recent quarter-end.

        Parameters
        ----------
        other : datetime or Timestamp

        Returns
        -------
        tuple of
        prev_year_end : Timestamp giving most recent fiscal year end
        num_qtrs : int
        tdelta : Timedelta
        """
        num_qtrs = 0

        # 将 `other` 转换为没有时区信息的时间戳
        norm = Timestamp(other).tz_localize(None)
        # 使用对象的偏移量方法获取最近的财政年度结束日期
        start = self._offset.rollback(norm)
        # 注意：start <= norm and self._offset.is_on_offset(start)

        if start < norm:
            # 如果开始日期早于 `norm`，进行调整
            # 获取每个季度长度的周数
            qtr_lens = self.get_weeks(norm)

            # 检查 `qtr_lens` 是否与 `self._offset` 的添加一致
            end = _shift_day(start, days=7 * sum(qtr_lens))
            # 断言结束日期处于偏移量上
            assert self._offset.is_on_offset(end), (start, end, qtr_lens)

            # 计算从开始日期到 `norm` 的时间差
            tdelta = norm - start
            for qlen in qtr_lens:
                if qlen * 7 <= tdelta.days:
                    # 如果季度长度乘以7小于等于时间差的天数，增加季度计数并调整时间差
                    num_qtrs += 1
                    tdelta -= (
                        <_Timedelta>Timedelta(days=qlen * 7)
                    )._as_creso(norm._creso)
                else:
                    break
        else:
            tdelta = Timedelta(0)

        # 注意：tdelta._value >= 0 总是成立
        return start, num_qtrs, tdelta

    # 应用装饰器，为方法或属性应用外部装饰器
    @apply_wraps
    # 应用偏移量规则到给定的日期时间对象上
    def _apply(self, other: datetime) -> datetime:
        # 注意：不允许 self.n == 0.

        n = self.n  # 将 self.n 的值赋给局部变量 n

        # 调用对象的 _rollback_to_year 方法，获取上一个年度结束日期、季度数和时间增量
        prev_year_end, num_qtrs, tdelta = self._rollback_to_year(other)
        res = prev_year_end  # 结果初始化为上一个年度结束日期
        n += num_qtrs  # 将季度数加到 n 上
        # 如果 self.n <= 0 并且 tdelta._value 大于 0，则 n 再加 1
        if self.n <= 0 and tdelta._value > 0:
            n += 1

        # 可能通过先处理年份来加快速度
        years = n // 4  # 计算 n 中包含的完整年数
        if years:
            res += self._offset * years  # 将年数乘以偏移量加到结果上
            n -= years * 4  # 更新 n，减去处理过的年数

        # 添加一天以确保获取到即将到来年度的季度长度，而不是前一年度的
        qtr_lens = self.get_weeks(res + Timedelta(days=1))

        # 注意：始终有 0 <= n < 4
        weeks = sum(qtr_lens[:n])  # 计算前 n 个季度的总周数
        if weeks:
            res = _shift_day(res, days=weeks * 7)  # 将结果向后偏移相应的周数天数

        return res  # 返回计算后的日期时间对象

    # 获取指定日期时间对象所在年度每个季度的周数列表
    def get_weeks(self, dt: datetime):
        ret = [13] * 4  # 初始化每个季度的周数为 13

        year_has_extra_week = self.year_has_extra_week(dt)

        if year_has_extra_week:
            ret[self.qtr_with_extra_week - 1] = 14  # 如果年度有额外的周，设置相应季度的周数为 14

        return ret  # 返回季度的周数列表

    # 判断指定日期时间对象所在年度是否有额外的周
    def year_has_extra_week(self, dt: datetime) -> bool:
        # 避免由于舍入误差而导致计算不准确，将日期时间对象规范化为 '370D' 格式
        norm = Timestamp(dt).normalize().tz_localize(None)

        next_year_end = self._offset.rollforward(norm)  # 获取下一个年度结束日期
        prev_year_end = norm - self._offset  # 获取前一个年度结束日期
        weeks_in_year = (next_year_end - prev_year_end).days / 7  # 计算年度总周数
        assert weeks_in_year in [52, 53], weeks_in_year  # 断言年度总周数应为 52 或 53
        return weeks_in_year == 53  # 返回是否有 53 周的布尔值

    # 判断指定日期时间对象是否符合偏移量规则
    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False  # 如果需要规范化并且日期时间对象未被规范化，则返回 False
        if self._offset.is_on_offset(dt):
            return True  # 如果日期时间对象符合偏移量规则，则返回 True

        next_year_end = dt - self._offset  # 获取偏移后的下一个年度结束日期

        qtr_lens = self.get_weeks(dt)  # 获取指定日期时间对象所在年度每个季度的周数列表

        current = next_year_end
        for qtr_len in qtr_lens:
            current = _shift_day(current, days=qtr_len * 7)  # 按每个季度的周数向后偏移日期时间对象
            if dt == current:
                return True  # 如果偏移后的日期时间对象等于原日期时间对象，则返回 True
        return False  # 如果未找到符合条件的季度偏移，返回 False

    # 返回规则代码的属性，格式为 'suffix-qtr'
    @property
    def rule_code(self) -> str:
        suffix = FY5253Mixin.rule_code.__get__(self)  # 获取基类的 rule_code 属性值
        qtr = self.qtr_with_extra_week  # 获取额外周所在的季度
        return f"{suffix}-{qtr}"  # 返回组合后的规则代码字符串

    # 从名称解析参数并创建对象的类方法
    @classmethod
    def _from_name(cls, *args):
        return cls(
            **dict(FY5253._parse_suffix(*args[:-1]), qtr_with_extra_week=int(args[-1]))
        )  # 使用解析后的参数创建并返回对象
cdef class Easter(SingleConstructorOffset):
    """
    DateOffset for the Easter holiday using logic defined in dateutil.

    Right now uses the revised method which is valid in years 1583-4099.

    Parameters
    ----------
    n : int, default 1
        The number of years represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 1, 1)
    >>> ts + pd.offsets.Easter()
    Timestamp('2022-04-17 00:00:00')
    """

    cpdef __setstate__(self, state):
        # 从状态中恢复 'n' 参数
        self.n = state.pop("n")
        # 从状态中恢复 'normalize' 参数
        self.normalize = state.pop("normalize")

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # 导入 dateutil 库中的 easter 函数
        from dateutil.easter import easter

        # 获取当前年份的复活节日期，并转换为 datetime 类型
        current_easter = easter(other.year)
        current_easter = datetime(
            current_easter.year, current_easter.month, current_easter.day
        )
        # 将当前复活节日期本地化，并与 other 的时区信息对齐
        current_easter = localize_pydatetime(current_easter, other.tzinfo)

        n = self.n
        # 根据 self.n 的值调整复活节年份
        if n >= 0 and other < current_easter:
            n -= 1
        elif n < 0 and other > current_easter:
            n += 1
        # TODO: Why does this handle the 0 case the opposite of others?

        # NOTE: easter 返回一个 datetime.date 对象，因此需要将其转换为与 other 相同类型的对象
        # 创建新的日期对象，表示调整后的复活节日期
        new = easter(other.year + n)
        new = datetime(
            new.year,
            new.month,
            new.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        return new

    def is_on_offset(self, dt: datetime) -> bool:
        # 如果启用了 normalize 选项，并且日期时间 dt 没有经过标准化，则返回 False
        if self.normalize and not _is_normalized(dt):
            return False

        # 导入 dateutil 库中的 easter 函数
        from dateutil.easter import easter

        # 检查给定日期 dt 是否为其年份的复活节日期
        return date(dt.year, dt.month, dt.day) == easter(dt.year)
    # 创建一个时间戳对象，表示2022年8月5日下午4点
    ts = pd.Timestamp(2022, 8, 5, 16)
    
    # 将自定义工作日偏移量应用于时间戳对象，返回新的时间戳对象
    ts + pd.offsets.CustomBusinessDay()
    Timestamp('2022-08-08 16:00:00')
    
    """
    Business days can be specified by ``weekmask`` parameter. To convert
    the returned datetime object to its string representation
    the function strftime() is used in the next example.
    """
    
    # 导入datetime模块并创建自定义工作日偏移量对象，指定工作日为周一、周三、周五
    freq = pd.offsets.CustomBusinessDay(weekmask="Mon Wed Fri")
    
    # 使用指定频率生成日期范围，返回日期时间索引，并格式化为指定的字符串表示形式
    pd.date_range(dt.datetime(2022, 12, 10), dt.datetime(2022, 12, 21),
                  freq=freq).strftime('%a %d %b %Y %H:%M')
    Index(['Mon 12 Dec 2022 00:00', 'Wed 14 Dec 2022 00:00',
           'Fri 16 Dec 2022 00:00', 'Mon 19 Dec 2022 00:00',
           'Wed 21 Dec 2022 00:00'],
           dtype='object')
    
    """
    Using NumPy business day calendar you can define custom holidays.
    """
    
    # 导入datetime模块并使用NumPy的工作日日历，定义自定义假期列表
    bdc = np.busdaycalendar(holidays=['2022-12-12', '2022-12-14'])
    
    # 使用自定义工作日日历生成日期范围，返回日期时间索引
    pd.date_range(dt.datetime(2022, 12, 10), dt.datetime(2022, 12, 25), freq=freq)
    DatetimeIndex(['2022-12-13', '2022-12-15', '2022-12-16', '2022-12-19',
                   '2022-12-20', '2022-12-21', '2022-12-22', '2022-12-23'],
                   dtype='datetime64[ns]', freq='C')
    
    """
    If you want to shift the result on n day you can use the parameter ``offset``.
    """
    
    # 将自定义工作日偏移量应用于时间戳对象，将结果向后移动1个工作日
    pd.Timestamp(2022, 8, 5, 16) + pd.offsets.CustomBusinessDay(1)
    Timestamp('2022-08-08 16:00:00')
    
    # 导入datetime模块并创建时间戳对象
    ts = pd.Timestamp(2022, 8, 5, 16)
    
    # 将自定义工作日偏移量应用于时间戳对象，将结果向后移动1个工作日，并使用timedelta指定额外的偏移量
    ts + pd.offsets.CustomBusinessDay(1, offset=dt.timedelta(days=1))
    Timestamp('2022-08-09 16:00:00')
    """
    
    _prefix = "C"
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )
    
    @property
    def _period_dtype_code(self):
        # GH#52534
        raise ValueError(f"{self.base} is not supported as period frequency")
    
    _apply_array = BaseOffset._apply_array
    
    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        # 调用父类BusinessDay的初始化方法，并初始化自定义工作日偏移量对象
        BusinessDay.__init__(self, n, normalize, offset)
        self._init_custom(weekmask, holidays, calendar)
    
    cpdef __setstate__(self, state):
        # 从状态中恢复假期和工作日掩码属性，调用父类BusinessDay的设置状态方法
        self.holidays = state.pop("holidays")
        self.weekmask = state.pop("weekmask")
        BusinessDay.__setstate__(self, state)
    
    @apply_wraps
    # 如果 self.n 小于等于 0，则设置滚动方向为前进
    if self.n <= 0:
        roll = "forward"
    else:
        # 否则设置滚动方向为后退
        roll = "backward"

    # 如果 other 是 Python 的 datetime 对象
    if PyDateTime_Check(other):
        # 将 other 转换为 numpy 的 datetime64 格式
        date_in = other
        np_dt = np.datetime64(date_in.date())

        # 使用 numpy 的 busday_offset 函数计算偏移后的日期
        np_incr_dt = np.busday_offset(
            np_dt, self.n, roll=roll, busdaycal=self.calendar
        )

        # 将 numpy 的 datetime64 转换为 Python 的 datetime 格式
        dt_date = np_incr_dt.astype(datetime)
        result = datetime.combine(dt_date, date_in.time())

        # 如果存在偏移量 self.offset，则将其加到结果中
        if self.offset:
            result = result + self.offset
        return result

    # 如果 other 是任何类型的 timedelta 标量
    elif is_any_td_scalar(other):
        # 将 self.offset 转换为 Timedelta 类型并加到 other 中
        td = Timedelta(self.offset) + other
        return BDay(self.n, offset=td.to_pytimedelta(), normalize=self.normalize)
    else:
        # 如果 other 类型不支持，抛出异常
        raise ApplyTypeError(
            "Only know how to combine trading day with "
            "datetime, datetime64 or timedelta."
        )

# 检查给定的日期 dt 是否是符合偏移条件的交易日
def is_on_offset(self, dt: datetime) -> bool:
    # 如果 self.normalize 为 True 并且 dt 不是规范化的日期时间，则返回 False
    if self.normalize and not _is_normalized(dt):
        return False
    # 将 dt 转换为 numpy 的 datetime64D 格式
    day64 = _to_dt64D(dt)
    # 使用 numpy 的 is_busday 函数判断是否是交易日
    return np.is_busday(day64, busdaycal=self.calendar)
# 定义一个自定义的业务小时类 CustomBusinessHour，它继承自 BusinessHour 类
cdef class CustomBusinessHour(BusinessHour):
    """
    DateOffset subclass representing possibly n custom business days.

    In CustomBusinessHour we can use custom weekmask, holidays, and calendar.

    Parameters
    ----------
    n : int, default 1
        The number of hours represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    start : str, time, or list of str/time, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, time, or list of str/time, default: "17:00"
        End time of your custom business hour in 24h format.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    Examples
    --------
    In the example below the default parameters give the next business hour.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour()
    Timestamp('2022-08-08 09:00:00')

    We can also change the start and the end of business hours.

    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour(start="11:00")
    Timestamp('2022-08-08 11:00:00')

    >>> from datetime import time as dt_time
    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour(end=dt_time(19, 0))
    Timestamp('2022-08-05 17:00:00')

    >>> ts = pd.Timestamp(2022, 8, 5, 22)
    >>> ts + pd.offsets.CustomBusinessHour(end=dt_time(19, 0))
    Timestamp('2022-08-08 10:00:00')

    You can divide your business day hours into several parts.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessHour(start=["06:00", "10:00", "15:00"],
    ...                                      end=["08:00", "12:00", "17:00"])
    >>> pd.date_range(dt.datetime(2022, 12, 9), dt.datetime(2022, 12, 13), freq=freq)
    DatetimeIndex(['2022-12-09 06:00:00', '2022-12-09 07:00:00',
                   '2022-12-09 10:00:00', '2022-12-09 11:00:00',
                   '2022-12-09 15:00:00', '2022-12-09 16:00:00',
                   '2022-12-12 06:00:00', '2022-12-12 07:00:00',
                   '2022-12-12 10:00:00', '2022-12-12 11:00:00',
                   '2022-12-12 15:00:00', '2022-12-12 16:00:00'],
                   dtype='datetime64[ns]', freq='cbh')

    Business days can be specified by ``weekmask`` parameter. To convert
    the returned datetime object to its string representation
    the function strftime() is used in the next example.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessHour(weekmask="Mon Wed Fri",
    ...                                      start="10:00", end="13:00")
    """
    # 定义一个类，继承自 BusinessHour，用于自定义工作小时，包括工作日设置、假期、日历等属性
    class CustomBusinessHour(BusinessHour):
    
        # 初始化方法，设置类的各种属性和参数
        def __init__(
            self,
            n=1,  # 工作小时的数量，默认为1
            normalize=False,  # 是否标准化时间，默认为 False
            weekmask="Mon Tue Wed Thu Fri",  # 定义工作日的掩码，默认为周一到周五
            holidays=None,  # 自定义假期列表，默认为空
            calendar=None,  # 自定义工作日历对象，默认为空
            start="09:00",  # 工作开始时间，默认为早上9点
            end="17:00",  # 工作结束时间，默认为下午5点
            offset=timedelta(0),  # 时间偏移量，默认为0
        ):
            # 调用父类 BusinessHour 的初始化方法，设置工作小时数、是否标准化、起始和结束时间、偏移量
            BusinessHour.__init__(self, n, normalize, start=start, end=end, offset=offset)
            # 调用类的自定义初始化方法，设置工作日掩码、假期和工作日历
            self._init_custom(weekmask, holidays, calendar)
    
        # 类的属性列表，包括标准属性和自定义属性
        _attributes = tuple(
            ["n", "normalize", "weekmask", "holidays", "calendar", "start", "end", "offset"]
        )
cdef class _CustomBusinessMonth(BusinessMixin):
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        # 调用父类的初始化方法，传递初始参数 n, normalize, offset
        BusinessMixin.__init__(self, n, normalize, offset)
        # 使用给定的参数初始化自定义月份对象
        self._init_custom(weekmask, holidays, calendar)

    @cache_readonly
    def cbday_roll(self):
        """
        Define default roll function to be called in apply method.
        """
        # 复制关键字参数
        cbday_kwds = self.kwds.copy()
        cbday_kwds["offset"] = timedelta(0)

        # 创建 CustomBusinessDay 对象
        cbday = CustomBusinessDay(n=1, normalize=False, **cbday_kwds)

        if self._prefix.endswith("S"):
            # 如果前缀以 "S" 结尾，则为月初
            roll_func = cbday.rollforward  # 设置为月初的推进函数
        else:
            # 否则为月末
            roll_func = cbday.rollback  # 设置为月末的回滚函数
        return roll_func

    @cache_readonly
    def m_offset(self):
        if self._prefix.endswith("S"):
            # 如果前缀以 "S" 结尾，则为月初
            moff = MonthBegin(n=1, normalize=False)  # 创建月初偏移对象
        else:
            # 否则为月末
            moff = MonthEnd(n=1, normalize=False)  # 创建月末偏移对象
        return moff

    @cache_readonly
    def month_roll(self):
        """
        Define default roll function to be called in apply method.
        """
        if self._prefix.endswith("S"):
            # 如果前缀以 "S" 结尾，则为月初
            roll_func = self.m_offset.rollback  # 设置为月初的回滚函数
        else:
            # 否则为月末
            roll_func = self.m_offset.rollforward  # 设置为月末的推进函数
        return roll_func

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # 首先移动到月份偏移
        cur_month_offset_date = self.month_roll(other)

        # 找到自定义月份偏移的比较日期
        compare_date = self.cbday_roll(cur_month_offset_date)
        # 根据自定义规则处理日期
        n = roll_convention(other.day, self.n, compare_date.day)

        # 计算新的日期
        new = cur_month_offset_date + n * self.m_offset
        # 应用推进/回滚函数
        result = self.cbday_roll(new)

        # 如果存在额外的偏移量，则应用
        if self.offset:
            result = result + self.offset
        return result


cdef class CustomBusinessMonthEnd(_CustomBusinessMonth):
    """
    DateOffset subclass representing custom business month(s).

    Increments between end of month dates.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.
    """
    # 定义一个类变量 _prefix，表示 CustomBusinessMonthEnd 的缩写
    _prefix = "CBME"
# 定义一个自定义的业务月份开始类，继承自 _CustomBusinessMonth 类
cdef class CustomBusinessMonthBegin(_CustomBusinessMonth):
    """
    DateOffset subclass representing custom business month(s).

    Increments between beginning of month dates.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : np.busdaycalendar
        Calendar to integrate.
    offset : timedelta, default timedelta(0)
        Time offset to apply.

    See Also
    --------
    :class:`~pandas.tseries.offsets.DateOffset` : Standard kind of date increment.

    Examples
    --------
    In the example below we use the default parameters.

    >>> ts = pd.Timestamp(2022, 8, 5)
    >>> ts + pd.offsets.CustomBusinessMonthBegin()
    Timestamp('2022-09-01 00:00:00')

    Custom business month start can be specified by ``weekmask`` parameter.
    To convert the returned datetime object to its string representation
    the function strftime() is used in the next example.

    >>> import datetime as dt
    >>> freq = pd.offsets.CustomBusinessMonthBegin(weekmask="Wed Thu")
    >>> pd.date_range(dt.datetime(2022, 7, 10), dt.datetime(2022, 12, 18),
    ...               freq=freq).strftime('%a %d %b %Y %H:%M')
    Index(['Wed 03 Aug 2022 00:00', 'Thu 01 Sep 2022 00:00',
           'Wed 05 Oct 2022 00:00', 'Wed 02 Nov 2022 00:00',
           'Thu 01 Dec 2022 00:00'],
           dtype='object')

    Using NumPy business day calendar you can define custom holidays.

    >>> import datetime as dt
    >>> bdc = np.busdaycalendar(holidays=['2022-08-01', '2022-09-30',
    ...                                   '2022-10-31', '2022-11-01'])
    >>> freq = pd.offsets.CustomBusinessMonthBegin(calendar=bdc)
    >>> pd.date_range(dt.datetime(2022, 7, 10), dt.datetime(2022, 11, 10), freq=freq)
    DatetimeIndex(['2022-08-02', '2022-09-01', '2022-10-03', '2022-11-02'],
                   dtype='datetime64[ns]', freq='CBMS')
    """

    # 类属性，用于标识该类的前缀
    _prefix = "CBMS"


# 别名定义
BDay = BusinessDay
BMonthEnd = BusinessMonthEnd
BMonthBegin = BusinessMonthBegin
CBMonthEnd = CustomBusinessMonthEnd
CBMonthBegin = CustomBusinessMonthBegin
CDay = CustomBusinessDay

# ----------------------------------------------------------------------
# to_offset helpers

# 前缀映射字典，将每个偏移量类的前缀映射到相应的偏移量类对象
prefix_mapping = {
    offset._prefix: offset
    for offset in [
        YearBegin,  # 年度开始时间偏移量，代号为 'YS'
        YearEnd,  # 年度结束时间偏移量，代号为 'YE'
        BYearBegin,  # 业务年度开始时间偏移量，代号为 'BYS'
        BYearEnd,  # 业务年度结束时间偏移量，代号为 'BYE'
        BusinessDay,  # 工作日时间偏移量，代号为 'B'
        BusinessMonthBegin,  # 工作月开始时间偏移量，代号为 'BMS'
        BusinessMonthEnd,  # 工作月结束时间偏移量，代号为 'BME'
        BQuarterEnd,  # 工作季度结束时间偏移量，代号为 'BQE'
        BQuarterBegin,  # 工作季度开始时间偏移量，代号为 'BQS'
        BusinessHour,  # 工作小时时间偏移量，代号为 'bh'
        CustomBusinessDay,  # 自定义工作日时间偏移量，代号为 'C'
        CustomBusinessMonthEnd,  # 自定义工作月结束时间偏移量，代号为 'CBME'
        CustomBusinessMonthBegin,  # 自定义工作月开始时间偏移量，代号为 'CBMS'
        CustomBusinessHour,  # 自定义工作小时时间偏移量，代号为 'cbh'
        MonthEnd,  # 月末时间偏移量，代号为 'ME'
        MonthBegin,  # 月初时间偏移量，代号为 'MS'
        Nano,  # 纳秒时间偏移量，代号为 'ns'
        SemiMonthEnd,  # 半月结束时间偏移量，代号为 'SME'
        SemiMonthBegin,  # 半月开始时间偏移量，代号为 'SMS'
        Week,  # 周时间偏移量，代号为 'W'
        Second,  # 秒时间偏移量，代号为 's'
        Minute,  # 分钟时间偏移量，代号为 'min'
        Micro,  # 微秒时间偏移量，代号为 'us'
        QuarterEnd,  # 季度结束时间偏移量，代号为 'QE'
        QuarterBegin,  # 季度开始时间偏移量，代号为 'QS'
        Milli,  # 毫秒时间偏移量，代号为 'ms'
        Hour,  # 小时时间偏移量，代号为 'h'
        Day,  # 天时间偏移量，代号为 'D'
        WeekOfMonth,  # 月中第几周时间偏移量，代号为 'WOM'
        FY5253,  # 财年52/53周时间偏移量
        FY5253Quarter,  # 财年52/53周季度时间偏移量
    ]
}

# 处理特定的 WOM-1MON 的hack
opattern = re.compile(
    r"([+\-]?\d*|[+\-]?\d*\.\d*)\s*([A-Za-z]+([\-][\dA-Za-z\-]+)?)"
)

# _lite_rule_alias 字典，用于简化频率别名
_lite_rule_alias = {
    "W": "W-SUN",       # 周频率，以星期日结束
    "QE": "QE-DEC",     # 季度结束，以12月结束

    "YE": "YE-DEC",     # 年结束，以12月结束
    "YS": "YS-JAN",     # 年开始，以1月开始
    "BYE": "BYE-DEC",   # 业务年度结束，以12月结束
    "BYS": "BYS-JAN",   # 业务年度开始，以1月开始

    "Min": "min",       # 分钟
    "min": "min",       # 分钟
    "ms": "ms",         # 毫秒
    "us": "us",         # 微秒
    "ns": "ns",         # 纳秒
}

# 不应大写的别名集合
_dont_uppercase = {"h", "bh", "cbh", "MS", "ms", "s"}

# 错误消息，用于无效频率
INVALID_FREQ_ERR_MSG = "Invalid frequency: {0}"

# TODO: 是否仍然需要？
# 缓存之前见过的偏移量映射
_offset_map = {}


def _validate_to_offset_alias(alias: str, is_period: bool) -> None:
    """
    根据别名验证偏移量是否有效，如果无效则抛出 ValueError 异常。

    Parameters
    ----------
    alias : str
        要验证的偏移量别名
    is_period : bool
        是否为周期

    Raises
    ------
    ValueError
        当别名不再支持或者不符合规范时

    """
    if not is_period:
        # 非周期性别名验证
        if alias.upper() in c_OFFSET_RENAMED_FREQSTR:
            raise ValueError(
                f"\'{alias}\' is no longer supported for offsets. Please "
                f"use \'{c_OFFSET_RENAMED_FREQSTR.get(alias.upper())}\' "
                f"instead."
            )
        if (alias.upper() != alias and
                alias.lower() not in {"s", "ms", "us", "ns"} and
                alias.upper().split("-")[0].endswith(("S", "E"))):
            raise ValueError(INVALID_FREQ_ERR_MSG.format(alias))
    if (is_period and
            alias.upper() in c_OFFSET_TO_PERIOD_FREQSTR and
            alias != "ms" and
            alias.upper().split("-")[0].endswith(("S", "E"))):
        if (alias.upper().startswith("B") or
                alias.upper().startswith("S") or
                alias.upper().startswith("C")):
            raise ValueError(INVALID_FREQ_ERR_MSG.format(alias))
        else:
            alias_msg = "".join(alias.upper().split("E", 1))
            raise ValueError(
                f"for Period, please use \'{alias_msg}\' "
                f"instead of \'{alias}\'"
            )


# TODO: 更好的名称？
def _get_offset(name: str) -> BaseOffset:
    """
    根据规则名称返回与之关联的 DateOffset 对象。

    Examples
    --------
    _get_offset('EOM') --> BMonthEnd(1)

    Parameters
    ----------
    name : str
        规则名称

    Returns
    -------
    BaseOffset
        与规则名称关联的偏移量对象

    """
    if (
        name not in _lite_rule_alias
        and (name.upper() in _lite_rule_alias)
        and name != "ms"
    ):
        warnings.warn(
            f"\'{name}\' is deprecated and will be removed "
            f"in a future version, please use \'{name.upper()}\' instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    elif (
        name not in _lite_rule_alias
        and (name.lower() in _lite_rule_alias)
        and name != "MS"
    ):
        warnings.warn(
            f"\'{name}\' is deprecated and will be removed "
            f"in a future version, please use \'{name.lower()}\' instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    if name not in _dont_uppercase:
        name = name.upper()
        name = _lite_rule_alias.get(name, name)
        name = _lite_rule_alias.get(name.lower(), name)
    else:
        # 如果名字不在别名映射表中，则直接使用原始名字
        name = _lite_rule_alias.get(name, name)

    # 如果名字不在偏移量映射表中
    if name not in _offset_map:
        try:
            # 尝试按照"-"分割名字
            split = name.split("-")
            # 根据分割后的第一个部分查找对应的类别
            klass = prefix_mapping[split[0]]
            # 处理没有后缀的情况（如果有太多的'-'将导致 TypeError）
            offset = klass._from_name(*split[1:])
        except (ValueError, TypeError, KeyError) as err:
            # 捕获可能的错误，如无效的前缀或后缀
            raise ValueError(INVALID_FREQ_ERR_MSG.format(
                f"{name}, failed to parse with error message: {repr(err)}")
            )
        # 将计算出的偏移量缓存起来
        _offset_map[name] = offset

    # 返回名字对应的偏移量
    return _offset_map[name]
# 根据输入的频率字符串或时间增量对象返回对应的 DateOffset 对象
cpdef to_offset(freq, bint is_period=False):
    """
    Return DateOffset object from string or datetime.timedelta object.

    Parameters
    ----------
    freq : str, datetime.timedelta, BaseOffset or None
        The frequency represented.
    is_period : bool, default False
        Convert string denoting period frequency to corresponding offsets
        frequency if is_period=True.

    Returns
    -------
    BaseOffset subclass or None

    Raises
    ------
    ValueError
        If freq is an invalid frequency

    See Also
    --------
    BaseOffset : Standard kind of date increment used for a date range.

    Examples
    --------
    >>> from pandas.tseries.frequencies import to_offset
    >>> to_offset("5min")
    <5 * Minutes>

    >>> to_offset("1D1h")
    <25 * Hours>

    >>> to_offset("2W")
    <2 * Weeks: weekday=6>

    >>> to_offset("2B")
    <2 * BusinessDays>

    >>> to_offset(pd.Timedelta(days=1))
    <Day>

    >>> to_offset(pd.offsets.Hour())
    <Hour>

    Passing the parameter ``is_period`` equal to True, you can use a string
    denoting period frequency:

    >>> freq = to_offset(freq="ME", is_period=False)
    >>> freq.rule_code
    'ME'

    >>> freq = to_offset(freq="M", is_period=True)
    >>> freq.rule_code
    'ME'
    """
    # 如果 freq 参数为 None，则直接返回 None
    if freq is None:
        return None

    # 如果 freq 是一个元组，则抛出类型错误
    if isinstance(freq, tuple):
        raise TypeError(
            f"to_offset does not support tuples {freq}, pass as a string instead"
        )

    # 如果 freq 是 BaseOffset 的实例，则直接赋值给 result
    if isinstance(freq, BaseOffset):
        result = freq

    # 如果 freq 是 PyDelta_Check 的实例，则调用 delta_to_tick 函数处理结果赋值给 result
    elif PyDelta_Check(freq):
        result = delta_to_tick(freq)

    else:
        result = None

    # 如果 result 为 None，则抛出值错误异常，指示 freq 是无效的频率
    if result is None:
        raise ValueError(INVALID_FREQ_ERR_MSG.format(freq))

    try:
        # 尝试获取 result 的 _period_dtype_code 属性，判断是否支持周期频率
        has_period_dtype_code = hasattr(result, "_period_dtype_code")
    except ValueError:
        has_period_dtype_code = False

    # 如果 is_period 为 True 且 result 没有 _period_dtype_code 属性，则根据 freq 类型抛出相应的值错误异常
    if is_period and not has_period_dtype_code:
        if isinstance(freq, str):
            raise ValueError(f"{result.name} is not supported as period frequency")
        else:
            raise ValueError(f"{freq} is not supported as period frequency")

    # 返回处理后的 result 结果
    return result


# ----------------------------------------------------------------------
# RelativeDelta Arithmetic

# 定义一个 C 语言级函数 _shift_day，用于处理日期偏移操作
cdef datetime _shift_day(datetime other, int days):
    """
    Increment the datetime `other` by the given number of days, retaining
    the time-portion of the datetime.  For tz-naive datetimes this is
    equivalent to adding a timedelta.  For tz-aware datetimes it is similar to
    dateutil's relativedelta.__add__, but handles pytz tzinfo objects.

    Parameters
    ----------
    other : datetime or Timestamp
        输入的日期时间对象，可以是带有时区信息的或者没有时区信息的。
    days : int
        要增加的天数。

    Returns
    -------
    shifted: datetime or Timestamp
        返回增加指定天数后的日期时间对象，保留原有的时分秒。

    """
    # 如果输入的日期时间对象没有时区信息，则直接加上 timedelta 并返回结果
    if other.tzinfo is None:
        return other + timedelta(days=days)

    # 如果有时区信息，则先提取时区信息，将日期时间对象转换为无时区信息的 naive 版本，
    # 然后加上 timedelta，并通过 localize_pydatetime 函数将结果重新本地化为指定时区的日期时间对象。
    tz = other.tzinfo
    naive = other.replace(tzinfo=None)
    shifted = naive + timedelta(days=days)
    return localize_pydatetime(shifted, tz)
cdef int year_add_months(npy_datetimestruct dts, int months) noexcept nogil:
    """
    Calculate the new year number after shifting the npy_datetimestruct by the specified number of months.

    Parameters
    ----------
    dts : npy_datetimestruct
        Input datetime structure.
    months : int
        Number of months to shift.

    Returns
    -------
    int
        New year number.
    """
    return dts.year + (dts.month + months - 1) // 12


cdef int month_add_months(npy_datetimestruct dts, int months) noexcept nogil:
    """
    Calculate the new month number after shifting the npy_datetimestruct by the specified number of months.

    Parameters
    ----------
    dts : npy_datetimestruct
        Input datetime structure.
    months : int
        Number of months to shift.

    Returns
    -------
    int
        New month number (1-12).
    """
    cdef:
        int new_month = (dts.month + months) % 12
    return 12 if new_month == 0 else new_month


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ndarray shift_quarters(
    ndarray dtindex,
    int quarters,
    int q1start_month,
    str day_opt,
    int modby=3,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    Shift all elements in the int64 array `dtindex` representing nanosecond timestamps by the specified
    number of quarters using DateOffset semantics.

    Parameters
    ----------
    dtindex : ndarray[int64_t]
        Timestamps for input dates.
    quarters : int
        Number of quarters to shift.
    q1start_month : int
        Month in which Q1 begins by convention (1-12).
    day_opt : str
        Option specifying the day manipulation ('start', 'end', 'business_start', 'business_end').
    modby : int, optional
        Modifier (3 for quarters, 12 for years), default is 3.
    reso : NPY_DATETIMEUNIT, optional
        Resolution of the output array, default is NPY_FR_ns.

    Returns
    -------
    ndarray[int64_t]
        Shifted timestamps.

    Raises
    ------
    ValueError
        If `day_opt` is not one of ['start', 'end', 'business_start', 'business_end'].
    """
    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    cdef:
        Py_ssize_t count = dtindex.size
        ndarray out = cnp.PyArray_EMPTY(dtindex.ndim, dtindex.shape, cnp.NPY_INT64, 0)
        Py_ssize_t i
        int64_t val, res_val
        int months_since, n
        npy_datetimestruct dts
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, dtindex)

    with nogil:
        for i in range(count):
            # Analogous to: val = dtindex[i]
            val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            if val == NPY_NAT:
                res_val = NPY_NAT
            else:
                pandas_datetime_to_datetimestruct(val, reso, &dts)
                n = quarters

                months_since = (dts.month - q1start_month) % modby
                n = _roll_qtrday(&dts, n, months_since, day_opt)

                dts.year = year_add_months(dts, modby * n - months_since)
                dts.month = month_add_months(dts, modby * n - months_since)
                dts.day = get_day_of_month(&dts, day_opt)

                res_val = npy_datetimestruct_to_datetime(reso, &dts)

            # Analogous to: out[i] = res_val
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

            cnp.PyArray_MultiIter_NEXT(mi)

    return out


@cython.wraparound(False)
@cython.boundscheck(False)
def shift_months(
    ndarray dtindex,  # int64_t, arbitrary ndim
    int months,
    str day_opt=None,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    Shift all elements in the int64 array `dtindex` representing nanosecond timestamps by the specified
    number of months.

    Parameters
    ----------
    dtindex : ndarray[int64_t]
        Timestamps for input dates.
    months : int
        Number of months to shift.
    day_opt : str, optional
        Option specifying the day manipulation ('start', 'end', 'business_start', 'business_end').
    reso : NPY_DATETIMEUNIT, optional
        Resolution of the output array, default is NPY_FR_ns.

    Returns
    -------
    ndarray[int64_t]
        Shifted timestamps.
    """
    Given an int64-based datetime index, shift all elements
    specified number of months using DateOffset semantics

    day_opt: {None, 'start', 'end', 'business_start', 'business_end'}
       * None: day of month
       * 'start' 1st day of month
       * 'end' last day of month
    """
    cdef:
        Py_ssize_t i                               # 定义循环计数器 i
        npy_datetimestruct dts                     # 定义 numpy 日期时间结构体变量 dts
        int count = dtindex.size                   # 获取 dtindex 的大小，并赋给 count
        ndarray out = cnp.PyArray_EMPTY(dtindex.ndim, dtindex.shape, cnp.NPY_INT64, 0)  # 创建一个与 dtindex 维度和形状相同的空的 int64 数组 out
        int months_to_roll                         # 定义用于记录要移动的月份数的变量 months_to_roll
        int64_t val, res_val                       # 定义 int64_t 类型的变量 val 和 res_val

        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, dtindex)  # 使用 out 和 dtindex 创建一个广播迭代器 mi

    if day_opt is not None and day_opt not in {
            "start", "end", "business_start", "business_end"
    }:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")  # 如果 day_opt 不是 None 并且不在指定的选项中，则抛出 ValueError 异常

    if day_opt is None:
        # TODO: can we combine this with the non-None case?
        with nogil:
            for i in range(count):
                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # 从 mi 中获取第二个数组（dtindex）中当前位置的值，赋给 val

                if val == NPY_NAT:  # 如果 val 是 NPY_NAT（numpy 中的缺失值）
                    res_val = NPY_NAT  # 则 res_val 也设置为 NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)  # 将 val 转换为 datetimestruct 结构体 dts
                    dts.year = year_add_months(dts, months)  # 根据给定的月份数移动 dts 的年份
                    dts.month = month_add_months(dts, months)  # 根据给定的月份数移动 dts 的月份

                    dts.day = min(dts.day, get_days_in_month(dts.year, dts.month))  # 将 dts 的日设置为当前月份的天数或该月最大天数
                    res_val = npy_datetimestruct_to_datetime(reso, &dts)  # 将 dts 转换回 datetime，赋给 res_val

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val  # 将 res_val 的值写入 out 数组的当前位置

                cnp.PyArray_MultiIter_NEXT(mi)  # 将 mi 迭代器移动到下一个位置

    else:
        with nogil:
            for i in range(count):

                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # 从 mi 中获取第二个数组（dtindex）中当前位置的值，赋给 val

                if val == NPY_NAT:  # 如果 val 是 NPY_NAT（numpy 中的缺失值）
                    res_val = NPY_NAT  # 则 res_val 也设置为 NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)  # 将 val 转换为 datetimestruct 结构体 dts
                    months_to_roll = months  # 将 months 赋给 months_to_roll

                    months_to_roll = _roll_qtrday(&dts, months_to_roll, 0, day_opt)  # 调用函数 _roll_qtrday 处理 dts 和 months_to_roll，并返回更新后的 months_to_roll

                    dts.year = year_add_months(dts, months_to_roll)  # 根据更新后的月份数移动 dts 的年份
                    dts.month = month_add_months(dts, months_to_roll)  # 根据更新后的月份数移动 dts 的月份
                    dts.day = get_day_of_month(&dts, day_opt)  # 获取 dts 的特定日（如月初、月末等）

                    res_val = npy_datetimestruct_to_datetime(reso, &dts)  # 将 dts 转换回 datetime，赋给 res_val

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val  # 将 res_val 的值写入 out 数组的当前位置

                cnp.PyArray_MultiIter_NEXT(mi)  # 将 mi 迭代器移动到下一个位置

    return out  # 返回处理后的 int64 数组 out
def shift_month(stamp: datetime, months: int, day_opt: object = None) -> datetime:
    """
    Given a datetime (or Timestamp) `stamp`, an integer `months` and an
    option `day_opt`, return a new datetimelike that many months later,
    with day determined by `day_opt` using relativedelta semantics.

    Scalar analogue of shift_months.

    Parameters
    ----------
    stamp : datetime or Timestamp
        The starting datetime object.
    months : int
        Number of months to shift `stamp` by.
    day_opt : None, 'start', 'end', 'business_start', 'business_end', or int
        None: returned datetimelike has the same day as the input, or the
              last day of the month if the new month is too short
        'start': returned datetimelike has day=1
        'end': returned datetimelike has day on the last day of the month
        'business_start': returned datetimelike has day on the first
            business day of the month
        'business_end': returned datetimelike has day on the last
            business day of the month
        int: returned datetimelike has day equal to day_opt
            (if within valid range)

    Returns
    -------
    shifted : datetime or Timestamp (same as input `stamp`)
        The datetime object shifted according to the specified parameters.
    """
    cdef:
        int year, month, day
        int days_in_month, dy

    dy = (stamp.month + months) // 12  # Calculate number of years to add
    month = (stamp.month + months) % 12  # Calculate resulting month index

    if month == 0:
        month = 12
        dy -= 1
    year = stamp.year + dy  # Calculate resulting year

    if day_opt is None:
        days_in_month = get_days_in_month(year, month)  # Get total days in month
        day = min(stamp.day, days_in_month)  # Choose minimum of current day or total days in month
    elif day_opt == "start":
        day = 1  # Set day to 1st day of the month
    elif day_opt == "end":
        day = get_days_in_month(year, month)  # Set day to last day of the month
    elif day_opt == "business_start":
        # Set day to the first business day of the month
        day = get_firstbday(year, month)
    elif day_opt == "business_end":
        # Set day to the last business day of the month
        day = get_lastbday(year, month)
    elif is_integer_object(day_opt):
        days_in_month = get_days_in_month(year, month)  # Get total days in month
        day = min(day_opt, days_in_month)  # Choose minimum of specified day_opt or total days in month
    else:
        raise ValueError(day_opt)  # Raise error if day_opt is invalid

    return stamp.replace(year=year, month=month, day=day)  # Return datetime with specified year, month, and day


cdef int get_day_of_month(npy_datetimestruct* dts, str day_opt) noexcept nogil:
    """
    Find the day in `other`'s month that satisfies a DateOffset's is_on_offset
    policy, as described by the `day_opt` argument.

    Parameters
    ----------
    dts : npy_datetimestruct*
        A numpy datetime structure.
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        'start': returns 1
        'end': returns last day of the month
        'business_start': returns the first business day of the month
        'business_end': returns the last business day of the month

    Returns
    -------
    day_of_month : int
        The day of the month according to the `day_opt` specified.

    Examples
    -------
    >>> other = datetime(2017, 11, 14)
    >>> get_day_of_month(other, 'start')
    1
    >>> get_day_of_month(other, 'end')
    30

    Notes
    -----
    Caller is responsible for ensuring one of the four accepted day_opt values
    is passed.
    """
    
    if day_opt == "start":
        return 1  # Return 1 for the start of the month
    elif day_opt == "end":
        # 如果 day_opt 等于 "end"，返回当前月份的总天数
        return get_days_in_month(dts.year, dts.month)
    elif day_opt == "business_start":
        # 如果 day_opt 等于 "business_start"，返回当前月份的第一个工作日
        return get_firstbday(dts.year, dts.month)
    else:
        # 否则，即 day_opt 等于 "business_end":
        # 返回当前月份的最后一个工作日
        return get_lastbday(dts.year, dts.month)
# 根据特定的约定可能增加或减少需要移动的周期数，基于前滚或后滚的约定。
cpdef int roll_convention(int other, int n, int compare) noexcept nogil:
    """
    可能根据前滚/后滚约定调整周期数的增减。

    Parameters
    ----------
    other : int，通常是日期时间的日组件
    n : 需要增加的周期数，在调整滚动之前
    compare : int，通常是与 `other` 相同月份的日期时间的日组件

    Returns
    -------
    n : int 需要增加的周期数
    """
    if n > 0 and other < compare:
        # 如果当前周期大于0且`other`小于`compare`，则减少周期数
        n -= 1
    elif n <= 0 and other > compare:
        # 如果当前周期小于等于0且`other`大于`compare`，则增加周期数，仿佛已经前滚过了
        n += 1
    return n


def roll_qtrday(other: datetime, n: int, month: int,
                day_opt: str, modby: int) -> int:
    """
    可能根据前滚/后滚约定调整周期数的增减。

    Parameters
    ----------
    other : datetime 或 Timestamp
    n : 需要增加的周期数，在调整滚动之前
    month : int 参考月份，给出年份的第一个月
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        用于查找给定月份中的日期的约定，用于前滚/后滚决策
    modby : int 3 表示季度，12 表示年

    Returns
    -------
    n : int 需要增加的周期数

    See Also
    --------
    get_day_of_month : 给定偏移量，找到月份中的日期。
    """
    cdef:
        int months_since
        npy_datetimestruct dts

    if day_opt not in ["start", "end", "business_start", "business_end"]:
        # 如果 `day_opt` 不在有效的选项中，则引发 ValueError 异常
        raise ValueError(day_opt)

    pydate_to_dtstruct(other, &dts)

    if modby == 12:
        # 我们关心年份中的月份，而不是季度中的月份，因此跳过模数运算
        months_since = other.month - month
    else:
        months_since = other.month % modby - month % modby

    return _roll_qtrday(&dts, n, months_since, day_opt)


cdef int _roll_qtrday(npy_datetimestruct* dts,
                      int n,
                      int months_since,
                      str day_opt) except? -1 nogil:
    """
    参见 roll_qtrday.__doc__
    """

    if n > 0:
        if months_since < 0 or (months_since == 0 and
                                dts.day < get_day_of_month(dts, day_opt)):
            # 如果周期数大于0且月份差小于0或者月份相同但日期小于对比日期，则假装后滚
            n -= 1
    else:
        if months_since > 0 or (months_since == 0 and
                                dts.day > get_day_of_month(dts, day_opt)):
            # 如果周期数小于等于0且月份差大于0或者月份相同但日期大于对比日期，则确保前滚，因此增加周期数
            n += 1
    return n
```