# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timestamps.pyx`

```
"""
_Timestamp is a c-defined subclass of datetime.datetime

_Timestamp is PITA. Because we inherit from datetime, which has very specific
construction requirements, we need to do object instantiation in python
(see Timestamp class below). This will serve as a C extension type that
shadows the python class, where we do any heavy lifting.
"""

# 引入警告模块，用于可能的警告信息
import warnings

# 引入Cython模块
cimport cython

# 引入NumPy库，并使用Cython来导入相应的C级别接口
import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

# 导入NumPy数组相关功能
cnp.import_array()

# 从Cython模块中导入datetime模块的部分内容
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    PyTZInfo_Check,
    datetime,
    import_datetime,
    time as dt_time,
    tzinfo as tzinfo_type,
)

# 从Cython模块中导入object模块的部分内容
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompare,
    PyObject_RichCompareBool,
)

# 调用import_datetime函数，导入datetime模块
import_datetime()

# 导入Python标准库中的datetime模块，并简写为dt
import datetime as dt

# 从pandas._libs.tslibs中导入ccalendar模块
from pandas._libs.tslibs cimport ccalendar

# 从pandas._libs.tslibs.base中导入ABCTimestamp抽象基类
from pandas._libs.tslibs.base cimport ABCTimestamp

# 从pandas.util._exceptions中导入find_stack_level函数
from pandas.util._exceptions import find_stack_level

# 从pandas._libs.tslibs.conversion中导入_TSObject和相关转换函数
from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    convert_datetime_to_tsobject,
    convert_to_tsobject,
    maybe_localize_tso,
)

# 从pandas._libs.tslibs.dtypes中导入各种数据类型相关的常量和函数
from pandas._libs.tslibs.dtypes cimport (
    npy_unit_to_abbrev,
    npy_unit_to_attrname,
    periods_per_day,
    periods_per_second,
)

# 从pandas._libs.tslibs.util中导入一些工具函数
from pandas._libs.tslibs.util cimport (
    is_array,
    is_integer_object,
)

# 从pandas._libs.tslibs.fields中导入各种字段相关的功能
from pandas._libs.tslibs.fields import (
    RoundTo,
    get_date_name_field,
    get_start_end_field,
    round_nsint64,
)

# 从pandas._libs.tslibs.nattype中导入自然时间相关的常量和类型
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)

# 从pandas._libs.tslibs.np_datetime中导入NumPy datetime相关的函数和常量
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    cmp_dtstructs,
    cmp_scalar,
    convert_reso,
    dts_to_iso_string,
    get_datetime64_unit,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dtstruct,
)

# 调用import_pandas_datetime函数，导入pandas中的datetime模块
import_pandas_datetime()

# 从pandas._libs.tslibs.np_datetime模块中导入OutOfBoundsDatetime和OutOfBoundsTimedelta异常
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

# 从pandas._libs.tslibs.offsets中导入to_offset函数
from pandas._libs.tslibs.offsets cimport to_offset

# 从pandas._libs.tslibs.timedeltas中导入_Timedelta和相关函数
from pandas._libs.tslibs.timedeltas cimport (
    _Timedelta,
    get_unit_for_round,
    is_any_td_scalar,
)

# 从pandas._libs.tslibs.timedeltas中导入Timedelta类
from pandas._libs.tslibs.timedeltas import Timedelta

# 从pandas._libs.tslibs.timezones中导入时区相关的函数和常量
from pandas._libs.tslibs.timezones cimport (
    get_timezone,
    is_utc,
    maybe_get_tz,
    treat_tz_as_pytz,
    utc_stdlib as UTC,
)

# 从pandas._libs.tslibs.tzconversion中导入时区转换相关的函数
from pandas._libs.tslibs.tzconversion cimport (
    tz_convert_from_utc_single,
    tz_localize_to_utc_single,
)

# ----------------------------------------------------------------------
# Constants
# 定义常量_zero_time，表示时间的0点0分
_zero_time = dt_time(0, 0)
# 定义常量_no_input，表示没有输入
_no_input = object()

# ----------------------------------------------------------------------


# 定义C级函数_create_timestamp_from_ts，用于创建_Timestamp对象
cdef _Timestamp create_timestamp_from_ts(
    int64_t value,  # 64位整数值
    npy_datetimestruct dts,  # NumPy日期时间结构体
    tzinfo tz,  # 时区信息对象
    bint fold,  # 折叠标志位，用于处理重复的时间
    NPY_DATETIMEUNIT reso=NPY_FR_ns,  # 时间单位，默认为纳秒
):
    """ convenience routine to construct a Timestamp from its parts """
    # 定义一个Cython对象_ts_base，用于存储时间戳的基本信息
    cdef:
        _Timestamp ts_base  # 声明一个_Timestamp类型的变量ts_base
        int64_t pass_year = dts.year  # 声明一个int64_t类型的变量pass_year，并初始化为dts.year的值

    # We pass year=1970/1972 here and set year below because with non-nanosecond
    #  resolution we may have datetimes outside of the stdlib pydatetime
    #  implementation bounds, which would raise.
    # NB: this means the C-API macro PyDateTime_GET_YEAR is unreliable.
    # 如果传入的年份在1到9999之间，则年份在标准库pydatetime的范围内，可以直接使用
    if 1 <= pass_year <= 9999:
        # we are in-bounds for pydatetime
        pass  # 如果年份在合法范围内，继续执行
    elif ccalendar.is_leapyear(dts.year):
        pass_year = 1972  # 如果是闰年，则将pass_year设为1972
    else:
        pass_year = 1970  # 否则将pass_year设为1970

    # 使用传入的各个时间部分和时区信息构造一个_Timestamp对象ts_base
    ts_base = _Timestamp.__new__(Timestamp, pass_year, dts.month,
                                 dts.day, dts.hour, dts.min,
                                 dts.sec, dts.us, tz, fold=fold)

    ts_base._value = value  # 设置ts_base的_value属性为value
    ts_base.year = dts.year  # 设置ts_base的year属性为dts.year
    ts_base.nanosecond = dts.ps // 1000  # 将dts的picosecond转换为nanosecond，并赋值给ts_base的nanosecond属性
    ts_base._creso = reso  # 设置ts_base的_creso属性为reso

    return ts_base  # 返回构造好的ts_base对象作为时间戳的表示
# 定义一个函数 `_unpickle_timestamp`，用于反序列化时间戳对象
def _unpickle_timestamp(value, freq, tz, reso=NPY_FR_ns):
    # 使用 Timestamp 类的类方法 `_from_value_and_reso` 创建一个时间戳对象 `ts`
    ts = Timestamp._from_value_and_reso(value, reso, tz)
    # 返回时间戳对象 `ts`
    return ts


# ----------------------------------------------------------------------

# 定义一个函数 `integer_op_not_supported`，用于处理不支持的整数操作
def integer_op_not_supported(obj):
    # 获取对象 `obj` 的类名
    cls = type(obj).__name__

    # 构建整数加减操作不支持的错误消息
    int_addsub_msg = (
        f"Addition/subtraction of integers and integer-arrays with {cls} is "
        "no longer supported.  Instead of adding/subtracting `n`, "
        "use `n * obj.freq`"
    )
    # 返回类型错误，包含错误消息 `int_addsub_msg`
    return TypeError(int_addsub_msg)


# 定义一个类 `MinMaxReso`，用于定义时间戳最小值、最大值和分辨率
class MinMaxReso:
    """
    We need to define min/max/resolution on both the Timestamp _instance_
    and Timestamp class.  On an instance, these depend on the object's _reso.
    On the class, we default to the values we would get with nanosecond _reso.

    See also: timedeltas.MinMaxReso
    """
    def __init__(self, name):
        # 初始化方法，接收一个名称 `name` 作为参数
        self._name = name

    def __get__(self, obj, type=None):
        # 获取方法，用于获取最小值、最大值或分辨率的值
        cls = Timestamp
        if self._name == "min":
            # 如果名称为 "min"，设置值为最小的整数值 + 1
            val = np.iinfo(np.int64).min + 1
        elif self._name == "max":
            # 如果名称为 "max"，设置值为最大的整数值
            val = np.iinfo(np.int64).max
        else:
            assert self._name == "resolution"
            # 如果名称为 "resolution"，设置值为默认分辨率 1
            val = 1
            cls = Timedelta

        if obj is None:
            # 如果对象为 None，表示在类上调用，返回相应的类实例
            return cls(val)
        elif self._name == "resolution":
            # 如果名称为 "resolution"，返回基于指定分辨率的 Timedelta 对象
            return Timedelta._from_value_and_reso(val, obj._creso)
        else:
            # 否则，返回基于指定值和分辨率的 Timestamp 对象
            return Timestamp._from_value_and_reso(val, obj._creso, tz=None)

    def __set__(self, obj, value):
        # 设置方法，抛出属性错误，表示该属性不可设置
        raise AttributeError(f"{self._name} is not settable.")


# ----------------------------------------------------------------------

# 定义一个 Cython 类 `_Timestamp`，继承自 `ABCTimestamp`
cdef class _Timestamp(ABCTimestamp):

    # 定义属性 `__array_priority__`，用于设置对象的数组优先级
    __array_priority__ = 100
    dayofweek = _Timestamp.day_of_week
    dayofyear = _Timestamp.day_of_year

    # 定义类属性 `min`、`max` 和 `resolution`，分别表示最小值、最大值和分辨率
    min = MinMaxReso("min")
    max = MinMaxReso("max")
    resolution = MinMaxReso("resolution")  # GH#21336, GH#21365

    @property
    # 定义 `value` 属性，用于获取时间戳的值
    def value(self) -> int:
        try:
            # 尝试将时间戳转换为指定单位（纳秒），并返回结果
            return convert_reso(self._value, self._creso, NPY_FR_ns, False)
        except OverflowError:
            # 如果发生溢出错误，抛出溢出错误，并提供相关的解决方案建议
            raise OverflowError(
                "Cannot convert Timestamp to nanoseconds without overflow. "
                "Use `.asm8.view('i8')` to cast represent Timestamp in its own "
                f"unit (here, {self.unit})."
            )

    @property
    def unit(self) -> str:
        """
        返回与 self._creso 相关联的时间单位的缩写。

        Examples
        --------
        >>> pd.Timestamp("2020-01-01 12:34:56").unit
        's'

        >>> pd.Timestamp("2020-01-01 12:34:56.123").unit
        'ms'

        >>> pd.Timestamp("2020-01-01 12:34:56.123456").unit
        'us'

        >>> pd.Timestamp("2020-01-01 12:34:56.123456789").unit
        'ns'
        """
        return npy_unit_to_abbrev(self._creso)

    # -----------------------------------------------------------------
    # Constructors

    @classmethod
    def _from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso, tzinfo tz):
        """
        根据给定的数值、时间单位和时区信息创建 Timestamp 对象。

        如果 value 等于 NPY_NAT，则返回 NaT。

        如果时间单位 reso 不在支持的范围内 ('s', 'ms', 'us', 'ns')，则抛出 NotImplementedError。

        创建 _TSObject 对象并初始化其值和时间单位，然后调用 pandas_datetime_to_datetimestruct
        函数填充时间结构体，最后根据时区信息可能调整时间对象的本地化。

        返回创建的 Timestamp 对象。
        """
        cdef:
            _TSObject obj = _TSObject()

        if value == NPY_NAT:
            return NaT

        if reso < NPY_DATETIMEUNIT.NPY_FR_s or reso > NPY_DATETIMEUNIT.NPY_FR_ns:
            raise NotImplementedError(
                "Only resolutions 's', 'ms', 'us', 'ns' are supported."
            )

        obj.value = value
        obj.creso = reso
        pandas_datetime_to_datetimestruct(value, reso, &obj.dts)
        maybe_localize_tso(obj, tz, reso)

        return create_timestamp_from_ts(
            value, obj.dts, tz=obj.tzinfo, fold=obj.fold, reso=reso
        )

    @classmethod
    def _from_dt64(cls, dt64: np.datetime64):
        """
        从 np.datetime64 对象构造一个 Timestamp 对象，并保留输入的时间分辨率。

        这主要是为了能够逐步实现非纳秒级别的支持（例如，最初只支持无时区的情况）。

        获取 np.datetime64 对象的时间单位 reso 和数值 value，然后调用 _from_value_and_reso 方法创建 Timestamp 对象。
        """
        cdef:
            int64_t value
            NPY_DATETIMEUNIT reso

        reso = get_datetime64_unit(dt64)
        value = cnp.get_datetime64_value(dt64)
        return cls._from_value_and_reso(value, reso, None)

    # -----------------------------------------------------------------

    def __hash__(_Timestamp self):
        """
        计算 Timestamp 对象的哈希值。

        如果 Timestamp 对象包含纳秒级别的时间信息，则返回基于 self._value 的哈希值。
        如果 self.year 不在 1 到 9999 之间，则返回 self._value 的哈希值。
        如果存在折叠时间（fold），则返回未折叠版本的哈希值。
        否则，返回未修改的 datetime.__hash__(self)。

        注意：该方法用于支持 Timestamp 对象的哈希比较。
        """
        if self.nanosecond:
            return hash(self._value)
        if not (1 <= self.year <= 9999):
            # 超出 pydatetime 的范围
            return hash(self._value)
        if self.fold:
            return datetime.__hash__(self.replace(fold=0))
        return datetime.__hash__(self)
    def __richcmp__(_Timestamp self, object other, int op):
        cdef:
            _Timestamp ots  # 声明一个变量ots，类型为_Timestamp，用于存储其他时间戳对象

        if isinstance(other, _Timestamp):  # 如果other是_Timestamp类型的实例
            ots = other  # 将ots设为other
        elif other is NaT:  # 如果other是NaT（Not a Time）对象
            return op == Py_NE  # 返回操作符op是否为不等于(Py_NE)
        elif cnp.is_datetime64_object(other):  # 如果other是numpy datetime64类型的对象
            ots = Timestamp(other)  # 使用other创建一个Timestamp对象，并赋值给ots
        elif PyDateTime_Check(other):  # 如果other是Python datetime.datetime类型的对象
            if self.nanosecond == 0:  # 如果self的纳秒部分为0
                val = self.to_pydatetime()  # 将self转换为Python的datetime.datetime对象
                return PyObject_RichCompareBool(val, other, op)  # 使用Python的对象比较函数判断self与other的关系

            try:
                ots = type(self)(other)  # 尝试使用self的类型创建一个ots对象
            except ValueError:
                return self._compare_outside_nanorange(other, op)  # 在纳秒范围之外的比较操作

        elif is_array(other):  # 如果other是数组类型
            # 避免递归错误 GH#15183
            if other.dtype.kind == "M":  # 如果数组的数据类型是时间类型
                if self.tz is None:  # 如果self没有时区信息
                    return PyObject_RichCompare(self.asm8, other, op)  # 比较self.asm8和other的关系
                elif op == Py_NE:  # 如果操作符是不等于
                    return np.ones(other.shape, dtype=np.bool_)  # 返回一个全为True的布尔数组
                elif op == Py_EQ:  # 如果操作符是等于
                    return np.zeros(other.shape, dtype=np.bool_)  # 返回一个全为False的布尔数组
                raise TypeError(  # 抛出类型错误异常
                    "Cannot compare tz-naive and tz-aware timestamps"
                )
            elif other.dtype.kind == "O":  # 如果数组的数据类型是对象类型
                # 逐元素操作
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],  # 对other数组中的每个元素与self进行比较
                    dtype=bool,
                )
            elif op == Py_NE:  # 如果操作符是不等于
                return np.ones(other.shape, dtype=np.bool_)  # 返回一个全为True的布尔数组
            elif op == Py_EQ:  # 如果操作符是等于
                return np.zeros(other.shape, dtype=np.bool_)  # 返回一个全为False的布尔数组
            return NotImplemented  # 如果无法比较，返回NotImplemented

        elif PyDate_Check(other):  # 如果other是Python datetime.date类型的对象
            # 返回NotImplemented将推迟到date的实现
            # 这里的标准库datetime会在比较前错误地删除时区信息并标准化为午夜
            # 我们遵循标准库datetime的行为，永远不相等
            if op == Py_EQ:  # 如果操作符是等于
                return False  # 返回False
            elif op == Py_NE:  # 如果操作符是不等于
                return True  # 返回True
            raise TypeError(  # 抛出类型错误异常
                "Cannot compare Timestamp with datetime.date. "
                "Use ts == pd.Timestamp(date) or ts.date() == date instead."
            )
        else:
            return NotImplemented  # 如果other类型无法处理，返回NotImplemented

        if not self._can_compare(ots):  # 如果无法比较self和ots
            if op == Py_NE or op == Py_EQ:  # 如果操作符是不等于或等于
                return NotImplemented  # 返回NotImplemented
            raise TypeError(  # 抛出类型错误异常
                "Cannot compare tz-naive and tz-aware timestamps"
            )
        if self._creso == ots._creso:  # 如果self的分辨率等于ots的分辨率
            return cmp_scalar(self._value, ots._value, op)  # 使用标量比较函数比较self和ots的值
        return self._compare_mismatched_resos(ots, op)  # 处理分辨率不匹配的情况的比较操作

    # TODO: copied from Timedelta; try to de-duplicate  # 提示：从Timedelta复制而来，尝试去重
    # 比较两个不匹配的时间戳对象是否相等或者大小关系，避免 numpy 自动溢出导致错误
    cdef bint _compare_mismatched_resos(self, _Timestamp other, int op):
        # 定义 numpy 的日期时间结构体变量，用于存储当前时间戳和其他时间戳的结构化信息
        cdef:
            npy_datetimestruct dts_self
            npy_datetimestruct dts_other
    
        # 调用 pandas_datetime_to_datetimestruct 函数，将当前时间戳转换为日期时间结构体
        pandas_datetime_to_datetimestruct(self._value, self._creso, &dts_self)
        # 调用 pandas_datetime_to_datetimestruct 函数，将其他时间戳转换为日期时间结构体
        pandas_datetime_to_datetimestruct(other._value, other._creso, &dts_other)
        # 调用 cmp_dtstructs 函数比较两个日期时间结构体的大小关系，并返回比较结果
        return cmp_dtstructs(&dts_self,  &dts_other, op)
    
    # 比较时间戳对象和日期对象之间的大小关系，处理可能的异常情况
    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1:
        # 将时间戳对象转换为 Python 的 datetime 对象，获取其值
        cdef:
            datetime dtval = self.to_pydatetime(warn=False)
    
        # 如果当前时间戳无法与其他日期对象进行比较，则返回 Not Implemented
        if not self._can_compare(other):
            return NotImplemented
    
        # 如果当前时间戳的纳秒部分为 0，则通过 PyObject_RichCompareBool 函数直接比较日期时间和其他日期对象
        if self.nanosecond == 0:
            return PyObject_RichCompareBool(dtval, other, op)
    
        # 否则，根据操作符 op 比较当前时间戳和其他日期对象的年份部分
        if op == Py_NE:
            return True
        if op == Py_EQ:
            return False
        if op == Py_LE or op == Py_LT:
            return self.year <= other.year
        if op == Py_GE or op == Py_GT:
            return self.year >= other.year
    
    # 判断当前时间戳对象是否能够与指定的日期对象进行比较
    cdef bint _can_compare(self, datetime other):
        # 如果当前时间戳对象具有时区信息，则判断指定的日期对象是否也具有时区信息
        if self.tzinfo is not None:
            return other.tzinfo is not None
        # 如果当前时间戳对象没有时区信息，则判断指定的日期对象是否也没有时区信息
        return other.tzinfo is None
    
    # 设置 Cython 函数的溢出检查为 True
    @cython.overflowcheck(True)
    # 定义特殊方法 __add__，用于处理时间增加操作
    def __add__(self, other):
        # 声明 nanos 变量，用于存储纳秒数，默认为 0
        cdef:
            int64_t nanos = 0

        # 如果 other 是任意时间增量标量，则转换为 Timedelta 对象处理
        if is_any_td_scalar(other):
            other = Timedelta(other)

            # TODO: 与 __sub__、Timedelta.__add__ 共享此部分逻辑
            # 匹配 numpy 的行为，将时间增量转换为更高精度。与 numpy 不同的是，我们在转换过程中遇到溢出会抛出异常，而不是静默处理。
            # 如果 self 的分辨率低于 other，则将 self 转换为具有 other 分辨率的对象（如果允许舍入）。
            if self._creso < other._creso:
                self = (<_Timestamp>self)._as_creso(other._creso, round_ok=True)
            # 如果 self 的分辨率高于 other，则将 other 转换为具有 self 分辨率的对象（如果允许舍入）。
            elif self._creso > other._creso:
                other = (<_Timedelta>other)._as_creso(self._creso, round_ok=True)

            # 获取 other 的时间增量值（以纳秒为单位）
            nanos = other._value

            try:
                # 尝试计算新的时间戳值
                new_value = self._value + nanos
                # 根据计算结果创建新的 _Timestamp 对象
                result = type(self)._from_value_and_reso(
                    new_value, reso=self._creso, tz=self.tzinfo
                )
            except OverflowError as err:
                # 处理溢出错误，构造错误信息并抛出 OutOfBoundsDatetime 异常
                new_value = int(self._value) + int(nanos)
                attrname = npy_unit_to_attrname[self._creso]
                raise OutOfBoundsDatetime(
                    f"Out of bounds {attrname} timestamp: {new_value}"
                ) from err

            return result

        # 如果 other 是整数对象，则抛出异常，不支持与整数的操作
        elif is_integer_object(other):
            raise integer_op_not_supported(self)

        # 如果 other 是数组对象，则根据其类型执行不同操作
        elif is_array(other):
            # 如果数组元素的数据类型是整数类型，则抛出异常，不支持与整数的操作
            if other.dtype.kind in "iu":
                raise integer_op_not_supported(self)
            # 如果数组元素的数据类型是时间类型 'm'
            if other.dtype.kind == "m":
                # 如果当前时间戳没有时区信息，则直接返回与数组相加后的结果
                if self.tz is None:
                    return self.asm8 + other
                # 如果当前时间戳有时区信息，则遍历数组每个元素进行递归加法运算，并返回结果数组
                return np.asarray(
                    [self + other[n] for n in range(len(other))],
                    dtype=object,
                )

        # 如果其他情况均不符合，则返回 NotImplemented，表示不支持当前操作
        return NotImplemented


    # 定义特殊方法 __radd__，处理反向加法操作
    def __radd__(self, other):
        # 为避免由于 NotImplemented 导致的无限递归，需在此处重复检查类型
        # 如果 other 是任意时间增量标量、整数对象或数组对象，则调用 __add__ 方法处理加法操作
        if is_any_td_scalar(other) or is_integer_object(other) or is_array(other):
            return self.__add__(other)
        # 如果不支持当前操作类型，则返回 NotImplemented
        return NotImplemented
    # 定义减法运算符重载方法，用于处理 self - other 的操作
    def __sub__(self, other):
        # 如果 other 是 NaT（Not a Time），返回 NaT
        if other is NaT:
            return NaT

        # 如果 other 是任意时间差标量或整数对象
        elif is_any_td_scalar(other) or is_integer_object(other):
            # 计算 other 的负值
            neg_other = -other
            # 返回 self 加上 neg_other 的结果
            return self + neg_other

        # 如果 other 是数组
        elif is_array(other):
            # 如果数组的数据类型是整数类型（unsigned 或者 signed），抛出异常
            if other.dtype.kind in "iu":
                raise integer_op_not_supported(self)
            # 如果数组的数据类型是日期时间类型
            if other.dtype.kind == "m":
                # 如果 self 的时区信息为 None，则执行 self.asm8 - other 操作
                if self.tz is None:
                    return self.asm8 - other
                # 否则，返回由 self 与 other 数组中每个元素相减得到的对象数组
                return np.asarray(
                    [self - other[n] for n in range(len(other))],
                    dtype=object,
                )
            # 其他情况返回 NotImplemented
            return NotImplemented

        # 如果需要的话，强制类型转换为 Timestamp 类型
        if PyDateTime_Check(other) or cnp.is_datetime64_object(other):
            # 判断是否 both_timestamps，用于确定 Timedelta(self - other) 是否应该
            # 抛出超出范围错误或者返回一个 timedelta。
            both_timestamps = isinstance(other, _Timestamp)
            other = type(self)(other)

            # 如果 self 和 other 的时区信息不一致，抛出类型错误
            if (self.tzinfo is None) ^ (other.tzinfo is None):
                raise TypeError(
                    "Cannot subtract tz-naive and tz-aware datetime-like objects."
                )

            # 匹配 numpy 的行为，将 self 和 other 转换为更高的分辨率。
            # 不同于 numpy，我们在转换期间遇到超出范围的情况时会抛出异常而不是默默溢出。
            if self._creso < other._creso:
                self = (<_Timestamp>self)._as_creso(other._creso, round_ok=True)
            elif self._creso > other._creso:
                other = (<_Timestamp>other)._as_creso(self._creso, round_ok=True)

            # 标量 Timestamp/datetime - Timestamp/datetime -> 返回一个 Timedelta
            try:
                # 计算 self._value - other._value，并返回 Timedelta 对象
                res_value = self._value - other._value
                return Timedelta._from_value_and_reso(res_value, self._creso)
            except (OverflowError, OutOfBoundsDatetime, OutOfBoundsTimedelta) as err:
                # 如果 both_timestamps 为 True，则抛出 OutOfBoundsDatetime 异常
                if both_timestamps:
                    raise OutOfBoundsDatetime(
                        "Result is too large for pandas.Timedelta. Convert inputs "
                        "to datetime.datetime with 'Timestamp.to_pydatetime()' "
                        "before subtracting."
                    ) from err
                # 在 stata 测试中到达此处，回退到标准库 datetime 方法，并返回标准库的 timedelta 对象
                pass

        # 如果上述条件都不满足，则返回 NotImplemented
        return NotImplemented

    # 右侧减法运算符重载方法，用于处理 other - self 的操作
    def __rsub__(self, other):
        # 如果 other 是 Python 标准库的 datetime 对象
        if PyDateTime_Check(other):
            try:
                # 返回 type(self)(other) - self 的结果
                return type(self)(other) - self
            except (OverflowError, OutOfBoundsDatetime) as err:
                # 在 stata 测试中到达此处，回退到标准库 datetime 方法，并返回标准库的 timedelta 对象
                pass
        # 如果 other 是 numpy 的 datetime64 对象
        elif cnp.is_datetime64_object(other):
            # 返回 type(self)(other) - self 的结果
            return type(self)(other) - self
        # 其他情况返回 NotImplemented
        return NotImplemented
    # -----------------------------------------------------------------

    cdef int64_t _maybe_convert_value_to_local(self) except? -1:
        """Convert UTC i8 value to local i8 value if tz exists"""
        # 定义本地变量和结构体
        cdef:
            int64_t val  # 用于存储转换后的时间值
            tzinfo own_tz = self.tzinfo  # 获取对象的时区信息
            npy_datetimestruct dts  # NumPy日期时间结构体

        # 如果存在时区并且不是UTC时区，则进行转换
        if own_tz is not None and not is_utc(own_tz):
            # 将Python datetime对象转换为日期时间结构体
            pydatetime_to_dtstruct(self, &dts)
            # 使用结构体转换为datetime并加上纳秒部分
            val = npy_datetimestruct_to_datetime(self._creso, &dts) + self.nanosecond
        else:
            # 否则直接使用原始的UTC时间值
            val = self._value
        return val  # 返回转换后的时间值

    @cython.boundscheck(False)
    cdef bint _get_start_end_field(self, str field, freq):
        # 定义本地变量和数据结构
        cdef:
            int64_t val  # 用于存储时间值
            dict kwds  # 用于存储关键字参数
            ndarray[uint8_t, cast=True] out  # 输出的NumPy数组
            int month_kw  # 月份关键字参数

        # 如果频率存在，则获取关键字参数和月份
        if freq:
            kwds = freq.kwds
            month_kw = kwds.get("startingMonth", kwds.get("month", 12))
            freq_name = freq.name
        else:
            # 否则默认月份为12
            month_kw = 12
            freq_name = None

        # 获取转换后的时间值
        val = self._maybe_convert_value_to_local()

        # 调用函数获取开始和结束字段的数据
        out = get_start_end_field(np.array([val], dtype=np.int64),
                                  field, freq_name, month_kw, self._creso)
        return out[0]  # 返回处理后的第一个元素

    @property
    def is_month_start(self) -> bool:
        """
        Check if the date is the first day of the month.

        Returns
        -------
        bool
            True if the date is the first day of the month.

        See Also
        --------
        Timestamp.is_month_end : Similar property indicating the last day of the month.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_month_start
        True
        """
        return self.day == 1  # 返回日期是否为月初的布尔值

    @property
    def is_month_end(self) -> bool:
        """
        Check if the date is the last day of the month.

        Returns
        -------
        bool
            True if the date is the last day of the month.

        See Also
        --------
        Timestamp.is_month_start : Similar property indicating month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_month_end
        True
        """
        return self.day == self.days_in_month  # 返回日期是否为月末的布尔值

    @property
    # 继续添加更多属性或方法
    def is_quarter_start(self) -> bool:
        """
        Check if the date is the first day of the quarter.

        Returns
        -------
        bool
            True if date is first day of the quarter.

        See Also
        --------
        Timestamp.is_quarter_end : Similar property indicating the quarter end.
        Timestamp.quarter : Return the quarter of the date.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_start
        False

        >>> ts = pd.Timestamp(2020, 4, 1)
        >>> ts.is_quarter_start
        True
        """
        # Check if the day is 1 (first day of any month) and the month modulo 3 equals 1 (quarter start)
        return self.day == 1 and self.month % 3 == 1

    @property
    def is_quarter_end(self) -> bool:
        """
        Check if date is last day of the quarter.

        Returns
        -------
        bool
            True if date is last day of the quarter.

        See Also
        --------
        Timestamp.is_quarter_start : Similar property indicating the quarter start.
        Timestamp.quarter : Return the quarter of the date.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_end
        False

        >>> ts = pd.Timestamp(2020, 3, 31)
        >>> ts.is_quarter_end
        True
        """
        # Check if the day is equal to the number of days in the month (last day of the month)
        return (self.month % 3) == 0 and self.day == self.days_in_month

    @property
    def is_year_start(self) -> bool:
        """
        Return True if date is first day of the year.

        Returns
        -------
        bool

        See Also
        --------
        Timestamp.is_year_end : Similar property indicating the end of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_year_start
        True
        """
        # Check if both day and month are 1 (January 1st)
        return self.day == self.month == 1

    @property
    def is_year_end(self) -> bool:
        """
        Return True if date is last day of the year.

        Returns
        -------
        bool

        See Also
        --------
        Timestamp.is_year_start : Similar property indicating the start of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_year_end
        True
        """
        # Check if the month is December (12) and the day is 31
        return self.month == 12 and self.day == 31

    @cython.boundscheck(False)
    cdef _get_date_name_field(self, str field, object locale):
        """
        This is a Cython function that retrieves a specific date name field.

        Parameters
        ----------
        field : str
            The field name to retrieve.
        locale : object
            The locale object for localization.

        Returns
        -------
        object
            The retrieved date name field.

        Notes
        -----
        This function is optimized for performance using Cython.

        See Also
        --------
        get_date_name_field : Function for retrieving date name fields.
        """
        cdef:
            int64_t val
            object[::1] out

        val = self._maybe_convert_value_to_local()

        out = get_date_name_field(np.array([val], dtype=np.int64),
                                  field, locale=locale, reso=self._creso)
        return out[0]
    def day_name(self, locale=None) -> str:
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str
            Day name based on the locale.

        See Also
        --------
        Timestamp.day_of_week : Return day of the week.
        Timestamp.day_of_year : Return day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """
        return self._get_date_name_field("day_name", locale)

    def month_name(self, locale=None) -> str:
        """
        Return the month name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str
            Month name based on the locale.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """
        return self._get_date_name_field("month_name", locale)

    @property
    def is_leap_year(self) -> bool:
        """
        Return True if year is a leap year.

        A leap year is a year, which has 366 days (instead of 365) including 29th of
        February as an intercalary day. Leap years are years which are multiples of
        four with the exception of years divisible by 100 but not by 400.

        Returns
        -------
        bool
            True if year is a leap year, else False

        See Also
        --------
        Period.is_leap_year : Return True if the period’s year is in a leap year.
        DatetimeIndex.is_leap_year : Boolean indicator if the date belongs to a
            leap year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_leap_year
        True
        """
        return bool(ccalendar.is_leapyear(self.year))

    @property
    def day_of_week(self) -> int:
        """
        Return day of the week.

        Returns
        -------
        int
            Numeric representation of the day of the week (Monday is 0, Sunday is 6).

        See Also
        --------
        Timestamp.isoweekday : Return the ISO day of the week represented by the date.
        Timestamp.weekday : Return the day of the week represented by the date.
        Timestamp.day_of_year : Return day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_week
        5
        """
        return self.weekday()

    @property
    def day_of_year(self) -> int:
        """
        Return the ordinal day of the year.

        Returns
        -------
        int
            Day of the year (1 through 366 for leap years).

        See Also
        --------
        Timestamp.day_of_week : Return day of the week.
        Timestamp.day_of_week : Return the day of the week represented by the date.
        Timestamp.is_leap_year : Return True if year is a leap year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_year
        74
        """
        return self._data.day_of_year
    def day_of_year(self) -> int:
        """
        Return the day of the year.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.day_of_week : Return day of the week.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_year
        74
        """
        # 使用自定义的日历模块获取当前日期在年中的第几天
        return ccalendar.get_day_of_year(self.year, self.month, self.day)

    @property
    def quarter(self) -> int:
        """
        Return the quarter of the year.

        Returns
        -------
        int

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.quarter
        1
        """
        # 计算当前日期所在的季度
        return ((self.month - 1) // 3) + 1

    @property
    def week(self) -> int:
        """
        Return the week number of the year.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.weekday : Return the day of the week.
        Timestamp.quarter : Return the quarter of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.week
        11
        """
        # 使用自定义的日历模块获取当前日期在年中的第几周
        return ccalendar.get_week_of_year(self.year, self.month, self.day)

    @property
    def days_in_month(self) -> int:
        """
        Return the number of days in the month.

        Returns
        -------
        int

        See Also
        --------
        Timestamp.month_name : Return the month name of the Timestamp with
            specified locale.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.days_in_month
        31
        """
        # 使用自定义的日历模块获取当前日期所在月份的天数
        return ccalendar.get_days_in_month(self.year, self.month)

    # -----------------------------------------------------------------
    # Transformation Methods

    def normalize(self) -> "Timestamp":
        """
        Normalize Timestamp to midnight, preserving tz information.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15, 30)
        >>> ts.normalize()
        Timestamp('2020-03-14 00:00:00')
        """
        # 将时间戳规范化为午夜，保留时区信息
        cdef:
            local_val = self._maybe_convert_value_to_local()
            int64_t normalized
            int64_t ppd = periods_per_day(self._creso)
            _Timestamp ts

        # 调用底层函数将时间戳规范化为整型表示的午夜时间戳
        normalized = normalize_i8_stamp(local_val, ppd)
        # 根据规范化后的时间戳创建新的 Timestamp 对象
        ts = type(self)._from_value_and_reso(normalized, reso=self._creso, tz=None)
        return ts.tz_localize(self.tzinfo)

    # -----------------------------------------------------------------
    # Pickle Methods

    def __reduce_ex__(self, protocol):
        # python 3.6 compat
        # https://bugs.python.org/issue28730
        # now __reduce_ex__ is defined and higher priority than __reduce__
        # 返回用于对象序列化的 reduce 方法
        return self.__reduce__()
    def __setstate__(self, state):
        # 将对象状态的第一个元素赋给对象的值
        self._value= state[0]
        # 将对象状态的第三个元素赋给对象的时区信息
        self.tzinfo = state[2]

        # 如果状态长度为3，表示旧版 pickle 格式
        if len(state) == 3:
            # 使用旧版的时间分辨率（numpy datetime64 中的标志）
            # TODO: 2022-05-10 无测试案例到达这里
            reso = NPY_FR_ns
        else:
            # 否则，使用状态的第五个元素作为时间分辨率
            reso = state[4]
        # 将时间分辨率保存在对象的私有变量中
        self._creso = reso

    def __reduce__(self):
        # 返回用于反序列化对象的函数和对象的状态
        object_state = self._value, None, self.tzinfo, self._creso
        return (_unpickle_timestamp, object_state)

    # -----------------------------------------------------------------
    # 渲染方法

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        """
        根据 ISO 8601 格式返回时间的字符串表示。

        完整格式为 'YYYY-MM-DD HH:MM:SS.mmmmmmnnn'。
        如果 self.microsecond == 0 且 self.nanosecond == 0，则默认不包含小数部分。

        如果 self.tzinfo 不为 None，则附加 UTC 偏移量，格式为 'YYYY-MM-DD HH:MM:SS.mmmmmmnnn+HH:MM'。

        Parameters
        ----------
        sep : str, 默认 'T'
            日期和时间之间的分隔符。

        timespec : str, 默认 'auto'
            指定要包含的时间部分的详细程度。
            有效值为 'auto', 'hours', 'minutes', 'seconds',
            'milliseconds', 'microseconds', 'nanoseconds'。

        Returns
        -------
        str

        See Also
        --------
        Timestamp.strftime : 返回格式化的字符串。
        Timestamp.isocalendar : 返回包含 ISO 年份、周数和星期几的元组。

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.isoformat()
        '2020-03-14T15:32:52.192548651'
        >>> ts.isoformat(timespec='microseconds')
        '2020-03-14T15:32:52.192548'
        """
        base_ts = "microseconds" if timespec == "nanoseconds" else timespec
        base = super(_Timestamp, self).isoformat(sep=sep, timespec=base_ts)
        # 将假的年份 1970 替换为真实的年份
        base = f"{self.year:04d}-" + base.split("-", 1)[1]

        # 如果 self.nanosecond 为 0 且 timespec 不是 'nanoseconds'，则直接返回 base
        if self.nanosecond == 0 and timespec != "nanoseconds":
            return base

        # 如果有时区信息
        if self.tzinfo is not None:
            base1, base2 = base[:-6], base[-6:]
        else:
            base1, base2 = base, ""

        # 如果要求 'nanoseconds' 或者 self.nanosecond 非零，处理纳秒部分
        if timespec == "nanoseconds" or (timespec == "auto" and self.nanosecond):
            if self.microsecond or timespec == "nanoseconds":
                base1 += f"{self.nanosecond:03d}"
            else:
                base1 += f".{self.nanosecond:09d}"

        return base1 + base2
    # 返回对象的字符串表示形式，应为一个时间戳的描述
    def __repr__(self) -> str:
        # 初始时间戳字符串基于基本表示
        stamp = self._repr_base
        zone = None

        # 如果有时区信息
        if self.tzinfo is not None:
            try:
                # 尝试获取时间戳的时区偏移
                stamp += self.strftime("%z")
            except ValueError:
                # 处理时间戳不支持 %z 格式的情况，回退到特定年份（例如2000年）再尝试
                year2000 = self.replace(year=2000)
                stamp += year2000.strftime("%z")

            # 获取时区对象
            zone = get_timezone(self.tzinfo)
            try:
                # 尝试获取时区的字符串表示形式
                stamp += zone.strftime(" %%Z")
            except AttributeError:
                # 捕获时区对象没有 `strftime` 方法的异常，例如 tzlocal
                pass

        # 如果存在时区信息，将其包含在返回字符串中
        tz = f", tz='{zone}'" if zone is not None else ""

        # 返回完整的时间戳字符串表示形式
        return f"Timestamp('{stamp}'{tz})"

    @property
    def _repr_base(self) -> str:
        # 返回时间戳基本表示形式，由日期和时间组成的字符串
        return f"{self._date_repr} {self._time_repr}"

    @property
    def _date_repr(self) -> str:
        # 返回日期的字符串表示形式，格式为 YYYY-MM-DD
        # 由于性能和兼容性，这里使用手动格式化日期
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

    @property
    def _time_repr(self) -> str:
        # 返回时间的字符串表示形式，格式为 HH:MM:SS.microseconds
        result = f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

        # 如果存在纳秒部分，则包含在时间字符串中
        if self.nanosecond != 0:
            result += f".{self.nanosecond + 1000 * self.microsecond:09d}"
        elif self.microsecond != 0:
            result += f".{self.microsecond:06d}"

        # 返回完整的时间字符串表示形式
        return result

    # -----------------------------------------------------------------
    # Conversion Methods

    @cython.cdivision(False)
    # 将时间戳转换为指定精度的 C 时间戳对象
    cdef _Timestamp _as_creso(self, NPY_DATETIMEUNIT creso, bint round_ok=True):
        cdef:
            int64_t value

        # 如果目标精度与当前精度相同，直接返回当前对象
        if creso == self._creso:
            return self

        # 尝试将当前时间戳转换为指定精度的值
        try:
            value = convert_reso(self._value, self._creso, creso, round_ok=round_ok)
        except OverflowError as err:
            # 处理转换溢出的异常情况，提供详细错误信息
            unit = npy_unit_to_abbrev(creso)
            raise OutOfBoundsDatetime(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

        # 返回新的时间戳对象，基于转换后的值和指定的精度
        return type(self)._from_value_and_reso(value, reso=creso, tz=self.tzinfo)


这些注释完整地解释了每个函数和相关的代码行的作用和含义，确保了每个语句的功能清晰可见。
    def as_unit(self, str unit, bint round_ok=True):
        """
        Convert the underlying int64 representation to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
            The unit to which the timestamp should be converted.
        round_ok : bool, default True
            If False and the conversion requires rounding, raise an exception.

        Returns
        -------
        Timestamp
            A new Timestamp object representing the timestamp in the specified unit.

        See Also
        --------
        Timestamp.asm8 : Return numpy datetime64 format in nanoseconds.
        Timestamp.to_pydatetime : Convert Timestamp object to a native
            Python datetime object.
        to_timedelta : Convert argument into timedelta object,
            which can represent differences in times.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 00:00:00.01')
        >>> ts
        Timestamp('2023-01-01 00:00:00.010000')
        >>> ts.unit
        'ms'
        >>> ts = ts.as_unit('s')
        >>> ts
        Timestamp('2023-01-01 00:00:00')
        >>> ts.unit
        's'
        """
        # Define the numpy dtype corresponding to the requested unit
        dtype = np.dtype(f"M8[{unit}]")
        # Determine the resolution corresponding to the dtype
        reso = get_unit_from_dtype(dtype)
        try:
            # Convert the timestamp to the specified resolution
            return self._as_creso(reso, round_ok=round_ok)
        except OverflowError as err:
            # Raise an exception if the conversion causes overflow
            raise OutOfBoundsDatetime(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

    @property
    def asm8(self) -> np.datetime64:
        """
        Return numpy datetime64 format in nanoseconds.

        See Also
        --------
        numpy.datetime64 : Numpy datatype for dates and times with high precision.
        Timestamp.to_numpy : Convert the Timestamp to a NumPy datetime64.
        to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15)
        >>> ts.asm8
        numpy.datetime64('2020-03-14T15:00:00.000000')
        """
        # Return the numpy datetime64 representation of the Timestamp
        return self.to_datetime64()

    def timestamp(self):
        """
        Return POSIX timestamp as float.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.timestamp()
        1584199972.192548
        """
        # Calculate the POSIX timestamp using the internal resolution
        denom = periods_per_second(self._creso)
        return round(self._value / denom, 6)
    # 定义一个 Cython 方法，将 Timestamp 对象转换为原生 Python datetime 对象
    def to_pydatetime(_Timestamp self, bint warn=True):
        """
        Convert a Timestamp object to a native Python datetime object.

        This method is useful for when you need to utilize a pandas Timestamp
        object in contexts where native Python datetime objects are expected
        or required. The conversion discards the nanoseconds component, and a
        warning can be issued in such cases if desired.

        Parameters
        ----------
        warn : bool, default True
            If True, issues a warning when the timestamp includes nonzero
            nanoseconds, as these will be discarded during the conversion.

        Returns
        -------
        datetime.datetime or NaT
            Returns a datetime.datetime object representing the timestamp,
            with year, month, day, hour, minute, second, and microsecond components.
            If the timestamp is NaT (Not a Time), returns NaT.

        See Also
        --------
        datetime.datetime : The standard Python datetime class that this method
            returns.
        Timestamp.timestamp : Convert a Timestamp object to POSIX timestamp.
        Timestamp.to_datetime64 : Convert a Timestamp object to numpy.datetime64.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """
        # 如果时间戳包含非零纳秒且 warn 为 True，则发出警告，因为这些纳秒将在转换过程中被丢弃
        if self.nanosecond != 0 and warn:
            warnings.warn("Discarding nonzero nanoseconds in conversion.",
                          UserWarning, stacklevel=find_stack_level())

        # 返回一个 datetime.datetime 对象，表示时间戳，包括年、月、日、时、分、秒和微秒组件
        return datetime(self.year, self.month, self.day,
                        self.hour, self.minute, self.second,
                        self.microsecond, self.tzinfo, fold=self.fold)


    # 定义一个 Cython 方法，将 Timestamp 对象转换为 numpy.datetime64 对象
    def to_datetime64(self):
        """
        Return a numpy.datetime64 object with same precision.

        Examples
        --------
        >>> ts = pd.Timestamp(year=2023, month=1, day=1,
        ...                   hour=10, second=15)
        >>> ts
        Timestamp('2023-01-01 10:00:15')
        >>> ts.to_datetime64()
        numpy.datetime64('2023-01-01T10:00:15.000000')
        """
        # TODO: find a way to construct dt64 directly from _reso
        # 将 _creso 转换为 numpy 日期时间单位的缩写
        abbrev = npy_unit_to_abbrev(self._creso)
        # 返回一个 numpy.datetime64 对象，精度与原对象相同
        return np.datetime64(self._value, abbrev)
    def to_numpy(self, dtype=None, copy=False) -> np.datetime64:
        """
        Convert the Timestamp to a NumPy datetime64.

        This is an alias method for `Timestamp.to_datetime64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Returns
        -------
        numpy.datetime64
            NumPy datetime64 representation of the Timestamp.

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.to_numpy()
        numpy.datetime64('2020-03-14T15:32:52.192548651')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64('NaT')
        """
        if dtype is not None or copy is not False:
            # 如果指定了dtype或copy不为False，则抛出数值错误
            raise ValueError(
                "Timestamp.to_numpy dtype and copy arguments are ignored."
            )
        # 调用实例的to_datetime64方法返回numpy.datetime64表示
        return self.to_datetime64()

    def to_period(self, freq=None):
        """
        Return an period of which this timestamp is an observation.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> # Year end frequency
        >>> ts.to_period(freq='Y')
        Period('2020', 'Y-DEC')

        >>> # Month end frequency
        >>> ts.to_period(freq='M')
        Period('2020-03', 'M')

        >>> # Weekly frequency
        >>> ts.to_period(freq='W')
        Period('2020-03-09/2020-03-15', 'W-SUN')

        >>> # Quarter end frequency
        >>> ts.to_period(freq='Q')
        Period('2020Q1', 'Q-DEC')
        """
        from pandas import Period

        if self.tz is not None:
            # 如果时间戳有时区信息，则发出警告，转换为Period将丢失时区信息
            warnings.warn(
                "Converting to Period representation will drop timezone information.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        # 返回一个Period对象，表示该时间戳是观测的周期
        return Period(self, freq=freq)
# ----------------------------------------------------------------------
# Python front end to C extension type _Timestamp
# This serves as the box for datetime64

# 定义 Timestamp 类，作为 datetime64 的 pandas 替代品
class Timestamp(_Timestamp):
    """
    Pandas replacement for python datetime.datetime object.

    Timestamp is the pandas equivalent of python's Datetime
    and is interchangeable with it in most cases. It's the type used
    for the entries that make up a DatetimeIndex, and other timeseries
    oriented data structures in pandas.

    Parameters
    ----------
    ts_input : datetime-like, str, int, float
        Value to be converted to Timestamp.
    year : int
        Value of year.
    month : int
        Value of month.
    day : int
        Value of day.
    hour : int, optional, default 0
        Value of hour.
    minute : int, optional, default 0
        Value of minute.
    second : int, optional, default 0
        Value of second.
    microsecond : int, optional, default 0
        Value of microsecond.
    tzinfo : datetime.tzinfo, optional, default None
        Timezone info.
    nanosecond : int, optional, default 0
        Value of nanosecond.
    tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
        Time zone for time which Timestamp will have.
    unit : str
        Unit used for conversion if ts_input is of type int or float. The
        valid values are 'W', 'D', 'h', 'm', 's', 'ms', 'us', and 'ns'. For
        example, 's' means seconds and 'ms' means milliseconds.

        For float inputs, the result will be stored in nanoseconds, and
        the unit attribute will be set as ``'ns'``.
    fold : {0, 1}, default None, keyword-only
        Due to daylight saving time, one wall clock time can occur twice
        when shifting from summer to winter time; fold describes whether the
        datetime-like corresponds  to the first (0) or the second time (1)
        the wall clock hits the ambiguous time.

    See Also
    --------
    Timedelta : Represents a duration, the difference between two dates or times.
    datetime.datetime : Python datetime.datetime object.

    Notes
    -----
    There are essentially three calling conventions for the constructor. The
    primary form accepts four parameters. They can be passed by position or
    keyword.

    The other two forms mimic the parameters from ``datetime.datetime``. They
    can be passed by either position or keyword, but not both mixed together.

    Examples
    --------
    Using the primary calling convention:

    This converts a datetime-like string

    >>> pd.Timestamp('2017-01-01T12')
    Timestamp('2017-01-01 12:00:00')

    This converts a float representing a Unix epoch in units of seconds

    >>> pd.Timestamp(1513393355.5, unit='s')
    Timestamp('2017-12-16 03:02:35.500000')

    This converts an int representing a Unix-epoch in units of weeks

    >>> pd.Timestamp(1535, unit='W')
    Timestamp('1999-06-03 00:00:00')
    This converts an int representing a Unix-epoch in units of seconds
    and for a particular timezone

    >>> pd.Timestamp(1513393355, unit='s', tz='US/Pacific')
    Timestamp('2017-12-15 19:02:35-0800', tz='US/Pacific')

    Using the other two forms that mimic the API for ``datetime.datetime``:

    >>> pd.Timestamp(2017, 1, 1, 12)
    Timestamp('2017-01-01 12:00:00')

    >>> pd.Timestamp(year=2017, month=1, day=1, hour=12)
    Timestamp('2017-01-01 12:00:00')
    """



    @classmethod
    def fromordinal(cls, ordinal, tz=None):
        """
        Construct a timestamp from a proleptic Gregorian ordinal.

        Parameters
        ----------
        ordinal : int
            Date corresponding to a proleptic Gregorian ordinal.
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for the Timestamp.

        Notes
        -----
        By definition there cannot be any tz info on the ordinal itself.

        Examples
        --------
        >>> pd.Timestamp.fromordinal(737425)
        Timestamp('2020-01-01 00:00:00')
        """
        # 使用给定的 proleptic Gregorian ordinal 构造一个 timestamp 对象，并可选地指定时区
        return cls(datetime.fromordinal(ordinal), tz=tz)

    @classmethod
    def now(cls, tz=None):
        """
        Return new Timestamp object representing current time local to tz.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.now()  # doctest: +SKIP
        Timestamp('2020-11-16 22:06:16.378782')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """
        # 返回一个代表当前本地时间的 Timestamp 对象，可以指定时区
        if isinstance(tz, str):
            tz = maybe_get_tz(tz)
        return cls(datetime.now(tz))

    @classmethod
    def today(cls, tz=None):
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
        # 返回当前本地时区的当前时间的 Timestamp 对象，可以指定时区
        return cls.now(tz)

    @classmethod


These annotations explain each method's purpose, parameter details, and usage examples, following the provided format and guidelines.
    @classmethod
    def utcnow(cls):
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        See Also
        --------
        Timestamp : Constructs an arbitrary datetime.
        Timestamp.now : Return the current local date and time, which
            can be timezone-aware.
        Timestamp.today : Return the current local date and time with
            timezone information set to None.
        to_datetime : Convert argument to datetime.
        date_range : Return a fixed frequency DatetimeIndex.
        Timestamp.utctimetuple : Return UTC time tuple, compatible with
            time.localtime().

        Examples
        --------
        >>> pd.Timestamp.utcnow()   # doctest: +SKIP
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """
        # 发出警告，通知用户函数即将被弃用
        warnings.warn(
            # 标准库中的 datetime.utcnow 已弃用，我们也弃用以匹配它
            "Timestamp.utcnow is deprecated and will be removed in a future "
            "version. Use Timestamp.now('UTC') instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        # 调用当前类的 now 方法，传递 UTC 时区，返回时间戳对象
        return cls.now(UTC)

    @classmethod
    def utcfromtimestamp(cls, ts):
        """
        Timestamp.utcfromtimestamp(ts)

        Construct a timezone-aware UTC datetime from a POSIX timestamp.

        Notes
        -----
        Timestamp.utcfromtimestamp behavior differs from datetime.utcfromtimestamp
        in returning a timezone-aware object.

        Examples
        --------
        >>> pd.Timestamp.utcfromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52+0000', tz='UTC')
        """
        # GH#22451
        # 发出警告，通知用户函数即将被弃用
        warnings.warn(
            # 标准库中的 datetime.utcfromtimestamp 已弃用，我们也弃用以匹配它
            "Timestamp.utcfromtimestamp is deprecated and will be removed in a "
            "future version. Use Timestamp.fromtimestamp(ts, 'UTC') instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        # 调用当前类的 fromtimestamp 方法，传递 POSIX 时间戳和 UTC 时区，返回时间戳对象
        return cls.fromtimestamp(ts, tz="UTC")

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        """
        Timestamp.fromtimestamp(ts)

        Transform timestamp[, tz] to tz's local time from POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.fromtimestamp(1584199972)  # doctest: +SKIP
        Timestamp('2020-03-14 15:32:52')

        Note that the output may change depending on your local time.
        """
        # 获取可能的时区对象
        tz = maybe_get_tz(tz)
        # 调用当前类的 fromtimestamp 方法，传递 POSIX 时间戳和指定时区，返回时间戳对象
        return cls(datetime.fromtimestamp(ts, tz))
    # 返回格式化后的时间戳字符串
    def strftime(self, format):
        """
        Return a formatted string of the Timestamp.

        Parameters
        ----------
        format : str
            Format string to convert Timestamp to string.
            See strftime documentation for more information on the format string:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        See Also
        --------
        Timestamp.isoformat : Return the time formatted according to ISO 8601.
        pd.to_datetime : Convert argument to datetime.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.strftime('%Y-%m-%d %X')
        '2020-03-14 15:32:52'
        """
        try:
            # 根据时间戳的各个组成部分创建一个 datetime 对象
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            # 如果组成部分不合法，抛出异常
            raise NotImplementedError(
                "strftime not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
                "For now, please call the components you need (such as `.year` "
                "and `.month`) and construct your string from there."
            ) from err
        # 使用 datetime 对象的 strftime 方法返回格式化后的字符串
        return _dt.strftime(format)

    # 返回与 ctime 格式相匹配的字符串
    def ctime(self):
        """
        Return ctime() style string.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.ctime()
        'Sun Jan  1 10:00:00 2023'
        """
        try:
            # 根据时间戳的各个组成部分创建一个 datetime 对象
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            # 如果组成部分不合法，抛出异常
            raise NotImplementedError(
                "ctime not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
                "For now, please call the components you need (such as `.year` "
                "and `.month`) and construct your string from there."
            ) from err
        # 使用 datetime 对象的 ctime 方法返回 ctime 格式的字符串
        return _dt.ctime()

    # 返回与日期部分相同的 date 对象
    def date(self):
        """
        Return date object with same year, month and day.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.date()
        datetime.date(2023, 1, 1)
        """
        try:
            # 根据时间戳的年月日部分创建一个 date 对象
            _dt = dt.date(self.year, self.month, self.day)
        except ValueError as err:
            # 如果组成部分不合法，抛出异常
            raise NotImplementedError(
                "date not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        # 返回构建的 date 对象
        return _dt
    def dst(self):
        """
        Return the daylight saving time (DST) adjustment.

        This method returns the DST adjustment as a `datetime.timedelta` object
        if the Timestamp is timezone-aware and DST is applicable.

        See Also
        --------
        Timestamp.tz_localize : Localize the Timestamp to a timezone.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.

        Examples
        --------
        >>> ts = pd.Timestamp('2000-06-01 00:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2000-06-01 00:00:00+0200', tz='Europe/Brussels')
        >>> ts.dst()
        datetime.timedelta(seconds=3600)
        """
        # 调用父类方法获取时间戳的夏令时调整
        return super().dst()

    def isocalendar(self):
        """
        Return a named tuple containing ISO year, week number, and weekday.

        See Also
        --------
        DatetimeIndex.isocalendar : Return a 3-tuple containing ISO year,
            week number, and weekday for the given DatetimeIndex object.
        datetime.date.isocalendar : The equivalent method for `datetime.date` objects.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isocalendar()
        datetime.IsoCalendarDate(year=2022, week=52, weekday=7)
        """
        # 尝试创建一个日期时间对象并返回其 ISO 日历格式的命名元组
        try:
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            # 如果出现值错误，则抛出未实现错误
            raise NotImplementedError(
                "isocalendar not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.isocalendar()

    def tzname(self):
        """
        Return time zone name.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.tzname()
        'CET'
        """
        # 调用父类方法返回时间戳的时区名称
        return super().tzname()

    def utcoffset(self):
        """
        Return utc offset.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utcoffset()
        datetime.timedelta(seconds=3600)
        """
        # 调用父类方法返回时间戳的 UTC 偏移量
        return super().utcoffset()
    def utctimetuple(self):
        """
        Return UTC time tuple, compatible with time.localtime().

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utctimetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1, tm_hour=9,
        tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=0)
        """
        # 调用父类方法获取 UTC 时间元组
        return super().utctimetuple()

    def time(self):
        """
        Return time object with same time but with tzinfo=None.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.time()
        datetime.time(10, 0)
        """
        # 调用父类方法返回时间对象，不带时区信息
        return super().time()

    def timetuple(self):
        """
        Return time tuple, compatible with time.localtime().

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.timetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1,
        tm_hour=10, tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=-1)
        """
        try:
            # 创建一个 datetime 对象来获取时间元组
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            # 如果值错误，则抛出未实现错误
            raise NotImplementedError(
                "timetuple not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.timetuple()

    def timetz(self):
        """
        Return time object with same time and tzinfo.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.timetz()
        datetime.time(10, 0, tzinfo=<DstTzInfo 'Europe/Brussels' CET+1:00:00 STD>)
        """
        # 调用父类方法返回带有相同时间和时区信息的时间对象
        return super().timetz()

    def toordinal(self):
        """
        Return proleptic Gregorian ordinal. January 1 of year 1 is day 1.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:50')
        >>> ts
        Timestamp('2023-01-01 10:00:50')
        >>> ts.toordinal()
        738521
        """
        try:
            # 创建一个 datetime 对象来获取日期的儒略日
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            # 如果值错误，则抛出未实现错误
            raise NotImplementedError(
                "toordinal not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.toordinal()

    # Issue 25016.
    @classmethod
    def strptime(cls, date_string, format):
        """
        Timestamp.strptime(string, format)

        Function is not implemented. Use pd.to_datetime().

        Examples
        --------
        >>> pd.Timestamp.strptime("2023-01-01", "%d/%m/%y")
        Traceback (most recent call last):
        NotImplementedError
        """
        # 抛出未实现错误，建议使用 pd.to_datetime() 替代
        raise NotImplementedError(
            "Timestamp.strptime() is not implemented. "
            "Use to_datetime() to parse date strings."
        )

    @classmethod
    def combine(cls, date, time):
        """
        Timestamp.combine(date, time)

        Combine date, time into datetime with same date and time fields.

        Examples
        --------
        >>> from datetime import date, time
        >>> pd.Timestamp.combine(date(2020, 3, 14), time(15, 30, 15))
        Timestamp('2020-03-14 15:30:15')
        """
        # 使用给定的 date 和 time 创建一个新的 Timestamp 对象
        return cls(datetime.combine(date, time))

    def __new__(
        cls,
        object ts_input=_no_input,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        tzinfo_type tzinfo=None,
        *,
        nanosecond=None,
        tz=_no_input,
        unit=None,
        fold=None,
    ):
        """
        Constructor for Timestamp objects.

        Parameters
        ----------
        ts_input : object, optional
            Timestamp input object.
        year : int, optional
            Year component of the timestamp.
        month : int, optional
            Month component of the timestamp.
        day : int, optional
            Day component of the timestamp.
        hour : int, optional
            Hour component of the timestamp.
        minute : int, optional
            Minute component of the timestamp.
        second : int, optional
            Second component of the timestamp.
        microsecond : int, optional
            Microsecond component of the timestamp.
        tzinfo : tzinfo_type, optional
            Timezone information.
        nanosecond : int, optional
            Nanosecond component of the timestamp.
        tz : object, optional
            Timezone input object.
        unit : str, optional
            Unit of the timestamp.
        fold : int, optional
            Fold parameter.

        Notes
        -----
        This method initializes a new Timestamp object with specified components.
        """
        # 略，构造函数的实现在这里未给出完整信息

    def _round(self, freq, mode, ambiguous="raise", nonexistent="raise"):
        """
        Round the timestamp to the specified frequency.

        Parameters
        ----------
        freq : object
            Frequency to round to.
        mode : object
            Rounding mode.
        ambiguous : str, optional
            Ambiguous parameter handling.
        nonexistent : str, optional
            Nonexistent parameter handling.

        Returns
        -------
        Timestamp
            Rounded timestamp object.

        Raises
        ------
        ValueError
            If division by zero occurs in rounding.

        Notes
        -----
        This method rounds the timestamp according to the provided frequency.
        """
        cdef:
            int64_t nanos

        # 将 freq 转换为 Offset 对象
        freq = to_offset(freq, is_period=False)
        # 获取频率对应的纳秒数
        nanos = get_unit_for_round(freq, self._creso)
        
        # 如果纳秒数为零，则返回自身，不进行进一步的处理
        if nanos == 0:
            if freq.nanos == 0:
                raise ValueError("Division by zero in rounding")

            # 例如，如果 self.unit == "s" 并且是次秒级别的频率
            return self

        # 如果存在时区信息，则将值进行本地化处理
        if self.tz is not None:
            value = self.tz_localize(None)._value
        else:
            value = self._value

        # 将数值转换为 int64 类型的数组
        value = np.array([value], dtype=np.int64)

        # 尝试对数值进行舍入操作
        try:
            r = round_nsint64(value, mode, nanos)[0]
        except OverflowError as err:
            raise OutOfBoundsDatetime(
                f"Cannot round {self} to freq={freq} without overflow"
            ) from err

        # 根据舍入后的值创建新的 Timestamp 对象
        result = Timestamp._from_value_and_reso(r, self._creso, None)
        
        # 如果存在时区信息，则将结果进行本地化处理
        if self.tz is not None:
            result = result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )
        
        # 返回舍入后的结果
        return result
    def round(self, freq, ambiguous="raise", nonexistent="raise"):
        """
        Round the Timestamp to the specified resolution.

        This method rounds the given Timestamp down to a specified frequency
        level. It is particularly useful in data analysis to normalize timestamps
        to regular frequency intervals. For instance, rounding to the nearest
        minute, hour, or day can help in time series comparisons or resampling
        operations.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
                     'NaT' will return NaT for a nonexistent time.

        """
        # 返回一个新的 Timestamp 对象，按照指定频率向下取整
        # 在数据分析中，将时间戳规范化到常规频率间隔非常有用
        return self._round(freq, ambiguous=ambiguous, nonexistent=nonexistent)


    def floor(self, freq, ambiguous="raise", nonexistent="raise"):
        """
        Return a new Timestamp floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
                     'NaT' will return NaT for a nonexistent time.

        """
        # 返回一个新的 Timestamp 对象，按照指定频率向下取整
        # 在数据分析中，将时间戳规范化到常规频率间隔非常有用
        return self._round(freq, ambiguous=ambiguous, nonexistent=nonexistent)
        """
        Floor the Timestamp to the nearest lower frequency.

        Parameters
        ----------
        freq : str or offset alias
            The frequency level to floor the timestamp to.
        ambiguous : 'NaT', bool, default 'raise'
            A nonexistent time is treated as NaT or raise an error.
        nonexistent : 'raise', 'shift_forward', 'shift_backward', 'NaT', timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        See Also
        --------
        Timestamp.ceil : Round up a Timestamp to the specified resolution.
        Timestamp.round : Round a Timestamp to the specified resolution.
        Series.dt.floor : Round down the datetime values in a Series.

        Notes
        -----
        If the Timestamp has a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be floored using multiple frequency units:

        >>> ts.floor(freq='h')  # hour
        Timestamp('2020-03-14 15:00:00')

        >>> ts.floor(freq='min')  # minute
        Timestamp('2020-03-14 15:32:00')

        >>> ts.floor(freq='s')  # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.floor(freq='ns')  # nanoseconds
        Timestamp('2020-03-14 15:32:52.192548651')

        ``freq`` can also be a multiple of a single unit, like '5min' (i.e.  5 minutes):

        >>> ts.floor(freq='5min')
        Timestamp('2020-03-14 15:30:00')

        or a combination of multiple units, like '1h30min' (i.e. 1 hour and 30 minutes):

        >>> ts.floor(freq='1h30min')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.floor()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 03:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.floor("2h", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.floor("2h", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)
    def ceil(self, freq, ambiguous="raise", nonexistent="raise"):
        """
        Return a new Timestamp ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward', 'NaT', \
            The behavior for nonexistent time is defined here:
            
            * 'raise' will raise a NonExistentTimeError for a time that doesn't exist due to DST transition.
            * 'shift_forward' will shift to the closest valid time after the non-existent time.
            * 'shift_backward' will shift to the closest valid time before the non-existent time.
            * 'NaT' will return NaT for a non-existent time.
            
            """
        timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        See Also
        --------
        Timestamp.floor : Round down a Timestamp to the specified resolution.
        Timestamp.round : Round a Timestamp to the specified resolution.
        Series.dt.ceil : Ceil the datetime values in a Series.

        Notes
        -----
        If the Timestamp has a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be ceiled using multiple frequency units:

        >>> ts.ceil(freq='h')  # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.ceil(freq='min')  # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.ceil(freq='s')  # seconds
        Timestamp('2020-03-14 15:32:53')

        >>> ts.ceil(freq='us')  # microseconds
        Timestamp('2020-03-14 15:32:52.192549')

        ``freq`` can also be a multiple of a single unit, like '5min' (i.e.  5 minutes):

        >>> ts.ceil(freq='5min')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1h30min' (i.e. 1 hour and 30 minutes):

        >>> ts.ceil(freq='1h30min')
        Timestamp('2020-03-14 16:30:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.ceil()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.ceil("h", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.ceil("h", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)
    def tz(self):
        """
        Alias for tzinfo.

        The `tz` property provides a simple and direct way to retrieve the timezone
        information of a `Timestamp` object. It is particularly useful when working
        with time series data that includes timezone information, allowing for easy
        access and manipulation of the timezone context.

        See Also
        --------
        Timestamp.tzinfo : Returns the timezone information of the Timestamp.
        Timestamp.tz_convert : Convert timezone-aware Timestamp to another time zone.
        Timestamp.tz_localize : Localize the Timestamp to a timezone.

        Examples
        --------
        >>> ts = pd.Timestamp(1584226800, unit='s', tz='Europe/Stockholm')
        >>> ts.tz
        <DstTzInfo 'Europe/Stockholm' CET+1:00:00 STD>
        """
        # 返回当前时间戳的时区信息
        return self.tzinfo

    @tz.setter
    def tz(self, value):
        # GH 3746: 防止通过直接设置 tz 来本地化或转换索引
        raise AttributeError(
            "Cannot directly set timezone. "
            "Use tz_localize() or tz_convert() as appropriate"
        )

    def tz_localize(self, tz, ambiguous="raise", nonexistent="raise"):
        """
        Localize the Timestamp to a timezone.

        Convert naive Timestamp to local time zone or remove
        timezone from timezone-aware Timestamp.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding local time.

        ambiguous : bool, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
            or dateutil.relativedelta.relativedelta, default 'raise'
            If 'raise', raises a NonExistentTimeError for nonexistent times.
            If 'shift_forward' forward shifts the ambiguous time to the
            next valid local time.
            If 'shift_backward' backward shifts the ambiguous time to the
            next valid local time.
            If 'NaT' returns NaT for a nonexistent time.
            If a timedelta or relativedelta object, shift the nonexistent
            time by the amount specified.

        Returns
        -------
        Timestamp or DatetimeIndex
            Timestamp or DatetimeIndex localized to `tz`.

        See Also
        --------
        Timestamp.tz_convert : Convert Timestamp from one timezone to another.
        Timestamp.tz : Alias for `tzinfo`.

        Notes
        -----
        In rare situations where localized time may not exist due to
        daylight saving time transitions, the `nonexistent` parameter
        provides methods to handle such cases.

        Examples
        --------
        Localize a naive Timestamp to a specific timezone:
        >>> ts = pd.Timestamp('2023-03-12 01:30:00')
        >>> ts.tz_localize('Europe/Berlin')
        Timestamp('2023-03-12 01:30:00+0100', tz='Europe/Berlin')

        Handle ambiguous times by shifting forward:
        >>> ts.tz_localize('Europe/Berlin', ambiguous='shift_forward')
        Timestamp('2023-03-12 03:30:00+0200', tz='Europe/Berlin')
        """
        # 将时间戳本地化到指定时区，处理可能的歧义和不存在的时间
        pass
        """
        default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            The behavior is as follows:

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        localized : Timestamp

        Raises
        ------
        TypeError
            If the Timestamp is tz-aware and tz is not None.

        Examples
        --------
        Create a naive timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651')

        Add 'Europe/Stockholm' as timezone:

        >>> ts.tz_localize(tz='Europe/Stockholm')
        Timestamp('2020-03-14 15:32:52.192548651+0100', tz='Europe/Stockholm')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_localize()
        NaT
        """
        # 检查 ambiguous 参数是否为布尔值或者合法的字符串
        if not isinstance(ambiguous, bool) and ambiguous not in {"NaT", "raise"}:
            raise ValueError(
                        "'ambiguous' parameter must be one of: "
                        "True, False, 'NaT', 'raise' (default)"
                    )

        # 检查 nonexistent 参数是否为合法选项或者 timedelta 对象
        nonexistent_options = ("raise", "NaT", "shift_forward", "shift_backward")
        if nonexistent not in nonexistent_options and not PyDelta_Check(nonexistent):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
            )

        if self.tzinfo is None:
            # 如果时间戳是无时区信息的，进行本地化操作
            tz = maybe_get_tz(tz)
            if not isinstance(ambiguous, str):
                ambiguous = [ambiguous]
            # 使用给定的参数进行本地化到 UTC
            value = tz_localize_to_utc_single(self._value, tz,
                                              ambiguous=ambiguous,
                                              nonexistent=nonexistent,
                                              creso=self._creso)
        elif tz is None:
            # 如果时间戳有时区信息但传入的 tz 为 None，则重置时区信息
            value = tz_convert_from_utc_single(self._value, self.tz, creso=self._creso)

        else:
            # 如果时间戳有时区信息且传入的 tz 不为 None，则抛出类型错误
            raise TypeError(
                "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
            )

        # 根据处理后的值和分辨率创建新的 Timestamp 对象
        out = type(self)._from_value_and_reso(value, self._creso, tz=tz)
        return out
    def tz_convert(self, tz):
        """
        Convert timezone-aware Timestamp to another time zone.

        This method is used to convert a timezone-aware Timestamp object to a
        different time zone. The original UTC time remains the same; only the
        time zone information is changed. If the Timestamp is timezone-naive, a
        TypeError is raised.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        See Also
        --------
        Timestamp.tz_localize : Localize the Timestamp to a timezone.
        DatetimeIndex.tz_convert : Convert a DatetimeIndex to another time zone.
        DatetimeIndex.tz_localize : Localize a DatetimeIndex to a specific time zone.
        datetime.datetime.astimezone : Convert a datetime object to another time zone.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
        if self.tzinfo is None:
            # If the Timestamp is timezone-naive, raise an error
            raise TypeError(
                "Cannot convert tz-naive Timestamp, use tz_localize to localize"
            )
        else:
            # Create a new Timestamp object with the same UTC time but in the specified timezone
            tz = maybe_get_tz(tz)  # Retrieve the timezone object or None
            out = type(self)._from_value_and_reso(self._value, reso=self._creso, tz=tz)
            return out

    astimezone = tz_convert

    def replace(
        self,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        nanosecond=None,
        tzinfo=object,
        fold=None,
    ):
        """
        Construct and return a new Timestamp object with the same attributes
        except for those attributes given new values by whichever keyword
        arguments are specified.

        This function is used to create a new Timestamp object with modified
        attributes. If no attributes are specified, the new Timestamp is a
        shallow copy of this instance. If any attributes are specified, they
        override the corresponding attributes of the new Timestamp object.

        Parameters
        ----------
        year, month, day, hour, minute, second, microsecond, nanosecond : int, optional
            New values for the Timestamp's attributes. If not provided, default
            to the corresponding attribute from the original Timestamp.
        tzinfo : datetime.tzinfo, optional, default object
            Time zone for the Timestamp. If not provided, defaults to the
            original Timestamp's time zone.
        fold : {0, 1}, optional
            This parameter is currently not used by pandas Timestamp.

        Returns
        -------
        Timestamp
            A new Timestamp object with updated attributes.

        See Also
        --------
        datetime.replace : Equivalent datetime function for Python's datetime objects.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='US/Pacific')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Replace the year:

        >>> ts.replace(year=2021)
        Timestamp('2021-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Replace the time zone:

        >>> ts.replace(tzinfo='UTC')
        Timestamp('2020-03-14 22:32:52.192548651+0000', tz='UTC')
        """
    def to_julian_date(self) -> np.float64:
        """
        Convert TimeStamp to a Julian Date.

        0 Julian date is noon January 1, 4713 BC.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52')
        >>> ts.to_julian_date()
        2458923.147824074
        """
        # 提取时间戳的年、月、日信息
        year = self.year
        month = self.month
        day = self.day
        # 如果月份小于等于2，则修正年份和月份
        if month <= 2:
            year -= 1
            month += 12
        # 计算并返回儒略日期
        return (day +
                np.fix((153 * month - 457) / 5) +
                365 * year +
                np.floor(year / 4) -
                np.floor(year / 100) +
                np.floor(year / 400) +
                1721118.5 +
                (self.hour +
                 self.minute / 60.0 +
                 self.second / 3600.0 +
                 self.microsecond / 3600.0 / 1e+6 +
                 self.nanosecond / 3600.0 / 1e+9
                 ) / 24.0)

    def isoweekday(self):
        """
        Return the day of the week represented by the date.

        Monday == 1 ... Sunday == 7.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isoweekday()
        7
        """
        # 返回ISO格式的星期几，与Python默认的weekday方法相比，基于1到7的星期表示
        # 调用了父类的weekday()方法，并将结果加1以调整到ISO星期表示
        return self.weekday() + 1

    def weekday(self):
        """
        Return the day of the week represented by the date.

        Monday == 0 ... Sunday == 6.

        See Also
        --------
        Timestamp.dayofweek : Return the day of the week with Monday=0, Sunday=6.
        Timestamp.isoweekday : Return the day of the week with Monday=1, Sunday=7.
        datetime.date.weekday : Equivalent method in datetime module.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01')
        >>> ts
        Timestamp('2023-01-01  00:00:00')
        >>> ts.weekday()
        6
        """
        # 返回基于Python默认格式的星期几，使用ccalendar.dayofweek方法计算
        # 调用了ccalendar.dayofweek方法，返回0到6的星期表示
        return ccalendar.dayofweek(self.year, self.month, self.day)
# Aliases
# 在 Timestamp 类中添加两个别名函数，使其功能类似于 Timestamp.week 和 Timestamp.days_in_month

Timestamp.weekofyear = Timestamp.week
Timestamp.daysinmonth = Timestamp.days_in_month


# ----------------------------------------------------------------------
# Scalar analogues to functions in vectorized.pyx

# 使用装饰器指定不允许使用 Cython 的除法
@cython.cdivision(False)
# 声明一个 C 函数，用于将本地化的纳秒级时间戳向下舍入到前一天的午夜时间点

cdef int64_t normalize_i8_stamp(int64_t local_val, int64_t ppd) noexcept nogil:
    """
    Round the localized nanosecond timestamp down to the previous midnight.

    Parameters
    ----------
    local_val : int64_t
        本地化的纳秒级时间戳
    ppd : int64_t
        时间戳分辨率下的每日周期数

    Returns
    -------
    int64_t
        调整后的时间戳
    """
    return local_val - (local_val % ppd)
```