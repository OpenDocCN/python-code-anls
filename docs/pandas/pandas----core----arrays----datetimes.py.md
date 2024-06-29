# `D:\src\scipysrc\pandas\pandas\core\arrays\datetimes.py`

```
# 从未来模块中导入注解支持
from __future__ import annotations

# 导入 datetime 相关模块
from datetime import (
    datetime,     # 日期时间类
    timedelta,    # 时间间隔类
    tzinfo,       # 时区信息基类
)

# 导入类型相关模块
from typing import (
    TYPE_CHECKING,  # 类型检查标志
    TypeVar,        # 泛型类型变量
    cast,           # 类型转换函数
    overload,       # 函数重载装饰器
)
import warnings  # 警告模块

import numpy as np  # 导入 NumPy 库

# 从 pandas 库中导入配置相关模块和函数
from pandas._config.config import get_option

# 从 pandas 库的内部模块中导入 C 扩展库和时间序列相关模块
from pandas._libs import (
    lib,    # C 扩展库
    tslib,  # 时间序列基础库
)

# 从 pandas 库的时间序列基础库中导入日期时间偏移类和一些相关的辅助函数和类
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,             # Not-a-Time 标识
    NaTType,         # NaT 类型
    Resolution,      # 时间分辨率枚举
    Timestamp,       # 时间戳类
    astype_overflowsafe,
    fields,
    get_resolution,
    get_supported_dtype,
    get_unit_from_dtype,
    ints_to_pydatetime,
    is_date_array_normalized,
    is_supported_dtype,
    is_unitless,
    normalize_i8_timestamps,
    timezones,
    to_offset,
    tz_convert_from_utc,
    tzconversion,
)

# 从 pandas 库的时间序列数据类型模块中导入常用类型映射函数
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit

# 从 pandas 库的错误模块中导入性能警告类
from pandas.errors import PerformanceWarning

# 从 pandas 库的实用工具异常模块中导入栈查找级别函数
from pandas.util._exceptions import find_stack_level

# 从 pandas 库的实用工具验证器模块中导入验证函数
from pandas.util._validators import validate_inclusive

# 从 pandas 核心数据类型常见模块中导入常用数据类型和相关函数
from pandas.core.dtypes.common import (
    DT64NS_DTYPE,    # datetime64[ns] 数据类型
    INT64_DTYPE,     # int64 数据类型
    is_bool_dtype,   # 是否布尔类型函数
    is_float_dtype,  # 是否浮点类型函数
    is_string_dtype, # 是否字符串类型函数
    pandas_dtype,    # 转换为 pandas 数据类型函数
)

# 从 pandas 核心数据类型模块中导入日期时间时区数据类型和扩展数据类型
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,   # 带时区的 datetime 类型
    ExtensionDtype,    # 扩展数据类型基类
    PeriodDtype,       # 时期数据类型
)

# 从 pandas 核心数据类型缺失模块中导入缺失值判断函数
from pandas.core.dtypes.missing import isna

# 从 pandas 核心数组模块中导入日期时间类数组和相关的日期时间操作模块
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range

# 导入 pandas 核心公共模块
import pandas.core.common as com

# 从 pandas 时间序列频率模块中导入周期别名获取函数
from pandas.tseries.frequencies import get_period_alias

# 从 pandas 时间序列偏移模块中导入日期和 Tick 偏移类
from pandas.tseries.offsets import (
    Day,   # 天偏移类
    Tick,  # Tick 偏移类
)

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 collections.abc 模块中导入生成器和迭代器抽象基类
    from collections.abc import (
        Generator,  # 生成器类型
        Iterator,   # 迭代器类型
    )

    # 从 pandas 的类型注解模块中导入数组样式、日期时间错误选择、数据类型对象等
    from pandas._typing import (
        ArrayLike,            # 数组样式类型
        DateTimeErrorChoices, # 日期时间错误选择
        DtypeObj,             # 数据类型对象
        IntervalClosedType,   # 区间闭合类型
        Self,                 # Self 类型变量
        TimeAmbiguous,        # 时间模糊类型
        TimeNonexistent,      # 时间不存在类型
        npt,                  # NumPy 类型变量
    )

    # 从 pandas 核心模块中导入 DataFrame 和 Timedelta 类
    from pandas import (
        DataFrame,   # 数据帧类
        Timedelta,   # 时间间隔类
    )
    # 从 pandas 核心数组模块中导入周期数组类
    from pandas.core.arrays import PeriodArray

    # 定义两个特定的时间戳类型变量，支持时间戳和 None
    _TimestampNoneT1 = TypeVar("_TimestampNoneT1", Timestamp, None)
    _TimestampNoneT2 = TypeVar("_TimestampNoneT2", Timestamp, None)


# 定义迭代块大小常量
_ITER_CHUNKSIZE = 10_000


# 函数重载：根据时区信息返回相应的 datetime64[ns] 数据类型或 DatetimeTZDtype 对象
@overload
def tz_to_dtype(tz: tzinfo, unit: str = ...) -> DatetimeTZDtype: ...


# 函数重载：如果时区信息为 None，则返回默认的 np.datetime64 数据类型
@overload
def tz_to_dtype(tz: None, unit: str = ...) -> np.dtype[np.datetime64]: ...


def tz_to_dtype(
    tz: tzinfo | None, unit: str = "ns"
) -> np.dtype[np.datetime64] | DatetimeTZDtype:
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None
        时区信息或 None
    unit : str, default "ns"
        时间单位，默认为纳秒

    Returns
    -------
    np.dtype or Datetime64TZDType
        返回对应的 NumPy 数据类型或 Datetime64TZDType 对象
    """
    if tz is None:
        return np.dtype(f"M8[{unit}]")  # 返回 np.datetime64 数据类型
    else:
        return DatetimeTZDtype(tz=tz, unit=unit)  # 返回带时区的 datetime 类型


# 定义字段访问器函数，用于访问对象的特定字段
def _field_accessor(name: str, field: str, docstring: str | None = None):
    # 定义一个名为 f 的方法，该方法是一个属性方法（property）
    def f(self):
        # 调用 self._local_timestamps() 方法获取时间戳数据
        values = self._local_timestamps()

        # 检查字段是否在布尔运算集合中
        if field in self._bool_ops:
            # 声明一个类型为 np.ndarray 的变量 result

            # 如果字段以 "start" 或 "end" 结尾
            if field.endswith(("start", "end")):
                # 获取频率信息
                freq = self.freq
                month_kw = 12
                # 如果存在频率信息
                if freq:
                    kwds = freq.kwds
                    # 从频率信息中获取起始月份，如果未指定，默认为 12
                    month_kw = kwds.get("startingMonth", kwds.get("month", month_kw))

                # 如果频率信息不为 None，则获取频率名称
                if freq is not None:
                    freq_name = freq.name
                else:
                    freq_name = None

                # 调用 fields 模块的 get_start_end_field 方法获取开始/结束字段的结果
                result = fields.get_start_end_field(
                    values, field, freq_name, month_kw, reso=self._creso
                )
            else:
                # 否则，调用 fields 模块的 get_date_field 方法获取日期字段的结果
                result = fields.get_date_field(values, field, reso=self._creso)

            # 返回布尔类型的结果
            return result

        # 如果字段在对象操作集合中
        if field in self._object_ops:
            # 调用 fields 模块的 get_date_name_field 方法获取日期名称字段的结果
            result = fields.get_date_name_field(values, field, reso=self._creso)
            # 对结果进行可能的遮蔽操作，使用 None 作为填充值
            result = self._maybe_mask_results(result, fill_value=None)

        else:
            # 否则，调用 fields 模块的 get_date_field 方法获取日期字段的结果
            result = fields.get_date_field(values, field, reso=self._creso)
            # 对结果进行可能的遮蔽操作，使用 None 作为填充值，将结果转换为 float64 类型
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        # 返回处理后的结果
        return result

    # 将方法名设置为 name
    f.__name__ = name
    # 将方法的文档字符串设置为 docstring
    f.__doc__ = docstring
    # 返回 f 方法作为属性
    return property(f)
# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):  # type: ignore[misc]
    """
    Pandas ExtensionArray for tz-naive or tz-aware datetime data.

    .. warning::

       DatetimeArray is currently experimental, and its API may change
       without warning. In particular, :attr:`DatetimeArray.dtype` is
       expected to change to always be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    data : Series, Index, DatetimeArray, ndarray
        The datetime data.

        For DatetimeArray `values` (or a Series or Index boxing one),
        `dtype` and `freq` will be extracted from `values`.

    dtype : numpy.dtype or DatetimeTZDtype
        Note that the only NumPy dtype allowed is 'datetime64[ns]'.
    freq : str or Offset, optional
        The frequency.
    copy : bool, default False
        Whether to copy the underlying array of values.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.DatetimeArray._from_sequence(
    ...     pd.DatetimeIndex(["2023-01-01", "2023-01-02"], freq="D")
    ... )
    <DatetimeArray>
    ['2023-01-01 00:00:00', '2023-01-02 00:00:00']
    Length: 2, dtype: datetime64[s]
    """

    _typ = "datetimearray"  # 定义 DatetimeArray 的类型字符串
    _internal_fill_value = np.datetime64("NaT", "ns")  # 内部填充值为 NaT (Not a Time) 的 numpy.datetime64 类型
    _recognized_scalars = (datetime, np.datetime64)  # 识别的标量类型为 datetime 和 numpy.datetime64
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, "M") or isinstance(
        x, DatetimeTZDtype
    )  # 判断是否为识别的 dtype，支持 numpy.datetime64 或 DatetimeTZDtype 类型

    _infer_matches = ("datetime", "datetime64", "date")  # 推断匹配的数据类型包括 datetime、datetime64 和 date

    @property
    def _scalar_type(self) -> type[Timestamp]:
        return Timestamp  # 返回此 DatetimeArray 实例中标量的类型为 Timestamp

    # define my properties & methods for delegation
    _bool_ops: list[str] = [  # 布尔操作列表，包括判断月初、月末、季初、季末、年初、年末、闰年等
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
    ]
    _object_ops: list[str] = ["freq", "tz"]  # 对象操作列表，包括频率和时区
    _field_ops: list[str] = [  # 字段操作列表，包括年、月、日、小时、分钟、秒、星期、年的第几天、季度等
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "days_in_month",
        "daysinmonth",
        "microsecond",
        "nanosecond",
    ]
    _other_ops: list[str] = ["date", "time", "timetz"]  # 其他操作列表，包括日期、时间、带时区的时间
    _datetimelike_ops: list[str] = (  # 日期时间操作列表，包括所有字段操作、对象操作、布尔操作、其他操作以及单位操作
        _field_ops + _object_ops + _bool_ops + _other_ops + ["unit"]
    )
    _datetimelike_methods: list[str] = [  # 日期时间方法列表，包括转换为周期、本地化时区、转换时区、规范化、格式化、舍入等
        "to_period",
        "tz_localize",
        "tz_convert",
        "normalize",
        "strftime",
        "round",
        "floor",
        "ceil",
        "month_name",
        "day_name",
        "as_unit",
    ]

    # ndim is inherited from ExtensionArray, must exist to ensure
    #  Timestamp.__richcmp__(DateTimeArray) operates pointwise
    # 设置数组优先级，确保 numpy 数组的操作优先使用我们的实现
    __array_priority__ = 1000

    # -----------------------------------------------------------------
    # 构造函数

    # _dtype 表示数据类型为 np.datetime64 或 DatetimeTZDtype
    _dtype: np.dtype[np.datetime64] | DatetimeTZDtype
    # _freq 表示频率，可以是 BaseOffset 类型或 None
    _freq: BaseOffset | None = None

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        # 检查 scalars 是否为 datetime 或 datetime64 类型
        if lib.infer_dtype(scalars, skipna=True) not in ["datetime", "datetime64"]:
            # 如果不是日期时间类型，则引发值错误
            raise ValueError
        # 根据传入的数据创建对象
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def _validate_dtype(cls, values, dtype):
        # 在 TimeLikeOps.__init__ 中使用
        # 验证日期时间数据类型，确保与指定的 dtype 兼容
        dtype = _validate_dt64_dtype(dtype)
        _validate_dt64_dtype(values.dtype)
        if isinstance(dtype, np.dtype):
            if values.dtype != dtype:
                raise ValueError("Values resolution does not match dtype.")
        else:
            vunit = np.datetime_data(values.dtype)[0]
            if vunit != dtype.unit:
                raise ValueError("Values resolution does not match dtype.")
        return dtype

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.datetime64],
        freq: BaseOffset | None = None,
        dtype: np.dtype[np.datetime64] | DatetimeTZDtype = DT64NS_DTYPE,
    ) -> Self:
        # 确保 values 是 np.ndarray 类型
        assert isinstance(values, np.ndarray)
        # 确保 dtype 是日期时间类型
        assert dtype.kind == "M"
        if isinstance(dtype, np.dtype):
            # 如果 dtype 是 np.dtype，则确保与 values 的 dtype 相同，并且不是无单位的
            assert dtype == values.dtype
            assert not is_unitless(dtype)
        else:
            # 如果是 DatetimeTZDtype，则确保与 values 的 dtype 的单位兼容
            assert dtype._creso == get_unit_from_dtype(values.dtype)

        # 调用父类的 _simple_new 方法创建新对象
        result = super()._simple_new(values, dtype)
        result._freq = freq  # 设置频率
        return result

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:
        # 从序列数据创建对象，不严格要求数据类型
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        *,
        dtype=None,
        copy: bool = False,
        tz=lib.no_default,
        freq: str | BaseOffset | lib.NoDefault | None = lib.no_default,
        dayfirst: bool = False,
        yearfirst: bool = False,
        ambiguous: TimeAmbiguous = "raise",
        # 以下省略部分参数注释
    def _from_sequence(
        self,
        data,
        dtype=None,
        copy=False,
        tz=None,
        dayfirst=False,
        yearfirst=False,
        ambiguous="raise",
        infer_datetime_format=False,
    ) -> Self:
        """
        A non-strict version of _from_sequence, called from DatetimeIndex.__new__.
        """

        # 如果用户显式传递了 tz=None 或者是一个 tz-naive 的 dtype，则禁止推断时区。
        explicit_tz_none = tz is None
        if tz is lib.no_default:
            tz = None
        else:
            tz = timezones.maybe_get_tz(tz)

        # 验证并规范化 dtype
        dtype = _validate_dt64_dtype(dtype)
        
        # 如果 dtype 包含时区信息，则从 dtype 中获取时区信息
        tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)

        # 根据 dtype 转换为单位信息
        unit = None
        if dtype is not None:
            unit = dtl.dtype_to_unit(dtype)

        # 确保 data 可以转换为数组，并返回转换后的数组和是否复制的标志
        data, copy = dtl.ensure_arraylike_for_datetimelike(
            data, copy, cls_name="DatetimeArray"
        )

        inferred_freq = None
        # 如果 data 是 DatetimeArray 类型，则推断其频率
        if isinstance(data, DatetimeArray):
            inferred_freq = data.freq

        # 将数据序列转换为 datetime64 数组
        subarr, tz = _sequence_to_dt64(
            data,
            copy=copy,
            tz=tz,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
            out_unit=unit,
        )

        # 再次调用以确保在可能推断时区后，时区信息的有效性
        _validate_tz_from_dtype(dtype, tz, explicit_tz_none)

        # 如果 tz 不为 None 且用户显式指定了 tz=None，则抛出 ValueError
        if tz is not None and explicit_tz_none:
            raise ValueError(
                "Passed data is timezone-aware, incompatible with 'tz=None'. "
                "Use obj.tz_localize(None) instead."
            )

        # 获取 subarr 的数据单位
        data_unit = np.datetime_data(subarr.dtype)[0]

        # 根据时区信息 tz 和数据单位 data_unit 创建对应的 dtype
        data_dtype = tz_to_dtype(tz, data_unit)

        # 使用 _simple_new 方法创建新的实例 result
        result = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)

        # 如果用户指定了 unit 并且不等于 result 的 unit，则进行单位转换
        if unit is not None and unit != result.unit:
            result = result.as_unit(unit)

        # 验证关键字参数，设置频率
        validate_kwds = {"ambiguous": ambiguous}
        result._maybe_pin_freq(freq, validate_kwds)

        # 返回最终结果 result
        return result
    def _check_compatible_with(self, other) -> None:
        # 如果 other 是 NaT（Not a Time），则直接返回，不执行后续操作
        if other is NaT:
            return
        # 断言当前对象与 other 兼容的时区感知性
        self._assert_tzawareness_compat(other)

    # -----------------------------------------------------------------
    # Descriptive Properties

    def _box_func(self, x: np.datetime64) -> Timestamp | NaTType:
        # GH#42228
        # 将 np.datetime64 转换为 int64 类型的视图
        value = x.view("i8")
        # 使用 Timestamp 的 _from_value_and_reso 方法创建时间戳对象
        ts = Timestamp._from_value_and_reso(value, reso=self._creso, tz=self.tz)
        # 返回时间戳对象 ts
        return ts

    @property
    # error: Return type "Union[dtype, DatetimeTZDtype]" of "dtype"
    # incompatible with return type "ExtensionDtype" in supertype
    # "ExtensionArray"
    def dtype(self) -> np.dtype[np.datetime64] | DatetimeTZDtype:  # type: ignore[override]
        """
        The dtype for the DatetimeArray.

        .. warning::

           A future version of pandas will change dtype to never be a
           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will
           always be an instance of an ``ExtensionDtype`` subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``
            is returned.

            If the values are tz-aware, then the ``DatetimeTZDtype``
            is returned.
        """
        # 返回当前对象的 _dtype 属性，即日期时间数组的数据类型
        return self._dtype

    @property
    def tz(self) -> tzinfo | None:
        """
        Return the timezone.

        Returns
        -------
        zoneinfo.ZoneInfo,, datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None
            Returns None when the array is tz-naive.

        See Also
        --------
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.
        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from
            one time zone to another.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.tz
        datetime.timezone.utc

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(
        ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
        ... )
        >>> idx.tz
        datetime.timezone.utc
        """  # noqa: E501
        # GH 18595
        # 返回当前对象的 dtype 属性的 tz 属性，即时区信息
        return getattr(self.dtype, "tz", None)

    @tz.setter
    def tz(self, value):
        # GH 3746: Prevent localizing or converting the index by setting tz
        # 抛出错误，禁止直接设置时区信息
        raise AttributeError(
            "Cannot directly set timezone. Use tz_localize() "
            "or tz_convert() as appropriate"
        )

    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
        # 返回 tz 属性的别名 tzinfo
        return self.tz

    @property  # NB: override with cache_readonly in immutable subclasses
    # 检查时间序列是否标准化，即所有日期是否都是午夜时间（无具体时分秒）
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        return is_date_array_normalized(self.asi8, self.tz, reso=self._creso)

    @property  # NB: override with cache_readonly in immutable subclasses
    # 返回分辨率对象，根据时间序列数据和时区信息确定分辨率
    def _resolution_obj(self) -> Resolution:
        return get_resolution(self.asi8, self.tz, reso=self._creso)

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if dtype is None and self.tz:
            # 如果未指定数据类型且有时区信息，则默认使用对象类型以保留时区信息
            dtype = object

        return super().__array__(dtype=dtype, copy=copy)

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the boxed values

        Yields
        ------
        tstamp : Timestamp
            时间戳对象的迭代器
        """
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # 为了效率，以1万条数据为一组进行迭代转换
            data = self.asi8
            length = len(self)
            chunksize = _ITER_CHUNKSIZE
            chunks = (length // chunksize) + 1

            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                # 将整数数组转换为 Python datetime 对象列表
                converted = ints_to_pydatetime(
                    data[start_i:end_i],
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                yield from converted
    def astype(self, dtype, copy: bool = True):
        # 处理数据类型转换的方法，接受目标数据类型和是否复制数据的标志

        # 将目标数据类型转换为 Pandas 数据类型
        dtype = pandas_dtype(dtype)

        # 如果目标数据类型和当前数据类型相同
        if dtype == self.dtype:
            # 如果需要复制数据，则返回数据的副本，否则返回自身引用
            if copy:
                return self.copy()
            return self

        # 如果目标数据类型是扩展数据类型
        elif isinstance(dtype, ExtensionDtype):
            # 如果不是 DatetimeTZDtype 类型
            if not isinstance(dtype, DatetimeTZDtype):
                # 返回父类方法处理数据类型转换
                return super().astype(dtype, copy=copy)
            # 如果数据是时区感知类型但是没有时区信息
            elif self.tz is None:
                # 抛出类型错误，提示不能使用 .astype 从时区无关的数据类型转换为时区相关的数据类型
                raise TypeError(
                    "Cannot use .astype to convert from timezone-naive dtype to "
                    "timezone-aware dtype. Use obj.tz_localize instead or "
                    "series.dt.tz_localize instead"
                )
            else:
                # 时区感知的单位转换，例如 datetime64[s, UTC]
                np_dtype = np.dtype(dtype.str)
                # 调用安全转换方法进行数据类型转换
                res_values = astype_overflowsafe(self._ndarray, np_dtype, copy=copy)
                # 返回一个新的对象实例，保留数据类型和频率信息
                return type(self)._simple_new(res_values, dtype=dtype, freq=self.freq)

        # 如果数据是时区无关的，并且目标数据类型是 NumPy 的日期时间类型
        elif (
            self.tz is None
            and lib.is_np_dtype(dtype, "M")
            and not is_unitless(dtype)
            and is_supported_dtype(dtype)
        ):
            # 单位转换，例如 datetime64[s]
            res_values = astype_overflowsafe(self._ndarray, dtype, copy=True)
            # 返回一个新的对象实例，保留新的数据类型
            return type(self)._simple_new(res_values, dtype=res_values.dtype)
            # TODO: 是否保留频率信息？

        # 如果数据是时区相关的，并且目标数据类型是 NumPy 的日期时间类型
        elif self.tz is not None and lib.is_np_dtype(dtype, "M"):
            # 抛出类型错误，提示不能使用 .astype 从时区相关的数据类型转换为时区无关的数据类型
            raise TypeError(
                "Cannot use .astype to convert from timezone-aware dtype to "
                "timezone-naive dtype. Use obj.tz_localize(None) or "
                "obj.tz_convert('UTC').tz_localize(None) instead."
            )

        # 如果数据是时区无关的，并且目标数据类型是 NumPy 的日期时间类型，且不是当前数据类型
        elif (
            self.tz is None
            and lib.is_np_dtype(dtype, "M")
            and dtype != self.dtype
            and is_unitless(dtype)
        ):
            # 抛出类型错误，提示不支持转换为无单位的日期时间类型 'datetime64'
            raise TypeError(
                "Casting to unit-less dtype 'datetime64' is not supported. "
                "Pass e.g. 'datetime64[ns]' instead."
            )

        # 如果目标数据类型是周期类型
        elif isinstance(dtype, PeriodDtype):
            # 调用 to_period 方法将数据转换为周期类型
            return self.to_period(freq=dtype.freq)
        
        # 调用 DatetimeLikeArrayMixin 类的 astype 方法处理剩余的情况
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)

    # -----------------------------------------------------------------
    # 渲染方法

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        # 如果日期格式为 None 且仅包含日期部分（无时区信息），提供一个默认格式
        if date_format is None and self._is_dates_only:
            date_format = "%Y-%m-%d"

        # 使用 tslib.format_array_from_datetime 格式化日期时间数组为字符串数组
        return tslib.format_array_from_datetime(
            self.asi8, tz=self.tz, format=date_format, na_rep=na_rep, reso=self._creso
        )

    # -----------------------------------------------------------------
    # Comparison Methods

    def _assert_tzawareness_compat(self, other) -> None:
        # 从 _Timestamp._assert_tzawareness_compat 改编而来
        other_tz = getattr(other, "tzinfo", None)
        other_dtype = getattr(other, "dtype", None)

        # 如果 other 的 dtype 是 DatetimeTZDtype，则获取其 tzinfo
        if isinstance(other_dtype, DatetimeTZDtype):
            other_tz = other.dtype.tz
        # 如果 other 是 NaT（Not a Time），则跳过比较
        if other is NaT:
            pass
        # 如果 self 是 tz-naive 而 other 是 tz-aware，则引发 TypeError
        elif self.tz is None:
            if other_tz is not None:
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware datetime-like objects."
                )
        # 如果 other 是 tz-naive 而 self 是 tz-aware，则引发 TypeError
        elif other_tz is None:
            raise TypeError(
                "Cannot compare tz-naive and tz-aware datetime-like objects"
            )

    # -----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, offset: BaseOffset) -> Self:
        # 确保 offset 不是 Tick 的实例
        assert not isinstance(offset, Tick)

        # 如果 self 是 tz-aware，则将其本地化为无时区
        if self.tz is not None:
            values = self.tz_localize(None)
        else:
            values = self

        try:
            # 尝试将偏移应用于 values 的数组形式
            res_values = offset._apply_array(values._ndarray)
            # 如果 res_values 的 dtype 是整数类型，则尝试将其视为 self 的 dtype
            if res_values.dtype.kind == "i":
                res_values = res_values.view(values.dtype)  # type: ignore[arg-type]
        except NotImplementedError:
            # 如果无法向量化地应用 DateOffset 到 Series 或 DatetimeIndex
            if get_option("performance_warnings"):
                # 发出性能警告
                warnings.warn(
                    "Non-vectorized DateOffset being applied to Series or "
                    "DatetimeIndex.",
                    PerformanceWarning,
                    stacklevel=find_stack_level(),
                )
            # 将 self 转换为对象数组并添加偏移量
            res_values = self.astype("O") + offset
            # TODO(GH#55564): as_unit 将不再必要
            # 将结果转换为与 self 类型相同的序列并作为 self 的单位返回
            result = type(self)._from_sequence(res_values).as_unit(self.unit)
            # 如果 self 长度为零，则 _from_sequence 无法推断 self.tz，需要本地化 self.tz
            if not len(self):
                return result.tz_localize(self.tz)

        else:
            # 如果成功应用了偏移量，则创建一个新的对象结果
            result = type(self)._simple_new(res_values, dtype=res_values.dtype)
            # 如果偏移量需要标准化，则标准化结果
            if offset.normalize:
                result = result.normalize()
                result._freq = None

            # 如果 self 是 tz-aware，则将结果本地化为 self.tz
            if self.tz is not None:
                result = result.tz_localize(self.tz)

        return result
    # -----------------------------------------------------------------
    # Timezone Conversion and Localization Methods

    # 定义一个方法，返回一个 numpy 数组（int64 类型），用来表示本地时区的时间戳
    def _local_timestamps(self) -> npt.NDArray[np.int64]:
        """
        Convert to an i8 (unix-like nanosecond timestamp) representation
        while keeping the local timezone and not using UTC.
        This is used to calculate time-of-day information as if the timestamps
        were timezone-naive.
        """
        # 如果时区为空或者是 UTC 时区，则直接返回原始的 i8 数组，避免进行时区转换时的复制操作
        if self.tz is None or timezones.is_utc(self.tz):
            return self.asi8
        # 否则，使用指定时区将 UTC 时间戳数组转换为本地时区的时间戳数组，并指定时间分辨率为 _creso
        return tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)
    def tz_convert(self, tz) -> Self:
        """
        Convert tz-aware Datetime Array/Index from one time zone to another.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index. A `tz` of None will
            convert to UTC and remove the timezone information.

        Returns
        -------
        Array or Index
            Datetme Array/Index with target `tz`.

        Raises
        ------
        TypeError
            If Datetime Array/Index is tz-naive.

        See Also
        --------
        DatetimeIndex.tz : A timezone that has a variable offset from UTC.
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.

        Examples
        --------
        With the `tz` parameter, we can change the DatetimeIndex
        to other time zones:

        >>> dti = pd.date_range(
        ...     start="2014-08-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
        ... )

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert("US/Central")
        DatetimeIndex(['2014-08-01 02:00:00-05:00',
                       '2014-08-01 03:00:00-05:00',
                       '2014-08-01 04:00:00-05:00'],
                      dtype='datetime64[ns, US/Central]', freq='h')

        With the ``tz=None``, we can remove the timezone (after converting
        to UTC if necessary):

        >>> dti = pd.date_range(
        ...     start="2014-08-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
        ... )

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                        dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert(None)
        DatetimeIndex(['2014-08-01 07:00:00',
                       '2014-08-01 08:00:00',
                       '2014-08-01 09:00:00'],
                        dtype='datetime64[ns]', freq='h')
        """  # noqa: E501
        # 通过 timezones.maybe_get_tz 函数获取有效的时区对象
        tz = timezones.maybe_get_tz(tz)

        # 如果当前 Datetime Array/Index 是 tz-naive，则抛出 TypeError
        if self.tz is None:
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )

        # 因为时间戳始终是 UTC，所以无需进行时区转换，直接使用目标时区创建新的数组或索引
        dtype = tz_to_dtype(tz, unit=self.unit)
        return self._simple_new(self._ndarray, dtype=dtype, freq=self.freq)

    @dtl.ravel_compat
    def tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        """
        Localize tz-naive Datetime Array/Index to tz-aware Datetime Array/Index.

        This method takes a time zone (tz) naive Datetime Array/Index object
        and makes this time zone aware. It does not move the time to another
        time zone.

        This method can also be used to do the inverse -- to create a time
        zone unaware object from an aware object. To that end, pass `tz=None`.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo,, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone to convert timestamps to. Passing ``None`` will
            remove the time zone information preserving local time.
        ambiguous : 'infer', 'NaT', bool array, default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False signifies a
              non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta,
            NaT does not exist, please specify what the time shift should
            be

        Returns
        -------
        Self
            A new instance with tz information.

        Notes
        -----
        This method does not move the time to another time zone.

        See Also
        --------
        DatetimeArray.tz_convert : Convert time zone using fixed offset or
            time zone string.

        """
        # Return instance with tz information
        return self._tz_convert(tz, ambiguous, nonexistent, infer=False)

    def to_pydatetime(self) -> npt.NDArray[np.object_]:
        """
        Return an ndarray of ``datetime.datetime`` objects.

        Returns
        -------
        numpy.ndarray
            An ndarray of ``datetime.datetime`` objects.

        See Also
        --------
        DatetimeIndex.to_julian_date : Converts Datetime Array to float64 ndarray
            of Julian Dates.

        Examples
        --------
        >>> idx = pd.date_range("2018-02-27", periods=3)
        >>> idx.to_pydatetime()
        array([datetime.datetime(2018, 2, 27, 0, 0),
               datetime.datetime(2018, 2, 28, 0, 0),
               datetime.datetime(2018, 3, 1, 0, 0)], dtype=object)
        """
        # Convert internal representation to Python datetime objects
        return ints_to_pydatetime(self.asi8, tz=self.tz, reso=self._creso)
    def normalize(self) -> Self:
        """
        Convert times to midnight.

        The time component of the date-time is converted to midnight i.e.
        00:00:00. This is useful in cases, when the time does not matter.
        Length is unaltered. The timezones are unaffected.

        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on Datetime Array/Index.

        Returns
        -------
        DatetimeArray, DatetimeIndex or Series
            The same type as the original data. Series will have the same
            name and index. DatetimeIndex will have the same name.

        See Also
        --------
        floor : Floor the datetimes to the specified freq.
        ceil : Ceil the datetimes to the specified freq.
        round : Round the datetimes to the specified freq.

        Examples
        --------
        >>> idx = pd.date_range(
        ...     start="2014-08-01 10:00", freq="h", periods=3, tz="Asia/Calcutta"
        ... )
        >>> idx
        DatetimeIndex(['2014-08-01 10:00:00+05:30',
                       '2014-08-01 11:00:00+05:30',
                       '2014-08-01 12:00:00+05:30'],
                       dtype='datetime64[ns, Asia/Calcutta]', freq='h')
        >>> idx.normalize()
        DatetimeIndex(['2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30'],
                       dtype='datetime64[ns, Asia/Calcutta]', freq=None)
        """
        # 调用 normalize_i8_timestamps 函数，将日期时间的时间部分转换为午夜时分（00:00:00）
        new_values = normalize_i8_timestamps(self.asi8, self.tz, reso=self._creso)
        
        # 将转换后的时间戳数组视为对应的 numpy 数据类型
        dt64_values = new_values.view(self._ndarray.dtype)

        # 使用当前对象的类型创建一个新的 DatetimeArray/Index 或 Series 对象
        dta = type(self)._simple_new(dt64_values, dtype=dt64_values.dtype)
        
        # 尝试推断频率并设置到新对象中
        dta = dta._with_freq("infer")
        
        # 如果原始对象有时区信息，则将时区信息重新应用到新对象中
        if self.tz is not None:
            dta = dta.tz_localize(self.tz)
        
        # 返回处理后的新对象
        return dta
    def to_period(self, freq=None) -> PeriodArray:
        """
        Cast to PeriodArray/PeriodIndex at a particular frequency.

        Converts DatetimeArray/Index to PeriodArray/PeriodIndex.

        Parameters
        ----------
        freq : str or Period, optional
            One of pandas' :ref:`period aliases <timeseries.period_aliases>`
            or a Period object. Will be inferred by default.

        Returns
        -------
        PeriodArray/PeriodIndex
            Immutable ndarray holding ordinal values at a particular frequency.

        Raises
        ------
        ValueError
            When converting a DatetimeArray/Index with non-regular values,
            so that a frequency cannot be inferred.

        See Also
        --------
        PeriodIndex: Immutable ndarray holding ordinal values.
        DatetimeIndex.to_pydatetime: Return DatetimeIndex as object.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"y": [1, 2, 3]},
        ...     index=pd.to_datetime(
        ...         [
        ...             "2000-03-31 00:00:00",
        ...             "2000-05-31 00:00:00",
        ...             "2000-08-31 00:00:00",
        ...         ]
        ...     ),
        ... )
        >>> df.index.to_period("M")
        PeriodIndex(['2000-03', '2000-05', '2000-08'],
                    dtype='period[M]')

        Infer the daily frequency

        >>> idx = pd.date_range("2017-01-01", periods=2)
        >>> idx.to_period()
        PeriodIndex(['2017-01-01', '2017-01-02'],
                    dtype='period[D]')
        """
        from pandas.core.arrays import PeriodArray

        # 如果存在时区信息，发出警告并丢弃时区信息
        if self.tz is not None:
            warnings.warn(
                "Converting to PeriodArray/Index representation "
                "will drop timezone information.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        # 如果未提供频率参数，尝试从当前索引推断频率
        if freq is None:
            # 从 freqstr 或 inferred_freq 推断频率
            freq = self.freqstr or self.inferred_freq

            # 如果当前频率是 BaseOffset 的实例并且有 _period_dtype_code 属性，获取频率字符串
            if isinstance(self.freq, BaseOffset) and hasattr(
                self.freq, "_period_dtype_code"
            ):
                freq = PeriodDtype(self.freq)._freqstr

            # 如果仍然无法推断频率，抛出 ValueError
            if freq is None:
                raise ValueError(
                    "You must pass a freq argument as current index has none."
                )

            # 获取频率的别名
            res = get_period_alias(freq)

            # 修复特定问题 https://github.com/pandas-dev/pandas/issues/33358
            if res is None:
                res = freq

            freq = res

        # 返回基于 datetime64 数据创建的 PeriodArray 对象
        return PeriodArray._from_datetime64(self._ndarray, freq, tz=self.tz)

    # -----------------------------------------------------------------
    # Properties - Vectorized Timestamp Properties/Methods


注释：
- `from pandas.core.arrays import PeriodArray`：导入 Pandas 中的 `PeriodArray` 类。
- `if self.tz is not None:`：检查当前对象是否有时区信息，并发出警告，因为转换为 `PeriodArray/Index` 将丢失时区信息。
- `if freq is None:`：如果未指定频率参数，则尝试从当前索引的 `freqstr` 或 `inferred_freq` 推断频率。
- `freq = PeriodDtype(self.freq)._freqstr`：如果当前频率是 `BaseOffset` 的实例并且具有 `_period_dtype_code` 属性，则获取频率字符串。
- `raise ValueError(...)`：如果仍然无法推断频率，则引发 ValueError。
- `res = get_period_alias(freq)` 和 `if res is None:`：获取频率的别名，修复特定问题。
- `return PeriodArray._from_datetime64(self._ndarray, freq, tz=self.tz)`：返回基于 `datetime64` 数据创建的 `PeriodArray` 对象，用指定的频率和时区。
    def month_name(self, locale=None) -> npt.NDArray[np.object_]:
        """
        Return the month names with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the month name.
            Default is English locale (``'en_US.utf8'``). Use the command
            ``locale -a`` on your terminal on Unix systems to find your locale
            language code.

        Returns
        -------
        Series or Index
            Series or Index of month names.

        See Also
        --------
        DatetimeIndex.day_name : Return the day names with specified locale.

        Examples
        --------
        >>> s = pd.Series(pd.date_range(start="2018-01", freq="ME", periods=3))
        >>> s
        0   2018-01-31
        1   2018-02-28
        2   2018-03-31
        dtype: datetime64[ns]
        >>> s.dt.month_name()
        0     January
        1    February
        2       March
        dtype: object

        >>> idx = pd.date_range(start="2018-01", freq="ME", periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='ME')
        >>> idx.month_name()
        Index(['January', 'February', 'March'], dtype='object')

        Using the ``locale`` parameter you can set a different locale language,
        for example: ``idx.month_name(locale='pt_BR.utf8')`` will return month
        names in Brazilian Portuguese language.

        >>> idx = pd.date_range(start="2018-01", freq="ME", periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='ME')
        >>> idx.month_name(locale="pt_BR.utf8")  # doctest: +SKIP
        Index(['Janeiro', 'Fevereiro', 'Março'], dtype='object')
        """
        # 获取本地时间戳的值
        values = self._local_timestamps()
        
        # 调用特定字段处理函数，获取月份名称的字段值
        result = fields.get_date_name_field(
            values, "month_name", locale=locale, reso=self._creso
        )
        
        # 处理结果，可能进行结果掩码化处理，填充值为None
        result = self._maybe_mask_results(result, fill_value=None)
        
        # 返回处理后的结果
        return result
    def day_name(self, locale=None) -> npt.NDArray[np.object_]:
        """
        Return the day names with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the day name.
            Default is English locale (``'en_US.utf8'``). Use the command
            ``locale -a`` on your terminal on Unix systems to find your locale
            language code.

        Returns
        -------
        Series or Index
            Series or Index of day names.

        See Also
        --------
        DatetimeIndex.month_name : Return the month names with specified locale.

        Examples
        --------
        >>> s = pd.Series(pd.date_range(start="2018-01-01", freq="D", periods=3))
        >>> s
        0   2018-01-01
        1   2018-01-02
        2   2018-01-03
        dtype: datetime64[ns]
        >>> s.dt.day_name()
        0       Monday
        1      Tuesday
        2    Wednesday
        dtype: object

        >>> idx = pd.date_range(start="2018-01-01", freq="D", periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name()
        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')

        Using the ``locale`` parameter you can set a different locale language,
        for example: ``idx.day_name(locale='pt_BR.utf8')`` will return day
        names in Brazilian Portuguese language.

        >>> idx = pd.date_range(start="2018-01-01", freq="D", periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name(locale="pt_BR.utf8")  # doctest: +SKIP
        Index(['Segunda', 'Terça', 'Quarta'], dtype='object')
        """
        # 获取本地时间戳数据
        values = self._local_timestamps()

        # 调用外部函数 fields.get_date_name_field() 获取指定日期名称字段（如 'day_name'）的结果
        # 可以通过 locale 参数设置语言环境，reso 参数为 self._creso 的值
        result = fields.get_date_name_field(
            values, "day_name", locale=locale, reso=self._creso
        )
        # 调用内部方法 _maybe_mask_results() 可能对结果进行掩码处理，使用 fill_value=None 填充未找到的值
        result = self._maybe_mask_results(result, fill_value=None)
        # 返回处理后的结果
        return result

    @property
    @property
    def time(self) -> npt.NDArray[np.object_]:
        """
        Returns numpy array of :class:`datetime.time` objects.

        The time part of the Timestamps.

        See Also
        --------
        DatetimeIndex.timetz : Returns numpy array of :class:`datetime.time`
            objects with timezones. The time part of the Timestamps.
        DatetimeIndex.date : Returns numpy array of python :class:`datetime.date`
            objects. Namely, the date part of Timestamps without time and timezone
            information.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.time
        0    10:00:00
        1    11:00:00
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(
        ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
        ... )
        >>> idx.time
        array([datetime.time(10, 0), datetime.time(11, 0)], dtype=object)
        """
        # If the Timestamps have a timezone that is not UTC,
        # convert them into their i8 representation while
        # keeping their timezone and not using UTC
        timestamps = self._local_timestamps()

        return ints_to_pydatetime(timestamps, box="time", reso=self._creso)

    @property
    def timetz(self) -> npt.NDArray[np.object_]:
        """
        Returns numpy array of :class:`datetime.time` objects with timezones.

        The time part of the Timestamps.

        See Also
        --------
        DatetimeIndex.time : Returns numpy array of :class:`datetime.time` objects.
            The time part of the Timestamps.
        DatetimeIndex.tz : Return the timezone.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.timetz
        0    10:00:00+00:00
        1    11:00:00+00:00
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(
        ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
        ... )
        >>> idx.timetz
        array([datetime.time(10, 0, tzinfo=datetime.timezone.utc),
        datetime.time(11, 0, tzinfo=datetime.timezone.utc)], dtype=object)
        """
        # Convert the array of integer timestamps to datetime objects with timezone information,
        # using the current resolution setting (_creso) for time accuracy
        return ints_to_pydatetime(self.asi8, self.tz, box="time", reso=self._creso)
    def date(self) -> npt.NDArray[np.object_]:
        """
        Returns numpy array of python :class:`datetime.date` objects.

        Namely, the date part of Timestamps without time and
        timezone information.

        See Also
        --------
        DatetimeIndex.time : Returns numpy array of :class:`datetime.time` objects.
            The time part of the Timestamps.
        DatetimeIndex.year : The year of the datetime.
        DatetimeIndex.month : The month as January=1, December=12.
        DatetimeIndex.day : The day of the datetime.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.date
        0    2020-01-01
        1    2020-02-01
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(
        ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
        ... )
        >>> idx.date
        array([datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)], dtype=object)
        """
        # 获取本地时间戳数组
        timestamps = self._local_timestamps()
        # 将整数时间戳转换为日期对象，保留时区信息，并使用指定的分辨率
        return ints_to_pydatetime(timestamps, box="date", reso=self._creso)

    def isocalendar(self) -> DataFrame:
        """
        Calculate year, week, and day according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
            With columns year, week and day.

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> idx = pd.date_range(start="2019-12-29", freq="D", periods=4)
        >>> idx.isocalendar()
                    year  week  day
        2019-12-29  2019    52    7
        2019-12-30  2020     1    1
        2019-12-31  2020     1    2
        2020-01-01  2020     1    3
        >>> idx.isocalendar().week
        2019-12-29    52
        2019-12-30     1
        2019-12-31     1
        2020-01-01     1
        Freq: D, Name: week, dtype: UInt32
        """
        # 从本地时间戳获取数值
        values = self._local_timestamps()
        # 构建ISO日历的结构化数组，使用给定的时间分辨率
        sarray = fields.build_isocalendar_sarray(values, reso=self._creso)
        # 创建DataFrame对象，包含ISO日历的年、周和日，数据类型为UInt32
        iso_calendar_df = DataFrame(
            sarray, columns=["year", "week", "day"], dtype="UInt32"
        )
        # 如果存在缺失值，将其置为None
        if self._hasna:
            iso_calendar_df.iloc[self._isnan] = None
        return iso_calendar_df
    # 创建 _field_accessor 函数调用，设置 year 属性的访问器函数
    year = _field_accessor(
        "year",
        "Y",
        """
        The year of the datetime.

        See Also
        --------
        DatetimeIndex.month: The month as January=1, December=12.
        DatetimeIndex.day: The day of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="YE")
        ... )
        >>> datetime_series
        0   2000-12-31
        1   2001-12-31
        2   2002-12-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.year
        0    2000
        1    2001
        2    2002
        dtype: int32
        """,
    )
    # 创建 _field_accessor 函数调用，设置 month 属性的访问器函数
    month = _field_accessor(
        "month",
        "M",
        """
        The month as January=1, December=12.

        See Also
        --------
        DatetimeIndex.year: The year of the datetime.
        DatetimeIndex.day: The day of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="ME")
        ... )
        >>> datetime_series
        0   2000-01-31
        1   2000-02-29
        2   2000-03-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.month
        0    1
        1    2
        2    3
        dtype: int32
        """,
    )
    # 创建 _field_accessor 函数调用，设置 day 属性的访问器函数
    day = _field_accessor(
        "day",
        "D",
        """
        The day of the datetime.

        See Also
        --------
        DatetimeIndex.year: The year of the datetime.
        DatetimeIndex.month: The month as January=1, December=12.
        DatetimeIndex.hour: The hours of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="D")
        ... )
        >>> datetime_series
        0   2000-01-01
        1   2000-01-02
        2   2000-01-03
        dtype: datetime64[ns]
        >>> datetime_series.dt.day
        0    1
        1    2
        2    3
        dtype: int32
        """,
    )
    # 创建 _field_accessor 函数调用，设置 hour 属性的访问器函数
    hour = _field_accessor(
        "hour",
        "h",
        """
        The hours of the datetime.

        See Also
        --------
        DatetimeIndex.day: The day of the datetime.
        DatetimeIndex.minute: The minutes of the datetime.
        DatetimeIndex.second: The seconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="h")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 01:00:00
        2   2000-01-01 02:00:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.hour
        0    0
        1    1
        2    2
        dtype: int32
        """,
    )
    minute = _field_accessor(
        "minute",
        "m",
        """
        The minutes of the datetime.

        See Also
        --------
        DatetimeIndex.hour: The hours of the datetime.
        DatetimeIndex.second: The seconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="min")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:01:00
        2   2000-01-01 00:02:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.minute
        0    0
        1    1
        2    2
        dtype: int32
        """,
    )



    second = _field_accessor(
        "second",
        "s",
        """
        The seconds of the datetime.

        See Also
        --------
        DatetimeIndex.minute: The minutes of the datetime.
        DatetimeIndex.microsecond: The microseconds of the datetime.
        DatetimeIndex.nanosecond: The nanoseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="s")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:00:01
        2   2000-01-01 00:00:02
        dtype: datetime64[ns]
        >>> datetime_series.dt.second
        0    0
        1    1
        2    2
        dtype: int32
        """,
    )



    microsecond = _field_accessor(
        "microsecond",
        "us",
        """
        The microseconds of the datetime.

        See Also
        --------
        DatetimeIndex.second: The seconds of the datetime.
        DatetimeIndex.nanosecond: The nanoseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="us")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00.000000
        1   2000-01-01 00:00:00.000001
        2   2000-01-01 00:00:00.000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.microsecond
        0       0
        1       1
        2       2
        dtype: int32
        """,
    )



    nanosecond = _field_accessor(
        "nanosecond",
        "ns",
        """
        The nanoseconds of the datetime.

        See Also
        --------
        DatetimeIndex.second: The seconds of the datetime.
        DatetimeIndex.microsecond: The microseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="ns")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00.000000000
        1   2000-01-01 00:00:00.000000001
        2   2000-01-01 00:00:00.000000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.nanosecond
        0       0
        1       1
        2       2
        dtype: int32
        """,
    )
    # 定义字符串变量，包含有关“星期几”的文档字符串
    _dayofweek_doc = """
    The day of the week with Monday=0, Sunday=6.

    Return the day of the week. It is assumed the week starts on
    Monday, which is denoted by 0 and ends on Sunday which is denoted
    by 6. This method is available on both Series with datetime
    values (using the `dt` accessor) or DatetimeIndex.

    Returns
    -------
    Series or Index
        Containing integers indicating the day number.

    See Also
    --------
    Series.dt.dayofweek : Alias.
    Series.dt.weekday : Alias.
    Series.dt.day_name : Returns the name of the day of the week.

    Examples
    --------
    >>> s = pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series()
    >>> s.dt.dayofweek
    2016-12-31    5
    2017-01-01    6
    2017-01-02    0
    2017-01-03    1
    2017-01-04    2
    2017-01-05    3
    2017-01-06    4
    2017-01-07    5
    2017-01-08    6
    Freq: D, dtype: int32
    """
    # 创建与“day_of_week”相关的访问器函数，并使用_dayofweek_doc作为文档字符串
    day_of_week = _field_accessor("day_of_week", "dow", _dayofweek_doc)
    # 别名dayofweek，引用day_of_week
    dayofweek = day_of_week
    # 别名weekday，引用day_of_week

    # 创建与“day_of_year”相关的访问器函数，包含有关“一年中的日期”的文档字符串
    day_of_year = _field_accessor(
        "dayofyear",
        "doy",
        """
        The ordinal day of the year.

        See Also
        --------
        DatetimeIndex.dayofweek : The day of the week with Monday=0, Sunday=6.
        DatetimeIndex.day : The day of the datetime.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.dayofyear
        0    1
        1   32
        dtype: int32

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.dayofyear
        Index([1, 32], dtype='int32')
        """,
    )
    # 别名dayofyear，引用day_of_year

    # 创建与“quarter”相关的访问器函数，包含有关“季度”的文档字符串
    quarter = _field_accessor(
        "quarter",
        "q",
        """
        The quarter of the date.

        See Also
        --------
        DatetimeIndex.snap : Snap time stamps to nearest occurring frequency.
        DatetimeIndex.time : Returns numpy array of datetime.time objects.
            The time part of the Timestamps.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "4/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-04-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.quarter
        0    1
        1    2
        dtype: int32

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.quarter
        Index([1, 1], dtype='int32')
        """,
    )
    # 别名quarter，引用quarter
    # 定义一个字段访问器函数 days_in_month，用于获取月份的天数信息
    days_in_month = _field_accessor(
        "days_in_month",  # 字段名称
        "dim",             # 字段别名
        """
        该月份的天数信息。
    
        参见
        --------
        Series.dt.day : 返回月份的日期。
        Series.dt.is_month_end : 返回一个布尔值，指示日期是否为月末。
        Series.dt.is_month_start : 返回一个布尔值，指示日期是否为月初。
        Series.dt.month : 返回月份，一月为1，十二月为12。
    
        示例
        --------
        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.daysinmonth
        0    31
        1    29
        dtype: int32
        """,
    )
    
    # 定义一个别名 daysinmonth，用于访问 days_in_month 函数
    daysinmonth = days_in_month
    
    # 定义一个文档字符串 _is_month_doc，用于描述月初或月末的日期信息
    _is_month_doc = """
        指示日期是否为月份的{first_or_last}天。
    
        返回
        -------
        Series 或 数组
            对于 Series，返回一个布尔值的 Series。
            对于 DatetimeIndex，返回一个布尔数组。
    
        参见
        --------
        is_month_start : 返回一个布尔值，指示日期是否为月初。
        is_month_end : 返回一个布尔值，指示日期是否为月末。
    
        示例
        --------
        该方法可用于具有日期时间值的 Series，在 ``.dt`` 访问器下使用，也可以直接在 DatetimeIndex 上使用。
    
        >>> s = pd.Series(pd.date_range("2018-02-27", periods=3))
        >>> s
        0   2018-02-27
        1   2018-02-28
        2   2018-03-01
        dtype: datetime64[ns]
        >>> s.dt.is_month_start
        0    False
        1    False
        2    True
        dtype: bool
        >>> s.dt.is_month_end
        0    False
        1    True
        2    False
        dtype: bool
    
        >>> idx = pd.date_range("2018-02-27", periods=3)
        >>> idx.is_month_start
        array([False, False, True])
        >>> idx.is_month_end
        array([False, True, False])
    """
    
    # 定义一个字段访问器函数 is_month_start，用于检查日期是否为月初
    is_month_start = _field_accessor(
        "is_month_start",                        # 字段名称
        "is_month_start",                        # 字段别名
        _is_month_doc.format(first_or_last="first")  # 使用 _is_month_doc 格式化后的文档字符串
    )
    
    # 定义一个字段访问器函数 is_month_end，用于检查日期是否为月末
    is_month_end = _field_accessor(
        "is_month_end",                          # 字段名称
        "is_month_end",                          # 字段别名
        _is_month_doc.format(first_or_last="last")   # 使用 _is_month_doc 格式化后的文档字符串
    )
    is_quarter_start = _field_accessor(
        "is_quarter_start",
        "is_quarter_start",
        """
        Indicator for whether the date is the first day of a quarter.

        Returns
        -------
        is_quarter_start : Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_end : Similar property for indicating the quarter end.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
        ...                   periods=4)})
        >>> df.assign(quarter=df.dates.dt.quarter,
        ...           is_quarter_start=df.dates.dt.is_quarter_start)
               dates  quarter  is_quarter_start
        0 2017-03-30        1             False
        1 2017-03-31        1             False
        2 2017-04-01        2              True
        3 2017-04-02        2             False

        >>> idx = pd.date_range('2017-03-30', periods=4)
        >>> idx
        DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_quarter_start
        array([False, False,  True, False])
        """,
    )



# 创建一个字段访问器函数 is_quarter_start，用于判断日期是否为季度的第一天
is_quarter_start = _field_accessor(
    # 字段名称为 "is_quarter_start"
    "is_quarter_start",
    # 文档字符串描述了日期是否为季度的第一天的指示器
    "is_quarter_start",
    """
    Indicator for whether the date is the first day of a quarter.

    Returns
    -------
    is_quarter_start : Series or DatetimeIndex
        The same type as the original data with boolean values. Series will
        have the same name and index. DatetimeIndex will have the same
        name.

    See Also
    --------
    quarter : Return the quarter of the date.
    is_quarter_end : Similar property for indicating the quarter end.

    Examples
    --------
    This method is available on Series with datetime values under
    the ``.dt`` accessor, and directly on DatetimeIndex.

    >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
    ...                   periods=4)})
    >>> df.assign(quarter=df.dates.dt.quarter,
    ...           is_quarter_start=df.dates.dt.is_quarter_start)
           dates  quarter  is_quarter_start
    0 2017-03-30        1             False
    1 2017-03-31        1             False
    2 2017-04-01        2              True
    3 2017-04-02        2             False

    >>> idx = pd.date_range('2017-03-30', periods=4)
    >>> idx
    DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                  dtype='datetime64[ns]', freq='D')

    >>> idx.is_quarter_start
    array([False, False,  True, False])
    """,
)
    # 创建一个函数 is_quarter_end，并通过 _field_accessor 方法进行封装
    is_quarter_end = _field_accessor(
        # 属性名为 "is_quarter_end"
        "is_quarter_end",
        # 文档字符串描述：指示日期是否是季度的最后一天
        """
        Indicator for whether the date is the last day of a quarter.

        Returns
        -------
        is_quarter_end : Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_start : Similar property indicating the quarter start.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
        ...                    periods=4)})
        >>> df.assign(quarter=df.dates.dt.quarter,
        ...           is_quarter_end=df.dates.dt.is_quarter_end)
               dates  quarter    is_quarter_end
        0 2017-03-30        1             False
        1 2017-03-31        1              True
        2 2017-04-01        2             False
        3 2017-04-02        2             False

        >>> idx = pd.date_range('2017-03-30', periods=4)
        >>> idx
        DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_quarter_end
        array([False,  True, False, False])
        """,
    )
    # 创建一个名为 is_year_start 的属性，其通过调用 _field_accessor 函数实现
    is_year_start = _field_accessor(
        "is_year_start",  # 属性的名称
        "is_year_start",  # 属性的别名
        """
        Indicate whether the date is the first day of a year.

        Returns
        -------
        Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        is_year_end : Similar property indicating the last day of the year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_start
        0    False
        1    False
        2    True
        dtype: bool

        >>> idx = pd.date_range("2017-12-30", periods=3)
        >>> idx
        DatetimeIndex(['2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_year_start
        array([False, False,  True])

        This method, when applied to Series with datetime values under
        the ``.dt`` accessor, will lose information about Business offsets.

        >>> dates = pd.Series(pd.date_range("2020-10-30", periods=4, freq="BYS"))
        >>> dates
        0   2021-01-01
        1   2022-01-03
        2   2023-01-02
        3   2024-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_start
        0    True
        1    False
        2    False
        3    True
        dtype: bool

        >>> idx = pd.date_range("2020-10-30", periods=4, freq="BYS")
        >>> idx
        DatetimeIndex(['2021-01-01', '2022-01-03', '2023-01-02', '2024-01-01'],
                      dtype='datetime64[ns]', freq='BYS-JAN')

        >>> idx.is_year_start
        array([ True,  True,  True,  True])
        """,  # 属性的文档字符串，详细描述了属性的功能、返回值类型及用法示例
    )
    is_year_end = _field_accessor(
        "is_year_end",
        "is_year_end",
        """
        判断日期是否为一年的最后一天。

        返回
        -------
        Series 或 DatetimeIndex
            与原始数据类型相同，但值为布尔类型。Series 将保留原名称和索引，DatetimeIndex 将保留原名称。

        另请参阅
        --------
        is_year_start : 类似的属性，指示一年的开始日期。

        示例
        --------
        该方法适用于包含日期时间值的 Series，在 ``.dt`` 访问器下使用；也可以直接用于 DatetimeIndex。

        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_end
        0    False
        1     True
        2    False
        dtype: bool

        >>> idx = pd.date_range("2017-12-30", periods=3)
        >>> idx
        DatetimeIndex(['2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_year_end
        array([False,  True, False])
        """,
    )
    is_leap_year = _field_accessor(
        "is_leap_year",
        "is_leap_year",
        """
        判断日期是否为闰年。

        闰年是指有 366 天（包括闰年的 2 月 29 日）的年份。
        闰年的条件是能被 4 整除但不能被 100 整除，或者能被 400 整除。

        返回
        -------
        Series 或 ndarray
             布尔值，指示日期是否为闰年。

        另请参阅
        --------
        DatetimeIndex.is_year_end : 判断日期是否为一年的最后一天。
        DatetimeIndex.is_year_start : 判断日期是否为一年的第一天。

        示例
        --------
        该方法适用于包含日期时间值的 Series，在 ``.dt`` 访问器下使用；也可以直接用于 DatetimeIndex。

        >>> idx = pd.date_range("2012-01-01", "2015-01-01", freq="YE")
        >>> idx
        DatetimeIndex(['2012-12-31', '2013-12-31', '2014-12-31'],
                      dtype='datetime64[ns]', freq='YE-DEC')
        >>> idx.is_leap_year
        array([ True, False, False])

        >>> dates_series = pd.Series(idx)
        >>> dates_series
        0   2012-12-31
        1   2013-12-31
        2   2014-12-31
        dtype: datetime64[ns]
        >>> dates_series.dt.is_leap_year
        0     True
        1    False
        2    False
        dtype: bool
        """,
    )
    def to_julian_date(self) -> npt.NDArray[np.float64]:
        """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """

        # 根据Julian日历算法将日期数组转换为float64类型的Julian日期数组
        # 0 Julian日期对应于公元前4713年1月1日中午
        # 参考链接：https://en.wikipedia.org/wiki/Julian_day

        # 将年份、月份、日期转换为NumPy数组
        year = np.asarray(self.year)
        month = np.asarray(self.month)
        day = np.asarray(self.day)

        # 对月份小于3的情况进行处理，根据Julian日历规则调整年份和月份
        testarr = month < 3
        year[testarr] -= 1
        month[testarr] += 12

        # 计算Julian日期的算法表达式，返回Julian日期的数组
        return (
            day
            + np.fix((153 * month - 457) / 5)
            + 365 * year
            + np.floor(year / 4)
            - np.floor(year / 100)
            + np.floor(year / 400)
            + 1_721_118.5
            + (
                self.hour
                + self.minute / 60
                + self.second / 3600
                + self.microsecond / 3600 / 10**6
                + self.nanosecond / 3600 / 10**9
            )
            / 24
        )

    # -----------------------------------------------------------------
    # Reductions

    # 标准差计算方法
    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
# -------------------------------------------------------------------
# Constructor Helpers

# 将序列转换为 datetime64 数据类型的辅助函数
def _sequence_to_dt64(
    data: ArrayLike,
    *,
    copy: bool = False,
    tz: tzinfo | None = None,
    dayfirst: bool = False,
    yearfirst: bool = False,
    ambiguous: TimeAmbiguous = "raise",
    out_unit: str | None = None,
) -> tuple[np.ndarray, tzinfo | None]:
    """
    Parameters
    ----------
    data : np.ndarray or ExtensionArray
        dtl.ensure_arraylike_for_datetimelike has already been called.
        已经调用 dtl.ensure_arraylike_for_datetimelike 处理过的 numpy 数组或扩展数组。
    copy : bool, default False
        是否复制输入数据。
    tz : tzinfo or None, default None
        时区信息，如果为 None 则表示无时区信息。
    dayfirst : bool, default False
        是否优先解析日期中的日。
    yearfirst : bool, default False
        是否优先解析日期中的年。
    ambiguous : str, bool, or arraylike, default 'raise'
        见 pandas._libs.tslibs.tzconversion.tz_localize_to_utc。
        处理时间模糊性的方式，如 'raise' 表示抛出异常。
    out_unit : str or None, default None
        期望的输出时间分辨率。

    Returns
    -------
    result : numpy.ndarray
        转换为 numpy 数组，并使用 dtype ``datetime64[unit]``。
        其中 `unit` 默认为 "ns"，除非通过 `out_unit` 指定了其他单位。
    tz : tzinfo or None
        用户提供的时区信息，或者从数据中推断出的时区信息。

    Raises
    ------
    TypeError : PeriodDType data is passed
        如果传递了 PeriodDType 类型的数据，则会引发此异常。
    """

    # 判断数据是否需要转换数据类型，并进行可能的复制操作
    data, copy = maybe_convert_dtype(data, copy, tz=tz)
    data_dtype = getattr(data, "dtype", None)

    # 输出数据类型默认为 datetime64[ns]
    out_dtype = DT64NS_DTYPE
    if out_unit is not None:
        out_dtype = np.dtype(f"M8[{out_unit}]")
    # 如果数据类型是 object 或者字符串类型，执行以下操作
    if data_dtype == object or is_string_dtype(data_dtype):
        # TODO: 我们没有针对字符串类型进行特定的测试，
        #  也没有测试复杂类型、分类类型或其他扩展类型的情况
        # 将 data 强制转换为 numpy 数组类型
        data = cast(np.ndarray, data)
        copy = False
        # 如果推断出 data 的类型为整数，比使用 array_to_datetime 更高效
        if lib.infer_dtype(data, skipna=False) == "integer":
            # 将 data 转换为 np.int64 类型
            data = data.astype(np.int64)
        # 如果指定了时区并且 ambiguous 设置为 "raise"
        elif tz is not None and ambiguous == "raise":
            # 将 data 转换为对象数组类型
            obj_data = np.asarray(data, dtype=object)
            # 调用 tslib.array_to_datetime_with_tz 函数处理带时区的日期时间数据
            result = tslib.array_to_datetime_with_tz(
                obj_data,
                tz=tz,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                creso=abbrev_to_npy_unit(out_unit),
            )
            return result, tz
        else:
            # 调用 objects_to_datetime64 处理日期时间对象转换
            converted, inferred_tz = objects_to_datetime64(
                data,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                allow_object=False,
                out_unit=out_unit,
            )
            copy = False
            # 如果指定了时区并且推断出时区不同
            if tz and inferred_tz:
                # 两个不同的时区：转换为指定的基于 UTC 的表示
                # 根据约定，这些已经是 UTC 时间
                result = converted
            elif inferred_tz:
                # 推断出时区不同，使用推断的时区
                tz = inferred_tz
                result = converted
            else:
                # 否则，调用 _construct_from_dt64_naive 处理 datetime64 类型的数据
                result, _ = _construct_from_dt64_naive(
                    converted, tz=tz, copy=copy, ambiguous=ambiguous
                )
            return result, tz

        # 更新 data_dtype 为 data 的 dtype
        data_dtype = data.dtype

    # 如果 data 的原始类型是 Categorical[datetime64[ns, tz]]，需要处理这些类型
    if isinstance(data_dtype, DatetimeTZDtype):
        # 将 DatetimeArray 转换为 ndarray 类型
        data = cast(DatetimeArray, data)
        # 推断时区并更新 tz
        tz = _maybe_infer_tz(tz, data.tz)
        # 获取 data 的 _ndarray 属性作为结果
        result = data._ndarray

    # 如果 data_dtype 是 np.datetime64 类型的，处理时区无关的 DatetimeArray 或者 ndarray[datetime64]
    elif lib.is_np_dtype(data_dtype, "M"):
        # 如果 data 是 DatetimeArray 类型，获取其 _ndarray 属性
        if isinstance(data, DatetimeArray):
            data = data._ndarray
        # 将 data 强制转换为 np.ndarray 类型
        data = cast(np.ndarray, data)
        # 调用 _construct_from_dt64_naive 处理 datetime64 类型的数据
        result, copy = _construct_from_dt64_naive(
            data, tz=tz, copy=copy, ambiguous=ambiguous
        )

    else:
        # 否则，假设数据类型是整数 dtype
        # 如果 data 的 dtype 不是 INT64_DTYPE，则强制转换为 np.int64 类型
        if data.dtype != INT64_DTYPE:
            data = data.astype(np.int64, copy=False)
            copy = False
        # 将 data 强制转换为 np.ndarray 类型
        data = cast(np.ndarray, data)
        # 将 data 视图转换为指定的输出 dtype
        result = data.view(out_dtype)

    # 如果需要复制 result
    if copy:
        # 复制 result
        result = result.copy()

    # 断言结果 result 是 np.ndarray 类型
    assert isinstance(result, np.ndarray), type(result)
    # 断言结果 result 的 dtype 的种类是 "M"
    assert result.dtype.kind == "M"
    # 断言结果 result 的 dtype 不是 "M8"
    assert result.dtype != "M8"
    # 断言结果 result 的 dtype 是支持的 dtype
    assert is_supported_dtype(result.dtype)
    # 返回结果 result 和时区 tz
    return result, tz
def _construct_from_dt64_naive(
    data: np.ndarray, *, tz: tzinfo | None, copy: bool, ambiguous: TimeAmbiguous
) -> tuple[np.ndarray, bool]:
    """
    Convert datetime64 data to a supported dtype, localizing if necessary.
    """
    # Caller is responsible for ensuring
    #  lib.is_np_dtype(data.dtype)
    
    # 确保调用者已经确保 data.dtype 是 NumPy 支持的数据类型

    new_dtype = data.dtype
    if not is_supported_dtype(new_dtype):
        # Cast to the nearest supported unit, generally "s"
        new_dtype = get_supported_dtype(new_dtype)
        data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
        copy = False

    if data.dtype.byteorder == ">":
        # TODO: better way to handle this?  non-copying alternative?
        #  without this, test_constructor_datetime64_bigendian fails
        # 如果数据类型的字节顺序是大端字节序，则将其转换为小端字节序以处理特定问题
        data = data.astype(data.dtype.newbyteorder("<"))
        new_dtype = data.dtype
        copy = False

    if tz is not None:
        # Convert tz-naive to UTC
        # TODO: if tz is UTC, are there situations where we *don't* want a
        #  copy?  tz_localize_to_utc always makes one.
        # 如果 tz 不为 None，则将时间从时区无关转换为 UTC 时间
        shape = data.shape
        if data.ndim > 1:
            data = data.ravel()

        data_unit = get_unit_from_dtype(new_dtype)
        data = tzconversion.tz_localize_to_utc(
            data.view("i8"), tz, ambiguous=ambiguous, creso=data_unit
        )
        data = data.view(new_dtype)
        data = data.reshape(shape)

    assert data.dtype == new_dtype, data.dtype
    result = data

    return result, copy


def objects_to_datetime64(
    data: np.ndarray,
    dayfirst,
    yearfirst,
    utc: bool = False,
    errors: DateTimeErrorChoices = "raise",
    allow_object: bool = False,
    out_unit: str | None = None,
) -> tuple[np.ndarray, tzinfo | None]:
    """
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert/localize timestamps to UTC.
    errors : {'raise', 'coerce'}
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    out_unit : str or None, default None
        None indicates we should do resolution inference.

    Returns
    -------
    result : ndarray
        np.datetime64[out_unit] if returned values represent wall times or UTC
        timestamps.
        object if mixed timezones
    inferred_tz : tzinfo or None
        If not None, then the datetime64 values in `result` denote UTC timestamps.

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    TypeError  : When a type cannot be converted to datetime
    """
    assert errors in ["raise", "coerce"]

    # if str-dtype, convert
    data = np.asarray(data, dtype=np.object_)
    
    # 将输入数据强制转换为 object 类型的 NumPy 数组
    result, tz_parsed = tslib.array_to_datetime(
        data,
        errors=errors,
        utc=utc,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        creso=abbrev_to_npy_unit(out_unit),
    )
    # 调用 tslib 模块的 array_to_datetime 函数，将数据转换为 datetime64 类型数组
    # 并接收返回的结果数组 result 和时区解析信息 tz_parsed

    if tz_parsed is not None:
        # 如果 tz_parsed 不为 None，表示结果数组 result 是在 UTC 时区下的 datetime64 numpy 数组
        # 可以直接返回结果和时区解析信息 tz_parsed
        return result, tz_parsed
    elif result.dtype.kind == "M":
        # 如果 result 的数据类型的种类为 "M"，表示结果数组 result 是 datetime64 类型
        # 但时区未知，因此也可以直接返回结果和时区解析信息 tz_parsed
        return result, tz_parsed
    elif result.dtype == object:
        # 如果 result 的数据类型为 object，可能有以下情况之一：
        # - 当通过 `pd.to_datetime` 调用时，返回的是 object 类型数组，这是被允许的；
        # - 当通过 `pd.DatetimeIndex` 调用时，必须返回 datetime64 类型数组，因此如果返回的是 object 类型数组，
        #   表示数据被识别为日期时间，但具有冲突的时区或意识，此时根据 allow_object 的设置决定是否返回结果和时区解析信息 tz_parsed
        if allow_object:
            return result, tz_parsed
        raise TypeError("DatetimeIndex has mixed timezones")
    else:  # pragma: no cover
        # 如果 result 的数据类型不是 datetime64 类型，也不是 object 类型，这种情况在测试覆盖率下不应该出现
        # 抛出 TypeError，因为预期上述情况应该可以处理所有可能的情况，这里表示存在未预期的类型错误
        raise TypeError(result)
def maybe_convert_dtype(data, copy: bool, tz: tzinfo | None = None):
    """
    Convert data based on dtype conventions, issuing
    errors where appropriate.

    Parameters
    ----------
    data : np.ndarray or pd.Index
        待转换的数据，可以是 NumPy 数组或 Pandas 索引对象
    copy : bool
        是否复制数据
    tz : tzinfo or None, default None
        时区信息，可以为 tzinfo 对象或 None，默认为 None

    Returns
    -------
    data : np.ndarray or pd.Index
        转换后的数据，与输入的数据类型相同
    copy : bool
        是否复制数据，与输入相同

    Raises
    ------
    TypeError : PeriodDType data is passed
        如果传递了 PeriodDType 类型的数据，则会引发 TypeError 异常
    """
    if not hasattr(data, "dtype"):
        # e.g. collections.deque
        # 如果数据对象没有 dtype 属性（如 collections.deque），直接返回原数据和复制标志
        return data, copy

    if is_float_dtype(data.dtype):
        # pre-2.0 we treated these as wall-times, inconsistent with ints
        # GH#23675, GH#45573 deprecated to treat symmetrically with integer dtypes.
        # Note: data.astype(np.int64) fails ARM tests, see
        # https://github.com/pandas-dev/pandas/issues/49468.
        # 如果数据类型是浮点数，将其转换为 datetime64[ns] 类型的整数视图
        data = data.astype(DT64NS_DTYPE).view("i8")
        copy = False

    elif lib.is_np_dtype(data.dtype, "m") or is_bool_dtype(data.dtype):
        # GH#29794 enforcing deprecation introduced in GH#23539
        # 如果数据类型是 datetime64[ns] 或布尔型，则抛出异常
        raise TypeError(f"dtype {data.dtype} cannot be converted to datetime64[ns]")
    elif isinstance(data.dtype, PeriodDtype):
        # Note: without explicitly raising here, PeriodIndex
        #  test_setops.test_join_does_not_recur fails
        # 如果数据类型是 PeriodDtype，则抛出异常
        raise TypeError(
            "Passing PeriodDtype data is invalid. Use `data.to_timestamp()` instead"
        )

    elif isinstance(data.dtype, ExtensionDtype) and not isinstance(
        data.dtype, DatetimeTZDtype
    ):
        # TODO: We have no tests for these
        # 如果数据类型是扩展类型，但不是 DatetimeTZDtype，则将数据转换为对象数组
        data = np.array(data, dtype=np.object_)
        copy = False

    return data, copy


# -------------------------------------------------------------------
# Validation and Inference


def _maybe_infer_tz(tz: tzinfo | None, inferred_tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is inferred from data, check that it is compatible with
    the user-provided timezone, if any.

    Parameters
    ----------
    tz : tzinfo or None
        用户提供的时区信息，可以为 tzinfo 对象或 None
    inferred_tz : tzinfo or None
        从数据中推断出的时区信息，可以为 tzinfo 对象或 None

    Returns
    -------
    tz : tzinfo or None
        返回适用的时区信息，可能为用户提供的时区信息或从数据中推断出的时区信息

    Raises
    ------
    TypeError : if both timezones are present but do not match
        如果用户提供的时区信息和从数据中推断出的时区信息存在且不匹配，则抛出 TypeError 异常
    """
    if tz is None:
        tz = inferred_tz
    elif inferred_tz is None:
        pass
    elif not timezones.tz_compare(tz, inferred_tz):
        # 如果两个时区信息都存在但不匹配，则抛出异常
        raise TypeError(
            f"data is already tz-aware {inferred_tz}, unable to "
            f"set specified tz: {tz}"
        )
    return tz


def _validate_dt64_dtype(dtype):
    """
    Check that a dtype, if passed, represents either a numpy datetime64[ns]
    dtype or a pandas DatetimeTZDtype.

    Parameters
    ----------
    dtype : object
        待验证的数据类型对象

    Returns
    -------
    dtype : None, numpy.dtype, or DatetimeTZDtype
        如果数据类型合法，则返回其本身或特定类型，否则返回 None

    Raises
    ------
    ValueError : invalid dtype
        如果传递的数据类型无效，则引发 ValueError 异常

    Notes
    -----
    Unlike _validate_tz_from_dtype, this does _not_ allow non-existent
    tz errors to go through
    """
    if dtype is not None:
        # 如果 dtype 参数不为 None，则进行以下处理

        dtype = pandas_dtype(dtype)
        # 将 dtype 转换为 pandas 中的数据类型

        if dtype == np.dtype("M8"):
            # 如果 dtype 是 datetime64 的一种，但没有指定精度，抛出异常
            msg = (
                "Passing in 'datetime64' dtype with no precision is not allowed. "
                "Please pass in 'datetime64[ns]' instead."
            )
            raise ValueError(msg)

        if (
            isinstance(dtype, np.dtype)
            and (dtype.kind != "M" or not is_supported_dtype(dtype))
        ) or not isinstance(dtype, (np.dtype, DatetimeTZDtype)):
            # 如果 dtype 不是 datetime64 或 DatetimeTZDtype 类型，或者不受支持的 dtype 类型，则抛出异常
            raise ValueError(
                f"Unexpected value for 'dtype': '{dtype}'. "
                "Must be 'datetime64[s]', 'datetime64[ms]', 'datetime64[us]', "
                "'datetime64[ns]' or DatetimeTZDtype'."
            )

        if getattr(dtype, "tz", None):
            # 如果 dtype 具有 tz 属性（时区信息）
            # 确保对于 pytz 对象，具有一个标准的时区
            # 没有这个处理，例如将一个时间增量数组和一个带有特定时区的时间戳相加，将会出现整体不正确的情况
            dtype = cast(DatetimeTZDtype, dtype)
            dtype = DatetimeTZDtype(
                unit=dtype.unit, tz=timezones.tz_standardize(dtype.tz)
            )

    return dtype
    # 返回处理后的 dtype 参数
# 从数据类型中验证时区信息，确保与给定的时区不冲突
def _validate_tz_from_dtype(
    dtype, tz: tzinfo | None, explicit_tz_none: bool = False
) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo
    explicit_tz_none : bool, default False
        Whether tz=None was passed explicitly, as opposed to lib.no_default.

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
    # 如果数据类型不为空
    if dtype is not None:
        # 如果数据类型是字符串
        if isinstance(dtype, str):
            try:
                # 尝试从字符串构造 DatetimeTZDtype 对象
                dtype = DatetimeTZDtype.construct_from_string(dtype)
            except TypeError:
                # 捕获异常，允许非存在的时区错误通过
                pass
        # 获取数据类型中的时区信息
        dtz = getattr(dtype, "tz", None)
        # 如果数据类型中存在时区信息
        if dtz is not None:
            # 如果给定的时区不为空且与数据类型中的时区不一致
            if tz is not None and not timezones.tz_compare(tz, dtz):
                raise ValueError("cannot supply both a tz and a dtype with a tz")
            # 如果显式传递了 tz=None
            if explicit_tz_none:
                raise ValueError("Cannot pass both a timezone-aware dtype and tz=None")
            # 更新时区信息为数据类型中的时区信息
            tz = dtz

        # 如果给定的时区不为空且数据类型是 numpy 日期时间类型
        if tz is not None and lib.is_np_dtype(dtype, "M"):
            # 检查用户是否传递了时区无关的数据类型
            if tz is not None and not timezones.tz_compare(tz, dtz):
                raise ValueError(
                    "cannot supply both a tz and a "
                    "timezone-naive dtype (i.e. datetime64[ns])"
                )

    # 返回最终的时区信息
    return tz


# 从起始点和结束点推断时区信息
def _infer_tz_from_endpoints(
    start: Timestamp, end: Timestamp, tz: tzinfo | None
) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        # 从起始点和结束点推断时区信息
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        # 如果传递的起始点和结束点有不同的时区，抛出异常
        raise TypeError(
            "Start and end cannot both be tz-aware with different timezones"
        ) from err

    # 获取推断的时区信息
    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    # 获取传递的时区信息
    tz = timezones.maybe_get_tz(tz)

    # 如果传递的时区信息和推断的时区信息都不为空
    if tz is not None and inferred_tz is not None:
        # 如果传递的时区信息和推断的时区信息不一致
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError("Inferred time zone not equal to passed time zone")
    # 如果推断出时区不为 None，则使用推断出的时区作为最终时区
    elif inferred_tz is not None:
        tz = inferred_tz

    # 返回最终确定的时区
    return tz
# 函数定义：根据需要可能标准化起始点和结束点
def _maybe_normalize_endpoints(
    start: _TimestampNoneT1, end: _TimestampNoneT2, normalize: bool
) -> tuple[_TimestampNoneT1, _TimestampNoneT2]:
    # 如果需要标准化
    if normalize:
        # 如果起始点不为 None，则标准化起始点
        if start is not None:
            start = start.normalize()

        # 如果结束点不为 None，则标准化结束点
        if end is not None:
            end = end.normalize()

    # 返回标准化后的起始点和结束点
    return start, end


# 函数定义：可能将时间点本地化
def _maybe_localize_point(
    ts: Timestamp | None, freq, tz, ambiguous, nonexistent
) -> Timestamp | None:
    """
    Localize a start or end Timestamp to the timezone of the corresponding
    start or end Timestamp

    Parameters
    ----------
    ts : start or end Timestamp to potentially localize
    freq : Tick, DateOffset, or None
    tz : str, timezone object or None
    ambiguous: str, localization behavior for ambiguous times
    nonexistent: str, localization behavior for nonexistent times

    Returns
    -------
    ts : Timestamp
    """
    # 如果时间点不为 None 并且未本地化
    if ts is not None and ts.tzinfo is None:
        # 确保如果：
        # 1) freq 是类似于 Timedelta 的频率 (Tick)
        # 2) freq 为 None，即生成一个线性间隔范围
        ambiguous = ambiguous if ambiguous != "infer" else False
        localize_args = {"ambiguous": ambiguous, "nonexistent": nonexistent, "tz": None}
        if isinstance(freq, Tick) or freq is None:
            localize_args["tz"] = tz
        # 将时间点本地化
        ts = ts.tz_localize(**localize_args)
    # 返回本地化后的时间点
    return ts


# 函数定义：生成一个时间范围的生成器
def _generate_range(
    start: Timestamp | None,
    end: Timestamp | None,
    periods: int | None,
    offset: BaseOffset,
    *,
    unit: str,
) -> Generator[Timestamp, None, None]:
    """
    Generates a sequence of dates corresponding to the specified time
    offset. Similar to dateutil.rrule except uses pandas DateOffset
    objects to represent time increments.

    Parameters
    ----------
    start : Timestamp or None
    end : Timestamp or None
    periods : int or None
    offset : DateOffset
    unit : str

    Notes
    -----
    * This method is faster for generating weekdays than dateutil.rrule
    * At least two of (start, end, periods) must be specified.
    * If both start and end are specified, the returned dates will
    satisfy start <= date <= end.

    Returns
    -------
    dates : generator object
    """
    # 将 offset 转换为 DateOffset 对象
    offset = to_offset(offset)

    # 如果起始点不为 NaT (not a time)，将其转换为指定单位的时间单位
    start = Timestamp(start)  # type: ignore[arg-type]
    if start is not NaT:
        start = start.as_unit(unit)
    else:
        start = None

    # 如果结束点不为 NaT (not a time)，将其转换为指定单位的时间单位
    end = Timestamp(end)  # type: ignore[arg-type]
    if end is not NaT:
        end = end.as_unit(unit)
    else:
        end = None
    # 如果指定了起始时间，并且起始时间不在偏移量的时间点上
    # GH #56147 考虑负方向和范围边界
    if start and not offset.is_on_offset(start):
        # 如果偏移量的天数大于等于零，则将起始时间向前滚动到最接近的偏移量时间点
        start = offset.rollforward(start)  # type: ignore[assignment]
        # 否则将起始时间向后滚动到最接近的偏移量时间点
        start = offset.rollback(start)  # type: ignore[assignment]

    # 如果周期数为 None 并且结束时间早于起始时间，并且偏移量的天数大于等于零
    # Unsupported operand types for < ("Timestamp" and "None")
    if periods is None and end < start and offset.n >= 0:  # type: ignore[operator]
        # 将结束时间设置为 None
        end = None
        # 将周期数设置为 0
        periods = 0

    # 如果结束时间为 None
    if end is None:
        # 将结束时间设置为起始时间加上周期数减去 1 的偏移量
        # error: No overload variant of "__radd__" of "BaseOffset" matches
        # argument type "None"
        end = start + (periods - 1) * offset  # type: ignore[operator]

    # 如果起始时间为 None
    if start is None:
        # 将起始时间设置为结束时间减去周期数减去 1 的偏移量
        # error: No overload variant of "__radd__" of "BaseOffset" matches
        # argument type "None"
        start = end - (periods - 1) * offset  # type: ignore[operator]

    # 将起始时间和结束时间强制转换为 Timestamp 类型
    start = cast(Timestamp, start)
    end = cast(Timestamp, end)

    # 设置当前时间为起始时间
    cur = start
    # 如果偏移量的天数大于等于零
    if offset.n >= 0:
        # 循环直到当前时间大于结束时间
        while cur <= end:
            # 生成当前时间
            yield cur

            # 如果当前时间等于结束时间，则避免在偏移量应用中进行加法操作
            if cur == end:
                # GH#24252 避免在不必要的情况下执行偏移量的增加操作
                break

            # 更快地获取下一个日期，避免使用 cur + offset
            next_date = offset._apply(cur)
            next_date = next_date.as_unit(unit)
            # 如果下一个日期小于等于当前日期，则抛出值错误异常
            if next_date <= cur:
                raise ValueError(f"Offset {offset} did not increment date")
            cur = next_date
    else:
        # 偏移量的天数小于零时的循环
        while cur >= end:
            # 生成当前时间
            yield cur

            # 如果当前时间等于结束时间，则避免在偏移量应用中进行加法操作
            if cur == end:
                # GH#24252 避免在不必要的情况下执行偏移量的增加操作
                break

            # 更快地获取下一个日期，避免使用 cur + offset
            next_date = offset._apply(cur)
            next_date = next_date.as_unit(unit)
            # 如果下一个日期大于等于当前日期，则抛出值错误异常
            if next_date >= cur:
                raise ValueError(f"Offset {offset} did not decrement date")
            cur = next_date
```