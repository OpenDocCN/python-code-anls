# `D:\src\scipysrc\pandas\pandas\core\arrays\period.py`

```
# 引入未来版本特性模块，用于支持类型注解中的类型自引用
from __future__ import annotations

# 引入时间间隔 timedelta 类
from datetime import timedelta

# 引入操作符模块，用于支持运算符的重载
import operator

# 引入类型提示相关模块
from typing import (
    TYPE_CHECKING,  # 类型检查开关
    Any,  # 任意类型
    Literal,  # 字面量类型
    TypeVar,  # 类型变量
    cast,  # 类型转换函数
    overload,  # 函数重载装饰器
)
# 引入警告模块
import warnings

# 引入 NumPy 库
import numpy as np

# 引入 pandas 库中的部分 C 语言扩展库
from pandas._libs import (
    algos as libalgos,  # 算法模块
    lib,  # 核心模块
)

# 引入 pandas 库中的数组支持模块
from pandas._libs.arrays import NDArrayBacked

# 引入 pandas 库中的时间序列库
from pandas._libs.tslibs import (
    BaseOffset,  # 基础偏移量类
    NaT,  # 不可用时间类
    NaTType,  # 不可用时间类型
    Timedelta,  # 时间间隔类
    add_overflowsafe,  # 安全加法函数
    astype_overflowsafe,  # 安全类型转换函数
    dt64arr_to_periodarr as c_dt64arr_to_periodarr,  # 将 numpy.datetime64 数组转换为 Period 数组
    get_unit_from_dtype,  # 从数据类型获取单位函数
    iNaT,  # 无效时间表示
    parsing,  # 解析函数
    period as libperiod,  # 时间段模块
    to_offset,  # 转换为偏移量函数
)

# 引入 pandas 库中的时间序列数据类型模块
from pandas._libs.tslibs.dtypes import (
    FreqGroup,  # 频率分组类
    PeriodDtypeBase,  # 时间段数据类型基类
)

# 引入 pandas 库中的字段判断模块
from pandas._libs.tslibs.fields import isleapyear_arr

# 引入 pandas 库中的时间偏移量模块
from pandas._libs.tslibs.offsets import (
    Tick,  # 增量
    delta_to_tick,  # 将增量转换为增量表示
)

# 引入 pandas 库中的时间段处理模块
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ,  # 不同频率错误
    IncompatibleFrequency,  # 不兼容频率异常
    Period,  # 时间段类
    get_period_field_arr,  # 获取时间段字段数组函数
    period_asfreq_arr,  # 时间段频率转换函数
)

# 引入 pandas 库中的工具装饰器模块
from pandas.util._decorators import (
    cache_readonly,  # 只读缓存装饰器
    doc,  # 文档装饰器
)

# 引入 pandas 核心数据类型通用模块
from pandas.core.dtypes.common import (
    ensure_object,  # 确保对象函数
    pandas_dtype,  # pandas 数据类型函数
)

# 引入 pandas 核心数据类型模块
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,  # 带时区的日期时间数据类型
    PeriodDtype,  # 时间段数据类型
)

# 引入 pandas 核心数据类型通用模块
from pandas.core.dtypes.generic import (
    ABCIndex,  # 抽象索引基类
    ABCPeriodIndex,  # 抽象时间段索引基类
    ABCSeries,  # 抽象序列基类
    ABCTimedeltaArray,  # 抽象时间差数组基类
)

# 引入 pandas 核心数据类型缺失值处理模块
from pandas.core.dtypes.missing import isna

# 引入 pandas 核心数组日期时间相关模块
from pandas.core.arrays import datetimelike as dtl

# 引入 pandas 核心公共模块
import pandas.core.common as com

# 如果是类型检查模式，引入额外的类型相关模块
if TYPE_CHECKING:
    from collections.abc import (
        Callable,  # 可调用对象抽象基类
        Sequence,  # 序列抽象基类
    )

    # 引入 pandas 类型提示模块
    from pandas._typing import (
        AnyArrayLike,  # 任意数组类型
        Dtype,  # 数据类型
        FillnaOptions,  # 填充选项
        NpDtype,  # NumPy 数据类型
        NumpySorter,  # NumPy 排序器
        NumpyValueArrayLike,  # NumPy 值数组类型
        Self,  # 自身类型
        npt,  # NumPy 类型提示
    )

    # 引入 pandas 核心数据类型扩展数据类型模块
    from pandas.core.dtypes.dtypes import ExtensionDtype

    # 引入 pandas 核心数组日期时间相关模块
    from pandas.core.arrays import (
        DatetimeArray,  # 日期时间数组类
        TimedeltaArray,  # 时间差数组类
    )
    # 引入 pandas 核心数组基类扩展数组模块
    from pandas.core.arrays.base import ExtensionArray

# 定义类型变量，限定为 BaseOffset 的子类
BaseOffsetT = TypeVar("BaseOffsetT", bound=BaseOffset)

# 共享文档关键字参数字典
_shared_doc_kwargs = {
    "klass": "PeriodArray",
}


def _field_accessor(name: str, docstring: str | None = None):
    # 定义字段访问器函数，返回指定名称字段的数组
    def f(self):
        base = self.dtype._dtype_code  # 获取基础数据类型码
        result = get_period_field_arr(name, self.asi8, base)  # 获取字段数组
        return result

    f.__name__ = name  # 设置函数名
    f.__doc__ = docstring  # 设置文档字符串
    return property(f)  # 返回属性访问器


# 错误: 在基类 "NDArrayBacked" 中的 "_concat_same_type" 的定义与基类 "ExtensionArray" 中的定义不兼容
class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):  # type: ignore[misc]
    """
    Pandas ExtensionArray for storing Period data.

    Users should use :func:`~pandas.array` to create new instances.

    Parameters
    ----------
    values : Union[PeriodArray, Series[period], ndarray[int], PeriodIndex]
        The data to store. These should be arrays that can be directly
        converted to ordinals without inference or copy (PeriodArray,
        ndarray[int64]), or a box around such an array (Series[period],
        PeriodIndex).
    dtype : PeriodDtype, optional
        A PeriodDtype instance from which to extract a `freq`. If both
        `freq` and `dtype` are specified, then the frequencies must match.
    copy : bool, default False
        Whether to copy the ordinals before storing.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Period: Represents a period of time.
    PeriodIndex : Immutable Index for period data.
    period_range: Create a fixed-frequency PeriodArray.
    array: Construct a pandas array.

    Notes
    -----
    There are two components to a PeriodArray

    - ordinals : integer ndarray
    - freq : pd.tseries.offsets.Offset

    The values are physically stored as a 1-D ndarray of integers. These are
    called "ordinals" and represent some kind of offset from a base.

    The `freq` indicates the span covered by each element of the array.
    All elements in the PeriodArray have the same `freq`.

    Examples
    --------
    >>> pd.arrays.PeriodArray(pd.PeriodIndex(["2023-01-01", "2023-01-02"], freq="D"))
    <PeriodArray>
    ['2023-01-01', '2023-01-02']
    Length: 2, dtype: period[D]
    """

    # array priority higher than numpy scalars
    __array_priority__ = 1000
    _typ = "periodarray"  # ABCPeriodArray
    _internal_fill_value = np.int64(iNaT)
    _recognized_scalars = (Period,)

    # Check if the input dtype is a PeriodDtype instance
    _is_recognized_dtype = lambda x: isinstance(
        x, PeriodDtype
    )  # check_compatible_with checks freq match

    _infer_matches = ("period",)

    @property
    def _scalar_type(self) -> type[Period]:
        return Period

    # Names others delegate to us
    _other_ops: list[str] = []
    _bool_ops: list[str] = ["is_leap_year"]
    _object_ops: list[str] = ["start_time", "end_time", "freq"]

    # Field operations available for period-like objects
    _field_ops: list[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekofyear",
        "weekday",
        "week",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "qyear",
        "days_in_month",
        "daysinmonth",
    ]

    # Operations applicable to period-like objects
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops

    # Methods available for period-like objects
    _datetimelike_methods: list[str] = ["strftime", "to_timestamp", "asfreq"]

    _dtype: PeriodDtype

    # --------------------------------------------------------------------
    # Constructors
    # 构造函数初始化方法，接受参数 values (值), dtype (数据类型，默认为 None), copy (是否复制数据，默认为 False)，无返回值
    def __init__(self, values, dtype: Dtype | None = None, copy: bool = False) -> None:
        # 如果指定了 dtype，则将其转换为 pandas 的数据类型对象
        if dtype is not None:
            dtype = pandas_dtype(dtype)
            # 如果 dtype 不是 PeriodDtype 类型，则抛出数值错误
            if not isinstance(dtype, PeriodDtype):
                raise ValueError(f"Invalid dtype {dtype} for PeriodArray")

        # 如果 values 是 ABCSeries 类型的实例，则获取其内部值 _values
        if isinstance(values, ABCSeries):
            values = values._values
            # 如果 values 不是当前类的实例，则抛出类型错误
            if not isinstance(values, type(self)):
                raise TypeError("Incorrect dtype")

        # 如果 values 是 ABCPeriodIndex 类型的实例，则获取其内部值 _values
        elif isinstance(values, ABCPeriodIndex):
            values = values._values

        # 如果 values 是当前类的实例，则进行进一步处理
        if isinstance(values, type(self)):
            # 如果指定了 dtype，并且与当前 values 的 dtype 不同，则抛出错误
            if dtype is not None and dtype != values.dtype:
                raise raise_on_incompatible(values, dtype.freq)
            # 获取当前 values 的 _ndarray 和 dtype
            values, dtype = values._ndarray, values.dtype

        # 如果不复制数据，则将 values 转换为 int64 类型的 numpy 数组
        if not copy:
            values = np.asarray(values, dtype="int64")
        else:
            # 否则，复制数据并转换为 int64 类型的 numpy 数组
            values = np.array(values, dtype="int64", copy=copy)
        
        # 如果未指定 dtype，则抛出数值错误
        if dtype is None:
            raise ValueError("dtype is not specified and cannot be inferred")
        
        # 将 dtype 强制转换为 PeriodDtype 类型
        dtype = cast(PeriodDtype, dtype)
        
        # 调用父类 NDArrayBacked 的初始化方法，传入 values 和 dtype
        NDArrayBacked.__init__(self, values, dtype)

    # 类方法 _simple_new，用于创建新的 PeriodArray 对象
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.int64],  # values 参数为 numpy 的 int64 类型数组
        dtype: PeriodDtype,  # dtype 参数为 PeriodDtype 类型
    ) -> Self:
        # 断言 values 是 numpy 数组且数据类型为 i8（即 int64）
        assertion_msg = "Should be numpy array of type i8"
        assert isinstance(values, np.ndarray) and values.dtype == "i8", assertion_msg
        # 调用当前类的构造函数，返回新创建的对象
        return cls(values, dtype=dtype)

    # 类方法 _from_sequence，从序列创建 PeriodArray 对象
    @classmethod
    def _from_sequence(
        cls,
        scalars,  # scalars 参数为输入序列
        *,
        dtype: Dtype | None = None,  # dtype 参数为数据类型，默认为 None
        copy: bool = False,  # copy 参数表示是否复制数据，默认为 False
    ) -> Self:
        # 如果指定了 dtype，则将其转换为 pandas 的数据类型对象
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        
        # 如果指定了 dtype，并且其类型为 PeriodDtype，则获取其频率 freq
        if dtype and isinstance(dtype, PeriodDtype):
            freq = dtype.freq
        else:
            freq = None
        
        # 如果 scalars 是当前类的实例，则验证其 dtype 的频率与当前 freq 是否一致
        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            # 如果需要复制数据，则复制 scalars
            if copy:
                scalars = scalars.copy()
            return scalars
        
        # 将 scalars 转换为对象数组 periods
        periods = np.asarray(scalars, dtype=object)
        
        # 如果 freq 为 None，则从 periods 中提取频率
        freq = freq or libperiod.extract_freq(periods)
        # 从 periods 中提取序数 ordinals
        ordinals = libperiod.extract_ordinals(periods, freq)
        # 创建 PeriodDtype 类型的 dtype 对象
        dtype = PeriodDtype(freq)
        # 调用当前类的构造函数，返回新创建的对象
        return cls(ordinals, dtype=dtype)

    # 类方法 _from_sequence_of_strings，从字符串序列创建 PeriodArray 对象
    @classmethod
    def _from_sequence_of_strings(
        cls, strings,  # strings 参数为字符串序列
        *, dtype: ExtensionDtype,  # dtype 参数为扩展数据类型
        copy: bool = False  # copy 参数表示是否复制数据，默认为 False
    ) -> Self:
        # 调用 _from_sequence 方法，传入字符串序列和 dtype 参数
        return cls._from_sequence(strings, dtype=dtype, copy=copy)
    def _from_datetime64(cls, data, freq, tz=None) -> Self:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
            Array of datetime values to convert into Periods.
        freq : str or Tick
            Frequency string or Tick object defining the period's frequency.
        tz : tzinfo, optional
            Timezone information for the datetime values.

        Returns
        -------
        PeriodArray[freq]
            Constructed PeriodArray based on the input datetime array.
        """
        if isinstance(freq, BaseOffset):
            freq = PeriodDtype(freq)._freqstr
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        dtype = PeriodDtype(freq)
        return cls(data, dtype=dtype)

    @classmethod
    def _generate_range(cls, start, end, periods, freq):
        """
        Generate a range of Periods based on start, end, or number of periods.

        Parameters
        ----------
        start : datetime64-like, optional
            Start of the period range.
        end : datetime64-like, optional
            End of the period range.
        periods : int
            Number of periods to generate.
        freq : str or Tick, optional
            Frequency string or Tick object defining the period's frequency.

        Returns
        -------
        subarr : ndarray of Period
            Array of generated Period objects.
        freq : str
            Frequency string used for the generated periods.
        """
        periods = dtl.validate_periods(periods)

        if freq is not None:
            freq = Period._maybe_convert_freq(freq)

        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError("Not enough parameters to construct Period range")

        return subarr, freq

    @classmethod
    def _from_fields(cls, *, fields: dict, freq) -> Self:
        """
        Construct a PeriodArray from individual fields.

        Parameters
        ----------
        fields : dict
            Dictionary containing fields necessary to construct Periods.
        freq : str or Tick
            Frequency string or Tick object defining the period's frequency.

        Returns
        -------
        PeriodArray[freq]
            Constructed PeriodArray based on the fields provided.
        """
        subarr, freq = _range_from_fields(freq=freq, **fields)
        dtype = PeriodDtype(freq)
        return cls._simple_new(subarr, dtype=dtype)

    # -----------------------------------------------------------------
    # DatetimeLike Interface

    # error: Argument 1 of "_unbox_scalar" is incompatible with supertype
    # "DatetimeLikeArrayMixin"; supertype defines the argument type as
    # "Union[Union[Period, Any, Timedelta], NaTType]"
    def _unbox_scalar(  # type: ignore[override]
        self,
        value: Period | NaTType,
    ) -> np.int64:
        """
        Convert a scalar value to np.int64 based on its type.

        Parameters
        ----------
        value : Period | NaTType
            Scalar value to convert.

        Returns
        -------
        np.int64
            Converted integer value.
        
        Raises
        ------
        ValueError
            If the value is not compatible with Period or NaTType.
        """
        if value is NaT:
            # error: Item "Period" of "Union[Period, NaTType]" has no attribute "value"
            return np.int64(value._value)  # type: ignore[union-attr]
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        """
        Create a Period object from a string representation.

        Parameters
        ----------
        value : str
            String representation of the Period.

        Returns
        -------
        Period
            Constructed Period object.
        """
        return Period(value, freq=self.freq)

    # error: Argument 1 of "_check_compatible_with" is incompatible with
    # supertype "DatetimeLikeArrayMixin"; supertype defines the argument type
    # as "Period | Timestamp | Timedelta | NaTType"
    def _check_compatible_with(self, other: Period | NaTType | PeriodArray) -> None:  # type: ignore[override]
        """
        Check compatibility with another Period, NaTType, or PeriodArray.

        Parameters
        ----------
        other : Period | NaTType | PeriodArray
            Object to check compatibility with.

        Raises
        ------
        AttributeError
            If the objects have mismatched frequencies.
        """
        if other is NaT:
            return
        # error: Item "NaTType" of "Period | NaTType | PeriodArray" has no attribute "freq"
        self._require_matching_freq(other.freq)  # type: ignore[union-attr]

    # --------------------------------------------------------------------
    # Data / Attributes

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        """
        Return the dtype (PeriodDtype) of the PeriodArray.

        Returns
        -------
        PeriodDtype
            dtype of the PeriodArray.
        """
        return self._dtype

    # error: Cannot override writeable attribute with read-only property
    @property  # type: ignore[override]
    def freq(self) -> BaseOffset:
        """
        Return the frequency object for this PeriodArray.
        """
        # 返回此 PeriodArray 的频率对象
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        # 返回与频率相关的字符串表示
        return PeriodDtype(self.freq)._freqstr

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if dtype == "i8":
            # 如果 dtype 是 "i8"，返回整数表示
            return self.asi8
        elif dtype == bool:
            # 如果 dtype 是布尔类型，返回非空值的布尔掩码
            return ~self._isnan

        # 对于非对象类型的 dtype，将自身转换为包含对象的 ndarray
        # 这里会为非对象类型的 dtype 抛出 TypeError 异常
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow

        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

        if type is not None:
            if pyarrow.types.is_integer(type):
                # 如果 type 是整数类型，使用 pyarrow 转换为数组
                return pyarrow.array(self._ndarray, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                # 确保频率与目标 type 相同
                if self.freqstr != type.freq:
                    raise TypeError(
                        "Not supported to convert PeriodArray to array with different "
                        f"'freq' ({self.freqstr} vs {type.freq})"
                    )
            else:
                # 不支持将 PeriodArray 转换为指定的 type 类型
                raise TypeError(
                    f"Not supported to convert PeriodArray to '{type}' type"
                )

        # 创建 ArrowPeriodType 对象，以及基于 int64 的存储数组，并返回扩展数组
        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type="int64")
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    # --------------------------------------------------------------------
    # Vectorized analogues of Period properties

    year = _field_accessor(
        "year",
        """
        The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.year
        Index([2023, 2024, 2025], dtype='int64')
        """,
    )
    month = _field_accessor(
        "month",
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.month
        Index([1, 2, 3], dtype='int64')
        """,
    )
    day = _field_accessor(
        "day",
        """
        The days of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')
        >>> idx.day
        Index([31, 28], dtype='int64')
        """,
    )
    hour = _field_accessor(
        "hour",
        """
        The hour of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00", "2023-01-01 11:00"], freq='h')
        >>> idx.hour
        Index([10, 11], dtype='int64')
        """,
    )
    minute = _field_accessor(
        "minute",
        """
        The minute of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:30:00",
        ...                       "2023-01-01 11:50:00"], freq='min')
        >>> idx.minute
        Index([30, 50], dtype='int64')
        """,
    )
    second = _field_accessor(
        "second",
        """
        The second of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00:30",
        ...                       "2023-01-01 10:00:31"], freq='s')
        >>> idx.second
        Index([30, 31], dtype='int64')
        """,
    )
    weekofyear = _field_accessor(
        "week",
        """
        The week ordinal of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.week  # It can be written `weekofyear`
        Index([5, 9, 13], dtype='int64')
        """,
    )
    # Alias `week` for `weekofyear`
    week = weekofyear
    day_of_week = _field_accessor(
        "day_of_week",
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01", "2023-01-02", "2023-01-03"], freq="D")
        >>> idx.weekday
        Index([6, 0, 1], dtype='int64')
        """,
    )
    # Aliases `dayofweek` and `weekday` for `day_of_week`
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor(
        "day_of_year",
        """
        The ordinal day of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-10", "2023-02-01", "2023-03-01"], freq="D")
        >>> idx.dayofyear
        Index([10, 32, 60], dtype='int64')

        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        >>> idx.dayofyear
        Index([365, 366, 365], dtype='int64')
        """,
    )
    quarter = _field_accessor(
        "quarter",
        """
        The quarter of the date.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.quarter
        Index([1, 1, 1], dtype='int64')
        """,
    )
    qyear = _field_accessor("qyear")
    days_in_month = _field_accessor(
        "days_in_month",
        """
        The number of days in the month.

        Examples
        --------
        For Series:

        >>> period = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')
        >>> s = pd.Series(period)
        >>> s
        0   2020-01
        1   2020-02
        2   2020-03
        dtype: period[M]
        >>> s.dt.days_in_month
        0    31
        1    29
        2    31
        dtype: int64

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.days_in_month   # It can be also entered as `daysinmonth`
        Index([31, 28, 31], dtype='int64')
        """,
    )
    # 将字段访问器 days_in_month 赋值给 daysinmonth 变量
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> npt.NDArray[np.bool_]:
        """
        Logical indicating if the date belongs to a leap year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.is_leap_year
        array([False,  True, False])
        """
        # 调用 isleapyear_arr 函数，判断 self.year 是否为闰年，返回布尔数组
        return isleapyear_arr(np.asarray(self.year))
    # 将 PeriodIndex 转换为 DatetimeArray/Index
    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeArray:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            目标频率。默认为 'D' 表示一周或更长时间，否则为 's'。
        how : {'s', 'e', 'start', 'end'}
            是否使用时间段的开始或结束进行转换。

        Returns
        -------
        DatetimeArray/Index

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq='MS')

        如果索引包含少于三个元素，或者索引的值不是严格单调递增，则不会推断频率：

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01'], dtype='datetime64[ns]', freq=None)

        >>> idx = pd.PeriodIndex(
        ...     ["2023-01", "2023-02", "2023-02", "2023-03"], freq="2M"
        ... )
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq=None)
        """
        # 导入 DatetimeArray 类
        from pandas.core.arrays import DatetimeArray

        # 验证结束别名
        how = libperiod.validate_end_alias(how)

        # 判断是否为结束时间
        end = how == "E"
        if end:
            if freq == "B" or self.freq == "B":
                # 向前滚动以确保落在 B 日期上
                adjust = Timedelta(1, "D") - Timedelta(1, "ns")
                return self.to_timestamp(how="start") + adjust
            else:
                adjust = Timedelta(1, "ns")
                return (self + self.freq).to_timestamp(how="start") - adjust

        if freq is None:
            # 获取频率代码
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            # 转换频率
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        # 调整频率
        new_parr = self.asfreq(freq, how=how)

        # 将 PeriodArray 转换为 Datetime64Array
        new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
        dta = DatetimeArray._from_sequence(new_data)

        if self.freq.name == "B":
            # 在无法区分 BDay 和 Day 的情况下保留 BDay
            diffs = libalgos.unique_deltas(self.asi8)
            if len(diffs) == 1:
                diff = diffs[0]
                if diff == self.dtype._n:
                    dta._freq = self.freq
                elif diff == 1:
                    dta._freq = self.freq.base
                # TODO: 其他情况？
            return dta
        else:
            return dta._with_freq("infer")
    # --------------------------------------------------------------------
    # _box_func 方法定义，接受参数 x，返回 Period 或 NaTType 类型
    def _box_func(self, x) -> Period | NaTType:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    # 使用 _shared_doc_kwargs 和其他参数调用 doc 函数，文档化 asfreq 方法
    @doc(**_shared_doc_kwargs, other="PeriodIndex", other_name="PeriodIndex")
    # asfreq 方法定义，转换当前 PeriodIndex 对象的频率为指定的 freq
    def asfreq(self, freq=None, how: str = "E") -> Self:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {{'E', 'S'}}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        {klass}
            The transformed {klass} with the new frequency.

        See Also
        --------
        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range("2010-01-01", "2015-01-01", freq="Y")
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[Y-DEC]')

        >>> pidx.asfreq("M")
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq("M", how="S")
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """
        # 验证 how 参数是否为有效的结束别名
        how = libperiod.validate_end_alias(how)
        # 如果 freq 是 BaseOffset 的实例并且具有 _period_dtype_code 属性，将 freq 转换为 PeriodDtype 对象的频率字符串
        if isinstance(freq, BaseOffset) and hasattr(freq, "_period_dtype_code"):
            freq = PeriodDtype(freq)._freqstr
        # 尝试将 freq 转换为 Period 频率对象
        freq = Period._maybe_convert_freq(freq)

        # 获取当前 PeriodIndex 的基础 dtype 代码
        base1 = self._dtype._dtype_code
        # 获取 freq 的周期 dtype 代码
        base2 = freq._period_dtype_code

        # 获取当前 PeriodIndex 对象的 asi8 属性
        asi8 = self.asi8
        # self.freq.n 不能为负数或 0
        end = how == "E"
        # 根据 end 的值计算 ordinal
        if end:
            ordinal = asi8 + self.dtype._n - 1
        else:
            ordinal = asi8

        # 使用 period_asfreq_arr 函数根据 ordinal、base1、base2 和 end 创建新数据
        new_data = period_asfreq_arr(ordinal, base1, base2, end)

        # 如果当前 PeriodIndex 对象包含缺失值，将其替换为 iNaT
        if self._hasna:
            new_data[self._isnan] = iNaT

        # 创建 PeriodDtype 对象，并返回一个新的 {klass} 对象，用新的 dtype
        dtype = PeriodDtype(freq)
        return type(self)(new_data, dtype=dtype)

    # ------------------------------------------------------------------
    # _formatter 方法定义，接受 boxed 参数，返回相应的格式化函数
    def _formatter(self, boxed: bool = False) -> Callable[[object], str]:
        if boxed:
            return str
        return "'{}'".format

    # _format_native_types 方法定义，接受 na_rep、date_format 和其他 kwargs 参数，返回格式化后的字符串
    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        """
        actually format my specific types
        """
        # 调用底层库函数 `libperiod.period_array_strftime`，将当前对象的整数表示 (`self.asi8`) 格式化为指定类型的字符串数组
        return libperiod.period_array_strftime(
            self.asi8, self.dtype._dtype_code, na_rep, date_format
        )

    # ------------------------------------------------------------------

    def astype(self, dtype, copy: bool = True):
        # We handle Period[T] -> Period[U]
        # Our parent handles everything else.
        # 将输入的数据类型转换为 Pandas 的数据类型
        dtype = pandas_dtype(dtype)
        # 如果目标数据类型与当前对象的数据类型相同
        if dtype == self._dtype:
            # 如果不需要复制，直接返回当前对象的引用
            if not copy:
                return self
            else:
                # 如果需要复制，返回当前对象的副本
                return self.copy()
        # 如果目标数据类型是 PeriodDtype 类型，则调用 `self.asfreq` 方法将当前对象转换为该类型
        if isinstance(dtype, PeriodDtype):
            return self.asfreq(dtype.freq)

        # 如果目标数据类型是 numpy 的日期时间类型 "M" 或者 DatetimeTZDtype 类型
        if lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            # GH#45038 匹配 PeriodIndex 的行为，将当前 Period 对象转换为 Timestamp，并进行时区本地化和单位转换
            tz = getattr(dtype, "tz", None)
            unit = dtl.dtype_to_unit(dtype)
            return self.to_timestamp().tz_localize(tz).as_unit(unit)

        # 对于其他情况，调用父类的 `astype` 方法进行转换
        return super().astype(dtype, copy=copy)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # 将输入的值 `value` 校验并转换为 "M8[ns]" 类型的 NumPy 数组
        npvalue = self._validate_setitem_value(value).view("M8[ns]")

        # 将当前对象 `_ndarray` 属性视图转换为 "M8[ns]" 类型，并调用其 `searchsorted` 方法查找 `npvalue` 在数组中的位置
        # 返回值为整数数组或整数，表示 `npvalue` 在 `_ndarray` 中的位置或插入点
        m8arr = self._ndarray.view("M8[ns]")
        return m8arr.searchsorted(npvalue, side=side, sorter=sorter)

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # 将当前对象视图转换为 "M8[ns]" 类型，以便在 `core.missing` 模块中被视为时间对象
        dta = self.view("M8[ns]")
        # 调用 `_pad_or_backfill` 方法进行填充或回填操作，返回填充后的结果
        result = dta._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )
        # 如果需要复制结果，则将结果视图转换为当前对象的数据类型后返回；否则直接返回当前对象
        if copy:
            return cast("Self", result.view(self.dtype))
        else:
            return self

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _addsub_int_array_or_scalar(
        self, other: np.ndarray | int, op: Callable[[Any, Any], Any]
    ) -> Self:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[int64] or int
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
        # 断言操作符 `op` 必须是加法或减法
        assert op in [operator.add, operator.sub]
        # 如果操作符是减法，则将 `other` 取负数
        if op is operator.sub:
            other = -other
        # 调用 `add_overflowsafe` 函数进行安全的加法操作，将结果封装成当前对象的类型并返回
        res_values = add_overflowsafe(self.asi8, np.asarray(other, dtype="i8"))
        return type(self)(res_values, dtype=self.dtype)
    # 确保参数 `other` 不是 Tick 类的实例，否则引发断言错误
    def _add_offset(self, other: BaseOffset):
        assert not isinstance(other, Tick)

        # 调用 `_require_matching_freq` 方法，确保当前对象与 `other` 频率匹配，基于基础频率进行检查
        self._require_matching_freq(other, base=True)

        # 调用 `_addsub_int_array_or_scalar` 方法，将 `other.n` 与当前对象进行加法运算
        return self._addsub_int_array_or_scalar(other.n, operator.add)

    # TODO: 可以与 `Period._add_timedeltalike_scalar` 方法进行代码去重
    def _add_timedeltalike_scalar(self, other):
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
        # 如果当前对象的频率不是 Tick，则不能将 timedelta-like 对象添加到非 Tick 类型的 PeriodArray 中
        if not isinstance(self.freq, Tick):
            raise raise_on_incompatible(self, other)

        # 如果 `other` 是空值（如 np.timedelta64("NaT")），则调用父类的 `_add_timedeltalike_scalar` 方法处理
        if isna(other):
            return super()._add_timedeltalike_scalar(other)

        # 将 `other` 转换为 numpy 的 timedelta 对象，然后转换为微秒级别的数组
        td = np.asarray(Timedelta(other).asm8)

        # 调用 `_add_timedelta_arraylike` 方法，将时间增量数组添加到当前对象中
        return self._add_timedelta_arraylike(td)

    # 将时间增量数组或 TimedeltaArray 添加到当前对象中
    def _add_timedelta_arraylike(
        self, other: TimedeltaArray | npt.NDArray[np.timedelta64]
    ) -> Self:
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        PeriodArray
        """
        # 如果当前对象的 dtype 不是 Tick 类型的时间间隔，则抛出类型错误
        if not self.dtype._is_tick_like():
            raise TypeError(
                f"Cannot add or subtract timedelta64[ns] dtype from {self.dtype}"
            )

        # 创建一个 `m8` 类型的 numpy dtype，其单位是当前对象的时间单位
        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")

        # 尝试将 `other` 转换为指定的 dtype，如果转换失败，则抛出 ValueError 异常
        try:
            delta = astype_overflowsafe(
                np.asarray(other), dtype=dtype, copy=False, round_ok=False
            )
        except ValueError as err:
            # 如果转换失败，可能是因为单位不匹配，抛出 IncompatibleFrequency 异常
            raise IncompatibleFrequency(
                "Cannot add/subtract timedelta-like from PeriodArray that is "
                "not an integer multiple of the PeriodArray's freq."
            ) from err

        # 对当前对象的整数表示（asi8）和转换后的时间增量数组进行安全加法运算
        res_values = add_overflowsafe(self.asi8, np.asarray(delta.view("i8")))

        # 返回与当前对象类型相同的 PeriodArray 对象，结果使用当前对象的 dtype
        return type(self)(res_values, dtype=self.dtype)
    def _check_timedeltalike_freq_compat(self, other):
        """
        确保 timedelta 类型和频率兼容性检查函数。

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex
            其他可能的时间增量或数组。

        Returns
        -------
        multiple : int or ndarray[int64]
            如果操作有效，则返回整数倍数；否则引发异常。

        Raises
        ------
        IncompatibleFrequency
            如果操作的频率不兼容，则引发此异常。
        """
        assert self.dtype._is_tick_like()  # 被调用函数检查 self 的类型是否为 Tick 类型

        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")  # 定义一个 Numpy dtype 对象，表示 timedelta 的单位

        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            td = np.asarray(Timedelta(other).asm8)  # 如果 other 是 timedelta 或者 np.timedelta64 或 Tick 类型，转换成 Numpy 数组
        else:
            td = np.asarray(other)  # 其他情况，直接转换成 Numpy 数组

        try:
            delta = astype_overflowsafe(td, dtype=dtype, copy=False, round_ok=False)
            # 尝试将 td 转换为指定 dtype 的 Numpy 数组，避免溢出，不允许四舍五入
        except ValueError as err:
            raise raise_on_incompatible(self, other) from err
            # 如果转换过程中出现 ValueError，则引发不兼容异常，传递原始错误信息

        delta = delta.view("i8")  # 将 delta 视图转换为 64 位整数类型
        return lib.item_from_zerodim(delta)
        # 返回 delta 的零维数组中的单个元素
# 渲染一致的错误消息以引发 IncompatibleFrequency 异常的辅助函数
def raise_on_incompatible(left, right) -> IncompatibleFrequency:
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
        左侧对象，期间数组
    right : None, DateOffset, Period, ndarray, or timedelta-like
        右侧对象，可以是 None、DateOffset、Period、ndarray 或类似 timedelta 的对象

    Returns
    -------
    IncompatibleFrequency
        被调用者将要引发的异常对象。
    """
    # GH#24283 error message format depends on whether right is scalar
    # GH#24283 的错误消息格式取决于 right 是否为标量
    if isinstance(right, (np.ndarray, ABCTimedeltaArray)) or right is None:
        other_freq = None
    elif isinstance(right, BaseOffset):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
            )
            other_freq = PeriodDtype(right)._freqstr
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period)):
        other_freq = right.freqstr
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr

    own_freq = PeriodDtype(left.freq)._freqstr
    # 使用 DIFFERENT_FREQ 的格式化字符串生成消息
    msg = DIFFERENT_FREQ.format(
        cls=type(left).__name__, own_freq=own_freq, other_freq=other_freq
    )
    return IncompatibleFrequency(msg)


# -------------------------------------------------------------------
# Constructor Helpers


def period_array(
    data: Sequence[Period | str | None] | AnyArrayLike,
    freq: str | Tick | BaseOffset | None = None,
    copy: bool = False,
) -> PeriodArray:
    """
    Construct a new PeriodArray from a sequence of Period scalars.

    Parameters
    ----------
    data : Sequence of Period objects
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : str, Tick, or Offset
        The frequency of every element of the array. This can be specified
        to avoid inferring the `freq` from `data`.
    copy : bool, default False
        Whether to ensure a copy of the data is made.

    Returns
    -------
    PeriodArray
        A new PeriodArray object constructed from the provided data.

    See Also
    --------
    PeriodArray
    pandas.PeriodIndex

    Examples
    --------
    >>> period_array([pd.Period("2017", freq="Y"), pd.Period("2018", freq="Y")])
    <PeriodArray>
    ['2017', '2018']
    Length: 2, dtype: period[Y-DEC]

    >>> period_array([pd.Period("2017", freq="Y"), pd.Period("2018", freq="Y"), pd.NaT])
    <PeriodArray>
    ['2017', '2018', 'NaT']
    Length: 3, dtype: period[Y-DEC]

    Integers that look like years are handled

    >>> period_array([2000, 2001, 2002], freq="D")
    <PeriodArray>
    ['2000-01-01', '2001-01-01', '2002-01-01']
    Length: 3, dtype: period[D]

    Datetime-like strings may also be passed

    >>> period_array(["2000-Q1", "2000-Q2", "2000-Q3", "2000-Q4"], freq="Q")
    <PeriodArray>
    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']
    Length: 4, dtype: period[Q-DEC]
    """
    data_dtype = getattr(data, "dtype", None)
    # 如果 data_dtype 是日期时间类型 "M"，则返回一个 PeriodArray 对象
    if lib.is_np_dtype(data_dtype, "M"):
        return PeriodArray._from_datetime64(data, freq)
    
    # 如果 data_dtype 是 PeriodDtype 的实例，则使用 data 构造一个 PeriodArray 对象
    if isinstance(data_dtype, PeriodDtype):
        out = PeriodArray(data)
        # 如果 freq 不为 None，并且与 data_dtype 中的频率相同，则直接返回 out
        if freq is not None:
            if freq == data_dtype.freq:
                return out
            # 否则，将 out 转换为指定频率的 PeriodArray 对象后返回
            return out.asfreq(freq)
        # 如果 freq 为 None，则直接返回 out
        return out

    # 如果 data 不是 ndarray、list、tuple 或 ABCSeries 的实例，则转换为列表
    if not isinstance(data, (np.ndarray, list, tuple, ABCSeries)):
        data = list(data)

    # 将 data 转换为 ndarray 类型
    arrdata = np.asarray(data)

    # 如果指定了 freq，则使用 PeriodDtype(freq) 构造 dtype，否则设为 None
    dtype: PeriodDtype | None
    if freq:
        dtype = PeriodDtype(freq)
    else:
        dtype = None

    # 如果 arrdata 的 dtype 的 kind 是 "f" 并且长度大于 0，则抛出 TypeError
    if arrdata.dtype.kind == "f" and len(arrdata) > 0:
        raise TypeError("PeriodIndex does not allow floating point in construction")

    # 如果 arrdata 的 dtype 的 kind 是 "iu"，则将其转换为 int64 类型数组 arr
    if arrdata.dtype.kind in "iu":
        arr = arrdata.astype(np.int64, copy=False)
        # 使用 libperiod.from_ordinals(arr, freq) 构造 ordinals
        ordinals = libperiod.from_ordinals(arr, freq)  # type: ignore[arg-type]
        # 返回一个 PeriodArray 对象，使用 ordinals 和指定的 dtype
        return PeriodArray(ordinals, dtype=dtype)

    # 将 arrdata 转换为确保是 object 类型的数组 data
    data = ensure_object(arrdata)
    # 如果 freq 为 None，则从 data 中提取频率
    if freq is None:
        freq = libperiod.extract_freq(data)
    # 构造 PeriodDtype(freq) 作为 dtype
    dtype = PeriodDtype(freq)
    # 返回一个 PeriodArray 对象，使用 data 和指定的 dtype
    return PeriodArray._from_sequence(data, dtype=dtype)
@overload
# 函数重载装饰器，定义了一种类型重载的方式
def validate_dtype_freq(dtype, freq: BaseOffsetT) -> BaseOffsetT: ...


@overload
# 函数重载装饰器，定义了另一种类型重载的方式
def validate_dtype_freq(dtype, freq: timedelta | str | None) -> BaseOffset: ...


def validate_dtype_freq(
    dtype, freq: BaseOffsetT | BaseOffset | timedelta | str | None
) -> BaseOffsetT:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
        数据类型，可能是任何类型
    freq : DateOffset or None
        时间偏移或者为空

    Returns
    -------
    freq : DateOffset
        返回时间偏移

    Raises
    ------
    ValueError : non-period dtype
        如果数据类型不是周期性的，抛出值错误异常
    IncompatibleFrequency : mismatch between dtype and freq
        如果指定的频率与数据类型的频率不匹配，抛出不兼容频率异常
    """
    if freq is not None:
        freq = to_offset(freq, is_period=True)

    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if not isinstance(dtype, PeriodDtype):
            raise ValueError("dtype must be PeriodDtype")
        if freq is None:
            freq = dtype.freq
        elif freq != dtype.freq:
            raise IncompatibleFrequency("specified freq and dtype are different")
    # error: Incompatible return value type (got "Union[BaseOffset, Any, None]",
    # expected "BaseOffset")
    return freq  # type: ignore[return-value]


def dt64arr_to_periodarr(
    data, freq, tz=None
) -> tuple[npt.NDArray[np.int64], BaseOffset]:
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]
        包含日期时间的数据结构，可以是 Series、DatetimeIndex 或者 ndarray
    freq : Optional[Union[str, Tick]]
        如果 data 是 DatetimeIndex 或 Series，则必须与 data 上的 freq 匹配
    tz : Optional[tzinfo]
        时区信息

    Returns
    -------
    ordinals : ndarray[int64]
        整数数组，表示时间周期的序数
    freq : Tick
        从 Series 或 DatetimeIndex 中提取的频率

    """
    if not isinstance(data.dtype, np.dtype) or data.dtype.kind != "M":
        raise ValueError(f"Wrong dtype: {data.dtype}")

    if freq is None:
        if isinstance(data, ABCIndex):
            data, freq = data._values, data.freq
        elif isinstance(data, ABCSeries):
            data, freq = data._values, data.dt.freq

    elif isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values

    reso = get_unit_from_dtype(data.dtype)
    freq = Period._maybe_convert_freq(freq)
    base = freq._period_dtype_code
    return c_dt64arr_to_periodarr(data.view("i8"), base, tz, reso=reso), freq


def _get_ordinal_range(start, end, periods, freq, mult: int = 1):
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )

    if freq is not None:
        freq = to_offset(freq, is_period=True)
        mult = freq.n

    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)

    is_start_per = isinstance(start, Period)
    # 检查 `end` 是否为 `Period` 类型的实例
    is_end_per = isinstance(end, Period)
    
    # 如果 `start` 和 `end` 都是 `Period` 类型的实例，并且它们的频率不同，则抛出值错误异常
    if is_start_per and is_end_per and start.freq != end.freq:
        raise ValueError("start and end must have same freq")
    
    # 如果 `start` 或 `end` 是 NaT（Not a Time），则抛出值错误异常
    if start is NaT or end is NaT:
        raise ValueError("start and end must not be NaT")
    
    # 如果未提供频率 `freq` 参数，则根据情况推断频率
    if freq is None:
        # 如果 `start` 是 `Period` 类型的实例，则使用其频率作为 `freq`
        if is_start_per:
            freq = start.freq
        # 如果 `end` 是 `Period` 类型的实例，则使用其频率作为 `freq`
        elif is_end_per:
            freq = end.freq
        else:  # pragma: no cover
            # 如果无法从 `start` 和 `end` 推断出频率，则抛出值错误异常
            raise ValueError("Could not infer freq from start/end")
    
        # 计算 `mult` 为频率 `freq` 的周期数
        mult = freq.n
    
    # 如果指定了 `periods` 参数，则将其乘以 `mult` 得到新的周期数
    if periods is not None:
        periods = periods * mult
        # 如果 `start` 未指定，则生成数据以填充从 `end` 往前推 `periods` 个周期的整数数组
        if start is None:
            data = np.arange(
                end.ordinal - periods + mult, end.ordinal + 1, mult, dtype=np.int64
            )
        else:
            # 否则，生成数据以填充从 `start` 开始 `periods` 个周期的整数数组
            data = np.arange(
                start.ordinal, start.ordinal + periods, mult, dtype=np.int64
            )
    else:
        # 如果未指定 `periods` 参数，则生成数据以填充从 `start` 到 `end` 的整数数组
        data = np.arange(start.ordinal, end.ordinal + 1, mult, dtype=np.int64)
    
    # 返回生成的整数数组 `data` 和推断得到的频率 `freq`
    return data, freq
# 根据给定的年、月、季度、日、时、分、秒以及频率，生成时间范围的数组和基础偏移对象
def _range_from_fields(
    year=None,
    month=None,
    quarter=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    freq=None,
) -> tuple[np.ndarray, BaseOffset]:
    # 如果小时未指定，默认为0
    if hour is None:
        hour = 0
    # 如果分钟未指定，默认为0
    if minute is None:
        minute = 0
    # 如果秒未指定，默认为0
    if second is None:
        second = 0
    # 如果日未指定，默认为1
    if day is None:
        day = 1

    # 存储时间周期的序数
    ordinals = []

    # 如果指定了季度
    if quarter is not None:
        # 如果频率未指定，默认为季度频率
        if freq is None:
            freq = to_offset("Q", is_period=True)
            base = FreqGroup.FR_QTR.value
        else:
            freq = to_offset(freq, is_period=True)
            base = libperiod.freq_to_dtype_code(freq)
            # 检查频率是否与季度频率相匹配
            if base != FreqGroup.FR_QTR.value:
                raise AssertionError("base must equal FR_QTR")

        # 获取频率字符串
        freqstr = freq.freqstr
        # 生成年和季度的数组
        year, quarter = _make_field_arrays(year, quarter)
        # 遍历年和季度，转换为日历年和月份
        for y, q in zip(year, quarter):
            calendar_year, calendar_month = parsing.quarter_to_myear(y, q, freqstr)
            # 计算序数值并添加到列表中
            val = libperiod.period_ordinal(
                calendar_year, calendar_month, 1, 1, 1, 1, 0, 0, base
            )
            ordinals.append(val)
    else:
        # 如果未指定季度，根据指定的频率生成时间数组
        freq = to_offset(freq, is_period=True)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        # 遍历生成的时间数组，计算并添加序数值
        for y, mth, d, h, mn, s in zip(*arrays):
            ordinals.append(libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base))

    # 将序数列表转换为 numpy 数组，并返回结果以及频率对象
    return np.array(ordinals, dtype=np.int64), freq


# 生成字段数组，确保长度匹配，用于时间范围计算
def _make_field_arrays(*fields) -> list[np.ndarray]:
    length = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if length is not None and len(x) != length:
                raise ValueError("Mismatched Period array lengths")
            if length is None:
                length = len(x)

    # 将每个字段转换为 numpy 数组或重复值数组（如果不是数组）
    return [
        np.asarray(x)
        if isinstance(x, (np.ndarray, list, ABCSeries))
        else np.repeat(x, length)  # type: ignore[arg-type]
        for x in fields
    ]
```