# `D:\src\scipysrc\pandas\pandas\core\indexes\period.py`

```
# 从未来导入注解支持，用于确保向后兼容性
from __future__ import annotations

# 导入 datetime 模块中的 datetime 和 timedelta 类
from datetime import (
    datetime,
    timedelta,
)

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入 NumPy 库，命名为 np
import numpy as np

# 导入 pandas 库中的私有模块
from pandas._libs import index as libindex
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    Period,
    Resolution,
    Tick,
)

# 导入 pandas 库中的时间相关数据类型
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR

# 导入 pandas 库中的装饰器和缓存相关工具
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

# 导入 pandas 库中的常用数据类型检查和时间数据类型
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype

# 导入 pandas 库中的 PeriodArray 相关功能
from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
    raise_on_incompatible,
    validate_dtype_freq,
)

# 导入 pandas 库中的公共工具模块和基础索引模块
import pandas.core.common as com
import pandas.core.indexes.base as ibase

# 导入 pandas 库中的索引相关功能
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    Index,
)

# 导入 pandas 库中的索引扩展相关功能
from pandas.core.indexes.extension import inherit_names

# 如果支持类型检查，导入额外的类型
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )

# 定义私有变量，用于文档生成的设置
_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({"target_klass": "PeriodIndex or list of Periods"})
_shared_doc_kwargs = {
    "klass": "PeriodArray",
}

# --- Period index sketch

# 定义一个辅助函数，用于创建新的 PeriodIndex 对象
def _new_PeriodIndex(cls, **d):
    # 用于反序列化的辅助功能，处理数据的恢复
    values = d.pop("data")
    if values.dtype == "int64":
        freq = d.pop("freq", None)
        # 如果数据类型是 int64，创建一个 PeriodDtype 对象
        dtype = PeriodDtype(freq)
        # 使用 PeriodArray 包装数据，构造新的 PeriodIndex 对象
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        # 否则直接创建 PeriodIndex 对象
        return cls(values, **d)

# 应用继承名称的装饰器，继承 PeriodArray 中的方法和属性
@inherit_names(
    ["strftime", "start_time", "end_time"] + PeriodArray._field_ops,
    PeriodArray,
    wrap=True,
)
# 继承 is_leap_year 方法的装饰器，继承 PeriodArray 中的属性和方法
@inherit_names(["is_leap_year"], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    dtype : str or PeriodDtype, default None
        A dtype from which to extract a freq.
    copy : bool
        Make a copy of input ndarray.
    name : str, default None
        Name of the resulting PeriodIndex.

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------

    """
    asfreq
    strftime
    to_timestamp
    from_fields
    from_ordinals



    # periodindex 对象提供的方法和属性

    Raises
    ------
    ValueError
        Passing the parameter data as a list without specifying either freq or
        dtype will raise a ValueError: "freq not specified and cannot be inferred"

    # 如果将参数 data 作为列表传递而不指定 freq 或 dtype，则会引发 ValueError 异常："未指定 freq 且无法推断"

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    # 参见其他相关对象：Index - 基本的 pandas Index 类型；Period - 表示时间段；DatetimeIndex - 具有 datetime64 数据的索引；TimedeltaIndex - 具有 timedelta64 数据的索引；period_range - 创建固定频率的 PeriodIndex。

    Examples
    --------
    >>> idx = pd.PeriodIndex(data=["2000Q1", "2002Q3"], freq="Q")
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """

    _typ = "periodindex"
    # 定义类型标识符为 "periodindex"

    _data: PeriodArray
    # PeriodArray 类型的数据

    freq: BaseOffset
    # BaseOffset 类型的频率

    dtype: PeriodDtype
    # PeriodDtype 类型的数据类型

    _data_cls = PeriodArray
    # 使用 PeriodArray 类

    _supports_partial_string_indexing = True
    # 支持部分字符串索引

    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]:
        return libindex.PeriodEngine
    # 返回 libindex.PeriodEngine 类型的引擎类型

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        # for compat with DatetimeIndex
        return self.dtype._resolution_obj
    # 返回与 DatetimeIndex 兼容的 _resolution_obj 对象

    # --------------------------------------------------------------------
    # methods that dispatch to array and wrap result in Index
    # These are defined here instead of via inherit_names for mypy

    @doc(
        PeriodArray.asfreq,
        other="arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs,
    )
    def asfreq(self, freq=None, how: str = "E") -> Self:
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)
    # 调用 PeriodArray 的 asfreq 方法，返回新的 PeriodIndex 对象

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeIndex:
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)
    # 调用 PeriodArray 的 to_timestamp 方法，返回新的 DatetimeIndex 对象

    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Index:
        return Index(self._data.hour, name=self.name)
    # 返回包含小时信息的 Index 对象

    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Index:
        return Index(self._data.minute, name=self.name)
    # 返回包含分钟信息的 Index 对象

    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Index:
        return Index(self._data.second, name=self.name)
    # 返回包含秒钟信息的 Index 对象

    # ------------------------------------------------------------------------
    # Index Constructors

    def __new__(
        cls,
        data=None,
        freq=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        refs = None
        # 如果不需要拷贝数据且数据是 Index 或者 ABCSeries 的实例，则保留数据的引用
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        # 提取名称，如果可能的话
        name = maybe_extract_name(name, data, cls)

        # 验证并返回有效的数据类型和频率
        freq = validate_dtype_freq(dtype, freq)

        # PeriodIndex 允许在 PeriodIndex(period_index, freq=different) 的情况下使用
        # 但在 PeriodArray 中不鼓励这种行为

        if freq and isinstance(data, cls) and data.freq != freq:
            # 如果指定了频率，并且数据是当前类的实例，并且数据的频率与指定的频率不同
            # 尝试按照指定的频率重新取样数据
            data = data.asfreq(freq)

        # 创建 PeriodArray 对象，用于存储数据
        data = period_array(data=data, freq=freq)

        if copy:
            # 如果需要拷贝数据，则进行数据的深拷贝
            data = data.copy()

        # 使用类方法 _simple_new 创建新的实例并返回
        return cls._simple_new(data, name=name, refs=refs)

    @classmethod
    def from_fields(
        cls,
        *,
        year=None,
        quarter=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        freq=None,
    ) -> Self:
        """
        从字段 (year, quarter, month, day 等) 构造 PeriodIndex 对象。

        Parameters
        ----------
        year : int, array, or Series, default None
            年份信息，可以是整数、数组或者序列，默认为 None
        quarter : int, array, or Series, default None
            季度信息，可以是整数、数组或者序列，默认为 None
        month : int, array, or Series, default None
            月份信息，可以是整数、数组或者序列，默认为 None
        day : int, array, or Series, default None
            日信息，可以是整数、数组或者序列，默认为 None
        hour : int, array, or Series, default None
            小时信息，可以是整数、数组或者序列，默认为 None
        minute : int, array, or Series, default None
            分钟信息，可以是整数、数组或者序列，默认为 None
        second : int, array, or Series, default None
            秒信息，可以是整数、数组或者序列，默认为 None
        freq : str or period object, optional
            字符串或者期间对象，表示频率，可选

        Returns
        -------
        PeriodIndex
            返回一个 PeriodIndex 对象

        Examples
        --------
        >>> idx = pd.PeriodIndex.from_fields(year=[2000, 2002], quarter=[1, 3])
        >>> idx
        PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
        """
        # 将所有非 None 的字段存储在字典中
        fields = {
            "year": year,
            "quarter": quarter,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        
        # 使用 PeriodArray 类方法 _from_fields 创建 PeriodArray 对象
        arr = PeriodArray._from_fields(fields=fields, freq=freq)
        
        # 使用类方法 _simple_new 创建新的实例并返回
        return cls._simple_new(arr)
    def from_ordinals(cls, ordinals, *, freq, name=None) -> Self:
        """
        从序数构造一个 PeriodIndex。

        Parameters
        ----------
        ordinals : array-like of int
            从格里高利历纪元开始的周期偏移量。
        freq : str or period object
            pandas周期字符串或相应对象之一。
        name : str, default None
            结果 PeriodIndex 的名称。

        Returns
        -------
        PeriodIndex

        Examples
        --------
        >>> idx = pd.PeriodIndex.from_ordinals([-1, 0, 1], freq="Q")
        >>> idx
        PeriodIndex(['1969Q4', '1970Q1', '1970Q2'], dtype='period[Q-DEC]')
        """
        # 将 ordinals 转换为 int64 类型的 numpy 数组
        ordinals = np.asarray(ordinals, dtype=np.int64)
        # 创建指定频率的 PeriodDtype 对象
        dtype = PeriodDtype(freq)
        # 使用 ordinals 和 dtype 创建 PeriodArray 对象
        data = PeriodArray._simple_new(ordinals, dtype=dtype)
        # 使用类方法 _simple_new 创建并返回 PeriodIndex 对象
        return cls._simple_new(data, name=name)

    # ------------------------------------------------------------------------
    # Data

    @property
    def values(self) -> npt.NDArray[np.object_]:
        # 将当前对象转换为包含对象的 numpy 数组并返回
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other) -> int | npt.NDArray[np.int64]:
        """
        将类似于时间差的输入转换为 self.freq 的整数倍

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : 如果输入无法写成 self.freq 的整数倍，则引发异常。
            注意，IncompatibleFrequency 是 ValueError 的子类。
        """
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                # 如果 self.freq 是 Tick 类型，则调用 _check_timedeltalike_freq_compat 方法
                # 如果不兼容会抛出异常
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n

            # 如果 base 不同，抛出异常
            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            assert isinstance(other, int)
            return other

        # 输入不包含 freq 时抛出异常
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        给定的 dtype 是否可与当前对象的 dtype 进行比较？
        """
        # 返回当前对象的 dtype 是否与给定的 dtype 相同
        return self.dtype == dtype

    # ------------------------------------------------------------------------
    # Index Methods
    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> np.ndarray:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        # 如果 `where` 参数是 DatetimeIndex 类型，则转换为 PeriodIndex 类型
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        # 如果 `where` 参数不是 PeriodIndex 类型，则抛出类型错误异常
        elif not isinstance(where, PeriodIndex):
            raise TypeError("asof_locs `where` must be DatetimeIndex or PeriodIndex")

        # 调用父类方法 super().asof_locs()，传入转换后的 where 和 mask 参数
        return super().asof_locs(where, mask)

    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        # 如果 PeriodIndex 长度为 0，则直接返回 True
        if len(self) == 0:
            return True
        # 如果 PeriodIndex 不是单调递增的，则抛出值错误异常
        if not self.is_monotonic_increasing:
            raise ValueError("Index is not monotonic")
        # 获取 PeriodIndex 的整数表示 values
        values = self.asi8
        # 返回值表达式，判断是否所有相邻元素的差小于 2
        return bool(((values[1:] - values[:-1]) < 2).all())

    @property
    def inferred_type(self) -> str:
        # 返回固定字符串 "period"，指示数据类型是周期类型
        return "period"

    # ------------------------------------------------------------------------
    # Indexing Methods

    def _convert_tolerance(self, tolerance, target):
        # 返回的容差 tolerance 必须与 dtype/units 一致，以使
        # `|self._get_engine_target() - target._engine_target()| <= tolerance`
        # 有意义。由于 PeriodIndex 返回 int64 作为 engine_target，因此可能需要将
        # timedelta64 的容差转换为 int64 类型。
        tolerance = super()._convert_tolerance(tolerance, target)

        # 如果 self 的 dtype 与 target 的 dtype 相同，则将容差 tolerance 转换为 int64 类型
        if self.dtype == target.dtype:
            tolerance = self._maybe_convert_timedelta(tolerance)

        return tolerance
    # 获取请求标签的整数位置。

    """
    Get integer location for requested label.

    Parameters
    ----------
    key : Period, NaT, str, or datetime
        String or datetime key must be parsable as Period.

    Returns
    -------
    loc : int or ndarray[int64]

    Raises
    ------
    KeyError
        Key is not present in the index.
    TypeError
        If key is listlike or otherwise not hashable.
    """

    orig_key = key  # 保存原始的 key 值

    self._check_indexing_error(key)  # 检查索引错误

    if is_valid_na_for_dtype(key, self.dtype):
        key = NaT  # 如果 key 是 NaT 类型，将 key 设为 NaT

    elif isinstance(key, str):
        try:
            parsed, reso = self._parse_with_reso(key)  # 尝试解析字符串 key
        except ValueError as err:
            # A string with invalid format
            raise KeyError(f"Cannot interpret '{key}' as period") from err

        if self._can_partial_date_slice(reso):
            try:
                return self._partial_date_slice(reso, parsed)  # 执行部分日期切片
            except KeyError as err:
                raise KeyError(key) from err

        if reso == self._resolution_obj:
            # 如果 reso 等于 self._resolution_obj，则通过 _cast_partial_indexing_scalar 处理索引标量
            key = self._cast_partial_indexing_scalar(parsed)
        else:
            raise KeyError(key)  # 否则抛出 KeyError

    elif isinstance(key, Period):
        self._disallow_mismatched_indexing(key)  # 检查不允许的索引方式

    elif isinstance(key, datetime):
        key = self._cast_partial_indexing_scalar(key)  # 将 datetime 转换为 Period

    else:
        # 如果 key 是其他类型（例如整数），则抛出 KeyError
        raise KeyError(key)

    try:
        return Index.get_loc(self, key)  # 调用父类 Index 的 get_loc 方法获取位置
    except KeyError as err:
        raise KeyError(orig_key) from err  # 如果获取位置失败，则抛出原始 key 的 KeyError

def _disallow_mismatched_indexing(self, key: Period) -> None:
    if key._dtype != self.dtype:
        raise KeyError(key)  # 如果 key 的数据类型与当前对象的 dtype 不匹配，则抛出 KeyError

def _cast_partial_indexing_scalar(self, label: datetime) -> Period:
    try:
        period = Period(label, freq=self.freq)  # 尝试构造 Period 对象
    except ValueError as err:
        # we cannot construct the Period
        raise KeyError(label) from err  # 如果构造失败，则抛出 KeyError
    return period  # 返回构造的 Period 对象

@doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
def _maybe_cast_slice_bound(self, label, side: str):
    if isinstance(label, datetime):
        label = self._cast_partial_indexing_scalar(label)  # 如果 label 是 datetime，则将其转换为 Period

    return super()._maybe_cast_slice_bound(label, side)  # 调用父类的 _maybe_cast_slice_bound 方法处理标签

def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime):
    freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
    iv = Period(parsed, freq=freq)  # 使用解析后的 datetime 创建 Period 对象
    return (iv.asfreq(self.freq, how="start"), iv.asfreq(self.freq, how="end"))  # 返回起始和结束时间戳

@doc(DatetimeIndexOpsMixin.shift)
    # 定义一个方法 `shift`，用于移动时间序列的时间点
    def shift(self, periods: int = 1, freq=None) -> Self:
        # 如果用户指定了 `freq` 参数，抛出类型错误异常，因为不支持这个参数
        if freq is not None:
            raise TypeError(
                f"`freq` argument is not supported for {type(self).__name__}.shift"
            )
        # 返回移动后的时间序列，移动的周期数由 `periods` 参数指定
        return self + periods
# 返回一个固定频率的 PeriodIndex 对象。
#
# 默认频率为日历日（"D"）。

def period_range(
    start=None,  # 生成周期的左边界，可以是字符串、日期时间、日期、Timestamp对象或类似周期的对象，默认为None
    end=None,    # 生成周期的右边界，可以是字符串、日期时间、日期、Timestamp对象或类似周期的对象，默认为None
    periods: int | None = None,  # 要生成的周期数，默认为None
    freq=None,   # 频率别名，可选参数。默认从start或end中获取频率（如果它们是Period对象），否则默认为"D"，即每日频率
    name: Hashable | None = None,  # 结果PeriodIndex的名称，默认为None
) -> PeriodIndex:  # 返回值为PeriodIndex对象

    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    Examples
    --------
    >>> pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
                 '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                 '2018-01'],
                dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(
    ...     start=pd.Period("2017Q1", freq="Q"),
    ...     end=pd.Period("2017Q2", freq="Q"),
    ...     freq="M",
    ... )
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """

    # 检查三个参数（start、end、periods）中确切有两个参数被指定，否则抛出ValueError异常
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )

    # 如果freq为None，并且start和end都不是Period对象，则将freq设置为默认值"D"（每日频率）
    if freq is None and (not isinstance(start, Period) and not isinstance(end, Period)):
        freq = "D"

    # 调用PeriodArray._generate_range方法生成数据和频率
    data, freq = PeriodArray._generate_range(start, end, periods, freq)

    # 创建PeriodDtype对象，指定频率dtype
    dtype = PeriodDtype(freq)

    # 创建PeriodArray对象，使用生成的数据和dtype
    data = PeriodArray(data, dtype=dtype)

    # 返回PeriodIndex对象，使用PeriodArray和指定名称
    return PeriodIndex(data, name=name)
```