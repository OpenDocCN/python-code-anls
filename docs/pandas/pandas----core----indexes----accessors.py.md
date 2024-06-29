# `D:\src\scipysrc\pandas\pandas\core\indexes\accessors.py`

```
"""
datetimelike delegation
"""

from __future__ import annotations  # 允许在类型注解中使用类本身作为类型

from typing import (  # 引入类型提示模块
    TYPE_CHECKING,  # 特殊标记，用于检查类型而不进行实际导入
    NoReturn,  # 特殊类型提示，表示函数没有返回值
    cast,  # 类型强制转换函数
)
import warnings  # 引入警告模块

import numpy as np  # 引入NumPy库

from pandas._libs import lib  # 导入Pandas库的底层C语言函数接口
from pandas.util._exceptions import find_stack_level  # 导入异常处理相关函数

from pandas.core.dtypes.common import (  # 导入常用数据类型判断函数
    is_integer_dtype,  # 判断是否为整数类型
    is_list_like,  # 判断是否为列表样式数据结构
)
from pandas.core.dtypes.dtypes import (  # 导入具体数据类型类定义
    ArrowDtype,  # Apache Arrow数据类型
    CategoricalDtype,  # 类别数据类型
    DatetimeTZDtype,  # 带时区的日期时间数据类型
    PeriodDtype,  # 时期数据类型
)
from pandas.core.dtypes.generic import ABCSeries  # 导入抽象基类Series定义

from pandas.core.accessor import (  # 导入访问器相关定义
    PandasDelegate,  # Pandas访问者基类
    delegate_names,  # 访问器名称列表
)
from pandas.core.arrays import (  # 导入数组相关定义
    DatetimeArray,  # 日期时间数组
    PeriodArray,  # 时期数组
    TimedeltaArray,  # 时间增量数组
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray  # 导入Arrow扩展数组定义
from pandas.core.base import (  # 导入Pandas基类
    NoNewAttributesMixin,  # 禁止新增属性混合类
    PandasObject,  # Pandas对象基类
)
from pandas.core.indexes.datetimes import DatetimeIndex  # 导入日期时间索引定义
from pandas.core.indexes.timedeltas import TimedeltaIndex  # 导入时间增量索引定义

if TYPE_CHECKING:  # 如果是类型检查模式
    from pandas import (  # 导入Pandas核心对象
        DataFrame,  # 数据框架
        Series,  # 系列数据
    )


class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {  # 属性设置，合并隐藏属性
        "orig",  # 原始属性
        "name",  # 名称属性
    }

    def __init__(self, data: Series, orig) -> None:  # 初始化方法
        if not isinstance(data, ABCSeries):  # 如果数据不是Pandas系列类型
            raise TypeError(  # 抛出类型错误异常
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data  # 将数据对象存储为父对象
        self.orig = orig  # 设置原始数据
        self.name = getattr(data, "name", None)  # 获取数据名称，若不存在则为None
        self._freeze()  # 冻结对象状态，禁止修改

    def _get_values(self):  # 获取值的内部方法
        data = self._parent  # 获取父对象数据
        if lib.is_np_dtype(data.dtype, "M"):  # 如果数据类型是日期时间类型
            return DatetimeIndex(data, copy=False, name=self.name)  # 返回日期时间索引对象

        elif isinstance(data.dtype, DatetimeTZDtype):  # 如果数据类型是带时区的日期时间类型
            return DatetimeIndex(data, copy=False, name=self.name)  # 返回日期时间索引对象

        elif lib.is_np_dtype(data.dtype, "m"):  # 如果数据类型是时间增量类型
            return TimedeltaIndex(data, copy=False, name=self.name)  # 返回时间增量索引对象

        elif isinstance(data.dtype, PeriodDtype):  # 如果数据类型是时期类型
            return PeriodArray(data, copy=False)  # 返回时期数组对象

        raise TypeError(  # 如果无法转换为日期时间类索引，则抛出类型错误异常
            f"cannot convert an object of type {type(data)} to a datetimelike index"
        )

    def _delegate_property_get(self, name: str):  # 委托属性获取方法
        from pandas import Series  # 导入Pandas系列对象

        values = self._get_values()  # 调用内部方法获取值

        result = getattr(values, name)  # 获取指定属性名的值

        # maybe need to upcast (ints)
        if isinstance(result, np.ndarray):  # 如果结果是NumPy数组
            if is_integer_dtype(result):  # 如果是整数类型
                result = result.astype("int64")  # 转换为int64类型

        elif not is_list_like(result):  # 如果结果不是列表样式数据结构
            return result  # 直接返回结果

        result = np.asarray(result)  # 将结果转换为NumPy数组

        if self.orig is not None:  # 如果存在原始数据
            index = self.orig.index  # 使用原始数据的索引
        else:  # 否则
            index = self._parent.index  # 使用父对象的索引

        return Series(result, index=index, name=self.name).__finalize__(self._parent)  # 返回结果作为系列对象
    # 抛出值错误异常，指示不支持对日期时间对象属性进行修改
    def _delegate_property_set(self, name: str, value, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "modifications to a property of a datetimelike object are not supported. "
            "Change values on the original."
        )

    # 委托方法调用到底层值对象，并返回结果
    def _delegate_method(self, name: str, *args, **kwargs):
        # 从 pandas 库导入 Series 类
        from pandas import Series
        
        # 获取底层值对象
        values = self._get_values()
        
        # 根据方法名获取对应的方法对象
        method = getattr(values, name)
        
        # 调用获取的方法，并传入参数，获取结果
        result = method(*args, **kwargs)
        
        # 如果结果不是类列表对象，直接返回结果
        if not is_list_like(result):
            return result
        
        # 将结果转换为 Series 对象，使用父对象的索引和名称，并终结后续操作
        return Series(result, index=self._parent.index, name=self.name).__finalize__(
            self._parent
        )
@delegate_names(
    delegate=ArrowExtensionArray,  # 使用ArrowExtensionArray作为委托对象
    accessors=TimedeltaArray._datetimelike_ops,  # 使用TimedeltaArray的日期时间操作作为访问器
    typ="property",  # 定义这个委托的类型为属性
    accessor_mapping=lambda x: f"_dt_{x}",  # 将访问器映射到"_dt_{x}"形式的属性名称
    raise_on_missing=False,  # 如果缺少则不引发异常
)
@delegate_names(
    delegate=ArrowExtensionArray,  # 使用ArrowExtensionArray作为委托对象
    accessors=TimedeltaArray._datetimelike_methods,  # 使用TimedeltaArray的日期时间方法作为访问器
    typ="method",  # 定义这个委托的类型为方法
    accessor_mapping=lambda x: f"_dt_{x}",  # 将访问器映射到"_dt_{x}"形式的属性名称
    raise_on_missing=False,  # 如果缺少则不引发异常
)
@delegate_names(
    delegate=ArrowExtensionArray,  # 使用ArrowExtensionArray作为委托对象
    accessors=DatetimeArray._datetimelike_ops,  # 使用DatetimeArray的日期时间操作作为访问器
    typ="property",  # 定义这个委托的类型为属性
    accessor_mapping=lambda x: f"_dt_{x}",  # 将访问器映射到"_dt_{x}"形式的属性名称
    raise_on_missing=False,  # 如果缺少则不引发异常
)
@delegate_names(
    delegate=ArrowExtensionArray,  # 使用ArrowExtensionArray作为委托对象
    accessors=DatetimeArray._datetimelike_methods,  # 使用DatetimeArray的日期时间方法作为访问器
    typ="method",  # 定义这个委托的类型为方法
    accessor_mapping=lambda x: f"_dt_{x}",  # 将访问器映射到"_dt_{x}"形式的属性名称
    raise_on_missing=False,  # 如果缺少则不引发异常
)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: Series, orig) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data  # 设置实例变量，指向传入的数据
        self._orig = orig  # 设置实例变量，指向传入的原始数据
        self._freeze()  # 冻结对象，使其不可变

    def _delegate_property_get(self, name: str):
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )
        result = getattr(self._parent.array, f"_dt_{name}")  # 获取属性名为"_dt_{name}"的属性

        if not is_list_like(result):  # 如果结果不是类列表的对象
            return result

        if self._orig is not None:
            index = self._orig.index  # 如果有原始数据，使用其索引
        else:
            index = self._parent.index  # 否则使用父数据的索引
        # 返回作为Series的结果，这是一个副本
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def _delegate_method(self, name: str, *args, **kwargs):
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )

        result = getattr(self._parent.array, f"_dt_{name}")(*args, **kwargs)  # 调用方法"_dt_{name}"

        if self._orig is not None:
            index = self._orig.index  # 如果有原始数据，使用其索引
        else:
            index = self._parent.index  # 否则使用父数据的索引
        # 返回作为Series的结果，这是一个副本
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result
    def to_pytimedelta(self):
        """
        # GH 57463
        发出警告，指明方法即将废弃，未来版本将返回包含 Python datetime.timedelta 对象的 Series，
        而不再是 ndarray。要保留旧行为，请在结果上调用 `np.array`。
        """
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pytimedelta is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime.timedelta objects instead of an ndarray. To retain the "
            "old behavior, call `np.array` on the result",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pytimedelta()

    def to_pydatetime(self) -> Series:
        """
        # GH#20306
        将 ArrowExtensionArray 转换为 Python datetime 对象的 Series。
        """
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pydatetime()

    def isocalendar(self) -> DataFrame:
        """
        生成包含年、周、日信息的 DataFrame。
        """
        from pandas import DataFrame
        
        result = (
            cast(ArrowExtensionArray, self._parent.array)
            ._dt_isocalendar()
            ._pa_array.combine_chunks()
        )
        iso_calendar_df = DataFrame(
            {
                col: type(self._parent.array)(result.field(i))  # type: ignore[call-arg]
                for i, col in enumerate(["year", "week", "day"])
            }
        )
        return iso_calendar_df

    @property
    def components(self) -> DataFrame:
        """
        生成包含日期时间组件信息的 DataFrame。
        """
        from pandas import DataFrame
        
        components_df = DataFrame(
            {
                col: getattr(self._parent.array, f"_dt_{col}")
                for col in [
                    "days",
                    "hours",
                    "minutes",
                    "seconds",
                    "milliseconds",
                    "microseconds",
                    "nanoseconds",
                ]
            }
        )
        return components_df
@delegate_names(
    delegate=DatetimeArray,
    accessors=DatetimeArray._datetimelike_ops + ["unit"],
    typ="property",
)
@delegate_names(
    delegate=DatetimeArray,
    accessors=DatetimeArray._datetimelike_methods + ["as_unit"],
    typ="method",
)
class DatetimeProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> seconds_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="s"))
    >>> seconds_series
    0   2000-01-01 00:00:00
    1   2000-01-01 00:00:01
    2   2000-01-01 00:00:02
    dtype: datetime64[ns]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    dtype: int32

    >>> hours_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="h"))
    >>> hours_series
    0   2000-01-01 00:00:00
    1   2000-01-01 01:00:00
    2   2000-01-01 02:00:00
    dtype: datetime64[ns]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    dtype: int32

    >>> quarters_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="QE"))
    >>> quarters_series
    0   2000-03-31
    1   2000-06-30
    2   2000-09-30
    dtype: datetime64[ns]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    dtype: int32

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """

    def to_pydatetime(self) -> Series:
        """
        Return the data as a Series of :class:`datetime.datetime` objects.

        Timezone information is retained if present.

        .. warning::

           Python's datetime uses microsecond resolution, which is lower than
           pandas (nanosecond). The values are truncated.

        Returns
        -------
        numpy.ndarray
            Object dtype array containing native Python datetime objects.

        See Also
        --------
        datetime.datetime : Standard library value for a datetime.

        Examples
        --------
        >>> s = pd.Series(pd.date_range("20180310", periods=2))
        >>> s
        0   2018-03-10
        1   2018-03-11
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        0    2018-03-10 00:00:00
        1    2018-03-11 00:00:00
        dtype: object

        pandas' nanosecond precision is truncated to microseconds.

        >>> s = pd.Series(pd.date_range("20180310", periods=2, freq="ns"))
        >>> s
        0   2018-03-10 00:00:00.000000000
        1   2018-03-10 00:00:00.000000001
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        0    2018-03-10 00:00:00
        1    2018-03-10 00:00:00
        dtype: object
        """
        # GH#20306
        from pandas import Series  # 导入 pandas 的 Series 类

        return Series(self._get_values().to_pydatetime(), dtype=object)  # 返回一个 Series 对象，其中包含转换为 Python datetime 对象的数据

    @property
    def freq(self):
        return self._get_values().inferred_freq  # 返回属性值 freq，通过调用 _get_values() 方法获取 Series 的值的推断频率
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
        >>> ser = pd.to_datetime(pd.Series(["2010-01-01", pd.NaT]))
        >>> ser.dt.isocalendar()
           year  week  day
        0  2009    53     5
        1  <NA>  <NA>  <NA>
        >>> ser.dt.isocalendar().week
        0      53
        1    <NA>
        Name: week, dtype: UInt32
        """
        # 调用 _get_values 方法获取数据，计算其 ISO 日历表示形式
        iso_calendar = self._get_values().isocalendar()
        # 将结果设为索引为父对象的索引，返回结果 DataFrame
        return iso_calendar.set_index(self._parent.index)
    # 使用 @delegate_names 装饰器将 TimedeltaArray 的属性和方法委托给 TimedeltaProperties 类
    # 通过 typ="property" 参数指定委托的类型为属性
    @delegate_names(
        delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ="property"
    )
    # 使用 @delegate_names 装饰器将 TimedeltaArray 的方法委托给 TimedeltaProperties 类
    # 通过 typ="method" 参数指定委托的类型为方法
    @delegate_names(
        delegate=TimedeltaArray,
        accessors=TimedeltaArray._datetimelike_methods,
        typ="method",
    )
    # 定义 TimedeltaProperties 类，用于访问 Series 值的日期时间属性
    class TimedeltaProperties(Properties):
        """
        Accessor object for datetimelike properties of the Series values.

        Returns a Series indexed like the original Series.
        Raises TypeError if the Series does not contain datetimelike values.

        Examples
        --------
        >>> seconds_series = pd.Series(
        ...     pd.timedelta_range(start="1 second", periods=3, freq="s")
        ... )
        >>> seconds_series
        0   0 days 00:00:01
        1   0 days 00:00:02
        2   0 days 00:00:03
        dtype: timedelta64[ns]
        >>> seconds_series.dt.seconds
        0    1
        1    2
        2    3
        dtype: int32
        """

        # 定义方法 to_pytimedelta，返回一个 numpy.ndarray，包含原始 Series 的 datetime.timedelta 对象
        def to_pytimedelta(self) -> np.ndarray:
            """
            Return an array of native :class:`datetime.timedelta` objects.

            Python's standard `datetime` library uses a different representation
            timedelta's. This method converts a Series of pandas Timedeltas
            to `datetime.timedelta` format with the same length as the original
            Series.

            Returns
            -------
            numpy.ndarray
                Array of 1D containing data with `datetime.timedelta` type.

            See Also
            --------
            datetime.timedelta : A duration expressing the difference
                between two date, time, or datetime.

            Examples
            --------
            >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="D"))
            >>> s
            0   0 days
            1   1 days
            2   2 days
            3   3 days
            4   4 days
            dtype: timedelta64[ns]

            >>> s.dt.to_pytimedelta()
            array([datetime.timedelta(0), datetime.timedelta(days=1),
            datetime.timedelta(days=2), datetime.timedelta(days=3),
            datetime.timedelta(days=4)], dtype=object)
            """
            # 发出警告，提醒用户方法的行为即将改变
            warnings.warn(
                f"The behavior of {type(self).__name__}.to_pytimedelta is deprecated, "
                "in a future version this will return a Series containing python "
                "datetime.timedelta objects instead of an ndarray. To retain the "
                "old behavior, call `np.array` on the result",
                FutureWarning,
                stacklevel=find_stack_level(),  # 获得当前调用堆栈的深度
            )
            # 调用 _get_values 方法获取 Series 的值，并将其转换为 datetime.timedelta 类型的 ndarray 返回
            return self._get_values().to_pytimedelta()

        @property
    def components(self) -> DataFrame:
        """
        Return a Dataframe of the components of the Timedeltas.

        Each row of the DataFrame corresponds to a Timedelta in the original
        Series and contains the individual components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedelta.

        Returns
        -------
        DataFrame

        See Also
        --------
        TimedeltaIndex.components : Return a DataFrame of the individual resolution
            components of the Timedeltas.
        Series.dt.total_seconds : Return the total number of seconds in the duration.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="s"))
        >>> s
        0   0 days 00:00:00
        1   0 days 00:00:01
        2   0 days 00:00:02
        3   0 days 00:00:03
        4   0 days 00:00:04
        dtype: timedelta64[ns]
        >>> s.dt.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     0      0        0        0             0             0            0
        1     0      0        0        1             0             0            0
        2     0      0        0        2             0             0            0
        3     0      0        0        3             0             0            0
        4     0      0        0        4             0             0            0
        """
        # 获取内部值并调用其 components 方法生成包含时间差各部分的 DataFrame
        return (
            self._get_values()  # 获取 Timedelta 对象的值
            .components.set_index(self._parent.index)  # 将生成的 DataFrame 设置索引为父级对象的索引
            .__finalize__(self._parent)  # 确保返回的对象与父级对象相关联
        )

    @property
    def freq(self):
        # 获取内部值并返回其推断频率
        return self._get_values().inferred_freq
@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ="method"
)
# 定义一个名为PeriodProperties的类，继承自Properties类，用于访问Series值的日期时间属性。
class PeriodProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.period_range(
    ...         start="2000-01-01 00:00:00", end="2000-01-01 00:00:03", freq="s"
    ...     )
    ... )
    >>> seconds_series
    0    2000-01-01 00:00:00
    1    2000-01-01 00:00:01
    2    2000-01-01 00:00:02
    3    2000-01-01 00:00:03
    dtype: period[s]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> hours_series = pd.Series(
    ...     pd.period_range(start="2000-01-01 00:00", end="2000-01-01 03:00", freq="h")
    ... )
    >>> hours_series
    0    2000-01-01 00:00
    1    2000-01-01 01:00
    2    2000-01-01 02:00
    3    2000-01-01 03:00
    dtype: period[h]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> quarters_series = pd.Series(
    ...     pd.period_range(start="2000-01-01", end="2000-12-31", freq="Q-DEC")
    ... )
    >>> quarters_series
    0    2000Q1
    1    2000Q2
    2    2000Q3
    3    2000Q4
    dtype: period[Q-DEC]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    3    4
    dtype: int64
    """


# 定义一个名为CombinedDatetimelikeProperties的类，继承自DatetimeProperties、TimedeltaProperties和PeriodProperties类，用于访问Series值的日期时间、时间差和周期属性。
class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    """
    Accessor object for Series values' datetime-like, timedelta and period properties.

    See Also
    --------
    DatetimeIndex : Index of datetime64 data.

    Examples
    --------
    >>> dates = pd.Series(
    ...     ["2024-01-01", "2024-01-15", "2024-02-5"], dtype="datetime64[ns]"
    ... )
    >>> dates.dt.day
    0     1
    1    15
    2     5
    dtype: int32
    >>> dates.dt.month
    0    1
    1    1
    2    2
    dtype: int32

    >>> dates = pd.Series(
    ...     ["2024-01-01", "2024-01-15", "2024-02-5"], dtype="datetime64[ns, UTC]"
    ... )
    >>> dates.dt.day
    0     1
    1    15
    2     5
    dtype: int32
    >>> dates.dt.month
    0    1
    1    1
    2    2
    dtype: int32
    """
    def __new__(cls, data: Series):  # pyright: ignore[reportInconsistentConstructor]
        # CombinedDatetimelikeProperties 类并不真正实例化。相反，我们需要选择适合的父类（datetime 或 timedelta）。
        # 由于我们无论如何都要检查 dtype，因此我们将在这里进行所有的验证。

        # 检查传入的 data 是否是 Pandas 的 Series 类型
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        # 如果 data 的 dtype 是 CategoricalDtype，则使用其原始数据
        orig = data if isinstance(data.dtype, CategoricalDtype) else None
        if orig is not None:
            # 使用原始数据创建一个新的 Series 对象
            data = data._constructor(
                orig.array,
                name=orig.name,
                copy=False,
                dtype=orig._values.categories.dtype,
                index=orig.index,
            )

        # 根据 data 的 dtype 类型选择合适的属性类进行返回
        if isinstance(data.dtype, ArrowDtype) and data.dtype.kind in "Mm":
            return ArrowTemporalProperties(data, orig)
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeProperties(data, orig)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeProperties(data, orig)
        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaProperties(data, orig)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodProperties(data, orig)

        # 如果没有匹配的 dtype 类型，则抛出异常
        raise AttributeError("Can only use .dt accessor with datetimelike values")
```