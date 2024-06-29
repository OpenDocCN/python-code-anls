# `D:\src\scipysrc\pandas\pandas\core\indexes\datetimelike.py`

```
"""
Base and utility classes for tseries type pandas objects.
"""

from __future__ import annotations  # 允许使用类型注释中的字符串形式的类

from abc import (  # 导入抽象基类模块中的 ABC 和 abstractmethod
    ABC,
    abstractmethod,
)
from typing import (  # 导入类型提示模块中的 TYPE_CHECKING 和 Any，cast，final 类型
    TYPE_CHECKING,
    Any,
    cast,
    final,
)

import numpy as np  # 导入 NumPy 库并使用 np 作为别名

from pandas._libs import (  # 从 pandas 私有库中导入 NaT, Timedelta, lib
    NaT,
    Timedelta,
    lib,
)
from pandas._libs.tslibs import (  # 从 pandas 私有库中导入 BaseOffset, Resolution, Tick, parsing, to_offset
    BaseOffset,
    Resolution,
    Tick,
    parsing,
    to_offset,
)
from pandas.compat.numpy import function as nv  # 从 NumPy 兼容模块导入 function 并使用 nv 作为别名
from pandas.errors import (  # 从 pandas 错误模块导入 InvalidIndexError, NullFrequencyError
    InvalidIndexError,
    NullFrequencyError,
)
from pandas.util._decorators import (  # 从 pandas 工具模块中导入 Appender, cache_readonly, doc 装饰器
    Appender,
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import (  # 从 pandas 核心数据类型模块导入 is_integer, is_list_like
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.concat import concat_compat  # 导入 pandas 核心数据类型模块中的 concat_compat 函数
from pandas.core.dtypes.dtypes import (  # 从 pandas 核心数据类型模块中导入 CategoricalDtype, PeriodDtype
    CategoricalDtype,
    PeriodDtype,
)

from pandas.core.arrays import (  # 从 pandas 核心数组模块中导入 DatetimeArray, ExtensionArray, PeriodArray, TimedeltaArray
    DatetimeArray,
    ExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin  # 从 pandas 核心数组日期类似模块中导入 DatetimeLikeArrayMixin 类
import pandas.core.common as com  # 导入 pandas 核心公共模块并使用 com 作为别名
import pandas.core.indexes.base as ibase  # 导入 pandas 核心索引基础模块并使用 ibase 作为别名
from pandas.core.indexes.base import (  # 从 pandas 核心索引基础模块中导入 Index, _index_shared_docs
    Index,
    _index_shared_docs,
)
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex  # 从 pandas 核心索引扩展模块中导入 NDArrayBackedExtensionIndex 类
from pandas.core.indexes.range import RangeIndex  # 从 pandas 核心索引范围模块中导入 RangeIndex 类
from pandas.core.tools.timedeltas import to_timedelta  # 从 pandas 核心工具时间增量模块中导入 to_timedelta 函数

if TYPE_CHECKING:  # 如果是类型检查模式
    from collections.abc import Sequence  # 从标准库集合抽象类中导入 Sequence 类
    from datetime import datetime  # 从标准库 datetime 模块中导入 datetime 类

    from pandas._typing import (  # 从 pandas 类型提示模块中导入 Axis, JoinHow, Self, npt
        Axis,
        JoinHow,
        Self,
        npt,
    )

    from pandas import CategoricalIndex  # 从 pandas 中导入 CategoricalIndex 类

_index_doc_kwargs = dict(ibase._index_doc_kwargs)  # 使用索引基础模块中 _index_doc_kwargs 的字典作为关键字参数

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """

    _can_hold_strings = False  # 设置 _can_hold_strings 属性为 False
    _data: DatetimeArray | TimedeltaArray | PeriodArray  # 设置 _data 属性为 DatetimeArray 或 TimedeltaArray 或 PeriodArray 类型

    @doc(DatetimeLikeArrayMixin.mean)  # 使用 DatetimeLikeArrayMixin 类中 mean 方法的文档字符串作为文档注释
    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._data.mean(skipna=skipna, axis=axis)  # 返回 _data 属性的均值，支持跳过 NaN 值和指定轴

    @property  # 声明属性
    def freq(self) -> BaseOffset | None:
        """
        Return the frequency object if it is set, otherwise None.

        To learn more about the frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        See Also
        --------
        DatetimeIndex.freq : Return the frequency object if it is set, otherwise None.
        PeriodIndex.freq : Return the frequency object if it is set, otherwise None.

        Examples
        --------
        >>> datetimeindex = pd.date_range(
        ...     "2022-02-22 02:22:22", periods=10, tz="America/Chicago", freq="h"
        ... )
        >>> datetimeindex
        DatetimeIndex(['2022-02-22 02:22:22-06:00', '2022-02-22 03:22:22-06:00',
                       '2022-02-22 04:22:22-06:00', '2022-02-22 05:22:22-06:00',
                       '2022-02-22 06:22:22-06:00', '2022-02-22 07:22:22-06:00',
                       '2022-02-22 08:22:22-06:00', '2022-02-22 09:22:22-06:00',
                       '2022-02-22 10:22:22-06:00', '2022-02-22 11:22:22-06:00'],
                      dtype='datetime64[ns, America/Chicago]', freq='h')
        >>> datetimeindex.freq
        <Hour>
        """
        return self._data.freq

    @freq.setter
    def freq(self, value) -> None:
        """
        Set the frequency object.

        Parameters
        ----------
        value : object
            The frequency object to be set.

        Notes
        -----
        This property is read-only for "PeriodArray".

        See Also
        --------
        DatetimeIndex.freq : Set the frequency object.
        PeriodIndex.freq : Set the frequency object.
        """
        self._data.freq = value  # type: ignore[misc]

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        """
        Return an array of integers representing the data in a fixed-width format.
        """
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> str:
        """
        Return the frequency string.

        Returns
        -------
        str
            The frequency string.

        Notes
        -----
        This property returns the frequency string if available for PeriodArray or PeriodIndex.
        """
        from pandas import PeriodIndex

        if self._data.freqstr is not None and isinstance(
            self._data, (PeriodArray, PeriodIndex)
        ):
            freq = PeriodDtype(self._data.freq)._freqstr
            return freq
        else:
            return self._data.freqstr  # type: ignore[return-value]

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        """
        Return the resolution object.

        This is an abstract method that should be implemented in subclasses.
        """

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> str:
        """
        Return the resolution string.
        """
        return self._data.resolution

    # ------------------------------------------------------------------------

    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return True if the data contains NaN values, False otherwise.
        """
        return self._data._hasna
    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        # 检查是否是同一个对象
        if self.is_(other):
            return True

        # 如果参数不是 Index 类型，直接返回 False
        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in "iufc":
            # 如果参数的数据类型是数值型（整数、浮点数等），返回 False
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            # 如果参数的数据类型是对象类型，尝试根据推断类型匹配
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                # 如果参数的数据类型是分类类型，尝试根据分类类别的推断类型匹配
                other = cast("CategoricalIndex", other)
                should_try = other.categories.inferred_type in inferable

            # 如果可以尝试进行类型转换
            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    # 捕获可能的异常情况，如数值转换错误或溢出
                    # 例如：ValueError -> 无法解析字符串条目，或者超出边界的日期时间
                    #      TypeError  -> 尝试将区间索引转换为日期时间索引
                    #      OverflowError -> Index([非常大的时间间隔])
                    return False

        # 如果对象的数据类型不同，返回 False
        if self.dtype != other.dtype:
            # 时区不同也会返回 False
            return False

        # 比较两个对象的元素是否相等
        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        # 计算键的哈希值
        hash(key)
        try:
            # 尝试获取键的位置
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            # 捕获可能的异常，如果获取位置失败则返回 False
            return False
        # 如果成功获取位置则返回 True
        return True

    def _convert_tolerance(self, tolerance, target):
        # 将容差转换为时间增量数组
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        # 调用父类方法进行进一步转换
        return super()._convert_tolerance(tolerance, target)

    # --------------------------------------------------------------------
    # Rendering Methods
    _default_na_rep = "NaT"

    def _format_with_header(
        self, *, header: list[str], na_rep: str, date_format: str | None = None
    ) -> list[str]:
        # TODO: not reached in tests 2023-10-11
        # 返回带有标题的格式化后的列表，用于CSV输出
        # 包含基类方法，但增加了空白填充和日期格式
        return header + list(
            self._get_values_for_csv(na_rep=na_rep, date_format=date_format)
        )

    @property
    def _formatter_func(self):
        # 返回格式化函数，使用内部数据的格式化方法
        return self._data._formatter()

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        # 调用父类方法获取格式化后的属性列表
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            # 遍历属性列表，并针对不同的属性进行特殊处理
            # 对于频率属性，获取其字符串表示形式
            if attrib == "freq":
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)  # 例如：D -> 'D'
                attrs.append(("freq", freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name=None) -> str:
        # 调用父类方法获取基本的摘要信息
        result = super()._summary(name=name)
        # 如果存在频率信息，则将频率字符串追加到结果中
        if self.freq:
            result += f"\nFreq: {self.freqstr}"
        
        # 返回最终的摘要结果
        return result

    # --------------------------------------------------------------------
    # Indexing Methods

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        # 判断是否可以对部分日期进行切片操作，返回一个布尔值
        # 例如：test_getitem_setitem_periodindex
        # 对话历史见 GH#3452, GH#3931, GH#2369, GH#14826
        return reso > self._resolution_obj
        # 注意：对于 DTI/PI，不适用于 TDI

    def _parsed_string_to_bounds(self, reso: Resolution, parsed):
        # 抛出未实现错误，子类需要实现具体的实现
        raise NotImplementedError

    def _parse_with_reso(self, label: str) -> tuple[datetime, Resolution]:
        # 被 TimedeltaIndex 覆盖的方法
        try:
            if self.freq is None or hasattr(self.freq, "rule_code"):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, "freqstr", getattr(self, "inferred_freq", None))
        
        freqstr: str | None
        # 如果 freq 不是 None 且不是字符串类型，则获取其规则代码
        if freq is not None and not isinstance(freq, str):
            freqstr = freq.rule_code
        else:
            freqstr = freq
        
        # 如果 label 是 np.str_ 类型，则将其转换为字符串
        if isinstance(label, np.str_):
            # GH#45580
            label = str(label)
        
        # 使用给定的标签和频率字符串解析日期时间，并返回解析结果和分辨率对象
        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return parsed, reso

    def _get_string_slice(self, key: str) -> slice | npt.NDArray[np.intp]:
        # 被 TimedeltaIndex 覆盖的方法，获取字符串切片
        parsed, reso = self._parse_with_reso(key)
        try:
            # 尝试使用部分日期切片方法来处理解析后的日期和分辨率
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            # 如果出现 KeyError，则将其向上抛出
            raise KeyError(key) from err

    @final
    def _partial_date_slice(
        self,
        reso: Resolution,
        parsed: datetime,
    ) -> slice | npt.NDArray[np.intp]:
        """
        Parameters
        ----------
        reso : Resolution
            分辨率对象
        parsed : datetime
            解析后的日期时间

        Returns
        -------
        slice or ndarray[intp]
            切片对象或整数数组
        """
        # 如果不能对部分日期进行切片，则抛出 ValueError
        if not self._can_partial_date_slice(reso):
            raise ValueError
        
        # 获取解析后日期时间的边界值 t1 和 t2
        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        # 获取数据的值和非盒化函数
        vals = self._data._ndarray
        unbox = self._data._unbox

        # 如果数据是单调递增的
        if self.is_monotonic_increasing:
            # 如果长度不为零，并且 t1 和 t2 超出了数据的范围，则抛出 KeyError
            if len(self) and (
                (t1 < self[0] and t2 < self[0]) or (t1 > self[-1] and t2 > self[-1])
            ):
                # 超出范围
                raise KeyError

            # TODO: 这是否依赖于单调递增？

            # 对于单调递增的系列，可以进行切片操作
            left = vals.searchsorted(unbox(t1), side="left")
            right = vals.searchsorted(unbox(t2), side="right")
            return slice(left, right)

        else:
            # 左侧掩码为大于等于 t1 的值
            lhs_mask = vals >= unbox(t1)
            # 右侧掩码为小于等于 t2 的值

            # 尝试找到日期
            return (lhs_mask & rhs_mask).nonzero()[0]
    def _maybe_cast_slice_bound(self, label, side: str):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
            The label to potentially cast to a scalar type.
        side : {'left', 'right'}
            Specifies whether to return the lower or upper bound.

        Returns
        -------
        object
            The label, possibly casted to a scalar type.

        Notes
        -----
        The value of `side` parameter should be validated in the caller.
        """
        if isinstance(label, str):
            try:
                # Attempt to parse the label with resolution
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                # Handle various parsing errors by raising an exception
                # DTI -> parsing.DateParseError
                # TDI -> 'unit abbreviation w/o a number'
                # PI -> string cannot be parsed as datetime-like
                self._raise_invalid_indexer("slice", label, err)

            # Convert parsed result and resolution to lower and upper bounds
            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            # If label is not a recognized scalar type, raise an invalid indexer exception
            self._raise_invalid_indexer("slice", label)

        # Return the label as-is if it doesn't need casting
        return label

    # --------------------------------------------------------------------
    # Arithmetic Methods

    def shift(self, periods: int = 1, freq=None) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr):
        """
        Cast a list-like indexer to appropriate form if possible.

        Parameters
        ----------
        keyarr : array-like
            The array-like object to potentially cast.

        Returns
        -------
        Index
            Index object containing the casted indexer.

        Notes
        -----
        This function handles validation and potential casting of list-like indexers,
        allowing for compatibility with various data structures and ensuring consistency
        in indexing operations.
        """
        try:
            # Validate and potentially cast the list-like indexer
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                # If the keyarr is not an ExtensionArray, cast it to a safe tuple-based array
                # e.g. we don't want to cast DTA to ndarray[object]
                res = com.asarray_tuplesafe(keyarr)
                # TODO: com.asarray_tuplesafe shouldn't cast e.g. DatetimeArray
            else:
                # Otherwise, simply retain the keyarr
                res = keyarr
        return Index(res, dtype=res.dtype)
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """

    # 数据成员，存储 DateTimeArray 或 TimedeltaArray 对象
    _data: DatetimeArray | TimedeltaArray
    # 可比较属性列表
    _comparables = ["name", "freq"]
    # 属性列表
    _attributes = ["name", "freq"]

    # 兼容性方法，用于推断频率，详见 GH#23789
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def unit(self) -> str:
        # 返回 _data 对象的单位属性
        return self._data.unit

    def as_unit(self, unit: str) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(["2020-01-02 01:02:03.004005006"])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit("s")
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(["1 day 3 min 2 us 42 ns"])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit("s")
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        # 调用 _data 对象的 as_unit 方法，并返回相同类型的新对象
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def _with_freq(self, freq):
        # 调用 _data 对象的 _with_freq 方法，并返回相同类型的新对象
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        # 注意事项：对于 Datetime64TZ 类型，此处可能会有信息丢失
        # 获取 _data 对象的 ndarray 属性
        data = self._data._ndarray
        # 创建一个数据视图
        data = data.view()
        # 设置数据视图为不可写
        data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq=None) -> Self:
        # 如果频率不为None且不等于当前对象的频率
        if freq is not None and freq != self.freq:
            # 如果freq是字符串，则转换为偏移量
            if isinstance(freq, str):
                freq = to_offset(freq)
            # 计算偏移量
            offset = periods * freq
            # 返回偏移后的对象
            return self + offset

        # 如果periods为0或者对象长度为0，返回对象的复制
        if periods == 0 or len(self) == 0:
            # GH#14811 空情况
            return self.copy()

        # 如果对象的频率为None，抛出异常
        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")

        # 计算移动后的起始时间和结束时间
        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq

        # 注意：在DatetimeTZ情况下，_generate_range将从`start`和`end`推断出适当的时区，
        # 因此不需要显式传递tz参数。
        # 使用_data._generate_range生成时间范围
        result = self._data._generate_range(
            start=start, end=end, periods=None, freq=self.freq, unit=self.unit
        )
        # 返回一个新的对象，类型与当前对象相同
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def inferred_freq(self) -> str | None:
        # 返回_data中推断出的频率
        return self._data.inferred_freq

    # --------------------------------------------------------------------
    # Set Operation Methods

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        # 将i8类型的表示转换为RangeIndex
        # 调用者需要检查self.freq是否是Tick的实例
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit(self.unit)._value
        # 使用步长创建RangeIndex
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other) -> bool:
        # 判断self.freq和other.freq是否都是Tick的实例
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other, res_i8) -> Self:
        new_freq = None
        # 如果结果为空，RangeIndex默认步长为1，我们不需要这个
        if not len(res_i8):
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            # 如果结果是RangeIndex，根据步长创建新的频率
            new_freq = to_offset(
                Timedelta(res_i8.step, unit=self.unit).as_unit(self.unit)
            )

        # TODO(GH#41493): 我们不能简单地使用以下方法
        # type(self._data)(res_i8.values, dtype=self.dtype, freq=new_freq)
        # 因为test_setops_preserve_freq失败了，_validate_frequency报错。
        # 这个错误是由于'on_freq'是不正确的。GH#41493将修复这个问题。
        # 将res_i8的值转换为self._data._ndarray的dtype
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        # 使用_simple_new创建新的对象
        result = type(self._data)._simple_new(
            res_values,
            dtype=self.dtype,  # type: ignore[arg-type]
            freq=new_freq,  # type: ignore[arg-type]
        )
        # 将结果转换为Self类型，并包装成set操作的结果
        return cast("Self", self._wrap_setop_result(other, result))
    def _range_intersect(self, other, sort) -> Self:
        # 调用 RangeIndex 的交集逻辑。
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other, sort) -> Self:
        # 调用 RangeIndex 的并集逻辑。
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _intersection(self, other: Index, sort: bool = False) -> Index:
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
        other = cast("DatetimeTimedeltaMixin", other)

        if self._can_range_setop(other):
            # 如果可以使用 RangeIndex 的集合操作，则调用 _range_intersect 方法
            return self._range_intersect(other, sort=sort)

        if not self._can_fast_intersect(other):
            # 如果不能使用快速交集操作，则调用 Index._intersection 方法
            result = Index._intersection(self, other, sort=sort)
            # 需要使频率无效，因为 Index._intersection 使用 self._data 的视图上的 _shallow_copy，
            # 如果不小心，会保留 self.freq。
            # 此时应该有 result.dtype == self.dtype，且 type(result) 是 self._data 的类型
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq("infer")

        else:
            # 如果可以使用快速交集操作，则调用 _fast_intersect 方法
            return self._fast_intersect(other, sort)

    def _fast_intersect(self, other, sort):
        # 为了简化操作，对两个范围进行排序
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        # 排序后，交集始终以 right 索引开始，并以最小的最后元素索引结束
        end = min(left[-1], right[-1])
        start = right[0]

        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]

        return result

    def _can_fast_intersect(self, other: Self) -> bool:
        # 注意：只有当 len(self) > 0 且 len(other) > 0 时才会执行到这里
        if self.freq is None:
            return False

        elif other.freq != self.freq:
            return False

        elif not self.is_monotonic_increasing:
            # 因为 freq 不为 None，所以必须是单调递减的
            return False

        # 这与匹配的 freq 确保我们“对齐”，因此交集将保留 freq
        # 注意我们假设没有 Tick，因为这些会通过 _range_intersect 处理
        # GH#42104
        return self.freq.n == 1
    def _can_fast_union(self, other: Self) -> bool:
        # Assumes that type(self) == type(other), as per the annotation
        # The ability to fast_union also implies that `freq` should be
        # retained on union.
        
        freq = self.freq  # 获取当前对象的频率信息

        if freq is None or freq != other.freq:
            return False  # 如果当前对象的频率为空，或者与其他对象的频率不同，则无法进行快速合并

        if not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            # TODO: do union on the reversed indexes?
            return False  # 如果当前对象不是单调递增的，则无法进行快速合并

        if len(self) == 0 or len(other) == 0:
            # only reached via union_many
            return True  # 如果当前对象或其他对象长度为0，则可以快速合并（通常在union_many中）

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        right_start = right[0]
        left_end = left[-1]

        # Only need to "adjoin", not overlap
        return (right_start == left_end + freq) or right_start in left
        # 返回是否可以快速合并的布尔值，根据右侧范围的起始点是否与左侧范围的末尾点相邻或者包含在左侧范围中

    def _fast_union(self, other: Self, sort=None) -> Self:
        # Caller is responsible for ensuring self and other are non-empty
        # 调用者负责确保self和other非空

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        elif sort is False:
            # TDIs are not in the "correct" order and we don't want
            # to sort but want to remove overlaps
            # 如果TDI不按“正确”的顺序排列，并且我们不想排序但想要移除重叠部分
            left, right = self, other
            left_start = left[0]
            loc = right.searchsorted(left_start, side="left")
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result  # 返回合并后的结果对象，移除了重叠的部分

        else:
            left, right = other, self

        left_end = left[-1]
        right_end = right[-1]

        # concatenate
        if left_end < right_end:
            loc = right.searchsorted(left_end, side="right")
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            # The can_fast_union check ensures that the result.freq
            # should match self.freq
            assert isinstance(dates, type(self._data))
            # error: Item "ExtensionArray" of "ExtensionArray |
            # ndarray[Any, Any]" has no attribute "_freq"
            assert dates._freq == self.freq  # type: ignore[union-attr]
            result = type(self)._simple_new(dates)
            return result  # 返回合并后的结果对象

        else:
            return left  # 如果右侧范围的末尾点小于左侧范围的末尾点，则返回左侧对象
    def _union(self, other, sort):
        # We are called by `union`, which is responsible for this validation
        # 断言确保参数 `other` 是与当前对象相同类型的对象
        assert isinstance(other, type(self))
        # 断言确保当前对象和 `other` 对象具有相同的数据类型
        assert self.dtype == other.dtype

        # 如果当前对象和 `other` 对象可以进行范围集合操作，则调用范围集合操作方法 `_range_union`
        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)

        # 如果当前对象和 `other` 对象可以进行快速联合操作，则调用快速联合操作方法 `_fast_union`
        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            # 当 sort=None 时，_can_fast_union 检查确保 result.freq == self.freq
            # 返回联合操作的结果
            return result
        else:
            # 否则调用超类的联合操作方法，并推断频率为 "infer"
            return super()._union(other, sort)._with_freq("infer")

    # --------------------------------------------------------------------
    # Join Methods

    def _get_join_freq(self, other):
        """
        Get the freq to attach to the result of a join operation.
        """
        freq = None
        # 如果当前对象和 `other` 对象可以进行快速联合操作，则将当前对象的频率赋给 `freq`
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def _wrap_join_result(
        self,
        joined,
        other,
        lidx: npt.NDArray[np.intp] | None,
        ridx: npt.NDArray[np.intp] | None,
        how: JoinHow,
    ) -> tuple[Self, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # 断言确保 `other` 对象和当前对象具有相同的数据类型
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        # 调用超类的 `_wrap_join_result` 方法，获取联合后的索引和索引位置
        join_index, lidx, ridx = super()._wrap_join_result(
            joined, other, lidx, ridx, how
        )
        # 将当前对象的频率赋给联合后的索引对象的频率属性 `_freq`
        join_index._data._freq = self._get_join_freq(other)
        # 返回联合后的索引对象、左侧索引位置和右侧索引位置
        return join_index, lidx, ridx

    def _get_engine_target(self) -> np.ndarray:
        # 返回当前对象的数据数组 `_data._ndarray`，视图类型转换为 "i8"
        # 引擎方法和连接方法需要将 dt64/td64 值转换为 i8
        return self._data._ndarray.view("i8")

    def _from_join_target(self, result: np.ndarray):
        # 将结果数组的视图类型转换回原始数据类型 `_data._ndarray.dtype`
        result = result.view(self._data._ndarray.dtype)
        # 从后台数据中创建新的数据对象，并返回
        return self._data._from_backing_data(result)

    # --------------------------------------------------------------------
    # List-like Methods

    def _get_delete_freq(self, loc: int | slice | Sequence[int]):
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        # 如果当前对象具有频率属性
        if self.freq is not None:
            # 如果位置参数 `loc` 是整数，并且在列表中的特定位置（首尾）之一
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                # 如果位置参数 `loc` 是类列表类型，将其转换为切片形式
                if is_list_like(loc):
                    loc = lib.maybe_indices_to_slice(  # type: ignore[assignment]
                        np.asarray(loc, dtype=np.intp), len(self)
                    )
                # 如果 `loc` 是切片并且步长为 1 或者 None，并且起始或结束位置在列表的边界
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq
    # 定义一个方法 `_get_insert_freq`，用于获取在插入操作后可能保留的频率信息
    def _get_insert_freq(self, loc: int, item):
        """
        Find the `freq` for self.insert(loc, item).
        """
        # 验证并封装插入项的数值
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)

        freq = None
        # 如果存在频率信息
        if self.freq is not None:
            # 在特殊情况下保留频率信息
            if self.size:
                # 处理边界情况
                if item is NaT:
                    pass
                elif loc in (0, -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif (loc == len(self)) and item - self.freq == self[-1]:
                    freq = self.freq
            else:
                # 在空索引中添加单个项可能会保留频率信息
                if isinstance(self.freq, Tick):
                    # 处理所有 TimedeltaIndex 情况；is_on_offset 可能会引发 TypeError
                    freq = self.freq
                elif self.freq.is_on_offset(item):
                    freq = self.freq
        return freq

    # 使用装饰器 doc，并提供 NDArrayBackedExtensionIndex.delete 方法的文档
    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc) -> Self:
        # 调用父类的 delete 方法执行删除操作
        result = super().delete(loc)
        # 获取删除操作后可能保留的频率信息
        result._data._freq = self._get_delete_freq(loc)
        return result

    # 使用装饰器 doc，并提供 NDArrayBackedExtensionIndex.insert 方法的文档
    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item):
        # 调用父类的 insert 方法执行插入操作
        result = super().insert(loc, item)
        # 如果 result 是当前类的实例，说明父类方法未进行类型转换
        if isinstance(result, type(self)):
            # 获取插入操作后可能保留的频率信息
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    # --------------------------------------------------------------------
    # NDArray-Like Methods

    # 使用 Appender 装饰器，提供 take 方法的文档，用于获取指定索引处的数据
    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        # 验证 take 方法的参数
        nv.validate_take((), kwargs)
        # 将 indices 转换为 np.ndarray 类型
        indices = np.asarray(indices, dtype=np.intp)

        # 调用 NDArrayBackedExtensionIndex.take 方法执行取值操作
        result = NDArrayBackedExtensionIndex.take(
            self, indices, axis, allow_fill, fill_value, **kwargs
        )

        # 如果 maybe_slice 是 slice 类型，获取该切片的频率信息并更新结果的频率
        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
```