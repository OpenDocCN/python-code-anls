# `D:\src\scipysrc\pandas\pandas\core\resample.py`

```
from __future__ import annotations
# 允许在类型提示中使用字符串形式的类型注释

import copy
# 导入用于复制对象的模块

from textwrap import dedent
# 导入用于移除多余缩进的文本包装函数

from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    final,
    no_type_check,
    overload,
)
# 导入类型提示相关的模块和类型

import warnings
# 导入用于警告处理的模块

import numpy as np
# 导入NumPy库，并使用“np”作为别名

from pandas._libs import lib
# 导入Pandas内部的C语言库

from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
# 导入时间序列相关的类型和函数

from pandas._typing import NDFrameT
# 导入Pandas类型注解相关的类型

from pandas.errors import AbstractMethodError
# 导入Pandas抽象方法错误异常

from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)
# 导入用于修饰器的辅助函数

from pandas.util._exceptions import (
    find_stack_level,
    rewrite_warning,
)
# 导入异常处理相关的函数

from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    PeriodDtype,
)
# 导入Pandas数据类型相关的定义

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
# 导入Pandas核心泛型数据结构的基类定义

import pandas.core.algorithms as algos
# 导入Pandas核心算法模块

from pandas.core.apply import ResamplerWindowApply
# 导入用于重采样窗口应用的函数

from pandas.core.arrays import ArrowExtensionArray
# 导入Arrow扩展数组

from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
# 导入Pandas核心基础对象和选择混合类

from pandas.core.generic import (
    NDFrame,
    _shared_docs,
)
# 导入Pandas核心泛型数据结构和共享文档

from pandas.core.groupby.groupby import (
    BaseGroupBy,
    GroupBy,
    _apply_groupings_depr,
    _pipe_template,
    get_groupby,
)
# 导入Pandas分组操作相关的函数和类

from pandas.core.groupby.grouper import Grouper
# 导入Pandas分组器类

from pandas.core.groupby.ops import BinGrouper
# 导入Pandas分组操作类

from pandas.core.indexes.api import MultiIndex
# 导入Pandas多重索引API

from pandas.core.indexes.base import Index
# 导入Pandas索引基类

from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
# 导入Pandas日期时间索引和日期范围函数

from pandas.core.indexes.period import (
    PeriodIndex,
    period_range,
)
# 导入Pandas周期索引和周期范围函数

from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)
# 导入Pandas时间增量索引和时间增量范围函数

from pandas.core.reshape.concat import concat
# 导入Pandas连接函数

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)
# 导入Pandas时间序列频率相关的函数

from pandas.tseries.offsets import (
    Day,
    Tick,
)
# 导入Pandas时间偏移类

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
    )
    # 在类型检查环境中，导入用于类型提示的标准库类型

    from pandas._typing import (
        Any,
        AnyArrayLike,
        Axis,
        Concatenate,
        FreqIndexT,
        Frequency,
        IndexLabel,
        InterpolateOptions,
        P,
        Self,
        T,
        TimedeltaConvertibleTypes,
        TimestampConvertibleTypes,
        npt,
    )
    # 在类型检查环境中，导入用于类型提示的Pandas扩展类型

    from pandas import (
        DataFrame,
        Series,
    )
    # 在类型检查环境中，导入Pandas核心数据结构类型

_shared_docs_kwargs: dict[str, str] = {}
# 创建一个空字典用于存储共享文档的关键字参数

class Resampler(BaseGroupBy, PandasObject):
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
        The input time series or DataFrame object to be resampled.
    groupby : TimeGrouper
        The time grouper object defining how to group the data for resampling.

    Returns
    -------
    a Resampler of the appropriate type
        A resampler object that allows applying aggregate, transform, and apply
        operations after resampling.

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """
    _grouper: BinGrouper
    # 内部变量，用于分组操作的具体实现
    _timegrouper: TimeGrouper
    # 内部变量，用于时间分组的具体实现
    binner: DatetimeIndex | TimedeltaIndex | PeriodIndex  # 依赖于子类的具体实现，可能是日期时间索引、时间增量索引或周期索引
    exclusions: frozenset[Hashable] = frozenset()  # 用于与 SelectionMixin 兼容的排除项集合
    _internal_names_set = set({"obj", "ax", "_indexer"})

    # 用于描述 groupby 对象的属性列表
    _attributes = [
        "freq",
        "closed",
        "label",
        "convention",
        "origin",
        "offset",
    ]

    def __init__(
        self,
        obj: NDFrame,
        timegrouper: TimeGrouper,
        *,
        gpr_index: Index,
        group_keys: bool = False,
        selection=None,
        include_groups: bool = True,
    ) -> None:
        # 初始化函数，设置对象的初始状态和属性
        self._timegrouper = timegrouper
        self.keys = None
        self.sort = True
        self.group_keys = group_keys
        self.as_index = True
        self.include_groups = include_groups

        # 设置对象的 obj、ax 和 _indexer 属性，使用 _timegrouper 的方法进行设置
        self.obj, self.ax, self._indexer = self._timegrouper._set_grouper(
            self._convert_obj(obj), sort=True, gpr_index=gpr_index
        )
        # 获取 binner 和 _grouper 属性的值
        self.binner, self._grouper = self._get_binner()
        self._selection = selection
        # 根据 _timegrouper 的 key 属性确定 exclusions 的值
        if self._timegrouper.key is not None:
            self.exclusions = frozenset([self._timegrouper.key])
        else:
            self.exclusions = frozenset()

    @final
    def __str__(self) -> str:
        """
        返回滚动对象的友好字符串表示形式。
        """
        # 生成对象的描述字符串，包括 _timegrouper 对象的属性
        attrs = (
            f"{k}={getattr(self._timegrouper, k)}"
            for k in self._attributes
            if getattr(self._timegrouper, k, None) is not None
        )
        return f"{type(self).__name__} [{', '.join(attrs)}]"

    @final
    def __getattr__(self, attr: str):
        # 获取对象的属性，如果属性在 _internal_names_set 中则直接返回
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        # 如果属性在 _attributes 中，则获取 _timegrouper 对象的对应属性
        if attr in self._attributes:
            return getattr(self._timegrouper, attr)
        # 如果属性在 obj 对象中，则返回对应的属性值
        if attr in self.obj:
            return self[attr]

        return object.__getattribute__(self, attr)

    @final
    @property
    def _from_selection(self) -> bool:
        """
        判断是否从 DataFrame 列或 MultiIndex 级别进行重采样。
        """
        # 如果 _timegrouper 不为 None，并且具有 key 或 level 属性，则返回 True
        # 用于捕获和抛出错误的状态
        return self._timegrouper is not None and (
            self._timegrouper.key is not None or self._timegrouper.level is not None
        )

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        """
        对对象进行必要的转换，以便正确处理。

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Series or DataFrame
        """
        # 对对象进行整理操作，并返回整理后的对象
        return obj._consolidate()

    def _get_binner_for_time(self):
        raise AbstractMethodError(self)

    @final
    def _get_binner(self):
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
        # 调用内部方法获取时间的分组信息，并获取返回的三个元素
        binner, bins, binlabels = self._get_binner_for_time()
        
        # 断言bins和binlabels长度相等
        assert len(bins) == len(binlabels)
        
        # 使用获取的bins和binlabels创建BinGrouper对象，传入self._indexer作为索引器
        bin_grouper = BinGrouper(bins, binlabels, indexer=self._indexer)
        
        # 返回binner和创建的bin_grouper对象
        return binner, bin_grouper

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    
    @final
    @Substitution(
        klass="Resampler",
        examples="""
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},
    ...                   index=pd.date_range('2012-08-02', periods=4))
    >>> df
                A
    2012-08-02  1
    2012-08-03  2
    2012-08-04  3
    2012-08-05  4

    To get the difference between each 2-day period's maximum and minimum
    value in one pass, you can do

    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())
                A
    2012-08-02  1
    2012-08-04  1""",
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # 调用父类的pipe方法，传入func、args和kwargs，并返回其结果
        return super().pipe(func, *args, **kwargs)

    _agg_see_also_doc = dedent(
        """
    See Also
    --------
    DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
        or list of string/callables.
    DataFrame.resample.transform : Transforms the Series on each group
        based on the given function.
    DataFrame.aggregate: Aggregate using one or more
        operations over the specified axis.
    """
    )

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5],
    ...               index=pd.date_range('20130101', periods=5, freq='s'))
    >>> s
    2013-01-01 00:00:00    1
    2013-01-01 00:00:01    2
    2013-01-01 00:00:02    3
    2013-01-01 00:00:03    4
    2013-01-01 00:00:04    5
    Freq: s, dtype: int64

    >>> r = s.resample('2s')

    >>> r.agg("sum")
    2013-01-01 00:00:00    3
    2013-01-01 00:00:02    7
    2013-01-01 00:00:04    5
    Freq: 2s, dtype: int64

    >>> r.agg(['sum', 'mean', 'max'])
                         sum  mean  max
    2013-01-01 00:00:00    3   1.5    2
    2013-01-01 00:00:02    7   3.5    4
    2013-01-01 00:00:04    5   5.0    5

    >>> r.agg({'result': lambda x: x.mean() / x.std(),
    ...        'total': "sum"})
                           result  total
    2013-01-01 00:00:00  2.121320      3
    2013-01-01 00:00:02  4.949747      7
    2013-01-01 00:00:04       NaN      5

    >>> r.agg(average="mean", total="sum")
                             average  total
    2013-01-01 00:00:00      1.5      3
    2013-01-01 00:00:02      3.5      7
    """
    )
    2013-01-01 00:00:04      5.0      5
    """
    )

    @final
    @doc(
        _shared_docs["aggregate"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
        klass="DataFrame",
        axis="",
    )
    # 使用 @final 装饰器标记该方法为最终方法，不应被子类重写
    # 使用 @doc 装饰器添加文档注释，指定文档中的聚合部分的共享文档、相关链接、示例以及类名和轴向
    def aggregate(self, func=None, *args, **kwargs):
        # 创建 ResamplerWindowApply 对象，传入当前对象、函数、位置参数和关键字参数，执行聚合操作
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        # 如果结果为空
        if result is None:
            # 将函数赋值给 how
            how = func
            # 调用 _groupby_and_aggregate 方法进行分组和聚合操作，传入函数及其它参数
            result = self._groupby_and_aggregate(how, *args, **kwargs)

        # 返回结果
        return result

    # 将 aggregate 方法赋值给 agg 和 apply 方法
    agg = aggregate
    apply = aggregate

    @final
    # 定义 transform 方法
    def transform(self, arg, *args, **kwargs):
        """
        Call function producing a like-indexed Series on each group.

        Return a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([1, 2], index=pd.date_range("20180101", periods=2, freq="1h"))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> resampled = s.resample("15min")
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        2018-01-01 00:00:00   NaN
        2018-01-01 01:00:00   NaN
        Freq: h, dtype: float64
        """
        # 调用 _selected_obj 上的 groupby 方法，使用 _timegrouper 进行分组，并应用 arg 函数及其它参数
        return self._selected_obj.groupby(self._timegrouper).transform(
            arg, *args, **kwargs
        )

    # 定义 _downsample 方法，抛出抽象方法错误
    def _downsample(self, f, **kwargs):
        raise AbstractMethodError(self)

    # 定义 _upsample 方法，抛出抽象方法错误，接受限制参数和填充值
    def _upsample(self, f, limit: int | None = None, fill_value=None):
        raise AbstractMethodError(self)

    # 定义 _gotitem 方法，返回被切片对象
    def _gotitem(self, key, ndim: int, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # 获取分组器
        grouper = self._grouper
        # 如果子集为空，将对象赋值给子集
        if subset is None:
            subset = self.obj
            # 如果键不为空，将子集切片
            if key is not None:
                subset = subset[key]
            else:
                # 当通过 Apply.agg_dict_like 达到时，选择为空且维度为1
                assert subset.ndim == 1
        # 如果维度为1，确保子集也为1维
        if ndim == 1:
            assert subset.ndim == 1

        # 调用 get_groupby 方法，传入子集、无分组键、分组器和分组键，返回分组结果
        grouped = get_groupby(
            subset, by=None, grouper=grouper, group_keys=self.group_keys
        )
        # 返回分组后的对象
        return grouped
    def _groupby_and_aggregate(self, how, *args, **kwargs):
        """
        Re-evaluate the obj with a groupby aggregation.
        """
        # 获取分组器对象
        grouper = self._grouper

        # 从 self._obj_with_exclusions 中排除 'on' 列（如果提供）
        obj = self._obj_with_exclusions

        # 使用 get_groupby 函数进行分组操作，不指定 by 参数，使用 grouper 进行分组，
        # group_keys 参数表示是否包含分组键
        grouped = get_groupby(obj, by=None, grouper=grouper, group_keys=self.group_keys)

        try:
            if callable(how):
                # 如果 how 是可调用的函数，则使用 lambda 函数创建一个新的函数 func，
                # 将 how 应用到 grouped 数据上，并传入 *args 和 **kwargs
                func = lambda x: how(x, *args, **kwargs)
                result = grouped.aggregate(func)
            else:
                # 否则直接将 how 当作聚合函数，将其应用到 grouped 数据上，并传入 *args 和 **kwargs
                result = grouped.aggregate(how, *args, **kwargs)
        except (AttributeError, KeyError):
            # 出现 AttributeError 或 KeyError 异常时，尝试应用 _apply 函数到 grouped 上，
            # 将 how 函数应用到数据上，并传入 *args 和 **kwargs，include_groups 表示是否包含组信息
            result = _apply(
                grouped, how, *args, include_groups=self.include_groups, **kwargs
            )

        except ValueError as err:
            if "Must produce aggregated value" in str(err):
                # 如果异常信息包含 "Must produce aggregated value"，则忽略异常
                # 见 _aggregate_named 的异常处理
                pass
            else:
                # 否则抛出异常
                raise

            # 出现异常时，同样尝试应用 _apply 函数到 grouped 上，
            # 将 how 函数应用到数据上，并传入 *args 和 **kwargs，include_groups 表示是否包含组信息
            result = _apply(
                grouped, how, *args, include_groups=self.include_groups, **kwargs
            )

        # 将结果使用 self._wrap_result 方法进行包装并返回
        return self._wrap_result(result)

    @final
    def _get_resampler_for_grouping(
        self, groupby: GroupBy, key, include_groups: bool = True
    ):
        """
        Return the correct class for resampling with groupby.
        """
        # 调用 _resampler_for_grouping 方法获取适合进行分组重新采样的类
        return self._resampler_for_grouping(
            groupby=groupby, key=key, parent=self, include_groups=include_groups
        )
    @final
    def ffill(self, limit: int | None = None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        An upsampled Series.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.

        Examples
        --------
        Here we only create a ``Series``.

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        Example for ``ffill`` with downsampling (we have fewer dates after resampling):

        >>> ser.resample("MS").ffill()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64

        Example for ``ffill`` with upsampling (fill the new dates with
        the previous value):

        >>> ser.resample("W").ffill()
        2023-01-01    1
        2023-01-08    1
        2023-01-15    2
        2023-01-22    2
        2023-01-29    2
        2023-02-05    3
        2023-02-12    3
        2023-02-19    4
        Freq: W-SUN, dtype: int64

        With upsampling and limiting (only fill the first new date with the
        previous value):

        >>> ser.resample("W").ffill(limit=1)
        2023-01-01    1.0
        2023-01-08    1.0
        2023-01-15    2.0
        2023-01-22    2.0
        2023-01-29    NaN
        2023-02-05    3.0
        2023-02-12    NaN
        2023-02-19    4.0
        Freq: W-SUN, dtype: float64
        """
        return self._upsample("ffill", limit=limit)


注释：

# 使用前向填充方法填充值的函数

# 参数
# ------
# limit : int, optional
#     填充值的数量限制。

# 返回
# -------
# 一个上采样的 Series。

# 参见
# --------
# Series.fillna: 使用指定方法填充 NA/NaN 值。
# DataFrame.fillna: 使用指定方法填充 NA/NaN 值。

# 示例
# --------
# 这里我们只创建一个 ``Series``。

# >>> ser = pd.Series(
# ...     [1, 2, 3, 4],
# ...     index=pd.DatetimeIndex(
# ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
# ...     ),
# ... )
# >>> ser
# 2023-01-01    1
# 2023-01-15    2
# 2023-02-01    3
# 2023-02-15    4
# dtype: int64

# ``ffill`` 示例，与下采样一起使用（重新采样后日期更少）：

# >>> ser.resample("MS").ffill()
# 2023-01-01    1
# 2023-02-01    3
# Freq: MS, dtype: int64

# ``ffill`` 示例，与上采样一起使用（使用前值填充新日期）：

# >>> ser.resample("W").ffill()
# 2023-01-01    1
# 2023-01-08    1
# 2023-01-15    2
# 2023-01-22    2
# 2023-01-29    2
# 2023-02-05    3
# 2023-02-12    3
# 2023-02-19    4
# Freq: W-SUN, dtype: int64

# 使用上采样和限制（仅使用前值填充第一个新日期）：

# >>> ser.resample("W").ffill(limit=1)
# 2023-01-01    1.0
# 2023-01-08    1.0
# 2023-01-15    2.0
# 2023-01-22    2.0
# 2023-01-29    NaN
# 2023-02-05    3.0
# 2023-02-12    NaN
# 2023-02-19    4.0
# Freq: W-SUN, dtype: float64
    def nearest(self, limit: int | None = None):
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        bfill : Backward fill the new missing values in the resampled data.
        ffill : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2], index=pd.date_range("20180101", periods=2, freq="1h"))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> s.resample("15min").nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15min, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample("15min").nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15min, dtype: float64
        """
        # 调用内部方法 `_upsample` 来进行最近邻值的上采样操作
        return self._upsample("nearest", limit=limit)

    @final
    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area=None,
        downcast=lib.no_default,
        **kwargs,
    @final
    def asfreq(self, fill_value=None):
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-31", "2023-02-01", "2023-02-28"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-31    2
        2023-02-01    3
        2023-02-28    4
        dtype: int64
        >>> ser.resample("MS").asfreq()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._upsample("asfreq", fill_value=fill_value)




    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        """
        Compute sum of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").sum()
        2023-01-01    3
        2023-02-01    7
        Freq: MS, dtype: int64
        """
        return self._downsample("sum", numeric_only=numeric_only, min_count=min_count)



    @final
    def prod(
        self,
        numeric_only: bool = False,
        min_count: int = 0,


**注释：**

- `asfreq(self, fill_value=None):`：返回按新频率重新索引后的值。

- `sum(self, numeric_only: bool = False, min_count: int = 0):`：计算分组值的总和。

- `prod(self, numeric_only: bool = False, min_count: int = 0,`：计算分组值的乘积。
    ):
        """
        Compute prod of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed prod of values within each group.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").prod()
        2023-01-01    2
        2023-02-01   12
        Freq: MS, dtype: int64
        """
        return self._downsample("prod", numeric_only=numeric_only, min_count=min_count)

    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        """
        Compute min value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed min value within each group.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").min()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._downsample("min", numeric_only=numeric_only, min_count=min_count)

    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        """
        Compute max value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed max value within each group.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").max()
        2023-01-01    2
        2023-02-01    4
        Freq: MS, dtype: int64
        """
        return self._downsample("max", numeric_only=numeric_only, min_count=min_count)

    @final
    @doc(GroupBy.first)
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ):
        """
        Compute first value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        Series or DataFrame
            First value within each group.

        Notes
        -----
        This method is an alias for ``GroupBy.first``.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").first()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._downsample("first", numeric_only=numeric_only, min_count=min_count, skipna=skipna)
    ):
        # 调用 _downsample 方法，使用 "first" 策略进行降采样
        return self._downsample(
            "first", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    @doc(GroupBy.last)
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ):
        # 调用 _downsample 方法，使用 "last" 策略进行降采样
        return self._downsample(
            "last", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    @doc(GroupBy.median)
    def median(self, numeric_only: bool = False):
        # 调用 _downsample 方法，使用 "median" 策略进行降采样
        return self._downsample("median", numeric_only=numeric_only)

    @final
    def mean(
        self,
        numeric_only: bool = False,
    ):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Mean of values within each group.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").mean()
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        # 调用 _downsample 方法，使用 "mean" 策略进行降采样
        return self._downsample("mean", numeric_only=numeric_only)

    @final
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Standard deviation of values within each group.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").std()
        2023-01-01    1.000000
        2023-02-01    2.645751
        Freq: MS, dtype: float64
        """
        # 调用 _downsample 方法，使用 "std" 策略进行降采样
        return self._downsample("std", ddof=ddof, numeric_only=numeric_only)

    @final
    @final
    # 声明方法为最终版本，禁止子类重写
    @doc(GroupBy.ohlc)
    # 使用 GroupBy.ohlc 方法的文档字符串作为当前方法的文档
    def ohlc(self):
        # 获取 self 对象的 ax 属性
        ax = self.ax
        # 获取带有排除项的 self._obj_with_exclusions
        obj = self._obj_with_exclusions
        # 如果 ax 的长度为 0
        if len(ax) == 0:
            # GH#42902
            # 复制 obj 对象
            obj = obj.copy()
            # 将 obj 的索引转换为与 self.freq 兼容的频率
            obj.index = _asfreq_compat(obj.index, self.freq)
            # 如果 obj 的维度为 1
            if obj.ndim == 1:
                # 将 obj 转换为 DataFrame 对象
                obj = obj.to_frame()
                # 重新索引 obj 的列为 ["open", "high", "low", "close"]
                obj = obj.reindex(["open", "high", "low", "close"], axis=1)
            else:
                # 创建多级索引 mi，包含 obj.columns 与 ["open", "high", "low", "close"] 的笛卡尔积
                mi = MultiIndex.from_product(
                    [obj.columns, ["open", "high", "low", "close"]]
                )
                # 重新索引 obj 的列为 mi
                obj = obj.reindex(mi, axis=1)
            # 返回 obj
            return obj

        # 调用 self._downsample 方法，执行 "ohlc" 操作
        return self._downsample("ohlc")
    def nunique(self):
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.

        See Also
        --------
        core.groupby.SeriesGroupBy.nunique : Method nunique for SeriesGroupBy.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 3],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    3
        dtype: int64
        >>> ser.resample("MS").nunique()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        return self._downsample("nunique")

    @final
    @doc(GroupBy.size)
    def size(self):
        # 使用 _downsample 方法计算分组后的大小（元素数量）
        result = self._downsample("size")

        # 如果结果是非空 DataFrame，则堆叠以得到一个 Series
        # GH 46826
        if isinstance(result, ABCDataFrame) and not result.empty:
            result = result.stack()

        # 如果 self.ax 长度为 0
        if not len(self.ax):
            from pandas import Series

            # 如果选中对象的维度为 1
            if self._selected_obj.ndim == 1:
                name = self._selected_obj.name
            else:
                name = None

            # 创建一个空的 Series，保留索引并指定数据类型和名称
            result = Series([], index=result.index, dtype="int64", name=name)
        return result

    @final
    @doc(GroupBy.count)
    def count(self):
        # 使用 _downsample 方法计算分组后的计数
        result = self._downsample("count")

        # 如果 self.ax 长度为 0
        if not len(self.ax):
            # 如果选中对象的维度为 1
            if self._selected_obj.ndim == 1:
                # 创建一个空的相同类型对象
                result = type(self._selected_obj)(
                    [], index=result.index, dtype="int64", name=self._selected_obj.name
                )
            else:
                from pandas import DataFrame

                # 创建一个空的 DataFrame，保留索引和列，并指定数据类型
                result = DataFrame(
                    [], index=result.index, columns=result.columns, dtype="int64"
                )

        return result

    @final
    def quantile(self, q: float | list[float] | AnyArrayLike = 0.5, **kwargs):
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            The quantile or sequence of quantiles to compute.

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the columns are groupby columns,
            and the values are its quantiles.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        Calculate quantile based on resampled data by calling _downsample method
        >>> ser.resample("MS").quantile()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64

        Calculate quantile based on resampled data by calling _downsample method and
        provide 25th percentile
        >>> ser.resample("MS").quantile(0.25)
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        return self._downsample("quantile", q=q, **kwargs)
class _GroupByMixin(PandasObject, SelectionMixin):
    """
    Provide the groupby facilities.
    """

    _attributes: list[str]  # in practice the same as Resampler._attributes
    _selection: IndexLabel | None = None
    _groupby: GroupBy
    _timegrouper: TimeGrouper

    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key=None,
        selection: IndexLabel | None = None,
        include_groups: bool = False,
    ) -> None:
        # reached via ._gotitem and _get_resampler_for_grouping
        
        # 确保 groupby 参数是 GroupBy 类型
        assert isinstance(groupby, GroupBy), type(groupby)

        # 确保 parent 参数始终是 Resampler 类型，有时也是 _GroupByMixin
        assert isinstance(parent, Resampler), type(parent)

        # 使用 resampler 的属性初始化 GroupByMixin 对象
        for attr in self._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection

        self.binner = parent.binner
        self.key = key

        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)

        self.ax = parent.ax
        self.obj = parent.obj
        self.include_groups = include_groups

    @no_type_check
    def _apply(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """

        def func(x):
            # 将 x 转换为适合的 resampler 类型对象
            x = self._resampler_cls(x, timegrouper=self._timegrouper, gpr_index=self.ax)

            if isinstance(f, str):
                # 如果 f 是字符串，则调用 x 对象的相应方法并传入 kwargs
                return getattr(x, f)(**kwargs)

            # 否则调用 x 对象的 apply 方法，并传入 f, args, kwargs
            return x.apply(f, *args, **kwargs)

        # 调用 _apply 函数处理 groupby 对象，并包含组信息（如果设置了 include_groups）
        result = _apply(self._groupby, func, include_groups=self.include_groups)
        return self._wrap_result(result)

    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

    @final
    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
            键：字符串或选择列表
        ndim : {1, 2}
            请求结果的维度
        subset : object, default None
            待操作的子集
        """
        # create a new object to prevent aliasing
        # 创建一个新对象以防止别名引用
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                # reached via Apply.agg_dict_like with selection=None, ndim=1
                # 通过 Apply.agg_dict_like 方法调用，selection=None，ndim=1 时执行
                assert subset.ndim == 1

        # Try to select from a DataFrame, falling back to a Series
        # 尝试从 DataFrame 中选择，如果失败则回退到 Series
        try:
            if isinstance(key, list) and self.key not in key and self.key is not None:
                key.append(self.key)
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby

        selection = self._infer_selection(key, subset)

        new_rs = type(self)(
            groupby=groupby,
            parent=cast(Resampler, self),
            selection=selection,
        )
        return new_rs
class DatetimeIndexResampler(Resampler):
    ax: DatetimeIndex

    @property
    def _resampler_for_grouping(self) -> type[DatetimeIndexResamplerGroupby]:
        # 返回用于分组的 DatetimeIndexResamplerGroupby 类型
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self):
        # 这里是创建时间分组器的方式
        if isinstance(self.ax, PeriodIndex):
            return self._timegrouper._get_time_period_bins(self.ax)
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        ax = self.ax

        # Excludes `on` column when provided
        obj = self._obj_with_exclusions

        if not len(ax):
            # 如果 ax 长度为零，重新设置索引到新的频率
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert obj.index.freq == self.freq, (obj.index.freq, self.freq)
            return obj

        # 是否有常规频率

        # error: Item "None" of "Optional[Any]" has no attribute "binlabels"
        if (
            (ax.freq is not None or ax.inferred_freq is not None)
            and len(self._grouper.binlabels) > len(ax)
            and how is None
        ):
            # 进行 asfreq 操作
            return self.asfreq()

        # 进行降采样
        # 在这里调用实际的分组方法
        result = obj.groupby(self._grouper).aggregate(how, **kwargs)
        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
        if self.closed == "right":
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner
    def _upsample(self, method, limit: int | None = None, fill_value=None):
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values
        """
        # 如果从选择操作中进行上采样，则抛出错误
        if self._from_selection:
            raise ValueError(
                "Upsampling from level= or on= selection "
                "is not supported, use .set_index(...) "
                "to explicitly set index to datetime-like"
            )

        # 获取当前轴和选定的对象
        ax = self.ax
        obj = self._selected_obj
        # 获取分组器
        binner = self.binner
        # 调整分组器以适应上采样
        res_index = self._adjust_binner_for_upsample(binner)

        # 如果采样频率与轴的推断频率相同，并且对象长度与结果索引长度相等，则直接复制对象并调整索引
        if (
            limit is None
            and to_offset(ax.inferred_freq) == self.freq
            and len(obj) == len(res_index)
        ):
            result = obj.copy()
            result.index = res_index
        else:
            # 如果方法为"asfreq"，则将其置为None
            if method == "asfreq":
                method = None
            # 重新索引对象
            result = obj.reindex(
                res_index, method=method, limit=limit, fill_value=fill_value
            )

        # 包装并返回结果
        return self._wrap_result(result)

    def _wrap_result(self, result):
        # 调用父类的包装结果方法
        result = super()._wrap_result(result)

        # 可能会得到不同于初始要求的索引类型，需要进行类型转换
        if isinstance(self.ax, PeriodIndex) and not isinstance(
            result.index, PeriodIndex
        ):
            if isinstance(result.index, MultiIndex):
                # 如果结果索引是多级索引，则需要确保最后一个级别是周期索引
                if not isinstance(result.index.levels[-1], PeriodIndex):
                    new_level = result.index.levels[-1].to_period(self.freq)
                    result.index = result.index.set_levels(new_level, level=-1)
            else:
                # 否则，将索引转换为周期索引
                result.index = result.index.to_period(self.freq)
        
        # 返回处理后的结果
        return result
# error: Definition of "ax" in base class "_GroupByMixin" is incompatible
# with definition in base class "DatetimeIndexResampler"
class DatetimeIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, DatetimeIndexResampler
):
    """
    Provides a resample of a groupby implementation
    """

    @property
    def _resampler_cls(self):
        # 返回 DatetimeIndexResampler 类型
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    # error: Incompatible types in assignment (expression has type "PeriodIndex", base
    # class "DatetimeIndexResampler" defined the type as "DatetimeIndex")
    ax: PeriodIndex  # type: ignore[assignment]

    @property
    def _resampler_for_grouping(self):
        # 发出警告，表示使用 PeriodIndex 进行分组已经被废弃
        warnings.warn(
            "Resampling a groupby with a PeriodIndex is deprecated. "
            "Cast to DatetimeIndex before resampling instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        # 返回 PeriodIndexResamplerGroupby 类型
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self):
        if isinstance(self.ax, DatetimeIndex):
            # 如果 ax 是 DatetimeIndex 类型，则调用父类方法
            return super()._get_binner_for_time()
        # 否则使用 self._timegrouper._get_period_bins 方法处理 ax
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        obj = super()._convert_obj(obj)

        if self._from_selection:
            # 见 GH 14008, GH 12871，如果是从 level= 或 on= 选择中进行重采样，
            # 并且使用 PeriodIndex，目前不支持，应显式使用 .set_index(...) 设置索引
            msg = (
                "Resampling from level= or on= selection "
                "with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )
            raise NotImplementedError(msg)

        # 将对象转换为时间戳
        if isinstance(obj, DatetimeIndex):
            obj = obj.to_timestamp(how=self.convention)

        return obj
    def _downsample(self, how, **kwargs):
        """
        Downsample the data based on the specified method.

        Parameters
        ----------
        how : string / cython mapped function
            Method or function used for downsampling.
        **kwargs : kw args passed to how function
            Additional keyword arguments passed to the downsampling function.
        """
        # 检查是否时间戳索引，如果是，则调用父类方法进行降采样
        if isinstance(self.ax, DatetimeIndex):
            return super()._downsample(how, **kwargs)

        ax = self.ax

        # 如果当前频率是目标频率的子周期，则进行降采样
        if is_subperiod(ax.freq, self.freq):
            return self._groupby_and_aggregate(how, **kwargs)
        # 如果当前频率是目标频率的超周期
        elif is_superperiod(ax.freq, self.freq):
            if how == "ohlc":
                # 处理 OHLC 方法，通过 _groupby_and_aggregate() 处理
                # GH #13083
                # 将超周期向下采样处理为 asfreq 方法，适用于纯聚合/降维方法
                # OHLC 沿时间维度减少数据，但为每个周期创建多个值 -> 使用 _groupby_and_aggregate() 处理
                return self._groupby_and_aggregate(how)
            return self.asfreq()
        elif ax.freq == self.freq:
            return self.asfreq()

        # 抛出频率不兼容的异常
        raise IncompatibleFrequency(
            f"Frequency {ax.freq} cannot be resampled to {self.freq}, "
            "as they are not sub or super periods"
        )

    def _upsample(self, method, limit: int | None = None, fill_value=None):
        """
        Upsample the data based on the specified method.

        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.
        """
        # 检查是否时间戳索引，如果是，则调用父类方法进行上采样
        if isinstance(self.ax, DatetimeIndex):
            return super()._upsample(method, limit=limit, fill_value=fill_value)

        ax = self.ax
        obj = self.obj
        new_index = self.binner

        # 根据约定将时间轴转换为特定频率的索引
        memb = ax.asfreq(self.freq, how=self.convention)

        # 获取填充索引器
        if method == "asfreq":
            method = None
        indexer = memb.get_indexer(new_index, method=method, limit=limit)

        # 根据新索引取新的对象数据
        new_obj = _take_new_index(
            obj,
            indexer,
            new_index,
        )

        # 封装结果并返回
        return self._wrap_result(new_obj)
# error: 在基类"_GroupByMixin"中定义的"ax"与基类"PeriodIndexResampler"中的定义不兼容
class PeriodIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, PeriodIndexResampler
):
    """
    提供了一个基于分组的重新采样实现。
    """

    @property
    def _resampler_cls(self):
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    # error: 在赋值中存在不兼容的类型（表达式的类型为"TimedeltaIndex"，
    # 基类"DatetimeIndexResampler"将类型定义为"DatetimeIndex"）
    ax: TimedeltaIndex  # type: ignore[assignment]

    @property
    def _resampler_for_grouping(self):
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self):
        return self._timegrouper._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner):
        """
        在上采样时调整我们的分组器。

        新索引的范围允许大于原始范围，因此我们不需要改变分组器的长度，参见 GH 13022
        """
        return binner


# error: 在基类"_GroupByMixin"中定义的"ax"与基类"DatetimeIndexResampler"中的定义不兼容
class TimedeltaIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, TimedeltaIndexResampler
):
    """
    提供了一个基于分组的重新采样实现。
    """

    @property
    def _resampler_cls(self):
        return TimedeltaIndexResampler


def get_resampler(obj: Series | DataFrame, **kwds) -> Resampler:
    """
    创建一个TimeGrouper并返回我们的重新采样器。
    """
    tg = TimeGrouper(obj, **kwds)  # type: ignore[arg-type]
    return tg._get_resampler(obj)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby: GroupBy,
    rule,
    how=None,
    fill_method=None,
    limit: int | None = None,
    on=None,
    include_groups: bool = True,
    **kwargs,
) -> Resampler:
    """
    当进行分组时返回适当的重新采样器。
    """
    # .resample 使用 'on'，类似于 .groupby 使用 'key'
    tg = TimeGrouper(freq=rule, key=on, **kwargs)
    resampler = tg._get_resampler(groupby.obj)
    return resampler._get_resampler_for_grouping(
        groupby=groupby, include_groups=include_groups, key=tg.key
    )


class TimeGrouper(Grouper):
    """
    用于时间间隔分组的自定义groupby类。

    参数
    ----------
    freq : pandas日期偏移或偏移别名，用于标识区间边界
    closed : 区间的闭合端; 'left' 或 'right'
    label : 用于标记的区间边界; 'left' 或 'right'
    convention : {'start', 'end', 'e', 's'}
        如果轴是 PeriodIndex
    """

    _attributes = Grouper._attributes + (
        "closed",
        "label",
        "how",
        "convention",
        "origin",
        "offset",
    )

    origin: TimeGrouperOrigin
    def __init__(
        self,
        obj: Grouper | None = None,
        freq: Frequency = "Min",
        key: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        how: str = "mean",
        fill_method=None,
        limit: int | None = None,
        convention: Literal["start", "end", "e", "s"] | None = None,
        origin: Literal["epoch", "start", "start_day", "end", "end_day"]
        | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
        **kwargs,
    ):
        """
        Initialize a TimeGrouper object with specified parameters.

        Parameters
        ----------
        obj : Grouper or None, optional
            The object to be grouped, default is None.
        freq : Frequency, optional
            The frequency at which to resample, default is "Min".
        key : str or None, optional
            Grouping key, default is None.
        closed : {"left", "right"} or None, optional
            Which side of bin interval is closed, default is None.
        label : {"left", "right"} or None, optional
            Which bin edge label to label bucket with, default is None.
        how : str, optional
            Resampling method, default is "mean".
        fill_method : optional
            Fill method for missing data, default is None.
        limit : int or None, optional
            Maximum number of consecutive NaNs to fill, default is None.
        convention : {"start", "end", "e", "s"} or None, optional
            Resampling convention, default is None.
        origin : {"epoch", "start", "start_day", "end", "end_day"} or TimestampConvertibleTypes, optional
            Timestamp conversion origin, default is "start_day".
        offset : TimedeltaConvertibleTypes or None, optional
            Offset for resampling, default is None.
        group_keys : bool, optional
            Whether to group keys, default is False.
        **kwargs : keyword arguments
            Additional arguments.

        """
        
    def _get_resampler(self, obj: NDFrame) -> Resampler:
        """
        Return the appropriate resampler object based on the type of index.

        Parameters
        ----------
        obj : Series or DataFrame
            The object to be resampled.

        Returns
        -------
        Resampler
            Resampler object based on the type of index.

        Raises
        ------
        TypeError
            If the index type of obj is incompatible for resampling.

        """
        # Obtain the grouper for the object
        _, ax, _ = self._set_grouper(obj, gpr_index=None)
        
        # Check the type of index and return the corresponding resampler
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, PeriodIndex):
            if isinstance(ax, PeriodIndex):
                # Issue a warning for using PeriodIndex for resampling
                warnings.warn(
                    "Resampling with a PeriodIndex is deprecated. "
                    "Cast index to DatetimeIndex before resampling instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return PeriodIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        
        # Raise TypeError if the index type is not supported
        raise TypeError(
            "Only valid with DatetimeIndex, "
            "TimedeltaIndex or PeriodIndex, "
            f"but got an instance of '{type(ax).__name__}'"
        )

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
    ) -> tuple[BinGrouper, NDFrameT]:
        """
        Obtain the grouper and the object for resampling.

        Parameters
        ----------
        obj : NDFrameT
            The object to be resampled.
        validate : bool, optional
            Whether to validate the operation, default is True.

        Returns
        -------
        tuple
            A tuple containing the BinGrouper and the resampled object (NDFrameT).

        """
        # Obtain the resampler for the object
        r = self._get_resampler(obj)
        
        # Return the grouper and the resampled object
        return r._grouper, cast(NDFrameT, r.obj)
    def _get_time_bins(self, ax: DatetimeIndex):
        # 检查传入的轴是否为 DatetimeIndex 类型，如果不是则抛出类型错误异常
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        # 如果轴的长度为 0，则创建一个空的 DatetimeIndex，并返回空的 binner、空列表和空的 labels
        if len(ax) == 0:
            binner = labels = DatetimeIndex(
                data=[], freq=self.freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels

        # 获取轴的最小和最大时间戳，并根据频率和其他参数计算第一个和最后一个时间边界
        first, last = _get_timestamp_range_edges(
            ax.min(),
            ax.max(),
            self.freq,
            unit=ax.unit,
            closed=self.closed,
            origin=self.origin,
            offset=self.offset,
        )

        # 创建一个日期范围（DatetimeIndex），表示时间段（bins），用于标记和分组时间数据
        # 使用 first 和 last 作为开始和结束时间，配置时区和其他参数
        binner = labels = date_range(
            freq=self.freq,
            start=first,
            end=last,
            tz=ax.tz,
            name=ax.name,
            ambiguous=True,
            nonexistent="shift_forward",
            unit=ax.unit,
        )

        # 获取轴的整数表示形式，并调整时间边界，确保 bins（时间段）和标签（labels）的一致性
        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)

        # 使用通用方法生成时间段（bins），不考虑相对频率
        bins = lib.generate_bins_dt64(
            ax_values, bin_edges, self.closed, hasnans=ax.hasnans
        )

        # 根据时间段的闭合方式和标签的位置，调整标签
        if self.closed == "right":
            labels = binner
            if self.label == "right":
                labels = labels[1:]
        elif self.label == "right":
            labels = labels[1:]

        # 如果轴包含缺失值，将 NaT（Not a Time）插入到 binner 和 labels 的开头
        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)

        # 如果标签数量多于时间段数量，截断多余的标签
        if len(bins) < len(labels):
            labels = labels[: len(bins)]

        # 返回时间段（binner）、时间段的边界（bins）和标签（labels）
        return binner, bins, labels
    ) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]:
        # 对于超过每日数据的一些修正，参见 issue #1471, #1458, #1483

        # 检查频率是否属于 ("BME", "ME", "W") 或以 ("BQE", "BYE", "QE", "YE", "W") 开头的频率
        if self.freq.name in ("BME", "ME", "W") or self.freq.name.split("-")[0] in (
            "BQE",
            "BYE",
            "QE",
            "YE",
            "W",
        ):
            # 如果右端点在月份的最后一天，则向前滚动直到那一天的最后时刻。
            # 注意，我们仅对与超过每日周期的偏移量执行此操作，例如，“月初”被排除在外。
            if self.closed == "right":
                # GH 21459, GH 9119: 调整与 wall time 相关的边界时间
                edges_dti = binner.tz_localize(None)
                edges_dti = (
                    edges_dti
                    + Timedelta(days=1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                    - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                )
                bin_edges = edges_dti.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8

            # 最后一天的日内值
            if bin_edges[-2] > ax_values.max():
                bin_edges = bin_edges[:-1]
                binner = binner[:-1]
        else:
            bin_edges = binner.asi8
        return binner, bin_edges

    def _get_time_delta_bins(self, ax: TimedeltaIndex):
        # 检查输入的轴是否为 TimedeltaIndex 类型，否则引发 TypeError 异常
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(
                "axis must be a TimedeltaIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        # 检查频率是否为 Tick 类型，否则引发 ValueError 异常
        if not isinstance(self.freq, Tick):
            # GH#51896
            raise ValueError(
                "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
                f"e.g. '24h' or '3D', not {self.freq}"
            )

        # 如果轴的长度为 0，则返回空的 binner、labels 和 TimedeltaIndex
        if not len(ax):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels

        # 获取轴的最小值和最大值
        start, end = ax.min(), ax.max()

        # 如果关闭端点为 "right"，则向结束时间添加频率
        if self.closed == "right":
            end += self.freq

        # 创建以频率为步长的 TimedeltaIndex，作为 binner 和 labels
        labels = binner = timedelta_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        # 如果关闭端点为 "left"，则向结束时间添加频率
        end_stamps = labels
        if self.closed == "left":
            end_stamps += self.freq

        # 根据结束时间戳在轴上的搜索结果，得到 bins
        bins = ax.searchsorted(end_stamps, side=self.closed)

        # 如果存在偏移量，则将其应用到 labels
        if self.offset:
            # GH 10530 & 31809
            labels += self.offset

        return binner, bins, labels
    # 定义一个方法来获取时间段的分组边界
    def _get_time_period_bins(self, ax: DatetimeIndex):
        # 检查输入参数 ax 是否为 DatetimeIndex 类型，如果不是则抛出类型错误异常
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        # 获取时间频率（频率信息应该是从对象的属性中获取）
        freq = self.freq

        # 如果时间索引 ax 长度为 0，则创建空的周期索引对象，并返回空的结果
        if len(ax) == 0:
            binner = labels = PeriodIndex(
                data=[], freq=freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels

        # 创建包含起始和结束时间点的周期索引，使用指定的频率和名称
        labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)

        # 计算每个时间段的结束时间戳，并将其重新取整到给定的频率上，精度为秒
        end_stamps = (labels + freq).asfreq(freq, "s").to_timestamp()

        # 如果时间索引 ax 具有时区信息，则将结束时间戳设置为该时区
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)

        # 在时间索引 ax 上搜索结束时间戳的位置，返回每个时间段的索引位置数组 bins
        bins = ax.searchsorted(end_stamps, side="left")

        # 返回周期索引对象 binner、时间段索引 bins 和标签 labels
        return binner, bins, labels
    # 定义一个方法来获取时间周期的分箱信息，接受一个 PeriodIndex 类型的参数 ax
    def _get_period_bins(self, ax: PeriodIndex):
        # 检查传入的参数 ax 是否是 PeriodIndex 类型，如果不是则抛出类型错误
        if not isinstance(ax, PeriodIndex):
            raise TypeError(
                "axis must be a PeriodIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        # 将 ax 转换为给定频率和约定的时间周期
        memb = ax.asfreq(self.freq, how=self.convention)

        # 处理 NaT 值，与 pandas._lib.lib.generate_bins_dt64() 中的处理方式一致
        nat_count = 0
        if memb.hasnans:
            # 统计 memb 中 NaT 值的数量
            nat_count = np.sum(memb._isnan)  # type: ignore[assignment]
            # 去除 memb 中的 NaT 值
            memb = memb[~memb._isnan]

        # 如果 memb 中没有有效值（非 NaT），返回空数组和空的 PeriodIndex
        if not len(memb):
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            # 如果 ax 不为空，则标记所有元素为 NaT
            if len(ax) > 0:
                binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
            return binner, bins, labels

        # 计算频率的倍数
        freq_mult = self.freq.n

        # 获取 ax 的最小值和最大值，并按照给定频率和约定转换为 PeriodIndex
        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how="end")
        bin_shift = 0

        # 如果频率是 Tick 类型
        if isinstance(self.freq, Tick):
            # 获取调整后的 bin 边缘标签，适用于 'origin' 和 'origin' 支持
            p_start, end = _get_period_range_edges(
                start,
                end,
                self.freq,
                closed=self.closed,
                origin=self.origin,
                offset=self.offset,
            )

            # 计算开始偏移量，用于调整 bin 边缘（而非标签边缘）
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            # 计算 bin_shift，考虑到频率倍数
            bin_shift = start_offset.n % freq_mult  # type: ignore[union-attr]
            start = p_start

        # 创建起始到结束时间范围的 PeriodIndex 标签
        labels = binner = period_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        # 将 memb 转换为 int64 类型
        i8 = memb.asi8

        # 当向上采样到子周期时，需要生成足够的 bins 数量
        expected_bins_count = len(binner) * freq_mult
        i8_extend = expected_bins_count - (i8[-1] - i8[0])
        # 生成连续的整数数组，以 freq_mult 为步长
        rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
        rng += freq_mult
        # 调整 bin 边缘索引，以考虑基础偏移
        rng -= bin_shift

        # 使用 PeriodArray 将 rng 包装为 PeriodArray，以便 PeriodArray.searchsorted 使用
        prng = type(memb._data)(rng, dtype=memb.dtype)
        # 使用 memb 的 searchsorted 方法，找到 prng 中每个元素的插入点索引，返回 bins 数组
        bins = memb.searchsorted(prng, side="left")

        # 如果 nat_count 大于 0，则将对应的信息插入到 binner、bins 和 labels 中
        if nat_count > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)

        # 返回 binner（PeriodIndex）、bins 和 labels
        return binner, bins, labels
    # 定义一个方法 `_set_grouper`，用于设置分组键
    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        # 调用父类的 `_set_grouper` 方法来设置分组键，并获取返回的结果
        obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
        
        # 检查索引对象 `ax` 的数据类型是否为 ArrowDtype，且类型种类为日期时间或时间间隔
        if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in "Mm":
            # 将对象的 ArrowDtype 设置为 ax 的数据类型
            self._arrow_dtype = ax.dtype
            # 如果 ax 的数组是 ArrowExtensionArray 类型，则尝试转换为日期时间数组
            ax = Index(
                cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array()
            )
        
        # 返回设置好的 obj, ax, indexer 组成的元组
        return obj, ax, indexer
# 定义函数签名，接受 DataFrame 和 Series 作为输入，返回相应的 DataFrame 或 Series 对象
@overload
def _take_new_index(
    obj: DataFrame, indexer: npt.NDArray[np.intp], new_index: Index
) -> DataFrame: ...

# 定义函数签名，接受 DataFrame 和 Series 作为输入，返回相应的 DataFrame 或 Series 对象
@overload
def _take_new_index(
    obj: Series, indexer: npt.NDArray[np.intp], new_index: Index
) -> Series: ...

# 实现根据索引进行重新索引的内部函数，接受 DataFrame 或 Series 对象作为输入
def _take_new_index(
    obj: DataFrame | Series,
    indexer: npt.NDArray[np.intp],
    new_index: Index,
) -> DataFrame | Series:
    # 如果输入对象是 Series 类型
    if isinstance(obj, ABCSeries):
        # 使用算法库中的 take_nd 函数，根据 indexer 提取新的值数组
        new_values = algos.take_nd(obj._values, indexer)
        # 返回一个新的 Series 对象，使用原始 Series 的构造函数，指定新的值数组和索引
        return obj._constructor(new_values, index=new_index, name=obj.name)
    # 如果输入对象是 DataFrame 类型
    elif isinstance(obj, ABCDataFrame):
        # 使用管理器对象的 reindex_indexer 方法，重新索引新的管理器
        new_mgr = obj._mgr.reindex_indexer(new_axis=new_index, indexer=indexer, axis=1)
        # 返回一个新的 DataFrame 对象，使用原始 DataFrame 的构造函数，指定新的管理器和轴
        return obj._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
    else:
        # 抛出数值错误，提示输入对象应为 Series 或 DataFrame
        raise ValueError("'obj' should be either a Series or a DataFrame")

# 定义函数用于获取时间戳范围的边界值
def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: str,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    """
    调整 `first` 时间戳到前一个在提供偏移量上的时间戳。
    调整 `last` 时间戳到后一个在提供偏移量上的时间戳。
    输入的时间戳如果已经在偏移量上，根据偏移类型和 `closed` 参数进行调整。

    Parameters
    ----------
    first : pd.Timestamp
        要调整的范围的开始时间戳。
    last : pd.Timestamp
        要调整的范围的结束时间戳。
    freq : pd.DateOffset
        时间戳将要调整的日期偏移量。
    closed : {'right', 'left'}, 默认为 "left"
        闭合的区间的哪一侧。
    origin : {'epoch', 'start', 'start_day'} 或 Timestamp, 默认为 'start_day'
        调整分组的时间戳。起始时间戳的时区必须与索引的时区匹配。
        如果未使用时间戳，还支持以下值：

        - 'epoch': `origin` 是 1970-01-01
        - 'start': `origin` 是时间序列的第一个值
        - 'start_day': `origin` 是时间序列的午夜的第一天
    offset : pd.Timedelta, 默认为 None
        添加到起始时间戳的偏移量。

    Returns
    -------
    长度为 2 的元组，包含调整后的 pd.Timestamp 对象。
    """
    # 如果频率（freq）是 Tick 类的实例
    if isinstance(freq, Tick):
        # 获取索引的时区
        index_tz = first.tz
        # 如果 origin 是 Timestamp 类型，并且其时区与索引时区不一致，则抛出异常
        if isinstance(origin, Timestamp) and (origin.tz is None) != (index_tz is None):
            raise ValueError("The origin must have the same timezone as the index.")
        # 如果 origin 等于 "epoch"
        if origin == "epoch":
            # 根据索引时区设置 epoch 时间，以便在不同时区上进行相同类型索引的重新采样时，结果相似
            origin = Timestamp("1970-01-01", tz=index_tz)

        # 如果频率（freq）是 Day 类的实例
        if isinstance(freq, Day):
            # _adjust_dates_anchored 假定 'D' 表示 24 小时，但 first/last 可能包含夏令时转换（23h、24h 或 25h）。
            # 因此，在调整端点时，“假装”日期是 naive（无时区）的
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            # 如果 origin 是 Timestamp 类型，则将其转换为 naive（无时区）的
            if isinstance(origin, Timestamp):
                origin = origin.tz_localize(None)

        # 调用 _adjust_dates_anchored 函数调整 first 和 last 的时间戳，
        # 根据频率（freq）、闭合方式（closed）、origin、offset 和 unit 来调整
        first, last = _adjust_dates_anchored(
            first, last, freq, closed=closed, origin=origin, offset=offset, unit=unit
        )

        # 如果频率（freq）是 Day 类的实例
        if isinstance(freq, Day):
            # 将 first 和 last 时间戳重新设置为索引时区
            first = first.tz_localize(index_tz)
            last = last.tz_localize(index_tz)
    else:
        # 如果频率（freq）不是 Tick 类的实例，则对 first 和 last 进行规范化处理
        first = first.normalize()
        last = last.normalize()

        # 如果闭合方式是 "left"
        if closed == "left":
            # 根据频率（freq）向前滚动 first 的时间戳
            first = Timestamp(freq.rollback(first))
        else:
            # 否则，根据频率（freq）向前调整 first 的时间戳
            first = Timestamp(first - freq)

        # 根据频率（freq）向后调整 last 的时间戳
        last = Timestamp(last + freq)

    # 返回调整后的 first 和 last 时间戳
    return first, last
# 调整给定的 `first` 和 `last` Periods，使其适应包含它们的给定偏移量的周期。

def _get_period_range_edges(
    first: Period,
    last: Period,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Period, Period]:
    """
    Adjust the provided `first` and `last` Periods to the respective Period of
    the given offset that encompasses them.

    Parameters
    ----------
    first : pd.Period
        The beginning Period of the range to be adjusted.
    last : pd.Period
        The ending Period of the range to be adjusted.
    freq : pd.DateOffset
        The freq to which the Periods will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'}, Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.

        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Period objects.
    """

    # 如果 `first` 和 `last` 不是 Period 类型的实例，则抛出类型错误异常
    if not all(isinstance(obj, Period) for obj in [first, last]):
        raise TypeError("'first' and 'last' must be instances of type Period")

    # GH 23882
    # 将 `first` 和 `last` 转换为对应的时间戳
    first_ts = first.to_timestamp()
    last_ts = last.to_timestamp()
    # 判断是否需要调整 `first` 和 `last` 时间戳
    adjust_first = not freq.is_on_offset(first_ts)
    adjust_last = freq.is_on_offset(last_ts)

    # 调用 `_get_timestamp_range_edges` 函数调整时间戳范围，返回调整后的时间戳
    first_ts, last_ts = _get_timestamp_range_edges(
        first_ts, last_ts, freq, unit="ns", closed=closed, origin=origin, offset=offset
    )

    # 根据需要调整的标志调整 `first` 和 `last` Period 对象
    first = (first_ts + int(adjust_first) * freq).to_period(freq)
    last = (last_ts - int(adjust_last) * freq).to_period(freq)
    return first, last


# 处理包含 NaT 的时间段索引
def _insert_nat_bin(
    binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int
) -> tuple[PeriodIndex, np.ndarray, PeriodIndex]:
    # NaT 处理，与 pandas._lib.lib.generate_bins_dt64() 中的处理方式一致
    # 将 bins 向右移动 NaT 的数量
    assert nat_count > 0
    bins += nat_count
    # 在 bins 数组开头插入 NaT 的位置
    bins = np.insert(bins, 0, nat_count)

    # 在 binner 中插入 NaT，类型注释忽略赋值不兼容的警告
    binner = binner.insert(0, NaT)  # type: ignore[assignment]
    # 在 labels 中插入 NaT，类型注释忽略赋值不兼容的警告
    labels = labels.insert(0, NaT)  # type: ignore[assignment]
    return binner, bins, labels


# 调整给定的时间戳 `first` 和 `last`，使其适应给定的频率 `freq`
def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
    unit: str = "ns",
# 返回一个元组，包含调整后的第一个和最后一个时间戳
def ) -> tuple[Timestamp, Timestamp]:
    # 从起始日期计算第一个和最后一个偏移量，以修复在多天重新采样时，一天周期不是频率的倍数而引起的错误。参见 GH 8683
    # 为处理不是天数倍数或不可被天数整除的频率，我们允许定义一个固定的起始时间戳。参见 GH 31809
    first = first.as_unit(unit)  # 将第一个时间戳转换为指定单位
    last = last.as_unit(unit)  # 将最后一个时间戳转换为指定单位
    if offset is not None:
        offset = offset.as_unit(unit)  # 如果有偏移量，也将其转换为指定单位

    freq_value = Timedelta(freq).as_unit(unit)._value  # 将频率转换为指定单位的时间增量值

    origin_timestamp = 0  # 初始时间戳设为0，表示“epoch”
    if origin == "start_day":
        origin_timestamp = first.normalize()._value  # 如果起始点是“start_day”，取第一个时间戳的规范化值
    elif origin == "start":
        origin_timestamp = first._value  # 如果起始点是“start”，直接取第一个时间戳的值
    elif isinstance(origin, Timestamp):
        origin_timestamp = origin.as_unit(unit)._value  # 如果起始点是时间戳对象，则取其转换为指定单位的值
    elif origin in ["end", "end_day"]:
        origin_last = last if origin == "end" else last.ceil("D")  # 如果起始点是“end”或“end_day”，取最后一个时间戳
        sub_freq_times = (origin_last._value - first._value) // freq_value  # 计算频率的整数倍数
        if closed == "left":
            sub_freq_times += 1
        first = origin_last - sub_freq_times * freq  # 计算新的第一个时间戳
        origin_timestamp = first._value
    origin_timestamp += offset._value if offset else 0  # 加上偏移量的值（如果存在）

    # GH 10117 & GH 19375. 如果第一个和最后一个时间戳包含时区信息，
    # 则在 UTC 中执行计算，以避免在模糊或不存在的时间上进行本地化。
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert("UTC")  # 如果第一个时间戳有时区信息，则转换为UTC时区
    if last_tzinfo is not None:
        last = last.tz_convert("UTC")  # 如果最后一个时间戳有时区信息，则转换为UTC时区

    foffset = (first._value - origin_timestamp) % freq_value  # 计算第一个时间戳相对于起始时间戳的偏移量
    loffset = (last._value - origin_timestamp) % freq_value  # 计算最后一个时间戳相对于起始时间戳的偏移量

    if closed == "right":
        if foffset > 0:
            fresult_int = first._value - foffset  # 如果偏移量大于0，向前滚动第一个时间戳
        else:
            fresult_int = first._value - freq_value  # 否则，第一个时间戳向前滚动一个完整的频率

        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)  # 如果偏移量大于0，向后滚动最后一个时间戳
        else:
            lresult_int = last._value  # 否则，最后一个时间戳保持不变
    else:  # closed == 'left'
        if foffset > 0:
            fresult_int = first._value - foffset  # 如果偏移量大于0，向前滚动第一个时间戳
        else:
            fresult_int = first._value  # 否则，第一个时间戳保持不变

        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)  # 如果偏移量大于0，向后滚动最后一个时间戳
        else:
            lresult_int = last._value + freq_value  # 否则，最后一个时间戳向后滚动一个完整的频率

    fresult = Timestamp(fresult_int, unit=unit)  # 创建调整后的第一个时间戳对象
    lresult = Timestamp(lresult_int, unit=unit)  # 创建调整后的最后一个时间戳对象
    if first_tzinfo is not None:
        fresult = fresult.tz_localize("UTC").tz_convert(first_tzinfo)  # 如果第一个时间戳有时区信息，则本地化和转换回原时区
    if last_tzinfo is not None:
        lresult = lresult.tz_localize("UTC").tz_convert(last_tzinfo)  # 如果最后一个时间戳有时区信息，则本地化和转换回原时区
    return fresult, lresult  # 返回调整后的第一个和最后一个时间戳的元组

# 对给定对象按照指定频率重新采样，并返回重新采样后的对象
def asfreq(
    obj: NDFrameT,
    freq,
    method=None,
    how=None,
    normalize: bool = False,
    fill_value=None,
) -> NDFrameT:
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
    # 检查索引是否为 PeriodIndex 类型
    if isinstance(obj.index, PeriodIndex):
        # 如果指定了 method 参数，抛出未实现的错误
        if method is not None:
            raise NotImplementedError("'method' argument is not supported")

        # 如果 how 参数为 None，则设为默认值 "E"
        if how is None:
            how = "E"

        # 如果 freq 参数是 BaseOffset 类型，则转换为 PeriodDtype 的频率字符串
        if isinstance(freq, BaseOffset):
            if hasattr(freq, "_period_dtype_code"):
                freq = PeriodDtype(freq)._freqstr

        # 复制原始对象
        new_obj = obj.copy()
        # 将索引转换为指定频率和处理方式的 PeriodIndex
        new_obj.index = obj.index.asfreq(freq, how=how)

    # 如果索引长度为 0，则复制原始对象，并根据 freq 调整索引
    elif len(obj.index) == 0:
        new_obj = obj.copy()
        new_obj.index = _asfreq_compat(obj.index, freq)

    else:
        unit = None
        # 如果索引是 DatetimeIndex 类型
        if isinstance(obj.index, DatetimeIndex):
            # TODO: 是否禁止非 DatetimeIndex 类型？
            unit = obj.index.unit
        # 根据最小和最大日期创建日期范围 dti
        dti = date_range(obj.index.min(), obj.index.max(), freq=freq, unit=unit)
        dti.name = obj.index.name
        # 重新索引对象，可选地使用 method 和 fill_value 进行填充
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        # 如果 normalize 为 True，则将索引标准化
        if normalize:
            new_obj.index = new_obj.index.normalize()

    # 返回调整后的新对象
    return new_obj
# 定义一个函数 `_asfreq_compat`，用于在空的日期时间索引（`DatetimeIndex`）或时间增量索引（`TimedeltaIndex`）上模拟 `asfreq` 的功能。
def _asfreq_compat(index: FreqIndexT, freq) -> FreqIndexT:
    """
    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.

    Parameters
    ----------
    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex
        输入参数，可以是周期索引、日期时间索引或时间增量索引。
    freq : DateOffset
        输入参数，表示要应用的日期偏移量。

    Returns
    -------
    FreqIndexT
        返回与输入索引相同类型的索引对象。
    """
    # 如果索引不为空，则抛出 ValueError 异常，因为此函数只能用于空的日期时间索引或时间增量索引。
    if len(index) != 0:
        raise ValueError(
            "Can only set arbitrary freq for empty DatetimeIndex or TimedeltaIndex"
        )
    # 根据输入索引的类型执行不同的操作
    if isinstance(index, PeriodIndex):
        new_index = index.asfreq(freq=freq)  # 将周期索引转换为指定频率的新索引
    elif isinstance(index, DatetimeIndex):
        new_index = DatetimeIndex([], dtype=index.dtype, freq=freq, name=index.name)
        # 创建空的日期时间索引，设置其数据类型、频率和名称
    elif isinstance(index, TimedeltaIndex):
        new_index = TimedeltaIndex([], dtype=index.dtype, freq=freq, name=index.name)
        # 创建空的时间增量索引，设置其数据类型、频率和名称
    else:  # pragma: no cover
        raise TypeError(type(index))  # 如果索引类型不符合预期，抛出 TypeError 异常
    return new_index  # 返回新创建的索引对象


def _apply(
    grouped: GroupBy, how: Callable, *args, include_groups: bool, **kwargs
) -> DataFrame:
    """
    Apply a function `how` to each group in `grouped`.

    Parameters
    ----------
    grouped : GroupBy
        输入参数，表示分组后的数据集。
    how : Callable
        输入参数，表示要应用于每个分组的函数。
    *args
        其他位置参数，传递给 `how` 函数。
    include_groups : bool
        输入参数，表示是否在应用函数时包含分组列。
    **kwargs
        其他关键字参数，传递给 `how` 函数。

    Returns
    -------
    DataFrame
        返回包含应用结果的数据帧。
    """
    # 重写警告消息，使其看起来来自 `.resample`
    target_message = "DataFrameGroupBy.apply operated on the grouping columns"
    new_message = _apply_groupings_depr.format("DataFrameGroupBy", "resample")
    with rewrite_warning(
        target_message=target_message,
        target_category=DeprecationWarning,
        new_message=new_message,
    ):
        # 应用函数 `how` 到分组对象 `grouped` 上，并返回结果
        result = grouped.apply(how, *args, include_groups=include_groups, **kwargs)
    return result  # 返回应用函数后的结果数据帧
```