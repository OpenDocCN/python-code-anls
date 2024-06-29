# `D:\src\scipysrc\pandas\pandas\core\indexes\interval.py`

```
# 引入未来的类型注解支持，允许在类型注解中使用类型本身
from __future__ import annotations

# 导入比较操作符中的 le（小于等于）和 lt（小于）函数
from operator import (
    le,
    lt,
)

# 导入文本包装模块
import textwrap

# 导入类型提示相关模块
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 导入 pandas 库的 C 扩展模块
from pandas._libs import lib

# 导入 pandas 库中与区间相关的模块
from pandas._libs.interval import (
    Interval,
    IntervalMixin,
    IntervalTree,
)

# 导入 pandas 库中与时间序列相关的模块
from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)

# 导入 pandas 库中的错误处理模块
from pandas.errors import InvalidIndexError

# 导入 pandas 工具模块中的装饰器相关函数
from pandas.util._decorators import (
    Appender,
    cache_readonly,
)

# 导入 pandas 工具模块中的异常处理相关函数
from pandas.util._exceptions import rewrite_exception

# 导入 pandas 核心数据类型转换相关函数
from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    maybe_box_datetimelike,
    maybe_downcast_numeric,
    maybe_upcast_numeric_to_64bit,
)

# 导入 pandas 核心数据类型判别函数
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_object_dtype,
    is_scalar,
    pandas_dtype,
)

# 导入 pandas 核心数据类型定义
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)

# 导入 pandas 核心数据类型缺失值处理相关函数
from pandas.core.dtypes.missing import is_valid_na_for_dtype

# 导入 pandas 核心算法模块
from pandas.core.algorithms import unique

# 导入 pandas 核心数组相关模块
from pandas.core.arrays.datetimelike import validate_periods

# 导入 pandas 核心数组中的区间数组和共享文档
from pandas.core.arrays.interval import (
    IntervalArray,
    _interval_shared_docs,
)

# 导入 pandas 核心通用功能模块
import pandas.core.common as com

# 导入 pandas 核心索引模块
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase

# 导入 pandas 核心索引中的基础类和共享文档
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    ensure_index,
    maybe_extract_name,
)

# 导入 pandas 核心日期时间索引相关模块
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)

# 导入 pandas 核心扩展索引相关模块
from pandas.core.indexes.extension import (
    ExtensionIndex,
    inherit_names,
)

# 导入 pandas 核心多级索引模块
from pandas.core.indexes.multi import MultiIndex

# 导入 pandas 核心时间差索引相关模块
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

# 如果是类型检查模式，导入哈希类型支持
if TYPE_CHECKING:
    from collections.abc import Hashable

    # 导入 pandas 中的类型提示相关定义
    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Self,
        npt,
    )

# 复制索引文档关键字参数
_index_doc_kwargs = dict(ibase._index_doc_kwargs)

# 更新索引文档关键字参数，用于 IntervalIndex
_index_doc_kwargs.update(
    {
        "klass": "IntervalIndex",
        "qualname": "IntervalIndex",
        "target_klass": "IntervalIndex or list of Intervals",
        "name": textwrap.dedent(
            """\
         name : object, optional
              Name to be stored in the index.
         """
        ),
    }
)

# 定义一个函数用于获取下一个标签
def _get_next_label(label):
    # 查看测试用例 test_slice_locs_with_ints_and_floats_succeeds
    # 获取标签的数据类型
    dtype = getattr(label, "dtype", type(label))

    # 如果标签是 Timestamp 或 Timedelta 类型，则将数据类型设为 datetime64[ns]
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"

    # 将数据类型转换为 pandas 中的数据类型对象
    dtype = pandas_dtype(dtype)

    # 如果数据类型是时间类型或具有 'mM' 类型（分钟或月份）或是带时区的日期时间类型
    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        # 返回增加一纳秒后的标签值
        return label + np.timedelta64(1, "ns")
    # 如果数据类型是整数类型
    elif is_integer_dtype(dtype):
        # 返回标签值加一
        return label + 1
    # 如果数据类型是浮点数类型
    elif is_float_dtype(dtype):
        # 返回比标签值大一点点的浮点数
        return np.nextafter(label, np.inf)
    else:
        # 如果不是已知的类型，则抛出类型错误异常，指明无法确定下一个标签的类型
        raise TypeError(f"cannot determine next label for type {type(label)!r}")
# 定义一个私有函数 `_get_prev_label`，用于获取给定标签的前一个标签
def _get_prev_label(label):
    # 获取标签的数据类型，如果标签是 Timestamp 或 Timedelta 对象，则将数据类型设为 "datetime64[ns]"
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    # 将数据类型转换为 pandas 中的数据类型
    dtype = pandas_dtype(dtype)

    # 根据数据类型判断如何计算前一个标签
    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        # 如果数据类型是日期时间相关的，使用 np.timedelta64 减去 1 纳秒
        return label - np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        # 如果数据类型是整数类型，直接减去 1
        return label - 1
    elif is_float_dtype(dtype):
        # 如果数据类型是浮点数类型，使用 np.nextafter 函数获取比当前标签小一点点的浮点数
        return np.nextafter(label, -np.inf)
    else:
        # 如果无法确定如何计算前一个标签，抛出类型错误异常
        raise TypeError(f"cannot determine next label for type {type(label)!r}")


# 定义一个私有函数 `_new_IntervalIndex`，用于从字典 d 中创建新的 IntervalIndex 对象
def _new_IntervalIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """
    return cls.from_arrays(**d)


# 使用装饰器定义 IntervalIndex 类，该类继承自 ExtensionIndex
@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalIndex",
        "summary": "Immutable index of intervals that are closed on the same side.",
        "name": _index_doc_kwargs["name"],
        "extra_attributes": "is_overlapping\nvalues\n",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalIndex`` is typically constructed using
    :func:`interval_range`:

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalIndex.from_arrays`,
    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.

    See further examples in the doc strings of ``interval_range`` and the
    mentioned constructor methods.
    """
        ),
    }
)
@inherit_names(["set_closed", "to_tuples"], IntervalArray, wrap=True)
@inherit_names(
    [
        "__array__",
        "overlaps",
        "contains",
        "closed_left",
        "closed_right",
        "open_left",
        "open_right",
        "is_empty",
    ],
    IntervalArray,
)
@inherit_names(["is_non_overlapping_monotonic", "closed"], IntervalArray, cache=True)
# 定义 IntervalIndex 类，用于表示区间的不可变索引，左右两侧的区间闭合
class IntervalIndex(ExtensionIndex):
    _typ = "intervalindex"

    # 定义 pinned 属性，这些属性是通过 inherit_names 方法固定的
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool

    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings = False
    _data_cls = IntervalArray

    # --------------------------------------------------------------------
    # Constructors

    # 定义构造函数 __new__，用于创建 IntervalIndex 对象
    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
        verify_integrity: bool = True,
    ) -> Self:
        # 从数据中可能提取名称，使用 maybe_extract_name 函数
        name = maybe_extract_name(name, data, cls)

        # 使用 rewrite_exception 包装，处理 IntervalArray 异常
        with rewrite_exception("IntervalArray", cls.__name__):
            # 使用 IntervalArray 类创建一个数组对象
            array = IntervalArray(
                data,
                closed=closed,
                copy=copy,
                dtype=dtype,
                verify_integrity=verify_integrity,
            )

        # 使用类方法 _simple_new 创建新的实例，并返回
        return cls._simple_new(array, name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        # 使用 rewrite_exception 包装，处理 IntervalArray 异常
        with rewrite_exception("IntervalArray", cls.__name__):
            # 使用 IntervalArray 类的 from_breaks 方法创建数组对象
            array = IntervalArray.from_breaks(
                breaks, closed=closed, copy=copy, dtype=dtype
            )
        # 使用类方法 _simple_new 创建新的实例，并返回
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        # 使用 rewrite_exception 包装，处理 IntervalArray 异常
        with rewrite_exception("IntervalArray", cls.__name__):
            # 使用 IntervalArray 类的 from_arrays 方法创建数组对象
            array = IntervalArray.from_arrays(
                left, right, closed, copy=copy, dtype=dtype
            )
        # 使用类方法 _simple_new 创建新的实例，并返回
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
                name : str, optional
                    Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
                Examples
                --------
                >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
                IntervalIndex([(0, 1], (1, 2]],
                               dtype='interval[int64, right]')
                """
            ),
        }
    )
    # 定义一个装饰器，用于添加文档字符串到 from_tuples 方法
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        # 使用 rewrite_exception 上下文管理器来捕获并处理异常
        with rewrite_exception("IntervalArray", cls.__name__):
            # 调用 IntervalArray 的 from_tuples 方法，生成一个 IntervalArray 对象
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        # 调用当前类的 _simple_new 方法创建一个新的 IntervalIndex 对象
        return cls._simple_new(arr, name=name)

    # --------------------------------------------------------------------
    # error: Return type "IntervalTree" of "_engine" incompatible with return type
    # "Union[IndexEngine, ExtensionEngine]" in supertype "Index"
    @cache_readonly
    # 定义一个只读缓存属性，返回 IntervalTree 对象
    def _engine(self) -> IntervalTree:  # type: ignore[override]
        # 将 self.left 和 self.right 转换为 64 位整数（如果可能）
        left = self._maybe_convert_i8(self.left)
        left = maybe_upcast_numeric_to_64bit(left)
        right = self._maybe_convert_i8(self.right)
        right = maybe_upcast_numeric_to_64bit(right)
        # 创建并返回一个 IntervalTree 对象
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key: Any) -> bool:
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
        # 对 key 进行哈希处理
        hash(key)
        # 如果 key 不是 Interval 类型
        if not isinstance(key, Interval):
            # 检查 key 是否是特定数据类型的 NaN 值，如果是返回 self.hasnans，否则返回 False
            if is_valid_na_for_dtype(key, self.dtype):
                return self.hasnans
            return False

        try:
            # 尝试获取 key 在索引中的位置，如果成功返回 True
            self.get_loc(key)
            return True
        except KeyError:
            # 如果获取位置失败，返回 False
            return False

    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        # 从 self._data 中获取 slobj 对应的切片结果
        res = self._data[slobj]
        # 调用当前类的 _simple_new 方法创建一个新的 IntervalIndex 对象，并返回
        return type(self)._simple_new(res, name=self._name)

    @cache_readonly
    # 定义一个只读缓存属性，返回 MultiIndex 对象
    def _multiindex(self) -> MultiIndex:
        # 使用 self.left 和 self.right 创建 MultiIndex 对象，指定列名为 "left" 和 "right"
        return MultiIndex.from_arrays([self.left, self.right], names=["left", "right"])

    def __reduce__(self):
        # 返回用于反序列化对象的数据
        d = {
            "left": self.left,
            "right": self.right,
            "closed": self.closed,
            "name": self.name,
        }
        return _new_IntervalIndex, (type(self), d), None

    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values"""
        # 返回推断出的值的类型的字符串描述
        return "interval"
    # 将 Index 类中的 memory_usage 文档字符串附加到当前方法上
    @Appender(Index.memory_usage.__doc__)  # type: ignore[has-type]
    def memory_usage(self, deep: bool = False) -> int:
        # 返回左右子索引的内存使用量之和，因为没有明确指定引擎
        # 因此在这里返回字节大小
        return self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep)

    # 由于 IntervalTree 没有 is_monotonic_decreasing 方法，需要重写 Index 实现
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        如果 IntervalIndex 是单调递减的（仅包含相等或递减的值），返回 True，否则返回 False
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def is_unique(self) -> bool:
        """
        如果 IntervalIndex 包含唯一元素，返回 True，否则返回 False。

        1. 检查是否有缺失值，如果超过 1 个，则直接返回 False。
        2. 检查左右子索引是否有唯一值，如果有任意一个是唯一的，则返回 True。
        3. 使用集合 seen_pairs 来记录已经出现过的左右子索引对，检查重复情况，如果有重复则返回 False。

        Returns
        -------
        bool
            表示 IntervalIndex 是否包含唯一元素的布尔值。
        """

        left = self.left
        right = self.right

        if self.isna().sum() > 1:
            return False

        if left.is_unique or right.is_unique:
            return True

        seen_pairs = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)

        return True

    @property
    def is_overlapping(self) -> bool:
        """
        如果 IntervalIndex 中存在重叠的区间，返回 True，否则返回 False。

        两个区间如果共享一个端点（包括闭合端点），则认为它们重叠。只有共享开放端点的区间不算重叠。

        Returns
        -------
        bool
            表示 IntervalIndex 是否包含重叠区间的布尔值。

        See Also
        --------
        Interval.overlaps : 检查两个 Interval 对象是否重叠。
        IntervalIndex.overlaps : 对 IntervalIndex 逐个元素检查是否重叠。

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        共享闭合端点的区间重叠：

        >>> index = pd.interval_range(0, 3, closed="both")
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        只共享开放端点的区间不重叠：

        >>> index = pd.interval_range(0, 3, closed="left")
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
        """
        # GH 23309
        return self._engine.is_overlapping
    def _needs_i8_conversion(self, key) -> bool:
        """
        Check if a given key needs i8 conversion. Conversion is necessary for
        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An
        Interval-like requires conversion if its endpoints are one of the
        aforementioned types.

        Assumes that any list-like data has already been cast to an Index.

        Parameters
        ----------
        key : scalar or Index-like
            The key that should be checked for i8 conversion

        Returns
        -------
        bool
            True if the key needs i8 conversion, otherwise False.
        """
        # 获取key的数据类型
        key_dtype = getattr(key, "dtype", None)
        
        # 如果key的数据类型是IntervalDtype或者key本身是Interval类型
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            # 递归调用以检查Interval的左端点是否需要i8转换
            return self._needs_i8_conversion(key.left)

        # 定义需要i8转换的类型集合
        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        
        # 检查key是否属于需要i8转换的类型集合
        return isinstance(key, i8_types)
    def _maybe_convert_i8(self, key):
        """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
        # 如果 key 是类似列表的对象
        if is_list_like(key):
            # 确保 key 是索引对象
            key = ensure_index(key)
            # 可能将数值类型提升为 64 位
            key = maybe_upcast_numeric_to_64bit(key)

        # 如果不需要进行 i8 转换，直接返回原始的 key
        if not self._needs_i8_conversion(key):
            return key

        # 判断 key 是否是标量
        scalar = is_scalar(key)
        # 获取 key 的数据类型
        key_dtype = getattr(key, "dtype", None)
        # 如果 key 的数据类型是 IntervalDtype 或者 key 是 Interval 对象
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            # 转换左右端点并重新构建 Interval 对象或 IntervalIndex 对象
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            # 根据是否是标量选择构造函数
            constructor = Interval if scalar else IntervalIndex.from_arrays
            # 返回构造的 Interval 或 IntervalIndex 对象
            return constructor(left, right, closed=self.closed)  # type: ignore[operator]

        # 如果 key 是标量
        if scalar:
            # 推断标量的数据类型和对应的 i8 值
            key_dtype, key_i8 = infer_dtype_from_scalar(key)
            # 如果 key 是 Period 对象，则获取其 ordinal 值
            if isinstance(key, Period):
                key_i8 = key.ordinal
            # 如果 key_i8 是 Timestamp 对象，则获取其 _value 值
            elif isinstance(key_i8, Timestamp):
                key_i8 = key_i8._value
            # 如果 key_i8 是 np.datetime64 或 np.timedelta64 对象，则转换为 "i8" 类型
            elif isinstance(key_i8, (np.datetime64, np.timedelta64)):
                key_i8 = key_i8.view("i8")
        else:
            # 如果 key 是 DatetimeIndex 或 TimedeltaIndex 对象
            key_dtype, key_i8 = key.dtype, Index(key.asi8)
            # 如果 key 中包含 NaN 值
            if key.hasnans:
                # 将 NaT 的 i8 值转换为 np.nan，以避免其被视为有效值，可能导致错误
                key_i8 = key_i8.where(~key._isnan)

        # 确保与 IntervalIndex 子类型的一致性
        subtype = self.dtype.subtype  # type: ignore[union-attr]

        # 如果 subtype 和 key 的数据类型不一致，则抛出 ValueError 异常
        if subtype != key_dtype:
            raise ValueError(
                f"Cannot index an IntervalIndex of subtype {subtype} with "
                f"values of dtype {key_dtype}"
            )

        # 返回转换后的 key_i8
        return key_i8
    # 在非重叠单调性索引上执行搜索，返回指定标签的位置
    def _searchsorted_monotonic(self, label, side: Literal["left", "right"] = "left"):
        # 如果索引不是非重叠单调的，抛出错误
        if not self.is_non_overlapping_monotonic:
            raise KeyError(
                "can only get slices from an IntervalIndex if bounds are "
                "non-overlapping and all monotonic increasing or decreasing"
            )

        # 如果标签是区间对象或区间索引，目前不支持，抛出未实现错误
        if isinstance(label, (IntervalMixin, IntervalIndex)):
            raise NotImplementedError("Interval objects are not currently supported")

        # 根据参数 `side` 的值选择左边界或右边界进行搜索
        # GH 20921: 对于第二个条件，使用 "not is_monotonic_increasing" 而不是 "is_monotonic_decreasing"
        # 这样可以处理单元素索引既增又减的情况
        if (side == "left" and self.left.is_monotonic_increasing) or (
            side == "right" and not self.left.is_monotonic_increasing
        ):
            # 如果选择左边界并且左边界是单调递增的，使用右边界进行搜索
            sub_idx = self.right
            # 如果开启了右开区间，调整标签为下一个标签
            if self.open_right:
                label = _get_next_label(label)
        else:
            # 否则使用左边界进行搜索
            sub_idx = self.left
            # 如果开启了左开区间，调整标签为前一个标签
            if self.open_left:
                label = _get_prev_label(label)

        # 递归调用子索引的 `_searchsorted_monotonic` 方法，返回结果
        return sub_idx._searchsorted_monotonic(label, side)

    # --------------------------------------------------------------------
    # 索引方法
    def get_loc(self, key) -> int | slice | np.ndarray:
        """
        Get integer location, slice or boolean mask for requested label.

        The `get_loc` method is used to retrieve the integer index, a slice for
        slicing objects, or a boolean mask indicating the presence of the label
        in the `IntervalIndex`.

        Parameters
        ----------
        key : label
            The value or range to find in the IntervalIndex.

        Returns
        -------
        int if unique index, slice if monotonic index, else mask
            The position or positions found. This could be a single
            number, a range, or an array of true/false values
            indicating the position(s) of the label.

        See Also
        --------
        IntervalIndex.get_indexer_non_unique : Compute indexer and
            mask for new index given the current index.
        Index.get_loc : Similar method in the base Index class.

        Examples
        --------
        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)
        >>> index = pd.IntervalIndex([i1, i2])
        >>> index.get_loc(1)
        0

        You can also supply a point inside an interval.

        >>> index.get_loc(1.5)
        1

        If a label is in several intervals, you get the locations of all the
        relevant intervals.

        >>> i3 = pd.Interval(0, 2)
        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])
        >>> overlapping_index.get_loc(0.5)
        array([ True, False,  True])

        Only exact matches will be returned if an interval is provided.

        >>> index.get_loc(pd.Interval(0, 1))
        0
        """
        # 检查索引是否有效，若无效则抛出错误
        self._check_indexing_error(key)

        # 如果key是Interval类型
        if isinstance(key, Interval):
            # 检查区间的闭合性是否与当前对象一致，若不一致则抛出错误
            if self.closed != key.closed:
                raise KeyError(key)
            # 创建一个布尔掩码，标识与key具有相同左右端点的位置
            mask = (self.left == key.left) & (self.right == key.right)
        # 如果key是NaN或者缺失值
        elif is_valid_na_for_dtype(key, self.dtype):
            # 创建一个布尔掩码，标识所有缺失值的位置
            mask = self.isna()
        else:
            # 假设key是标量值
            # 根据区间的闭合性选择适当的比较操作符
            op_left = le if self.closed_left else lt
            op_right = le if self.closed_right else lt
            try:
                # 创建一个布尔掩码，标识key在区间内的位置
                mask = op_left(self.left, key) & op_right(key, self.right)
            except TypeError as err:
                # 如果标量值无法比较，抛出错误
                raise KeyError(key) from err

        # 计算匹配项的数量
        matches = mask.sum()
        # 如果没有匹配项，抛出错误
        if matches == 0:
            raise KeyError(key)
        # 如果只有一个匹配项，返回该项的位置
        if matches == 1:
            return mask.argmax()

        # 将布尔掩码转换为切片或者保持为布尔数组
        res = lib.maybe_booleans_to_slice(mask.view("u1"))
        if isinstance(res, slice) and res.stop is None:
            # 如果切片的终点是None，则将其调整为与对象长度相符
            # TODO: 在maybe_booleans_to_slice中处理这个问题？
            res = slice(res.start, len(self), res.step)
        return res
    ) -> npt.NDArray[np.intp]:
        if isinstance(target, IntervalIndex):
            # 如果目标是 IntervalIndex 类型
            # 只有在不重叠的情况下才会进入这里
            # 想要精确匹配 -> 需要左右两边都匹配，因此延迟到 left/right get_indexer，逐个比较元素，相等则匹配
            indexer = self._get_indexer_unique_sides(target)

        elif not is_object_dtype(target.dtype):
            # 如果目标不是对象类型
            # 单一的标量索引：使用 IntervalTree
            # 我们应该总是有 self._should_partial_index(target) 为 True
            target = self._maybe_convert_i8(target)
            indexer = self._engine.get_indexer(target.values)
        else:
            # 如果目标是异构的标量索引：逐个元素延迟到 get_loc 处理
            # 我们应该总是有 self._should_partial_index(target) 为 True
            return self._get_indexer_pointwise(target)[0]

        return ensure_platform_int(indexer)

    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        target = ensure_index(target)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # 例如 IntervalIndex 具有不同的 closed 或不兼容的子类型 -> 没有匹配项
            return self._get_indexer_non_comparable(target, None, unique=False)

        elif isinstance(target, IntervalIndex):
            if self.left.is_unique and self.right.is_unique:
                # 即使我们没有 self._index_as_unique，也可以使用快速路径
                indexer = self._get_indexer_unique_sides(target)
                missing = (indexer == -1).nonzero()[0]
            else:
                return self._get_indexer_pointwise(target)

        elif is_object_dtype(target.dtype) or not self._should_partial_index(target):
            # 目标可能包含区间：逐个元素延迟到 get_loc 处理
            return self._get_indexer_pointwise(target)

        else:
            # 注意：这种情况与其他 Index 子类的行为不同，因为 IntervalIndex 支持部分整数索引
            target = self._maybe_convert_i8(target)
            indexer, missing = self._engine.get_indexer_non_unique(target.values)

        return ensure_platform_int(indexer), ensure_platform_int(missing)

    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
        # 调用者负责检查 `self.left.is_unique and self.right.is_unique`

        left_indexer = self.left.get_indexer(target.left)
        right_indexer = self.right.get_indexer(target.right)
        indexer = np.where(left_indexer == right_indexer, left_indexer, -1)
        return indexer
    def _get_indexer_pointwise(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
        # 初始化空列表，用于存放索引和缺失值索引
        indexer, missing = [], []
        # 遍历目标索引对象中的每个元素及其对应索引
        for i, key in enumerate(target):
            try:
                # 获取关键字 key 在当前索引对象中的位置
                locs = self.get_loc(key)
                # 如果 locs 是切片对象，只有在 get_indexer_non_unique 中才需要
                if isinstance(locs, slice):
                    # 将切片转换为整数数组
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype="intp")
                elif lib.is_integer(locs):
                    # 如果 locs 是整数，转换为包含一个元素的一维数组
                    locs = np.array(locs, ndmin=1)
                else:
                    # 否则 locs 应为布尔数组，获取其为 True 的索引
                    locs = np.where(locs)[0]
            except KeyError:
                # 处理 KeyError 异常，表示未找到关键字 key
                missing.append(i)
                locs = np.array([-1])
            except InvalidIndexError:
                # 处理 InvalidIndexError 异常，表示关键字 key 不是标量，如元组
                missing.append(i)
                locs = np.array([-1])

            # 将 locs 添加到索引器列表中
            indexer.append(locs)

        # 将索引器列表连接为一个 ndarray
        indexer = np.concatenate(indexer)
        # 返回索引器和缺失值列表
        return ensure_platform_int(indexer), ensure_platform_int(missing)

    @cache_readonly
    def _index_as_unique(self) -> bool:
        # 返回布尔值，指示索引是否唯一且没有 NaN 值
        return not self.is_overlapping and self._engine._na_count < 2

    _requires_unique_msg = (
        "cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique"
    )

    def _convert_slice_indexer(self, key: slice, kind: Literal["loc", "getitem"]):
        # 检查切片索引 key 的步长是否为 1 或者为 None
        if not (key.step is None or key.step == 1):
            # 如果是基于标签的切片且步长不为 1，抛出 ValueError 异常
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            if kind == "loc":
                raise ValueError(msg)
            if kind == "getitem":
                # 如果是基于位置的切片，检查其是否为有效的位置切片
                if not is_valid_positional_slice(key):
                    # 如果不能解释为位置切片，抛出 ValueError 异常
                    raise ValueError(msg)

        # 调用父类的方法处理切片索引器并返回结果
        return super()._convert_slice_indexer(key, kind)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # 返回布尔值，指示在 Series.__getitem__ 中是否明确使用位置索引
        # 如果 dtype 的子类型为 'm' 或 'M'，说明是日期时间类型，应该使用位置索引
        return self.dtype.subtype.kind in "mM"  # type: ignore[union-attr]

    def _maybe_cast_slice_bound(self, label, side: str):
        # 调用相应边界对象的 _maybe_cast_slice_bound 方法处理标签
        return getattr(self, side)._maybe_cast_slice_bound(label, side)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # 检查传入的 dtype 是否为 IntervalDtype 类型
        if not isinstance(dtype, IntervalDtype):
            return False
        # 查找 self.dtype 和传入的 dtype 的公共子类型
        common_subtype = find_common_type([self.dtype, dtype])
        # 如果公共子类型不是对象类型，则返回 True，表示可以比较
        return not is_object_dtype(common_subtype)
    # --------------------------------------------------------------------
    
    @cache_readonly
    def left(self) -> Index:
        """
        Return left bounds of the intervals in the IntervalIndex.
    
        The left bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the left bounds is the
        same as the datatype of the endpoints of the intervals.
    
        Returns
        -------
        Index
            An Index containing the left bounds of the intervals.
    
        See Also
        --------
        IntervalIndex.right : Return the right bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.mid : Return the mid-point of the intervals in
            the IntervalIndex.
        IntervalIndex.length : Return the length of the intervals in
            the IntervalIndex.
    
        Examples
        --------
        >>> iv_idx = pd.IntervalIndex.from_arrays([1, 2, 3], [4, 5, 6], closed="right")
        >>> iv_idx.left
        Index([1, 2, 3], dtype='int64')
    
        >>> iv_idx = pd.IntervalIndex.from_tuples(
        ...     [(1, 4), (2, 5), (3, 6)], closed="left"
        ... )
        >>> iv_idx.left
        Index([1, 2, 3], dtype='int64')
        """
        # 使用 IntervalIndex 对象的 _data 属性获取左边界数据，以 Index 对象形式返回
        return Index(self._data.left, copy=False)
    
    @cache_readonly
    def right(self) -> Index:
        """
        Return right bounds of the intervals in the IntervalIndex.
    
        The right bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the right bounds is the
        same as the datatype of the endpoints of the intervals.
    
        Returns
        -------
        Index
            An Index containing the right bounds of the intervals.
    
        See Also
        --------
        IntervalIndex.left : Return the left bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.mid : Return the mid-point of the intervals in
            the IntervalIndex.
        IntervalIndex.length : Return the length of the intervals in
            the IntervalIndex.
    
        Examples
        --------
        >>> iv_idx = pd.IntervalIndex.from_arrays([1, 2, 3], [4, 5, 6], closed="right")
        >>> iv_idx.right
        Index([4, 5, 6], dtype='int64')
    
        >>> iv_idx = pd.IntervalIndex.from_tuples(
        ...     [(1, 4), (2, 5), (3, 6)], closed="left"
        ... )
        >>> iv_idx.right
        Index([4, 5, 6], dtype='int64')
        """
        # 使用 IntervalIndex 对象的 _data 属性获取右边界数据，以 Index 对象形式返回
        return Index(self._data.right, copy=False)
    
    @cache_readonly
    def mid(self) -> Index:
        """
        Return the midpoint of each interval in the IntervalIndex as an Index.

        Each midpoint is calculated as the average of the left and right bounds
        of each interval. The midpoints are returned as a pandas Index object.

        Returns
        -------
        pandas.Index
            An Index containing the midpoints of each interval.

        See Also
        --------
        IntervalIndex.left : Return the left bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.right : Return the right bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.length : Return the length of the intervals in
            the IntervalIndex.

        Notes
        -----
        The midpoint is the average of the interval bounds, potentially resulting
        in a floating-point number even if bounds are integers. The returned Index
        will have a dtype that accurately holds the midpoints. This computation is
        the same regardless of whether intervals are open or closed.

        Examples
        --------
        >>> iv_idx = pd.IntervalIndex.from_arrays([1, 2, 3], [4, 5, 6])
        >>> iv_idx.mid
        Index([2.5, 3.5, 4.5], dtype='float64')

        >>> iv_idx = pd.IntervalIndex.from_tuples([(1, 4), (2, 5), (3, 6)])
        >>> iv_idx.mid
        Index([2.5, 3.5, 4.5], dtype='float64')
        """
        # 返回 IntervalIndex 对象中每个区间的中点作为 Index 对象
        return Index(self._data.mid, copy=False)

    @property
    def length(self) -> Index:
        """
        Calculate the length of each interval in the IntervalIndex.

        This method returns a new Index containing the lengths of each interval
        in the IntervalIndex. The length of an interval is defined as the difference
        between its end and its start.

        Returns
        -------
        Index
            An Index containing the lengths of each interval.

        See Also
        --------
        Interval.length : Return the length of the Interval.

        Examples
        --------
        >>> intervals = pd.IntervalIndex.from_arrays(
        ...     [1, 2, 3], [4, 5, 6], closed="right"
        ... )
        >>> intervals.length
        Index([3, 3, 3], dtype='int64')

        >>> intervals = pd.IntervalIndex.from_tuples([(1, 5), (6, 10), (11, 15)])
        >>> intervals.length
        Index([4, 4, 4], dtype='int64')
        """
        # 返回 IntervalIndex 对象中每个区间的长度作为 Index 对象
        return Index(self._data.length, copy=False)

    # --------------------------------------------------------------------
    # Set Operations
    def _intersection(self, other, sort):
        """
        intersection specialized to the case with matching dtypes.
        """
        # 如果左右端点均唯一，则使用_unique方法求交集
        if self.left.is_unique and self.right.is_unique:
            taken = self._intersection_unique(other)
        # 如果对方的左右端点均唯一且自身的缺失值不超过1个，则交换self/other，并使用_other_unique方法求交集
        elif other.left.is_unique and other.right.is_unique and self.isna().sum() <= 1:
            taken = other._intersection_unique(self)
        else:
            # 存在重复值，则使用_non_unique方法求交集
            taken = self._intersection_non_unique(other)

        # 如果sort参数为None，则对结果进行排序
        if sort is None:
            taken = taken.sort_values()

        return taken

    def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does not have any common endpoint,
        no matter left or right.
        Return the intersection with another IntervalIndex.
        Parameters
        ----------
        other : IntervalIndex
        Returns
        -------
        IntervalIndex
        """
        # 注意：这比super()._intersection(other)更高效
        # 获取self和other的左端点索引器和右端点索引器
        lindexer = self.left.get_indexer(other.left)
        rindexer = self.right.get_indexer(other.right)

        # 找到匹配的索引，并去除重复
        match = (lindexer == rindexer) & (lindexer != -1)
        indexer = lindexer.take(match.nonzero()[0])
        indexer = unique(indexer)

        return self.take(indexer)

    def _intersection_non_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does have some common endpoints,
        on either sides.
        Return the intersection with another IntervalIndex.

        Parameters
        ----------
        other : IntervalIndex

        Returns
        -------
        IntervalIndex
        """
        # 注意：这比super()._intersection(other)快大约3.25倍
        # 在self上创建一个布尔掩码
        mask = np.zeros(len(self), dtype=bool)

        # 如果self和other都有缺失值，找到第一个NaN位置并标记为True
        if self.hasnans and other.hasnans:
            first_nan_loc = np.arange(len(self))[self.isna()][0]
            mask[first_nan_loc] = True

        # 将other的左右端点作为元组存入集合other_tups
        other_tups = set(zip(other.left, other.right))
        # 遍历self的左右端点，如果存在于other_tups中，则将对应位置的mask设为True
        for i, tup in enumerate(zip(self.left, self.right)):
            if tup in other_tups:
                mask[i] = True

        return self[mask]

    # --------------------------------------------------------------------

    def _get_engine_target(self) -> np.ndarray:
        # 注意：我们本可以使用libjoin函数，通过转换为对象dtype或构造元组（比构造Interval更快），
        # 但在这些情况下，libjoin的快速路径已不再快速。
        raise NotImplementedError(
            "IntervalIndex does not use libjoin fastpaths or pass values to "
            "IndexEngine objects"
        )
    # 定义一个私有方法 _from_join_target，接受参数 result
    def _from_join_target(self, result):
        # 抛出 NotImplementedError 异常，提示 IntervalIndex 类不使用 libjoin 的快速路径
        raise NotImplementedError("IntervalIndex does not use libjoin fastpaths")

    # TODO: arithmetic operations
    # TODO 注释：此处应该实现与算术相关的操作，但当前尚未实现
def _is_valid_endpoint(endpoint) -> bool:
    """
    Helper for interval_range to check if start/end are valid types.
    """
    # 返回一个布尔值，判断 endpoint 是否是合法的起始或结束类型
    return any(
        [
            is_number(endpoint),  # 检查 endpoint 是否是数字类型
            isinstance(endpoint, Timestamp),  # 检查 endpoint 是否是 Timestamp 类型
            isinstance(endpoint, Timedelta),  # 检查 endpoint 是否是 Timedelta 类型
            endpoint is None,  # 检查 endpoint 是否为 None
        ]
    )


def _is_type_compatible(a, b) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
    # 返回一个布尔值，判断 a 和 b 是否是兼容的类型用于 start、end 或 freq
    is_ts_compat = lambda x: isinstance(x, (Timestamp, BaseOffset))  # 检查是否是 Timestamp 或 BaseOffset 类型
    is_td_compat = lambda x: isinstance(x, (Timedelta, BaseOffset))  # 检查是否是 Timedelta 或 BaseOffset 类型
    return (
        (is_number(a) and is_number(b))  # a 和 b 都是数字类型
        or (is_ts_compat(a) and is_ts_compat(b))  # a 和 b 都是 Timestamp 或 BaseOffset 类型
        or (is_td_compat(a) and is_td_compat(b))  # a 和 b 都是 Timedelta 或 BaseOffset 类型
        or com.any_none(a, b)  # a 和 b 中至少有一个为 None
    )


def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name: Hashable | None = None,
    closed: IntervalClosedType = "right",
) -> IntervalIndex:
    """
    Return a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric or datetime-like, default None
        Left bound for generating intervals.
    end : numeric or datetime-like, default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate.
    freq : numeric, str, Timedelta, datetime.timedelta, or DateOffset, default None
        The length of each interval. Must be consistent with the type of start
        and end, e.g. 2 for numeric, or '5H' for datetime-like.  Default is 1
        for numeric and 'D' for datetime-like.
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both
        or neither.

    Returns
    -------
    IntervalIndex
        Object with a fixed frequency.

    See Also
    --------
    IntervalIndex : An Index of intervals that are all closed on the same side.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``IntervalIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end``, inclusively.

    To learn more about datetime-like frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    Examples
    --------
    Numeric ``start`` and  ``end`` is supported.

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    Additionally, datetime-like input is also supported.

    >>> pd.interval_range(
    ...     start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-01-04")
    ... )
    """
    # 返回一个固定频率的 IntervalIndex 对象

    # 以下是函数实现的详细过程，根据参数生成 IntervalIndex 对象
    pass  # 实际的生成逻辑未在这里实现，仅展示了函数的目的和用法说明
    IntervalIndex([(2017-01-01 00:00:00, 2017-01-02 00:00:00],
                   (2017-01-02 00:00:00, 2017-01-03 00:00:00],
                   (2017-01-03 00:00:00, 2017-01-04 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    The ``freq`` parameter specifies the frequency between the left and right
    endpoints of the individual intervals within the ``IntervalIndex``. For
    numeric ``start`` and ``end``, the frequency must also be numeric.
    
    >>> pd.interval_range(start=0, periods=4, freq=1.5)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
                  dtype='interval[float64, right]')

    Similarly, for datetime-like ``start`` and ``end``, the frequency must be
    convertible to a DateOffset.
    
    >>> pd.interval_range(start=pd.Timestamp("2017-01-01"), periods=3, freq="MS")
    IntervalIndex([(2017-01-01 00:00:00, 2017-02-01 00:00:00],
                   (2017-02-01 00:00:00, 2017-03-01 00:00:00],
                   (2017-03-01 00:00:00, 2017-04-01 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).
    
    >>> pd.interval_range(start=0, end=6, periods=4)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
              dtype='interval[float64, right]')

    The ``closed`` parameter specifies which endpoints of the individual
    intervals within the ``IntervalIndex`` are closed.
    
    >>> pd.interval_range(end=5, periods=4, closed="both")
    IntervalIndex([[1, 2], [2, 3], [3, 4], [4, 5]],
                  dtype='interval[int64, both]')
    """
    # 将输入的 start 参数转换为日期时间对象（如果可能）
    start = maybe_box_datetimelike(start)
    # 将输入的 end 参数转换为日期时间对象（如果可能）
    end = maybe_box_datetimelike(end)
    # 如果 start 存在则将其用作 endpoint，否则使用 end 作为 endpoint
    endpoint = start if start is not None else end

    # 如果 freq 为 None，并且期间（periods）、start、end 中有一个为 None，则设置默认的 freq
    if freq is None and com.any_none(periods, start, end):
        freq = 1 if is_number(endpoint) else "D"

    # 检查四个参数中是否有三个已定义，否则引发 ValueError
    if com.count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

    # 检查 start 是否是有效的端点（要求是数值或类似日期时间的对象）
    if not _is_valid_endpoint(start):
        raise ValueError(f"start must be numeric or datetime-like, got {start}")
    # 检查 end 是否是有效的端点（要求是数值或类似日期时间的对象）
    if not _is_valid_endpoint(end):
        raise ValueError(f"end must be numeric or datetime-like, got {end}")

    # 验证 periods 的值是否有效
    periods = validate_periods(periods)

    # 如果 freq 不为 None 并且不是数值类型，则尝试将其转换为 DateOffset
    if freq is not None and not is_number(freq):
        try:
            freq = to_offset(freq)
        except ValueError as err:
            raise ValueError(
                f"freq must be numeric or convertible to DateOffset, got {freq}"
            ) from err

    # 验证类型的兼容性，确保 start、end、freq 类型相容
    if not all(
        [
            _is_type_compatible(start, end),
            _is_type_compatible(start, freq),
            _is_type_compatible(end, freq),
        ]
    ):
        raise TypeError("start, end, freq need to be type compatible")
    # +1 to convert interval count to breaks count (n breaks = n-1 intervals)
    # 如果 periods 不为 None，则将其加一，以将间隔计数转换为断点计数（n 个断点对应 n-1 个间隔）
    if periods is not None:
        periods += 1

    # breaks 的类型可以是 np.ndarray、TimedeltaIndex 或者 DatetimeIndex
    breaks: np.ndarray | TimedeltaIndex | DatetimeIndex

    # 如果 endpoint 是数字类型
    if is_number(endpoint):
        # 默认数据类型为 int64
        dtype: np.dtype = np.dtype("int64")
        # 如果 start、end 和 freq 均不为 None
        if com.all_not_none(start, end, freq):
            # 如果 start、end 或 freq 中有任一为 float 或 np.float16 类型，则数据类型设为 float64
            if (
                isinstance(start, (float, np.float16))
                or isinstance(end, (float, np.float16))
                or isinstance(freq, (float, np.float16))
            ):
                dtype = np.dtype("float64")
            # 如果 start 和 end 是整数或浮点数，并且它们的 dtype 相同，则数据类型设为 start 的 dtype
            elif (
                isinstance(start, (np.integer, np.floating))
                and isinstance(end, (np.integer, np.floating))
                and start.dtype == end.dtype
            ):
                dtype = start.dtype
            # 使用 np.arange 生成一组断点，确保最后一个断点可以捕获到 end
            breaks = np.arange(start, end + (freq * 0.1), freq)
            # 将 breaks 转换为指定的 dtype 类型
            breaks = maybe_downcast_numeric(breaks, dtype)
        else:
            # 如果未指定 periods，则根据频率 freq 计算出 periods
            if periods is None:
                periods = int((end - start) // freq) + 1
            elif start is None:
                # 如果未指定 start，则计算出 start，使得 end - start = (periods - 1) * freq
                start = end - (periods - 1) * freq
            elif end is None:
                # 如果未指定 end，则计算出 end，使得 end - start = (periods - 1) * freq
                end = start + (periods - 1) * freq

            # 使用 np.linspace 生成一组断点
            breaks = np.linspace(start, end, periods)
        
        # 如果 start、end、freq 均为整数，则 breaks 中的所有元素都为整数
        if all(is_integer(x) for x in com.not_none(start, end, freq)):
            # np.linspace 总是生成浮点数输出
            
            # 错误："maybe_downcast_numeric" 的第一个参数的类型不兼容，预期为 "ndarray[Any, Any]"
            # breaks 可能是 ndarray、TimedeltaIndex 或者 DatetimeIndex，需要将其转换为指定的 dtype 类型
            breaks = maybe_downcast_numeric(
                breaks,  # type: ignore[arg-type]
                dtype,
            )
    else:
        # 如果 endpoint 不是数字类型，则委托给适当的范围函数处理
        if isinstance(endpoint, Timestamp):
            # 使用 date_range 生成时间范围的断点
            breaks = date_range(start=start, end=end, periods=periods, freq=freq)
        else:
            # 使用 timedelta_range 生成时间间隔范围的断点
            breaks = timedelta_range(start=start, end=end, periods=periods, freq=freq)

    # 使用 IntervalIndex.from_breaks 方法根据 breaks 生成 IntervalIndex 对象，并设置相关参数
    return IntervalIndex.from_breaks(
        breaks,
        name=name,
        closed=closed,
        dtype=IntervalDtype(subtype=breaks.dtype, closed=closed),
    )
```