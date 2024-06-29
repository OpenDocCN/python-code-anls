# `D:\src\scipysrc\pandas\pandas\core\arrays\interval.py`

```
from __future__ import annotations
# 从未来导入注释：允许使用类型注释的特性，即使在 Python 3.7 之前的版本也可以。

import operator
# 导入 operator 模块：用于函数操作的标准操作符的函数集合。
from operator import (
    le,
    lt,
)
# 从 operator 模块导入 le（小于等于）和 lt（小于）操作符函数。

import textwrap
# 导入 textwrap 模块：用于文本包装和填充的工具。

from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    overload,
)
# 导入 typing 模块中的类型定义：用于静态类型检查和注释。

import numpy as np
# 导入 numpy 库：用于数值计算的强大库。

from pandas._libs import lib
# 导入 pandas 私有库中的 lib 模块。

from pandas._libs.interval import (
    VALID_CLOSED,
    Interval,
    IntervalMixin,
    intervals_to_interval_bounds,
)
# 从 pandas 私有库中的 interval 模块导入相关类和函数：用于处理区间和间隔的操作。

from pandas._libs.missing import NA
# 导入 pandas 私有库中的 missing 模块：用于处理缺失值。

from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    IntervalClosedType,
    NpDtype,
    PositionalIndexer,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    SortKind,
    TimeArrayLike,
    npt,
)
# 导入 pandas 中的类型定义：用于类型注解和静态类型检查。

from pandas.compat.numpy import function as nv
# 导入 pandas 兼容的 numpy 中的 function 函数。

from pandas.errors import IntCastingNaNError
# 导入 pandas 中的 IntCastingNaNError 异常：用于整数转换时的 NaN 错误处理。

from pandas.util._decorators import Appender
# 导入 pandas 工具模块中的 Appender 装饰器：用于添加功能。

from pandas.core.dtypes.cast import (
    LossySetitemError,
    maybe_upcast_numeric_to_64bit,
)
# 导入 pandas 核心数据类型转换模块中的异常和函数。

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
# 导入 pandas 核心数据类型常用函数和判断方法。

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
# 导入 pandas 核心数据类型模块中的分类数据类型和区间数据类型。

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCPeriodIndex,
)
# 导入 pandas 核心数据类型通用模块中的抽象 DataFrame 和时间索引类。

from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    notna,
)
# 导入 pandas 核心数据类型缺失值模块中的相关函数：用于处理 NaN 和缺失值。

from pandas.core.algorithms import (
    isin,
    take,
    unique,
    value_counts_internal as value_counts,
)
# 导入 pandas 核心算法模块中的算法函数：用于元素在数组中的检查、取值、唯一值和计数。

from pandas.core.arrays import ArrowExtensionArray
# 导入 pandas 核心数组模块中的 ArrowExtensionArray 类：用于处理 Arrow 扩展数组。

from pandas.core.arrays.base import (
    ExtensionArray,
    _extension_array_shared_docs,
)
# 导入 pandas 核心数组基类和共享文档。

from pandas.core.arrays.datetimes import DatetimeArray
# 导入 pandas 核心日期时间数组模块中的 DatetimeArray 类。

from pandas.core.arrays.timedeltas import TimedeltaArray
# 导入 pandas 核心时间差数组模块中的 TimedeltaArray 类。

import pandas.core.common as com
# 导入 pandas 核心通用模块中的 com。

from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
# 导入 pandas 核心构造模块中的数组、确保封装若为日期时间和提取数组相关函数。

from pandas.core.indexers import check_array_indexer
# 导入 pandas 核心索引模块中的 check_array_indexer 函数。

from pandas.core.ops import (
    invalid_comparison,
    unpack_zerodim_and_defer,
)
# 导入 pandas 核心操作模块中的无效比较和零维数据的延迟解包函数。

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )
    # 如果是类型检查阶段，则导入 collections.abc 模块中的 Callable、Iterator 和 Sequence 类。

    from pandas import (
        Index,
        Series,
    )
    # 如果是类型检查阶段，则导入 pandas 中的 Index 和 Series 类。

IntervalSide = Union[TimeArrayLike, np.ndarray]
# 定义 IntervalSide 类型别名：可以是时间数组或 numpy 数组。

IntervalOrNA = Union[Interval, float]
# 定义 IntervalOrNA 类型别名：可以是区间对象或浮点数。

_interval_shared_docs: dict[str, str] = {}
# 初始化 _interval_shared_docs 字典：用于存储区间相关的共享文档内容。

_shared_docs_kwargs = {
    "klass": "IntervalArray",
    "qualname": "arrays.IntervalArray",
    "name": "",
}
# 初始化 _shared_docs_kwargs 字典：用于存储共享文档中的类名、资格名称和名称字段。

_interval_shared_docs["class"] = """
%(summary)s

Parameters
----------
data : array-like (1-dimensional)
    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
    Interval objects from which to build the %(klass)s.
closed : {'left', 'right', 'both', 'neither'}, default 'right'
    Whether the intervals are closed on the left-side, right-side, both or
    neither.
dtype : dtype or None, default None
    If None, dtype will be inferred.
copy : bool, default False
    Copy the input data.
%(name)s\

"""
# 设置 _interval_shared_docs 中 "class" 键的值：包含区间数组的类文档模板，用于描述类的构建参数和选项。
# verify_integrity : bool, default True
#     确定 %(klass)s 是否有效的标志。

# Attributes
# ----------
# left
# right
# closed
# mid
# length
# is_empty
# is_non_overlapping_monotonic
# %(extra_attributes)s

# Methods
# -------
# from_arrays
# from_tuples
# from_breaks
# contains
# overlaps
# set_closed
# to_tuples
# %(extra_methods)s

# See Also
# --------
# Index : pandas 的基础索引类型。
# Interval : 具有界限的切片样式间隔；一个 %(klass)s 的元素。
# interval_range : 创建固定频率 IntervalIndex 的函数。
# cut : 将值分为离散的间隔。
# qcut : 根据排名或样本分位数将值分为相等大小的间隔。

# Notes
# -----
# 更多信息请参阅 `用户指南
# <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__。

# %(examples)s
"""


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalArray",
        "summary": "Pandas array for interval data that are closed on the same side.",
        "name": "",
        "extra_attributes": "",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalArray`` can be constructed directly from an array-like of
    ``Interval`` objects:

    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    <IntervalArray>
    [(0, 1], (1, 5]]
    Length: 2, dtype: interval[int64, right]

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalArray.from_arrays`,
    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.
    """
        ),
    }
)
class IntervalArray(IntervalMixin, ExtensionArray):
    # 可以容纳缺失值
    can_hold_na = True
    # 缺失值和填充值均为 NaN
    _na_value = _fill_value = np.nan

    @property
    def ndim(self) -> Literal[1]:
        return 1

    # To make mypy recognize the fields
    _left: IntervalSide
    _right: IntervalSide
    _dtype: IntervalDtype

    # ---------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ) -> Self:
        # 使用 extract_array 函数从 data 中提取数组，如果需要，将其转换为 NumPy 数组
        data = extract_array(data, extract_numpy=True)

        # 如果 data 是当前类的实例，则继承其左右边界和闭合属性，或者使用传入的参数
        if isinstance(data, cls):
            left: IntervalSide = data._left
            right: IntervalSide = data._right
            closed = closed or data.closed
            dtype = IntervalDtype(left.dtype, closed=closed)
        else:
            # 不允许处理标量数据
            if is_scalar(data):
                msg = (
                    f"{cls.__name__}(...) must be called with a collection "
                    f"of some kind, {data} was passed"
                )
                raise TypeError(msg)

            # 可能需要转换空或全 na 数据
            data = _maybe_convert_platform_interval(data)
            # 根据数据推断左右边界和闭合属性
            left, right, infer_closed = intervals_to_interval_bounds(
                data, validate_closed=closed is None
            )
            # 如果左边界是对象类型，则尝试将其转换
            if left.dtype == object:
                left = lib.maybe_convert_objects(left)
                right = lib.maybe_convert_objects(right)
            # 确定最终的闭合属性
            closed = closed or infer_closed

            # 确保输入是简单的新数据的方法，处理左右边界、闭合属性、复制标志和数据类型
            left, right, dtype = cls._ensure_simple_new_inputs(
                left,
                right,
                closed=closed,
                copy=copy,
                dtype=dtype,
            )

        # 如果需要验证数据的完整性，则调用 _validate 方法
        if verify_integrity:
            cls._validate(left, right, dtype=dtype)

        # 返回使用简单新数据创建的类的实例
        return cls._simple_new(
            left,
            right,
            dtype=dtype,
        )

    @classmethod
    def _simple_new(
        cls,
        left: IntervalSide,
        right: IntervalSide,
        dtype: IntervalDtype,
    ) -> Self:
        # 创建一个新的 IntervalMixin 类实例，设置其左边界、右边界和数据类型
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype

        return result

    @classmethod
    def _ensure_simple_new_inputs(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ):
        # 确保输入参数是简单的新数据，处理左右边界、闭合属性、复制标志和数据类型
        ...

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        # 从序列创建新的 Interval 类实例，指定数据类型和是否复制数据
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self:
        # 从因子化数据创建新的 Interval 类实例，通过调用 _from_sequence 处理
        return cls._from_sequence(values, dtype=original.dtype)
    # 设置_shared_docs字典的"from_breaks"条目，将其格式化为文本包含的字符串模板
    _interval_shared_docs["from_breaks"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct from a left and right array.
        %(klass)s.from_tuples : Construct from a sequence of tuples.

        %(examples)s\
        """
    )

    # 使用修饰符@classmethod声明类方法from_breaks，并附加文档字符串
    @classmethod
    @Appender(
        # 使用_shared_docs字典中的"from_breaks"条目格式化文档字符串
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalArray",  # 替换模板中的%(klass)s为"IntervalArray"
            "name": "",  # 替换模板中的%(name)s为空字符串
            # 替换模板中的%(examples)s为格式化后的文本字符串模板
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    # 定义from_breaks方法，接受多个参数并返回自身类型
    def from_breaks(
        cls,
        breaks,  # 数组或类似数组的断点列表
        closed: IntervalClosedType | None = "right",  # 指定间隔的闭合方式，默认右闭合
        copy: bool = False,  # 是否复制数据，默认不复制
        dtype: Dtype | None = None,  # 数据类型或None，默认为推断类型
    ) -> Self:
        # 调用_maybe_convert_platform_interval函数处理断点列表
        breaks = _maybe_convert_platform_interval(breaks)

        # 调用类方法from_arrays，使用处理后的断点列表创建IntervalArray对象
        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)
    _interval_shared_docs["from_arrays"] = textwrap.dedent(
        """
        构造函数，基于定义左右边界的两个数组。

        Parameters
        ----------
        left : array-like (1-dimensional)
            每个区间的左边界。
        right : array-like (1-dimensional)
            每个区间的右边界。
        closed : {'left', 'right', 'both', 'neither'}, 默认 'right'
            区间左侧、右侧、两侧或无边界的闭合情况。
        %(name)s
        copy : bool, 默认 False
            是否复制数据。
        dtype : dtype, 可选
            如果为 None，则推断 dtype。

        Returns
        -------
        %(klass)s

        Raises
        ------
        ValueError
            当 `left` 或 `right` 中只有一个缺少值时。
            当 `left` 中的某个值大于相应的 `right` 值时。

        See Also
        --------
        interval_range : 创建固定频率的 IntervalIndex 的函数。
        %(klass)s.from_breaks : 从一个数组的分割点构造 %(klass)s。
        %(klass)s.from_tuples : 从元组的数组构造 %(klass)s。

        Notes
        -----
        `left` 中的每个元素必须小于或等于相同位置的 `right` 元素。
        如果某个元素缺失，那么 `left` 和 `right` 中都必须缺失。
        当 `left` 或 `right` 使用不支持的类型时，会引发 TypeError。
        目前不支持 'category'、'object' 和 'string' 子类型。

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        left = _maybe_convert_platform_interval(left)  # 可能将 left 转换为平台特定的区间表示
        right = _maybe_convert_platform_interval(right)  # 可能将 right 转换为平台特定的区间表示

        left, right, dtype = cls._ensure_simple_new_inputs(  # 确保输入的简单新值
            left,
            right,
            closed=closed,
            copy=copy,
            dtype=dtype,
        )
        cls._validate(left, right, dtype=dtype)  # 验证左右边界和 dtype 是否有效

        return cls._simple_new(left, right, dtype=dtype)  # 创建并返回新的 IntervalArray 对象
    # 将 "from_tuples" 方法的文档字符串定义为 _interval_shared_docs 字典中的一个条目，用于类的文档说明
    _interval_shared_docs["from_tuples"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct an %(klass)s from a left and
                                    right array.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
                                    splits.

        %(examples)s\
        """
    )

    @classmethod
    # 使用 @classmethod 装饰器声明以下方法为类方法
    @Appender(
        # 将 _interval_shared_docs 中的 "from_tuples" 条目引入，并格式化其中的文本占位符
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalArray",  # klass 替换为 "IntervalArray"
            "name": "",  # name 留空
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        """
            ),
        }
    )
    # 定义类方法 from_tuples，用于从元组数组构建 IntervalArray 对象
    def from_tuples(
        cls,
        data,  # 参数 data，表示一个类似数组的元组序列
        closed: IntervalClosedType | None = "right",  # 参数 closed，表示区间的闭合方式，默认为右闭
        copy: bool = False,  # 参数 copy，表示是否复制数据，默认为 False
        dtype: Dtype | None = None,  # 参数 dtype，表示数据类型，默认为 None，自动推断
    ) -> Self:
        if len(data):  # 如果 data 长度大于 0
            left, right = [], []  # 初始化 left 和 right 为空列表
        else:  # 否则（data 长度为 0）
            # 确保空数据保留输入的 dtype
            left = right = data  # 将 left 和 right 均设置为 data

        for d in data:  # 遍历 data 中的每个元素 d
            if not isinstance(d, tuple) and isna(d):  # 如果 d 不是元组且为缺失值
                lhs = rhs = np.nan  # 将 lhs 和 rhs 均设为 NaN
            else:  # 否则
                name = cls.__name__  # 获取当前类的名称
                try:
                    # 需要长度为 2 的元组列表，例如 [(0, 1), (1, 2), ...]
                    lhs, rhs = d  # 尝试解构元组 d，分别赋给 lhs 和 rhs
                except ValueError as err:
                    msg = f"{name}.from_tuples requires tuples of length 2, got {d}"
                    raise ValueError(msg) from err  # 如果解构失败，抛出 ValueError 异常
                except TypeError as err:
                    msg = f"{name}.from_tuples received an invalid item, {d}"
                    raise TypeError(msg) from err  # 如果类型错误，抛出 TypeError 异常
            left.append(lhs)  # 将 lhs 添加到 left 列表末尾
            right.append(rhs)  # 将 rhs 添加到 right 列表末尾

        # 调用 from_arrays 方法，使用 left 和 right 数组构建 IntervalArray 对象，并返回结果
        return cls.from_arrays(left, right, closed, copy=False, dtype=dtype)

    @classmethod
    def _validate(cls, left, right, dtype: IntervalDtype) -> None:
        """
        Verify that the IntervalArray is valid.

        Checks that

        * dtype is correct
        * left and right match lengths
        * left and right have the same missing values
        * left is always below right
        """
        # 检查 dtype 是否为 IntervalDtype 类型
        if not isinstance(dtype, IntervalDtype):
            msg = f"invalid dtype: {dtype}"
            raise ValueError(msg)
        # 检查 left 和 right 是否长度相同
        if len(left) != len(right):
            msg = "left and right must have the same length"
            raise ValueError(msg)
        # 检查 left 和 right 是否具有相同的缺失值位置
        left_mask = notna(left)
        right_mask = notna(right)
        if not (left_mask == right_mask).all():
            msg = (
                "missing values must be missing in the same "
                "location both left and right sides"
            )
            raise ValueError(msg)
        # 检查左侧的值是否始终小于等于右侧的值
        if not (left[left_mask] <= right[left_mask]).all():
            msg = "left side of interval must be <= right side"
            raise ValueError(msg)

    def _shallow_copy(self, left, right) -> Self:
        """
        Return a new IntervalArray with the replacement attributes

        Parameters
        ----------
        left : Index
            Values to be used for the left-side of the intervals.
        right : Index
            Values to be used for the right-side of the intervals.
        """
        # 创建新的 IntervalDtype 对象
        dtype = IntervalDtype(left.dtype, closed=self.closed)
        # 确保输入的 left 和 right 符合简单输入的要求
        left, right, dtype = self._ensure_simple_new_inputs(left, right, dtype=dtype)

        # 返回用新属性替换的 IntervalArray 的浅拷贝
        return self._simple_new(left, right, dtype=dtype)

    # ---------------------------------------------------------------------
    # Descriptive

    @property
    def dtype(self) -> IntervalDtype:
        # 返回 IntervalArray 的 dtype 属性
        return self._dtype

    @property
    def nbytes(self) -> int:
        # 返回左右两侧值的内存大小总和
        return self.left.nbytes + self.right.nbytes

    @property
    def size(self) -> int:
        # 避免实例化 self.values，返回左侧值的大小
        return self.left.size

    # ---------------------------------------------------------------------
    # EA Interface

    def __iter__(self) -> Iterator:
        # 返回数组的迭代器
        return iter(np.asarray(self))

    def __len__(self) -> int:
        # 返回 _left 的长度
        return len(self._left)

    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalOrNA: ...
    # 略（重载方法的声明，具体实现未提供）
    
    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self: ...
    # 略（重载方法的声明，具体实现未提供）
    # 定义特殊方法 "__getitem__"，用于实现索引操作，返回索引位置的值或区间对象
    def __getitem__(self, key: PositionalIndexer) -> Self | IntervalOrNA:
        # 检查并规范化索引器，确保其符合要求
        key = check_array_indexer(self, key)
        # 获取左边界数组中索引位置的值
        left = self._left[key]
        # 获取右边界数组中索引位置的值
        right = self._right[key]

        # 如果左边界不是数组或扩展数组
        if not isinstance(left, (np.ndarray, ExtensionArray)):
            # 如果左边界是标量且为缺失值，则返回填充值
            if is_scalar(left) and isna(left):
                return self._fill_value
            # 否则返回一个区间对象，表示该索引位置的区间
            return Interval(left, right, self.closed)
        
        # 如果左边界是多维数组，则抛出数值错误
        if np.ndim(left) > 1:
            raise ValueError("multi-dimensional indexing not allowed")
        
        # 否则调用 "_simple_new" 方法创建新的 IntervalArray 对象，返回结果
        return self._simple_new(left, right, dtype=self.dtype)  # type: ignore[arg-type]

    # 定义特殊方法 "__setitem__"，用于实现赋值操作
    def __setitem__(self, key, value) -> None:
        # 验证并返回赋值操作后的左右边界值
        value_left, value_right = self._validate_setitem_value(value)
        # 检查并规范化索引器，确保其符合要求
        key = check_array_indexer(self, key)

        # 将赋值操作后的左边界值存入 _left 数组对应索引位置
        self._left[key] = value_left
        # 将赋值操作后的右边界值存入 _right 数组对应索引位置
        self._right[key] = value_right

    # 使用装饰器定义 "__eq__" 方法，实现对象之间的等于比较
    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        # 调用 _cmp_method 方法，比较当前对象和另一个对象是否相等
        return self._cmp_method(other, operator.eq)

    # 使用装饰器定义 "__ne__" 方法，实现对象之间的不等于比较
    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        # 调用 _cmp_method 方法，比较当前对象和另一个对象是否不相等
        return self._cmp_method(other, operator.ne)

    # 使用装饰器定义 "__gt__" 方法，实现对象之间的大于比较
    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        # 调用 _cmp_method 方法，比较当前对象是否大于另一个对象
        return self._cmp_method(other, operator.gt)

    # 使用装饰器定义 "__ge__" 方法，实现对象之间的大于等于比较
    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        # 调用 _cmp_method 方法，比较当前对象是否大于等于另一个对象
        return self._cmp_method(other, operator.ge)

    # 使用装饰器定义 "__lt__" 方法，实现对象之间的小于比较
    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        # 调用 _cmp_method 方法，比较当前对象是否小于另一个对象
        return self._cmp_method(other, operator.lt)

    # 使用装饰器定义 "__le__" 方法，实现对象之间的小于等于比较
    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        # 调用 _cmp_method 方法，比较当前对象是否小于等于另一个对象
        return self._cmp_method(other, operator.le)

    # 定义方法 "argsort"，用于返回按指定条件排序后的索引数组
    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        # 验证并返回排序时使用的升序标志
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        # 如果升序且排序算法为 "quicksort" 且缺失值位置为 "last"
        if ascending and kind == "quicksort" and na_position == "last":
            # 返回基于左右边界数组的词典序排序结果
            return np.lexsort((self.right, self.left))

        # 否则调用超类的 argsort 方法进行排序，返回结果
        return super().argsort(
            ascending=ascending, kind=kind, na_position=na_position, **kwargs
        )
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        # 验证轴参数的有效性
        nv.validate_minmax_axis(axis, self.ndim)

        # 如果数组长度为0，则返回NA值
        if not len(self):
            return self._na_value

        # 创建一个布尔掩码，标记缺失值位置
        mask = self.isna()
        
        # 如果存在缺失值且不跳过缺失值，则返回NA值
        if mask.any():
            if not skipna:
                return self._na_value
            # 从数组中排除缺失值
            obj = self[~mask]
        else:
            obj = self

        # 找到最小值的索引并返回对应的值
        indexer = obj.argsort()[0]
        return obj[indexer]

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        # 验证轴参数的有效性
        nv.validate_minmax_axis(axis, self.ndim)

        # 如果数组长度为0，则返回NA值
        if not len(self):
            return self._na_value

        # 创建一个布尔掩码，标记缺失值位置
        mask = self.isna()
        
        # 如果存在缺失值且不跳过缺失值，则返回NA值
        if mask.any():
            if not skipna:
                return self._na_value
            # 从数组中排除缺失值
            obj = self[~mask]
        else:
            obj = self

        # 找到最大值的索引并返回对应的值
        indexer = obj.argsort()[-1]
        return obj[indexer]

    def fillna(self, value, limit: int | None = None, copy: bool = True) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        limit : int, default None
            (Not implemented yet for IntervalArray)
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        """
        # 如果不允许复制原始数据，则抛出NotImplementedError
        if copy is False:
            raise NotImplementedError
        # 目前不支持对IntervalArray设置填充限制，因此抛出ValueError
        if limit is not None:
            raise ValueError("limit must be None")

        # 验证填充值，并分别填充左右两侧的IntervalArray
        value_left, value_right = self._validate_scalar(value)
        left = self.left.fillna(value=value_left)
        right = self.right.fillna(value=value_right)

        # 返回填充后的IntervalArray，确保只复制浅层数据
        return self._shallow_copy(left, right)
    def equals(self, other) -> bool:
        # 检查对象类型是否相同，如果不相同则返回 False
        if type(self) != type(other):
            return False

        # 检查对象的属性是否完全相等，包括闭合属性和左右端点的索引是否相同
        return bool(
            self.closed == other.closed
            and self.left.equals(other.left)
            and self.right.equals(other.right)
        )

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[IntervalArray]) -> Self:
        """
        Concatenate multiple IntervalArray

        Parameters
        ----------
        to_concat : sequence of IntervalArray
            List of IntervalArray objects to concatenate

        Returns
        -------
        IntervalArray
            Concatenated IntervalArray object
        """
        # Collect the set of 'closed' properties from all IntervalArray objects
        closed_set = {interval.closed for interval in to_concat}
        # Ensure all intervals have the same 'closed' property; otherwise, raise an error
        if len(closed_set) != 1:
            raise ValueError("Intervals must all be closed on the same side.")
        # Determine the 'closed' property for the concatenated IntervalArray
        closed = closed_set.pop()

        # Concatenate the 'left' properties of all IntervalArray objects
        left: IntervalSide = np.concatenate([interval.left for interval in to_concat])
        # Concatenate the 'right' properties of all IntervalArray objects
        right: IntervalSide = np.concatenate([interval.right for interval in to_concat])

        # Ensure inputs are in a simple form and get the resulting dtype
        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed)

        # Create a new IntervalArray object using the concatenated properties
        return cls._simple_new(left, right, dtype=dtype)

    def copy(self) -> Self:
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
            Copy of the IntervalArray object
        """
        # Create copies of the 'left' and 'right' properties
        left = self._left.copy()
        right = self._right.copy()
        # Retrieve the dtype of the IntervalArray
        dtype = self.dtype
        # Return a new IntervalArray object with copied properties
        return self._simple_new(left, right, dtype=dtype)

    def isna(self) -> np.ndarray:
        """
        Check for missing values in the 'left' property.

        Returns
        -------
        np.ndarray
            Boolean array indicating missing values
        """
        return isna(self._left)

    def shift(self, periods: int = 1, fill_value: object = None) -> IntervalArray:
        """
        Shift the IntervalArray by a specified number of periods.

        Parameters
        ----------
        periods : int, optional
            Number of periods to shift (default is 1)
        fill_value : object, optional
            Value to use for filling any missing elements

        Returns
        -------
        IntervalArray
            Shifted IntervalArray object
        """
        if not len(self) or periods == 0:
            # If the IntervalArray is empty or no shift is required, return a copy of itself
            return self.copy()

        self._validate_scalar(fill_value)

        # Determine the length of the empty portion to be filled
        empty_len = min(abs(periods), len(self))

        if isna(fill_value):
            from pandas import Index

            # Retrieve the appropriate fill value for NaN
            fill_value = Index(self._left, copy=False)._na_value
            # Create an IntervalArray filled with NaN values
            empty = IntervalArray.from_breaks([fill_value] * (empty_len + 1))
        else:
            # Create an IntervalArray filled with custom fill values
            empty = self._from_sequence([fill_value] * empty_len, dtype=self.dtype)

        if periods > 0:
            # Shift the IntervalArray forward
            a = empty
            b = self[:-periods]
        else:
            # Shift the IntervalArray backward
            a = self[abs(periods) :]
            b = empty

        # Concatenate the shifted parts and return a new IntervalArray object
        return self._concat_same_type([a, b])

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value=None,
        axis=None,
        **kwargs,
    ):
        """
        Return the elements at the given indices.

        Parameters
        ----------
        indices : array-like
            Indices of elements to retrieve
        allow_fill : bool, optional
            Whether to allow filling with a value for indices out of bounds (default is False)
        fill_value : object, optional
            Value to use for filling if allow_fill is True
        axis : int, optional
            Not used in this function, retained for compatibility
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        IntervalArray
            IntervalArray containing elements at specified indices
        """
        # Implementation details not provided in the comment block.
        pass
    ) -> Self:
        """
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray
            A new IntervalArray containing elements taken based on `indices`.

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        # Validate the arguments passed to the function
        nv.validate_take((), kwargs)

        # Determine left and right fill values based on allow_fill option
        fill_left = fill_right = fill_value
        if allow_fill:
            # Validate and assign fill values if filling is allowed
            fill_left, fill_right = self._validate_scalar(fill_value)

        # Perform 'take' operation on the left and right endpoints of intervals
        left_take = take(
            self._left, indices, allow_fill=allow_fill, fill_value=fill_left
        )
        right_take = take(
            self._right, indices, allow_fill=allow_fill, fill_value=fill_right
        )

        # Return a shallow copy of IntervalArray with updated left and right endpoints
        return self._shallow_copy(left_take, right_take)

    def _validate_listlike(self, value):
        """
        Validate a list-like input for the IntervalArray.

        Parameters
        ----------
        value : sequence of intervals
            Input values to validate.

        Returns
        -------
        value_left : array-like
            Left endpoints of validated intervals.
        value_right : array-like
            Right endpoints of validated intervals.

        Raises
        ------
        TypeError
            If `value` is not a valid interval type or NA.
        """
        # Attempt to create an IntervalArray from the input value
        try:
            array = IntervalArray(value)
            # Check if the created IntervalArray matches closed status
            self._check_closed_matches(array, name="value")
            # Extract left and right endpoints from the IntervalArray
            value_left, value_right = array.left, array.right
        except TypeError as err:
            # Handle TypeError if input is not a valid interval type or NA
            msg = f"'value' should be an interval type, got {type(value)} instead."
            raise TypeError(msg) from err

        # Validate the left endpoint using the current IntervalArray
        try:
            self.left._validate_fill_value(value_left)
        except (LossySetitemError, TypeError) as err:
            # Handle TypeError or LossySetitemError during left endpoint validation
            msg = (
                "'value' should be a compatible interval type, "
                f"got {type(value)} instead."
            )
            raise TypeError(msg) from err

        # Return validated left and right endpoints
        return value_left, value_right
    # 验证单个值是否为有效的区间对象，返回区间的左右边界值
    def _validate_scalar(self, value):
        # 如果值是 Interval 对象
        if isinstance(value, Interval):
            # 检查闭合性匹配
            self._check_closed_matches(value, name="value")
            # 获取区间的左右边界值
            left, right = value.left, value.right
            # TODO: 检查是否需要像 _validate_setitem_value 方法那样进行子类型匹配？
        
        # 如果值是适合数据类型的 NA 值
        elif is_valid_na_for_dtype(value, self.left.dtype):
            # GH#18295
            # 使用左边界的 NA 值作为左右边界
            left = right = self.left._na_value
        
        else:
            # 抛出类型错误，只能插入 Interval 对象和 NA 值到 IntervalArray 中
            raise TypeError(
                "can only insert Interval objects and NA into an IntervalArray"
            )
        
        # 返回左右边界值
        return left, right

    # 验证要设置的值是否有效，返回值的左右边界
    def _validate_setitem_value(self, value):
        # 如果值是适合数据类型的 NA 值
        if is_valid_na_for_dtype(value, self.left.dtype):
            # NA 值：需要特殊处理直接设置到 numpy 数组上
            value = self.left._na_value
            # 如果数据类型的子类型是整数
            if is_integer_dtype(self.dtype.subtype):
                # 无法在 numpy 整数数组上设置 NaN
                # GH#45484 TypeError，而不是 ValueError，与非 NA 不可保持值所得到的匹配一致。
                raise TypeError("Cannot set float NaN to integer-backed IntervalArray")
            # 左右边界值都是该值
            value_left, value_right = value, value

        # 如果值是 Interval 对象
        elif isinstance(value, Interval):
            # 标量区间
            self._check_closed_matches(value, name="value")
            # 获取区间的左右边界值
            value_left, value_right = value.left, value.right
            # 验证填充值的有效性
            self.left._validate_fill_value(value_left)
            self.left._validate_fill_value(value_right)

        else:
            # 返回值的列表化验证结果
            return self._validate_listlike(value)

        # 返回值的左右边界
        return value_left, value_right

    def value_counts(self, dropna: bool = True) -> Series:
        """
        返回一个包含每个区间计数的 Series。

        Parameters
        ----------
        dropna : bool, 默认为 True
            是否不包括 NaN 值的计数。

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        # TODO: 以非朴素方式实现这个方法！
        # 调用 value_counts 函数，统计数组中每个区间的计数
        result = value_counts(np.asarray(self), dropna=dropna)
        # 将结果的索引转换为指定的数据类型
        result.index = result.index.astype(self.dtype)
        # 返回结果
        return result

    # ---------------------------------------------------------------------
    # 渲染方法

    def _formatter(self, boxed: bool = False) -> Callable[[object], str]:
        # 返回 'str'，导致我们以如 "(0, 1]" 的形式呈现，而不是 "Interval(0, 1, closed='right')"。
        return str

    # ---------------------------------------------------------------------
    # 矢量化区间属性/特性

    @property
    def left(self) -> Index:
        """
        Return the left endpoints of each Interval in the IntervalArray as an Index.
        
        Examples
        --------
        
        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.left
        Index([0, 2], dtype='int64')
        """
        from pandas import Index
        
        # 返回 IntervalArray 中每个 Interval 的左端点作为 Index 对象
        return Index(self._left, copy=False)

    @property
    def right(self) -> Index:
        """
        Return the right endpoints of each Interval in the IntervalArray as an Index.
        
        Examples
        --------
        
        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.right
        Index([1, 5], dtype='int64')
        """
        from pandas import Index
        
        # 返回 IntervalArray 中每个 Interval 的右端点作为 Index 对象
        return Index(self._right, copy=False)

    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.
        
        Examples
        --------
        
        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.length
        Index([1, 4], dtype='int64')
        """
        # 返回一个 Index，其中每个条目表示每个 Interval 的长度
        return self.right - self.left

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.
        
        Examples
        --------
        
        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.mid
        Index([0.5, 3.0], dtype='float64')
        """
        try:
            # 尝试计算每个 Interval 的中点并返回作为 Index 对象
            return 0.5 * (self.left + self.right)
        except TypeError:
            # 处理日期时间安全版本的中点计算
            return self.left + 0.5 * self.length
    # 将文档字符串分配给 _interval_shared_docs 字典的 "overlaps" 键，描述了 Interval 类的 overlaps 方法的作用和用法
    _interval_shared_docs["overlaps"] = textwrap.dedent(
        """
        Check elementwise if an Interval overlaps the values in the %(klass)s.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : %(klass)s
            Interval to check against for an overlap.

        Returns
        -------
        ndarray
            Boolean array positionally indicating where an overlap occurs.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.

        Examples
        --------
        %(examples)s
        >>> intervals.overlaps(pd.Interval(0.5, 1.5))
        array([ True,  True, False])

        Intervals that share closed endpoints overlap:

        >>> intervals.overlaps(pd.Interval(1, 3, closed='left'))
        array([ True,  True, True])

        Intervals that only have an open endpoint in common do not overlap:

        >>> intervals.overlaps(pd.Interval(1, 2, closed='right'))
        array([False,  True, False])
        """
    )

    # 将装饰器 @Appender 应用到 overlaps 方法上，将 _interval_shared_docs["overlaps"] 格式化并附加到方法文档中
    @Appender(
        _interval_shared_docs["overlaps"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """\
        >>> data = [(0, 1), (1, 3), (2, 4)]
        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    # 定义 overlaps 方法，用于确定当前 IntervalArray 对象的各个 Interval 是否与给定的 Interval 对象 other 重叠
    def overlaps(self, other):
        # 如果 other 是 IntervalArray 或 ABCIntervalIndex 类型，则抛出 NotImplementedError
        if isinstance(other, (IntervalArray, ABCIntervalIndex)):
            raise NotImplementedError
        # 如果 other 不是 Interval 类型，则抛出 TypeError
        if not isinstance(other, Interval):
            msg = f"`other` must be Interval-like, got {type(other).__name__}"
            raise TypeError(msg)

        # 根据当前 IntervalArray 对象的闭合属性和 other 的闭合属性，选择相应的比较操作符
        # 如果两个 Interval 在端点处都是闭合的，则使用小于等于操作符（le），否则使用小于操作符（lt）
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # 判断两个 Interval 是否重叠，返回一个布尔数组，指示各个 Interval 是否重叠
        # 重叠定义为两个 Interval 至少有一个公共点，包括闭合端点
        # 判断两个 Interval 是否不重叠等价于判断它们是否是不相交的
        # 即 (A.left > B.right) or (B.left > A.right) 的否定形式
        return op1(self.left, other.right) & op2(other.left, self.right)

    # ---------------------------------------------------------------------
    def closed(self) -> IntervalClosedType:
        """
        Return a string describing the inclusive side of the intervals.

        This method returns one of {'left', 'right', 'both', 'neither'}, indicating whether
        intervals are closed on the left side, right side, both sides, or neither side.

        Returns
        -------
        IntervalClosedType
            The inclusive side of the intervals.

        See Also
        --------
        IntervalArray.closed : Returns inclusive side of the IntervalArray.
        Interval.closed : Returns inclusive side of the Interval.
        IntervalIndex.closed : Returns inclusive side of the IntervalIndex.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.closed
        'right'

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.closed
        'right'
        """
        return self.dtype.closed

    _interval_shared_docs["set_closed"] = textwrap.dedent(
        """
        Return an identical %(klass)s closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.

        Returns
        -------
        %(klass)s

        %(examples)s\
        """
    )

    def set_closed(self, closed: IntervalClosedType) -> Self:
        """
        Return an identical IntervalArray closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Specifies the side(s) on which the intervals are closed.

        Returns
        -------
        Self
            A new instance of the IntervalArray with the specified closure settings.

        Raises
        ------
        ValueError
            If the specified `closed` parameter is not one of {'left', 'right', 'both', 'neither'}.

        See Also
        --------
        IntervalArray.closed : Returns inclusive side of the Interval.
        arrays.IntervalArray.closed : Returns inclusive side of the IntervalArray.

        Examples
        --------
        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))
        >>> index
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        >>> index.set_closed("both")
        <IntervalArray>
        [[0, 1], [1, 2], [2, 3]]
        Length: 3, dtype: interval[int64, both]
        """
        if closed not in VALID_CLOSED:
            msg = f"invalid option for 'closed': {closed}"
            raise ValueError(msg)

        left, right = self._left, self._right
        dtype = IntervalDtype(left.dtype, closed=closed)
        return self._simple_new(left, right, dtype=dtype)
    _interval_shared_docs["is_non_overlapping_monotonic"] = """
        Return a boolean whether the %(klass)s is non-overlapping and monotonic.

        Non-overlapping means (no Intervals share points), and monotonic means
        either monotonic increasing or monotonic decreasing.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        True

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),
        ...                                       pd.Interval(-1, 0.1)])
        >>> interv_arr
        <IntervalArray>
        [(0.0, 1.0], (-1.0, 0.1]]
        Length: 2, dtype: interval[float64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        False

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.is_non_overlapping_monotonic
        True

        >>> interv_idx = pd.interval_range(start=0, end=2, closed='both')
        >>> interv_idx
        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
        >>> interv_idx.is_non_overlapping_monotonic
        False
        """

    @property



    # 将文档字符串存储在_interval_shared_docs字典中，用于描述is_non_overlapping_monotonic属性的功能和示例
    _interval_shared_docs["is_non_overlapping_monotonic"] = """
        Return a boolean whether the %(klass)s is non-overlapping and monotonic.

        Non-overlapping means (no Intervals share points), and monotonic means
        either monotonic increasing or monotonic decreasing.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        True

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),
        ...                                       pd.Interval(-1, 0.1)])
        >>> interv_arr
        <IntervalArray>
        [(0.0, 1.0], (-1.0, 0.1]]
        Length: 2, dtype: interval[float64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        False

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.is_non_overlapping_monotonic
        True

        >>> interv_idx = pd.interval_range(start=0, end=2, closed='both')
        >>> interv_idx
        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
        >>> interv_idx.is_non_overlapping_monotonic
        False
        """
    # 返回一个布尔值，指示 IntervalArray/IntervalIndex 是否非重叠且单调。

    # 非重叠意味着没有区间共享点，单调意味着要么单调递增，要么单调递减。

    # 参考
    # --------
    # overlaps : 检查两个 IntervalIndex 对象是否重叠。

    # 示例
    # --------
    # 对于 IntervalArray:

    # >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    # >>> interv_arr
    # <IntervalArray>
    # [(0, 1], (1, 5]]
    # Length: 2, dtype: interval[int64, right]
    # >>> interv_arr.is_non_overlapping_monotonic
    # True

    # >>> interv_arr = pd.arrays.IntervalArray(
    # ...     [pd.Interval(0, 1), pd.Interval(-1, 0.1)]
    # ... )
    # >>> interv_arr
    # <IntervalArray>
    # [(0.0, 1.0], (-1.0, 0.1]]
    # Length: 2, dtype: interval[float64, right]
    # >>> interv_arr.is_non_overlapping_monotonic
    # False

    # 对于 IntervalIndex:

    # >>> interv_idx = pd.interval_range(start=0, end=2)
    # >>> interv_idx
    # IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
    # >>> interv_idx.is_non_overlapping_monotonic
    # True

    # >>> interv_idx = pd.interval_range(start=0, end=2, closed="both")
    # >>> interv_idx
    # IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
    # >>> interv_idx.is_non_overlapping_monotonic
    # False

    # ---------------------------------------------------------------------
    # 转换

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ):
    ) -> np.ndarray:
        """
        将 IntervalArray 的数据作为包含 Interval 对象的 numpy 数组（dtype='object'）返回
        """
        # 获取 IntervalArray 的左边界数组
        left = self._left
        # 获取 IntervalArray 的右边界数组
        right = self._right
        # 获取 IntervalArray 的缺失值掩码
        mask = self.isna()
        # 获取 IntervalArray 的闭合属性
        closed = self.closed

        # 创建一个空的对象数组，用于存储 Interval 对象
        result = np.empty(len(left), dtype=object)
        # 遍历左边界数组
        for i, left_value in enumerate(left):
            # 如果当前位置有缺失值，将结果数组相应位置设置为 NaN
            if mask[i]:
                result[i] = np.nan
            else:
                # 否则，根据左右边界和闭合属性创建 Interval 对象，并存储到结果数组中
                result[i] = Interval(left_value, right[i], closed)
        # 返回存储 Interval 对象的 numpy 数组
        return result

    def __arrow_array__(self, type=None):
        """
        将自身转换为 pyarrow Array。
        """
        import pyarrow

        from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

        try:
            # 尝试从 numpy 数据类型中获取子类型信息
            subtype = pyarrow.from_numpy_dtype(self.dtype.subtype)
        except TypeError as err:
            # 如果转换失败，抛出类型错误异常
            raise TypeError(
                f"Conversion to arrow with subtype '{self.dtype.subtype}' "
                "is not supported"
            ) from err
        # 使用子类型和闭合属性创建 ArrowIntervalType 对象
        interval_type = ArrowIntervalType(subtype, self.closed)
        # 根据左右边界数组创建结构化数组存储
        storage_array = pyarrow.StructArray.from_arrays(
            [
                pyarrow.array(self._left, type=subtype, from_pandas=True),
                pyarrow.array(self._right, type=subtype, from_pandas=True),
            ],
            names=["left", "right"],
        )
        # 获取 IntervalArray 的缺失值掩码
        mask = self.isna()
        # 如果存在缺失值，设置存储数组的有效位图
        if mask.any():
            null_bitmap = pyarrow.array(~mask).buffers()[1]
            storage_array = pyarrow.StructArray.from_buffers(
                storage_array.type,
                len(storage_array),
                [null_bitmap],
                children=[storage_array.field(0), storage_array.field(1)],
            )

        # 如果指定了转换的类型
        if type is not None:
            # 检查是否与 ArrowIntervalType 的存储类型相同
            if type.equals(interval_type.storage_type):
                return storage_array
            elif isinstance(type, ArrowIntervalType):
                # 确保具有相同的子类型和闭合属性
                if not type.equals(interval_type):
                    raise TypeError(
                        "Not supported to convert IntervalArray to type with "
                        f"different 'subtype' ({self.dtype.subtype} vs {type.subtype}) "
                        f"and 'closed' ({self.closed} vs {type.closed}) attributes"
                    )
            else:
                # 如果不是 ArrowIntervalType 类型，则抛出类型错误异常
                raise TypeError(
                    f"Not supported to convert IntervalArray to '{type}' type"
                )

        # 返回使用 ArrowIntervalType 和存储数组创建的扩展数组
        return pyarrow.ExtensionArray.from_storage(interval_type, storage_array)
    # 将 "to_tuples" 添加到共享文档中，用于生成文档字符串
    _interval_shared_docs["to_tuples"] = textwrap.dedent(
        """
        Return an %(return_type)s of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        tuples: %(return_type)s
        %(examples)s\
        """
    )

    # 定义方法 "to_tuples"，返回表示区间的元组数组或索引
    def to_tuples(self, na_tuple: bool = True) -> np.ndarray:
        """
        Return an ndarray (if self is IntervalArray) or Index \
        (if self is IntervalIndex) of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        ndarray or Index
            An ndarray of tuples representing the intervals
                if `self` is an IntervalArray.
            An Index of tuples representing the intervals
                if `self` is an IntervalIndex.

        See Also
        --------
        IntervalArray.to_list : Convert IntervalArray to a list of tuples.
        IntervalArray.to_numpy : Convert IntervalArray to a numpy array.
        IntervalArray.unique : Find unique intervals in an IntervalArray.

        Examples
        --------
        For :class:`pandas.IntervalArray`:

        >>> idx = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        >>> idx
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        >>> idx.to_tuples()
        array([(0, 1), (1, 2)], dtype=object)

        For :class:`pandas.IntervalIndex`:

        >>> idx = pd.interval_range(start=0, end=2)
        >>> idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> idx.to_tuples()
        Index([(0, 1), (1, 2)], dtype='object')
        """
        # 调用 asarray_tuplesafe 函数处理左右端点，生成元组数组
        tuples = com.asarray_tuplesafe(zip(self._left, self._right))
        if not na_tuple:
            # 如果不返回 NA 元组，则将元组中的 NA 替换为 nan（GH 18756）
            tuples = np.where(~self.isna(), tuples, np.nan)
        return tuples

    # ---------------------------------------------------------------------

    # 定义方法 "_putmask"，用于根据掩码修改区间的值
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # 验证设置项的值，并分别获取左右端点的值
        value_left, value_right = self._validate_setitem_value(value)

        if isinstance(self._left, np.ndarray):
            # 如果左端点是 ndarray 类型，则根据掩码设置左右端点的值
            np.putmask(self._left, mask, value_left)
            assert isinstance(self._right, np.ndarray)
            np.putmask(self._right, mask, value_right)
        else:
            # 否则递归调用左右端点对象的 "_putmask" 方法
            self._left._putmask(mask, value_left)
            assert not isinstance(self._right, np.ndarray)
            self._right._putmask(mask, value_right)
    # 在 IntervalArray 中插入一个新的 Interval 对象到指定位置
    def insert(self, loc: int, item: Interval) -> Self:
        """
        Return a new IntervalArray inserting new item at location. Follows
        Python numpy.insert semantics for negative values.  Only Interval
        objects and NA can be inserted into an IntervalIndex

        Parameters
        ----------
        loc : int
            插入位置的索引
        item : Interval
            要插入的 Interval 对象

        Returns
        -------
        IntervalArray
            返回一个新的 IntervalArray 对象
        """
        # 验证并获取要插入的 Interval 对象的左右边界
        left_insert, right_insert = self._validate_scalar(item)

        # 在当前对象的左边界数组中插入新的左边界
        new_left = self.left.insert(loc, left_insert)
        # 在当前对象的右边界数组中插入新的右边界
        new_right = self.right.insert(loc, right_insert)

        # 返回一个浅拷贝的新 IntervalArray 对象，更新了左右边界数组
        return self._shallow_copy(new_left, new_right)

    # 从 IntervalArray 中删除指定位置的 Interval 对象
    def delete(self, loc) -> Self:
        new_left: np.ndarray | DatetimeArray | TimedeltaArray
        new_right: np.ndarray | DatetimeArray | TimedeltaArray
        # 如果左边界是 numpy 数组，则从左右边界数组中删除对应位置的元素
        if isinstance(self._left, np.ndarray):
            new_left = np.delete(self._left, loc)
            assert isinstance(self._right, np.ndarray)
            new_right = np.delete(self._right, loc)
        else:
            # 如果左边界不是 numpy 数组，则调用左右边界对象的 delete 方法删除对应位置的元素
            new_left = self._left.delete(loc)
            assert not isinstance(self._right, np.ndarray)
            new_right = self._right.delete(loc)
        # 返回一个浅拷贝的新 IntervalArray 对象，更新了左右边界数组
        return self._shallow_copy(left=new_left, right=new_right)

    # 重复 IntervalArray 中的元素指定的次数
    @Appender(_extension_array_shared_docs["repeat"] % _shared_docs_kwargs)
    def repeat(
        self,
        repeats: int | Sequence[int],
        axis: AxisInt | None = None,
    ) -> Self:
        # 验证重复操作的参数是否有效
        nv.validate_repeat((), {"axis": axis})
        # 对当前对象的左右边界数组进行重复操作
        left_repeat = self.left.repeat(repeats)
        right_repeat = self.right.repeat(repeats)
        # 返回一个浅拷贝的新 IntervalArray 对象，更新了左右边界数组
        return self._shallow_copy(left=left_repeat, right=right_repeat)

    # 检查 IntervalArray 中是否包含特定值的元素
    _interval_shared_docs["contains"] = textwrap.dedent(
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the %(klass)s.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array
            返回一个布尔数组，指示是否包含特定值

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        %(klass)s.overlaps : Check if an Interval overlaps the values in the
            %(klass)s.

        Examples
        --------
        %(examples)s
        >>> intervals.contains(0.5)
        array([ True, False, False])
    """
    )
    def contains(self, other):
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the IntervalArray.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array
            A boolean mask whether the value is contained in the Intervals.

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        IntervalArray.overlaps : Check if an Interval overlaps the values in the
            IntervalArray.

        Examples
        --------
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.contains(0.5)
        array([ True, False, False])
        """
        if isinstance(other, Interval):
            # 如果 other 是 Interval 类型，则抛出未实现的错误，因为不支持两个 Interval 对象之间的 contains 操作
            raise NotImplementedError("contains not implemented for two intervals")

        return (self._left < other if self.open_left else self._left <= other) & (
            other < self._right if self.open_right else other <= self._right
        )

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        if isinstance(values, IntervalArray):
            if self.closed != values.closed:
                # 如果当前 IntervalArray 与传入的 values 的闭合性质不同，则返回全 False 的布尔数组，表示无重叠
                return np.zeros(self.shape, dtype=bool)

            if self.dtype == values.dtype:
                # 如果当前 IntervalArray 与 values 的数据类型相同，则进行比较
                # 使用 complex128 视图提高性能，而不是转换为对象类型进行操作
                left = self._combined.view("complex128")
                right = values._combined.view("complex128")
                # 使用 np.isin 检查 left 中的元素是否在 right 中，返回展平后的结果数组
                return np.isin(left, right).ravel()  # type: ignore[arg-type]

            elif needs_i8_conversion(self.left.dtype) ^ needs_i8_conversion(
                values.left.dtype
            ):
                # 如果当前 IntervalArray 与 values 的数据类型不同，且其中一个需要将 int64 转换为 int，则返回全 False 的布尔数组
                return np.zeros(self.shape, dtype=bool)

        # 如果 values 不是 IntervalArray 类型，则将当前 IntervalArray 和 values 转换为对象类型进行比较
        return isin(self.astype(object), values.astype(object))

    @property
    def _combined(self) -> IntervalSide:
        """
        Combine left and right interval values into a structured array.
        """
        # 将左侧值重塑为列向量
        left = self.left._values.reshape(-1, 1)  # type: ignore[union-attr]
        # 将右侧值重塑为列向量
        right = self.right._values.reshape(-1, 1)  # type: ignore[union-attr]
        if needs_i8_conversion(left.dtype):
            # 如果左侧值需要转换成i8类型，则使用特定的方法进行连接
            comb = left._concat_same_type(  # type: ignore[union-attr]
                [left, right], axis=1
            )
        else:
            # 否则使用numpy的concatenate函数进行连接
            comb = np.concatenate([left, right], axis=1)
        return comb

    def _from_combined(self, combined: np.ndarray) -> IntervalArray:
        """
        Create a new IntervalArray from a 1D complex128 ndarray.
        """
        # 将复合数组视图转换为i8类型，并按2列重塑
        nc = combined.view("i8").reshape(-1, 2)

        dtype = self._left.dtype
        if needs_i8_conversion(dtype):
            # 如果需要将dtype转换为i8类型，则从序列中创建新的左右数组
            assert isinstance(self._left, (DatetimeArray, TimedeltaArray))
            new_left: DatetimeArray | TimedeltaArray | np.ndarray = type(
                self._left
            )._from_sequence(nc[:, 0], dtype=dtype)
            assert isinstance(self._right, (DatetimeArray, TimedeltaArray))
            new_right: DatetimeArray | TimedeltaArray | np.ndarray = type(
                self._right
            )._from_sequence(nc[:, 1], dtype=dtype)
        else:
            # 否则直接从nc数组中视图创建左右数组
            assert isinstance(dtype, np.dtype)
            new_left = nc[:, 0].view(dtype)
            new_right = nc[:, 1].view(dtype)
        return self._shallow_copy(left=new_left, right=new_right)

    def unique(self) -> IntervalArray:
        """
        Return unique elements of the combined interval values.
        """
        # 从复合视图中选择第一列，并确保唯一性
        nc = unique(
            self._combined.view("complex128")[:, 0]  # type: ignore[call-overload]
        )
        # 将结果转换为列向量
        nc = nc[:, None]
        return self._from_combined(nc)
# 尝试进行平台转换，特别处理 IntervalArray。
# 包装了 maybe_convert_platform 的函数，在某些情况下修改默认返回的 dtype 以兼容 IntervalArray。
# 例如，空列表返回整数 dtype 而不是对象 dtype，因为对象 dtype 对于 IntervalArray 是不允许的。

def _maybe_convert_platform_interval(values) -> ArrayLike:
    """
    Try to do platform conversion, with special casing for IntervalArray.
    Wrapper around maybe_convert_platform that alters the default return
    dtype in certain cases to be compatible with IntervalArray.  For example,
    empty lists return with integer dtype instead of object dtype, which is
    prohibited for IntervalArray.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    array
    """
    if isinstance(values, (list, tuple)) and len(values) == 0:
        # 如果是空列表或空元组
        # GH 19016
        # 空列表/元组默认会返回对象 dtype，但这对于 IntervalArray 是禁止的，所以强制返回整数 dtype
        return np.array([], dtype=np.int64)
    elif not is_list_like(values) or isinstance(values, ABCDataFrame):
        # 如果不是类列表结构或者是 DataFrame 类型
        # 这会在后面引发错误，但我们避免将其传递给 maybe_convert_platform
        return values
    elif isinstance(getattr(values, "dtype", None), CategoricalDtype):
        # 如果 values 具有 CategoricalDtype 类型的 dtype
        values = np.asarray(values)
    elif not hasattr(values, "dtype") and not isinstance(values, (list, tuple, range)):
        # 如果 values 没有 dtype 属性且不是列表、元组或范围类型
        # TODO: 我们应该直接将其转换为列表吗？
        return values
    else:
        # 从 values 中提取数组，使用 extract_array 函数，确保返回一个 numpy 数组
        values = extract_array(values, extract_numpy=True)

    if not hasattr(values, "dtype"):
        # 如果 values 没有 dtype 属性
        values = np.asarray(values)
        if values.dtype.kind in "iu" and values.dtype != np.int64:
            # 如果 values 的 dtype 种类是整数或无符号整数，并且不是 np.int64
            values = values.astype(np.int64)
    return values
```