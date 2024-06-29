# `D:\src\scipysrc\pandas\pandas\core\indexing.py`

```
from __future__ import annotations
# 导入未来支持的类型注释特性，使得可以在类型提示中使用字符串形式的类名称

from contextlib import suppress
# 导入用于忽略异常的上下文管理器

import sys
# 导入系统相关的功能模块

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    final,
)
# 导入类型提示相关模块，包括类型检查器、通用类型变量、类型强制转换、最终方法装饰器等

import warnings
# 导入警告相关模块

import numpy as np
# 导入NumPy库，用于数值计算

from pandas._libs.indexing import NDFrameIndexerBase
# 导入Pandas内部的索引基类

from pandas._libs.lib import item_from_zerodim
# 导入Pandas内部的从零维对象获取单个元素的方法

from pandas.compat import PYPY
# 导入Pandas的兼容性模块，用于检查是否在PyPy环境中运行

from pandas.errors import (
    AbstractMethodError,
    ChainedAssignmentError,
    IndexingError,
    InvalidIndexError,
    LossySetitemError,
)
# 导入Pandas的错误类，包括抽象方法错误、链式赋值错误、索引错误、无效索引错误、损失性设置项错误

from pandas.errors.cow import _chained_assignment_msg
# 导入Pandas的链式赋值错误消息

from pandas.util._decorators import doc
# 导入Pandas的装饰器模块

from pandas.util._exceptions import find_stack_level
# 导入Pandas的异常处理模块，用于查找堆栈级别

from pandas.core.dtypes.cast import (
    can_hold_element,
    maybe_promote,
)
# 导入Pandas的数据类型转换模块，包括判断是否能容纳元素、可能的升级方法

from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_sequence,
)
# 导入Pandas的常用数据类型判断方法，包括数组类、布尔型、可哈希类型、整数类型、迭代器、列表类、数值类型、对象类型、标量、序列类

from pandas.core.dtypes.concat import concat_compat
# 导入Pandas的数据合并兼容模块

from pandas.core.dtypes.dtypes import ExtensionDtype
# 导入Pandas的扩展数据类型模块

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
# 导入Pandas的通用数据框架基类和序列基类

from pandas.core.dtypes.missing import (
    construct_1d_array_from_inferred_fill_value,
    infer_fill_value,
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)
# 导入Pandas的缺失数据处理模块，包括从推断的填充值构造一维数组、推断填充值、检查数据类型是否有效的NA值、判断是否为NA值、根据数据类型获取NA值

from pandas.core import algorithms as algos
# 导入Pandas的算法模块

import pandas.core.common as com
# 导入Pandas的常用工具模块

from pandas.core.construction import (
    array as pd_array,
    extract_array,
)
# 导入Pandas的构造模块，包括数组、提取数组等

from pandas.core.indexers import (
    check_array_indexer,
    is_list_like_indexer,
    is_scalar_indexer,
    length_of_indexer,
)
# 导入Pandas的索引器模块，包括检查数组索引器、判断是否为列表索引器、判断是否为标量索引器、获取索引器长度

from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)
# 导入Pandas的索引API模块，包括索引、多级索引

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )
    # 如果是类型检查阶段，则从标准库中导入可哈希对象和序列类型

    from pandas._typing import (
        Axis,
        AxisInt,
        Self,
        npt,
    )
    # 导入Pandas的类型提示模块，包括轴、整数轴、自身类型、NumPy类型

    from pandas import (
        DataFrame,
        Series,
    )
    # 导入Pandas的数据框架和序列类

T = TypeVar("T")
# 定义泛型类型变量"T"

# "null slice"
_NS = slice(None, None)
# 定义特殊的空切片对象，用于表示完整切片范围

_one_ellipsis_message = "indexer may only contain one '...' entry"
# 定义错误消息，指出索引器中只能包含一个省略号的条目

# the public IndexSlicerMaker
class _IndexSlice:
    """
    Create an object to more easily perform multi-index slicing.

    See Also
    --------
    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.

    Notes
    -----
    See :ref:`Defined Levels <advanced.shown_levels>`
    for further info on slicing a MultiIndex.

    Examples
    --------
    >>> midx = pd.MultiIndex.from_product([["A0", "A1"], ["B0", "B1", "B2", "B3"]])
    >>> columns = ["foo", "bar"]
    >>> dfmi = pd.DataFrame(
    ...     np.arange(16).reshape((len(midx), len(columns))),
    ...     index=midx,
    ...     columns=columns,
    ... )

    Using the default slice command:

    >>> dfmi.loc[(slice(None), slice("B0", "B1")), :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11

    Using the IndexSlice class for a more intuitive command:

    >>> idx = pd.IndexSlice

    """
    # 定义公共的IndexSlice类，用于更轻松地执行多级索引切片操作
    """
        >>> dfmi.loc[idx[:, "B0":"B1"], :]
                   foo  bar
            A0 B0    0    1
               B1    2    3
            A1 B0    8    9
               B1   10   11
        """
    
        # 定义一个特殊方法 __getitem__，用于对象的索引操作
        def __getitem__(self, arg):
            # 直接返回参数 arg，即实现了简单的索引功能
            return arg
# 创建一个 _IndexSlice 对象的实例，并将其赋值给 IndexSlice 变量
IndexSlice = _IndexSlice()

# 定义一个 Mixin 类，用于为 Dataframes 和 Series 添加 .loc/.iloc/.at/.iat 功能
class IndexingMixin:
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """

    @property
    # 定义一个 at 属性，返回一个 _AtIndexer 对象
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.

        Similar to ``loc``, in that both provide label-based lookups. Use
        ``at`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        KeyError
            If getting a value and 'label' does not exist in a DataFrame or Series.

        ValueError
            If row/column label pair is not a tuple or if any label
            from the pair is not a scalar for DataFrame.
            If label is list-like (*excluding* NamedTuple) for Series.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column pair by label.
        DataFrame.iat : Access a single value for a row/column pair by integer
            position.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer
            position(s).
        Series.at : Access a single value by label.
        Series.iat : Access a single value by integer position.
        Series.loc : Access a group of rows by label(s).
        Series.iloc : Access a group of rows by integer position(s).

        Notes
        -----
        See :ref:`Fast scalar value getting and setting <indexing.basics.get_value>`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...     index=[4, 5, 6],
        ...     columns=["A", "B", "C"],
        ... )
        >>> df
            A   B   C
        4   0   2   3
        5   0   4   1
        6  10  20  30

        Get value at specified row/column pair

        >>> df.at[4, "B"]
        2

        Set value at specified row/column pair

        >>> df.at[4, "B"] = 10
        >>> df.at[4, "B"]
        10

        Get value within a Series

        >>> df.loc[5].at["B"]
        4
        """
        return _AtIndexer("at", self)

    @property
    # 以下部分缺失了具体的代码，无法添加注释
    # 定义一个方法 `iat`，用于通过整数位置访问行/列对应的单个值。

    # 与 `iloc` 类似，都提供基于整数的查找。如果只需在 DataFrame 或 Series 中获取或设置单个值，则使用 `iat`。

    # 当整数位置超出范围时，抛出 IndexError 异常。

    # 参见：
    # DataFrame.at：通过行/列标签对访问单个值。
    # DataFrame.loc：通过标签访问一组行和列。
    # DataFrame.iloc：通过整数位置访问一组行和列。

    # 示例：
    # 创建一个 DataFrame
    # >>> df = pd.DataFrame(
    # ...     [[0, 2, 3], [0, 4, 1], [10, 20, 30]], columns=["A", "B", "C"]
    # ... )
    # >>> df
    #    A   B   C
    # 0   0   2   3
    # 1   0   4   1
    # 2  10  20  30

    # 获取指定行/列位置处的值
    # >>> df.iat[1, 2]
    # 1

    # 设置指定行/列位置处的值
    # >>> df.iat[1, 2] = 10
    # >>> df.iat[1, 2]
    # 10

    # 在 Series 内部获取值
    # >>> df.loc[0].iat[1]

    """
    返回一个 `_iAtIndexer` 对象，标识为 'iat'，传入当前对象的引用（self）。
    """
    return _iAtIndexer("iat", self)
class _LocationIndexer(NDFrameIndexerBase):
    _valid_types: str
    axis: AxisInt | None = None

    # sub-classes need to set _takeable
    _takeable: bool

    @final
    def __call__(self, axis: Axis | None = None) -> Self:
        # 创建一个新的 _LocationIndexer 对象，以确保返回自身的副本
        new_self = type(self)(self.name, self.obj)

        if axis is not None:
            # 如果指定了轴，则获取该轴的整数编号
            axis_int_none = self.obj._get_axis_number(axis)
        else:
            # 否则将轴设置为 None
            axis_int_none = axis
        # 更新新对象的轴属性
        new_self.axis = axis_int_none
        return new_self

    def _get_setitem_indexer(self, key):
        """
        Convert a potentially-label-based key into a positional indexer.
        """
        if self.name == "loc":
            # 由于 iloc 会覆盖 _get_setitem_indexer，这里总是成立
            self._ensure_listlike_indexer(key, axis=self.axis)

        if isinstance(key, tuple):
            # 检查元组中的每个元素是否为字典或集合，用于多索引处理
            for x in key:
                check_dict_or_set_indexers(x)

        if self.axis is not None:
            # 将索引键转换为适合指定轴的元组形式
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)

        ax = self.obj._get_axis(0)

        if (
            isinstance(ax, MultiIndex)
            and self.name != "iloc"
            and is_hashable(key)
            and not isinstance(key, slice)
        ):
            with suppress(KeyError, InvalidIndexError):
                # 尝试获取键在 MultiIndex 中的位置
                return ax.get_loc(key)

        if isinstance(key, tuple):
            with suppress(IndexingError):
                # 如果出现索引错误，忽略该异常
                return self._convert_tuple(key)

        if isinstance(key, range):
            # 处理 range 类型的索引键，将其转换为列表形式
            key = list(key)

        # 将索引键转换为适合于轴 0 的位置索引器
        return self._convert_to_indexer(key, axis=0)

    @final
    # 如果索引器是元组且长度为2，且值是 Series 或 DataFrame 类型
    # 则进行条件判断和处理，类似于 Series.__setitem__
    if (
        isinstance(indexer, tuple)
        and len(indexer) == 2
        and isinstance(value, (ABCSeries, ABCDataFrame))
    ):
        # 取出索引器和列索引器
        pi, icols = indexer
        # 获取 value 的维度信息
        ndim = value.ndim
        # 如果 pi 是布尔索引器且长度与 value 相同
        if com.is_bool_indexer(pi) and len(value) == len(pi):
            # 找出 pi 中为真的索引位置
            newkey = pi.nonzero()[0]

            # 如果 icols 是标量索引器且 value 是一维的
            if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                # 对齐 Series，处理对应位置的值
                value = self.obj.iloc._align_series(indexer, value)
                # 更新索引器为新的索引和列索引
                indexer = (newkey, icols)

            # 如果 icols 是整数 ndarray 且长度为1
            elif (
                isinstance(icols, np.ndarray)
                and icols.dtype.kind == "i"
                and len(icols) == 1
            ):
                # 如果 value 是一维的
                if ndim == 1:
                    # 对齐 Series，处理对应位置的值
                    value = self.obj.iloc._align_series(indexer, value)
                    # 更新索引器为新的索引和列索引
                    indexer = (newkey, icols)

                # 如果 value 是二维的且列数为1
                elif ndim == 2 and value.shape[1] == 1:
                    # 对齐 DataFrame，处理对应位置的值
                    value = self.obj.iloc._align_frame(indexer, value)
                    # 更新索引器为新的索引和列索引

# 如果索引器是布尔索引器，则找出为真的索引位置
elif com.is_bool_indexer(indexer):
    indexer = indexer.nonzero()[0]

# 返回更新后的索引器和值
return indexer, value
    # 确保传入的索引器是类列表的列标签，如果不存在则添加
    def _ensure_listlike_indexer(self, key, axis=None, value=None) -> None:
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
        # 默认列轴为1
        column_axis = 1

        # 如果数据框不是二维的，则直接返回
        if self.ndim != 2:
            return

        # 如果 key 是元组并且长度大于1，则假设我们是在使用 .loc
        if isinstance(key, tuple) and len(key) > 1:
            # 如果 key 是元组且长度大于1，则将 key 设置为列部分，除非已经指定了轴
            if axis is None:
                axis = column_axis
            key = key[axis]

        # 如果轴是列轴并且列不是 MultiIndex，并且 key 是类列表的索引器，并且不是布尔索引器，并且所有的 key 都是可哈希的
        if (
            axis == column_axis
            and not isinstance(self.obj.columns, MultiIndex)
            and is_list_like_indexer(key)
            and not com.is_bool_indexer(key)
            and all(is_hashable(k) for k in key)
        ):
            # 合并现有的列名和新的 key，不进行排序
            keys = self.obj.columns.union(key, sort=False)
            # 找出差异的部分
            diff = Index(key).difference(self.obj.columns, sort=False)

            # 如果存在差异
            if len(diff):
                # 例如，如果我们执行 df.loc[:, ["A", "B"]] = 7，并且 "B" 是一个新列，则添加新列，使用 dtype=np.void
                # 这样当我们后续执行 setitem_single_column 时，会使用 isetitem。
                # 如果没有这一步，下面的 reindex_axis 将会在此示例中创建 float64 列，它们可以成功保存 7，所以最终会得到错误的 dtype。
                indexer = np.arange(len(keys), dtype=np.intp)
                indexer[len(self.obj.columns) :] = -1
                # 使用重新索引器重新索引管理器，仅使用切片，使用 NA 代理
                new_mgr = self.obj._mgr.reindex_indexer(
                    keys, indexer=indexer, axis=0, only_slice=True, use_na_proxy=True
                )
                self.obj._mgr = new_mgr
                return

            # 使用重新索引器重新索引轴，仅使用切片
            self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, only_slice=True)

    # 最终的方法，用于设置项目的值，当 PYPY 不为真时，检查引用计数是否小于等于2，如果是则发出警告
    @final
    def __setitem__(self, key, value) -> None:
        if not PYPY:
            if sys.getrefcount(self.obj) <= 2:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        # 检查 key 是否为字典或设置索引器
        check_dict_or_set_indexers(key)

        # 如果 key 是元组，则将其转换为列表（如果 x 是迭代器则转换为列表），然后将其应用于 self.obj
        if isinstance(key, tuple):
            key = (list(x) if is_iterator(x) else x for x in key)
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            # 如果 key 可调用，则将其应用于 self.obj
            maybe_callable = com.apply_if_callable(key, self.obj)
            key = self._raise_callable_usage(key, maybe_callable)

        # 获取设置项目的索引器
        indexer = self._get_setitem_indexer(key)

        # 检查索引器是否有效
        self._has_valid_setitem_indexer(key)

        # 如果 self.name 是 "iloc"，则 iloc 是当前对象，否则使用 self.obj.iloc
        iloc = self if self.name == "iloc" else self.obj.iloc

        # 使用索引器设置项目值
        iloc._setitem_with_indexer(indexer, value, self.name)
    def _validate_key(self, key, axis: AxisInt) -> None:
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
        # 抽象方法错误，子类必须实现具体的验证逻辑
        raise AbstractMethodError(self)

    @final
    def _expand_ellipsis(self, tup: tuple) -> tuple:
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
        # 检查元组中是否包含省略号（Ellipsis）
        if any(x is Ellipsis for x in tup):
            # 如果有多个省略号，抛出索引错误
            if tup.count(Ellipsis) > 1:
                raise IndexingError(_one_ellipsis_message)

            # 如果元组长度等于对象的维度，则将省略号替换为单个空切片
            if len(tup) == self.ndim:
                # 这里假设只有一个省略号，找到省略号的位置并替换为 (_NS,)
                i = tup.index(Ellipsis)
                new_key = tup[:i] + (_NS,) + tup[i + 1 :]
                return new_key

            # TODO: 其他情况需要处理吗？目前只有一个测试会到达这里，并且已被 _validate_key_length 覆盖

        # 如果没有省略号，直接返回原始元组
        return tup

    @final
    def _validate_tuple_indexer(self, key: tuple) -> tuple:
        """
        Check the key for valid keys across my indexer.
        """
        # 首先检查键的长度是否有效
        key = self._validate_key_length(key)
        # 扩展元组中的省略号
        key = self._expand_ellipsis(key)
        # 逐个验证元组中的每个键
        for i, k in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                # 如果验证失败，抛出具体错误信息
                raise ValueError(
                    "Location based indexing can only have "
                    f"[{self._valid_types}] types"
                ) from err
        return key

    @final
    def _is_nested_tuple_indexer(self, tup: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
        # 如果对象的轴中有任何一个是 MultiIndex 类型，则检查元组是否嵌套
        if any(isinstance(ax, MultiIndex) for ax in self.obj.axes):
            return any(is_nested_tuple(tup, ax) for ax in self.obj.axes)
        return False

    @final
    def _convert_tuple(self, key: tuple) -> tuple:
        # 注意：假设 _tupleize_axis_indexer 已经被调用，如果有必要的话。
        # 首先验证键的长度是否有效
        self._validate_key_length(key)
        # 将元组中的每个键转换为索引器
        keyidx = [self._convert_to_indexer(k, axis=i) for i, k in enumerate(key)]
        return tuple(keyidx)

    @final
    def _validate_key_length(self, key: tuple) -> tuple:
        # 检查索引器 key 的长度是否超过 self.ndim
        if len(key) > self.ndim:
            # 如果 key 的第一个元素是 Ellipsis（省略号），例如 Series.iloc[..., 3] 可简化为 Series.iloc[3]
            if key[0] is Ellipsis:
                # 去除第一个 Ellipsis，继续递归检查剩余的 key
                key = key[1:]
                # 如果剩余 key 中还有 Ellipsis，则抛出索引错误异常
                if Ellipsis in key:
                    raise IndexingError(_one_ellipsis_message)
                # 递归调用 _validate_key_length 函数检查修正后的 key
                return self._validate_key_length(key)
            # 如果 key 的长度超过 self.ndim 但第一个元素不是 Ellipsis，则抛出索引错误异常
            raise IndexingError("Too many indexers")
        # 如果 key 的长度未超过 self.ndim，则直接返回 key
        return key

    @final
    def _getitem_tuple_same_dim(self, tup: tuple):
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.

        This is only called after a failed call to _getitem_lowerdim.
        """
        # 将初始返回值设置为 self.obj
        retval = self.obj
        # 优先选择列再选择行，可以显著提高性能
        start_val = (self.ndim - len(tup)) + 1
        # 逆序遍历 tup 中的索引器
        for i, key in enumerate(reversed(tup)):
            # 计算当前轴的索引值
            i = self.ndim - i - start_val
            # 如果 key 是空的切片（null slice），则跳过当前循环
            if com.is_null_slice(key):
                continue

            # 调用 getattr 获取当前 retval 对象中的 self.name 属性，再调用 _getitem_axis 方法进行索引操作
            retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
            # 确保 retval 的维度不会小于 self.ndim，因为这种情况应该在上面的 _getitem_lowerdim 调用中处理
            assert retval.ndim == self.ndim

        # 如果 retval 仍然是初始的 self.obj，说明所有轴都是空切片（df.loc[:, :]），确保返回一个新对象
        if retval is self.obj:
            retval = retval.copy(deep=False)

        # 返回最终结果
        return retval

    @final
    # 处理低维度下的索引操作，接收一个元组作为参数
    def _getitem_lowerdim(self, tup: tuple):
        # 如果指定了轴（axis），直接使用轴的结果
        if self.axis is not None:
            # 获取指定轴的编号
            axis = self.obj._get_axis_number(self.axis)
            # 调用 _getitem_axis 方法进行索引操作
            return self._getitem_axis(tup, axis=axis)

        # 处理可能存在的嵌套元组索引
        if self._is_nested_tuple_indexer(tup):
            return self._getitem_nested_tuple(tup)

        # 处理可能使用元组表示多个维度的情况
        ax0 = self.obj._get_axis(0)
        # 但是 iloc 应该将元组视为简单的整数位置索引，而不是多索引表示（GH 13797）
        if (
            isinstance(ax0, MultiIndex)
            and self.name != "iloc"
            and not any(isinstance(x, slice) for x in tup)
        ):
            # 注意：在所有现有的测试用例中，将切片条件替换为
            # `all(is_hashable(x) or com.is_null_slice(x) for x in tup)`
            # 是等效的。
            # （参见我们调用 _handle_lowerdim_multi_index_axis0 的另一个地方）
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)

        # 验证键的长度
        tup = self._validate_key_length(tup)

        # 遍历元组中的每个键值对
        for i, key in enumerate(tup):
            # 如果键是标签样式
            if is_label_like(key):
                # 这里不需要检查元组，因为这些情况在上面的 _is_nested_tuple_indexer 检查中已经捕获了。
                # 通过指定的轴（axis=i），获取对应轴上的切片
                section = self._getitem_axis(key, axis=i)

                # 我们不应该在这里有标量的 section，因为 _getitem_lowerdim 只在检查 is_scalar_access 之后调用，而那时它会返回 True。
                if section.ndim == self.ndim:
                    # 我们正在通过 MultiIndex 切片，根据 `section` 修改键为插入 _NS 的新键
                    new_key = tup[:i] + (_NS,) + tup[i + 1 :]

                else:
                    # 注意：上面的 section.ndim == self.ndim 检查排除了 DataFrame 的情况，所以我们不需要担心转置。
                    new_key = tup[:i] + tup[i + 1 :]

                    # 如果新键的长度为 1，将其转换为单个元素而不是元组
                    if len(new_key) == 1:
                        new_key = new_key[0]

                # 切片应该返回视图，但是用空切片调用 iloc/loc 会返回一个新对象。
                if com.is_null_slice(new_key):
                    return section
                # 这是一个省略的递归调用 iloc/loc
                return getattr(section, self.name)[new_key]

        # 如果以上条件都不适用，抛出索引错误
        raise IndexingError("not applicable")

    @final
    ```python`
        def _getitem_nested_tuple(self, tup: tuple):
            # 定义一个方法，处理包含嵌套元组的索引，tup 为输入的元组
            # we have a nested tuple so have at least 1 multi-index level
            # we should be able to match up the dimensionality here
    
            def _contains_slice(x: object) -> bool:
                # 检查对象是否为 slice 或包含 slice 的元组
                # Check if object is a slice or a tuple containing a slice
                if isinstance(x, tuple):
                    return any(isinstance(v, slice) for v in x)
                elif isinstance(x, slice):
                    return True
                return False
    
            # 对元组中的每个键调用 check_dict_or_set_indexers 函数
            for key in tup:
                check_dict_or_set_indexers(key)
    
            # 如果元组中索引的数量超过数据维度，但有至少一个多维索引，尝试处理如元组传递给具有多索引的序列的情况
            # we have too many indexers for our dim, but have at least 1
            # multi-index dimension, try to see if we have something like
            # a tuple passed to a series with a multi-index
            if len(tup) > self.ndim:
                if self.name != "loc":
                    # 如果数据的名称不是 "loc"，抛出 ValueError 异常
                    # This should never be reached, but let's be explicit about it
                    raise ValueError("Too many indices")  # pragma: no cover
                if all(
                    (is_hashable(x) and not _contains_slice(x)) or com.is_null_slice(x)
                    for x in tup
                ):
                    # 如果所有元素都是可哈希的且不包含 slice，或是空 slice，尝试降低多维索引的维度
                    # GH#10521 Series should reduce MultiIndex dimensions instead of
                    #  DataFrame, IndexingError is not raised when slice(None,None,None)
                    #  with one row.
                    with suppress(IndexingError):
                        return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(
                            tup
                        )
                elif isinstance(self.obj, ABCSeries) and any(
                    isinstance(k, tuple) for k in tup
                ):
                    # 如果是 Series 对象并且元组中包含元组，抛出 IndexingError 异常
                    # GH#35349 Raise if tuple in tuple for series
                    # Do this after the all-hashable-or-null-slice check so that
                    #  we are only getting non-hashable tuples, in particular ones
                    #  that themselves contain a slice entry
                    # See test_loc_series_getitem_too_many_dimensions
                    raise IndexingError("Too many indexers")
    
                # 如果是具有多重索引的 Series，使用传递的元组选择器处理多维索引
                # this is a series with a multi-index specified a tuple of
                # selectors
                axis = self.axis or 0
                return self._getitem_axis(tup, axis=axis)
    
            # 处理多轴情况，进行切片并减少维度，这是一个迭代过程
            obj = self.obj
            # GH#41369 反向循环确保沿列索引后再沿行索引，这只选择必要的块，避免数据类型转换
            # Loop in reverse order ensures indexing along columns before rows
            # which selects only necessary blocks which avoids dtype conversion if possible
            axis = len(tup) - 1
            for key in reversed(tup):
                if com.is_null_slice(key):
                    axis -= 1
                    continue
    
                # 根据当前键和轴索引，获取对象的切片
                obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
                axis -= 1
    
                # 如果对象是标量，或没有属性 "ndim"，则结束循环
                # if we have a scalar, we are done
                if is_scalar(obj) or not hasattr(obj, "ndim"):
                    break
    
            return obj
    
        def _convert_to_indexer(self, key, axis: AxisInt):
            # 抛出抽象方法错误，要求子类实现此方法
            raise AbstractMethodError(self)
    def _raise_callable_usage(self, key: Any, maybe_callable: T) -> T:
        # 如果操作名称为 "iloc"，且 key 是可调用的且 maybe_callable 是元组类型，则抛出值错误异常
        if self.name == "iloc" and callable(key) and isinstance(maybe_callable, tuple):
            raise ValueError(
                "Returning a tuple from a callable with iloc is not allowed.",
            )
        # 返回 maybe_callable，可能会被改变或者保持不变
        return maybe_callable

    @final
    def __getitem__(self, key):
        # 检查索引键是否是字典或集合
        check_dict_or_set_indexers(key)
        
        # 如果键的类型是元组
        if type(key) is tuple:
            # 如果元组中的每个元素是迭代器，则将其转换为列表
            key = (list(x) if is_iterator(x) else x for x in key)
            # 对元组中的每个元素应用可能的可调用函数，并重新构建元组
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
            # 如果是标量访问，则调用对象的 _get_value 方法获取值
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            # 否则，调用 _getitem_tuple 方法处理元组键
            return self._getitem_tuple(key)
        else:
            # 根据定义，我们只有第0轴
            axis = self.axis or 0

            # 对键应用可能的可调用函数
            maybe_callable = com.apply_if_callable(key, self.obj)
            # 调用 _raise_callable_usage 方法处理可能的可调用情况
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            # 调用 _getitem_axis 方法处理键值获取操作
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key: tuple):
        # 抛出未实现错误，子类需要实现该方法
        raise NotImplementedError

    def _getitem_tuple(self, tup: tuple):
        # 抛出抽象方法错误，子类需要实现该方法
        raise AbstractMethodError(self)

    def _getitem_axis(self, key, axis: AxisInt):
        # 抛出未实现错误，子类需要实现该方法
        raise NotImplementedError

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        # 抛出抽象方法错误，子类需要实现该方法
        raise AbstractMethodError(self)

    @final
    def _getbool_axis(self, key, axis: AxisInt):
        # 调用者负责确保轴不为 None
        # 获取对象在给定轴上的标签
        labels = self.obj._get_axis(axis)
        # 检查布尔索引器的有效性，并返回调用对象的 take 方法结果
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj.take(inds, axis=axis)
# 根据索引混合类 IndexingMixin.loc 的文档说明，为 _LocIndexer 类添加文档注释
@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    # _takeable 属性用于指示是否可以采取操作
    _takeable: bool = False
    # _valid_types 定义了有效的索引类型，支持标签、标签切片（包含两端点）、整数切片（仅当索引为整数时）、标签列表和布尔类型
    _valid_types = (
        "labels (MUST BE IN THE INDEX), slices of labels (BOTH "
        "endpoints included! Can be slices of integers if the "
        "index is integers), listlike of labels, boolean"
    )

    # -------------------------------------------------------------------
    # Key Checks

    # 根据 _LocationIndexer._validate_key 的文档说明，验证给定的键 key 是否有效
    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key, axis: Axis) -> None:
        # 获取指定轴的对象
        ax = self.obj._get_axis(axis)
        # 如果 key 是布尔型并且轴的数据类型不是布尔型，或者轴不是布尔型且不支持布尔索引（对于 MultiIndex 的第一级）
        # 则抛出 KeyError 异常
        if isinstance(key, bool) and not (
            is_bool_dtype(ax.dtype)
            or ax.dtype.name == "boolean"
            or isinstance(ax, MultiIndex)
            and is_bool_dtype(ax.get_level_values(0).dtype)
        ):
            raise KeyError(
                f"{key}: boolean label can not be used without a boolean index"
            )

        # 如果 key 是切片并且起始点或结束点是布尔值，则抛出 TypeError 异常
        if isinstance(key, slice) and (
            isinstance(key.start, bool) or isinstance(key.stop, bool)
        ):
            raise TypeError(f"{key}: boolean values can not be used in a slice")

    # 判断是否具有有效的 setitem 索引器，始终返回 True
    def _has_valid_setitem_indexer(self, indexer) -> bool:
        return True

    # 判断是否是标量访问，通过检查键 key 的长度和每个元素是否为标量来判断
    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
        # 如果键 key 的长度不等于对象的维度，则返回 False
        if len(key) != self.ndim:
            return False

        # 遍历键 key 的每个元素 k 和对应的轴 ax
        for i, k in enumerate(key):
            # 如果 k 不是标量，则返回 False
            if not is_scalar(k):
                return False

            # 获取对象的第 i 个轴
            ax = self.obj.axes[i]
            # 如果该轴是 MultiIndex 类型，则返回 False
            if isinstance(ax, MultiIndex):
                return False

            # 如果 k 是字符串并且该轴支持部分字符串索引，则返回 False
            if isinstance(k, str) and ax._supports_partial_string_indexing:
                return False

            # 如果该轴不是以唯一索引的形式存在，则返回 False
            if not ax._index_as_unique:
                return False

        # 如果所有条件都通过，则返回 True
        return True

    # -------------------------------------------------------------------
    # MultiIndex Handling
    def _multi_take_opportunity(self, tup: tuple) -> bool:
        """
        检查是否有可能使用 ``_multi_take`` 方法。

        当前的限制是所有被索引的轴必须使用类似列表的索引。

        Parameters
        ----------
        tup : tuple
            每个轴对应的索引器的元组。

        Returns
        -------
        bool
            当前的索引操作是否可以通过 `_multi_take` 方法处理。
        """
        # 检查是否所有的索引器都是类似列表的索引
        if not all(is_list_like_indexer(x) for x in tup):
            return False

        # 如果存在布尔型索引器，返回 False
        return not any(com.is_bool_indexer(x) for x in tup)

    def _multi_take(self, tup: tuple):
        """
        为传入的键的元组创建索引器，并执行 take 操作。
        这允许一次性执行 take 操作，而不是每个维度都执行一次，从而提高效率。

        Parameters
        ----------
        tup : tuple
            每个轴对应的索引器的元组。

        Returns
        -------
        values: 与被索引对象相同类型的对象
        """
        # GH 836
        # 创建一个字典，将每个轴的索引器与对应的键关联起来
        d = {
            axis: self._get_listlike_indexer(key, axis)
            for (key, axis) in zip(tup, self.obj._AXIS_ORDERS)
        }
        # 调用对象的 `_reindex_with_indexers` 方法重新索引，并允许重复索引
        return self.obj._reindex_with_indexers(d, allow_dups=True)

    # -------------------------------------------------------------------

    def _getitem_iterable(self, key, axis: AxisInt):
        """
        使用可迭代的键集合对当前对象进行索引。

        Parameters
        ----------
        key : iterable
            目标标签。
        axis : int
            进行索引的维度。

        Raises
        ------
        KeyError
            如果未找到任何键。未来将更改为仅在未找到所有键时引发错误。

        Returns
        -------
        scalar, DataFrame, or Series：索引后的值。
        """
        # 假设不是布尔型索引器，因为在此之前已经处理过。
        self._validate_key(key, axis)

        # 获取类似列表的索引器和索引数组
        keyarr, indexer = self._get_listlike_indexer(key, axis)
        # 调用对象的 `_reindex_with_indexers` 方法重新索引，并允许重复索引
        return self.obj._reindex_with_indexers(
            {axis: [keyarr, indexer]}, allow_dups=True
        )

    def _getitem_tuple(self, tup: tuple):
        """
        根据元组进行索引操作。

        Parameters
        ----------
        tup : tuple
            索引器的元组。

        Returns
        -------
        scalar, DataFrame, or Series：索引后的值。
        """
        with suppress(IndexingError):
            # 展开省略号（如果有），然后进行下一级别的索引操作
            tup = self._expand_ellipsis(tup)
            return self._getitem_lowerdim(tup)

        # 如果没有多级索引，验证所有索引器
        tup = self._validate_tuple_indexer(tup)

        # 对于 GH #836 的丑陋处理
        if self._multi_take_opportunity(tup):
            # 如果有多重索引的机会，执行 `_multi_take` 操作
            return self._multi_take(tup)

        # 否则，按相同维度的元组进行索引
        return self._getitem_tuple_same_dim(tup)

    def _get_label(self, label, axis: AxisInt):
        """
        获取指定标签在指定轴上的对象。

        Parameters
        ----------
        label : object
            目标标签。
        axis : int
            进行索引的维度。

        Returns
        -------
        与对象相同类型的标量、DataFrame 或 Series：索引后的值。
        """
        # 如果标签不在轴上，则会失败（GH #5567）
        return self.obj.xs(label, axis=axis)
    def _handle_lowerdim_multi_index_axis0(self, tup: tuple):
        # 处理具有 axis0 多级索引的情况，或者引发异常
        axis = self.axis or 0  # 获取当前对象的轴向或默认为0
        try:
            # 对于 Series 或者不包含切片的 tup，采用快速路径
            return self._get_label(tup, axis=axis)

        except KeyError as ek:
            # 如果索引器数量匹配，则引发 KeyError
            # 否则将引发 IndexingError
            if self.ndim < len(tup) <= self.obj.index.nlevels:
                raise ek
            raise IndexingError("No label returned") from ek

    def _getitem_axis(self, key, axis: AxisInt):
        key = item_from_zerodim(key)  # 将零维 key 转换为合适的值
        if is_iterator(key):
            key = list(key)  # 如果 key 是迭代器，则转换为列表
        if key is Ellipsis:
            key = slice(None)  # 如果 key 是省略号，则转换为全部切片

        labels = self.obj._get_axis(axis)  # 获取指定轴向的标签

        if isinstance(key, tuple) and isinstance(labels, MultiIndex):
            key = tuple(key)  # 如果 key 是元组且标签是 MultiIndex，则保持为元组

        if isinstance(key, slice):
            self._validate_key(key, axis)  # 验证切片 key 的有效性
            return self._get_slice_axis(key, axis=axis)  # 获取切片操作的结果
        elif com.is_bool_indexer(key):
            return self._getbool_axis(key, axis=axis)  # 处理布尔索引
        elif is_list_like_indexer(key):
            # 可迭代的多选操作
            if not (isinstance(key, tuple) and isinstance(labels, MultiIndex)):
                if hasattr(key, "ndim") and key.ndim > 1:
                    raise ValueError("Cannot index with multidimensional key")
                return self._getitem_iterable(key, axis=axis)  # 处理可迭代对象的索引

            # 嵌套元组切片
            if is_nested_tuple(key, labels):
                locs = labels.get_locs(key)
                indexer: list[slice | npt.NDArray[np.intp]] = [slice(None)] * self.ndim
                indexer[axis] = locs
                return self.obj.iloc[tuple(indexer)]

        # 直接查找
        self._validate_key(key, axis)  # 验证 key 的有效性
        return self._get_label(key, axis=axis)  # 获取标签的数据

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        """
        This is pretty simple as we just have to deal with labels.
        """
        # 调用者需保证 axis 不为 None
        obj = self.obj
        if not need_slice(slice_obj):
            return obj.copy(deep=False)  # 如果不需要切片，则返回对象的浅拷贝

        labels = obj._get_axis(axis)  # 获取指定轴向的标签
        indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step)  # 获取切片的索引器

        if isinstance(indexer, slice):
            return self.obj._slice(indexer, axis=axis)  # 对象支持切片操作
        else:
            # DatetimeIndex 覆盖了 Index.slice_indexer，可能返回 DatetimeIndex 而不是切片对象
            return self.obj.take(indexer, axis=axis)  # 使用索引器获取数据
    # 将索引键转换为可以在 ndarray 上执行花式索引的格式
    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Convert indexing key into something we can use to do actual fancy
        indexing on a ndarray.

        Examples
        ix[:5] -> slice(0, 5)
        ix[[1,2,3]] -> [1,2,3]
        ix[['foo', 'bar', 'baz']] -> [i, j, k] (indices of foo, bar, baz)

        Going by Zen of Python?
        'In the face of ambiguity, refuse the temptation to guess.'
        raise AmbiguousIndexError with integer labels?
        - No, prefer label-based indexing
        """
        # 获取指定轴的标签对象
        labels = self.obj._get_axis(axis)

        # 如果索引键是 slice 类型，则转换为 loc 索引器
        if isinstance(key, slice):
            return labels._convert_slice_indexer(key, kind="loc")

        # 如果索引键是元组，并且标签不是 MultiIndex 且维度小于 2 并且键的长度大于 1，则抛出索引错误
        if (
            isinstance(key, tuple)
            and not isinstance(labels, MultiIndex)
            and self.ndim < 2
            and len(key) > 1
        ):
            raise IndexingError("Too many indexers")

        # 检查键中是否包含 slice 类型的项
        contains_slice = False
        if isinstance(key, tuple):
            contains_slice = any(isinstance(v, slice) for v in key)

        # 如果键是标量或者是 MultiIndex 且可散列且不包含 slice，则返回其位置索引
        if is_scalar(key) or (
            isinstance(labels, MultiIndex) and is_hashable(key) and not contains_slice
        ):
            # 否则 get_loc 将引发 InvalidIndexError

            # 如果是标签，则返回其位置
            try:
                return labels.get_loc(key)
            except LookupError:
                if isinstance(key, tuple) and isinstance(labels, MultiIndex):
                    if len(key) == labels.nlevels:
                        return {"key": key}
                    raise
            except InvalidIndexError:
                # GH35015, 使用 datetime 作为列索引会引发异常
                if not isinstance(labels, MultiIndex):
                    raise
            except ValueError:
                if not is_integer(key):
                    raise
                return {"key": key}

        # 如果键是嵌套元组，则返回其位置索引
        if is_nested_tuple(key, labels):
            if self.ndim == 1 and any(isinstance(k, tuple) for k in key):
                # 如果系列中存在元组中的元组，则引发索引错误
                raise IndexingError("Too many indexers")
            return labels.get_locs(key)

        # 如果键是类似列表的索引器，则根据情况处理
        elif is_list_like_indexer(key):
            if is_iterator(key):
                key = list(key)

            # 如果是布尔索引器，则验证并返回其处理结果
            if com.is_bool_indexer(key):
                key = check_bool_indexer(labels, key)
                return key
            else:
                # 否则，返回列表类索引器的处理结果的第二个元素
                return self._get_listlike_indexer(key, axis)[1]
        else:
            # 否则，尝试获取键的位置索引
            try:
                return labels.get_loc(key)
            except LookupError:
                # 如果是设置操作，并且未找到键，则返回一个包含键的字典
                if not is_list_like_indexer(key):
                    return {"key": key}
                raise
    def _get_listlike_indexer(self, key, axis: AxisInt):
        """
        Transform a list-like of keys into a new index and an indexer.

        Parameters
        ----------
        key : list-like
            Targeted labels.
        axis:  int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.

        Returns
        -------
        keyarr: Index
            New index (coinciding with 'key' if the axis is unique).
        values : array-like
            Indexer for the return object, -1 denotes keys not found.
        """
        # 获取指定轴上的索引器和新的索引数组
        ax = self.obj._get_axis(axis)
        # 获取轴的名称
        axis_name = self.obj._get_axis_name(axis)

        # 严格获取索引器和索引数组
        keyarr, indexer = ax._get_indexer_strict(key, axis_name)

        # 返回新的索引数组和索引器
        return keyarr, indexer
@doc(IndexingMixin.iloc)
class _iLocIndexer(_LocationIndexer):
    _valid_types = (
        "integer, integer slice (START point is INCLUDED, END "
        "point is EXCLUDED), listlike of integers, boolean array"
    )
    _takeable = True

    # -------------------------------------------------------------------
    # Key Checks

    # 定义方法用于验证索引键的有效性
    def _validate_key(self, key, axis: AxisInt) -> None:
        # 如果键是布尔索引器
        if com.is_bool_indexer(key):
            # 如果键具有索引且索引是索引对象
            if hasattr(key, "index") and isinstance(key.index, Index):
                # 如果索引的推断类型为整数，则抛出未实现错误
                if key.index.inferred_type == "integer":
                    raise NotImplementedError(
                        "iLocation based boolean "
                        "indexing on an integer type "
                        "is not available"
                    )
                # 否则，抛出值错误，说明不能使用索引作为掩码进行布尔索引
                raise ValueError(
                    "iLocation based boolean indexing cannot use "
                    "an indexable as a mask"
                )
            return

        # 如果键是切片对象，则直接返回
        if isinstance(key, slice):
            return
        # 如果键是整数类型
        elif is_integer(key):
            # 调用私有方法验证整数索引的有效性
            self._validate_integer(key, axis)
        # 如果键是元组类型
        elif isinstance(key, tuple):
            # 元组类型已经在此之前被捕获，因此不应视为有效的索引器，抛出索引错误
            raise IndexingError("Too many indexers")
        # 如果键是类列表索引器
        elif is_list_like_indexer(key):
            # 如果键是 Pandas 的序列对象
            if isinstance(key, ABCSeries):
                arr = key._values
            # 如果键是类数组对象
            elif is_array_like(key):
                arr = key
            # 否则，将键转换为 NumPy 数组
            else:
                arr = np.array(key)
            # 获取轴的长度
            len_axis = len(self.obj._get_axis(axis))

            # 检查键的数据类型是否为数值型
            if not is_numeric_dtype(arr.dtype):
                raise IndexError(f".iloc requires numeric indexers, got {arr}")

            # 检查键的值是否超出索引的最大范围
            if len(arr) and (arr.max() >= len_axis or arr.min() < -len_axis):
                raise IndexError("positional indexers are out-of-bounds")
        else:
            # 如果键类型不符合预期，则抛出值错误，说明只能使用特定类型的索引器进行定位索引
            raise ValueError(f"Can only index by location with a [{self._valid_types}]")
    def _has_valid_setitem_indexer(self, indexer) -> bool:
        """
        Validate that a positional indexer cannot enlarge its target
        will raise if needed, does not modify the indexer externally.

        Parameters
        ----------
        indexer : object
            The indexer to validate.

        Returns
        -------
        bool
            True if the indexer is valid.

        Raises
        ------
        IndexError
            If the indexer is a dictionary or contains elements that would enlarge the target object.
        TypeError
            If the indexer is an instance of ABCDataFrame.

        Notes
        -----
        This method ensures that the indexer is suitable for setting items in the object.
        """
        if isinstance(indexer, dict):
            # Raise an error if the indexer is a dictionary
            raise IndexError("iloc cannot enlarge its target object")

        if isinstance(indexer, ABCDataFrame):
            # Raise a type error if the indexer is an instance of ABCDataFrame
            raise TypeError(
                "DataFrame indexer for .iloc is not supported. "
                "Consider using .loc with a DataFrame indexer for automatic alignment.",
            )

        if not isinstance(indexer, tuple):
            # Convert indexer to a tuple if it's not already one
            indexer = _tuplify(self.ndim, indexer)

        # Validate each element of the indexer
        for ax, i in zip(self.obj.axes, indexer):
            if isinstance(i, slice):
                # For slice objects, no further validation is needed
                pass
            elif is_list_like_indexer(i):
                # For list-like indexers, further checks could be implemented
                pass
            elif is_integer(i):
                # Validate integer indexers against axis length
                if i >= len(ax):
                    raise IndexError("iloc cannot enlarge its target object")
            elif isinstance(i, dict):
                # Raise an error if any indexer element is a dictionary
                raise IndexError("iloc cannot enlarge its target object")

        return True

    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Check if the given key represents scalar access.

        Parameters
        ----------
        key : tuple
            Key to check for scalar access.

        Returns
        -------
        bool
            True if all elements of the key are integers, indicating scalar access.

        Notes
        -----
        This method checks if the key can be used for accessing single scalar elements.
        """
        # Check if the length of the key matches the object's number of dimensions
        if len(key) != self.ndim:
            return False

        # Return True if all elements of the key are integers
        return all(is_integer(k) for k in key)

    def _validate_integer(self, key: int | np.integer, axis: AxisInt) -> None:
        """
        Validate that 'key' is a valid position in the desired axis.

        Parameters
        ----------
        key : int | np.integer
            Requested position.
        axis : AxisInt
            Desired axis.

        Raises
        ------
        IndexError
            If 'key' is out of bounds for the specified axis.

        Notes
        -----
        This method checks if the given key is within valid bounds for indexing.
        """
        len_axis = len(self.obj._get_axis(axis))
        if key >= len_axis or key < -len_axis:
            raise IndexError("single positional indexer is out-of-bounds")

    # -------------------------------------------------------------------

    def _getitem_tuple(self, tup: tuple):
        """
        Retrieve item(s) from the object using a tuple indexer.

        Parameters
        ----------
        tup : tuple
            Tuple indexer.

        Returns
        -------
        object
            Retrieved item(s) from the object.

        Notes
        -----
        This method handles tuple indexers and attempts to retrieve items
        using different strategies depending on the dimensions and type of the object.
        """
        # Validate the tuple indexer
        tup = self._validate_tuple_indexer(tup)
        
        # Attempt to retrieve items using a lower-dimensional getter method
        with suppress(IndexingError):
            return self._getitem_lowerdim(tup)

        # If lower-dimensional access fails, fallback to same-dimensional retrieval
        return self._getitem_tuple_same_dim(tup)
    def _get_list_axis(self, key, axis: AxisInt):
        """
        Return Series values by list or array of integers.

        Parameters
        ----------
        key : list-like positional indexer
            List or array of integers representing positions to select from the Series.
        axis : int
            Axis along which to take elements. Must be 0.

        Returns
        -------
        Series object
            Series object containing the selected elements.

        Notes
        -----
        `axis` can only be zero.
        """
        try:
            # Use the `take` method of `self.obj` to select elements at positions `key` along `axis`.
            return self.obj.take(key, axis=axis)
        except IndexError as err:
            # If an IndexError occurs, raise it with a specific message.
            raise IndexError("positional indexers are out-of-bounds") from err

    def _getitem_axis(self, key, axis: AxisInt):
        """
        Retrieve elements from the object along the specified axis using `key`.

        Parameters
        ----------
        key : indexer
            Indexer used to select elements.
        axis : int
            Axis along which to select elements.

        Returns
        -------
        Series or DataFrame object
            Object containing the selected elements.

        Raises
        ------
        IndexError
            If `key` is a DataFrame indexer or an invalid type.

        Notes
        -----
        Raises an IndexError if `key` is a DataFrame indexer. Handles various types of `key`
        such as slices, iterators, lists, and single integers.
        """
        if key is Ellipsis:
            key = slice(None)
        elif isinstance(key, ABCDataFrame):
            # Raise an IndexError if `key` is a DataFrame indexer.
            raise IndexError(
                "DataFrame indexer is not allowed for .iloc\n"
                "Consider using .loc for automatic alignment."
            )

        if isinstance(key, slice):
            # If `key` is a slice, retrieve elements using `_get_slice_axis`.
            return self._get_slice_axis(key, axis=axis)

        if is_iterator(key):
            # Convert `key` from an iterator to a list.
            key = list(key)

        if isinstance(key, list):
            # Convert `key` from a list to a NumPy array.
            key = np.asarray(key)

        if com.is_bool_indexer(key):
            # Validate and retrieve elements using boolean indexing.
            self._validate_key(key, axis)
            return self._getbool_axis(key, axis=axis)

        # Handle a list of integers
        elif is_list_like_indexer(key):
            # Retrieve elements using `_get_list_axis` for a list-like indexer.
            return self._get_list_axis(key, axis=axis)

        # Handle a single integer
        else:
            key = item_from_zerodim(key)
            if not is_integer(key):
                # Raise a TypeError if `key` is not an integer.
                raise TypeError("Cannot index by location index with a non-integer key")

            # Validate the integer key and retrieve the corresponding element.
            self._validate_integer(key, axis)
            return self.obj._ixs(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        """
        Retrieve elements from the object along the specified axis using a slice.

        Parameters
        ----------
        slice_obj : slice
            Slice object specifying the range of elements to select.
        axis : int
            Axis along which to select elements.

        Returns
        -------
        Series or DataFrame object
            Object containing the selected elements.

        Notes
        -----
        Validates the positional slice `slice_obj` against the axis labels before retrieving elements.
        """
        # The caller is responsible for ensuring `axis` is non-None.
        obj = self.obj

        if not need_slice(slice_obj):
            # If `slice_obj` does not need slicing, return a shallow copy of `obj`.
            return obj.copy(deep=False)

        labels = obj._get_axis(axis)
        # Validate the positional slice `slice_obj` against the axis labels.
        labels._validate_positional_slice(slice_obj)
        # Retrieve elements using `_slice` method of `self.obj`.
        return self.obj._slice(slice_obj, axis=axis)

    def _convert_to_indexer(self, key: T, axis: AxisInt) -> T:
        """
        Convert the key to an indexer format.

        Parameters
        ----------
        key : T
            Key to be converted.
        axis : int
            Axis along which to perform the conversion.

        Returns
        -------
        T
            Converted key.

        Notes
        -----
        This method performs a straightforward conversion as all types are valid indexers.
        """
        return key

    def _get_setitem_indexer(self, key):
        """
        Convert `key` to an indexer for setting items.

        Parameters
        ----------
        key
            Key to be converted.

        Returns
        -------
        indexer
            Converted indexer for setting items.

        Notes
        -----
        If `key` is an iterator, it is converted to a list. Handles special cases based on `self.axis`.
        """
        # Fall through to let NumPy handle validation if `key` is an iterator.
        if is_iterator(key):
            key = list(key)

        if self.axis is not None:
            # Tupleize `key` if `self.axis` is not None.
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)

        return key

    # -------------------------------------------------------------------
    # 处理二维值的索引器设置操作，对应情况 np.ndim(value) == 2，DataFrame 除外，后者通过 _setitem_with_indexer_frame_value 处理
    pi = indexer[0]  # 取得索引器中的第一个元素作为主索引位置

    ilocs = self._ensure_iterable_column_indexer(indexer[1])  # 确保列索引器是可迭代的

    if not is_array_like(value):
        # 如果 value 不是类数组，将其转换为 ndarray
        value = np.array(value, dtype=object)
    if len(ilocs) != value.shape[1]:
        # 当 ndarray 长度与 ilocs 不相等时，抛出 ValueError 异常
        raise ValueError(
            "Must have equal len keys and value when setting with an ndarray"
        )

    for i, loc in enumerate(ilocs):
        # 遍历 ilocs，对每个位置 loc 进行设置
        value_col = value[:, i]  # 取得 ndarray 中的第 i 列数据
        if is_object_dtype(value_col.dtype):
            # 如果数据类型是对象类型，将其转换为列表，以便在 _setitem_single_column 中进行类型推断
            value_col = value_col.tolist()
        self._setitem_single_column(loc, value_col, pi)  # 调用 _setitem_single_column 进行单列设置操作

def _setitem_with_indexer_frame_value(
    self, indexer, value: DataFrame, name: str
) -> None:
    ilocs = self._ensure_iterable_column_indexer(indexer[1])  # 确保列索引器是可迭代的

    sub_indexer = list(indexer)  # 将 indexer 转换为列表
    pi = indexer[0]  # 取得索引器中的第一个元素作为主索引位置

    multiindex_indexer = isinstance(self.obj.columns, MultiIndex)  # 判断 self.obj.columns 是否为 MultiIndex 对象

    unique_cols = value.columns.is_unique  # 判断 value 的列是否唯一

    # 对于 "iloc" 情况，不进行值的对齐操作，直接设置
    if name == "iloc":
        for i, loc in enumerate(ilocs):
            val = value.iloc[:, i]  # 取得 DataFrame value 的第 i 列数据
            self._setitem_single_column(loc, val, pi)  # 调用 _setitem_single_column 进行单列设置操作

    # 如果 value 的列不唯一且与 self.obj.columns 相等，则假定已经对齐
    elif not unique_cols and value.columns.equals(self.obj.columns):
        for loc in ilocs:
            item = self.obj.columns[loc]  # 取得 self.obj 的列名
            if item in value:
                sub_indexer[1] = item
                val = self._align_series(
                    tuple(sub_indexer),
                    value.iloc[:, loc],
                    multiindex_indexer,
                )
            else:
                val = np.nan

            self._setitem_single_column(loc, val, pi)  # 调用 _setitem_single_column 进行单列设置操作

    # 如果 value 的列不唯一但不与 self.obj.columns 相等，则抛出 ValueError 异常
    elif not unique_cols:
        raise ValueError("Setting with non-unique columns is not allowed.")

    # 否则，处理每个 loc 的值对齐操作
    else:
        for loc in ilocs:
            item = self.obj.columns[loc]  # 取得 self.obj 的列名
            if item in value:
                sub_indexer[1] = item
                val = self._align_series(
                    tuple(sub_indexer),
                    value[item],
                    multiindex_indexer,
                    using_cow=True,
                )
            else:
                val = np.nan

            self._setitem_single_column(loc, val, pi)  # 调用 _setitem_single_column 进行单列设置操作
    def _setitem_single_block(self, indexer, value, name: str) -> None:
        """
        _setitem_with_indexer for the case when we have a single Block.
        """
        from pandas import Series  # 导入 pandas 的 Series 类

        # 如果值是 ABCSeries 类型且不是通过 "iloc" 设置，或者值是字典类型
        if (isinstance(value, ABCSeries) and name != "iloc") or isinstance(value, dict):
            # TODO(EA): ExtensionBlock.setitem this causes issues with
            # setting for extensionarrays that store dicts. Need to decide
            # if it's worth supporting that.
            # 对于存储字典的扩展数组，这会导致设置问题。需要决定是否值得支持。
            value = self._align_series(indexer, Series(value))  # 调用 _align_series 方法对 Series 进行对齐处理

        info_axis = self.obj._info_axis_number  # 获取对象的信息轴编号
        item_labels = self.obj._get_axis(info_axis)  # 获取信息轴上的标签
        if isinstance(indexer, tuple):
            # 如果我们只在信息轴上设置
            # 使用这些方法进行设置，以避免在此处拆分块逻辑
            if (
                self.ndim == len(indexer) == 2  # 如果维度为 2
                and is_integer(indexer[1])  # 并且第二个索引是整数
                and com.is_null_slice(indexer[0])  # 并且第一个索引是空切片
            ):
                col = item_labels[indexer[info_axis]]  # 获取列名
                if len(item_labels.get_indexer_for([col])) == 1:
                    # 例如，test_loc_setitem_empty_append_expands_rows
                    loc = item_labels.get_loc(col)  # 获取列的位置
                    self._setitem_single_column(loc, value, indexer[0])  # 调用 _setitem_single_column 方法设置单列
                    return

            indexer = maybe_convert_ix(*indexer)  # 例如，test_setitem_frame_align

        if isinstance(value, ABCDataFrame) and name != "iloc":
            # 如果值是 ABCDataFrame 类型且不是通过 "iloc" 设置
            value = self._align_frame(indexer, value)._values  # 调用 _align_frame 方法对 DataFrame 进行对齐处理并获取值

        # 实际进行设置操作
        self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)  # 调用对象的 _mgr 属性的 setitem 方法进行设置

    def _ensure_iterable_column_indexer(self, column_indexer):
        """
        Ensure that our column indexer is something that can be iterated over.
        """
        ilocs: Sequence[int | np.integer] | np.ndarray
        if is_integer(column_indexer):
            ilocs = [column_indexer]  # 如果 column_indexer 是整数，转换为列表
        elif isinstance(column_indexer, slice):
            ilocs = np.arange(len(self.obj.columns))[column_indexer]  # 如果是切片，获取列的索引
        elif (
            isinstance(column_indexer, np.ndarray) and column_indexer.dtype.kind == "b"
        ):
            ilocs = np.arange(len(column_indexer))[column_indexer]  # 如果是布尔型 ndarray，获取列的索引
        else:
            ilocs = column_indexer  # 否则直接使用 column_indexer

        return ilocs

    def _align_series(
        self,
        indexer,
        ser: Series,
        multiindex_indexer: bool = False,
        using_cow: bool = False,
    ):
        """
        Align Series to current DataFrame index.
        """
        # 对 Series 进行对齐处理到当前 DataFrame 的索引
        pass  # 此处为示例注释，实际应根据方法实现添加注释
    # 定义一个方法 `_align_frame`，用于对齐索引并返回对齐后的 DataFrame
    def _align_frame(self, indexer, df: DataFrame) -> DataFrame:
        # 判断当前对象是否为 DataFrame
        is_frame = self.ndim == 2

        # 如果 indexer 是 tuple 类型，处理多索引对齐情况
        if isinstance(indexer, tuple):
            # 初始化 idx 和 cols 为 None
            idx, cols = None, None
            # 存储单索引的位置列表
            sindexers = []
            # 遍历 indexer 中的每个索引
            for i, ix in enumerate(indexer):
                # 获取当前轴对象
                ax = self.obj.axes[i]
                # 如果 ix 是序列或切片类型
                if is_sequence(ix) or isinstance(ix, slice):
                    # 如果 ix 是 numpy 数组，将其展平
                    if isinstance(ix, np.ndarray):
                        ix = ix.reshape(-1)
                    # 如果 idx 为空，根据 ix 选择轴上的数据
                    if idx is None:
                        idx = ax[ix]
                    # 如果 cols 为空，根据 ix 选择轴上的数据
                    elif cols is None:
                        cols = ax[ix]
                    else:
                        break
                else:
                    # 如果 ix 不是序列或切片类型，记录其位置
                    sindexers.append(i)

            # 如果 idx 和 cols 都不为空
            if idx is not None and cols is not None:
                # 如果传入的 DataFrame 的索引和列与 idx 和 cols 对应，则直接复制 DataFrame
                if df.index.equals(idx) and df.columns.equals(cols):
                    val = df.copy()
                else:
                    # 否则，重新索引 DataFrame
                    val = df.reindex(idx, columns=cols)
                return val

        # 如果 indexer 是 slice 或 list-like 索引器，并且当前对象是 DataFrame
        elif (isinstance(indexer, slice) or is_list_like_indexer(indexer)) and is_frame:
            # 根据 indexer 选择当前对象的索引
            ax = self.obj.index[indexer]
            # 如果传入的 DataFrame 的索引与 ax 相同，则直接复制 DataFrame
            if df.index.equals(ax):
                val = df.copy()
            else:
                # 否则，根据 ax 重新索引 DataFrame
                val = df.reindex(index=ax)
            return val

        # 如果以上条件均不满足，抛出值错误异常
        raise ValueError("Incompatible indexer with DataFrame")
class _ScalarAccessIndexer(NDFrameIndexerBase):
    """
    Access scalars quickly.
    """

    # sub-classes need to set _takeable
    _takeable: bool

    def _convert_key(self, key):
        # 抽象方法错误，子类需要实现该方法
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            # 可能有可转换的项目（例如 Timestamp）
            if not is_list_like_indexer(key):
                key = (key,)
            else:
                # 获取标量访问的无效调用
                raise ValueError("Invalid call for scalar access (getting)!")

        key = self._convert_key(key)
        # 使用 _get_value 方法获取数据
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            # 对元组中的每个元素应用可调用函数
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            # 标量可调用函数可能返回元组
            key = com.apply_if_callable(key, self.obj)

        if not isinstance(key, tuple):
            # 将标量键转换为元组
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key))
        if len(key) != self.ndim:
            # 对于标量访问，索引器不足
            raise ValueError("Not enough indexers for scalar access (setting)!")

        # 使用 _set_value 方法设置值
        self.obj._set_value(*key, value=value, takeable=self._takeable)


@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    _takeable = False

    def _convert_key(self, key):
        """
        Require they keys to be the same type as the index. (so we don't
        fallback)
        """
        # 对于系列，需要键的类型与索引相同，避免回退
        # GH 26989
        # 对于系列，如果 len(key) > 1，则需要展开键以得到标签
        if self.ndim == 1 and len(key) > 1:
            key = (key,)

        return key

    @property
    def _axes_are_unique(self) -> bool:
        # 仅适用于 self.ndim == 2
        assert self.ndim == 2
        # 索引和列必须唯一
        return self.obj.index.is_unique and self.obj.columns.is_unique

    def __getitem__(self, key):
        if self.ndim == 2 and not self._axes_are_unique:
            # GH#33041 回退到 .loc
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                # 无效的标量访问调用（获取）
                raise ValueError("Invalid call for scalar access (getting)!")
            # 返回通过 .loc 获取的对象
            return self.obj.loc[key]

        # 调用父类的 __getitem__ 方法
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        if self.ndim == 2 and not self._axes_are_unique:
            # GH#33041 回退到 .loc
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                # 无效的标量访问调用（设置）
                raise ValueError("Invalid call for scalar access (setting)!")

            # 通过 .loc 设置值
            self.obj.loc[key] = value
            return

        # 调用父类的 __setitem__ 方法
        return super().__setitem__(key, value)


@doc(IndexingMixin.iat)
class _iAtIndexer(_ScalarAccessIndexer):
    _takeable = True
    def _convert_key(self, key):
        """
        将键转换为整数参数（并转换为标签参数）
        """
        # 遍历键中的每个元素
        for i in key:
            # 检查当前元素是否为整数，若不是则抛出值错误异常
            if not is_integer(i):
                raise ValueError("iAt based indexing can only have integer indexers")
        # 返回转换后的键
        return key
# 创建一个由多个 Hashable 类型或者 slice 类型元素组成的列表，用于表示多维索引器的元组
def _tuplify(ndim: int, loc: Hashable) -> tuple[Hashable | slice, ...]:
    _tup: list[Hashable | slice]
    # 初始化一个包含 ndim 个 slice(None, None) 的列表
    _tup = [slice(None, None) for _ in range(ndim)]
    # 将列表的第一个元素替换为 loc
    _tup[0] = loc
    # 返回转换为元组的 _tup 列表
    return tuple(_tup)


# 根据指定的轴和索引键创建一个元组，使得索引操作与轴无关
def _tupleize_axis_indexer(ndim: int, axis: AxisInt, key) -> tuple:
    new_key = [slice(None)] * ndim
    # 将 new_key 中的第 axis 个位置替换为 key
    new_key[axis] = key
    return tuple(new_key)


# 检查 key 是否是一个有效的布尔类型索引器，并进行必要的重新索引或转换
def check_bool_indexer(index: Index, key) -> np.ndarray:
    result = key
    # 如果 key 是 ABCSeries 的实例，并且其索引与 index 不相等
    if isinstance(key, ABCSeries) and not key.index.equals(index):
        # 获取 key 对 index 的索引器
        indexer = result.index.get_indexer_for(index)
        # 如果 indexer 中有 -1 存在，抛出 IndexingError
        if -1 in indexer:
            raise IndexingError(
                "Unalignable boolean Series provided as "
                "indexer (index of the boolean Series and of "
                "the indexed object do not match)."
            )
        # 使用 indexer 对 result 进行重新索引
        result = result.take(indexer)

        # 如果 result 的 dtype 不是 ExtensionDtype 类型，转换为 bool 类型的数组
        if not isinstance(result.dtype, ExtensionDtype):
            return result.astype(bool)._values

    # 如果 key 是 object dtype，转换为 bool 类型的数组
    if is_object_dtype(key):
        result = np.asarray(result, dtype=bool)
    # 如果 result 不是类数组对象，使用 pd_array 将其转换为 bool 类型的数组
    elif not is_array_like(result):
        result = pd_array(result, dtype=bool)
    # 返回对 index 和 result 进行检查后的结果
    return check_array_indexer(index, result)


# 反向转换一个缺失索引器（一般为字典形式），返回标量索引器和是否转换的布尔值
def convert_missing_indexer(indexer):
    if isinstance(indexer, dict):
        # 获取字典中的 "key" 键对应的值
        indexer = indexer["key"]
        # 如果 indexer 是布尔值，抛出 KeyError
        if isinstance(indexer, bool):
            raise KeyError("cannot use a single bool to index into setitem")
        # 返回处理后的 indexer 和 True，表示已转换
        return indexer, True

    # 如果 indexer 不是字典，直接返回 indexer 和 False，表示未转换
    return indexer, False


# 创建一个不包含任何缺失索引器的过滤索引器
def convert_from_missing_indexer_tuple(indexer, axes):
    def get_indexer(_i, _idx):
        # 如果 _idx 是字典，使用 axes[_i] 获取其 "key" 键对应的位置索引，否则直接使用 _idx
        return axes[_i].get_loc(_idx["key"]) if isinstance(_idx, dict) else _idx
    # 返回一个元组，其中包含根据索引器索引生成的结果
    return tuple(get_indexer(_i, _idx) for _i, _idx in enumerate(indexer))
# 检查参数是否可以转换为交叉产品的形式
def maybe_convert_ix(*args):
    for arg in args:
        # 如果参数不是 ndarray、list、ABCSeries 或 Index 类型的实例，则直接返回参数元组
        if not isinstance(arg, (np.ndarray, list, ABCSeries, Index)):
            return args
    # 使用 np.ix_ 函数生成交叉产品
    return np.ix_(*args)


# 检查是否为嵌套元组且包含 MultiIndex 标签
def is_nested_tuple(tup, labels) -> bool:
    # 如果 tup 不是 tuple 类型，则返回 False
    if not isinstance(tup, tuple):
        return False

    # 遍历 tup 中的元素
    for k in tup:
        # 如果 k 是类似于 list 的对象或者是 slice 类型，则检查 labels 是否为 MultiIndex 类型
        if is_list_like(k) or isinstance(k, slice):
            return isinstance(labels, MultiIndex)

    return False


# 检查是否类似于标签的键
def is_label_like(key) -> bool:
    # 如果 key 不是 slice 类型，并且不是类似于 list 的索引器，并且不是省略符号 Ellipsis，则返回 True
    return (
        not isinstance(key, slice)
        and not is_list_like_indexer(key)
        and key is not Ellipsis
    )


# 检查是否需要切片
def need_slice(obj: slice) -> bool:
    # 如果 obj 的 start 不为 None，或者 obj 的 stop 不为 None，或者 obj 的 step 不为 None 且不等于 1，则返回 True
    return (
        obj.start is not None
        or obj.stop is not None
        or (obj.step is not None and obj.step != 1)
    )


# 检查字典或集合索引器
def check_dict_or_set_indexers(key) -> None:
    # 如果 key 是 set 类型，或者 key 是 tuple 类型且包含 set 类型的元素，则抛出 TypeError
    if (
        isinstance(key, set)
        or isinstance(key, tuple)
        and any(isinstance(x, set) for x in key)
    ):
        raise TypeError(
            "Passing a set as an indexer is not supported. Use a list instead."
        )

    # 如果 key 是 dict 类型，或者 key 是 tuple 类型且包含 dict 类型的元素，则抛出 TypeError
    if (
        isinstance(key, dict)
        or isinstance(key, tuple)
        and any(isinstance(x, dict) for x in key)
    ):
        raise TypeError(
            "Passing a dict as an indexer is not supported. Use a list instead."
        )
```