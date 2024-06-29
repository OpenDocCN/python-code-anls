# `D:\src\scipysrc\pandas\pandas\core\frame.py`

```
"""
DataFrame
---------
An efficient 2D container for potentially mixed-type time series or other
labeled data series.

Similar to its R counterpart, data.frame, except providing automatic data
alignment and a host of useful data manipulation methods having to do with the
labeling information
"""

# 导入必要的模块和库

from __future__ import annotations

import collections                    # 引入collections模块
from collections import abc           # 从collections模块中导入abc
from collections.abc import (         # 从collections.abc模块中导入以下类型
    Callable,                        # 可调用对象
    Hashable,                        # 可散列对象
    Iterable,                        # 可迭代对象
    Iterator,                        # 迭代器对象
    Mapping,                         # 映射对象
    Sequence                         # 序列对象
)
import functools                     # 引入functools模块
from io import StringIO              # 从io模块中导入StringIO类
import itertools                     # 引入itertools模块
import operator                      # 引入operator模块
import sys                           # 引入sys模块
from textwrap import dedent          # 从textwrap模块中导入dedent函数
from typing import (                 # 从typing模块中导入以下类型
    TYPE_CHECKING,                  # 类型检查
    Any,                            # 任意类型
    Literal,                        # 字面值类型
    cast,                           # 类型强制转换
    overload                        # 函数重载
)
import warnings                      # 引入warnings模块

import numpy as np                   # 引入numpy并重命名为np
from numpy import ma                 # 从numpy中导入ma模块

from pandas._config import get_option            # 从pandas._config模块中导入get_option函数

from pandas._libs import (                       # 从pandas._libs模块中导入以下内容
    algos as libalgos,                           # 导入algos并重命名为libalgos
    lib,                                         # 导入lib
    properties                                   # 导入properties
)
from pandas._libs.hashtable import duplicated    # 从pandas._libs.hashtable模块中导入duplicated函数
from pandas._libs.lib import is_range_indexer    # 从pandas._libs.lib模块中导入is_range_indexer函数
from pandas.compat import PYPY                    # 从pandas.compat模块中导入PYPY
from pandas.compat._constants import REF_COUNT    # 从pandas.compat._constants模块中导入REF_COUNT
from pandas.compat._optional import import_optional_dependency  # 从pandas.compat._optional模块中导入import_optional_dependency函数
from pandas.compat.numpy import function as nv    # 从pandas.compat.numpy模块中导入function并重命名为nv
from pandas.errors import (                       # 从pandas.errors模块中导入以下错误类型
    ChainedAssignmentError,                      # 连锁赋值错误
    InvalidIndexError                            # 无效索引错误
)
from pandas.errors.cow import (                   # 从pandas.errors.cow模块中导入以下内容
    _chained_assignment_method_msg,               # 连锁赋值方法消息
    _chained_assignment_msg                       # 连锁赋值消息
)
from pandas.util._decorators import (             # 从pandas.util._decorators模块中导入以下装饰器
    Appender,                                     # Appender装饰器
    Substitution,                                 # Substitution装饰器
    deprecate_nonkeyword_arguments,               # deprecate_nonkeyword_arguments装饰器
    doc,                                          # doc装饰器
    set_module                                    # set_module装饰器
)
from pandas.util._exceptions import (              # 从pandas.util._exceptions模块中导入以下异常处理函数
    find_stack_level,                             # find_stack_level函数
    rewrite_warning                               # rewrite_warning函数
)
from pandas.util._validators import (              # 从pandas.util._validators模块中导入以下验证函数
    validate_ascending,                            # validate_ascending函数
    validate_bool_kwarg,                           # validate_bool_kwarg函数
    validate_percentile                           # validate_percentile函数
)

from pandas.core.dtypes.cast import (              # 从pandas.core.dtypes.cast模块中导入以下类型转换函数
    LossySetitemError,                            # 丢失设置项错误
    can_hold_element,                             # can_hold_element函数
    construct_1d_arraylike_from_scalar,           # construct_1d_arraylike_from_scalar函数
    construct_2d_arraylike_from_scalar,           # construct_2d_arraylike_from_scalar函数
    find_common_type,                             # find_common_type函数
    infer_dtype_from_scalar,                      # infer_dtype_from_scalar函数
    invalidate_string_dtypes,                     # invalidate_string_dtypes函数
    maybe_downcast_to_dtype                       # maybe_downcast_to_dtype函数
)
from pandas.core.dtypes.common import (            # 从pandas.core.dtypes.common模块中导入以下常用类型判断函数
    infer_dtype_from_object,                      # infer_dtype_from_object函数
    is_1d_only_ea_dtype,                          # is_1d_only_ea_dtype函数
    is_array_like,                                # is_array_like函数
    is_bool_dtype,                                # is_bool_dtype函数
    is_dataclass,                                 # is_dataclass函数
    is_dict_like,                                 # is_dict_like函数
    is_float,                                     # is_float函数
    is_float_dtype,                               # is_float_dtype函数
    is_hashable,                                  # is_hashable函数
    is_integer,                                   # is_integer函数
    is_integer_dtype,                             # is_integer_dtype函数
    is_iterator,                                  # is_iterator函数
    is_list_like,                                 # is_list_like函数
    is_scalar,                                    # is_scalar函数
    is_sequence,                                  # is_sequence函数
    needs_i8_conversion,                          # needs_i8_conversion函数
    pandas_dtype                                  # pandas_dtype函数
)
from pandas.core.dtypes.concat import concat_compat  # 从pandas.core.dtypes.concat模块中导入concat_compat函数
from pandas.core.dtypes.dtypes import (             # 从pandas.core.dtypes.dtypes模块中导入以下数据类型
    ArrowDtype,                                     # ArrowDtype类型
    BaseMaskedDtype,                                # BaseMaskedDtype类型
    ExtensionDtype                                  # ExtensionDtype类型
)
from pandas.core.dtypes.missing import (            # 从pandas.core.dtypes.missing模块中导入以下缺失值处理函数
    isna,                                           # isna函数
    notna                                           # notna函数
)

from pandas.core import (                           # 从pandas.core模块中导入以下核心功能
    algorithms,                                     # 算法模块
    common as com,                                  # 通用模块并重命名为com
    nanops,                                         # nanops模块
    ops,                                            # ops模块
    roperator                                       # roperator模块
)
from pandas.core.accessor import Accessor            # 从pandas.core.accessor模块中导入Accessor类
from pandas.core.apply import reconstruct_and_relabel_result  # 从pandas.core.apply模块中导入reconstruct_and_relabel_result函数
from pandas.core.array_algos.take import take_2d_multi  # 从pandas.core.array_algos.take模块中导入take_2d_multi函数
from pandas.core.arraylike import OpsMixin           # 从pandas.core.arraylike模块中导入OpsMixin类
from pandas.core.arrays import (                     # 从pandas.core.arrays模块中导入以下数组类型
    BaseMaskedArray,                                 # BaseMaskedArray类型
    DatetimeArray,                                   # DatetimeArray类型
    ExtensionArray,                                  # ExtensionArray类型
    PeriodArray,                                     # PeriodArray类型
    TimedeltaArray                                   # TimedeltaArray类型
)
# 从 pandas 库中导入稀疏框架访问器
from pandas.core.arrays.sparse import SparseFrameAccessor
# 从 pandas 库中导入数据结构构建相关函数
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    sanitize_array,
    sanitize_masked_array,
)
# 从 pandas 库中导入通用数据结构 NDFrame 和文档生成函数
from pandas.core.generic import (
    NDFrame,
    make_doc,
)
# 从 pandas 库中导入索引检查函数
from pandas.core.indexers import check_key_length
# 从 pandas 库中导入索引相关的 API 函数
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    PeriodIndex,
    default_index,
    ensure_index,
    ensure_index_from_sequences,
)
# 从 pandas 库中导入多重索引相关函数和方法
from pandas.core.indexes.multi import (
    MultiIndex,
    maybe_droplevels,
)
# 从 pandas 库中导入索引器检查函数
from pandas.core.indexing import (
    check_bool_indexer,
    check_dict_or_set_indexers,
)
# 从 pandas 库中导入数据块管理器
from pandas.core.internals import BlockManager
# 从 pandas 库中导入数据结构构建相关函数
from pandas.core.internals.construction import (
    arrays_to_mgr,
    dataclasses_to_dicts,
    dict_to_mgr,
    ndarray_to_mgr,
    nested_data_to_arrays,
    rec_array_to_mgr,
    reorder_arrays,
    to_arrays,
    treat_as_nested,
)
# 从 pandas 库中导入选择操作方法
from pandas.core.methods import selectn
# 从 pandas 库中导入数据重塑相关函数
from pandas.core.reshape.melt import melt
# 从 pandas 库中导入序列数据结构
from pandas.core.series import Series
# 从 pandas 库中导入共享文档相关模块
from pandas.core.shared_docs import _shared_docs
# 从 pandas 库中导入排序相关函数
from pandas.core.sorting import (
    get_group_index,
    lexsort_indexer,
    nargsort,
)

# 从 pandas.io.common 模块中导入获取文件句柄函数
from pandas.io.common import get_handle
# 从 pandas.io.formats 模块中导入控制台格式化、格式化和数据框信息相关函数
from pandas.io.formats import (
    console,
    format as fmt,
)
# 从 pandas.io.formats.info 模块中导入信息字符串、数据框信息和子参数提取函数
from pandas.io.formats.info import (
    INFO_DOCSTRING,
    DataFrameInfo,
    frame_sub_kwargs,
)
# 导入 pandas 的绘图模块
import pandas.plotting

# 如果类型检查为真，导入特定类型的模块和类型定义
if TYPE_CHECKING:
    import datetime  # 导入日期时间模块

    from pandas._libs.internals import BlockValuesRefs  # 导入内部块值引用
    from pandas._typing import (  # 导入类型定义
        AggFuncType,
        AnyAll,
        AnyArrayLike,
        ArrayLike,
        Axes,
        Axis,
        AxisInt,
        ColspaceArgType,
        CompressionOptions,
        CorrelationMethod,
        DropKeep,
        Dtype,
        DtypeObj,
        FilePath,
        FloatFormatType,
        FormattersType,
        Frequency,
        FromDictOrient,
        HashableT,
        HashableT2,
        IgnoreRaise,
        IndexKeyFunc,
        IndexLabel,
        JoinValidate,
        Level,
        ListLike,
        MergeHow,
        MergeValidate,
        MutableMappingT,
        NaPosition,
        NsmallestNlargestKeep,
        PythonFuncType,
        QuantileInterpolation,
        ReadBuffer,
        ReindexMethod,
        Renamer,
        Scalar,
        Self,
        SequenceNotStr,
        SortKind,
        StorageOptions,
        Suffixes,
        T,
        ToStataByteorder,
        ToTimestampHow,
        UpdateJoin,
        ValueKeyFunc,
        WriteBuffer,
        XMLParsers,
        npt,
    )

    from pandas.core.groupby.generic import DataFrameGroupBy  # 导入数据框分组通用模块
    from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg  # 导入数据帧协议
    from pandas.core.internals.managers import SingleBlockManager  # 导入单块管理器

    from pandas.io.formats.style import Styler  # 导入样式模块

# ---------------------------------------------------------------------
# 文档字符串模板

_shared_doc_kwargs = {
    "axes": "index, columns",  # "axes" 参数的描述
    "klass": "DataFrame",  # "klass" 参数的描述
}
    # 定义参数 axes_single_arg，可以接受值 0 或 'index'，1 或 'columns'
    "axes_single_arg": "{0 or 'index', 1 or 'columns'}",
    
    # 定义参数 axis，可以接受值 0 或 'index'，1 或 'columns'，默认为 0
    # 当为 0 或 'index' 时，将函数应用于每一列
    # 当为 1 或 'columns' 时，将函数应用于每一行
    "axis": """axis : {0 or 'index', 1 or 'columns'}, default 0
        If 0 or 'index': apply function to each column.
        If 1 or 'columns': apply function to each row.""",
    
    # 定义参数 inplace，可以接受布尔值，默认为 False
    # 当为 True 时，修改原始 DataFrame，而不是创建一个新的 DataFrame
    "inplace": """
    inplace : bool, default False
        Whether to modify the DataFrame rather than creating a new one.""",
    
    # 定义参数 optional_by，暂时没有提供具体的说明
    "optional_by": """
{
    "by": """
Name or list of names to sort by.

- if `axis` is 0 or `'index'` then `by` may contain index
  levels and/or column labels.
- if `axis` is 1 or `'columns'` then `by` may contain column
  levels and/or index labels.
""",
    "optional_reindex": """
labels : array-like, optional
New labels / index to conform the axis specified by 'axis' to.
index : array-like, optional
New labels for the index. Preferably an Index object to avoid
duplicating data.
columns : array-like, optional
New labels for the columns. Preferably an Index object to avoid
duplicating data.
axis : int or str, optional
Axis to target. Can be either the axis name ('index', 'columns')
or number (0, 1).
""",
}

_merge_doc = """
Merge DataFrame or named Series objects with a database-style join.

A named Series object is treated as a DataFrame with a single named column.

The join is done on columns or indexes. If joining columns on
columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
on indexes or indexes on a column or columns, the index will be passed on.
When performing a cross merge, no column specifications to merge on are
allowed.

.. warning::

If both key columns contain rows where the key is a null value, those
rows will be matched against each other. This is different from usual SQL
join behaviour and can lead to unexpected results.

Parameters
----------%s
right : DataFrame or named Series
Object to merge with.
how : {'left', 'right', 'outer', 'inner', 'cross'}, default 'inner'
Type of merge to be performed.

* left: use only keys from left frame, similar to a SQL left outer join;
preserve key order.
* right: use only keys from right frame, similar to a SQL right outer join;
preserve key order.
* outer: use union of keys from both frames, similar to a SQL full outer
join; sort keys lexicographically.
* inner: use intersection of keys from both frames, similar to a SQL inner
join; preserve the order of the left keys.
* cross: creates the cartesian product from both frames, preserves the order
of the left keys.
on : label or list
Column or index level names to join on. These must be found in both
DataFrames. If `on` is None and not merging on indexes then this defaults
to the intersection of the columns in both DataFrames.
left_on : label or list, or array-like
Column or index level names to join on in the left DataFrame. Can also
be an array or list of arrays of the length of the left DataFrame.
These arrays are treated as if they are columns.
right_on : label or list, or array-like
Column or index level names to join on in the right DataFrame. Can also
be an array or list of arrays of the length of the right DataFrame.
These arrays are treated as if they are columns.
left_index : bool, default False
Use the index from the left DataFrame as the join key(s). If it is a
    MultiIndex, the number of keys in the other DataFrame (either the index
    or a number of columns) must match the number of levels.
# right_index : bool, default False
# 将右侧数据框的索引作为连接键使用。与 left_index 一样存在相同的注意事项。

# sort : bool, default False
# 在结果数据框中按字典顺序对连接键进行排序。如果为 False，则连接键的顺序取决于连接类型（通过 how 关键字指定）。

# suffixes : list-like, default is ("_x", "_y")
# 长度为2的序列，其中每个元素可选地表示要添加到左侧和右侧数据框中重叠列名的后缀。
# 将 None 替换为字符串以指示应该保留来自左侧或右侧的列名，不添加后缀。至少一个值必须不是 None。

# copy : bool, default False
# 如果为 False，则尽可能避免复制。

# .. note::
#     `copy` 关键字将在 pandas 3.0 中更改行为。
#     将启用默认的 Copy-on-Write 机制，这意味着所有带有 `copy` 关键字的方法将使用延迟复制机制来推迟复制并忽略 `copy` 关键字。
#     `copy` 关键字将在将来的 pandas 版本中被移除。
# 
#     您可以通过启用 copy on write `pd.options.mode.copy_on_write = True` 来获取未来的行为和改进。

# .. deprecated:: 3.0.0
# indicator : bool or str, default False
# 如果为 True，则向输出数据框添加名为 "_merge" 的列，列中包含每行数据的来源信息。
# 可以通过提供一个字符串参数来指定不同的列名。
# 列将具有分类类型，对于只出现在左侧数据框中的合并键的观察值，值为 "left_only"；
# 对于只出现在右侧数据框中的合并键的观察值，值为 "right_only"；
# 如果合并键在两个数据框中都找到，则值为 "both"。

# validate : str, optional
# 如果指定，则检查合并的类型。

# * "one_to_one" 或 "1:1"：检查左右数据集中的合并键是否在各自数据集中唯一。
# * "one_to_many" 或 "1:m"：检查左侧数据集中的合并键是否唯一。
# * "many_to_one" 或 "m:1"：检查右侧数据集中的合并键是否唯一。
# * "many_to_many" 或 "m:m"：允许，但不进行检查。
# -----------------------------------------------------------------------
# DataFrame class


@set_module("pandas")
class DataFrame(NDFrame, OpsMixin):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Data structure also contains labeled axes (rows and columns).
    Arithmetic operations align on both row and column labels. Can be
    thought of as a dict-like container for Series objects. The primary
    pandas data structure.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Dict can contain Series, arrays, constants, dataclass or list-like objects. If
        data is a dict, column order follows insertion-order. If a dict contains Series
        which have an index defined, it is aligned by its index. This alignment also
        occurs if data is a Series or a DataFrame itself. Alignment is done on
        Series/DataFrame inputs.

        If data is a list of dicts, column order follows insertion-order.

    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    """
    # columns : Index or array-like
    # 结果DataFrame的列标签，当数据没有列标签时使用，默认为RangeIndex(0, 1, 2, ..., n)。
    # 如果数据包含列标签，则执行列选择而不是默认生成的列标签。
    dtype : dtype, default None
    # 强制的数据类型。只允许单个数据类型。如果为None，则推断数据类型。
    copy : bool or None, default None
    # 是否复制输入数据。
    # 对于字典数据，默认值为None时相当于``copy=True``。
    # 对于DataFrame或2维ndarray输入，默认值为None时相当于``copy=False``。
    # 如果数据是包含一个或多个Series的字典（可能具有不同的数据类型），
    # ``copy=False`` 将确保这些输入不被复制。

    # .. versionchanged:: 1.3.0
    # 请参阅版本变更说明，了解更多信息。

    See Also
    --------
    DataFrame.from_records : 从元组构造DataFrame，也适用于记录数组。
    DataFrame.from_dict : 从字典构造DataFrame，字典可以包含Series、数组或字典。
    read_csv : 从逗号分隔值（csv）文件读取数据到DataFrame。
    read_table : 从通用分隔文件读取数据到DataFrame。
    read_clipboard : 从剪贴板读取文本数据到DataFrame。

    Notes
    -----
    请参考 :ref:`用户指南 <basics.dataframe>` 获取更多信息。

    Examples
    --------
    从字典构造DataFrame。

    >>> d = {"col1": [1, 2], "col2": [3, 4]}
    >>> df = pd.DataFrame(data=d)
    >>> df
       col1  col2
    0     1     3
    1     2     4

    注意推断出的数据类型为int64。

    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    强制使用单一数据类型：

    >>> df = pd.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object

    从包含Series的字典构造DataFrame：

    >>> d = {"col1": [0, 1, 2, 3], "col2": pd.Series([2, 3], index=[2, 3])}
    >>> pd.DataFrame(data=d, index=[0, 1, 2, 3])
       col1  col2
    0     0   NaN
    1     1   NaN
    2     2   2.0
    3     3   3.0

    从numpy ndarray构造DataFrame：

    >>> df2 = pd.DataFrame(
    ...     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    ... )
    >>> df2
       a  b  c
    0  1  2  3
    1  4  5  6
    2  7  8  9

    从具有标签列的numpy ndarray构造DataFrame：

    >>> data = np.array(
    ...     [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    ...     dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")],
    ... )
    >>> df3 = pd.DataFrame(data, columns=["c", "a"])
    >>> df3
       c  a
    0  3  1
    1  6  4
    2  9  7

    从dataclass构造DataFrame：

    >>> from dataclasses import make_dataclass
    >>> Point = make_dataclass("Point", [("x", int), ("y", int)])
    >>> pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])
       x  y
    0  0  0
    1  0  3
    2  2  3

    从Series/DataFrame构造DataFrame：

    >>> ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
    # 创建一个 DataFrame，使用指定的数据（ser），并指定行索引为 ["a", "c"]
    >>> df = pd.DataFrame(data=ser, index=["a", "c"])
    # 显示 DataFrame df 的内容
    >>> df
       0
    a  1
    c  3

    # 创建一个包含数据 [1, 2, 3] 的 DataFrame，指定行索引为 ["a", "b", "c"]，列名为 ["x"]
    >>> df1 = pd.DataFrame([1, 2, 3], index=["a", "b", "c"], columns=["x"])
    # 创建一个新的 DataFrame df2，使用 df1 的数据，并指定行索引为 ["a", "c"]
    >>> df2 = pd.DataFrame(data=df1, index=["a", "c"])
    # 显示 DataFrame df2 的内容
    >>> df2
       x
    a  1
    c  3
    """

    # _internal_names_set 是一个集合，包含了 "columns", "index" 和 NDFrame._internal_names_set 的元素
    _internal_names_set = {"columns", "index"} | NDFrame._internal_names_set
    # _typ 是字符串 "dataframe"
    _typ = "dataframe"
    # _HANDLED_TYPES 是一个元组，包含了 Series、Index、ExtensionArray 和 np.ndarray
    _HANDLED_TYPES = (Series, Index, ExtensionArray, np.ndarray)
    # _accessors 是一个集合，包含字符串 "sparse"
    _accessors: set[str] = {"sparse"}
    # _hidden_attrs 是一个不可变集合，包含 NDFrame._hidden_attrs 的所有元素，以及一个空的字符串集合
    _hidden_attrs: frozenset[str] = NDFrame._hidden_attrs | frozenset([])
    # _mgr 是一个 BlockManager 对象

    # __pandas_priority__ 是一个整数，设定为 4000，表示 DataFrame 在数组操作中的优先级高于 Series、Index 和 ExtensionArray
    __pandas_priority__ = 4000

    @property
    # _constructor 是一个属性，返回 DataFrame 类型
    def _constructor(self) -> type[DataFrame]:
        return DataFrame

    # _constructor_from_mgr 是一个方法，根据给定的 mgr 和 axes 创建一个 DataFrame 对象
    def _constructor_from_mgr(self, mgr, axes) -> DataFrame:
        df = DataFrame._from_mgr(mgr, axes=axes)

        if type(self) is DataFrame:
            # 如果 self 是 DataFrame 类型，则返回 df
            return df

        elif type(self).__name__ == "GeoDataFrame":
            # 如果 self 的类名为 "GeoDataFrame"，则调用 self._constructor(mgr) 返回结果
            return self._constructor(mgr)

        # 假设子类的 __init__ 方法知道如何处理 pd.DataFrame 对象，则返回 self._constructor(df)
        return self._constructor(df)

    # _constructor_sliced 是一个 Callable 对象，用于创建 Series 类型的对象
    _constructor_sliced: Callable[..., Series] = Series

    # _constructor_sliced_from_mgr 是一个方法，根据给定的 mgr 和 axes 创建一个 Series 对象
    def _constructor_sliced_from_mgr(self, mgr, axes) -> Series:
        ser = Series._from_mgr(mgr, axes)
        ser._name = None  # 调用者负责设置真实的名称

        if type(self) is DataFrame:
            # 如果 self 是 DataFrame 类型，则返回 ser
            return ser

        # 假设子类的 __init__ 方法知道如何处理 pd.Series 对象，则返回 self._constructor_sliced(ser)
        return self._constructor_sliced(ser)

    # ----------------------------------------------------------------------
    # Constructors

    # __init__ 是构造方法，用于初始化 DataFrame 对象的各个属性和数据
    def __init__(
        self,
        data=None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
    # ----------------------------------------------------------------------
    
    # __dataframe__ 是一个方法，用于返回当前对象自身，支持设置 nan_as_null 和 allow_copy 参数
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> DataFrameXchg:
        """
        Return the dataframe interchange object implementing the interchange protocol.

        Parameters
        ----------
        nan_as_null : bool, default False
            `nan_as_null` is DEPRECATED and has no effect. Please avoid using
            it; it will be removed in a future release.
        allow_copy : bool, default True
            Whether to allow memory copying when exporting. If set to False
            it would cause non-zero-copy exports to fail.

        Returns
        -------
        DataFrame interchange object
            The object which consuming library can use to ingress the dataframe.

        See Also
        --------
        DataFrame.from_records : Constructor from tuples, also record arrays.
        DataFrame.from_dict : From dicts of Series, arrays, or dicts.

        Notes
        -----
        Details on the interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html

        Examples
        --------
        >>> df_not_necessarily_pandas = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> interchange_object = df_not_necessarily_pandas.__dataframe__()
        >>> interchange_object.column_names()
        Index(['A', 'B'], dtype='object')
        >>> df_pandas = pd.api.interchange.from_dataframe(
        ...     interchange_object.select_columns_by_name(["A"])
        ... )
        >>> df_pandas
             A
        0    1
        1    2

        These methods (``column_names``, ``select_columns_by_name``) should work
        for any dataframe library which implements the interchange protocol.
        """

        from pandas.core.interchange.dataframe import PandasDataFrameXchg
        # 导入 PandasDataFrameXchg 类

        return PandasDataFrameXchg(self, allow_copy=allow_copy)
        # 返回一个 PandasDataFrameXchg 对象，用于数据帧的交换，参数包括是否允许复制内存

    def __arrow_c_stream__(self, requested_schema=None):
        """
        Export the pandas DataFrame as an Arrow C stream PyCapsule.

        This relies on pyarrow to convert the pandas DataFrame to the Arrow
        format (and follows the default behaviour of ``pyarrow.Table.from_pandas``
        in its handling of the index, i.e. store the index as a column except
        for RangeIndex).
        This conversion is not necessarily zero-copy.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the dataframe should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.

        Returns
        -------
        PyCapsule
            A PyCapsule object representing the Arrow C stream.

        """
        pa = import_optional_dependency("pyarrow", min_version="14.0.0")
        # 导入 pyarrow 库，版本需至少为 14.0.0
        if requested_schema is not None:
            requested_schema = pa.Schema._import_from_c_capsule(requested_schema)
            # 如果指定了请求的 schema，则从 C Capsule 中导入 ArrowSchema 表示
        table = pa.Table.from_pandas(self, schema=requested_schema)
        # 使用 pyarrow 将 pandas DataFrame 转换为 Arrow Table 对象
        return table.__arrow_c_stream__()
        # 返回 Arrow Table 对象的 Arrow C stream 表示

    # ----------------------------------------------------------------------
    
    @property
    def axes(self) -> list[Index]:
        """
        Return a list representing the axes of the DataFrame.

        It has the row axis labels and column axis labels as the only members.
        They are returned in that order.

        See Also
        --------
        DataFrame.index: The index (row labels) of the DataFrame.
        DataFrame.columns: The column labels of the DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.axes
        [RangeIndex(start=0, stop=2, step=1), Index(['col1', 'col2'],
        dtype='object')]
        """
        # 返回包含 DataFrame 的行和列轴标签的列表
        return [self.index, self.columns]

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return a tuple representing the dimensionality of the DataFrame.

        See Also
        --------
        ndarray.shape : Tuple of array dimensions.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.shape
        (2, 2)

        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        >>> df.shape
        (2, 3)
        """
        # 返回一个元组，表示 DataFrame 的维度（行数和列数）
        return len(self.index), len(self.columns)

    @property
    def _is_homogeneous_type(self) -> bool:
        """
        Whether all the columns in a DataFrame have the same type.

        Returns
        -------
        bool

        Examples
        --------
        >>> DataFrame({"A": [1, 2], "B": [3, 4]})._is_homogeneous_type
        True
        >>> DataFrame({"A": [1, 2], "B": [3.0, 4.0]})._is_homogeneous_type
        False

        Items with the same type but different sizes are considered
        different types.

        >>> DataFrame(
        ...     {
        ...         "A": np.array([1, 2], dtype=np.int32),
        ...         "B": np.array([1, 2], dtype=np.int64),
        ...     }
        ... )._is_homogeneous_type
        False
        """
        # 检查 DataFrame 中所有列是否具有相同的数据类型
        # "<=" 这里的 "<" 是为了处理空 DataFrame 的情况
        return len({block.values.dtype for block in self._mgr.blocks}) <= 1

    @property
    def _can_fast_transpose(self) -> bool:
        """
        Can we transpose this DataFrame without creating any new array objects.
        """
        blocks = self._mgr.blocks
        # 如果 DataFrame 的数据块数量不为 1，则无法进行快速转置
        if len(blocks) != 1:
            return False

        dtype = blocks[0].dtype
        # TODO(EA2D) 特殊情况应该与 2D EAs 无关
        # 判断是否可以快速转置，不需要创建新的数组对象
        return not is_1d_only_ea_dtype(dtype)
    def _values(self) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray:
        """
        返回与 ._values 对应的方法，可能返回一个二维 ExtensionArray。
        """
        mgr = self._mgr  # 获取对象的 _mgr 属性

        blocks = mgr.blocks  # 获取 _mgr 属性中的 blocks 属性
        if len(blocks) != 1:  # 如果 blocks 不只有一个块
            return ensure_wrapped_if_datetimelike(self.values)  # 返回确保为日期时间类型的包装后的值

        arr = blocks[0].values  # 获取第一个块的值
        if arr.ndim == 1:  # 如果 arr 是一维的 ExtensionArray
            # 非二维 ExtensionArray
            return self.values  # 返回当前对象的值

        # 更一般地，允许在 NDArrayBackedExtensionBlock 中的各种类型
        arr = cast("np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray", arr)
        return arr.T  # 返回 arr 的转置

    # ----------------------------------------------------------------------
    # 渲染方法

    def _repr_fits_vertical_(self) -> bool:
        """
        检查长度是否小于等于 max_rows。
        """
        max_rows = get_option("display.max_rows")  # 获取显示选项中的 max_rows
        return len(self) <= max_rows  # 返回当前对象的长度是否小于等于 max_rows

    def _repr_fits_horizontal_(self) -> bool:
        """
        检查完整 repr 是否适应显示选项中的水平边界。
        """
        width, height = console.get_console_size()  # 获取控制台的宽度和高度
        max_columns = get_option("display.max_columns")  # 获取显示选项中的 max_columns
        nb_columns = len(self.columns)  # 获取当前对象的列数

        # 超出最大列数
        if (max_columns and nb_columns > max_columns) or (
            width and nb_columns > (width // 2)
        ):
            return False  # 返回 False，表示不适合水平显示

        # 用于 IPython 笔记本或脚本忽略终端维度的 repr_html
        if width is None or not console.in_interactive_session():
            return True  # 返回 True，表示适合水平显示

        if get_option("display.width") is not None or console.in_ipython_frontend():
            # 至少检查列行是否过宽
            max_rows = 1
        else:
            max_rows = get_option("display.max_rows")

        # 当自动检测时，width=None 并且不在 ipython 前端时
        # 实际检查渲染 repr 的宽度是否适合水平显示
        buf = StringIO()

        # 只关心实际打印出来的内容
        # 对整个 frame 进行 to_string 可能比较昂贵
        d = self

        if max_rows is not None:  # 限制行数
            # 取两者中的最小值，其中一个可能为 None
            d = d.iloc[: min(max_rows, len(d))]
        else:
            return True

        d.to_string(buf=buf)
        value = buf.getvalue()
        repr_width = max(len(line) for line in value.split("\n"))

        return repr_width < width  # 返回 repr 宽度是否小于控制台宽度

    def _info_repr(self) -> bool:
        """
        如果 repr 应显示信息视图，则返回 True。
        """
        info_repr_option = get_option("display.large_repr") == "info"  # 获取显示选项中的 large_repr 是否为 info
        return info_repr_option and not (
            self._repr_fits_horizontal_() and self._repr_fits_vertical_()
        )  # 返回是否需要显示信息视图的逻辑结果
    def __repr__(self) -> str:
        """
        Return a string representation for a particular DataFrame.
        """
        # 如果数据帧有信息的表现形式，则使用字符串缓冲区来获取信息，并返回其值
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            return buf.getvalue()

        # 否则，获取数据帧的格式化参数并返回其字符串表示形式
        repr_params = fmt.get_dataframe_repr_params()
        return self.to_string(**repr_params)

    def _repr_html_(self) -> str | None:
        """
        Return a html representation for a particular DataFrame.

        Mainly for IPython notebook.
        """
        # 如果数据帧有信息的表现形式，则使用字符串缓冲区来获取信息
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            # 需要转义<class>，应该是第一行。
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return f"<pre>{val}</pre>"

        # 如果 IPython 笔记本显示设置允许，则生成 HTML 表示
        if get_option("display.notebook_repr_html"):
            max_rows = get_option("display.max_rows")
            min_rows = get_option("display.min_rows")
            max_cols = get_option("display.max_columns")
            show_dimensions = get_option("display.show_dimensions")

            # 创建 DataFrameFormatter 对象，用于格式化数据帧
            formatter = fmt.DataFrameFormatter(
                self,
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                escape=True,
                max_rows=max_rows,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=".",
            )
            # 返回 HTML 格式的数据帧表示
            return fmt.DataFrameRenderer(formatter).to_html(notebook=True)
        else:
            return None

    @overload
    def to_string(
        self,
        buf: None = ...,
        *,
        columns: Axes | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: fmt.FormattersType | None = ...,
        float_format: fmt.FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool = ...,
        decimal: str = ...,
        line_width: int | None = ...,
        min_rows: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: str | None = ...,
    ) -> str: ...
    # 定义一个方法，将数据框转换为字符串格式输出
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: Axes | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: fmt.FormattersType | None = ...,
        float_format: fmt.FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool = ...,
        decimal: str = ...,
        line_width: int | None = ...,
        min_rows: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: str | None = ...
    ) -> None: ...

    # 使用装饰器Substitution，替换参数和返回值的说明
    @Substitution(
        header_type="bool or list of str",
        header="Write out the column names. If a list of columns "
        "is given, it is assumed to be aliases for the "
        "column names",
        col_space_type="int, list or dict of int",
        col_space="The minimum width of each column. If a list of ints is given "
        "every integers corresponds with one column. If a dict is given, the key "
        "references the column, while the value defines the space to use.",
    )
    # 使用Substitution装饰器，替换共享参数和返回值的说明
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    # 重新定义to_string方法，增加默认参数值
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        columns: Axes | None = None,
        col_space: int | list[int] | dict[Hashable, int] | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: fmt.FormattersType | None = None,
        float_format: fmt.FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        justify: str | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: bool = False,
        decimal: str = ".",
        line_width: int | None = None,
        min_rows: int | None = None,
        max_colwidth: int | None = None,
        encoding: str | None = None,
    ) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.
        %(shared_params)s
        line_width : int, optional
            Width to wrap a line in characters.
        min_rows : int, optional
            The number of rows to display in the console in a truncated repr
            (when number of rows is above `max_rows`).
        max_colwidth : int, optional
            Max width to truncate each column in characters. By default, no limit.
        encoding : str, default "utf-8"
            Set character encoding.
        %(returns)s
        See Also
        --------
        to_html : Convert DataFrame to HTML.

        Examples
        --------
        >>> d = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        >>> df = pd.DataFrame(d)
        >>> print(df.to_string())
           col1  col2
        0     1     4
        1     2     5
        2     3     6
        """
        from pandas import option_context
        
        # 使用 option_context 设置显示选项，设置 max_colwidth 的最大列宽度
        with option_context("display.max_colwidth", max_colwidth):
            # 创建 DataFrameFormatter 对象来格式化 DataFrame
            formatter = fmt.DataFrameFormatter(
                self,
                columns=columns,
                col_space=col_space,
                na_rep=na_rep,
                formatters=formatters,
                float_format=float_format,
                sparsify=sparsify,
                justify=justify,
                index_names=index_names,
                header=header,
                index=index,
                min_rows=min_rows,
                max_rows=max_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=decimal,
            )
            # 使用 DataFrameRenderer 将格式化后的 DataFrame 转换为字符串形式输出
            return fmt.DataFrameRenderer(formatter).to_string(
                buf=buf,
                encoding=encoding,
                line_width=line_width,
            )

    def _get_values_for_csv(
        self,
        *,
        float_format: FloatFormatType | None,
        date_format: str | None,
        decimal: str,
        na_rep: str,
        quoting,  # int csv.QUOTE_FOO from stdlib
    ) -> DataFrame:
        # helper used by to_csv
        # 调用内部方法 _mgr.get_values_for_csv 获取适合导出到 CSV 的数据
        mgr = self._mgr.get_values_for_csv(
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
            na_rep=na_rep,
            quoting=quoting,
        )
        # 使用 _constructor_from_mgr 方法构造一个新的 DataFrame 对象并返回
        return self._constructor_from_mgr(mgr, axes=mgr.axes)

    # ----------------------------------------------------------------------
    
    @property
    def style(self) -> Styler:
        """
        Returns a Styler object.

        Contains methods for building a styled HTML representation of the DataFrame.

        See Also
        --------
        io.formats.style.Styler : Helps style a DataFrame or Series according to the
            data with HTML and CSS.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3]})
        >>> df.style  # doctest: +SKIP

        Please see
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        from pandas.io.formats.style import Styler

        # 返回一个 Styler 对象，用于构建 DataFrame 的 HTML 样式表示
        return Styler(self)

    _shared_docs["items"] = r"""
        Iterate over (column name, Series) pairs.

        Iterates over the DataFrame columns, returning a tuple with
        the column name and the content as a Series.

        Yields
        ------
        label : object
            The column names for the DataFrame being iterated over.
        content : Series
            The column entries belonging to each label, as a Series.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as
            (index, Series) pairs.
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples
            of the values.

        Examples
        --------
        >>> df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],
        ...                   'population': [1864, 22000, 80000]},
        ...                   index=['panda', 'polar', 'koala'])
        >>> df
                species   population
        panda   bear      1864
        polar   bear      22000
        koala   marsupial 80000
        >>> for label, content in df.items():
        ...     print(f'label: {label}')
        ...     print(f'content: {content}', sep='\n')
        ...
        label: species
        content:
        panda         bear
        polar         bear
        koala    marsupial
        Name: species, dtype: object
        label: population
        content:
        panda     1864
        polar    22000
        koala    80000
        Name: population, dtype: int64
        """

    @Appender(_shared_docs["items"])
    def items(self) -> Iterable[tuple[Hashable, Series]]:
        # 遍历 DataFrame 的列，返回每一列的列名和作为 Series 的内容
        for i, k in enumerate(self.columns):
            # 返回列名 k 和其在 axis=1 上索引为 i 的内容作为元组
            yield k, self._ixs(i, axis=1)
    def iterrows(self) -> Iterable[tuple[Hashable, Series]]:
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        See Also
        --------
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples of the values.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames).

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.

        Examples
        --------

        >>> df = pd.DataFrame([[1, 1.5]], columns=["int", "float"])
        >>> row = next(df.iterrows())[1]
        >>> row
        int      1.0
        float    1.5
        Name: 0, dtype: float64
        >>> print(row["int"].dtype)
        float64
        >>> print(df["int"].dtype)
        int64
        """
        # 获取数据框的列名
        columns = self.columns
        # 获取数据框的行构造函数
        klass = self._constructor_sliced
        # 遍历数据框的索引和值
        for k, v in zip(self.index, self.values):
            # 使用行的值构造一个新的 Series 对象
            s = klass(v, index=columns, name=k).__finalize__(self)
            # 如果数据框的内部管理器表示为单一块
            if self._mgr.is_single_block:
                # 将新建的 Series 对象添加到内部管理器的引用中
                s._mgr.add_references(self._mgr)
            # 生成当前行的索引和新建的 Series 对象
            yield k, s

    def itertuples(
        self, index: bool = True, name: str | None = "Pandas"
        # 方法签名：迭代生成命名元组，可选择是否包含索引，命名元组的名称默认为 "Pandas"
    ) -> Iterable[tuple[Any, ...]]:
        """
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str or None, default "Pandas"
            The name of the returned namedtuples or None to return regular
            tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)
            pairs.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        The column names will be renamed to positional names if they are
        invalid Python identifiers, repeated, or start with an underscore.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"num_legs": [4, 2], "num_wings": [0, 2]}, index=["dog", "hawk"]
        ... )
        >>> df
              num_legs  num_wings
        dog          4          0
        hawk         2          2
        >>> for row in df.itertuples():
        ...     print(row)
        Pandas(Index='dog', num_legs=4, num_wings=0)
        Pandas(Index='hawk', num_legs=2, num_wings=2)

        By setting the `index` parameter to False we can remove the index
        as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        Pandas(num_legs=4, num_wings=0)
        Pandas(num_legs=2, num_wings=2)

        With the `name` parameter set we set a custom name for the yielded
        namedtuples:

        >>> for row in df.itertuples(name="Animal"):
        ...     print(row)
        Animal(Index='dog', num_legs=4, num_wings=0)
        Animal(Index='hawk', num_legs=2, num_wings=2)
        """
        # 创建一个空列表，用于存储每一行数据的数组
        arrays = []
        # 获取 DataFrame 的所有列名
        fields = list(self.columns)
        # 如果 index 参数为 True，则将 DataFrame 的索引作为数组的第一个元素，并将 "Index" 插入到列名列表的第一个位置
        if index:
            arrays.append(self.index)
            fields.insert(0, "Index")

        # 使用整数索引访问列数据，以处理可能存在重复列名的情况
        arrays.extend(self.iloc[:, k] for k in range(len(self.columns)))

        # 如果指定了 name 参数
        if name is not None:
            # 创建一个命名元组，用给定的名称和字段名（可能已重命名）来生成命名元组对象，并返回其迭代器
            itertuple = collections.namedtuple(
                name, fields, rename=True
            )
            return map(itertuple._make, zip(*arrays))

        # 否则返回普通元组的迭代器
        return zip(*arrays)

    def __len__(self) -> int:
        """
        Returns length of info axis, but here we use the index.
        """
        # 返回 DataFrame 的索引的长度
        return len(self.index)

    @overload
    def dot(self, other: Series) -> Series: ...
    # 省略了一些代码，继续注释
    # 定义方法 `dot`，用于计算与另一个 DataFrame、Index 或 ArrayLike 对象的点乘结果，返回 DataFrame 对象
    def dot(self, other: DataFrame | Index | ArrayLike) -> DataFrame: ...

    # 定义方法 `__matmul__` 的重载，用于处理与 Series 对象的矩阵乘法，返回 Series 对象
    @overload
    def __matmul__(self, other: Series) -> Series: ...

    # 定义方法 `__matmul__` 的重载，用于处理与 AnyArrayLike 或 DataFrame 对象的矩阵乘法，返回 DataFrame 或 Series 对象
    @overload
    def __matmul__(self, other: AnyArrayLike | DataFrame) -> DataFrame | Series: ...

    # 实现矩阵乘法运算的方法 `__matmul__`，根据操作符 `@` 调用 `dot` 方法进行计算
    def __matmul__(self, other: AnyArrayLike | DataFrame) -> DataFrame | Series:
        """
        Matrix multiplication using binary `@` operator.
        """
        return self.dot(other)

    # 实现右侧矩阵乘法运算的方法 `__rmatmul__`
    def __rmatmul__(self, other) -> DataFrame:
        """
        Matrix multiplication using binary `@` operator.
        """
        try:
            # 尝试执行右侧矩阵乘法运算，先转置当前对象再与 `other` 进行乘法运算，最后再次转置
            return self.T.dot(np.transpose(other)).T
        except ValueError as err:
            # 捕获值错误异常
            if "shape mismatch" not in str(err):
                # 如果错误消息中不包含 "shape mismatch"，则抛出异常
                raise
            # 如果包含 "shape mismatch"，则生成具体的异常消息，显示原始形状信息
            msg = f"shapes {np.shape(other)} and {self.shape} not aligned"
            raise ValueError(msg) from err

    # ----------------------------------------------------------------------
    # IO methods (to / from other formats)

    # 类方法 `from_dict`，用于从字典数据创建 DataFrame 对象
    @classmethod
    def from_dict(
        cls,
        data: dict,
        orient: FromDictOrient = "columns",
        dtype: Dtype | None = None,
        columns: Axes | None = None,
    # 实例方法 `to_numpy`，用于将 DataFrame 转换为 NumPy 数组
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the DataFrame to a NumPy array.

        By default, the dtype of the returned array will be the common NumPy
        dtype of all types in the DataFrame. For example, if the dtypes are
        ``float16`` and ``float32``, the result's dtype will be ``float32``.
        This may require copying data and coercing values, which can be
        expensive.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensures that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the dtypes of the DataFrame columns.

        Returns
        -------
        numpy.ndarray
            The NumPy array representing the values in the DataFrame.

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
        array([[1, 3],
               [2, 4]])

        With heterogeneous data, the lowest common type will be used.

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
        >>> df.to_numpy()
        array([[1. , 3. ],
               [2. , 4.5]])

        For a mix of numeric and non-numeric types, the output array will
        have object dtype.

        >>> df["C"] = pd.date_range("2000", periods=2)
        >>> df.to_numpy()
        array([[1, 3.0, Timestamp('2000-01-01 00:00:00')],
               [2, 4.5, Timestamp('2000-01-02 00:00:00')]], dtype=object)
        """
        # Check if dtype is specified and convert it to numpy.dtype object
        if dtype is not None:
            dtype = np.dtype(dtype)
        
        # Convert DataFrame to NumPy array using internal _mgr object
        result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
        
        # Ensure the resulting dtype matches the requested dtype
        if result.dtype is not dtype:
            result = np.asarray(result, dtype=dtype)

        return result
    # 定义一个方法 `to_dict`，将对象转换为字典列表格式
    def to_dict(
        self,
        orient: Literal["records"],
        *,
        into: type[dict] = ...,
        index: bool = ...,
    ) -> list[dict]: ...

    # 错误：参数 "into" 的默认值不兼容（默认类型为 "type[dict[Any, Any]]"，而参数类型为 "type[MutableMappingT] | MutableMappingT"）
    def to_dict(
        self,
        orient: Literal[
            "dict", "list", "series", "split", "tight", "records", "index"
        ] = "dict",
        *,
        into: type[MutableMappingT] | MutableMappingT = dict,  # type: ignore[assignment]
        index: bool = True,
    )

    @classmethod
    # 类方法：从记录数组创建 DataFrame 对象
    def from_records(
        cls,
        data,
        index=None,
        exclude=None,
        columns=None,
        coerce_float: bool = False,
        nrows: int | None = None,
    )

    # 实例方法：将 DataFrame 转换为记录数组
    def to_records(
        self, index: bool = True, column_dtypes=None, index_dtypes=None
    )

    @classmethod
    # 私有类方法：从列数组列表创建 DataFrame 对象
    def _from_arrays(
        cls,
        arrays,
        columns,
        index,
        dtype: Dtype | None = None,
        verify_integrity: bool = True,
    ) -> Self:
        """
        从对应列数组列表创建 DataFrame 对象。

        Parameters
        ----------
        arrays : list-like of arrays
            列表中的每个数组依次对应一个列。
        columns : list-like, Index
            结果 DataFrame 的列名。
        index : list-like, Index
            结果 DataFrame 的行标签。
        dtype : dtype, optional
            可选的 dtype，用于强制所有数组。
        verify_integrity : bool, default True
            验证并同质化所有输入。如果设置为 False，则假设 `arrays` 中的所有元素都是实际的数组，
            它们将如何存储在块中（numpy ndarray 或 ExtensionArray），并且长度与索引对齐，
            假定 `columns` 和 `index` 已确保为一个 Index 对象。

        Returns
        -------
        DataFrame
        """
        # 如果指定了 dtype，则转换为 pandas 的 dtype 对象
        if dtype is not None:
            dtype = pandas_dtype(dtype)

        # 确保 columns 是一个 Index 对象
        columns = ensure_index(columns)
        # 如果 columns 的长度不等于 arrays 的长度，则抛出 ValueError 异常
        if len(columns) != len(arrays):
            raise ValueError("len(columns) must match len(arrays)")

        # 将数组转换为数据管理器（mgr）
        mgr = arrays_to_mgr(
            arrays,
            columns,
            index,
            dtype=dtype,
            verify_integrity=verify_integrity,
        )
        # 使用 mgr 创建 DataFrame，并返回
        return cls._from_mgr(mgr, axes=mgr.axes)

    @doc(
        storage_options=_shared_docs["storage_options"],
        compression_options=_shared_docs["compression_options"] % "path",
    )
    def to_stata(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        convert_dates: dict[Hashable, str] | None = None,
        write_index: bool = True,
        byteorder: ToStataByteorder | None = None,
        time_stamp: datetime.datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Hashable, str] | None = None,
        version: int | None = 114,
        convert_strl: Sequence[Hashable] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ):
        """
        Write a DataFrame to Stata format.

        Parameters
        ----------
        path : str, path object, or file-like object
            The location to save the Stata file. Can be a string, path object,
            or file-like object.
        convert_dates : dict, optional
            Dictionary mapping columns to Stata date formats.
        write_index : bool, default True
            Whether to write the DataFrame index.
        byteorder : ToStataByteorder, optional
            Byte order to use for writing the file.
        time_stamp : datetime.datetime, optional
            Time stamp to associate with the dataset.
        data_label : str, optional
            Label for the dataset.
        variable_labels : dict, optional
            Dictionary mapping columns to variable labels.
        version : int, optional
            Stata file version (default is 114).
        convert_strl : sequence, optional
            Sequence of column names to convert to Stata strl format.
        compression : CompressionOptions, default 'infer'
            Compression options for the file.
        storage_options : StorageOptions, optional
            Additional options for file storage.
        value_labels : dict, optional
            Dictionary mapping columns to value labels.

        Notes
        -----
        This function writes the DataFrame to Stata format (.dta).

        See Also
        --------
        DataFrame.to_csv : Write DataFrame to a CSV file.
        DataFrame.to_excel : Write DataFrame to an Excel file.
        DataFrame.to_parquet : Write DataFrame to the Parquet format.
        DataFrame.to_feather : Write DataFrame to the Feather format.
        """
        pass

    def to_feather(self, path: FilePath | WriteBuffer[bytes], **kwargs) -> None:
        """
        Write a DataFrame to the binary Feather format.

        Parameters
        ----------
        path : str, path object, file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. If a string or a path,
            it will be used as Root Directory path when writing a partitioned dataset.
        **kwargs :
            Additional keywords passed to :func:`pyarrow.feather.write_feather`.
            This includes the `compression`, `compression_level`, `chunksize`
            and `version` keywords.

        See Also
        --------
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.
        DataFrame.to_excel : Write object to an Excel sheet.
        DataFrame.to_sql : Write to a sql table.
        DataFrame.to_csv : Write a csv file.
        DataFrame.to_json : Convert the object to a JSON string.
        DataFrame.to_html : Render a DataFrame as an HTML table.
        DataFrame.to_string : Convert DataFrame to a string.

        Notes
        -----
        This function writes the dataframe as a `feather file
        <https://arrow.apache.org/docs/python/feather.html>`_. Requires a default
        index. For saving the DataFrame with your custom index use a method that
        supports custom indices e.g. `to_parquet`.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        >>> df.to_feather("file.feather")  # doctest: +SKIP
        """
        from pandas.io.feather_format import to_feather

        to_feather(self, path, **kwargs)

    @overload
    def to_markdown(
        self,
        buf: None = ...,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions | None = ...,
        **kwargs,
    ) -> str: ...

    @overload
    def to_markdown(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions | None = ...,
        **kwargs,
    ) -> None: ...

    @overload
    # 将数据框架转换为 Markdown 格式的字符串表示
    def to_markdown(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,  # 以下参数为关键字参数，不能直接通过位置传递
        mode: str = "wt",  # 写入模式，默认为文本写入
        index: bool = True,  # 是否包含索引，默认为包含
        storage_options: StorageOptions | None = None,  # 存储选项，用于配置数据的存储细节
        **kwargs,  # 其他关键字参数，用于接收额外的自定义参数
    ) -> str | None:  # 返回值为字符串或空值
    def to_markdown(
        buf: Optional[Union[str, Path, StringIO]] = None,
        mode: str = "wt",
        index: bool = True,
        storage_options: Optional[dict] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Print DataFrame in Markdown-friendly format.
    
        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.
    
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.
    
        **kwargs
            These parameters will be passed to `tabulate <https://pypi.org/project/tabulate>`_.
    
        Returns
        -------
        str
            DataFrame in Markdown-friendly format.
    
        Raises
        ------
        ValueError
            If 'showindex' is found in kwargs (use 'index' instead).
    
        See Also
        --------
        DataFrame.to_html : Render DataFrame to HTML-formatted table.
        DataFrame.to_latex : Render DataFrame to LaTeX-formatted table.
    
        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.
    
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
        ... )
        >>> print(df.to_markdown())
        |    | animal_1   | animal_2   |
        |---:|:-----------|:-----------|
        |  0 | elk        | dog        |
        |  1 | pig        | quetzal    |
    
        Output markdown with a tabulate option.
    
        >>> print(df.to_markdown(tablefmt="grid"))
        +----+------------+------------+
        |    | animal_1   | animal_2   |
        +====+============+============+
        |  0 | elk        | dog        |
        +----+------------+------------+
        |  1 | pig        | quetzal    |
        +----+------------+------------+
        """
        if "showindex" in kwargs:
            raise ValueError("Pass 'index' instead of 'showindex")
    
        # Set default tabulate options
        kwargs.setdefault("headers", "keys")
        kwargs.setdefault("tablefmt", "pipe")
        kwargs.setdefault("showindex", index)
    
        # Import tabulate as an optional dependency
        tabulate = import_optional_dependency("tabulate")
    
        # Generate Markdown table representation of the DataFrame
        result = tabulate.tabulate(self, **kwargs)
    
        # If buf is None, return the Markdown table as a string
        if buf is None:
            return result
    
        # Otherwise, write the Markdown table to the specified buffer
        with get_handle(buf, mode, storage_options=storage_options) as handles:
            handles.handle.write(result)
    
        # Return None if writing to a buffer
        return None
    def to_parquet(
        self,
        path: None = ...,
        *,
        engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
        compression: str | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs,
    ) -> bytes:
        ...

    @overload
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
        compression: str | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs,
    ) -> None:
        ...

    @doc(storage_options=_shared_docs["storage_options"])
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes] | None = None,
        *,
        engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
        compression: str | None = "snappy",
        index: bool | None = None,
        partition_cols: list[str] | None = None,
        storage_options: StorageOptions | None = None,
        **kwargs,
    ) -> None:
        ...

    @overload
    def to_orc(
        self,
        path: None = ...,
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> bytes:
        ...

    @overload
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> None:
        ...

    @overload
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes] | None,
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> bytes | None:
        ...

    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes] | None = None,
        *,
        engine: Literal["pyarrow"] = "pyarrow",
        index: bool | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        ...


注释：

    # 定义一个方法 `to_parquet`，用于将数据写入 Parquet 文件格式
    def to_parquet(
        self,
        path: None = ...,
        *,
        engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
        compression: str | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs,
    ) -> bytes:
        ...

    # `to_parquet` 方法的重载，用于将数据写入 Parquet 文件格式，但返回为空
    @overload
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
        compression: str | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs,
    ) -> None:
        ...

    # `to_parquet` 方法的重载，使用文档注释 `storage_options` 来自共享文档
    @doc(storage_options=_shared_docs["storage_options"])
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes] | None = None,
        *,
        engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
        compression: str | None = "snappy",
        index: bool | None = None,
        partition_cols: list[str] | None = None,
        storage_options: StorageOptions | None = None,
        **kwargs,
    ) -> None:
        ...

    # 定义一个方法 `to_orc`，用于将数据写入 ORC 文件格式
    @overload
    def to_orc(
        self,
        path: None = ...,
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> bytes:
        ...

    # `to_orc` 方法的重载，用于将数据写入 ORC 文件格式，但返回为空
    @overload
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> None:
        ...

    # `to_orc` 方法的重载，用于将数据写入 ORC 文件格式，返回值可能为空
    @overload
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes] | None,
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> bytes | None:
        ...

    # 定义一个方法 `to_orc`，用于将数据写入 ORC 文件格式
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes] | None = None,
        *,
        engine: Literal["pyarrow"] = "pyarrow",
        index: bool | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        ...
    # 定义一个将数据框对象转换为 HTML 格式的方法
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: Axes | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: bool = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool | str = ...,
        decimal: str = ...,
        bold_rows: bool = ...,
        classes: str | list | tuple | None = ...,
        escape: bool = ...,
        notebook: bool = ...,
        border: int | bool | None = ...,
        table_id: str | None = ...,
        render_links: bool = ...,
        encoding: str | None = ...,
    ) -> None:
        ...

    # 另一个方法签名的重载，将数据框对象转换为 HTML 字符串
    @overload
    def to_html(
        self,
        buf: None = ...,
        *,
        columns: Axes | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: bool = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool | str = ...,
        decimal: str = ...,
        bold_rows: bool = ...,
        classes: str | list | tuple | None = ...,
        escape: bool = ...,
        notebook: bool = ...,
        border: int | bool | None = ...,
        table_id: str | None = ...,
        render_links: bool = ...,
        encoding: str | None = ...,
    ) -> str:
        ...

    # 使用装饰器对文档字符串进行参数替换，添加注释
    @Substitution(
        header_type="bool",
        header="Whether to print column labels, default True",
        col_space_type="str or int, list or dict of int or str",
        col_space="The minimum width of each column in CSS length "
                  "units.  An int is assumed to be px units.",
    )
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    # 将数据框（DataFrame）转换为 HTML 格式的字符串

    # buf 参数用于指定输出的目标文件路径或写入缓冲区，可以是文件路径、写入缓冲区对象或 None
    # columns 参数用于指定要包含在输出中的列，可以是轴对象或 None
    # col_space 参数用于指定列之间的空间，可以是列间距参数或 None
    # header 参数指示是否包括表头，默认为 True
    # index 参数指示是否包括行索引，默认为 True
    # na_rep 参数指定在数据中遇到的 NaN 值的表示方式，默认为 "NaN"
    # formatters 参数用于指定格式化函数，可以是格式化函数列表或 None
    # float_format 参数用于指定浮点数格式，可以是浮点格式字符串或 None
    # sparsify 参数指示是否稀疏化输出，默认为 None
    # index_names 参数指示是否包括行索引名称，默认为 True
    # justify 参数用于指定文本对齐方式，可以是对齐方式字符串或 None
    # max_rows 参数用于指定最大输出行数限制，可以是整数或 None
    # max_cols 参数用于指定最大输出列数限制，可以是整数或 None
    # show_dimensions 参数指示是否显示数据框的维度信息，可以是布尔值或字符串，默认为 False
    # decimal 参数用于指定输出浮点数时使用的小数点符号，默认为 "."
    # bold_rows 参数指示是否为行添加粗体样式，默认为 True
    # classes 参数用于指定输出 HTML 表格的 CSS 类，可以是字符串、列表、元组或 None
    # escape 参数指示是否转义特殊字符，默认为 True
    # notebook 参数指示是否在 Notebook 环境中生成输出，默认为 False
    # border 参数用于指定表格的边框样式，可以是整数、布尔值或 None
    # table_id 参数用于指定输出 HTML 表格的 id 属性，可以是字符串或 None
    # render_links 参数指示是否呈现数据中的链接，默认为 False
    # encoding 参数用于指定生成 HTML 的字符编码方式，可以是字符串或 None
    ) -> str | None:
        """
        Render a DataFrame as an HTML table.
        %(shared_params)s
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            `<table>` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        encoding : str, default "utf-8"
            Set character encoding.
        %(returns)s
        See Also
        --------
        to_string : Convert DataFrame to a string.

        Examples
        --------
        >>> df = pd.DataFrame(data={"col1": [1, 2], "col2": [4, 3]})
        >>> html_string = '''<table border="1" class="dataframe">
        ...   <thead>
        ...     <tr style="text-align: right;">
        ...       <th></th>
        ...       <th>col1</th>
        ...       <th>col2</th>
        ...     </tr>
        ...   </thead>
        ...   <tbody>
        ...     <tr>
        ...       <th>0</th>
        ...       <td>1</td>
        ...       <td>4</td>
        ...     </tr>
        ...     <tr>
        ...       <th>1</th>
        ...       <td>2</td>
        ...       <td>3</td>
        ...     </tr>
        ...   </tbody>
        ... </table>'''
        >>> assert html_string == df.to_html()
        """
        # 如果 justify 参数不为 None 并且其值不在有效的格式化参数列表中，则抛出 ValueError
        if justify is not None and justify not in fmt.VALID_JUSTIFY_PARAMETERS:
            raise ValueError("Invalid value for justify parameter")

        # 创建一个 DataFrameFormatter 对象，用于格式化 DataFrame
        formatter = fmt.DataFrameFormatter(
            self,
            columns=columns,
            col_space=col_space,
            na_rep=na_rep,
            header=header,
            index=index,
            formatters=formatters,
            float_format=float_format,
            bold_rows=bold_rows,
            sparsify=sparsify,
            justify=justify,
            index_names=index_names,
            escape=escape,
            decimal=decimal,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
        )
        # 创建一个 DataFrameRenderer 对象并调用其 to_html 方法，将 DataFrame 转换为 HTML 字符串
        return fmt.DataFrameRenderer(formatter).to_html(
            buf=buf,
            classes=classes,
            notebook=notebook,
            border=border,
            encoding=encoding,
            table_id=table_id,
            render_links=render_links,
        )

    @overload
    def to_xml(
        self,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        index: bool = True,
        root_name: str | None = "data",
        row_name: str | None = "row",
        na_rep: str | None = None,
        attr_cols: list[str] | None = None,
        elem_cols: list[str] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = "utf-8",
        xml_declaration: bool | None = True,
        pretty_print: bool | None = True,
        parser: XMLParsers | None = "lxml",
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ) -> str:
    '''
    转换对象数据为 XML 格式字符串。

    Parameters:
    - path_or_buffer: 文件路径或写入缓冲区，允许为 None。
    - index: 是否包含索引信息，默认为 True。
    - root_name: XML 根节点名称，默认为 "data"。
    - row_name: XML 行节点名称，默认为 "row"。
    - na_rep: 用于表示缺失值的字符串，默认为 None。
    - attr_cols: 应该作为 XML 属性的列名列表，默认为 None。
    - elem_cols: 应该作为 XML 元素的列名列表，默认为 None。
    - namespaces: XML 命名空间字典，默认为 None。
    - prefix: XML 命名空间前缀，默认为 None。
    - encoding: XML 编码方式，默认为 "utf-8"。
    - xml_declaration: 是否包含 XML 声明，默认为 True。
    - pretty_print: 是否美化输出，默认为 True。
    - parser: XML 解析器类型，默认为 "lxml"。
    - stylesheet: XML 样式表文件路径或缓冲区，默认为 None。
    - compression: 数据压缩选项，默认为 "infer"。
    - storage_options: 存储选项，用于控制文件读写行为，默认为 None。

    Returns:
    - 转换后的 XML 格式字符串。
    '''



    @overload
    def to_xml(
        self,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        *,
        index: bool = ...,
        root_name: str | None = ...,
        row_name: str | None = ...,
        na_rep: str | None = ...,
        attr_cols: list[str] | None = ...,
        elem_cols: list[str] | None = ...,
        namespaces: dict[str | None, str] | None = ...,
        prefix: str | None = ...,
        encoding: str = ...,
        xml_declaration: bool | None = ...,
        pretty_print: bool | None = ...,
        parser: XMLParsers | None = ...,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions | None = ...,
    ) -> None:
    '''
    转换对象数据为 XML 格式并写入指定的文件或缓冲区。

    Parameters:
    - path_or_buffer: 文件路径或写入缓冲区，不能为空。
    - 其余参数同上述函数。
    '''



    @doc(
        storage_options=_shared_docs["storage_options"],
        compression_options=_shared_docs["compression_options"] % "path_or_buffer",
    )
    def to_xml(
        self,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        index: bool = True,
        root_name: str | None = "data",
        row_name: str | None = "row",
        na_rep: str | None = None,
        attr_cols: list[str] | None = None,
        elem_cols: list[str] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = "utf-8",
        xml_declaration: bool | None = True,
        pretty_print: bool | None = True,
        parser: XMLParsers | None = "lxml",
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ):
    '''
    为 to_xml 方法添加文档字符串，包括存储选项和压缩选项。

    Parameters:
    - storage_options: 存储选项，控制文件读写行为。
    - compression_options: 压缩选项说明，包括路径或缓冲区。

    Note:
    - 此函数是为了提供 to_xml 方法的文档说明和功能扩展。
    '''



    @doc(INFO_DOCSTRING, **frame_sub_kwargs)
    def info(
        self,
        verbose: bool | None = None,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        memory_usage: bool | str | None = None,
        show_counts: bool | None = None,
    ):
    '''
    打印对象的信息摘要。

    Parameters:
    - verbose: 是否详细显示信息，默认为 None。
    - buf: 写入缓冲区，用于存储信息输出。
    - max_cols: 最大列数限制，默认为 None。
    - memory_usage: 是否显示内存使用信息，默认为 None。
    - show_counts: 是否显示计数信息，默认为 None。

    Note:
    - 此函数为对象的信息打印功能，使用了 INFO_DOCSTRING 和 frame_sub_kwargs。
    '''
    ) -> None:
        # 创建一个 DataFrameInfo 对象，用于展示关于数据帧的信息
        info = DataFrameInfo(
            data=self,
            memory_usage=memory_usage,
        )
        # 渲染并显示数据帧信息
        info.render(
            buf=buf,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )

    def transpose(
        self,
        *args,
        copy: bool | lib.NoDefault = lib.no_default,
    @property
    def T(self) -> DataFrame:
        """
        The transpose of the DataFrame.

        Returns
        -------
        DataFrame
            The transposed DataFrame.

        See Also
        --------
        DataFrame.transpose : Transpose index and columns.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4

        >>> df.T
              0  1
        col1  1  2
        col2  3  4
        """
        # 返回当前数据帧的转置
        return self.transpose()

    # ----------------------------------------------------------------------
    # Indexing Methods

    def _ixs(self, i: int, axis: AxisInt = 0) -> Series:
        """
        Parameters
        ----------
        i : int
            行索引号
        axis : int
            轴的方向

        Returns
        -------
        Series
            返回数据帧中指定行或列的数据作为 Series 对象
        """
        # 如果轴是0，获取指定行的数据
        if axis == 0:
            new_mgr = self._mgr.fast_xs(i)

            # 从新的管理器中创建切片数据帧，并指定轴
            result = self._constructor_sliced_from_mgr(new_mgr, axes=new_mgr.axes)
            result._name = self.index[i]
            return result.__finalize__(self)

        # 如果轴是1，获取指定列的数据
        else:
            col_mgr = self._mgr.iget(i)
            return self._box_col_values(col_mgr, i)

    def _get_column_array(self, i: int) -> ArrayLike:
        """
        Get the values of the i'th column (ndarray or ExtensionArray, as stored
        in the Block)

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution (for read-only purposes).
        """
        # 获取第 i 列的值数组（存储在块中的 ndarray 或 ExtensionArray）
        return self._mgr.iget_values(i)

    def _iter_column_arrays(self) -> Iterator[ArrayLike]:
        """
        Iterate over the arrays of all columns in order.
        This returns the values as stored in the Block (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution (for read-only purposes).
        """
        # 依次迭代所有列的数组，按顺序返回块中存储的值（ndarray 或 ExtensionArray）
        for i in range(len(self.columns)):
            yield self._get_column_array(i)
    # 定义特殊方法 __getitem__，用于实现索引操作
    def __getitem__(self, key):
        # 检查索引键是否符合字典或集合的要求，如不符则引发异常
        check_dict_or_set_indexers(key)
        # 将零维度的索引键转换为标量值
        key = lib.item_from_zerodim(key)
        # 如果索引键是可调用的，则应用并获取结果
        key = com.apply_if_callable(key, self)

        # 如果索引键是可散列的，且不是迭代器或切片类型
        if is_hashable(key) and not is_iterator(key) and not isinstance(key, slice):
            # is_iterator用于排除生成器，例如test_getitem_listlike
            # 在 Python 3.12 中，切片也是可散列的，这可能会影响 MultiIndex (GH#57500)

            # 如果索引键存在于列中，则快速返回相应项
            is_mi = isinstance(self.columns, MultiIndex)
            # GH#45316 如果索引键不重复，则返回视图
            # 仅在存在重复值时使用 drop_duplicates 来提升性能
            if not is_mi and (
                self.columns.is_unique
                and key in self.columns
                or key in self.columns.drop_duplicates(keep=False)
            ):
                return self._get_item(key)

            # 如果是 MultiIndex 且索引键存在于列中，则调用 _getitem_multilevel 方法
            elif is_mi and self.columns.is_unique and key in self.columns:
                return self._getitem_multilevel(key)

        # 如果索引键是切片类型，则调用 _getitem_slice 方法处理行的切片操作
        if isinstance(key, slice):
            return self._getitem_slice(key)

        # 如果索引键是 DataFrame 类型，则调用 where 方法处理布尔值 DataFrame
        if isinstance(key, DataFrame):
            return self.where(key)

        # 如果索引键是布尔值数组或 DataFrame，则调用 _getitem_bool_array 处理布尔值索引
        if com.is_bool_indexer(key):
            return self._getitem_bool_array(key)

        # 处理剩余的两种情况：单个索引键或多个索引键集合
        # 对于非 MultiIndex，将元组视为集合处理
        is_single_key = isinstance(key, tuple) or not is_list_like(key)

        if is_single_key:
            # 如果列具有多层级，则调用 _getitem_multilevel 方法处理多层级索引
            if self.columns.nlevels > 1:
                return self._getitem_multilevel(key)
            # 获取单个索引键在列中的位置索引
            indexer = self.columns.get_loc(key)
            # 如果索引位置是整数，则转换为列表形式
            if is_integer(indexer):
                indexer = [indexer]
        else:
            # 如果索引键是迭代器，则转换为列表形式
            if is_iterator(key):
                key = list(key)
            # 获取索引键在列中的位置索引
            indexer = self.columns._get_indexer_strict(key, "columns")[1]

        # 如果索引器的数据类型是布尔型，则转换为布尔数组的索引位置
        if getattr(indexer, "dtype", None) == bool:
            indexer = np.where(indexer)[0]

        # 如果索引器是切片类型，则调用 _slice 方法进行切片操作
        if isinstance(indexer, slice):
            return self._slice(indexer, axis=1)

        # 使用索引器获取数据
        data = self.take(indexer, axis=1)

        if is_single_key:
            # 处理单个索引键在非唯一索引上的情况
            # 行为不一致：它返回 Series，但在以下情况下可能不同：
            # - 键本身重复（见 data.shape，＃9519），或
            # - 列具有 MultiIndex（见 self.columns，＃21309）
            if data.shape[1] == 1 and not isinstance(self.columns, MultiIndex):
                # GH#26490 使用 data[key] 可能导致 RecursionError
                return data._get_item(key)

        # 返回获取的数据
        return data
    def _getitem_bool_array(self, key):
        # 如果 key 是 Series 类型，并且其索引与 DataFrame 的索引不相同，
        # 发出警告，表明布尔类型的 Series 将重新索引以匹配 DataFrame 的索引。
        if isinstance(key, Series) and not key.index.equals(self.index):
            warnings.warn(
                "Boolean Series key will be reindexed to match DataFrame index.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        # 如果 key 的长度与 DataFrame 的索引长度不相等，抛出 ValueError 异常。
        elif len(key) != len(self.index):
            raise ValueError(
                f"Item wrong length {len(key)} instead of {len(self.index)}."
            )

        # 检查布尔索引器，如果 Series key 无法重新索引以匹配 DataFrame 的行，则会抛出异常。
        key = check_bool_indexer(self.index, key)

        # 如果 key 中所有值都为 True，返回当前 DataFrame 的副本（浅拷贝）。
        if key.all():
            return self.copy(deep=False)

        # 找到 key 中非零元素的索引，然后按照这些索引取出 DataFrame 的行。
        indexer = key.nonzero()[0]
        return self.take(indexer, axis=0)

    def _getitem_multilevel(self, key):
        # self.columns 是一个 MultiIndex

        # 获取 key 在 MultiIndex 中的位置 loc
        loc = self.columns.get_loc(key)

        # 如果 loc 是 slice 或者 ndarray 类型
        if isinstance(loc, (slice, np.ndarray)):
            # 获取新的列名 new_columns，去除 key 相关的多级索引
            new_columns = self.columns[loc]
            result_columns = maybe_droplevels(new_columns, key)
            # 取出相应的列数据
            result = self.iloc[:, loc]
            result.columns = result_columns

            # 如果只返回了一个列，并且其列名是空字符串或者元组的第一个元素是空字符串，
            # 则将空字符串视为占位符，并按照用户提供的空字符串的方式返回列。
            if len(result.columns) == 1:
                top = result.columns[0]
                if isinstance(top, tuple):
                    top = top[0]
                if top == "":
                    result = result[""]
                    if isinstance(result, Series):
                        # 如果结果是一个 Series，排除隐含的空字符串，返回具有给定名称的 Series。
                        result = self._constructor_sliced(
                            result, index=self.index, name=key
                        )

            return result
        else:
            # loc 不是 slice 或者 ndarray 类型，必须是一个整数，调用 _ixs 方法进行索引。
            return self._ixs(loc, axis=1)
    def _get_value(self, index, col, takeable: bool = False) -> Scalar:
        """
        Quickly retrieve single value at passed column and index.

        Parameters
        ----------
        index : row label
            行标签，指定要检索的行
        col : column label
            列标签，指定要检索的列
        takeable : interpret the index/col as indexers, default False
            takeable参数，如果为True，则将index/col视为索引器，否则按标签处理

        Returns
        -------
        scalar
            返回单个标量值

        Notes
        -----
        Assumes that both `self.index._index_as_unique` and
        `self.columns._index_as_unique`; Caller is responsible for checking.
        假设 `self.index._index_as_unique` 和 `self.columns._index_as_unique` 都为唯一索引；调用者负责检查。
        """
        if takeable:
            series = self._ixs(col, axis=1)
            return series._values[index]

        series = self._get_item(col)

        if not isinstance(self.index, MultiIndex):
            # CategoricalIndex: Trying to use the engine fastpath may give incorrect
            #  results if our categories are integers that dont match our codes
            # IntervalIndex: IntervalTree has no get_loc
            # 如果索引不是MultiIndex类型，根据行标签获取行索引
            row = self.index.get_loc(index)
            return series._values[row]

        # For MultiIndex going through engine effectively restricts us to
        #  same-length tuples; see test_get_set_value_no_partial_indexing
        # 对于MultiIndex，通过引擎获取行位置
        loc = self.index._engine.get_loc(index)
        return series._values[loc]
    def isetitem(self, loc, value) -> None:
        """
        Set the given value in the column with position `loc`.

        This is a positional analogue to ``__setitem__``.

        Parameters
        ----------
        loc : int or sequence of ints
            Index position for the column.
        value : scalar or arraylike
            Value(s) for the column.

        See Also
        --------
        DataFrame.iloc : Purely integer-location based indexing for selection by
            position.

        Notes
        -----
        ``frame.isetitem(loc, value)`` is an in-place method as it will
        modify the DataFrame in place (not returning a new object). In contrast to
        ``frame.iloc[:, i] = value`` which will try to update the existing values in
        place, ``frame.isetitem(loc, value)`` will not update the values of the column
        itself in place, it will instead insert a new array.

        In cases where ``frame.columns`` is unique, this is equivalent to
        ``frame[frame.columns[i]] = value``.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.isetitem(1, [5, 6])
        >>> df
              A  B
        0     1  5
        1     2  6
        """
        # 如果 value 是 DataFrame 类型
        if isinstance(value, DataFrame):
            # 如果 loc 是整数，则将其转换为列表
            if is_integer(loc):
                loc = [loc]

            # 检查 loc 的长度是否与 value 的列数相等，若不相等则抛出 ValueError
            if len(loc) != len(value.columns):
                raise ValueError(
                    f"Got {len(loc)} positions but value has {len(value.columns)} "
                    f"columns."
                )

            # 遍历 loc 中的索引位置和 value 中的列
            for i, idx in enumerate(loc):
                # 对 value 的第 i 列进行清理和引用处理，返回清理后的 arraylike 和引用 refs
                arraylike, refs = self._sanitize_column(value.iloc[:, i])
                # 调用 _iset_item_mgr 方法插入新的 arraylike 数组到 DataFrame 中的 idx 位置
                self._iset_item_mgr(idx, arraylike, inplace=False, refs=refs)
            return

        # 对单个 value 进行清理和引用处理，返回清理后的 arraylike 和引用 refs
        arraylike, refs = self._sanitize_column(value)
        # 调用 _iset_item_mgr 方法插入新的 arraylike 到 DataFrame 中的 loc 位置
        self._iset_item_mgr(loc, arraylike, inplace=False, refs=refs)
    # 实现 __setitem__ 方法，用于设置对象的项（键值对）
    def __setitem__(self, key, value) -> None:
        # 如果不是在 PyPy 环境下，并且对象的引用计数小于等于3，则发出警告
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        # 将 key 应用于对象自身，如果 key 是可调用的，则将其应用
        key = com.apply_if_callable(key, self)

        # 检查是否可以对行进行切片操作
        if isinstance(key, slice):
            # 将切片索引器转换为实际的索引器，用于获取行数据
            slc = self.index._convert_slice_indexer(key, kind="getitem")
            return self._setitem_slice(slc, value)

        # 如果 key 是 DataFrame 或者具有二维属性的对象，则调用 _setitem_frame 方法
        if isinstance(key, DataFrame) or getattr(key, "ndim", None) == 2:
            self._setitem_frame(key, value)
        # 如果 key 是 Series、ndarray、list 或 Index 对象，则调用 _setitem_array 方法
        elif isinstance(key, (Series, np.ndarray, list, Index)):
            self._setitem_array(key, value)
        # 如果 value 是 DataFrame，则调用 _set_item_frame_value 方法
        elif isinstance(value, DataFrame):
            self._set_item_frame_value(key, value)
        # 如果 value 是列表样式且列不是唯一的，并且长度与 key 的索引器长度相同，则调用 _setitem_array 方法
        elif (
            is_list_like(value)
            and not self.columns.is_unique
            and 1 < len(self.columns.get_indexer_for([key])) == len(value)
        ):
            # 要设置的列是重复的
            self._setitem_array([key], value)
        else:
            # 设置列
            self._set_item(key, value)

    # 处理对行的切片设置
    def _setitem_slice(self, key: slice, value) -> None:
        # 注意：不能简单地使用 self.loc[key] = value，因为这将基于标签进行操作，我们需要基于位置进行操作以保持向后兼容性，参考 GH#31469
        self.iloc[key] = value

    # 处理对数组样式的键设置
    def _setitem_array(self, key, value) -> None:
        # 如果 key 是布尔索引器，则沿着行进行索引
        if com.is_bool_indexer(key):
            # 如果索引器长度与索引的长度不一致，则引发 ValueError
            if len(key) != len(self.index):
                raise ValueError(
                    f"Item wrong length {len(key)} instead of {len(self.index)}!"
                )
            # 检查并转换布尔索引器，获取有效的索引
            key = check_bool_indexer(self.index, key)
            # 获取布尔索引器中为真的位置
            indexer = key.nonzero()[0]
            # 如果 value 是 DataFrame，则重新索引它以便与当前对象的索引对齐
            if isinstance(value, DataFrame):
                value = value.reindex(self.index.take(indexer))
            # 使用有效索引设置值
            self.iloc[indexer] = value

        else:
            # 注意：与 self.iloc[:, indexer] = value 不同，这里永远不会尝试原地重写值

            # 如果 value 是 DataFrame，则检查 key 和 value 的长度并依次设置
            if isinstance(value, DataFrame):
                check_key_length(self.columns, key, value)
                for k1, k2 in zip(key, value.columns):
                    self[k1] = value[k2]

            # 如果 value 不是列表样式，则依次为 key 中的每列设置值
            elif not is_list_like(value):
                for col in key:
                    self[col] = value

            # 如果 value 是二维 ndarray，则调用 _iset_not_inplace 方法设置值
            elif isinstance(value, np.ndarray) and value.ndim == 2:
                self._iset_not_inplace(key, value)

            # 如果 value 的维度大于1，则将其转换为 DataFrame 的值并递归调用 _setitem_array 方法
            elif np.ndim(value) > 1:
                # 列表的列表情况
                value = DataFrame(value).values
                self._setitem_array(key, value)

            else:
                # 否则，调用 _iset_not_inplace 方法设置值
                self._iset_not_inplace(key, value)
    def _iset_not_inplace(self, key, value) -> None:
        # 当使用 df[key] = obj 形式设置数据时，如果 key 和 value 都是类似列表的结构，
        # 则逐个设置列数据。这与使用 `self.loc[:, key] = value` 不同，
        # 因为 loc.__setitem__ 可能会原地覆盖数据，而这里会插入新的数组。

        def igetitem(obj, i: int):
            # 注意：在到达这里之前我们捕获了 DataFrame obj，但假设在这里会返回 obj.iloc[:, i]
            if isinstance(obj, np.ndarray):
                return obj[..., i]
            else:
                return obj[i]

        if self.columns.is_unique:
            if np.shape(value)[-1] != len(key):
                raise ValueError("Columns must be same length as key")

            for i, col in enumerate(key):
                self[col] = igetitem(value, i)

        else:
            ilocs = self.columns.get_indexer_non_unique(key)[0]
            if (ilocs < 0).any():
                # key 中的条目不在 self.columns 中
                raise NotImplementedError

            if np.shape(value)[-1] != len(ilocs):
                raise ValueError("Columns must be same length as key")

            assert np.ndim(value) <= 2

            orig_columns = self.columns

            # 使用 self.iloc[:, i] = ... 可能会原地设置值，
            # 但按照惯例我们在 __setitem__ 中不这样做
            try:
                self.columns = Index(range(len(self.columns)))
                for i, iloc in enumerate(ilocs):
                    self[iloc] = igetitem(value, i)
            finally:
                self.columns = orig_columns

    def _setitem_frame(self, key, value) -> None:
        # 支持使用 DataFrame 输入进行布尔设置，例如 df[df > df2] = 0
        if isinstance(key, np.ndarray):
            if key.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
            key = self._constructor(key, **self._construct_axes_dict(), copy=False)

        if key.size and not all(is_bool_dtype(dtype) for dtype in key.dtypes):
            raise TypeError(
                "Must pass DataFrame or 2-d ndarray with boolean values only"
            )

        self._where(-key, value, inplace=True)
    # 设置 DataFrame 中指定键（列名）的数值，确保值是有效的索引
    def _set_item_frame_value(self, key, value: DataFrame) -> None:
        self._ensure_valid_index(value)

        # 如果键已经存在于 DataFrame 的列中
        if key in self.columns:
            # 获取键在列中的位置
            loc = self.columns.get_loc(key)
            # 获取该位置的列或列组
            cols = self.columns[loc]
            # 确定 cols 的长度，如果是标量或元组则长度为1，否则为实际列数
            len_cols = 1 if is_scalar(cols) or isinstance(cols, tuple) else len(cols)
            # 如果传入的 DataFrame 的列数与当前键对应的列数不同，抛出值错误
            if len_cols != len(value.columns):
                raise ValueError("Columns must be same length as key")

            # 如果 DataFrame 的列是多级索引并且 self[key] 是一个子框架
            if isinstance(self.columns, MultiIndex) and isinstance(
                loc, (slice, Series, np.ndarray, Index)
            ):
                # 可能去除 cols 的级别
                cols_droplevel = maybe_droplevels(cols, key)
                # 如果去除级别后的 cols 不等于传入 DataFrame 的列，则重新索引值
                if len(cols_droplevel) and not cols_droplevel.equals(value.columns):
                    value = value.reindex(cols_droplevel, axis=1)

                # 逐列更新 self 中的值
                for col, col_droplevel in zip(cols, cols_droplevel):
                    self[col] = value[col_droplevel]
                return

            # 如果 cols 是标量，则直接使用 value 的第一列值进行赋值
            if is_scalar(cols):
                self[cols] = value[value.columns[0]]
                return

            # 如果 loc 是切片，则获取 loc 的索引数组
            locs: np.ndarray | list
            if isinstance(loc, slice):
                locs = np.arange(loc.start, loc.stop, loc.step)
            elif is_scalar(loc):
                locs = [loc]
            else:
                locs = loc.nonzero()[0]

            # 调用 self 的 isetitem 方法来设置值
            return self.isetitem(locs, value)

        # 如果传入的 DataFrame 列数大于1，则抛出值错误
        if len(value.columns) > 1:
            raise ValueError(
                "Cannot set a DataFrame with multiple columns to the single "
                f"column {key}"
            )
        # 如果传入的 DataFrame 没有列，则抛出值错误
        elif len(value.columns) == 0:
            raise ValueError(
                f"Cannot set a DataFrame without columns to the column {key}"
            )

        # 否则，直接将传入 DataFrame 的第一列值赋给 self[key]
        self[key] = value[value.columns[0]]

    # 设置 DataFrame 的管理器中的项
    def _iset_item_mgr(
        self,
        loc: int | slice | np.ndarray,
        value,
        inplace: bool = False,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        # 当从 _set_item_mgr 调用时，loc 可以是从 get_loc 返回的任何值
        self._mgr.iset(loc, value, inplace=inplace, refs=refs)

    # 设置 DataFrame 的管理器中的项
    def _set_item_mgr(
        self, key, value: ArrayLike, refs: BlockValuesRefs | None = None
    ) -> None:
        try:
            # 尝试获取 key 在信息轴上的位置
            loc = self._info_axis.get_loc(key)
        except KeyError:
            # 如果 key 不存在，则直接在末尾插入新项
            self._mgr.insert(len(self._info_axis), key, value, refs)
        else:
            # 否则，调用 _iset_item_mgr 设置项
            self._iset_item_mgr(loc, value, refs=refs)

    # 设置 DataFrame 中指定位置的项
    def _iset_item(self, loc: int, value: Series, inplace: bool = True) -> None:
        # 只从 _replace_columnwise 调用，确保无需重新索引
        self._iset_item_mgr(loc, value._values, inplace=inplace, refs=value._references)
    def _set_item(self, key, value) -> None:
        """
        Add series to DataFrame in specified column.

        If series is a numpy-array (not a Series/TimeSeries), it must be the
        same length as the DataFrames index or an error will be thrown.

        Series/TimeSeries will be conformed to the DataFrames index to
        ensure homogeneity.
        """
        # 对值进行清理和参考信息提取
        value, refs = self._sanitize_column(value)

        # 如果列名已存在且值是一维数组且不是扩展类型
        if (
            key in self.columns
            and value.ndim == 1
            and not isinstance(value.dtype, ExtensionDtype)
        ):
            # 如果列名不唯一或者是多重索引，则进行跨列广播
            if not self.columns.is_unique or isinstance(self.columns, MultiIndex):
                existing_piece = self[key]
                # 如果已存在的片段是 DataFrame，则将值广播到多列
                if isinstance(existing_piece, DataFrame):
                    value = np.tile(value, (len(existing_piece.columns), 1)).T
                    refs = None

        # 使用管理器设置项目
        self._set_item_mgr(key, value, refs)

    def _set_value(
        self, index: IndexLabel, col, value: Scalar, takeable: bool = False
    ) -> None:
        """
        Put single value at passed column and index.

        Parameters
        ----------
        index : Label
            row label
        col : Label
            column label
        value : scalar
        takeable : bool, default False
            Sets whether or not index/col interpreted as indexers
        """
        try:
            if takeable:
                # 如果指定为可接受索引，则直接使用传入的索引和列
                icol = col
                iindex = cast(int, index)
            else:
                # 否则，获取列的位置和索引的位置
                icol = self.columns.get_loc(col)
                iindex = self.index.get_loc(index)
            # 在管理器中设置列值，仅限于原地修改
            self._mgr.column_setitem(icol, iindex, value, inplace_only=True)

        except (KeyError, TypeError, ValueError, LossySetitemError):
            # 如果 get_loc 抛出 KeyError（标签缺失），则使用 (i)loc 进行索引扩展
            # column_setitem 将进行验证，可能引发 TypeError、ValueError 或 LossySetitemError
            # 使用非递归方法设置并重置缓存
            if takeable:
                self.iloc[index, col] = value
            else:
                self.loc[index, col] = value

        except InvalidIndexError as ii_err:
            # GH48729: 似乎您正尝试在只允许标量选项时向行赋值
            raise InvalidIndexError(
                f"You can only assign a scalar value not a {type(value)}"
            ) from ii_err
    # 确保如果没有索引，可以从传入的值创建一个索引
    def _ensure_valid_index(self, value) -> None:
        """
        Ensure that if we don't have an index, that we can create one from the
        passed value.
        """
        # GH5632, 确保我们可以将其转换为 Series
        if not len(self.index) and is_list_like(value) and len(value):
            # 如果值是列表形式并且非空，且不是 DataFrame 类型，则尝试转换为 Series
            if not isinstance(value, DataFrame):
                try:
                    value = Series(value)
                except (ValueError, NotImplementedError, TypeError) as err:
                    # 如果转换失败，则抛出 ValueError
                    raise ValueError(
                        "Cannot set a frame with no defined index "
                        "and a value that cannot be converted to a Series"
                    ) from err

            # GH31368 保持索引的名称
            index_copy = value.index.copy()
            if self.index.name is not None:
                index_copy.name = self.index.name

            # 使用复制后的索引重新索引化列数据，用 NaN 填充
            self._mgr = self._mgr.reindex_axis(index_copy, axis=1, fill_value=np.nan)

    # 将列的值装箱为 Series 对象
    def _box_col_values(self, values: SingleBlockManager, loc: int) -> Series:
        """
        Provide boxed values for a column.
        """
        # 从列的位置 loc 获取列的名称
        name = self.columns[loc]
        # 由于 values 是 SingleBlockManager，使用 _constructor_sliced_from_mgr 函数创建对象
        obj = self._constructor_sliced_from_mgr(values, axes=values.axes)
        # 设置对象的名称为列的名称
        obj._name = name
        # 返回最终的对象，确保与当前对象关联
        return obj.__finalize__(self)

    # 获取指定项（item）对应的 Series 对象
    def _get_item(self, item: Hashable) -> Series:
        # 获取指定项 item 在列中的位置 loc
        loc = self.columns.get_loc(item)
        # 根据 loc 和轴向 1 获取相应的切片对象，返回作为 Series
        return self._ixs(loc, axis=1)

    # ----------------------------------------------------------------------
    # 未排序的方法

    # 以下是 query 方法的函数重载
    @overload
    def query(
        self, expr: str, *, inplace: Literal[False] = ..., **kwargs
    ) -> DataFrame: ...
    
    @overload
    def query(self, expr: str, *, inplace: Literal[True], **kwargs) -> None: ...

    @overload
    def query(
        self, expr: str, *, inplace: bool = ..., **kwargs
    ) -> DataFrame | None: ...

    # 以下是 eval 方法的函数重载
    @overload
    def eval(self, expr: str, *, inplace: Literal[False] = ..., **kwargs) -> Any: ...

    @overload
    def eval(self, expr: str, *, inplace: Literal[True], **kwargs) -> None: ...

    # 在指定位置 loc 插入列 column，赋值为 value，并允许重复值
    def insert(
        self,
        loc: int,
        column: Hashable,
        value: object,
        allow_duplicates: bool | lib.NoDefault = lib.no_default,
    ) -> None:
        """
        Insert column into DataFrame at specified location.

        Raises a ValueError if `column` is already contained in the DataFrame,
        unless `allow_duplicates` is set to True.

        Parameters
        ----------
        loc : int
            Insertion index. Must verify 0 <= loc <= len(columns).
        column : str, number, or hashable object
            Label of the inserted column.
        value : Scalar, Series, or array-like
            Content of the inserted column.
        allow_duplicates : bool, optional, default lib.no_default
            Allow duplicate column labels to be created.

        See Also
        --------
        Index.insert : Insert new item by index.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4
        >>> df.insert(1, "newcol", [99, 99])
        >>> df
           col1  newcol  col2
        0     1      99     3
        1     2      99     4
        >>> df.insert(0, "col1", [100, 100], allow_duplicates=True)
        >>> df
           col1  col1  newcol  col2
        0   100     1      99     3
        1   100     2      99     4

        Notice that pandas uses index alignment in case of `value` from type `Series`:

        >>> df.insert(0, "col0", pd.Series([5, 6], index=[1, 2]))
        >>> df
           col0  col1  col1  newcol  col2
        0   NaN   100     1      99     3
        1   5.0   100     2      99     4
        """
        # 如果 allow_duplicates 没有指定，则默认为 False
        if allow_duplicates is lib.no_default:
            allow_duplicates = False
        # 如果 allow_duplicates 为 True 且不允许重复标签，则引发 ValueError
        if allow_duplicates and not self.flags.allows_duplicate_labels:
            raise ValueError(
                "Cannot specify 'allow_duplicates=True' when "
                "'self.flags.allows_duplicate_labels' is False."
            )
        # 如果不允许重复且 column 已存在于 self.columns 中，则引发 ValueError
        if not allow_duplicates and column in self.columns:
            raise ValueError(f"cannot insert {column}, already exists")
        # 如果 loc 不是整数类型，则引发 TypeError
        if not is_integer(loc):
            raise TypeError("loc must be int")
        # 将 loc 转换为整数以满足类型检查
        loc = int(loc)
        # 如果 value 是 DataFrame 且列数大于 1，则引发 ValueError
        if isinstance(value, DataFrame) and len(value.columns) > 1:
            raise ValueError(
                f"Expected a one-dimensional object, got a DataFrame with "
                f"{len(value.columns)} columns instead."
            )
        # 如果 value 是 DataFrame，则仅取其第一列进行插入
        elif isinstance(value, DataFrame):
            value = value.iloc[:, 0]

        # 对插入的 value 进行清理和引用处理
        value, refs = self._sanitize_column(value)
        # 在内部数据管理器中插入新列
        self._mgr.insert(loc, column, value, refs=refs)
    def assign(self, **kwargs) -> DataFrame:
        r"""
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable or Series}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though pandas doesn't check it).
            If the values are not callable, (e.g. a Series, scalar, or array),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible.
        Later items in '\*\*kwargs' may refer to newly created or modified
        columns in 'df'; items are computed and assigned into 'df' in order.

        Examples
        --------
        >>> df = pd.DataFrame({"temp_c": [17.0, 25.0]}, index=["Portland", "Berkeley"])
        >>> df
                  temp_c
        Portland    17.0
        Berkeley    25.0

        Where the value is a callable, evaluated on `df`:

        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence:

        >>> df.assign(temp_f=df["temp_c"] * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        You can create multiple columns within the same assign where one
        of the columns depends on another one defined within the same assign:

        >>> df.assign(
        ...     temp_f=lambda x: x["temp_c"] * 9 / 5 + 32,
        ...     temp_k=lambda x: (x["temp_f"] + 459.67) * 5 / 9,
        ... )
                  temp_c  temp_f  temp_k
        Portland    17.0    62.6  290.15
        Berkeley    25.0    77.0  298.15
        """
        # Create a shallow copy of the DataFrame to avoid modifying the original data
        data = self.copy(deep=False)

        # Iterate over each key-value pair in kwargs
        for k, v in kwargs.items():
            # Apply the callable function v to the DataFrame data and assign the result to column k
            data[k] = com.apply_if_callable(v, data)
        
        # Return the modified DataFrame with new columns added
        return data
    # 确保值有效，例如索引
    self._ensure_valid_index(value)

    # 如果值是 DataFrame，则断言不是
    assert not isinstance(value, DataFrame)
    
    # 如果值是类似字典的结构
    if is_dict_like(value):
        # 如果值不是 Series，则转换为 Series
        if not isinstance(value, Series):
            value = Series(value)
        # 调用 _reindex_for_setitem 函数，重新索引值，并返回处理后的结果
        return _reindex_for_setitem(value, self.index)
    
    # 如果值类似于列表
    if is_list_like(value):
        # 要求长度匹配索引
        com.require_length_match(value, self.index)
    
    # 对值进行数组的清洗和转换，确保返回一个数组及其对应的 BlockValuesRefs（若有）
    return sanitize_array(value, self.index, copy=True, allow_2d=True), None

@property
def _series(self):
    # 返回一个字典，其中包含每列的索引值
    return {item: self._ixs(idx, axis=1) for idx, item in enumerate(self.columns)}

# ----------------------------------------------------------------------
# 重新索引和对齐

def _reindex_multi(self, axes: dict[str, Index], fill_value) -> DataFrame:
    """
    确保轴上没有空值。
    """
    
    # 重新索引行，并返回新的索引和行索引器
    new_index, row_indexer = self.index.reindex(axes["index"])
    
    # 重新索引列，并返回新的列标签和列索引器
    new_columns, col_indexer = self.columns.reindex(axes["columns"])
    
    # 如果行索引器和列索引器都不为空
    if row_indexer is not None and col_indexer is not None:
        # 快速路径。通过同时执行两次 'take'，避免不必要的复制。
        # 我们只有在 `self._can_fast_transpose` 为真时才会执行到这里，这几乎确保了 self.values 的便宜性。
        # 可能值得将此条件更加具体化。
        indexer = row_indexer, col_indexer
        # 使用 take_2d_multi 函数，基于索引器在两个维度上重新索引值
        new_values = take_2d_multi(self.values, indexer, fill_value=fill_value)
        # 返回一个新的 DataFrame 对象，不复制数据
        return self._constructor(
            new_values, index=new_index, columns=new_columns, copy=False
        )
    else:
        # 否则，使用 _reindex_with_indexers 方法进行重新索引，基于给定的索引器和填充值
        return self._reindex_with_indexers(
            {0: [new_index, row_indexer], 1: [new_columns, col_indexer]},
            fill_value=fill_value,
        )

@Appender(
    """
    示例
    --------
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    更改行标签。

    >>> df.set_axis(['a', 'b', 'c'], axis='index')
       A  B
    a  1  4
    b  2  5
    c  3  6

    更改列标签。

    >>> df.set_axis(['I', 'II'], axis='columns')
       I  II
    0  1   4
    1  2   5
    2  3   6
    """
)
    @Substitution(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
        extended_summary_sub=" column or",
        axis_description_sub=", and 1 identifies the columns",
        see_also_sub=" or columns",
    )
    # 使用 Substitution 装饰器，替换文档字符串中的特定关键字参数，以生成文档
    @Appender(NDFrame.set_axis.__doc__)
    # 使用 Appender 装饰器，将 set_axis 方法的文档字符串追加到当前方法的文档中
    def set_axis(
        self,
        labels,
        *,
        axis: Axis = 0,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> DataFrame:
        return super().set_axis(labels, axis=axis, copy=copy)
        # 调用父类的 set_axis 方法，并返回结果 DataFrame 对象

    @doc(
        NDFrame.reindex,
        klass=_shared_doc_kwargs["klass"],
        optional_reindex=_shared_doc_kwargs["optional_reindex"],
    )
    # 使用 doc 装饰器，关联当前方法与 reindex 方法的文档，并传入相关参数
    def reindex(
        self,
        labels=None,
        *,
        index=None,
        columns=None,
        axis: Axis | None = None,
        method: ReindexMethod | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        level: Level | None = None,
        fill_value: Scalar | None = np.nan,
        limit: int | None = None,
        tolerance=None,
    ) -> DataFrame:
        return super().reindex(
            labels=labels,
            index=index,
            columns=columns,
            axis=axis,
            method=method,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
            copy=copy,
        )
        # 调用父类的 reindex 方法，并返回结果 DataFrame 对象

    @overload
    # 定义方法的重载，用于类型提示和文档生成
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...

    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame: ...

    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame | None: ...
    # 定义 drop 方法的多个重载，分别对应不同的参数组合和返回类型

    def drop(
        self,
        labels: IndexLabel | ListLike = None,
        *,
        axis: Axis = 0,
        index: IndexLabel | ListLike = None,
        columns: IndexLabel | ListLike = None,
        level: Level | None = None,
        inplace: bool = False,
        errors: IgnoreRaise = "raise",
    ):
    # drop 方法的实际定义，包含参数注释和默认值设定
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,  # 以下为关键字参数，不能被位置参数覆盖
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: Literal[True],  # 标志是否在原地修改，默认为True
        level: Level = ...,  # 操作的索引级别，默认为...
        errors: IgnoreRaise = ...,  # 指定遇到错误时的处理方式，默认为忽略错误
    ) -> None:  # 方法返回类型为None，即不返回任何值
        ...

    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: Literal[False] = ...,  # 标志是否在原地修改，默认为False
        level: Level = ...,  # 操作的索引级别，默认为...
        errors: IgnoreRaise = ...,  # 指定遇到错误时的处理方式，默认为忽略错误
    ) -> DataFrame:  # 方法返回类型为DataFrame对象
        ...

    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = ...,  # 标志是否在原地修改，默认为False
        level: Level = ...,  # 操作的索引级别，默认为...
        errors: IgnoreRaise = ...,  # 指定遇到错误时的处理方式，默认为忽略错误
    ) -> DataFrame | None:  # 方法返回类型为DataFrame对象或None
        ...

    def rename(
        self,
        mapper: Renamer | None = None,
        *,  # 以下为关键字参数，不能被位置参数覆盖
        index: Renamer | None = None,
        columns: Renamer | None = None,
        axis: Axis | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = False,  # 标志是否在原地修改，默认为False
        level: Level | None = None,  # 操作的索引级别，默认为None
        errors: IgnoreRaise = "ignore",  # 指定遇到错误时的处理方式，默认为忽略错误
    ):
        """
        重命名DataFrame的索引或列。

        Parameters
        ----------
        mapper : Renamer或None，可选
            用于重命名的映射器，如果为None则不重命名。
        index : Renamer或None，可选
            用于重命名索引的映射器，如果为None则不重命名。
        columns : Renamer或None，可选
            用于重命名列的映射器，如果为None则不重命名。
        axis : Axis或None，可选
            指定操作轴，如果为None则根据mapper进行选择。
        copy : bool或lib.NoDefault，可选
            是否复制DataFrame，默认为lib.no_default。
        inplace : bool，可选
            是否在原地修改，默认为False。
        level : Level或None，可选
            操作的索引级别，默认为None。
        errors : IgnoreRaise，可选
            指定遇到错误时的处理方式，默认为"ignore"。

        Returns
        -------
        DataFrame或None
            如果inplace为True，则返回None；否则返回重命名后的DataFrame副本。

        Notes
        -----
        该方法允许根据提供的映射器或名称重命名DataFrame的索引或列。
        """
        ...

    def pop(self, item: Hashable) -> Series:
        """
        从DataFrame中弹出指定的列，并返回弹出的列作为Series。

        Parameters
        ----------
        item : label
            要弹出的列的标签。

        Returns
        -------
        Series
            表示被弹出的列的Series对象。

        See Also
        --------
        DataFrame.drop: 从行或列中删除指定的标签。
        DataFrame.drop_duplicates: 返回删除重复行后的DataFrame。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("falcon", "bird", 389.0),
        ...         ("parrot", "bird", 24.0),
        ...         ("lion", "mammal", 80.5),
        ...         ("monkey", "mammal", np.nan),
        ...     ],
        ...     columns=("name", "class", "max_speed"),
        ... )
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        >>> df.pop("class")
        0      bird
        1      bird
        2    mammal
        3    mammal
        Name: class, dtype: object

        >>> df
             name  max_speed
        0  falcon      389.0
        1  parrot       24.0
        2    lion       80.5
        3  monkey        NaN
        """
        return super().pop(item=item)

    @overload
    def _replace_columnwise(
        self, mapping: dict[Hashable, tuple[Any, Any]], inplace: Literal[True], regex
    ) -> None:
        """
        Replace values column-wise in the DataFrame using specified mapping.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping column names to tuples (target, value).
        inplace : bool
            Whether to modify the DataFrame inplace (always True for this signature).
        regex : bool or same types as `to_replace` in DataFrame.replace
            Whether to interpret the keys in the mapping as regular expressions.

        Returns
        -------
        None
        """
        # Determine if operation is inplace or requires a copy
        res = self if inplace else self.copy(deep=False)
        
        # Get column labels
        ax = self.columns

        # Iterate over columns
        for i, ax_value in enumerate(ax):
            if ax_value in mapping:
                # Select column as a Series
                ser = self.iloc[:, i]

                # Extract target and value to replace
                target, value = mapping[ax_value]

                # Replace values in the Series
                newobj = ser.replace(target, value, regex=regex)

                # Set the modified Series back into the DataFrame
                res._iset_item(i, newobj, inplace=inplace)

        # If inplace operation, return None; otherwise, finalize and return a new DataFrame
        if inplace:
            return None
        return res.__finalize__(self)

    @doc(NDFrame.shift, klass=_shared_doc_kwargs["klass"])
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq: Frequency | None = None,
        axis: Axis = 0,
        fill_value: Hashable = lib.no_default,
        suffix: str | None = None,
    ) -> DataFrame | None:
        """
        Shift index by desired number of periods with optional time frequency.

        Parameters
        ----------
        periods : int or sequence of int
            Number of periods to shift. If sequence, shifts each index by the corresponding value.
        freq : str or DateOffset, optional
            Offset to use from the pandas time series API (e.g., 'M' for month-end).
        axis : int or str, default 0
            Axis to shift along.
        fill_value : scalar, optional
            Fill value for newly introduced missing values.
        suffix : str, optional
            Suffix to add to column names.

        Returns
        -------
        DataFrame or None
            Shifted DataFrame or None if inplace operation is performed.
        """
        pass  # Implementation details are not provided in the snippet.

    @overload
    def set_index(
        self,
        keys,
        *,
        drop: bool = ...,
        append: bool = ...,
        inplace: Literal[False] = ...,
        verify_integrity: bool = ...,
    ) -> DataFrame:
        ...

    @overload
    def set_index(
        self,
        keys,
        *,
        drop: bool = ...,
        append: bool = ...,
        inplace: Literal[True],
        verify_integrity: bool = ...,
    ) -> None:
        ...

    def set_index(
        self,
        keys,
        *,
        drop: bool = True,
        append: bool = False,
        inplace: bool = False,
        verify_integrity: bool = False,
    ) -> DataFrame | None:
        """
        Set the DataFrame index using existing columns.

        Parameters
        ----------
        keys : label or list of labels
            Column label or list of column labels to use as index.
        drop : bool, default True
            Whether to drop columns used for indexing from DataFrame.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            Modify the DataFrame inplace or return a new DataFrame.
        verify_integrity : bool, default False
            Check the new index for duplicates.

        Returns
        -------
        DataFrame or None
            DataFrame with new index or None if inplace operation is performed.
        """
        pass  # Implementation details are not provided in the snippet.

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        inplace: Literal[False] = ...,
        col_level: Hashable = ...,
        col_fill: Hashable = ...,
        allow_duplicates: bool | lib.NoDefault = ...,
        names: Hashable | Sequence[Hashable] | None = None,
    ) -> DataFrame:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        inplace: Literal[True],
        col_level: Hashable = ...,
        col_fill: Hashable = ...,
        allow_duplicates: bool | lib.NoDefault = ...,
        names: Hashable | Sequence[Hashable] | None = None,
    ) -> None:
        ...

    def reset_index(
        self,
        level: IndexLabel = None,
        *,
        drop: bool = False,
        inplace: bool = False,
        col_level: Hashable = 0,
        col_fill: Hashable | None = '',
        allow_duplicates: bool | lib.NoDefault = False,
        names: Hashable | Sequence[Hashable] | None = None,
    ) -> DataFrame | None:
        """
        Reset index of the DataFrame, moving index labels to columns.

        Parameters
        ----------
        level : int, str, tuple, or list, optional
            Level(s) of index to be removed (can be a single level or list of levels).
        drop : bool, default False
            Whether to drop the removed index levels from DataFrame.
        inplace : bool, default False
            Modify the DataFrame inplace or return a new DataFrame.
        col_level : int or str, default 0
            If the columns have multiple levels, determine which level is reset.
        col_fill : scalar or None, default ''
            Value to use when inserting new columns after resetting index.
        allow_duplicates : bool, default False
            Whether to allow duplicate index entries.
        names : label or list of labels, optional
            Names to use for the resulting columns when the columns have multiple levels.

        Returns
        -------
        DataFrame or None
            DataFrame with reset index or None if inplace operation is performed.
        """
        pass  # Implementation details are not provided in the snippet.
    # ----------------------------------------------------------------------
    # 重新索引为基础的选择方法

    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    # 使用文档字符串 NDFrame.isna，应用于当前类（_shared_doc_kwargs["klass"]）
    def isna(self) -> DataFrame:
        # 调用底层管理器的 isna 方法，生成结果管理器
        res_mgr = self._mgr.isna(func=isna)
        # 从结果管理器构造新的对象，使用相同的轴
        result = self._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        # 用当前对象进行最终初始化，并标记为 isna 方法
        return result.__finalize__(self, method="isna")

    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    # 使用文档字符串 NDFrame.isna，应用于当前类（_shared_doc_kwargs["klass"]）
    def isnull(self) -> DataFrame:
        """
        DataFrame.isnull is an alias for DataFrame.isna.
        """
        # 返回 isna 方法的结果
        return self.isna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    # 使用文档字符串 NDFrame.notna，应用于当前类（_shared_doc_kwargs["klass"]）
    def notna(self) -> DataFrame:
        # 返回非空值的逻辑非
        return ~self.isna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    # 使用文档字符串 NDFrame.notna，应用于当前类（_shared_doc_kwargs["klass"]）
    def notnull(self) -> DataFrame:
        """
        DataFrame.notnull is an alias for DataFrame.notna.
        """
        # 返回 notna 方法的结果
        return ~self.isna()

    @overload
    # 方法的重载声明

    def dropna(
        self,
        *,
        axis: Axis = ...,
        how: AnyAll | lib.NoDefault = ...,
        thresh: int | lib.NoDefault = ...,
        subset: IndexLabel = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...,
    ) -> DataFrame:
    # 返回 DataFrame 对象

    @overload
    # 方法的重载声明

    def dropna(
        self,
        *,
        axis: Axis = ...,
        how: AnyAll | lib.NoDefault = ...,
        thresh: int | lib.NoDefault = ...,
        subset: IndexLabel = ...,
        inplace: Literal[True],
        ignore_index: bool = ...,
    ) -> None:
    # 不返回任何内容

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll | lib.NoDefault = lib.no_default,
        thresh: int | lib.NoDefault = lib.no_default,
        subset: IndexLabel | AnyArrayLike | None = None,
        inplace: bool = False,
        ignore_index: bool = False,
    # ----------------------------------------------------------------------
    # 删除重复值

    @overload
    # 方法的重载声明

    def drop_duplicates(
        self,
        subset: Hashable | Sequence[Hashable] | None = ...,
        *,
        keep: DropKeep = ...,
        inplace: Literal[True],
        ignore_index: bool = ...,
    ) -> None:
    # 不返回任何内容

    @overload
    # 方法的重载声明

    def drop_duplicates(
        self,
        subset: Hashable | Sequence[Hashable] | None = ...,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...,
    ) -> DataFrame:
    # 返回 DataFrame 对象
    # 定义在 DataFrame 上操作的方法，用于去除重复的行。
    def drop_duplicates(
        self,
        subset: Hashable | Sequence[Hashable] | None = None,
        *,
        keep: DropKeep = "first",  # 确定保留重复行中的哪些实例
        inplace: bool = False,  # 是否在原地修改 DataFrame
        ignore_index: bool = False,  # 是否忽略索引重新排序
    ) -> DataFrame | None: ...
    
    # 定义在 DataFrame 上操作的方法，用于检测和标记重复的行。
    def duplicated(
        self,
        subset: Hashable | Sequence[Hashable] | None = None,
        keep: DropKeep = "first",  # 确定保留重复行中的哪些实例
    ) -> Series:  # 返回一个布尔值 Series，表示每行是否为重复行
    
    # ----------------------------------------------------------------------
    # Sorting
    # error: Signature of "sort_values" incompatible with supertype "NDFrame"
    # 以下是对 sort_values 方法的重载，根据不同的参数签名进行不同的操作
    
    @overload  # 标记重载
    def sort_values(
        self,
        by: IndexLabel,
        *,
        axis: Axis = ...,  # 沿着哪个轴进行排序，默认为0（行）
        ascending=...,  # 是否升序，默认为 True
        inplace: Literal[False] = ...,  # 是否在原地修改 DataFrame，默认为 False
        kind: SortKind = ...,  # 排序算法，默认为 "quicksort"
        na_position: NaPosition = ...,  # NaN 值的位置，默认为 "last"
        ignore_index: bool = ...,  # 是否忽略索引重新排序，默认为 False
        key: ValueKeyFunc = ...,  # 自定义排序键函数，默认为 None
    ) -> DataFrame: ...
    
    @overload
    def sort_values(
        self,
        by: IndexLabel,
        *,
        axis: Axis = ...,  # 沿着哪个轴进行排序，默认为0（行）
        ascending=...,  # 是否升序，默认为 True
        inplace: Literal[True],  # 是否在原地修改 DataFrame，此处强制为 True
        kind: SortKind = ...,  # 排序算法，默认为 "quicksort"
        na_position: str = ...,  # NaN 值的位置，默认为 "last"
        ignore_index: bool = ...,  # 是否忽略索引重新排序，默认为 False
        key: ValueKeyFunc = ...,  # 自定义排序键函数，默认为 None
    ) -> None: ...
    
    def sort_values(
        self,
        by: IndexLabel,
        *,
        axis: Axis = 0,  # 沿着哪个轴进行排序，默认为0（行）
        ascending: bool | list[bool] | tuple[bool, ...] = True,  # 是否升序，默认为 True
        inplace: bool = False,  # 是否在原地修改 DataFrame，默认为 False
        kind: SortKind = "quicksort",  # 排序算法，默认为 "quicksort"
        na_position: str = "last",  # NaN 值的位置，默认为 "last"
        ignore_index: bool = False,  # 是否忽略索引重新排序，默认为 False
        key: ValueKeyFunc | None = None,  # 自定义排序键函数，默认为 None
    ) -> DataFrame: ...
    
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,  # 沿着哪个轴进行排序，默认为0（行）
        level: IndexLabel = ...,  # 指定层次化索引的级别，默认为 None
        ascending: bool | Sequence[bool] = ...,  # 是否升序，默认为 True
        inplace: Literal[True],  # 是否在原地修改 DataFrame，此处强制为 True
        kind: SortKind = ...,  # 排序算法，默认为 "quicksort"
        na_position: NaPosition = ...,  # NaN 值的位置，默认为 "last"
        sort_remaining: bool = ...,  # 是否同时排序其余轴，默认为 False
        ignore_index: bool = ...,  # 是否忽略索引重新排序，默认为 False
        key: IndexKeyFunc = ...,  # 自定义排序键函数，默认为 None
    ) -> None: ...
    
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,  # 沿着哪个轴进行排序，默认为0（行）
        level: IndexLabel = ...,  # 指定层次化索引的级别，默认为 None
        ascending: bool | Sequence[bool] = ...,  # 是否升序，默认为 True
        inplace: Literal[False] = ...,  # 是否在原地修改 DataFrame，默认为 False
        kind: SortKind = ...,  # 排序算法，默认为 "quicksort"
        na_position: NaPosition = ...,  # NaN 值的位置，默认为 "last"
        sort_remaining: bool = ...,  # 是否同时排序其余轴，默认为 False
        ignore_index: bool = ...,  # 是否忽略索引重新排序，默认为 False
        key: IndexKeyFunc = ...,  # 自定义排序键函数，默认为 None
    ) -> DataFrame: ...
    
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,  # 沿着哪个轴进行排序，默认为0（行）
        level: IndexLabel = ...,  # 指定层次化索引的级别，默认为 None
        ascending: bool | Sequence[bool] = ...,  # 是否升序，默认为 True
        inplace: bool = ...,  # 是否在原地修改 DataFrame，默认为 False
        kind: SortKind = ...,  # 排序算法，默认为 "quicksort"
        na_position: NaPosition = ...,  # NaN 值的位置，默认为 "last"
        sort_remaining: bool = ...,  # 是否同时排序其余轴，默认为 False
        ignore_index: bool = ...,  # 是否忽略索引重新排序，默认为 False
        key: IndexKeyFunc = ...,  # 自定义排序键函数，默认为 None
    ) -> DataFrame | None: ...
    def sort_index(
        self,
        *,
        axis: Axis = 0,  # 定义排序的轴，默认为0（行索引）
        level: IndexLabel | None = None,  # 指定多层索引的级别，默认为None
        ascending: bool | Sequence[bool] = True,  # 控制排序顺序，单个或多个布尔值，默认升序
        inplace: bool = False,  # 是否在原地排序，默认为False
        kind: SortKind = "quicksort",  # 指定排序算法，默认为快速排序
        na_position: NaPosition = "last",  # 缺失值的处理方式，默认放在最后
        sort_remaining: bool = True,  # 是否排序剩余的索引，默认为True
        ignore_index: bool = False,  # 是否忽略索引，默认为False
        key: IndexKeyFunc | None = None,  # 自定义排序键函数，默认为None
    ) -> None:
        """
        Sort index labels along a specified axis.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to sort along.
        level : int or str or tuple of int/str, default None
            If the axis is a MultiIndex (hierarchical), sort by a specific
            level or levels.
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders.
        inplace : bool, default False
            If True, perform operation in place.
        kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
            Choice of sorting algorithm.
        na_position : {'first', 'last'}, default 'last'
            Where to place NaNs.
        sort_remaining : bool, default True
            Sort non-specified levels along with the specified level in a
            multi-index.
        ignore_index : bool, default False
            If True, the index will not be sorted.
        key : callable, optional
            Apply the key function to the values before sorting.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If attempting to reorder levels on a non-hierarchical axis.

        See Also
        --------
        DataFrame.sort_values : Sort DataFrame by one or more columns.
        """
        pass

    def value_counts(
        self,
        subset: IndexLabel | None = None,  # 指定要计数的子集，默认为None
        normalize: bool = False,  # 是否返回相对频率，默认为False
        sort: bool = True,  # 是否按计数值排序，默认为True
        ascending: bool = False,  # 是否按升序排序，默认为False
        dropna: bool = True,  # 是否排除缺失值，默认为True
    ) -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        subset : array-like, optional
            List of index labels to include in the count.
        normalize : bool, default False
            If True, return relative frequencies.
        sort : bool, default True
            Sort by counts.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Exclude NA/null values.

        Returns
        -------
        Series
            Series with counts of unique values.

        See Also
        --------
        Series.nlargest : Return the largest n elements.
        Series.nsmallest : Return the smallest n elements.
        """
        pass

    def nlargest(
        self, n: int, columns: IndexLabel, keep: NsmallestNlargestKeep = "first"
    ) -> DataFrame:
        """
        Return the first n rows ordered by columns in descending order.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : str or list of str
            Column(s) to order by.
        keep : {'first', 'last'}, default 'first'
            Where values are equal, return either 'first' or 'last' rows.

        Returns
        -------
        DataFrame
            First n rows ordered by columns in descending order.

        See Also
        --------
        DataFrame.nsmallest : Return the first n rows ordered by columns in ascending order.
        """
        pass

    def nsmallest(
        self, n: int, columns: IndexLabel, keep: NsmallestNlargestKeep = "first"
    ) -> DataFrame:
        """
        Return the first n rows ordered by columns in ascending order.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : str or list of str
            Column(s) to order by.
        keep : {'first', 'last'}, default 'first'
            Where values are equal, return either 'first' or 'last' rows.

        Returns
        -------
        DataFrame
            First n rows ordered by columns in ascending order.

        See Also
        --------
        DataFrame.nlargest : Return the first n rows ordered by columns in descending order.
        """
        pass

    def reorder_levels(self, order: Sequence[int | str], axis: Axis = 0) -> DataFrame:
        """
        Rearrange index or column levels using input ``order``.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Where to reorder levels.

        Returns
        -------
        DataFrame
            DataFrame with indices or columns with reordered levels.

        Raises
        ------
        TypeError
            If attempting to reorder levels on a non-hierarchical axis.

        See Also
        --------
            DataFrame.swaplevel : Swap levels i and j in a MultiIndex.

        Examples
        --------
        >>> data = {
        ...     "class": ["Mammals", "Mammals", "Reptiles"],
        ...     "diet": ["Omnivore", "Carnivore", "Carnivore"],
        ...     "species": ["Humans", "Dogs", "Snakes"],
        ... }
        >>> df = pd.DataFrame(data, columns=["class", "diet", "species"])
        >>> df = df.set_index(["class", "diet"])
        >>> df
                                          species
        class      diet
        Mammals    Omnivore                Humans
                   Carnivore                 Dogs
        Reptiles   Carnivore               Snakes

        Let's reorder the levels of the index:

        >>> df.reorder_levels(["diet", "class"])
                                          species
        diet      class
        Omnivore  Mammals                  Humans
        Carnivore Mammals                    Dogs
                  Reptiles                 Snakes
        """
        axis = self._get_axis_number(axis)
        if not isinstance(self._get_axis(axis), MultiIndex):  # pragma: no cover
            raise TypeError("Can only reorder levels on a hierarchical axis.")

        result = self.copy(deep=False)

        if axis == 0:
            assert isinstance(result.index, MultiIndex)
            result.index = result.index.reorder_levels(order)
        else:
            assert isinstance(result.columns, MultiIndex)
            result.columns = result.columns.reorder_levels(order)
        return result
    # ----------------------------------------------------------------------
    # Arithmetic Methods

    # 定义比较方法，用于处理比较操作
    def _cmp_method(self, other, op):
        axis: Literal[1] = 1  # 只有在处理 Series 其他情况时才相关

        # 对操作数进行对齐以进行操作
        self, other = self._align_for_op(other, axis, flex=False, level=None)

        # 调用分派框架操作方法处理操作，并获取新的数据
        new_data = self._dispatch_frame_op(other, op, axis=axis)
        # 构造并返回操作结果
        return self._construct_result(new_data)

    # 定义算术方法，用于处理算术操作
    def _arith_method(self, other, op):
        # 如果需要重新索引来执行帧操作，则调用具有重新索引的算术方法
        if self._should_reindex_frame_op(other, op, 1, None, None):
            return self._arith_method_with_reindex(other, op)

        axis: Literal[1] = 1  # 只有在处理 Series 其他情况时才相关

        # 将标量操作数准备为可操作形式
        other = ops.maybe_prepare_scalar_for_op(other, (self.shape[axis],))

        # 对操作数进行对齐以进行操作，启用灵活模式
        self, other = self._align_for_op(other, axis, flex=True, level=None)

        # 使用忽略所有错误的状态处理，调度帧操作
        with np.errstate(all="ignore"):
            new_data = self._dispatch_frame_op(other, op, axis=axis)
        # 构造并返回操作结果
        return self._construct_result(new_data)

    # 逻辑方法与算术方法相同
    _logical_method = _arith_method

    # 分派帧操作的方法定义
    def _dispatch_frame_op(
        self, right, func: Callable, axis: AxisInt | None = None
        """
        Evaluate the frame operation func(left, right) by evaluating
        column-by-column, dispatching to the Series implementation.

        Parameters
        ----------
        right : scalar, Series, or DataFrame
            右侧操作数，可以是标量、Series 或者 DataFrame
        func : arithmetic or comparison operator
            算术或比较运算符，用于操作每一列的值
        axis : {None, 0, 1}
            轴向信息，确定操作是在列（0）还是行（1）上进行

        Returns
        -------
        DataFrame
            返回一个新的 DataFrame 对象，包含操作后的结果

        Notes
        -----
        Caller is responsible for setting np.errstate where relevant.
        调用者负责在相关情况下设置 np.errstate。

        """
        # Get the appropriate array-op to apply to each column/block's values.
        # 获取适当的数组操作以应用于每一列/块的值。
        array_op = ops.get_array_op(func)

        right = lib.item_from_zerodim(right)
        if not is_list_like(right):
            # i.e. scalar, faster than checking np.ndim(right) == 0
            # 即标量，比检查 np.ndim(right) == 0 更快
            bm = self._mgr.apply(array_op, right=right)
            return self._constructor_from_mgr(bm, axes=bm.axes)

        elif isinstance(right, DataFrame):
            assert self.index.equals(right.index)
            assert self.columns.equals(right.columns)
            # TODO: The previous assertion `assert right._indexed_same(self)`
            #  fails in cases with empty columns reached via
            #  _frame_arith_method_with_reindex
            # TODO: 上述断言 `assert right._indexed_same(self)` 在通过
            #  _frame_arith_method_with_reindex 达到空列的情况下失败

            # TODO operate_blockwise expects a manager of the same type
            #  TODO operate_blockwise 需要一个相同类型的管理器
            bm = self._mgr.operate_blockwise(
                right._mgr,
                array_op,
            )
            return self._constructor_from_mgr(bm, axes=bm.axes)

        elif isinstance(right, Series) and axis == 1:
            # axis=1 means we want to operate row-by-row
            # axis=1 表示我们希望逐行进行操作
            assert right.index.equals(self.columns)

            right = right._values
            # maybe_align_as_frame ensures we do not have an ndarray here
            # maybe_align_as_frame 确保这里没有 ndarray

            assert not isinstance(right, np.ndarray)

            arrays = [
                array_op(_left, _right)
                for _left, _right in zip(self._iter_column_arrays(), right)
            ]

        elif isinstance(right, Series):
            assert right.index.equals(self.index)
            right = right._values

            arrays = [array_op(left, right) for left in self._iter_column_arrays()]

        else:
            raise NotImplementedError(right)

        return type(self)._from_arrays(
            arrays, self.columns, self.index, verify_integrity=False
        )
    def _combine_frame(self, other: DataFrame, func, fill_value=None):
        # 在这一步我们假设 `self._indexed_same(other)` 已经成立

        if fill_value is None:
            # 如果 fill_value 为 None，则避免在循环中调用 _arith_op 函数，以减少函数调用开销
            _arith_op = func

        else:

            def _arith_op(left, right):
                # 对于混合类型情况，在迭代列时，_arith_op(left, right) 等同于
                # left._binop(right, func, fill_value=fill_value)
                left, right = ops.fill_binop(left, right, fill_value)
                return func(left, right)

        # 使用 _arith_op 函数分派帧操作
        new_data = self._dispatch_frame_op(other, _arith_op)
        return new_data

    def _arith_method_with_reindex(self, right: DataFrame, op) -> DataFrame:
        """
        用于需要重新索引的 DataFrame 与 DataFrame 操作，仅操作共享列，然后重新索引。

        Parameters
        ----------
        right : DataFrame
        op : 二元运算符

        Returns
        -------
        DataFrame
        """
        left = self

        # GH#31623，只在共享列上操作
        cols, lcol_indexer, rcol_indexer = left.columns.join(
            right.columns, how="inner", return_indexers=True
        )

        # 根据 lcol_indexer 和 rcol_indexer 对 left 和 right 进行切片操作
        new_left = left if lcol_indexer is None else left.iloc[:, lcol_indexer]
        new_right = right if rcol_indexer is None else right.iloc[:, rcol_indexer]
        
        # 使用给定的操作符 op 对 new_left 和 new_right 进行操作
        result = op(new_left, new_right)

        # 使用列的 join 来执行连接，而不是使用 left._align_for_op
        # 以避免构造两个可能很大/稀疏的 DataFrame
        join_columns = left.columns.join(right.columns, how="outer")

        if result.columns.has_duplicates:
            # 避免使用重复的轴进行重新索引
            # https://github.com/pandas-dev/pandas/issues/35194
            indexer, _ = result.columns.get_indexer_non_unique(join_columns)
            indexer = algorithms.unique1d(indexer)
            result = result._reindex_with_indexers(
                {1: [join_columns, indexer]}, allow_dups=True
            )
        else:
            result = result.reindex(join_columns, axis=1)

        return result
    def _should_reindex_frame_op(self, right, op, axis: int, fill_value, level) -> bool:
        """
        Check if this is an operation between DataFrames that will need to reindex.
        """
        if op is operator.pow or op is roperator.rpow:
            # 如果操作是乘幂运算或右侧乘幂运算，返回 False
            # GH#32685 pow 对于空值具有特殊语义
            return False

        if not isinstance(right, DataFrame):
            # 如果右侧不是 DataFrame 对象，返回 False
            return False

        if fill_value is None and level is None and axis == 1:
            # 如果填充值为 None，级别为 None，且轴为 1
            # Intersection 总是唯一的，因此我们需要检查唯一的列
            left_uniques = self.columns.unique()
            right_uniques = right.columns.unique()
            cols = left_uniques.intersection(right_uniques)
            if len(cols) and not (
                len(cols) == len(left_uniques) and len(cols) == len(right_uniques)
            ):
                # 如果 cols 的长度不为 0，并且左右两侧的唯一列数不相等
                # TODO: 当 len(cols) == 0 时是否有更快的处理方式？
                return True

        return False

    def _align_for_op(
        self,
        other,
        axis: AxisInt,
        flex: bool | None = False,
        level: Level | None = None,
    ):
        """
        Align the objects for a binary operation involving this DataFrame and another object.
        """
        # 对于涉及本 DataFrame 和另一个对象的二元操作，对其进行对齐
        pass

    def _maybe_align_series_as_frame(self, series: Series, axis: AxisInt):
        """
        Align a Series operand as a DataFrame if necessary.
        If the Series operand is not EA-dtype, we can broadcast to 2D and operate
        blockwise.
        """
        rvalues = series._values
        if not isinstance(rvalues, np.ndarray):
            # 如果 rvalues 不是 ndarray 对象
            if rvalues.dtype in ("datetime64[ns]", "timedelta64[ns]"):
                # 如果数据类型是日期时间或时间间隔
                # 可以损失地且廉价地转换为 ndarray
                rvalues = np.asarray(rvalues)
            else:
                return series

        if axis == 0:
            # 如果轴为 0，则将 rvalues 重塑为 (-1, 1)
            rvalues = rvalues.reshape(-1, 1)
        else:
            # 否则将 rvalues 重塑为 (1, -1)
            rvalues = rvalues.reshape(1, -1)

        rvalues = np.broadcast_to(rvalues, self.shape)
        # 传递 dtype 以避免推断
        # 返回一个与 self 具有相同索引和列的新对象，使用 rvalues 作为数据，dtype 为 rvalues 的数据类型
        return self._constructor(
            rvalues,
            index=self.index,
            columns=self.columns,
            dtype=rvalues.dtype,
        )

    def _flex_arith_method(
        self, other, op, *, axis: Axis = "columns", level=None, fill_value=None
    ):
        """
        Perform flexible arithmetic operation involving this DataFrame and another object.
        """
        # 执行涉及本 DataFrame 和另一个对象的灵活算术操作
        pass
    @Appender(ops.make_flex_doc("le", "dataframe"))
    def le(self, other, axis: Axis = "columns", level=None) -> DataFrame:
        """
        Compare this DataFrame with another DataFrame or Series using <= operator.

        Parameters
        ----------
        other : DataFrame or Series
            The other object to compare with.
        axis : Axis, optional, default 'columns'
            The axis to compare along.
        level : int or label, optional
            Level of MultiIndex or column name to compare on.

        Returns
        -------
        DataFrame
            A new DataFrame containing boolean values indicating if each element
            is less than or equal to the corresponding element in `other`.
        """
        # Determine the numerical axis index if specified, otherwise default to axis 1
        axis = self._get_axis_number(axis) if axis is not None else 1

        # Align self and other objects based on the specified axis and level
        self, other = self._align_for_op(other, axis, flex=True, level=level)

        # Dispatch the comparison operation between self and other based on their types and axis
        new_data = self._dispatch_frame_op(other, operator.le, axis=axis)

        # Construct and return a new DataFrame with the comparison results
        return self._construct_result(new_data)
    # 定义 DataFrame 类的方法，用于执行小于或等于比较操作
    def le(self, other, axis: Axis = "columns", level=None) -> DataFrame:
        return self._flex_cmp_method(other, operator.le, axis=axis, level=level)

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行小于比较操作
    @Appender(ops.make_flex_doc("lt", "dataframe"))
    def lt(self, other, axis: Axis = "columns", level=None) -> DataFrame:
        return self._flex_cmp_method(other, operator.lt, axis=axis, level=level)

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行大于或等于比较操作
    @Appender(ops.make_flex_doc("ge", "dataframe"))
    def ge(self, other, axis: Axis = "columns", level=None) -> DataFrame:
        return self._flex_cmp_method(other, operator.ge, axis=axis, level=level)

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行大于比较操作
    @Appender(ops.make_flex_doc("gt", "dataframe"))
    def gt(self, other, axis: Axis = "columns", level=None) -> DataFrame:
        return self._flex_cmp_method(other, operator.gt, axis=axis, level=level)

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行加法操作
    @Appender(ops.make_flex_doc("add", "dataframe"))
    def add(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, operator.add, level=level, fill_value=fill_value, axis=axis
        )

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行反向加法操作
    @Appender(ops.make_flex_doc("radd", "dataframe"))
    def radd(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, roperator.radd, level=level, fill_value=fill_value, axis=axis
        )

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行减法操作
    @Appender(ops.make_flex_doc("sub", "dataframe"))
    def sub(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, operator.sub, level=level, fill_value=fill_value, axis=axis
        )

    # 定义别名函数，将 sub 方法映射到 subtract
    subtract = sub

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行反向减法操作
    @Appender(ops.make_flex_doc("rsub", "dataframe"))
    def rsub(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, roperator.rsub, level=level, fill_value=fill_value, axis=axis
        )

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行乘法操作
    @Appender(ops.make_flex_doc("mul", "dataframe"))
    def mul(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, operator.mul, level=level, fill_value=fill_value, axis=axis
        )

    # 定义别名函数，将 mul 方法映射到 multiply
    multiply = mul

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行反向乘法操作
    @Appender(ops.make_flex_doc("rmul", "dataframe"))
    def rmul(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, roperator.rmul, level=level, fill_value=fill_value, axis=axis
        )

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行真除法操作
    @Appender(ops.make_flex_doc("truediv", "dataframe"))
    def truediv(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        return self._flex_arith_method(
            other, operator.truediv, level=level, fill_value=fill_value, axis=axis
        )

    # 定义别名函数，将 truediv 方法映射到 div 和 divide
    div = truediv
    divide = truediv

    # 使用装饰器向文档添加额外信息，并定义 DataFrame 类的方法，用于执行反向真除法操作
    @Appender(ops.make_flex_doc("rtruediv", "dataframe"))
    # 定义实例方法 rtruediv，用于执行真除操作
    def rtruediv(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行真除操作
        return self._flex_arith_method(
            other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis
        )

    # 创建别名 rdiv，指向 rtruediv 方法
    rdiv = rtruediv

    # 定义装饰器函数 floordiv，用于执行向下取整除法操作
    @Appender(ops.make_flex_doc("floordiv", "dataframe"))
    def floordiv(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行向下取整除法操作
        return self._flex_arith_method(
            other, operator.floordiv, level=level, fill_value=fill_value, axis=axis
        )

    # 定义装饰器函数 rfloordiv，用于执行反向向下取整除法操作
    @Appender(ops.make_flex_doc("rfloordiv", "dataframe"))
    def rfloordiv(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行反向向下取整除法操作
        return self._flex_arith_method(
            other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis
        )

    # 定义装饰器函数 mod，用于执行取模操作
    @Appender(ops.make_flex_doc("mod", "dataframe"))
    def mod(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行取模操作
        return self._flex_arith_method(
            other, operator.mod, level=level, fill_value=fill_value, axis=axis
        )

    # 定义装饰器函数 rmod，用于执行反向取模操作
    @Appender(ops.make_flex_doc("rmod", "dataframe"))
    def rmod(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行反向取模操作
        return self._flex_arith_method(
            other, roperator.rmod, level=level, fill_value=fill_value, axis=axis
        )

    # 定义装饰器函数 pow，用于执行幂运算操作
    @Appender(ops.make_flex_doc("pow", "dataframe"))
    def pow(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行幂运算操作
        return self._flex_arith_method(
            other, operator.pow, level=level, fill_value=fill_value, axis=axis
        )

    # 定义装饰器函数 rpow，用于执行反向幂运算操作
    @Appender(ops.make_flex_doc("rpow", "dataframe"))
    def rpow(
        self, other, axis: Axis = "columns", level=None, fill_value=None
    ) -> DataFrame:
        # 调用内部灵活算术方法 _flex_arith_method，执行反向幂运算操作
        return self._flex_arith_method(
            other, roperator.rpow, level=level, fill_value=fill_value, axis=axis
        )

    # ----------------------------------------------------------------------
    # Combination-Related

    # 定义实例方法 compare，用于比较两个 DataFrame 对象
    def compare(
        self,
        other: DataFrame,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ("self", "other"),
    ) -> DataFrame:
        # 调用父类的 compare 方法，进行比较操作
        return super().compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )

    # 定义实例方法 combine，用于将两个 DataFrame 对象组合起来
    def combine(
        self,
        other: DataFrame,
        func: Callable[[Series, Series], Series | Hashable],
        fill_value=None,
        overwrite: bool = True,
    ) -> DataFrame:
        # 未完整定义，应该继续添加该方法的其余部分
    def combine_first(self, other: DataFrame) -> DataFrame:
        """
        Update null elements with value in the same location in `other`.

        Combine two DataFrame objects by filling null values in one DataFrame
        with non-null values from other DataFrame. The row and column indexes
        of the resulting DataFrame will be the union of the two. The resulting
        dataframe contains the 'first' dataframe values and overrides the
        second one values where both first.loc[index, col] and
        second.loc[index, col] are not missing values, upon calling
        first.combine_first(second).

        Parameters
        ----------
        other : DataFrame
            Provided DataFrame to use to fill null values.

        Returns
        -------
        DataFrame
            The result of combining the provided DataFrame with the other object.

        See Also
        --------
        DataFrame.combine : Perform series-wise operation on two DataFrames
            using a given function.

        Examples
        --------
        >>> df1 = pd.DataFrame({"A": [None, 0], "B": [None, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})
        >>> df1.combine_first(df2)
             A    B
        0  1.0  3.0
        1  0.0  4.0

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> df1 = pd.DataFrame({"A": [None, 0], "B": [4, None]})
        >>> df2 = pd.DataFrame({"B": [3, 3], "C": [1, 1]}, index=[1, 2])
        >>> df1.combine_first(df2)
             A    B    C
        0  NaN  4.0  NaN
        1  0.0  3.0  1.0
        2  NaN  3.0  1.0
        """
        
        # 导入表达式模块，用于处理数据操作
        from pandas.core.computation import expressions

        # 定义一个内部函数 combiner，用于组合两个 Series 的值
        def combiner(x: Series, y: Series):
            # 创建一个布尔掩码，标识 x 中的空值
            mask = x.isna()._values

            # 获取 x 和 y 的数值
            x_values = x._values
            y_values = y._values

            # 如果 other DataFrame 中的列 y 不在当前 DataFrame 中，直接返回 y_values
            if y.name not in self.columns:
                return y_values

            # 使用表达式模块的 where 方法，根据 mask 来选择填充值
            return expressions.where(mask, y_values, x_values)

        # 如果 other DataFrame 为空，则重新索引当前 DataFrame 的列，加入 other 的列，并转换为 other 的数据类型
        if len(other) == 0:
            combined = self.reindex(
                self.columns.append(other.columns.difference(self.columns)), axis=1
            )
            combined = combined.astype(other.dtypes)
        else:
            # 合并当前 DataFrame 和 other，使用 combiner 函数进行合并，不覆盖已有值
            combined = self.combine(other, combiner, overwrite=False)

        # 查找当前 DataFrame 和 other DataFrame 共同列的数据类型，更新到 combined 中
        dtypes = {
            col: find_common_type([self.dtypes[col], other.dtypes[col]])
            for col in self.columns.intersection(other.columns)
            if combined.dtypes[col] != self.dtypes[col]
        }

        # 如果存在需要更新的数据类型，则将其更新到 combined 中
        if dtypes:
            combined = combined.astype(dtypes)

        # 返回最终合并后的 DataFrame，并使用当前对象来完成 finalize 操作
        return combined.__finalize__(self, method="combine_first")

    def update(
        self,
        other,
        join: UpdateJoin = "left",
        overwrite: bool = True,
        filter_func=None,
        errors: IgnoreRaise = "ignore",
    # ----------------------------------------------------------------------
    # Data reshaping
    
    # 将数据进行分组操作，返回一个 DataFrameGroupBy 对象
    @Appender(_shared_docs["groupby"] % _shared_doc_kwargs)
    def groupby(
        self,
        by=None,  # 分组依据，可以是列名、列表或者其他分组键
        level: IndexLabel | None = None,  # 指定分组的索引级别
        as_index: bool = True,  # 是否将分组键作为索引
        sort: bool = True,  # 是否对分组结果进行排序
        group_keys: bool = True,  # 是否显示组键
        observed: bool = True,  # 是否按照观测的顺序进行分组
        dropna: bool = True,  # 是否丢弃缺失值
    ) -> DataFrameGroupBy:  # 返回类型为 DataFrameGroupBy 对象
        from pandas.core.groupby.generic import DataFrameGroupBy

        # 如果未指定分组键和索引级别，则抛出异常
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")

        # 返回根据指定条件分组后的 DataFrameGroupBy 对象
        return DataFrameGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    # 根据指定的列和索引进行数据透视，返回一个 DataFrame 对象
    @Substitution("")
    @Appender(_shared_docs["pivot"])
    def pivot(
        self, *, columns, index=lib.no_default, values=lib.no_default
    ) -> DataFrame:
        from pandas.core.reshape.pivot import pivot

        return pivot(self, index=index, columns=columns, values=values)

    # 根据指定的索引、列和聚合函数创建数据透视表，返回一个 DataFrame 对象
    @Substitution("")
    @Appender(_shared_docs["pivot_table"])
    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc: AggFuncType = "mean",
        fill_value=None,
        margins: bool = False,
        dropna: bool = True,
        margins_name: Level = "All",
        observed: bool = True,
        sort: bool = True,
        **kwargs,
    ) -> DataFrame:
        from pandas.core.reshape.pivot import pivot_table

        # 返回根据指定条件创建的数据透视表 DataFrame 对象
        return pivot_table(
            self,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=sort,
            **kwargs,
        )

    # 将数据按指定的索引级别堆叠（重塑），返回一个 DataFrame 对象
    def stack(
        self,
        level: IndexLabel = -1,  # 指定堆叠的索引级别，默认为最内层
        dropna: bool | lib.NoDefault = lib.no_default,  # 是否丢弃缺失值
        sort: bool | lib.NoDefault = lib.no_default,  # 是否对结果进行排序
        future_stack: bool = True,  # 是否执行未来堆叠操作（保留未来兼容性）
    ):
        # 详细堆叠操作的具体实现可以在相关函数中查看

    # 将指定列中的每个元素扩展为单独的行，返回一个 DataFrame 对象
    def explode(
        self,
        column: IndexLabel,  # 指定要扩展的列
        ignore_index: bool = False,  # 是否忽略索引
    ):
        # 详细扩展操作的具体实现可以在相关函数中查看

    # 将数据按指定的索引级别展开（重塑），返回一个 DataFrame 对象
    def unstack(
        self, level: IndexLabel = -1, fill_value=None, sort: bool = True
    ):
        # 详细展开操作的具体实现可以在相关函数中查看
    ) -> DataFrame | Series:
        """
        Pivot a level of the (necessarily hierarchical) index labels.

        Returns a DataFrame having a new level of column labels whose inner-most
        level consists of the pivoted index labels.

        If the index is not a MultiIndex, the output will be a Series
        (the analogue of stack when the columns are not a MultiIndex).

        Parameters
        ----------
        level : int, str, or list of these, default -1 (last level)
            Level(s) of index to unstack, can pass level name.
        fill_value : int, str or dict
            Replace NaN with this value if the unstack produces missing values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        Series or DataFrame
            If index is a MultiIndex: DataFrame with pivoted index labels as new
            inner-most level column labels, else Series.

        See Also
        --------
        DataFrame.pivot : Pivot a table based on column values.
        DataFrame.stack : Pivot a level of the column labels (inverse operation
            from `unstack`).

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> index = pd.MultiIndex.from_tuples(
        ...     [("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")]
        ... )
        >>> s = pd.Series(np.arange(1.0, 5.0), index=index)
        >>> s
        one  a   1.0
             b   2.0
        two  a   3.0
             b   4.0
        dtype: float64

        >>> s.unstack(level=-1)
             a   b
        one  1.0  2.0
        two  3.0  4.0

        >>> s.unstack(level=0)
           one  two
        a  1.0   3.0
        b  2.0   4.0

        >>> df = s.unstack(level=0)
        >>> df.unstack()
        one  a  1.0
             b  2.0
        two  a  3.0
             b  4.0
        dtype: float64
        """
        # 导入 pandas 内部的 unstack 函数，用于实现数据透视
        from pandas.core.reshape.reshape import unstack

        # 调用 unstack 函数进行数据透视操作，返回结果存储在 result 中
        result = unstack(self, level, fill_value, sort)

        # 将结果对象使用当前对象进行 finalization，返回新的 DataFrame 或 Series
        return result.__finalize__(self, method="unstack")

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name: Hashable = "value",
        col_level: Level | None = None,
        ignore_index: bool = True,
    ):
        # ----------------------------------------------------------------------
        # Time series-related
    @doc(
        Series.diff,  # 使用 doc 方法为 diff 方法添加文档字符串
        klass="DataFrame",  # 设置 klass 参数为 "DataFrame"
        extra_params="axis : {0 or 'index', 1 or 'columns'}, default 0\n    "
        "Take difference over rows (0) or columns (1).\n",  # 描述 extra_params 参数的作用和默认值
        other_klass="Series",  # 设置 other_klass 参数为 "Series"
        examples=dedent(  # 提供一些使用示例
            """
        Difference with previous row

        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8],
        ...                    'c': [1, 4, 9, 16, 25, 36]})
        >>> df
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36

        >>> df.diff()
             a    b     c
        0  NaN  NaN   NaN
        1  1.0  0.0   3.0
        2  1.0  1.0   5.0
        3  1.0  1.0   7.0
        4  1.0  2.0   9.0
        5  1.0  3.0  11.0

        Difference with previous column

        >>> df.diff(axis=1)
            a  b   c
        0 NaN  0   0
        1 NaN -1   3
        2 NaN -1   7
        3 NaN -1  13
        4 NaN  0  20
        5 NaN  2  28

        Difference with 3rd previous row

        >>> df.diff(periods=3)
             a    b     c
        0  NaN  NaN   NaN
        1  NaN  NaN   NaN
        2  NaN  NaN   NaN
        3  3.0  2.0  15.0
        4  3.0  4.0  21.0
        5  3.0  6.0  27.0

        Difference with following row

        >>> df.diff(periods=-1)
             a    b     c
        0 -1.0  0.0  -3.0
        1 -1.0 -1.0  -5.0
        2 -1.0 -1.0  -7.0
        3 -1.0 -2.0  -9.0
        4 -1.0 -3.0 -11.0
        5  NaN  NaN   NaN

        Overflow in input dtype

        >>> df = pd.DataFrame({'a': [1, 0]}, dtype=np.uint8)
        >>> df.diff()
               a
        0    NaN
        1  255.0"""
        ),
    )
    def diff(self, periods: int = 1, axis: Axis = 0) -> DataFrame:
        if not lib.is_integer(periods):  # 检查 periods 是否为整数，如果不是则抛出 ValueError
            if not (is_float(periods) and periods.is_integer()):  # 如果 periods 不是浮点数或不是整数
                raise ValueError("periods must be an integer")  # 抛出值错误异常
            periods = int(periods)  # 将 periods 转换为整数类型

        axis = self._get_axis_number(axis)  # 获取轴的编号
        if axis == 1:  # 如果轴是列轴
            if periods != 0:
                # 在 periods == 0 情况下，这相当于轴为 0 的 diff，而且 Manager 方法可能更高效，因此我们在这种情况下分派。
                return self - self.shift(periods, axis=axis)  # 返回当前对象与向右偏移 periods 的对象的差值
            # 对于 periods=0，这相当于轴为 0 的 diff
            axis = 0  # 设置轴为 0

        new_data = self._mgr.diff(n=periods)  # 使用 _mgr 对象计算差分后的新数据
        res_df = self._constructor_from_mgr(new_data, axes=new_data.axes)  # 使用新数据创建结果 DataFrame
        return res_df.__finalize__(self, "diff")  # 保留当前对象的元数据并返回差分后的结果 DataFrame

    # ----------------------------------------------------------------------
    # Function application

    def _gotitem(
        self,
        key: IndexLabel,  # 键的标签
        ndim: int,  # 数组的维数
        subset: DataFrame | Series | None = None,  # 子集 DataFrame 或 Series，或者为 None
    ) -> DataFrame | Series:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
            The key or list of keys to slice the object.
        ndim : {1, 2}
            The requested number of dimensions for the result.
        subset : object, default None
            The subset of data to operate on.
        """
        if subset is None:
            subset = self  # Assign the current object if subset is None
        elif subset.ndim == 1:  # Check if subset is a Series (1-dimensional)
            return subset  # Return subset if it is a Series

        # TODO: _shallow_copy(subset)?
        return subset[key]  # Return the subset indexed by key

    _agg_see_also_doc = dedent(
        """
        See Also
        --------
        DataFrame.apply : Perform any type of operations.
        DataFrame.transform : Perform transformation type operations.
        DataFrame.groupby : Perform operations over groups.
        DataFrame.resample : Perform operations over resampled bins.
        DataFrame.rolling : Perform operations over rolling window.
        DataFrame.expanding : Perform operations over expanding window.
        core.window.ewm.ExponentialMovingWindow : Perform operation over exponential
            weighted window.
        """
    )

    _agg_examples_doc = dedent(
        """
        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3],
        ...                    [4, 5, 6],
        ...                    [7, 8, 9],
        ...                    [np.nan, np.nan, np.nan]],
        ...                   columns=['A', 'B', 'C'])

        Aggregate these functions over the rows.

        >>> df.agg(['sum', 'min'])
                A     B     C
        sum  12.0  15.0  18.0
        min   1.0   2.0   3.0

        Different aggregations per column.

        >>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
                A    B
        sum  12.0  NaN
        min   1.0  2.0
        max   NaN  8.0

        Aggregate different functions over the columns and rename the index of the resulting
        DataFrame.

        >>> df.agg(x=('A', 'max'), y=('B', 'min'), z=('C', 'mean'))
             A    B    C
        x  7.0  NaN  NaN
        y  NaN  2.0  NaN
        z  NaN  NaN  6.0

        Aggregate over the columns.

        >>> df.agg("mean", axis="columns")
        0    2.0
        1    5.0
        2    8.0
        3    NaN
        dtype: float64
        """
    )

    @doc(
        _shared_docs["aggregate"],  # Use the shared documentation for aggregate method
        klass=_shared_doc_kwargs["klass"],  # Class name for documentation context
        axis=_shared_doc_kwargs["axis"],  # Axis parameter description
        see_also=_agg_see_also_doc,  # Related methods and functions
        examples=_agg_examples_doc,  # Examples of usage
    )
    def aggregate(self, func=None, axis: Axis = 0, *args, **kwargs):
        from pandas.core.apply import frame_apply

        axis = self._get_axis_number(axis)  # Convert axis to numeric representation

        op = frame_apply(self, func=func, axis=axis, args=args, kwargs=kwargs)  # Apply function to DataFrame
        result = op.agg()  # Aggregate the operation results
        result = reconstruct_and_relabel_result(result, func, **kwargs)  # Reconstruct and relabel the resulting DataFrame
        return result  # Return the final aggregated result DataFrame

    agg = aggregate  # Alias aggregate method as 'agg'

    @doc(
        _shared_docs["transform"],  # Use the shared documentation for transform method
        klass=_shared_doc_kwargs["klass"],  # Class name for documentation context
        axis=_shared_doc_kwargs["axis"],  # Axis parameter description
    )
    def transform(
        self, func: AggFuncType, axis: Axis = 0, *args, **kwargs
    # 定义一个方法 apply，接受一个函数 func 作为参数，并返回一个 DataFrame 对象
    def apply(
        self,
        func: AggFuncType,
        axis: Axis = 0,
        raw: bool = False,
        result_type: Literal["expand", "reduce", "broadcast"] | None = None,
        args=(),
        by_row: Literal[False, "compat"] = "compat",
        engine: Literal["python", "numba"] = "python",
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs,
    ):
        # 导入 pandas 库中的 frame_apply 方法
        from pandas.core.apply import frame_apply
        # 使用 frame_apply 方法对当前对象 self 应用 func 函数，指定参数和选项
        op = frame_apply(self, func=func, axis=axis, args=args, kwargs=kwargs)
        # 对操作结果进行变换
        result = op.transform()
        # 断言结果是一个 DataFrame 对象
        assert isinstance(result, DataFrame)
        # 返回变换后的结果 DataFrame
        return result

    # 定义一个方法 map，接受一个函数 func 作为参数，并返回一个 DataFrame 对象
    def map(
        self, func: PythonFuncType, na_action: Literal["ignore"] | None = None, **kwargs
    ):
    ) -> DataFrame:
        """
        Apply a function to a Dataframe elementwise.

        .. versionadded:: 2.1.0

           DataFrame.applymap was deprecated and renamed to DataFrame.map.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.replace: Replace values given in `to_replace` with `value`.
        Series.map : Apply a function elementwise on a Series.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> df.map(lambda x: len(str(x)))
           0  1
        0  3  4
        1  5  5

        Like Series.map, NA values can be ignored:

        >>> df_copy = df.copy()
        >>> df_copy.iloc[0, 0] = pd.NA
        >>> df_copy.map(lambda x: len(str(x)), na_action="ignore")
             0  1
        0  NaN  4
        1  5.0  5

        It is also possible to use `map` with functions that are not
        `lambda` functions:

        >>> df.map(round, ndigits=1)
             0    1
        0  1.0  2.1
        1  3.4  4.6

        Note that a vectorized version of `func` often exists, which will
        be much faster. You could square each number elementwise.

        >>> df.map(lambda x: x**2)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489

        But it's better to avoid map in that case.

        >>> df**2
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489
        """
        # 检查 na_action 参数是否合法，只能是 'ignore' 或 None
        if na_action not in {"ignore", None}:
            raise ValueError(f"na_action must be 'ignore' or None. Got {na_action!r}")

        # 如果 DataFrame 为空，返回其副本
        if self.empty:
            return self.copy()

        # 将 func 函数与额外的关键字参数绑定，生成一个偏函数
        func = functools.partial(func, **kwargs)

        # 定义一个函数 infer，用于应用 func 函数到每个元素
        def infer(x):
            return x._map_values(func, na_action=na_action)

        # 对 DataFrame 应用 infer 函数，并用原始对象进行最终化处理
        return self.apply(infer).__finalize__(self, "map")

    # ----------------------------------------------------------------------
    # Merging / joining methods

    def _append(
        self,
        other,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
        ) -> DataFrame:
        # 定义方法签名，指定返回类型为 DataFrame

        if isinstance(other, (Series, dict)):
            # 检查 other 是否为 Series 或 dict 类型
            if isinstance(other, dict):
                # 如果 other 是 dict 类型
                if not ignore_index:
                    # 如果 ignore_index 不为 True，则抛出类型错误异常
                    raise TypeError("Can only append a dict if ignore_index=True")
                # 将 other 转换为 Series 类型
                other = Series(other)
            # 如果 other 没有名称且 ignore_index 不为 True，则抛出类型错误异常
            if other.name is None and not ignore_index:
                raise TypeError(
                    "Can only append a Series if ignore_index=True "
                    "or if the Series has a name"
                )

            # 创建索引对象 index
            index = Index(
                [other.name],
                name=self.index.names
                if isinstance(self.index, MultiIndex)
                else self.index.name,
            )
            # 将 other 转换为 DataFrame 格式的行数据
            row_df = other.to_frame().T
            # 对 row_df 进行类型推断，需要用于特定测试
            #  test_append_empty_frame_to_series_with_dateutil_tz
            other = row_df.infer_objects().rename_axis(index.names)

        elif isinstance(other, list):
            # 如果 other 是 list 类型
            if not other:
                # 如果 other 是空列表，则什么都不做
                pass
            elif not isinstance(other[0], DataFrame):
                # 如果 other 的第一个元素不是 DataFrame 类型，则将其转换为 DataFrame
                other = DataFrame(other)
                # 如果当前对象的索引名称不为 None 且 ignore_index 不为 True，则将 other 的索引名称设置为当前对象的索引名称

        from pandas.core.reshape.concat import concat

        # 根据 other 的类型确定要拼接的对象列表
        if isinstance(other, (list, tuple)):
            to_concat = [self, *other]
        else:
            to_concat = [self, other]

        # 使用 concat 方法进行拼接操作
        result = concat(
            to_concat,
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )
        # 返回拼接后的结果，并使用当前对象的方法 "append" 进行最终化处理
        return result.__finalize__(self, method="append")

    def join(
        self,
        other: DataFrame | Series | Iterable[DataFrame | Series],
        on: IndexLabel | None = None,
        how: MergeHow = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
        validate: JoinValidate | None = None,
    ):
        # 定义 join 方法，接受的参数及其默认值

        @Substitution("")
        @Appender(_merge_doc, indents=2)
        def merge(
            self,
            right: DataFrame | Series,
            how: MergeHow = "inner",
            on: IndexLabel | AnyArrayLike | None = None,
            left_on: IndexLabel | AnyArrayLike | None = None,
            right_on: IndexLabel | AnyArrayLike | None = None,
            left_index: bool = False,
            right_index: bool = False,
            sort: bool = False,
            suffixes: Suffixes = ("_x", "_y"),
            copy: bool | lib.NoDefault = lib.no_default,
            indicator: str | bool = False,
            validate: MergeValidate | None = None,
        ):
            # 定义 merge 方法，接受的参数及其默认值
    ) -> DataFrame:
        # 检查是否弃用复制操作
        self._check_copy_deprecation(copy)

        # 从 pandas 核心库中导入合并函数 merge
        from pandas.core.reshape.merge import merge

        # 调用 merge 函数进行数据框合并操作
        return merge(
            self,
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
        )

    def round(
        self, decimals: int | dict[IndexLabel, int] | Series = 0, *args, **kwargs
    ):
        # ----------------------------------------------------------------------
        # 统计方法等

    def corr(
        self,
        method: CorrelationMethod = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ):
        # ----------------------------------------------------------------------
        # 协方差计算方法

    def cov(
        self,
        min_periods: int | None = None,
        ddof: int | None = 1,
        numeric_only: bool = False,
    ):
        # ----------------------------------------------------------------------
        # 与另一个数据框或序列计算相关性

    def corrwith(
        self,
        other: DataFrame | Series,
        axis: Axis = 0,
        drop: bool = False,
        method: CorrelationMethod = "pearson",
        numeric_only: bool = False,
        min_periods: int | None = None,
    ):
        # ----------------------------------------------------------------------
        # 类似 ndarray 的统计方法
    # 定义一个方法用于计算每列或每行中非空单元格的数量
    def count(self, axis: Axis = 0, numeric_only: bool = False) -> Series:
        """
        Count non-NA cells for each column or row.

        The values `None`, `NaN`, `NaT`, ``pandas.NA`` are considered NA.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or 'index' counts are generated for each column.
            If 1 or 'columns' counts are generated for each row.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.value_counts: Count unique combinations of columns.
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = pd.DataFrame(
        ...     {
        ...         "Person": ["John", "Myla", "Lewis", "John", "Myla"],
        ...         "Age": [24.0, np.nan, 21.0, 33, 26],
        ...         "Single": [False, True, True, True, False],
        ...     }
        ... )
        >>> df
           Person   Age  Single
        0    John  24.0   False
        1    Myla   NaN    True
        2   Lewis  21.0    True
        3    John  33.0    True
        4    Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64

        Counts for each **row**:

        >>> df.count(axis="columns")
        0    3
        1    2
        2    3
        3    3
        4    3
        dtype: int64
        """
        # 获取指定轴的编号
        axis = self._get_axis_number(axis)

        # 如果 numeric_only 为 True，则获取仅包含 float、int 或 boolean 数据的 DataFrame
        if numeric_only:
            frame = self._get_numeric_data()
        else:
            frame = self

        # GH #423：处理轴为空的情况
        if len(frame._get_axis(axis)) == 0:
            # 如果轴为空，则创建一个空的 Series 对象
            result = self._constructor_sliced(0, index=frame._get_agg_axis(axis))
        else:
            # 否则，调用 notna 函数计算非 NA 值的数量，按指定轴进行求和
            result = notna(frame).sum(axis=axis)

        # 将结果转换为 int64 类型，并确保其与原始对象（self）具有相同的属性
        return result.astype("int64").__finalize__(self, method="count")

    def _reduce(
        self,
        op,
        name: str,
        *,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        filter_type=None,
        **kwds,
    # error: Signature of "any" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # 声明一个类型重载的装饰器，用于支持多个重载的any方法
    def any(
        self,
        *,
        axis: Axis = ...,
        bool_only: bool = ...,
        skipna: bool = ...,
        **kwargs,
    ) -> Series: ...

    @overload
    # 另一个any方法的重载，处理axis为None的情况，返回布尔值
    def any(
        self,
        *,
        axis: None,
        bool_only: bool = ...,
        skipna: bool = ...,
        **kwargs,
    ) -> bool: ...

    @overload
    # 第三个any方法的重载，处理不同的axis类型情况，返回Series或布尔值
    def any(
        self,
        *,
        axis: Axis | None,
        bool_only: bool = ...,
        skipna: bool = ...,
        **kwargs,
    ) -> Series | bool: ...

    @doc(make_doc("any", ndim=1))
    # any方法的文档生成，指定维度为1
    def any(
        self,
        *,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        # 调用_logical_func方法，传递"any"作为操作类型，进行逻辑操作
        result = self._logical_func(
            "any", nanops.nanany, axis, bool_only, skipna, **kwargs
        )
        # 如果结果是Series类型，则使用当前对象的finalize方法进行处理
        if isinstance(result, Series):
            result = result.__finalize__(self, method="any")
        # 返回最终结果
        return result

    @overload
    # 类似any方法，重载了all方法，支持多种参数组合
    def all(
        self,
        *,
        axis: Axis = ...,
        bool_only: bool = ...,
        skipna: bool = ...,
        **kwargs,
    ) -> Series: ...

    @overload
    def all(
        self,
        *,
        axis: None,
        bool_only: bool = ...,
        skipna: bool = ...,
        **kwargs,
    ) -> bool: ...

    @overload
    def all(
        self,
        *,
        axis: Axis | None,
        bool_only: bool = ...,
        skipna: bool = ...,
        **kwargs,
    ) -> Series | bool: ...

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="all")
    @doc(make_doc("all", ndim=1))
    # all方法的文档生成，指定维度为1
    # 定义 all 方法，用于对 Series 进行逻辑操作，返回结果为 Series 或 bool 值
    def all(
        self,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        # 调用 _logical_func 方法执行 "all" 操作，处理 NaN 值并返回结果
        result = self._logical_func(
            "all", nanops.nanall, axis, bool_only, skipna, **kwargs
        )
        # 如果结果是 Series 类型，则使用当前对象完成最后操作
        if isinstance(result, Series):
            result = result.__finalize__(self, method="all")
        return result

    # error: Signature of "min" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # min 方法的重载，针对不同的参数类型返回不同的类型注释
    def min(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...

    @overload
    def min(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...

    @overload
    def min(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...

    # 标记 min 方法的非关键字参数即将废弃，并生成 min 方法的文档
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="min")
    @doc(make_doc("min", ndim=2))
    # min 方法的实现，通过调用父类的 min 方法计算最小值并处理结果
    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        result = super().min(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        # 如果结果是 Series 类型，则使用当前对象完成最后操作
        if isinstance(result, Series):
            result = result.__finalize__(self, method="min")
        return result

    # error: Signature of "max" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # max 方法的重载，针对不同的参数类型返回不同的类型注释
    def max(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...

    @overload
    def max(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...

    @overload
    def max(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...

    # 标记 max 方法的非关键字参数即将废弃，并生成 max 方法的文档
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="max")
    @doc(make_doc("max", ndim=2))
    # max 方法的实现，通过调用父类的 max 方法计算最大值并处理结果
    def max(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        result = super().max(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        # 如果结果是 Series 类型，则使用当前对象完成最后操作
        if isinstance(result, Series):
            result = result.__finalize__(self, method="max")
        return result

    # 标记 sum 方法的非关键字参数即将废弃
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sum")
    # sum 方法的实现，计算 Series 对象沿指定轴的和并返回结果
    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    # 使用 @deprecate_nonkeyword_arguments 装饰器标记该方法已弃用，版本为 4.0，
    # 允许的参数为 ["self"]，函数名为 "prod"
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="prod")
    # 定义一个 prod 方法，属于某个类（未显示给出）
    def prod(
        self,
        axis: Axis | None = 0,  # 定义 axis 参数，表示计算的轴向，默认为索引轴 (0)
        skipna: bool = True,  # skipna 参数表示是否排除 NA/null 值，默认为 True
        numeric_only: bool = False,  # numeric_only 参数表示是否只包括 float、int、boolean 列，默认为 False
        min_count: int = 0,  # min_count 参数表示执行操作所需的最小有效值数，默认为 0
        **kwargs,  # **kwargs 收集额外的关键字参数
    ) -> Series:  # 方法返回一个 Series 对象
        """
        Return the product of the values over the requested axis.
    
        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
    
            .. warning::
    
                The behavior of DataFrame.prod with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).
    
            .. versionadded:: 2.0.0
    
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
            
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.
    
        Returns
        -------
        Series or scalar
            The product of the values over the requested axis.
    
        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.
    
        Examples
        --------
        By default, the product of an empty or all-NA Series is ``1``
    
        >>> pd.Series([], dtype="float64").prod()
        1.0
    
        This can be controlled with the ``min_count`` parameter
    
        >>> pd.Series([], dtype="float64").prod(min_count=1)
        nan
    
        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.
    
        >>> pd.Series([np.nan]).prod()
        1.0
    
        >>> pd.Series([np.nan]).prod(min_count=1)
        nan
        """
        # 调用父类的 prod 方法进行计算
        result = super().prod(
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )
        # 如果返回结果是一个 Series 对象，则使用 __finalize__ 方法进行最后的定制
        if isinstance(result, Series):
            result = result.__finalize__(self, method="prod")
        # 返回计算结果
        return result
    # error: Signature of "mean" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # 声明一个类型重载的装饰器，用于 mean 方法，告诉类型检查系统忽略覆盖错误

    def mean(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    # mean 方法的类型重载，指定返回类型为 Series，接受指定的参数，并且可能接受任意的关键字参数

    @overload
    # 继续声明类型重载的装饰器，忽略覆盖错误

    def mean(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...
    # mean 方法的类型重载，指定返回类型为 Any，当 axis 参数为 None 时

    @overload
    # 继续声明类型重载的装饰器，忽略覆盖错误

    def mean(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...
    # mean 方法的类型重载，指定返回类型为 Series 或 Any，接受 axis 参数为 Axis 或 None，接受任意的关键字参数

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="mean")
    # 使用装饰器标记 mean 方法的非关键字参数已被弃用，推荐使用关键字参数，版本为 4.0

    @doc(make_doc("mean", ndim=2))
    # 使用装饰器为 mean 方法添加文档，指定数据结构为 2 维

    def mean(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        # 计算均值，并返回结果
        result = super().mean(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        # 如果结果是 Series 类型，则使用当前对象进行最终处理
        if isinstance(result, Series):
            result = result.__finalize__(self, method="mean")
        return result
    # 返回计算结果

    # error: Signature of "median" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # 声明一个类型重载的装饰器，用于 median 方法，告诉类型检查系统忽略覆盖错误

    def median(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    # median 方法的类型重载，指定返回类型为 Series，接受指定的参数，并且可能接受任意的关键字参数

    @overload
    # 继续声明类型重载的装饰器，忽略覆盖错误

    def median(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...
    # median 方法的类型重载，指定返回类型为 Any，当 axis 参数为 None 时

    @overload
    # 继续声明类型重载的装饰器，忽略覆盖错误

    def median(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...
    # median 方法的类型重载，指定返回类型为 Series 或 Any，接受 axis 参数为 Axis 或 None，接受任意的关键字参数

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="median")
    # 使用装饰器标记 median 方法的非关键字参数已被弃用，推荐使用关键字参数，版本为 4.0

    @doc(make_doc("median", ndim=2))
    # 使用装饰器为 median 方法添加文档，指定数据结构为 2 维

    def median(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        # 计算中位数，并返回结果
        result = super().median(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        # 如果结果是 Series 类型，则使用当前对象进行最终处理
        if isinstance(result, Series):
            result = result.__finalize__(self, method="median")
        return result
    # 返回计算结果

    # error: Signature of "sem" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # 声明一个类型重载的装饰器，用于 sem 方法，告诉类型检查系统忽略覆盖错误

    def sem(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    # sem 方法的类型重载，指定返回类型为 Series，接受指定的参数，并且可能接受任意的关键字参数

    @overload
    # 继续声明类型重载的装饰器，忽略覆盖错误

    def sem(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...
    # sem 方法的类型重载，指定返回类型为 Any，当 axis 参数为 None 时
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sem")
    # 使用装饰器 @deprecate_nonkeyword_arguments 对 sem 方法进行标记为过时，将在 4.0 版本移除非关键字参数，允许的参数为 ["self"]
    def sem(
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sem with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        Series or DataFrame (if level specified)
            Unbiased standard error of the mean over requested axis.

        See Also
        --------
        DataFrame.var : Return unbiased variance over requested axis.
        DataFrame.std : Returns sample standard deviation over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.sem().round(6)
        0.57735

        With a DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
               a   b
        tiger  1   2
        zebra  2   3
        >>> df.sem()
        a   0.5
        b   0.5
        dtype: float64

        Using axis=1

        >>> df.sem(axis=1)
        tiger   0.5
        zebra   0.5
        dtype: float64

        In this case, `numeric_only` should be set to `True`
        to avoid getting an error.

        >>> df = pd.DataFrame({"a": [1, 2], "b": ["T", "Z"]}, index=["tiger", "zebra"])
        >>> df.sem(numeric_only=True)
        a   0.5
        dtype: float64
        """
        # 调用父类（superclass）的 sem 方法，传递相同的参数并获取结果
        result = super().sem(
            axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs
        )
        # 如果结果是 Series 类型，则用当前对象（self）进行最终化处理，方法为 "sem"
        if isinstance(result, Series):
            result = result.__finalize__(self, method="sem")
        # 返回结果
        return result
    # 声明函数重载，用于类型检查和提示，忽略覆盖错误
    @overload
    def var(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    
    # 另一种函数重载，处理axis为None的情况
    @overload
    def var(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...
    
    # 进行轴和返回类型的组合重载，支持axis为Axis或None的情况
    @overload
    def var(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...
    
    # 使用装饰器标记函数的过时非关键字参数，指定版本为4.0，允许的参数为"self"，函数名为"var"
    def var(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
    ) -> Series | Any:
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.var with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        Series or scalar
            Unbiased variance over requested axis.

        See Also
        --------
        numpy.var : Equivalent function in NumPy.
        Series.var : Return unbiased variance over Series values.
        Series.std : Return standard deviation over Series values.
        DataFrame.std : Return standard deviation of the values over
            the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.var()
        age       352.916667
        height      0.056367
        dtype: float64

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

        >>> df.var(ddof=0)
        age       264.687500
        height      0.042275
        dtype: float64
        """
        # 调用父类 NDFrame 的 var 方法来计算沿指定轴的无偏方差
        result = super().var(
            axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs
        )
        # 如果结果是 Series 类型，则使用当前对象（self）来完成最终处理
        if isinstance(result, Series):
            result = result.__finalize__(self, method="var")
        # 返回计算结果
        return result

    # error: Signature of "std" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    def std(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    # 定义 std 方法的第一部分，用于没有明确指定 axis 参数时的类型提示
    def std(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...
    
    # 定义 std 方法的第二部分，用于带有 Series 返回类型提示的重载情况
    @overload
    def std(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        ddof: int = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...
    
    # 使用 @deprecate_nonkeyword_arguments 装饰器对 std 方法进行修饰，声明即将弃用非关键字参数的使用
    def std(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
    ) -> Series | Any:
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.std with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Standard deviation over requested axis.

        See Also
        --------
        Series.std : Return standard deviation over Series values.
        DataFrame.mean : Return the mean of the values over the requested axis.
        DataFrame.mediam : Return the mediam of the values over the requested axis.
        DataFrame.mode : Get the mode(s) of each element along the requested axis.
        DataFrame.sum : Return the sum of the values over the requested axis.

        Notes
        -----
        To have the same behaviour as `numpy.std`, use `ddof=0` (instead of the
        default `ddof=1`)

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        The standard deviation of the columns can be found as follows:

        >>> df.std()
        age       18.786076
        height     0.237417
        dtype: float64

        Alternatively, `ddof=0` can be set to normalize by N instead of N-1:

        >>> df.std(ddof=0)
        age       16.269219
        height     0.205609
        dtype: float64
        """
        # Call the superclass method to compute the standard deviation
        result = super().std(
            axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs
        )
        # If the result is a Series, finalize it as a Series with the method "std"
        if isinstance(result, Series):
            result = result.__finalize__(self, method="std")
        # Return the computed result
        return result

    # error: Signature of "skew" incompatible with supertype "NDFrame"
    @overload  # type: ignore[override]
    # 声明函数重载，用于类型检查时忽略覆盖警告
    def skew(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    # 返回一个 Series 对象，计算请求轴上的无偏倾斜度

    @overload
    # 另一种重载声明
    def skew(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...
    # 返回任意类型，计算请求轴上的无偏倾斜度，轴为 None 时适用

    @overload
    # 第三种重载声明
    def skew(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...
    # 返回 Series 或任意类型，计算请求轴上的无偏倾斜度，可指定轴或使用 None

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="skew")
    # 使用装饰器标记该函数不再支持非关键字参数，并指定版本和允许的参数
    def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        """
        Return unbiased skew over requested axis.

        Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Unbiased skew over requested axis.

        See Also
        --------
        Dataframe.kurt : Returns unbiased kurtosis over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.skew()
        0.0

        With a DataFrame

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3], "b": [2, 3, 4], "c": [1, 3, 5]},
        ...     index=["tiger", "zebra", "cow"],
        ... )
        >>> df
                a   b   c
        tiger   1   2   1
        zebra   2   3   3
        cow     3   4   5
        >>> df.skew()
        a   0.0
        b   0.0
        c   0.0
        dtype: float64

        Using axis=1

        >>> df.skew(axis=1)
        tiger   1.732051
        zebra  -1.732051
        cow     0.000000
        dtype: float64

        In this case, `numeric_only` should be set to `True` to avoid
        getting an error.

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3], "b": ["T", "Z", "X"]}, index=["tiger", "zebra", "cow"]
        ... )
        >>> df.skew(numeric_only=True)
        a   0.0
        dtype: float64
        """
        result = super().skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        # 调用父类的 skew 方法，计算结果赋给 result
        if isinstance(result, Series):
            result = result.__finalize__(self, method="skew")
            # 如果结果是 Series 类型，使用当前对象来 finalize 结果，方法为 'skew'
        return result
    # 返回计算的结果，可以是 Series 或标量
    # 使用 @overload 装饰器标记方法的多态定义，用于类型检查
    @overload  # type: ignore[override]
    # 声明方法的第一个重载：返回一个 Series 类型，计算给定轴向上的峰度
    def kurt(
        self,
        *,
        axis: Axis = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...

    # 声明方法的第二个重载：返回任意类型，计算全局的峰度
    @overload
    def kurt(
        self,
        *,
        axis: None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Any: ...

    # 声明方法的第三个重载：返回 Series 或任意类型，支持轴向和全局的峰度计算
    @overload
    def kurt(
        self,
        *,
        axis: Axis | None,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series | Any: ...

    # 使用 @deprecate_nonkeyword_arguments 装饰器标记方法的弃用非关键字参数，版本为 4.0，允许 "self" 参数
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="kurt")
    # 实现主体方法定义：计算峰度的核心逻辑，支持指定轴向、是否跳过 NaN 值、仅计算数值型数据
    def kurt(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        """
        返回请求轴上的无偏峰度。

        使用 Fisher 的峰度定义（正态分布的峰度为 0.0）。通过 N-1 进行标准化。

        Parameters
        ----------
        axis : {index (0), columns (1)}
            要应用函数的轴。
            对于 `Series`，此参数未使用，默认为 0。

            对于 DataFrames，指定 ``axis=None`` 将在两个轴上应用聚合。

            .. versionadded:: 2.0.0

        skipna : bool, default True
            计算结果时是否排除 NA/null 值。

        numeric_only : bool, default False
            是否仅包括 float、int、boolean 列。

        **kwargs
            要传递给函数的额外关键字参数。

        Returns
        -------
        Series or scalar
            请求轴上的无偏峰度。

        See Also
        --------
        Dataframe.kurtosis : 返回请求轴上的无偏峰度。

        Examples
        --------
        >>> s = pd.Series([1, 2, 2, 3], index=["cat", "dog", "dog", "mouse"])
        >>> s
        cat    1
        dog    2
        dog    2
        mouse  3
        dtype: int64
        >>> s.kurt()
        1.5

        对于 DataFrame

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 2, 3], "b": [3, 4, 4, 4]},
        ...     index=["cat", "dog", "dog", "mouse"],
        ... )
        >>> df
               a   b
          cat  1   3
          dog  2   4
          dog  2   4
        mouse  3   4
        >>> df.kurt()
        a   1.5
        b   4.0
        dtype: float64

        使用 axis=None

        >>> df.kurt(axis=None).round(6)
        -0.988693

        使用 axis=1

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2], "b": [3, 4], "c": [3, 4], "d": [1, 2]},
        ...     index=["cat", "dog"],
        ... )
        >>> df.kurt(axis=1)
        cat   -6.0
        dog   -6.0
        dtype: float64
        """
        result = super().kurt(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        if isinstance(result, Series):
            result = result.__finalize__(self, method="kurt")
        return result

    # 错误: 分配中的类型不兼容
    kurtosis = kurt  # type: ignore[assignment]
    product = prod

    @doc(make_doc("cummin", ndim=2))
    def cummin(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ) -> Self:
        """
        返回按轴累计最小值的函数。

        Parameters
        ----------
        axis : Axis, optional
            操作的轴，默认为 0。
        
        skipna : bool, default True
            是否排除 NA/null 值。

        numeric_only : bool, default False
            是否仅包括数值型数据列。

        *args, **kwargs
            其他传递给函数的参数。

        Returns
        -------
        Self
            累计最小值后的结果。

        See Also
        --------
        DataFrame.cummin : 返回按轴累计最小值的函数。

        """
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cummin(data, axis, skipna, *args, **kwargs)

    @doc(make_doc("cummax", ndim=2))
    def cummax(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        *args,
        **kwargs,

        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ) -> Self:
        """
        返回按轴累计最大值的函数。

        Parameters
        ----------
        axis : Axis, optional
            操作的轴，默认为 0。
        
        skipna : bool, default True
            是否排除 NA/null 值。

        numeric_only : bool, default False
            是否仅包括数值型数据列。

        *args, **kwargs
            其他传递给函数的参数。

        Returns
        -------
        Self
            累计最大值后的结果。

        See Also
        --------
        DataFrame.cummax : 返回按轴累计最大值的函数。

        """
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cummax(data, axis, skipna, *args, **kwargs)
    ) -> Self:
        # 如果 numeric_only 为 True，则仅获取数值型数据，否则使用全部数据
        data = self._get_numeric_data() if numeric_only else self
        # 调用 NDFrame.cummax 方法计算沿指定轴的累积最大值
        return NDFrame.cummax(data, axis, skipna, *args, **kwargs)

    @doc(make_doc("cumsum", ndim=2))
    def cumsum(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ) -> Self:
        # 如果 numeric_only 为 True，则仅获取数值型数据，否则使用全部数据
        data = self._get_numeric_data() if numeric_only else self
        # 调用 NDFrame.cumsum 方法计算沿指定轴的累积和
        return NDFrame.cumsum(data, axis, skipna, *args, **kwargs)

    @doc(make_doc("cumprod", 2))
    def cumprod(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        *args,
        **kwargs,
    ) -> Self:
        # 如果 numeric_only 为 True，则仅获取数值型数据，否则使用全部数据
        data = self._get_numeric_data() if numeric_only else self
        # 调用 NDFrame.cumprod 方法计算沿指定轴的累积乘积
        return NDFrame.cumprod(data, axis, skipna, *args, **kwargs)

    def nunique(self, axis: Axis = 0, dropna: bool = True) -> Series:
        """
        Count number of distinct elements in specified axis.

        Return Series with number of distinct elements. Can ignore NaN
        values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
            column-wise.
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series
            Series with counts of unique values per row or column, depending on `axis`.

        See Also
        --------
        Series.nunique: Method nunique for Series.
        DataFrame.count: Count non-NA cells for each column or row.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [4, 5, 6], "B": [4, 1, 1]})
        >>> df.nunique()
        A    3
        B    2
        dtype: int64

        >>> df.nunique(axis=1)
        0    1
        1    2
        2    2
        dtype: int64
        """
        # 应用 Series.nunique 方法计算沿指定轴的唯一值数量
        return self.apply(Series.nunique, axis=axis, dropna=dropna)

    @doc(_shared_docs["idxmin"], numeric_only_default="False")
    def idxmin(
        self, axis: Axis = 0, skipna: bool = True, numeric_only: bool = False
    ) -> Self:
        # 返回最小值的索引，忽略 NaN 值
        return NDFrame.idxmin(self, axis=axis, skipna=skipna, numeric_only=numeric_only)
    @doc(_shared_docs["idxmax"], numeric_only_default="False")
    def idxmax(
        self, axis: Axis = 0, skipna: bool = True, numeric_only: bool = False
    ) -> Series:
        # 获取轴的编号，确保 axis 是有效的轴索引
        axis = self._get_axis_number(axis)

        # 如果 Series 是空的，并且指定轴不为空，返回一个具有指定轴数据类型的切片构造器对象
        if self.empty and len(self.axes[axis]):
            axis_dtype = self.axes[axis].dtype
            return self._constructor_sliced(dtype=axis_dtype)

        # 如果指定了 numeric_only，只获取数值数据；否则使用当前数据
        if numeric_only:
            data = self._get_numeric_data()
        else:
            data = self

        # 使用 _reduce 方法计算数据的最大值索引，返回的是一个包含索引值的结果对象
        res = data._reduce(
            nanops.nanargmax, "argmax", axis=axis, skipna=skipna, numeric_only=False
        )
        # 获取结果对象中的索引值
        indices = res._values
        # indices 将始终是一维数组，因为 axis 不是 None

        # 如果 indices 中存在值为 -1 的情况，发出警告
        if (indices == -1).any():
            warnings.warn(
                f"The behavior of {type(self).__name__}.idxmax with all-NA "
                "values, or any-NA and skipna=False, is deprecated. In a future "
                "version this will raise ValueError",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        # 获取数据的指定轴对象
        index = data._get_axis(axis)
        # 使用算法 take 从索引值中取出相应的值，允许填充空值，填充值为索引对象的 NA 值
        result = algorithms.take(
            index._values, indices, allow_fill=True, fill_value=index._na_value
        )
        # 使用切片构造器创建最终结果对象，指定索引为数据的聚合轴
        final_result = data._constructor_sliced(result, index=data._get_agg_axis(axis))
        # 返回最终结果，并通过 __finalize__ 方法保留附加的元数据
        return final_result.__finalize__(self, method="idxmax")

    def _get_agg_axis(self, axis_num: int) -> Index:
        """
        Let's be explicit about this.
        """
        # 根据指定的轴编号返回相应的轴对象，如果轴编号不是 0 或 1，则引发 ValueError 异常
        if axis_num == 0:
            return self.columns
        elif axis_num == 1:
            return self.index
        else:
            raise ValueError(f"Axis must be 0 or 1 (got {axis_num!r})")
    # 定义一个方法 `mode`，用于计算数据的众数
    def mode(
        self, axis: Axis = 0, numeric_only: bool = False, dropna: bool = True
    ) -> DataFrame:
        """
        Get the mode(s) of each element along the selected axis.

        The mode of a set of values is the value that appears most often.
        It can be multiple values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to iterate over while searching for the mode:

            * 0 or 'index' : get mode of each column
            * 1 or 'columns' : get mode of each row.

        numeric_only : bool, default False
            If True, only apply to numeric columns.
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        DataFrame
            The modes of each column or row.

        See Also
        --------
        Series.mode : Return the highest frequency value in a Series.
        Series.value_counts : Return the counts of values in a Series.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("bird", 2, 2),
        ...         ("mammal", 4, np.nan),
        ...         ("arthropod", 8, 0),
        ...         ("bird", 2, np.nan),
        ...     ],
        ...     index=("falcon", "horse", "spider", "ostrich"),
        ...     columns=("species", "legs", "wings"),
        ... )
        >>> df
                   species  legs  wings
        falcon        bird     2    2.0
        horse       mammal     4    NaN
        spider   arthropod     8    0.0
        ostrich       bird     2    NaN

        By default, missing values are not considered, and the mode of wings
        are both 0 and 2. Because the resulting DataFrame has two rows,
        the second row of ``species`` and ``legs`` contains ``NaN``.

        >>> df.mode()
          species  legs  wings
        0    bird   2.0    0.0
        1     NaN   NaN    2.0

        Setting ``dropna=False`` ``NaN`` values are considered and they can be
        the mode (like for wings).

        >>> df.mode(dropna=False)
          species  legs  wings
        0    bird     2    NaN

        Setting ``numeric_only=True``, only the mode of numeric columns is
        computed, and columns of other types are ignored.

        >>> df.mode(numeric_only=True)
           legs  wings
        0   2.0    0.0
        1   NaN    2.0

        To compute the mode over columns and not rows, use the axis parameter:

        >>> df.mode(axis="columns", numeric_only=True)
                   0    1
        falcon   2.0  NaN
        horse    4.0  NaN
        spider   0.0  8.0
        ostrich  2.0  NaN
        """
        # If numeric_only is True, select numeric columns, otherwise use all columns
        data = self if not numeric_only else self._get_numeric_data()

        # Define a function f to compute mode for a Series s
        def f(s):
            return s.mode(dropna=dropna)

        # Apply function f along the specified axis (0 for columns, 1 for rows)
        data = data.apply(f, axis=axis)

        # Ensure index is type stable (should always use int index)
        if data.empty:
            data.index = default_index(0)

        # Return the DataFrame containing modes of each column or row
        return data

    @overload
    # 定义 quantile 方法，用于计算指定分位数的值
    def quantile(
        self,
        q: float | AnyArrayLike | Sequence[float] = 0.5,  # 分位数，可以是单个值、数组或序列，默认为0.5
        axis: Axis = 0,  # 沿着哪个轴计算分位数，默认为0（行）
        numeric_only: bool = False,  # 是否仅考虑数值类型，默认为False
        interpolation: QuantileInterpolation = "linear",  # 插值方法，默认为线性插值
        method: Literal["single", "table"] = "single",  # 计算方法，可以是单值或表格法，默认为单值
    ) -> Series | DataFrame:  # 返回值可以是 Series 或 DataFrame

    # 定义 quantile 方法的重载，支持多种参数形式
    @overload
    def quantile(
        self,
        q: AnyArrayLike | Sequence[float],  # 分位数，可以是数组或序列
        axis: Axis = 0,  # 沿着哪个轴计算分位数，默认为0（行）
        numeric_only: bool = False,  # 是否仅考虑数值类型，默认为False
        interpolation: QuantileInterpolation = "linear",  # 插值方法，默认为线性插值
        method: Literal["single", "table"] = "single",  # 计算方法，可以是单值或表格法，默认为单值
    ) -> Series | DataFrame:  # 返回值可以是 Series 或 DataFrame

    # 定义 quantile 方法的重载，支持多种参数形式
    @overload
    def quantile(
        self,
        q: float | AnyArrayLike | Sequence[float] = ...,  # 分位数，可以是单个值、数组或序列，默认为未指定
        axis: Axis = 0,  # 沿着哪个轴计算分位数，默认为0（行）
        numeric_only: bool = False,  # 是否仅考虑数值类型，默认为False
        interpolation: QuantileInterpolation = "linear",  # 插值方法，默认为线性插值
        method: Literal["single", "table"] = "single",  # 计算方法，可以是单值或表格法，默认为单值
    ) -> Series | DataFrame:  # 返回值可以是 Series 或 DataFrame

    # 定义 to_timestamp 方法，将索引转换为时间戳
    def to_timestamp(
        self,
        freq: Frequency | None = None,  # 频率，可以是字符串或 None
        how: ToTimestampHow = "start",  # 转换方式，可以是开始或结束，默认为开始
        axis: Axis = 0,  # 沿着哪个轴转换，默认为0（行）
        copy: bool | lib.NoDefault = lib.no_default,  # 是否复制，默认为 lib.no_default
    ):

    # 定义 to_period 方法，将索引转换为周期
    def to_period(
        self,
        freq: Frequency | None = None,  # 频率，可以是字符串或 None
        axis: Axis = 0,  # 沿着哪个轴转换，默认为0（行）
        copy: bool | lib.NoDefault = lib.no_default,  # 是否复制，默认为 lib.no_default
    ):
    ) -> DataFrame:
        """
        Convert DataFrame from DatetimeIndex to PeriodIndex.

        Convert DataFrame from DatetimeIndex to PeriodIndex with desired
        frequency (inferred from index if not passed). Either index of columns can be
        converted, depending on `axis` argument.

        Parameters
        ----------
        freq : str, default
            Frequency of the PeriodIndex.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to convert (the index by default).
        copy : bool, default False
            If False then underlying input data is not copied.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0

        Returns
        -------
        DataFrame
            The DataFrame with the converted PeriodIndex.

        See Also
        --------
        Series.to_period: Equivalent method for Series.
        Series.dt.to_period: Convert DateTime column values.

        Examples
        --------
        >>> idx = pd.to_datetime(
        ...     [
        ...         "2001-03-31 00:00:00",
        ...         "2002-05-31 00:00:00",
        ...         "2003-08-31 00:00:00",
        ...     ]
        ... )

        >>> idx
        DatetimeIndex(['2001-03-31', '2002-05-31', '2003-08-31'],
        dtype='datetime64[s]', freq=None)

        >>> idx.to_period("M")
        PeriodIndex(['2001-03', '2002-05', '2003-08'], dtype='period[M]')

        For the yearly frequency

        >>> idx.to_period("Y")
        PeriodIndex(['2001', '2002', '2003'], dtype='period[Y-DEC]')
        """
        # Check for deprecation warning related to copy behavior
        self._check_copy_deprecation(copy)
        
        # Create a shallow copy of the DataFrame
        new_obj = self.copy(deep=False)

        # Determine the name of the axis to convert
        axis_name = self._get_axis_name(axis)
        
        # Retrieve the current axis object
        old_ax = getattr(self, axis_name)
        
        # Ensure the current axis is of type DatetimeIndex; otherwise, raise an error
        if not isinstance(old_ax, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(old_ax).__name__}")

        # Convert the DatetimeIndex to PeriodIndex with the specified frequency
        new_ax = old_ax.to_period(freq=freq)

        # Set the converted PeriodIndex back to the new DataFrame object
        setattr(new_obj, axis_name, new_ax)
        
        # Return the modified DataFrame with PeriodIndex
        return new_obj

    # ----------------------------------------------------------------------
    # Add index and columns
    # Define a constant list of axis orders for indexing
    _AXIS_ORDERS: list[Literal["index", "columns"]] = ["index", "columns"]
    
    # Map axis types to their corresponding integer representation
    _AXIS_TO_AXIS_NUMBER: dict[Axis, int] = {
        **NDFrame._AXIS_TO_AXIS_NUMBER,
        1: 1,
        "columns": 1,
    }
    
    # Calculate the length of the axis orders list
    _AXIS_LEN = len(_AXIS_ORDERS)
    
    # Specify the axis number for information purposes
    _info_axis_number: Literal[1] = 1
    # 定义私有属性 _info_axis_name，表示轴名称为 "columns"
    _info_axis_name: Literal["columns"] = "columns"

    # 创建 index 属性，表示 DataFrame 的索引（行标签）
    index = properties.AxisProperty(
        axis=1,
        doc="""
        DataFrame 的索引（行标签）。

        DataFrame 的索引是一系列标签，用于标识每一行。
        这些标签可以是整数、字符串或任何其他可散列的类型。
        索引用于基于标签的访问和对齐，可以使用此属性访问或修改。

        返回
        -------
        pandas.Index
            DataFrame 的索引标签。

        另请参阅
        --------
        DataFrame.columns : DataFrame 的列标签。
        DataFrame.to_numpy : 将 DataFrame 转换为 NumPy 数组。

        示例
        --------
        >>> df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Aritra'],
        ...                    'Age': [25, 30, 35],
        ...                    'Location': ['Seattle', 'New York', 'Kona']},
        ...                   index=([10, 20, 30]))
        >>> df.index
        Index([10, 20, 30], dtype='int64')

        在此示例中，我们创建了一个包含 3 行和 3 列的 DataFrame，包括 Name、Age 和 Location 信息。
        我们将索引标签设置为整数 10、20 和 30。然后我们访问 DataFrame 的 `index` 属性，
        返回一个包含索引标签的 `Index` 对象。

        >>> df.index = [100, 200, 300]
        >>> df
            Name  Age Location
        100  Alice   25  Seattle
        200    Bob   30 New York
        300  Aritra  35    Kona

        在此示例中，我们通过将新标签列表分配给 `index` 属性来修改 DataFrame 的索引标签。
        DataFrame 随后使用新标签进行更新，并输出显示修改后的 DataFrame。
        """,
    )

    # 创建 columns 属性，表示 DataFrame 的列标签
    columns = properties.AxisProperty(
        axis=0,
        doc=dedent(
            """
            DataFrame 的列标签。

            另请参阅
            --------
            DataFrame.index : DataFrame 的索引（行标签）。
            DataFrame.axes : 返回表示 DataFrame 轴的列表。

            示例
            --------
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> df
                 A  B
            0    1  3
            1    2  4
            >>> df.columns
            Index(['A', 'B'], dtype='object')
            """
        ),
    )

    # ----------------------------------------------------------------------
    # 为 DataFrame 添加绘图方法

    # 使用 Accessor 将 "plot" 作为属性名，关联到 pandas.plotting.PlotAccessor
    plot = Accessor("plot", pandas.plotting.PlotAccessor)

    # 直方图绘制方法，使用 pandas.plotting.hist_frame
    hist = pandas.plotting.hist_frame

    # 箱线图绘制方法，使用 pandas.plotting.boxplot_frame
    boxplot = pandas.plotting.boxplot_frame

    # 使用 Accessor 将 "sparse" 作为属性名，关联到 SparseFrameAccessor
    sparse = Accessor("sparse", SparseFrameAccessor)

    # ----------------------------------------------------------------------
    # 内部接口方法
    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of the DataFrame.

        .. warning::

           We recommend using :meth:`DataFrame.to_numpy` instead.

        Only the values in the DataFrame will be returned, the axes labels
        will be removed.

        Returns
        -------
        numpy.ndarray
            The values of the DataFrame.

        See Also
        --------
        DataFrame.to_numpy : Recommended alternative to this method.
        DataFrame.index : Retrieve the index labels.
        DataFrame.columns : Retrieving the column names.

        Notes
        -----
        The dtype will be a lower-common-denominator dtype (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen. Use this
        with care if you are not dealing with the blocks.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32.  If dtypes are int32 and uint8, dtype will be upcast to
        int32. By :func:`numpy.find_common_type` convention, mixing int64
        and uint64 will result in a float64 dtype.

        Examples
        --------
        A DataFrame where all columns are the same type (e.g., int64) results
        in an array of the same type.

        >>> df = pd.DataFrame(
        ...     {"age": [3, 29], "height": [94, 170], "weight": [31, 115]}
        ... )
        >>> df
           age  height  weight
        0    3      94      31
        1   29     170     115
        >>> df.dtypes
        age       int64
        height    int64
        weight    int64
        dtype: object
        >>> df.values
        array([[  3,  94,  31],
               [ 29, 170, 115]])

        A DataFrame with mixed type columns(e.g., str/object, int64, float32)
        results in an ndarray of the broadest type that accommodates these
        mixed types (e.g., object).

        >>> df2 = pd.DataFrame(
        ...     [
        ...         ("parrot", 24.0, "second"),
        ...         ("lion", 80.5, 1),
        ...         ("monkey", np.nan, None),
        ...     ],
        ...     columns=("name", "max_speed", "rank"),
        ... )
        >>> df2.dtypes
        name          object
        max_speed    float64
        rank          object
        dtype: object
        >>> df2.values
        array([['parrot', 24.0, 'second'],
               ['lion', 80.5, 1],
               ['monkey', nan, None]], dtype=object)
        """
        # 调用底层数据管理器的方法将DataFrame转换为NumPy数组表示
        return self._mgr.as_array()
# 将嵌套字典转换为默认字典的字典结构
def _from_nested_dict(
    data: Mapping[HashableT, Mapping[HashableT2, T]],
) -> collections.defaultdict[HashableT2, dict[HashableT, T]]:
    # 创建一个空的默认字典，内部值为普通字典
    new_data: collections.defaultdict[HashableT2, dict[HashableT, T]] = (
        collections.defaultdict(dict)
    )
    # 遍历输入的嵌套字典中的每一个条目
    for index, s in data.items():
        # 遍历每个嵌套字典中的键值对
        for col, v in s.items():
            # 将值按照索引和列存储到新的默认字典结构中
            new_data[col][index] = v
    # 返回转换后的默认字典的字典结构
    return new_data


# 根据需要重新索引 DataFrame 或 Series 对象的值和引用
def _reindex_for_setitem(
    value: DataFrame | Series, index: Index
) -> tuple[ArrayLike, BlockValuesRefs | None]:
    # 如果索引匹配或索引长度为零，则直接返回当前值和引用
    if value.index.equals(index) or not len(index):
        if isinstance(value, Series):
            return value._values, value._references
        return value._values.copy(), None

    # 在需要时重新索引值
    try:
        reindexed_value = value.reindex(index)._values
    except ValueError as err:
        # 如果值在 MultiIndex.from_tuples 中引发异常，则检查是否索引不唯一
        if not value.index.is_unique:
            # 抛出值错误异常
            raise err

        # 如果索引不兼容，则抛出类型错误异常
        raise TypeError(
            "incompatible index of inserted column with frame index"
        ) from err
    # 返回重新索引后的值和空的引用
    return reindexed_value, None
```