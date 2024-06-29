# `D:\src\scipysrc\pandas\pandas\core\groupby\groupby.py`

```
"""
提供分组应用-组合范式中的分组功能。定义 GroupBy 类提供操作的基类。

SeriesGroupBy 和 DataFrameGroupBy 是其子类
（定义在 pandas.core.groupby.generic 中）
提供这些用户可见的对象，以提供特定功能。
"""

from __future__ import annotations

from collections.abc import (
    Callable,           # 可调用对象的抽象基类
    Hashable,           # 可散列对象的抽象基类
    Iterable,           # 可迭代对象的抽象基类
    Iterator,           # 迭代器对象的抽象基类
    Mapping,            # 映射对象的抽象基类
    Sequence,           # 序列对象的抽象基类
)
import datetime         # 日期时间处理模块
from functools import (
    partial,            # 偏函数工具
    wraps,              # 装饰器工具，用于保留函数的元数据
)
from textwrap import dedent  # 文本包装工具，用于缩减文本的缩进
from typing import (
    TYPE_CHECKING,      # 类型检查标记
    Literal,            # 字面值类型标记
    TypeVar,            # 类型变量标记
    Union,              # 联合类型标记
    cast,               # 类型强制转换工具
    final,              # 最终方法修饰器
    overload,           # 函数重载修饰器
)
import warnings         # 警告模块，用于处理警告信息

import numpy as np      # 数值计算工具包

from pandas._libs import (
    Timestamp,          # 时间戳类型
    lib,                # pandas 底层 C 实现的库
)
from pandas._libs.algos import rank_1d  # 排序算法
import pandas._libs.groupby as libgroupby  # pandas 底层分组实现
from pandas._libs.missing import NA    # 缺失值标识
from pandas._typing import (
    AnyArrayLike,       # 任意类数组类型
    ArrayLike,          # 数组类类型
    DtypeObj,           # 数据类型对象
    IndexLabel,         # 索引标签类型
    IntervalClosedType, # 区间闭合类型
    NDFrameT,           # NDFrame 类型
    PositionalIndexer,  # 位置索引器类型
    RandomState,        # 随机状态类型
    npt,                # numpy 类型
)
from pandas.compat.numpy import function as nv  # 兼容 numpy 的函数
from pandas.errors import (
    AbstractMethodError,    # 抽象方法错误
    DataError,              # 数据错误
)
from pandas.util._decorators import (
    Appender,           # 追加装饰器
    Substitution,       # 替换装饰器
    cache_readonly,     # 只读缓存装饰器
    doc,                # 文档装饰器
)
from pandas.util._exceptions import find_stack_level  # 查找堆栈层级异常

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,       # 强制索引器数据类型转换
    ensure_dtype_can_hold_na,   # 确保数据类型可以包含 NA 值
)
from pandas.core.dtypes.common import (
    is_bool_dtype,              # 是否为布尔类型
    is_float_dtype,             # 是否为浮点类型
    is_hashable,                # 是否为可散列对象
    is_integer,                 # 是否为整数
    is_integer_dtype,           # 是否为整数类型
    is_list_like,               # 是否为类列表对象
    is_numeric_dtype,           # 是否为数值类型
    is_object_dtype,            # 是否为对象类型
    is_scalar,                  # 是否为标量
    needs_i8_conversion,        # 是否需要 i8 转换
    pandas_dtype,               # pandas 数据类型
)
from pandas.core.dtypes.missing import (
    isna,                       # 是否为缺失值
    na_value_for_dtype,         # 返回指定数据类型的缺失值
    notna,                      # 是否不为缺失值
)

from pandas.core import (
    algorithms,     # 算法模块
    sample,         # 抽样模块
)
from pandas.core._numba import executor  # numba 执行器
from pandas.core.arrays import (
    ArrowExtensionArray,            # Arrow 扩展数组
    BaseMaskedArray,                # 基础掩码数组
    ExtensionArray,                 # 扩展数组
    FloatingArray,                  # 浮点数数组
    IntegerArray,                   # 整数数组
    SparseArray,                    # 稀疏数组
)
from pandas.core.arrays.string_ import StringDtype  # 字符串类型
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,               # Arrow 字符串数组
    ArrowStringArrayNumpySemantics, # Arrow 字符串数组的 numpy 语义
)
from pandas.core.base import (
    PandasObject,       # Pandas 对象基类
    SelectionMixin,    # 选择混合类
)
import pandas.core.common as com  # pandas 核心通用工具
from pandas.core.frame import DataFrame  # DataFrame 类
from pandas.core.generic import NDFrame  # 泛型 NDFrame 类
from pandas.core.groupby import (
    base,       # 分组基类
    numba_,     # numba 实现
    ops,        # 操作实现
)
from pandas.core.groupby.grouper import get_grouper  # 获取分组器函数
from pandas.core.groupby.indexing import (
    GroupByIndexingMixin,       # 分组索引混合类
    GroupByNthSelector,         # 分组第 N 选择器类
)
from pandas.core.indexes.api import (
    Index,          # 索引类
    MultiIndex,     # 多重索引类
    default_index,  # 默认索引
)
from pandas.core.internals.blocks import ensure_block_shape  # 确保块形状函数
from pandas.core.series import Series     # Series 类
from pandas.core.sorting import get_group_index_sorter  # 获取分组索引排序函数
from pandas.core.util.numba_ import (
    get_jit_arguments,      # 获取 jit 参数函数
    maybe_use_numba,        # 可能使用 numba 函数
)

if TYPE_CHECKING:
    from pandas._libs.tslibs import BaseOffset  # 时间序列库中的基本偏移量类型
    # 从 pandas._typing 模块中导入指定类型
    from pandas._typing import (
        Any,
        Concatenate,
        P,
        Self,
        T,
    )

    # 从 pandas.core.indexers.objects 模块中导入 BaseIndexer 类
    from pandas.core.indexers.objects import BaseIndexer

    # 从 pandas.core.resample 模块中导入 Resampler 类
    from pandas.core.resample import Resampler

    # 从 pandas.core.window 模块中导入以下窗口类型
    # ExpandingGroupby：扩展组合聚合窗口
    # ExponentialMovingWindowGroupby：指数移动窗口组合聚合
    # RollingGroupby：滚动组合聚合窗口
    from pandas.core.window import (
        ExpandingGroupby,
        ExponentialMovingWindowGroupby,
        RollingGroupby,
    )
# 定义一个多行字符串，用于显示相关方法的 "See Also" 部分的帮助文档
_common_see_also = """
        See Also
        --------
        Series.%(name)s : Apply a function %(name)s to a Series.
        DataFrame.%(name)s : Apply a function %(name)s
            to each row or column of a DataFrame.
"""

# 定义一个模板字符串，用于生成关于 groupby 聚合方法的帮助文档
_groupby_agg_method_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns.

    .. versionchanged:: 2.0.0
        numeric_only no longer accepts ``None``.

min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.

Examples
--------
{example}
"""

# 定义一个模板字符串，用于生成关于 groupby 聚合方法的引擎选择的帮助文档
_groupby_agg_method_engine_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns.

    .. versionchanged:: 2.0.0
        numeric_only no longer accepts ``None``.

min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

engine : str, default None {e}
    * ``'cython'`` : Runs rolling apply through C-extensions from cython.
    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
        Only available when ``raw`` is set to ``True``.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None {ek}
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
        and ``parallel`` dictionary keys. The values must either be ``True`` or
        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
        applied to both the ``func`` and the ``apply`` groupby aggregation.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.

Examples
--------
{example}
"""

# 定义一个模板字符串，用于生成关于 .pipe 方法的帮助文档
_pipe_template = """
Apply a ``func`` with arguments to this %(klass)s object and return its result.

Use `.pipe` when you want to improve readability by chaining together
functions that expect Series, DataFrames, GroupBy or Resampler objects.
Instead of writing

>>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
>>> g = lambda x, arg1: x * 5 / arg1
>>> f = lambda x: x ** 4
>>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])
>>> h(g(f(df.groupby('group')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

You can write

>>> (df.groupby('group')
...    .pipe(f)
...    .pipe(g, arg1=1)
...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP

which is much more readable.

Parameters
----------
func : callable or tuple of (callable, str)
    Function to apply to this %(klass)s object or, alternatively,
    # 定义一个元组 `(callable, data_keyword)`，其中 `data_keyword` 是一个字符串，表示 `callable` 所需的关键字，该关键字期望是 %(klass)s 对象。
    a `(callable, data_keyword)` tuple where `data_keyword` is a
    string indicating the keyword of `callable` that expects the
    %(klass)s object.
*args : iterable, optional
       位置参数，传递给 `func` 函数的可迭代对象。

**kwargs : dict, optional
         关键字参数，传递给 `func` 函数的字典对象。

Returns
-------
%(klass)s
    应用了函数 `func` 后的原始对象。

See Also
--------
Series.pipe : 将带有参数的函数应用到系列对象上。
DataFrame.pipe: 将带有参数的函数应用到数据帧上。
apply : 将函数应用于每个分组而不是整个 %(klass)s 对象。

Notes
-----
详细信息请参阅 `这里
<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

Examples
--------
%(examples)s
"""

_transform_template = """
对每个分组调用生成一个同索引 %(klass)s 的函数。

返回一个 %(klass)s，其索引与原始对象相同，填充了转换后的值。

Parameters
----------
func : function, str
    要应用于每个分组的函数。有关要求，请参见下面的注释部分。

    可接受的输入为：

    - 字符串
    - Python 函数
    - 带 ``engine='numba'`` 参数指定的 Numba JIT 函数。

    仅支持传递单个函数与此引擎一起使用。
    如果选择 ``'numba'`` 引擎，则函数必须是带有 ``values`` 和 ``index``
    作为函数签名中的第一个和第二个参数的用户定义函数。
    每个组的索引将传递给用户定义函数，并可用于使用。

    如果选择字符串，则必须是要使用的 groupby 方法的名称。
*args
    传递给 func 的位置参数。
engine : str, default None
    * ``'cython'`` : 通过来自 Cython 的 C 扩展运行函数。
    * ``'numba'`` : 通过来自 Numba 的 JIT 编译代码运行函数。
    * ``None`` : 默认为 ``'cython'`` 或全局设置 ``compute.use_numba``。

engine_kwargs : dict, default None
    * 对于 ``'cython'`` 引擎，不接受 ``engine_kwargs``。
    * 对于 ``'numba'`` 引擎，引擎可以接受 ``nopython``、``nogil``
      和 ``parallel`` 字典键。值必须为 ``True`` 或 ``False``。
      ``'numba'`` 引擎的默认 ``engine_kwargs`` 为
      ``{'nopython': True, 'nogil': False, 'parallel': False}``，并将应用于函数。

**kwargs
    传递给 func 的关键字参数。

Returns
-------
%(klass)s
    具有与原始对象相同索引的 %(klass)s，填充了转换后的值。

See Also
--------
%(klass)s.groupby.apply : 对每个分组应用函数 ``func`` 并将结果组合在一起。
%(klass)s.groupby.aggregate : 使用一个或多个操作进行聚合。
%(klass)s.transform : 对自身调用 ``func``，生成与自身相同轴形状的 %(klass)s。

Notes
-----
每个组都有属性 'name'，以便您知道正在处理哪个组。

当前实现对函数 f 强加了三个要求：
"""
* f must return a value that either has the same shape as the input
  subframe or can be broadcast to the shape of the input subframe.
  For example, if `f` returns a scalar it will be broadcast to have the
  same shape as the input subframe.
* if this is a DataFrame, f must support application column-by-column
  in the subframe. If f also supports application to the entire subframe,
  then a fast path is used starting from the second chunk.
* f must not mutate groups. Mutation is not supported and may
  produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.

When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.

.. versionchanged:: 2.0.0

    When using ``.transform`` on a grouped DataFrame and the transformation function
    returns a DataFrame, pandas now aligns the result's index
    with the input's index. You can call ``.to_numpy()`` on the
    result of the transformation function to avoid alignment.

Examples
--------
%(example)s"""

_agg_template_series = """
Aggregate using one or more operations.

Parameters
----------
func : function, str, list, dict or None
    Function to use for aggregating the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
    - None, in which case ``**kwargs`` are used with Named Aggregation. Here the
      output has one column for each element in ``**kwargs``. The name of the
      column is keyword, whereas the value determines the aggregation used to compute
      the values in the column.

      Can also accept a Numba JIT function with
      ``engine='numba'`` specified. Only passing a single function is supported
      with this engine.

      If the ``'numba'`` engine is chosen, the function must be
      a user defined function with ``values`` and ``index`` as the
      first and second arguments respectively in the function signature.
      Each group's index will be passed to the user defined function
      and optionally available for use.

    .. deprecated:: 2.1.0

        Passing a dictionary is deprecated and will raise in a future version
        of pandas. Pass a list of aggregations instead.
*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None

"""
    * 对于 `'cython'` 引擎，不接受任何 `engine_kwargs`。
    * 对于 `'numba'` 引擎，该引擎可以接受 `nopython`、`nogil` 和 `parallel` 作为字典键。这些键的值必须是 `True` 或 `False`。`'numba'` 引擎的默认 `engine_kwargs` 是 `{'nopython': True, 'nogil': False, 'parallel': False}`，将应用于函数中。
"""
**kwargs
    * 如果 ``func`` 为 None，则使用 ``**kwargs`` 定义输出名称和聚合方式，通过命名聚合来完成。参见 ``func`` 条目。
    * 否则，这些关键字参数将传递给 func。

Returns
-------
{klass}
    返回一个 {klass} 对象。

See Also
--------
{klass}.groupby.apply : 对每个分组应用函数 func 并将结果组合在一起。
{klass}.groupby.transform : 根据给定的函数对每个分组中的 Series 进行变换。
{klass}.aggregate : 使用一个或多个操作进行聚合。

Notes
-----
当使用 ``engine='numba'`` 时，内部将不会有"回退"行为。
分组数据和分组索引将作为 numpy 数组传递给 JIT 编译的用户定义函数，不会尝试其他执行方式。

对传入对象进行变异的函数可能会导致意外行为或错误，并且不受支持。详见 :ref:`gotchas.udf-mutation` 获取更多信息。

.. versionchanged:: 1.3.0

    结果 dtype 将反映传递的 ``func`` 的返回值类型，参见下面的示例。
{examples}"""

_agg_template_frame = """
Aggregate using one or more operations.

Parameters
----------
func : function, str, list, dict or None
    用于聚合数据的函数。如果是函数，必须能够处理一个 {klass} 对象或者能够传递给 {klass}.apply。

    可接受的组合包括：

    - 函数
    - 字符串函数名
    - 函数和/或函数名列表，例如 ``[np.sum, 'mean']``
    - 索引标签 -> 函数、函数名或这些的列表的字典
    - None，此时使用 ``**kwargs`` 并通过命名聚合。输出将有一个列用于 ``**kwargs`` 中的每个元素。
      列名为关键字，值决定用于计算列中值的聚合方式。

      还可以接受一个带有 ``engine='numba'`` 的 Numba JIT 函数。只支持传递单个函数。

      如果选择 ``'numba'`` 引擎，则函数必须是一个带有 ``values`` 和 ``index`` 作为函数签名中的第一个和第二个参数的用户定义函数。
      每个分组的索引将传递给用户定义的函数，并可选用于操作。

*args
    传递给 func 的位置参数。
engine : str, 默认 None
    * ``'cython'`` : 通过 Cython 中的 C 扩展运行函数。
    * ``'numba'`` : 通过 Numba 中的 JIT 编译代码运行函数。
    * ``None`` : 默认为 ``'cython'`` 或全局设置的 ``compute.use_numba``。

engine_kwargs : dict, 默认 None
    * 对于 ``'cython'`` 引擎，不接受 ``engine_kwargs``。

"""
    * 对于 'numba' 引擎，该引擎可以接受 'nopython'、'nogil' 和 'parallel' 三个字典键。这些键的值必须是布尔值 True 或 False。
      'numba' 引擎的默认 'engine_kwargs' 是 {'nopython': True, 'nogil': False, 'parallel': False}，将会应用于函数。
# 根据传入的关键字参数设置输出的名称和聚合操作，若 func 为 None，则通过命名聚合来定义。参见 func 条目。
# 否则，将关键字参数传递给 func。
**kwargs
    * 如果 ``func`` 为 None，则使用 ``**kwargs`` 来定义输出名称和聚合操作，通过命名聚合来实现。参见 ``func`` 条目。
    * 否则，传递给 func 的关键字参数。

Returns
-------
{klass}
    返回一个 {klass} 对象。

See Also
--------
{klass}.groupby.apply : 对每个组应用函数 func，并将结果组合在一起。
{klass}.groupby.transform : 根据给定的函数，对每个组中的 Series 进行转换。
{klass}.aggregate : 使用一个或多个操作进行聚合。

Notes
-----
当使用 ``engine='numba'`` 时，内部不会有“回退”行为。
组数据和组索引将作为 numpy 数组传递给 JIT 编译的用户定义函数，并且不会尝试其他执行方式。

会改变传入对象的函数可能会产生意外的行为或错误，并且不受支持。详见 :ref:`gotchas.udf-mutation` 获取更多详情。

.. versionchanged:: 1.3.0
    结果的 dtype 将反映传递给 ``func`` 的返回值类型，参见下面的示例。
{examples}
    示例。
"""


@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: GroupBy) -> None:
        self._groupby = groupby

    def __call__(self, *args, **kwargs):
        def f(self):
            return self.plot(*args, **kwargs)

        f.__name__ = "plot"
        return self._groupby._python_apply_general(f, self._groupby._selected_obj)

    def __getattr__(self, name: str):
        def attr(*args, **kwargs):
            def f(self):
                return getattr(self.plot, name)(*args, **kwargs)

            return self._groupby._python_apply_general(f, self._groupby._selected_obj)

        return attr


_KeysArgType = Union[
    Hashable,
    list[Hashable],
    Callable[[Hashable], Hashable],
    list[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]


class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {
        "as_index",
        "dropna",
        "exclusions",
        "grouper",
        "group_keys",
        "keys",
        "level",
        "obj",
        "observed",
        "sort",
    }

    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None = None
    level: IndexLabel | None = None
    group_keys: bool

    @final
    def __len__(self) -> int:
        return self._grouper.ngroups

    @final
    def __repr__(self) -> str:
        # TODO: Better repr for GroupBy object
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> dict[Hashable, Index]:
        """
        返回一个字典，包含组名到组标签的映射。

        示例
        --------

        对于 SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).groups
        {'a': ['a', 'a'], 'b': ['b']}

        对于 DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> df.groupby(by=["a"]).groups
        {1: [0, 1], 7: [2]}

        对于 Resampler:

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
        >>> ser.resample("MS").groups
        {Timestamp('2023-01-01 00:00:00'): 2, Timestamp('2023-02-01 00:00:00'): 4}
        """
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        """
        返回分组数。

        """
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        返回一个字典，包含组名到组索引数组的映射。

        示例
        --------

        对于 SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).indices
        {'a': array([0, 1]), 'b': array([2])}

        对于 DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).indices
        {1: array([0, 1]), 7: array([2])}

        对于 Resampler:

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
        >>> ser.resample("MS").indices
        defaultdict(<class 'list'>, {Timestamp('2023-01-01 00:00:00'): [0, 1],
        Timestamp('2023-02-01 00:00:00'): [2, 3]})
        """
        return self._grouper.indices
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """

        def get_converter(s):
            # 定义一个内部函数，根据不同的日期类型返回对应的转换函数
            if isinstance(s, datetime.datetime):
                return lambda key: Timestamp(key)
            elif isinstance(s, np.datetime64):
                return lambda key: Timestamp(key).asm8
            else:
                return lambda key: key

        if len(names) == 0:
            return []

        if len(self.indices) > 0:
            index_sample = next(iter(self.indices))
        else:
            index_sample = None  # Dummy sample

        name_sample = names[0]
        if isinstance(index_sample, tuple):
            # 如果索引样本是一个元组，并且名字样本不是元组，则抛出值错误
            if not isinstance(name_sample, tuple):
                msg = "must supply a tuple to get_group with multiple grouping keys"
                raise ValueError(msg)
            # 如果名字样本长度与索引样本长度不相等，则尝试返回原始分组键的索引列表
            if not len(name_sample) == len(index_sample):
                try:
                    # 如果原始分组键是一个元组
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    # 原来它不是一个元组
                    msg = (
                        "must supply a same-length tuple to get_group "
                        "with multiple grouping keys"
                    )
                    raise ValueError(msg) from err

            # 使用索引样本中的每个元素生成转换器
            converters = (get_converter(s) for s in index_sample)
            # 对于每个名字，应用相应的转换器并返回新的元组列表
            names = (tuple(f(n) for f, n in zip(converters, name)) for name in names)

        else:
            # 否则，使用索引样本生成一个转换器并应用于每个名字
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)

        # 返回每个名字对应的索引列表，如果没有匹配则返回空列表
        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
        # 返回指定名字的索引，作为列表的第一个元素
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        # 注意：对于 SeriesGroupBy，_selected_obj 始终是 `self.obj`
        if isinstance(self.obj, Series):
            return self.obj

        if self._selection is not None:
            if is_hashable(self._selection):
                # 即一个单一的键，因此选择它将返回一个 Series。
                # 在这种情况下，_obj_with_exclusions 将键包装在列表中，并返回一个单列 DataFrame。
                return self.obj[self._selection]

            # 否则 _selection 等同于 _selection_list，所以 _selected_obj 与 _obj_with_exclusions 匹配，可以重用它而避免复制。
            return self._obj_with_exclusions

        # 返回原始对象 self.obj
        return self.obj

    @final
    def _dir_additions(self) -> set[str]:
        # 返回 self.obj 的 _dir_additions 方法的结果
        return self.obj._dir_additions()

    @overload
    # 定义类方法 pipe，用于数据处理管道操作
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # 返回 com 模块的 pipe 函数的结果，传入当前对象 self、处理函数 func、额外参数 args 和关键字参数 kwargs
        return com.pipe(self, func, *args, **kwargs)
    # 定义一个方法用于从给定名称的组中构造 DataFrame 或 Series

    keys = self.keys
    level = self.level
    # 获取对象的键和级别信息

    # mypy 无法识别 level/keys 在传递给 len 时作为大小的一部分
    if (is_list_like(level) and len(level) == 1) or (  # type: ignore[arg-type]
        is_list_like(keys) and len(keys) == 1  # type: ignore[arg-type]
    ):
        # 检查级别或键是否为单一值，若是则执行以下操作
        if isinstance(name, tuple) and len(name) == 1:
            # 如果 name 是元组且长度为 1，则将其转换为单一值
            name = name[0]
        else:
            # 否则抛出 KeyError
            raise KeyError(name)

    inds = self._get_index(name)
    # 获取指定名称的索引

    if not len(inds):
        # 如果索引长度为 0，则抛出 KeyError
        raise KeyError(name)

    return self._selected_obj.iloc[inds]
    # 返回根据索引选择的对象的切片
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator.

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> for x, y in ser.groupby(level=0):
        ...     print(f"{x}\\n{y}\\n")
        a
        a    1
        a    2
        dtype: int64
        b
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> for x, y in df.groupby(by=["a"]):
        ...     print(f"{x}\\n{y}\\n")
        (1,)
           a  b  c
        0  1  2  3
        1  1  5  6
        (7,)
           a  b  c
        2  7  8  9

        For Resampler:

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
        >>> for x, y in ser.resample("MS"):
        ...     print(f"{x}\\n{y}\\n")
        2023-01-01 00:00:00
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        2023-02-01 00:00:00
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        """
        # 获取分组的关键字
        keys = self.keys
        # 获取分组的层级
        level = self.level
        # 获取分组迭代器
        result = self._grouper.get_iterator(self._selected_obj)
        # 如果 level 是列表形式且长度为1，或者 keys 是列表形式且长度为1，则生成单元素元组的结果
        if (
            (is_list_like(level) and len(level) == 1)  # type: ignore[arg-type]
            or (isinstance(keys, list) and len(keys) == 1)
        ):
            # 当 keys 是列表时，即使长度为1，也返回元组形式的结果
            result = (((key,), group) for key, group in result)
        # 返回分组结果的迭代器
        return result
# 定义一个类型变量，用于指示输出为 DataFrame 或 Series 类型
OutputFrameOrSeries = TypeVar("OutputFrameOrSeries", bound=NDFrame)

class GroupBy(BaseGroupBy[NDFrameT]):
    """
    Class for grouping and aggregating relational data.

    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.groupby(...) to use GroupBy, but you can also do:

    ::

        grouped = groupby(obj, ...)

    Parameters
    ----------
    obj : pandas object
        要分组的 pandas 对象，可以是 DataFrame 或 Series
    level : int, default None
        多级索引的级别
    groupings : list of Grouping objects
        大多数用户应该忽略这个参数
    exclusions : array-like, optional
        要排除的列的列表
    name : str
        大多数用户应该忽略这个参数

    Returns
    -------
    **Attributes**
    groups : dict
        分组结果的字典，键为分组名，值为分组的标签
    len(grouped) : int
        分组后的组数目

    Notes
    -----
    After grouping, see aggregate, apply, and transform functions. Here are
    some other brief notes about usage. When grouping by multiple groups, the
    result index will be a MultiIndex (hierarchical) by default.

    Iteration produces (key, group) tuples, i.e. chunking the data by group. So
    you can write code like:

    ::

        grouped = obj.groupby(keys)
        for key, group in grouped:
            # do something with the data

    Function calls on GroupBy, if not specially implemented, "dispatch" to the
    grouped data. So if you group a DataFrame and wish to invoke the std()
    method on each group, you can simply do:

    ::

        df.groupby(mapper).std()

    rather than

    ::

        df.groupby(mapper).aggregate(np.std)

    You can pass arguments to these "wrapped" functions, too.

    See the online documentation for full exposition on these topics and much
    more
    """

    _grouper: ops.BaseGrouper
    as_index: bool

    @final
    def __init__(
        self,
        obj: NDFrameT,
        keys: _KeysArgType | None = None,
        level: IndexLabel | None = None,
        grouper: ops.BaseGrouper | None = None,
        exclusions: frozenset[Hashable] | None = None,
        selection: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the GroupBy object.

        Parameters
        ----------
        obj : NDFrameT
            The pandas object to be grouped, can be DataFrame or Series.
        keys : _KeysArgType | None, optional
            Grouping keys, by default None.
        level : IndexLabel | None, optional
            Level of MultiIndex, by default None.
        grouper : ops.BaseGrouper | None, optional
            Grouping information, by default None.
        exclusions : frozenset[Hashable] | None, optional
            Columns to exclude from grouping, by default None.
        selection : IndexLabel | None, optional
            Selection of data, by default None.
        as_index : bool, optional
            Whether to return the grouped object as DataFrame with index or not, by default True.
        sort : bool, optional
            Whether to sort the group keys, by default True.
        group_keys : bool, optional
            Whether to add group keys to the resulting index, by default True.
        observed : bool, optional
            Whether to use the observed values for categorical data, by default False.
        dropna : bool, optional
            Whether to exclude NA/null values when grouping, by default True.
        **kwargs : Any, optional
            Additional keyword arguments for customization.
        """
        # Initialize the BaseGroupBy object
        super().__init__(
            obj,
            keys=keys,
            level=level,
            grouper=grouper,
            exclusions=exclusions,
            selection=selection,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
            **kwargs,
        )
    ) -> None:
        self._selection = selection  # 初始化选择器的值

        assert isinstance(obj, NDFrame), type(obj)  # 断言确保 obj 是 NDFrame 类型

        self.level = level  # 设置分组级别
        self.as_index = as_index  # 指定是否按照分组键作为索引
        self.keys = keys  # 指定用于分组的键
        self.sort = sort  # 指定是否对分组键进行排序
        self.group_keys = group_keys  # 指定是否显示分组键
        self.dropna = dropna  # 指定是否丢弃缺失值

        if grouper is None:
            # 如果未指定 grouper，则调用 get_grouper 获取分组器对象
            grouper, exclusions, obj = get_grouper(
                obj,
                keys,
                level=level,
                sort=sort,
                observed=observed,
                dropna=self.dropna,
            )

        self.observed = observed  # 设置是否观察到所有可能的分组键
        self.obj = obj  # 设置当前操作的对象
        self._grouper = grouper  # 设置分组器对象
        self.exclusions = frozenset(exclusions) if exclusions else frozenset()  # 设置要排除的分组键集合

    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)  # 返回对象的属性
        if attr in self.obj:
            return self[attr]  # 返回分组后的对象中的属性

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )  # 抛出属性错误异常

    @final
    def _op_via_apply(self, name: str, *args, **kwargs):
        """Compute the result of an operation by using GroupBy's apply."""
        f = getattr(type(self._obj_with_exclusions), name)  # 获取对象的指定属性名对应的方法

        def curried(x):
            return f(x, *args, **kwargs)  # 执行指定方法并返回结果

        # 保留方法名，以便在调用绘图方法时检测到，避免重复
        curried.__name__ = name

        # 对于特殊情况，避免在捕获异常时创建额外的绘图
        if name in base.plotting_methods:
            return self._python_apply_general(curried, self._selected_obj)

        is_transform = name in base.transformation_kernels
        # 使用通用的 Python 应用方法进行操作
        result = self._python_apply_general(
            curried,
            self._obj_with_exclusions,
            is_transform=is_transform,
            not_indexed_same=not is_transform,
        )

        if self._grouper.has_dropped_na and is_transform:
            # 如果分组器删除了 NaN 值，结果将缺少部分行，需要用空值填充，并确保索引与输入相同
            result = self._set_result_index_ordered(result)
        return result

    # -----------------------------------------------------------------
    # Dispatch/Wrapping

    @final
    def _concat_objects(
        self,
        values,
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ):
        # 从 pandas 核心库中导入连接函数 concat
        from pandas.core.reshape.concat import concat

        # 如果设置了分组键且不是变换操作
        if self.group_keys and not is_transform:
            # 如果设置了作为索引
            if self.as_index:
                # 获取分组键、分组水平和分组名称
                group_keys = self._grouper.result_index
                group_levels = self._grouper.levels
                group_names = self._grouper.names

                # 使用 concat 函数按行连接数据
                result = concat(
                    values,
                    axis=0,
                    keys=group_keys,
                    levels=group_levels,
                    names=group_names,
                    sort=False,
                )
            else:
                # 否则直接按行连接数据
                result = concat(values, axis=0)

        # 如果不是同一索引
        elif not not_indexed_same:
            # 按行连接数据
            result = concat(values, axis=0)

            # 获取当前选定对象的索引
            ax = self._selected_obj.index
            # 如果需要删除缺失值
            if self.dropna:
                # 获取分组标签和掩码
                labels = self._grouper.ids
                mask = labels != -1
                ax = ax[mask]

            # 当索引存在重复时，无法使用 reindex 恢复原始顺序
            # 因此使用以下方法处理
            if ax.has_duplicates and not result.axes[0].equals(ax):
                # 获取目标索引，并根据目标索引获取索引器
                target = algorithms.unique1d(ax._values)
                indexer, _ = result.index.get_indexer_non_unique(target)
                # 按照索引器重新排序结果
                result = result.take(indexer, axis=0)
            else:
                # 否则按照索引重新排列结果
                result = result.reindex(ax, axis=0)

        else:
            # 否则直接按行连接数据
            result = concat(values, axis=0)

        # 如果对象维度为 1
        if self.obj.ndim == 1:
            # 获取对象的名称
            name = self.obj.name
        # 如果选定对象可哈希
        elif is_hashable(self._selection):
            # 设置名称为选定对象
            name = self._selection
        else:
            # 否则名称为空
            name = None

        # 如果结果为 Series 类型且名称不为空
        if isinstance(result, Series) and name is not None:
            # 设置结果的名称
            result.name = name

        # 返回结果
        return result

    # 定义函数 _set_result_index_ordered，接受 result 参数
    @final
    def _set_result_index_ordered(
        self, result: OutputFrameOrSeries
    ) -> OutputFrameOrSeries:
        # 设置结果的索引为对象的索引
        index = self.obj.index

        if self._grouper.is_monotonic and not self._grouper.has_dropped_na:
            # 如果分组器已经是单调的且没有删除缺失值，则使用快速路径
            result = result.set_axis(index, axis=0)
            return result

        # 如果行顺序被打乱 => 按照原始索引中的位置对行进行排序
        original_positions = Index(self._grouper.result_ilocs)
        result = result.set_axis(original_positions, axis=0)
        result = result.sort_index(axis=0)
        if self._grouper.has_dropped_na:
            # 由于删除了缺失值，需要补回任何缺失的行 - 这里的索引是整数，可以使用 RangeIndex
            result = result.reindex(default_index(len(index)), axis=0)
        result = result.set_axis(index, axis=0)

        return result

    @final
    def _insert_inaxis_grouper(
        self, result: Series | DataFrame, qs: npt.NDArray[np.float64] | None = None
    ) -> DataFrame:
        if isinstance(result, Series):
            # 如果结果是 Series，则转换为 DataFrame
            result = result.to_frame()

        n_groupings = len(self._grouper.groupings)

        if qs is not None:
            # 如果存在 qs，则在位置 0 处插入一个新的列，列名为 "level_n"，n 表示分组的数量
            result.insert(
                0, f"level_{n_groupings}", np.tile(qs, len(result) // len(qs))
            )

        # 反向遍历分组器的名称和级别，以便始终在位置 0 处插入
        for level, (name, lev) in enumerate(
            zip(
                reversed(self._grouper.names),
                self._grouper.get_group_levels(),
            )
        ):
            if name is None:
                # 当级别未命名时，行为类似于 .reset_index()
                name = (
                    "index"
                    if n_groupings == 1 and qs is None
                    else f"level_{n_groupings - level - 1}"
                )

            # GH #28549
            # 当使用 .apply(-) 时，name 已经存在于列中
            if name not in result.columns:
                # 如果不是 in_axis：
                if qs is None:
                    result.insert(0, name, lev)
                else:
                    result.insert(0, name, Index(np.repeat(lev, len(qs))))

        return result

    @final
    def _wrap_aggregated_output(
        self,
        result: Series | DataFrame,
        qs: npt.NDArray[np.float64] | None = None,
    ):
        """
        Wraps the output of GroupBy aggregations into the expected result.

        Parameters
        ----------
        result : Series, DataFrame
            The output from GroupBy aggregation, either a Series or DataFrame.

        Returns
        -------
        Series or DataFrame
            Aggregated result wrapped into a Series or DataFrame based on input type.
        """
        # ATM we do not get here for SeriesGroupBy; when we do, we will
        # need to require that result.name already match self.obj.name

        if not self.as_index:
            # `not self.as_index` is only relevant for DataFrameGroupBy,
            # enforced in __init__
            result = self._insert_inaxis_grouper(result, qs=qs)
            result = result._consolidate()
            result.index = default_index(len(result))

        else:
            index = self._grouper.result_index
            if qs is not None:
                # We get here with len(qs) != 1 and not self.as_index
                # in test_pass_args_kwargs
                index = _insert_quantile_level(index, qs)
            result.index = index

        return result

    def _wrap_applied_output(
        self,
        data,
        values: list,
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ):
        """
        Placeholder method raising an error when called.

        Parameters
        ----------
        data : object
            Data to be processed.
        values : list
            List of values.
        not_indexed_same : bool, optional
            Indicator if indexes are different, by default False.
        is_transform : bool, optional
            Indicator if transformation is applied, by default False.

        Raises
        ------
        AbstractMethodError
            Always raises this error when called.
        """
        raise AbstractMethodError(self)

    # -----------------------------------------------------------------
    # numba

    @final
    def _numba_prep(self, data: DataFrame):
        """
        Prepare data for numba-based aggregation.

        Parameters
        ----------
        data : DataFrame
            The data to be prepared for aggregation.

        Returns
        -------
        tuple
            Tuple containing starts, ends, sorted index data, and sorted data arrays.
        """
        ngroups = self._grouper.ngroups
        sorted_index = self._grouper.result_ilocs
        sorted_ids = self._grouper._sorted_ids

        sorted_data = data.take(sorted_index, axis=0).to_numpy()
        # GH 46867
        index_data = data.index
        if isinstance(index_data, MultiIndex):
            if len(self._grouper.groupings) > 1:
                raise NotImplementedError(
                    "Grouping with more than 1 grouping labels and "
                    "a MultiIndex is not supported with engine='numba'"
                )
            group_key = self._grouper.groupings[0].name
            index_data = index_data.get_level_values(group_key)
        sorted_index_data = index_data.take(sorted_index).to_numpy()

        starts, ends = lib.generate_slices(sorted_ids, ngroups)
        return (
            starts,
            ends,
            sorted_index_data,
            sorted_data,
        )

    def _numba_agg_general(
        self,
        func: Callable,
        dtype_mapping: dict[np.dtype, Any],
        engine_kwargs: dict[str, bool] | None,
        **aggregator_kwargs,
    ):
        """
        Perform general numba-based aggregation.

        Parameters
        ----------
        func : callable
            Function to be applied during aggregation.
        dtype_mapping : dict
            Mapping of data types.
        engine_kwargs : dict or None
            Optional keyword arguments for the engine.
        **aggregator_kwargs : dict
            Additional keyword arguments for the aggregator.

        Returns
        -------
        object
            Aggregated result based on the function and arguments provided.
        """
    ):
        """
        Perform groupby with a standard numerical aggregation function (e.g. mean)
        with Numba.
        """
        if not self.as_index:
            # 如果不是按照索引分组，抛出 NotImplementedError
            raise NotImplementedError(
                "as_index=False is not supported. Use .reset_index() instead."
            )

        data = self._obj_with_exclusions
        # 如果数据是一维的，将其转换为 DataFrame 对象
        df = data if data.ndim == 2 else data.to_frame()

        # 生成共享的聚合器，使用 Numba 执行
        aggregator = executor.generate_shared_aggregator(
            func,
            dtype_mapping,
            True,  # is_grouped_kernel
            **get_jit_arguments(engine_kwargs),
        )

        # 直接将分组的 IDs 传递给内核（如果内核能够处理），这比排序更快
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups

        # 应用聚合器到数据的管理器上
        res_mgr = df._mgr.apply(
            aggregator, labels=ids, ngroups=ngroups, **aggregator_kwargs
        )

        # 更新结果的列索引为分组的结果索引
        res_mgr.axes[1] = self._grouper.result_index

        # 从管理器中构造结果 DataFrame
        result = df._constructor_from_mgr(res_mgr, axes=res_mgr.axes)

        # 如果数据是一维的，压缩结果到一列，并设置结果的名称
        if data.ndim == 1:
            result = result.squeeze("columns")
            result.name = data.name
        else:
            # 如果数据是二维的，设置结果的列名
            result.columns = data.columns

        return result

    @final
    def _transform_with_numba(self, func, *args, engine_kwargs=None, **kwargs):
        """
        Perform groupby transform routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
        data = self._obj_with_exclusions
        # 获取结果排序后的索引
        index_sorting = self._grouper.result_ilocs
        # 如果数据是一维的，转换为 DataFrame 对象
        df = data if data.ndim == 2 else data.to_frame()

        # 准备数据以供 Numba 使用，获取起始点、结束点、排序后的索引和数据
        starts, ends, sorted_index, sorted_data = self._numba_prep(df)

        # 验证用户定义的函数是否可以使用 Numba 运行
        numba_.validate_udf(func)

        # 生成 Numba 加速的转换函数
        numba_transform_func = numba_.generate_numba_transform_func(
            func, **get_jit_arguments(engine_kwargs, kwargs)
        )

        # 调用 Numba 加速的转换函数处理数据
        result = numba_transform_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(df.columns),
            *args,
        )

        # 将结果按照原始排序重新排序，以便恢复原始数据的顺序
        result = result.take(np.argsort(index_sorting), axis=0)

        # 获取原始数据的索引
        index = data.index

        # 如果数据是一维的，设置结果的名称
        if data.ndim == 1:
            result_kwargs = {"name": data.name}
            result = result.ravel()
        else:
            # 如果数据是二维的，设置结果的列名
            result_kwargs = {"columns": data.columns}

        # 构造最终的结果对象
        return data._constructor(result, index=index, **result_kwargs)

    @final
    def _aggregate_with_numba(self, func, *args, engine_kwargs=None, **kwargs):
        """
        Perform groupby aggregation routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
        # 获取带有排除项的对象数据
        data = self._obj_with_exclusions
        # 将数据转换为DataFrame格式（如果不是），以便进行处理
        df = data if data.ndim == 2 else data.to_frame()

        # 准备数据并进行排序以便使用Numba引擎
        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        
        # 验证用户定义的函数是否符合Numba的要求
        numba_.validate_udf(func)
        
        # 生成用于Numba聚合函数的函数
        numba_agg_func = numba_.generate_numba_agg_func(
            func, **get_jit_arguments(engine_kwargs, kwargs)
        )
        
        # 调用Numba聚合函数处理数据
        result = numba_agg_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(df.columns),
            *args,
        )
        
        # 获取分组结果的索引
        index = self._grouper.result_index
        
        # 根据数据维度调整结果格式
        if data.ndim == 1:
            result_kwargs = {"name": data.name}
            result = result.ravel()
        else:
            result_kwargs = {"columns": data.columns}
        
        # 根据结果重新构建Series或DataFrame对象
        res = data._constructor(result, index=index, **result_kwargs)
        
        # 如果不需要作为索引，则重新插入分组器
        if not self.as_index:
            res = self._insert_inaxis_grouper(res)
            res.index = default_index(len(res))
        
        return res

    # -----------------------------------------------------------------
    # apply/agg/transform

    @final
    def _python_apply_general(
        self,
        f: Callable,
        data: DataFrame | Series,
        not_indexed_same: bool | None = None,
        is_transform: bool = False,
        is_agg: bool = False,
    ) -> NDFrameT:
        """
        Apply function f in python space

        Parameters
        ----------
        f : callable
            Function to apply
        data : Series or DataFrame
            Data to apply f to
        not_indexed_same: bool, optional
            When specified, overrides the value of not_indexed_same. Apply behaves
            differently when the result index is equal to the input index, but
            this can be coincidental leading to value-dependent behavior.
        is_transform : bool, default False
            Indicator for whether the function is actually a transform
            and should not have group keys prepended.
        is_agg : bool, default False
            Indicator for whether the function is an aggregation. When the
            result is empty, we don't want to warn for this case.
            See _GroupBy._python_agg_general.

        Returns
        -------
        Series or DataFrame
            data after applying f
        """
        # 在Python空间中应用函数f

        # 使用分组器对象执行按组应用函数操作
        values, mutated = self._grouper.apply_groupwise(f, data)
        
        # 如果未指定not_indexed_same参数，则根据实际情况决定
        if not_indexed_same is None:
            not_indexed_same = mutated
        
        # 封装应用后的输出结果
        return self._wrap_applied_output(
            data,
            values,
            not_indexed_same,
            is_transform,
        )

    @final
    @final
    # 以纯Python方式聚合数据，作为在_cython_operation抛出NotImplementedError时的后备方案
    def _agg_general(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        *,
        alias: str,
        npfunc: Callable | None = None,
        **kwargs,
    ):
        # 调用_cython_agg_general方法执行聚合操作
        result = self._cython_agg_general(
            how=alias,
            alt=npfunc,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )
        # 返回结果并以当前对象作为finalizer，以确保正确的groupby方法被应用
        return result.__finalize__(self.obj, method="groupby")

    # 如果_cython_operation抛出NotImplementedError，则使用纯Python方式进行聚合
    def _agg_py_fallback(
        self, how: str, values: ArrayLike, ndim: int, alt: Callable
    ) -> ArrayLike:
        """
        如果_cython_operation抛出NotImplementedError，则使用纯Python方式进行聚合。
        """
        # 确保alt参数不为None
        assert alt is not None

        if values.ndim == 1:
            # 对于DataFrameGroupBy，仅使用ExtensionArray时会进入此分支
            ser = Series(values, copy=False)
        else:
            # 对于values.dtype == object的情况进入此分支
            df = DataFrame(values.T, dtype=values.dtype)
            # 由于在grouped_reduce中我们分割了object块，因此只有1列
            # 否则，我们需要关注块分割的问题，参见GH#39329
            assert df.shape[1] == 1
            # 避免在DataFrame减少中调用self.values，参见GH#28949
            ser = df.iloc[:, 0]

        # 我们不会使用用户定义函数（UDFs），因此我们知道我们的dtype应始终由实现的聚合保留
        # TODO: 这是否完全正确？请参见WrappedCythonOp get_result_dtype？
        try:
            # 调用_grouper.agg_series方法对Series进行聚合操作，保留dtype
            res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
        except Exception as err:
            # 如果聚合函数失败，则抛出特定类型的异常
            msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
            raise type(err)(msg) from err

        if ser.dtype == object:
            # 如果原始Series的dtype为object，则将结果转换为object类型
            res_values = res_values.astype(object, copy=False)

        # 如果是DataFrameGroupBy并且经过了SeriesGroupByPath，则需要重塑结果的形状
        # GH#32223 包括IntegerArray值，ndarray类型的res_values
        # 对于具有object dtype值的test_groupby_duplicate_columns
        return ensure_block_shape(res_values, ndim=ndim)

    @final
    # 使用_cython_operation执行通用聚合操作
    def _cython_agg_general(
        self,
        how: str,
        alt: Callable | None = None,
        numeric_only: bool = False,
        min_count: int = -1,
        **kwargs,
    ):
        # Note: we never get here with how="ohlc" for DataFrameGroupBy;
        #  that goes through SeriesGroupBy
        # 注释：对于 DataFrameGroupBy 使用 how="ohlc" 的情况，不会进入这里；
        #      这种情况会通过 SeriesGroupBy 处理

        data = self._get_data_to_aggregate(numeric_only=numeric_only, name=how)
        # 注释：获取用于聚合的数据，可以是数值型数据，根据 how 参数进行命名

        def array_func(values: ArrayLike) -> ArrayLike:
            # 注释：定义一个处理数组的函数 array_func，输入和输出都是类数组对象
            try:
                result = self._grouper._cython_operation(
                    "aggregate",
                    values,
                    how,
                    axis=data.ndim - 1,
                    min_count=min_count,
                    **kwargs,
                )
            except NotImplementedError:
                # 注释：捕获 NotImplementedError 异常，通常是在 numeric_only=False 和不适用的函数下发生
                #      TODO: 是否应该考虑 min_count？
                #      TODO: 避免在这里特别处理 SparseArray
                if how in ["any", "all"] and isinstance(values, SparseArray):
                    pass
                elif alt is None or how in ["any", "all", "std", "sem"]:
                    raise  # TODO: 应该重新抛出作为 TypeError？不应该到达这里
            else:
                return result

            assert alt is not None
            # 注释：确保 alt 不为空
            result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
            return result

        new_mgr = data.grouped_reduce(array_func)
        # 注释：使用 array_func 对数据进行分组聚合操作，生成新的管理器对象 new_mgr
        res = self._wrap_agged_manager(new_mgr)
        # 注释：将聚合后的管理器对象 res 进行包装处理
        if how in ["idxmin", "idxmax"]:
            res = self._wrap_idxmax_idxmin(res)
            # 注释：如果 how 是 "idxmin" 或 "idxmax"，则对结果 res 进行特殊处理
        out = self._wrap_aggregated_output(res)
        # 注释：对聚合输出结果 res 进行最终的包装处理
        return out
    def _transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        # 如果传入的 func 不是字符串，则调用通用的转换方法 _transform_general
        if not isinstance(func, str):
            return self._transform_general(func, engine, engine_kwargs, *args, **kwargs)

        # 如果 func 不在允许的转换核函数名单中，抛出 ValueError 异常
        elif func not in base.transform_kernel_allowlist:
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        
        # 如果 func 是已经使用 Cython 优化或者是预定义的转换核函数
        elif func in base.cythonized_kernels or func in base.transformation_kernels:
            # 如果提供了 engine 参数，则将其传递给 kwargs
            if engine is not None:
                kwargs["engine"] = engine
                kwargs["engine_kwargs"] = engine_kwargs
            # 调用实例对象中的 func 方法，并传递其余参数
            return getattr(self, func)(*args, **kwargs)

        else:
            # 如果 func 在降维核函数名单中，并且对象已经观察到数据
            if self.observed:
                # 使用 _reduction_kernel_transform 方法进行处理
                return self._reduction_kernel_transform(
                    func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
                )

            # 在临时更改对象的 observed 属性为 True 和 _grouper 属性为 observed_grouper 时执行
            with (
                com.temp_setattr(self, "observed", True),
                com.temp_setattr(self, "_grouper", self._grouper.observed_grouper),
            ):
                # 使用 _reduction_kernel_transform 方法进行处理
                return self._reduction_kernel_transform(
                    func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
                )

    @final
    def _reduction_kernel_transform(
        self, func, *args, engine=None, engine_kwargs=None, **kwargs
    ):
        # GH#30918 当我们知道 func 是聚合函数时，使用 _transform_fast
        # 如果 func 是一个减少函数，我们需要将结果广播到整个组中。
        # 计算 func 的结果，并处理可能的广播情况。
        with com.temp_setattr(self, "as_index", True):
            # GH#49834 - 对于 _wrap_transform_fast_result 需要在索引中包含分组
            if func in ["idxmin", "idxmax"]:
                func = cast(Literal["idxmin", "idxmax"], func)
                # 调用 _idxmax_idxmin 方法处理索引最大最小值
                result = self._idxmax_idxmin(func, True, *args, **kwargs)
            else:
                # 如果提供了 engine 参数，则将其传递给 kwargs
                if engine is not None:
                    kwargs["engine"] = engine
                    kwargs["engine_kwargs"] = engine_kwargs
                # 调用实例对象中的 func 方法，并传递其余参数
                result = getattr(self, func)(*args, **kwargs)

        # 对结果使用 _wrap_transform_fast_result 方法进行处理
        return self._wrap_transform_fast_result(result)

    @final
    # 将结果快速转换为聚合的路径
    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
        """
        Fast transform path for aggregations.
        """
        # 获取带有排除项的对象
        obj = self._obj_with_exclusions

        # 对于每一列，通过take操作重塑为原始框架的大小
        ids = self._grouper.ids
        result = result.reindex(self._grouper.result_index, axis=0)

        if self.obj.ndim == 1:
            # 即SeriesGroupBy
            out = algorithms.take_nd(result._values, ids)
            output = obj._constructor(out, index=obj.index, name=obj.name)
        else:
            # `.size()`在DataFrame输入上给出Series输出，需要轴0
            # GH#46209
            # 不转换索引：负索引需要在结果中产生空值
            new_ax = result.index.take(ids)
            output = result._reindex_with_indexers({0: (new_ax, ids)}, allow_dups=True)
            output = output.set_axis(obj.index, axis=0)
        return output

    # -----------------------------------------------------------------
    # 实用工具

    @final
    def _apply_filter(self, indices, dropna):
        # 如果索引长度为0，则创建空数组
        if len(indices) == 0:
            indices = np.array([], dtype="int64")
        else:
            # 对索引进行排序并连接
            indices = np.sort(np.concatenate(indices))
        if dropna:
            # 根据索引过滤选定的对象
            filtered = self._selected_obj.take(indices, axis=0)
        else:
            mask = np.empty(len(self._selected_obj.index), dtype=bool)
            mask.fill(False)
            mask[indices.astype(int)] = True
            # 当传递给where时，掩码无法广播；手动广播
            mask = np.tile(mask, list(self._selected_obj.shape[1:]) + [1]).T
            filtered = self._selected_obj.where(mask)  # 用NaN填充
        return filtered

    @final
    def _cumcount_array(self, ascending: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Notes
        -----
        this is currently implementing sort=False
        (though the default is sort=True) for groupby in general
        """

        # 获取分组后的标识符数组
        ids = self._grouper.ids
        # 获取分组的数量
        ngroups = self._grouper.ngroups
        # 根据分组标识符排序并返回排序后的索引
        sorter = get_group_index_sorter(ids, ngroups)
        # 根据排序后的索引重新排列ids和计数
        ids, count = ids[sorter], len(ids)

        # 如果没有数据，返回空的 int64 数组
        if count == 0:
            return np.empty(0, dtype=np.int64)

        # 标识连续相同值的位置
        run = np.r_[True, ids[:-1] != ids[1:]]
        # 计算每个连续值的重复次数
        rep = np.diff(np.r_[np.nonzero(run)[0], count])
        # 计算累积计数
        out = (~run).cumsum()

        # 如果指定为升序
        if ascending:
            # 减去重复值对应的累积计数
            out -= np.repeat(out[run], rep)
        else:
            # 计算降序情况下的累积计数
            out = np.repeat(out[np.r_[run[1:], True]], rep) - out

        # 如果存在缺失值，则将对应位置设置为 NaN
        if self._grouper.has_dropped_na:
            out = np.where(ids == -1, np.nan, out.astype(np.float64, copy=False))
        else:
            out = out.astype(np.int64, copy=False)

        # 创建一个逆排序的索引数组
        rev = np.empty(count, dtype=np.intp)
        rev[sorter] = np.arange(count, dtype=np.intp)
        # 根据逆排序的索引数组返回结果数组
        return out[rev]

    # -----------------------------------------------------------------

    @final
    @property
    def _obj_1d_constructor(self) -> Callable:
        # 保留子类化的 Series/DataFrame 对象构造器
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced
        # 确认对象是 Series 类型
        assert isinstance(self.obj, Series)
        # 返回 Series 对象的构造器
        return self.obj._constructor

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def any(self, skipna: bool = True) -> NDFrameT:
        """
        Return True if any value in the group is truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if any element
            is True within its respective group, False otherwise.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).any()
        a     True
        b    False
        dtype: bool

        For DataFrameGroupBy:

        >>> data = [[1, 0, 3], [1, 0, 6], [7, 1, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["ostrich", "penguin", "parrot"]
        ... )
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  0  6
        parrot   7  1  9
        >>> df.groupby(by=["a"]).any()
               b      c
        a
        1  False   True
        7   True   True
        """
        # 使用 _cython_agg_general 方法进行通用聚合操作，调用 "any" 聚合类型
        # alt 函数处理每个分组，将结果转换为 Series，并调用其 any 方法进行聚合
        # skipna 参数用于控制是否跳过 NaN 值
        return self._cython_agg_general(
            "any",
            alt=lambda x: Series(x, copy=False).any(skipna=skipna),
            skipna=skipna,
        )
    # 返回当前分组中的所有值是否全部为真的布尔结果
    def all(self, skipna: bool = True) -> NDFrameT:
        """
        Return True if all values in the group are truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if all elements
            are True within its respective group, False otherwise.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).all()
        a     True
        b    False
        dtype: bool

        For DataFrameGroupBy:

        >>> data = [[1, 0, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["ostrich", "penguin", "parrot"]
        ... )
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  5  6
        parrot   7  8  9
        >>> df.groupby(by=["a"]).all()
               b      c
        a
        1  False   True
        7   True   True
        """
        # 使用 Cython 实现的通用聚合方法来计算所有值是否为真
        return self._cython_agg_general(
            "all",
            # 使用替代方法 alt 来计算每个分组是否所有元素为真
            alt=lambda x: Series(x, copy=False).all(skipna=skipna),
            skipna=skipna,
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    # 定义一个方法用于计算分组的计数，不包括缺失值。

    # 获取需要聚合的数据
    data = self._get_data_to_aggregate()
    
    # 获取分组的标识符
    ids = self._grouper.ids
    
    # 获取分组的数量
    ngroups = self._grouper.ngroups
    
    # 创建一个布尔掩码，用于标识有效的分组
    mask = ids != -1
    
    # 判断数据是否为 Series
    is_series = data.ndim == 1

    # 定义一个内部函数 hfunc，用于处理数据并返回计数结果
    def hfunc(bvalues: ArrayLike) -> ArrayLike:
        # 如果数据是一维的，创建一个掩码，排除缺失值
        if bvalues.ndim == 1:
            masked = mask & ~isna(bvalues).reshape(1, -1)
        else:
            masked = mask & ~isna(bvalues)
        
        # 调用 C 库函数进行二维计数
        counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups)
        
        # 处理特定类型的返回值
        if isinstance(bvalues, BaseMaskedArray):
            return IntegerArray(
                counted[0], mask=np.zeros(counted.shape[1], dtype=np.bool_)
            )
        elif isinstance(bvalues, ArrowExtensionArray) and not isinstance(
            bvalues.dtype, StringDtype
        ):
            dtype = pandas_dtype("int64[pyarrow]")
            return type(bvalues)._from_sequence(counted[0], dtype=dtype)
        
        # 如果是 Series，则返回二维计数的第一行
        if is_series:
            assert counted.ndim == 2
            assert counted.shape[0] == 1
            return counted[0]
        
        # 对于其他情况，直接返回计数结果
        return counted

    # 对数据进行分组聚合操作，得到新的数据管理器
    new_mgr = data.grouped_reduce(hfunc)
    
    # 将新的数据管理器进行包装，得到新的对象
    new_obj = self._wrap_agged_manager(new_mgr)
    
    # 对聚合后的输出结果进行包装
    result = self._wrap_aggregated_output(new_obj)
    
    # 返回最终的结果
    return result
    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0
                numeric_only no longer accepts ``None`` and defaults to ``False``.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Mean of values within each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": [1, 1, 2, 1, 2], "B": [np.nan, 2, 3, 4, 5], "C": [1, 2, 1, 1, 2]},
        ...     columns=["A", "B", "C"],
        ... )

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby("A").mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> df.groupby(["A", "B"]).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> df.groupby("A")["B"].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """

        if maybe_use_numba(engine):
            # Importing specialized function for mean computation using Numba
            from pandas.core._numba.kernels import grouped_mean
            # Call Numba accelerated aggregation function
            return self._numba_agg_general(
                grouped_mean,
                executor.float_dtype_mapping,
                engine_kwargs,
                min_periods=0,
            )
        else:
            # Call Cython accelerated aggregation function
            result = self._cython_agg_general(
                "mean",
                alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
                numeric_only=numeric_only,
            )
            # Finalize the result with the original object, preserving metadata
            return result.__finalize__(self.obj, method="groupby")

    @final
    def median(self, numeric_only: bool = False) -> NDFrameT:
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to False.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).median()
        a    7.0
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(
        ...     data, index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
        ... )
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).median()
                 a    b
        dog    3.0  4.0
        mouse  7.0  3.0

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 3, 4, 5],
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
        >>> ser.resample("MS").median()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64
        """
        # 使用 _cython_agg_general 方法计算中位数，传递给 alt 参数一个 lambda 函数以调用 Series 的 median 方法
        result = self._cython_agg_general(
            "median",
            alt=lambda x: Series(x, copy=False).median(numeric_only=numeric_only),
            numeric_only=numeric_only,
        )
        # 调用 __finalize__ 方法，将结果与原始对象关联起来，并指定方法为 groupby
        return result.__finalize__(self.obj, method="groupby")

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def std(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
    ):
        """
        Calculate standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is ``N - ddof``,
            where ``N`` represents the number of elements.

        engine : {'cython', 'numba'} or None, default None
            Engine to use for computation. If None, defaults to 'cython' for
            Series/DataFrames and 'numba' for GroupBy objects.

        engine_kwargs : dict[str, bool] or None, default None
            Optional keyword arguments for the computation engine.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Standard deviation within each group.

        See Also
        --------
        groupby : Group by operation.
        """
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def var(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
    ):
        """
        Calculate variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is ``N - ddof``,
            where ``N`` represents the number of elements.

        engine : {'cython', 'numba'} or None, default None
            Engine to use for computation. If None, defaults to 'cython' for
            Series/DataFrames and 'numba' for GroupBy objects.

        engine_kwargs : dict[str, bool] or None, default None
            Optional keyword arguments for the computation engine.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Variance within each group.

        See Also
        --------
        groupby : Group by operation.
        """
        pass
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)


    # 设置方法为最终方法，不允许子类覆盖
    # 使用 Substitution 装饰器，替换参数 name 为 "groupby"
    # 使用 Substitution 装饰器，替换参数 see_also 为 _common_see_also 的值
    def size(self) -> DataFrame | Series:
        """
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a     1
        a     2
        b     3
        dtype: int64
        >>> ser.groupby(level=0).size()
        a    2
        b    1
        dtype: int64

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby("a").size()
        a
        1    2
        7    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3],
        ...     index=pd.DatetimeIndex(["2023-01-01", "2023-01-15", "2023-02-01"]),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        dtype: int64
        >>> ser.resample("MS").size()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        # 调用内部方法 _grouper.size() 计算分组大小
        result = self._grouper.size()

        # 初始化 dtype_backend 变量为 None，表示数据类型后端暂未确定
        dtype_backend: None | Literal["pyarrow", "numpy_nullable"] = None

        # 如果 self.obj 是 Series 类型
        if isinstance(self.obj, Series):
            # 如果 self.obj 的 array 是 ArrowExtensionArray 类型
            if isinstance(self.obj.array, ArrowExtensionArray):
                # 如果 self.obj 的 array 是 ArrowStringArrayNumpySemantics 类型
                if isinstance(self.obj.array, ArrowStringArrayNumpySemantics):
                    dtype_backend = None
                # 如果 self.obj 的 array 是 ArrowStringArray 类型
                elif isinstance(self.obj.array, ArrowStringArray):
                    dtype_backend = "numpy_nullable"
                else:
                    dtype_backend = "pyarrow"
            # 如果 self.obj 的 array 是 BaseMaskedArray 类型
            elif isinstance(self.obj.array, BaseMaskedArray):
                dtype_backend = "numpy_nullable"

        # TODO: 对于 DataFrame，如果列混合使用 arrow/numpy/masked 类型，需要进一步处理

        # GH28330 通过调用保留子类化的 Series/DataFrames
        if isinstance(self.obj, Series):
            result = self._obj_1d_constructor(result, name=self.obj.name)
        else:
            result = self._obj_1d_constructor(result)

        # 如果 dtype_backend 不为 None，则使用 convert_dtypes 转换结果
        if dtype_backend is not None:
            result = result.convert_dtypes(
                infer_objects=False,
                convert_string=False,
                convert_boolean=False,
                convert_floating=False,
                dtype_backend=dtype_backend,
            )

        # 如果 as_index 为 False，则重命名结果并重置索引
        if not self.as_index:
            # error: Incompatible types in assignment (expression has
            # type "DataFrame", variable has type "Series")
            result = result.rename("size").reset_index()  # type: ignore[assignment]

        # 返回最终结果
        return result

    @final
    @doc(
        _groupby_agg_method_engine_template,  # 使用指定的文档模板注释此函数
        fname="sum",  # 函数名为 "sum"
        no=False,  # 不禁用此函数
        mc=0,  # 最小计数参数为 0
        e=None,  # 引擎参数为 None
        ek=None,  # 引擎关键字参数为 None
        example=dedent(  # 提供的示例字符串文档
            """\
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).sum()
        a    3
        b    7
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").sum()
             b   c
        a
        1   10   7
        2   11  17"""
        ),
    )
    def sum(
        self,
        numeric_only: bool = False,  # 是否仅针对数字类型进行计算，默认为 False
        min_count: int = 0,  # 最小计数，用于计算时的参数，默认为 0
        engine: Literal["cython", "numba"] | None = None,  # 引擎选择，可以是 "cython", "numba" 或 None
        engine_kwargs: dict[str, bool] | None = None,  # 引擎关键字参数，类型为字典，包含布尔值，默认为 None
    ):
        if maybe_use_numba(engine):  # 如果可能使用 Numba 引擎
            from pandas.core._numba.kernels import grouped_sum  # 导入 Numba 的 grouped_sum 函数

            return self._numba_agg_general(  # 调用 Numba 版本的聚合函数
                grouped_sum,  # 使用 grouped_sum 函数
                executor.default_dtype_mapping,  # 默认的数据类型映射
                engine_kwargs,  # 引擎关键字参数
                min_periods=min_count,  # 最小计数参数
            )
        else:
            # 如果我们在分类上进行分组，希望未观察到的分类返回零，而不是默认的 NaN
            # 这是因为 _agg_general() 在重新索引时返回 NaN，详见 GH #31422
            with com.temp_setattr(self, "observed", True):  # 临时设置 self.observed 为 True
                result = self._agg_general(  # 调用通用的聚合函数
                    numeric_only=numeric_only,  # 是否仅对数字类型进行计算
                    min_count=min_count,  # 最小计数参数
                    alias="sum",  # 别名为 "sum"
                    npfunc=np.sum,  # 使用 numpy 的 sum 函数进行计算
                )

            return result  # 返回计算结果

    @final
    @doc(
        _groupby_agg_method_template,  # 使用指定的文档模板注释此函数
        fname="prod",  # 函数名为 "prod"
        no=False,  # 不禁用此函数
        mc=0,  # 最小计数参数为 0
        example=dedent(  # 提供的示例字符串文档
            """\
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).prod()
        a    2
        b   12
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").prod()
             b    c
        a
        1   16   10
        2   30   72"""
        ),
    )
    # 定义 prod 方法，用于对数据进行乘积运算
    def prod(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT:
        # 调用 _agg_general 方法进行通用聚合操作，计算乘积
        return self._agg_general(
            numeric_only=numeric_only, min_count=min_count, alias="prod", npfunc=np.prod
        )

    @final
    @doc(
        _groupby_agg_method_engine_template,
        fname="min",
        no=False,
        mc=-1,
        e=None,
        ek=None,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).min()
        a    1
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").min()
            b  c
        a
        1   2  2
        2   5  8"""
        ),
    )
    # 定义 min 方法，用于对数据进行最小值聚合
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 若选择使用 numba 引擎，则调用相应的 numba 实现函数进行聚合操作
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max

            return self._numba_agg_general(
                grouped_min_max,
                executor.identity_dtype_mapping,
                engine_kwargs,
                min_periods=min_count,
                is_max=False,
            )
        else:
            # 否则，调用通用聚合方法 _agg_general 进行最小值计算
            return self._agg_general(
                numeric_only=numeric_only,
                min_count=min_count,
                alias="min",
                npfunc=np.min,
            )

    @final
    @doc(
        _groupby_agg_method_engine_template,
        fname="max",
        no=False,
        mc=-1,
        e=None,
        ek=None,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).max()
        a    2
        b    4
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").max()
            b  c
        a
        1   8  5
        2   6  9"""
        ),
    )
    # 定义 max 方法，用于对数据进行最大值聚合
    # 定义一个 max 方法，用于计算最大值
    def max(
        self,
        numeric_only: bool = False,  # 控制是否仅对数字进行计算，默认为 False
        min_count: int = -1,  # 最小有效观测数量，默认为 -1 表示无最小限制
        engine: Literal["cython", "numba"] | None = None,  # 指定计算引擎，可以是 "cython" 或 "numba"，默认为 None
        engine_kwargs: dict[str, bool] | None = None,  # 引擎参数的字典，键为字符串，值为布尔类型，或者为 None
    ):
        # 如果可能使用 numba 引擎，则导入相应的模块
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max

            # 调用 numba 加速的聚合方法 grouped_min_max
            return self._numba_agg_general(
                grouped_min_max,  # 使用 numba 的聚合函数
                executor.identity_dtype_mapping,  # 数据类型映射
                engine_kwargs,  # 引擎参数
                min_periods=min_count,  # 最小观测数
                is_max=True,  # 指定计算最大值
            )
        else:
            # 使用普通的 Pandas 方法进行聚合计算
            return self._agg_general(
                numeric_only=numeric_only,  # 是否仅对数字进行计算
                min_count=min_count,  # 最小有效观测数量
                alias="max",  # 聚合别名为最大值
                npfunc=np.max,  # NumPy 提供的最大值计算函数
            )

    # final 修饰符确保该方法在子类中不能被重写
    @final
    def first(
        self, numeric_only: bool = False,  # 控制是否仅对数字进行计算，默认为 False
        min_count: int = -1,  # 最小有效观测数量，默认为 -1 表示无最小限制
        skipna: bool = True  # 是否跳过 NA/null 值，默认为 True
    ) -> NDFrameT:
        """
        Compute the first entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            First values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        core.groupby.DataFrameGroupBy.last : Compute the last non-null entry
            of each column.
        core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     dict(
        ...         A=[1, 1, 3],
        ...         B=[None, 5, 6],
        ...         C=[1, 2, 3],
        ...         D=["3/11/2000", "3/12/2000", "3/13/2000"],
        ...     )
        ... )
        >>> df["D"] = pd.to_datetime(df["D"])
        >>> df.groupby("A").first()
             B  C          D
        A
        1  5.0  1 2000-03-11
        3  6.0  3 2000-03-13
        >>> df.groupby("A").first(min_count=2)
            B    C          D
        A
        1 NaN  1.0 2000-03-11
        3 NaN  NaN        NaT
        >>> df.groupby("A").first(numeric_only=True)
             B  C
        A
        1  5.0  1
        3  6.0  3
        """

        def first_compat(obj: NDFrameT):
            """Compute the first non-null value across DataFrame or Series."""
            def first(x: Series):
                """Helper function for first item that isn't NA."""
                # Extract non-null elements from the Series
                arr = x.array[notna(x.array)]
                # Return NA value if all elements are null, otherwise return the first non-null value
                if not len(arr):
                    return x.array.dtype.na_value
                return arr[0]

            if isinstance(obj, DataFrame):
                # Apply the 'first' function to each column of the DataFrame
                return obj.apply(first)
            elif isinstance(obj, Series):
                # Apply the 'first' function directly to the Series
                return first(obj)
            else:  # pragma: no cover
                # Raise an error if the input type is neither DataFrame nor Series
                raise TypeError(type(obj))

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="first",
            npfunc=first_compat,
            skipna=skipna,
        )
    ) -> NDFrameT:
        """
        Compute the last entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            Last of values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        core.groupby.DataFrameGroupBy.first : Compute the first non-null entry
            of each column.
        core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[5, None, 6], C=[1, 2, 3]))
        >>> df.groupby("A").last()
             B  C
        A
        1  5.0  2
        3  6.0  3
        """

        def last_compat(obj: NDFrameT):
            """
            Compatibility function to compute the last valid entry of each column.

            Parameters
            ----------
            obj : NDFrameT
                The input object, which can be either a DataFrame or a Series.

            Returns
            -------
            Series or scalar
                Series if obj is a DataFrame, otherwise a scalar (last valid entry).

            Raises
            ------
            TypeError
                If obj is neither a DataFrame nor a Series.

            Notes
            -----
            This function handles DataFrame and Series inputs differently to compute
            the last non-null value in each column or the last non-null value in the
            Series itself.

            """
            def last(x: Series):
                """Helper function for last item that isn't NA."""
                arr = x.array[notna(x.array)]  # Filter out NA values from the array
                if not len(arr):
                    return x.array.dtype.na_value  # Return NA value if array is empty
                return arr[-1]  # Return the last valid value from the array

            if isinstance(obj, DataFrame):
                return obj.apply(last)  # Apply the 'last' function across each column
            elif isinstance(obj, Series):
                return last(obj)  # Compute the last valid value for the Series
            else:  # pragma: no cover
                raise TypeError(type(obj))  # Raise error if obj type is unexpected

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="last",
            npfunc=last_compat,
            skipna=skipna,
        )

    @final
    @doc(DataFrame.describe)
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ) -> NDFrameT:
        obj = self._obj_with_exclusions

        # 如果对象为空，则描述空对象，返回空 DataFrame
        if len(obj) == 0:
            described = obj.describe(
                percentiles=percentiles, include=include, exclude=exclude
            )
            # 如果对象是一维的，则直接返回描述结果
            if obj.ndim == 1:
                result = described
            else:
                # 否则，对描述结果进行堆叠操作
                result = described.unstack()
            # 将结果转换为 DataFrame 格式并返回，注意空对象的情况返回空的行
            return result.to_frame().T.iloc[:0]

        # 临时设置 self.as_index 为 True，应用描述函数于对象，并处理结果
        with com.temp_setattr(self, "as_index", True):
            result = self._python_apply_general(
                lambda x: x.describe(
                    percentiles=percentiles, include=include, exclude=exclude
                ),
                obj,
                not_indexed_same=True,
            )

        # 处理分组列的情况，确保正确处理
        result = result.unstack()
        # 如果 self.as_index 不为 True，则插入轴向分组器，并使用默认索引
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))

        return result

    @final
    @final
    def rolling(
        self,
        window: int | datetime.timedelta | str | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: bool = False,
        win_type: str | None = None,
        on: str | None = None,
        closed: IntervalClosedType | None = None,
        method: str = "single",
    ):
        """
        Return a rolling grouper, providing rolling functionality per group.
        """
        # 省略 rolling 方法的代码块

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def expanding(self, *args, **kwargs) -> ExpandingGroupby:
        """
        Return an expanding grouper, providing expanding
        functionality per group.

        Returns
        -------
        pandas.api.typing.ExpandingGroupby
        """
        # 导入 ExpandingGroupby 类并返回其实例化结果
        from pandas.core.window import ExpandingGroupby

        return ExpandingGroupby(
            self._selected_obj,
            *args,
            _grouper=self._grouper,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ewm(self, *args, **kwargs) -> ExponentialMovingWindowGroupby:
        """
        Return an ewm grouper, providing ewm functionality per group.

        Returns
        -------
        pandas.api.typing.ExponentialMovingWindowGroupby
        """
        # 导入 ExponentialMovingWindowGroupby 类并返回其实例化结果
        from pandas.core.window import ExponentialMovingWindowGroupby

        return ExponentialMovingWindowGroupby(
            self._selected_obj,
            *args,
            _grouper=self._grouper,
            **kwargs,
        )

    @final
    def _fill(self, direction: Literal["ffill", "bfill"], limit: int | None = None):
        """
        Shared function for `pad` and `backfill` to call Cython method.

        Parameters
        ----------
        direction : {'ffill', 'bfill'}
            Direction passed to underlying Cython function. `bfill` will cause
            values to be filled backwards. `ffill` and any other values will
            default to a forward fill
        limit : int, default None
            Maximum number of consecutive values to fill. If `None`, this
            method will convert to -1 prior to passing to Cython

        Returns
        -------
        `Series` or `DataFrame` with filled values

        See Also
        --------
        pad : Returns Series with minimum number of char in object.
        backfill : Backward fill the missing values in the dataset.
        """
        # 将 limit 转换为整数，用于传递给 Cython 方法
        if limit is None:
            limit = -1

        # 获取分组的 ids 和分组的数量
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups

        # 部分函数定义，使用 Cython 方法进行填充 NaN 值的索引
        col_func = partial(
            libgroupby.group_fillna_indexer,
            labels=ids,
            limit=limit,
            compute_ffill=(direction == "ffill"),
            ngroups=ngroups,
        )

        # 块函数定义，用于处理数据块的填充操作
        def blk_func(values: ArrayLike) -> ArrayLike:
            mask = isna(values)
            if values.ndim == 1:
                # 对于一维数据，创建一个与原始数据形状相同的索引器
                indexer = np.empty(values.shape, dtype=np.intp)
                col_func(out=indexer, mask=mask)
                return algorithms.take_nd(values, indexer)

            else:
                # 对于多维数据，创建一个与原始数据形状相同的数组用于填充
                # 并调用 group_fillna_indexer 进行列-wise 的填充操作
                if isinstance(values, np.ndarray):
                    dtype = values.dtype
                    if self._grouper.has_dropped_na:
                        # 如果有删除 NaN 的分组，则结果中会有 NaN
                        dtype = ensure_dtype_can_hold_na(values.dtype)
                    out = np.empty(values.shape, dtype=dtype)
                else:
                    # 对于非 ndarray 的情况，直接使用原始数据类型创建空数组
                    out = type(values)._empty(values.shape, dtype=values.dtype)

                for i, value_element in enumerate(values):
                    # 对每个数据元素调用 group_fillna_indexer 进行填充索引操作
                    indexer = np.empty(values.shape[1], dtype=np.intp)
                    col_func(out=indexer, mask=mask[i])
                    out[i, :] = algorithms.take_nd(value_element, indexer)
                return out

        # 获取要聚合的数据
        mgr = self._get_data_to_aggregate()
        # 应用块函数处理数据
        res_mgr = mgr.apply(blk_func)

        # 封装处理后的数据到新的对象中
        new_obj = self._wrap_agged_manager(res_mgr)
        new_obj.index = self.obj.index
        return new_obj

    @final
    @Substitution(name="groupby")
    def ffill(self, limit: int | None = None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.ffill: Returns Series with minimum number of char in object.
        DataFrame.ffill: Object with missing values filled or None if inplace=True.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.

        Examples
        --------

        For SeriesGroupBy:

        >>> key = [0, 0, 1, 1]
        >>> ser = pd.Series([np.nan, 2, 3, np.nan], index=key)
        >>> ser
        0    NaN
        0    2.0
        1    3.0
        1    NaN
        dtype: float64
        >>> ser.groupby(level=0).ffill()
        0    NaN
        0    2.0
        1    3.0
        1    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> df = pd.DataFrame(
        ...     {
        ...         "key": [0, 0, 1, 1, 1],
        ...         "A": [np.nan, 2, np.nan, 3, np.nan],
        ...         "B": [2, 3, np.nan, np.nan, np.nan],
        ...         "C": [np.nan, np.nan, 2, np.nan, np.nan],
        ...     }
        ... )
        >>> df
           key    A    B   C
        0    0  NaN  2.0 NaN
        1    0  2.0  3.0 NaN
        2    1  NaN  NaN 2.0
        3    1  3.0  NaN NaN
        4    1  NaN  NaN NaN

        Propagate non-null values forward or backward within each group along columns.

        >>> df.groupby("key").ffill()
             A    B   C
        0  NaN  2.0 NaN
        1  2.0  3.0 NaN
        2  NaN  NaN 2.0
        3  3.0  NaN 2.0
        4  3.0  NaN NaN

        Propagate non-null values forward or backward within each group along rows.

        >>> df.T.groupby(np.array([0, 0, 1, 1])).ffill().T
           key    A    B    C
        0  0.0  0.0  2.0  2.0
        1  0.0  2.0  3.0  3.0
        2  1.0  1.0  NaN  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  1.0  NaN  NaN

        Only replace the first NaN element within a group along rows.

        >>> df.groupby("key").ffill(limit=1)
             A    B    C
        0  NaN  2.0  NaN
        1  2.0  3.0  NaN
        2  NaN  NaN  2.0
        3  3.0  NaN  2.0
        4  3.0  NaN  NaN
        """
        # 调用内部方法 `_fill`，执行前向填充操作，并返回结果
        return self._fill("ffill", limit=limit)

    @final
    @Substitution(name="groupby")
    def bfill(self, limit: int | None = None):
        """
        Backward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.bfill : Backward fill the missing values in the dataset.
        DataFrame.bfill: Backward fill the missing values in the dataset.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.

        Examples
        --------

        With Series:

        >>> index = ["Falcon", "Falcon", "Parrot", "Parrot", "Parrot"]
        >>> s = pd.Series([None, 1, None, None, 3], index=index)
        >>> s
        Falcon    NaN
        Falcon    1.0
        Parrot    NaN
        Parrot    NaN
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill()
        Falcon    1.0
        Falcon    1.0
        Parrot    3.0
        Parrot    3.0
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill(limit=1)
        Falcon    1.0
        Falcon    1.0
        Parrot    NaN
        Parrot    3.0
        Parrot    3.0
        dtype: float64

        With DataFrame:

        >>> df = pd.DataFrame(
        ...     {"A": [1, None, None, None, 4], "B": [None, None, 5, None, 7]},
        ...     index=index,
        ... )
        >>> df
                  A        B
        Falcon    1.0      NaN
        Falcon    NaN      NaN
        Parrot    NaN      5.0
        Parrot    NaN      NaN
        Parrot    4.0      7.0
        >>> df.groupby(level=0).bfill()
                  A        B
        Falcon    1.0      NaN
        Falcon    NaN      NaN
        Parrot    4.0      5.0
        Parrot    4.0      7.0
        Parrot    4.0      7.0
        >>> df.groupby(level=0).bfill(limit=1)
                  A        B
        Falcon    1.0      NaN
        Falcon    NaN      NaN
        Parrot    NaN      5.0
        Parrot    4.0      7.0
        Parrot    4.0      7.0
        """
        # 调用内部方法 `_fill`，使用 'bfill' 方式填充缺失值，传入可选的填充限制 `limit`
        return self._fill("bfill", limit=limit)

    @final
    @property
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def nth(self) -> GroupByNthSelector:
        """
        从每个分组中获取第n行（如果n是整数），否则获取行的子集。

        可以是函数调用或索引操作。使用索引操作时不支持dropna。
        索引操作接受一个逗号分隔的整数和切片列表。

        如果指定dropna，则获取第n个非空行。dropna可以是'all'或'any'，等同于在groupby之前调用dropna(how=dropna)。

        Parameters
        ----------
        n : int, slice or list of ints and slices
            行的单个第n值或第n值列表或切片。

            .. versionchanged:: 1.4.0
                添加了切片和包含切片的列表。
                添加了索引操作。

        dropna : {'any', 'all', None}, default None
            计算第n行之前，应用指定的dropna操作。仅当n为整数时支持。

        Returns
        -------
        Series or DataFrame
            每个分组内的第n个值。
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame(
        ...     {"A": [1, 1, 2, 1, 2], "B": [np.nan, 2, 3, 4, 5]}, columns=["A", "B"]
        ... )
        >>> g = df.groupby("A")
        >>> g.nth(0)
           A   B
        0  1 NaN
        2  2 3.0
        >>> g.nth(1)
           A   B
        1  1 2.0
        4  2 5.0
        >>> g.nth(-1)
           A   B
        3  1 4.0
        4  2 5.0
        >>> g.nth([0, 1])
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0
        4  2 5.0
        >>> g.nth(slice(None, -1))
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0

        可以使用索引操作

        >>> g.nth[0, 1]
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0
        4  2 5.0
        >>> g.nth[:-1]
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0

        指定`dropna`参数允许忽略``NaN``值

        >>> g.nth(0, dropna="any")
           A   B
        1  1 2.0
        2  2 3.0

        当指定的``n``大于任何分组的长度时，将返回一个空的DataFrame

        >>> g.nth(3, dropna="any")
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
        return GroupByNthSelector(self)

    def _nth(
        self,
        n: PositionalIndexer | tuple,
        dropna: Literal["any", "all", None] = None,
        ) -> NDFrameT:
        # 如果 dropna 参数为 False，则生成一个从位置索引器中创建的掩码
        if not dropna:
            mask = self._make_mask_from_positional_indexer(n)

            # 获取分组的 IDs
            ids = self._grouper.ids

            # 在分组中去除 NA 值
            mask = mask & (ids != -1)

            # 使用掩码选择对象中的数据
            out = self._mask_selected_obj(mask)
            return out

        # 如果 dropna 参数为真
        if not is_integer(n):
            # 如果 n 不是整数，则抛出值错误异常
            raise ValueError("dropna option only supported for an integer argument")

        if dropna not in ["any", "all"]:
            # 如果 dropna 不在 ["any", "all"] 中，则抛出值错误异常
            # 注意：当进行聚合操作时，picker 不会引发此异常，只会返回 NaN
            raise ValueError(
                "For a DataFrame or Series groupby.nth, dropna must be "
                "either None, 'any' or 'all', "
                f"(was passed {dropna})."
            )

        # 旧行为，但支持 DataFrames 的 all 和 any
        # 在 GH 7559 中进行了修改以提升性能
        n = cast(int, n)
        # 删除包含 NA 值的对象的行或列
        dropped = self._selected_obj.dropna(how=dropna, axis=0)

        # 获取新的 grouper 对象用于 dropped 对象
        grouper: np.ndarray | Index | ops.BaseGrouper
        if len(dropped) == len(self._selected_obj):
            # 没有删除任何行，可以使用相同的 grouper
            grouper = self._grouper
        else:
            # 我们没有可用的 grouper 信息
            # （例如，我们已经选择了一个当前对象中不存在的列）
            axis = self._grouper.axis
            # 从 grouper 中选择符合 dropped.index 的 codes_info
            grouper = self._grouper.codes_info[axis.isin(dropped.index)]
            if self._grouper.has_dropped_na:
                # 当传递给 groupby 时，需要将空组仍然编码为 -1
                nulls = grouper == -1
                # 错误：没有匹配 "where" 的重载变体
                # values = np.where(nulls, NA, grouper)  # type: ignore[call-overload]
                values = np.where(nulls, NA, grouper.astype("Int64"))  # type: ignore[call-overload]
                grouper = Index(values, dtype="Int64")

        # 对 dropped 对象进行分组操作
        grb = dropped.groupby(grouper, as_index=self.as_index, sort=self.sort)
        # 返回分组后的第 n 个元素
        return grb.nth(n)

    @final
    def quantile(
        self,
        q: float | AnyArrayLike = 0.5,
        interpolation: str = "linear",
        numeric_only: bool = False,
    @final
    @Substitution(name="groupby")
    def ngroup(self, ascending: bool = True):
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Groups with missing keys (where `pd.isna()` is True) will be labeled with `NaN`
        and will be skipped from the count.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = pd.DataFrame({"color": ["red", None, "red", "blue", "blue", "red"]})
        >>> df
           color
        0    red
        1   None
        2    red
        3   blue
        4   blue
        5    red
        >>> df.groupby("color").ngroup()
        0    1.0
        1    NaN
        2    1.0
        3    0.0
        4    0.0
        5    1.0
        dtype: float64
        >>> df.groupby("color", dropna=False).ngroup()
        0    1
        1    2
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby("color", dropna=False).ngroup(ascending=False)
        0    1
        1    0
        2    1
        3    2
        4    2
        5    1
        dtype: int64
        """
        # 获取包含数据和索引的对象
        obj = self._obj_with_exclusions
        # 获取索引
        index = obj.index
        # 获取分组器的 IDs
        comp_ids = self._grouper.ids

        # 初始化 dtype
        dtype: type
        # 如果分组器包含缺失值，则将 comp_ids 中的 -1 替换为 NaN，dtype 设置为 float64
        if self._grouper.has_dropped_na:
            comp_ids = np.where(comp_ids == -1, np.nan, comp_ids)
            dtype = np.float64
        else:
            # 否则，dtype 设置为 int64
            dtype = np.int64

        # 如果有任何分组器传递了分类数据
        if any(ping._passed_categorical for ping in self._grouper.groupings):
            # 重新计算 comp_ids，排除未观察到的分组
            comp_ids = rank_1d(comp_ids, ties_method="dense") - 1

        # 构建结果 Series，使用 comp_ids、index 和指定的 dtype
        result = self._obj_1d_constructor(comp_ids, index, dtype=dtype)
        # 如果不是升序，将结果反转，从 ngroups - 1 到 0
        if not ascending:
            result = self.ngroups - 1 - result
        return result
    def cumcount(self, ascending: bool = True):
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to

        .. code-block:: python

            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Returns
        -------
        Series
            Sequence number of each element within each group.

        See Also
        --------
        .ngroup : Number the groups themselves.

        Examples
        --------
        >>> df = pd.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], columns=["A"])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby("A").cumcount()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby("A").cumcount(ascending=False)
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
        # 获取索引信息
        index = self._obj_with_exclusions.index
        # 计算每个元素在各组内的序号数组
        cumcounts = self._cumcount_array(ascending=ascending)
        # 将计算结果构造成 Series 对象，并使用索引信息
        return self._obj_1d_constructor(cumcounts, index)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cumprod(self, *args, **kwargs) -> NDFrameT:
        """
        Cumulative product for each group.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to `func`.
        **kwargs : dict
            Additional/specific keyword arguments to be passed to the function,
            such as `numeric_only` and `skipna`.

        Returns
        -------
        Series or DataFrame
            Cumulative product for each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumprod()
        a    6
        a   12
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["cow", "horse", "bull"]
        ... )
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   2   5
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: ['cow', 'horse'], 2: ['bull']}
        >>> df.groupby("a").cumprod()
                b   c
        cow     8   2
        horse  16  10
        bull    6   9
        """
        # 验证传入的函数和参数，确保有效性
        nv.validate_groupby_func("cumprod", args, kwargs, ["numeric_only", "skipna"])
        # 调用底层的 Cython 函数执行累积乘积操作，并传递额外的关键字参数
        return self._cython_transform("cumprod", **kwargs)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cumsum(self, *args, **kwargs) -> NDFrameT:
        """
        Cumulative sum for each group.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to `func`.
        **kwargs : dict
            Additional/specific keyword arguments to be passed to the function,
            such as `numeric_only` and `skipna`.

        Returns
        -------
        Series or DataFrame
            Cumulative sum for each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumsum()
        a    6
        a    8
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["fox", "gorilla", "lion"]
        ... )
        >>> df
                  a   b   c
        fox       1   8   2
        gorilla   1   2   5
        lion      2   6   9
        >>> df.groupby("a").groups
        {1: ['fox', 'gorilla'], 2: ['lion']}
        >>> df.groupby("a").cumsum()
                  b   c
        fox       8   2
        gorilla  10   7
        lion      6   9
        """

        # 使用 nv 模块验证 groupby 函数的参数，并确保参数有效性
        nv.validate_groupby_func("cumsum", args, kwargs, ["numeric_only", "skipna"])
        # 调用底层方法 _cython_transform，执行累积求和操作，传递 kwargs 给底层函数
        return self._cython_transform("cumsum", **kwargs)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cummin(
        self,
        numeric_only: bool = False,
        **kwargs,
    ) -> NDFrameT:
        """
        Cumulative min for each group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the function, such as `skipna`,
            to control whether NA/null values are ignored.

        Returns
        -------
        Series or DataFrame
            Cumulative min for each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([1, 6, 2, 3, 0, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    0
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummin()
        a    1
        a    1
        a    1
        b    3
        b    0
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 0, 2], [1, 1, 5], [6, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["snake", "rabbit", "turtle"]
        ... )
        >>> df
                a   b   c
        snake   1   0   2
        rabbit  1   1   5
        turtle  6   6   9
        >>> df.groupby("a").groups
        {1: ['snake', 'rabbit'], 6: ['turtle']}
        >>> df.groupby("a").cummin()
                b   c
        snake   0   2
        rabbit  0   2
        turtle  6   9
        """
        # 获取是否忽略 NA/null 值的标志，默认为 True
        skipna = kwargs.get("skipna", True)
        # 调用底层的 Cython 函数进行累积最小值计算，并返回结果
        return self._cython_transform(
            "cummin", numeric_only=numeric_only, skipna=skipna
        )
    ) -> NDFrameT:
        """
        Cumulative max for each group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the function, such as `skipna`,
            to control whether NA/null values are ignored.

        Returns
        -------
        Series or DataFrame
            Cumulative max for each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([1, 6, 2, 3, 1, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    1
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummax()
        a    1
        a    6
        a    6
        b    3
        b    3
        b    4
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 1, 0], [2, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["cow", "horse", "bull"]
        ... )
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   1   0
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: ['cow', 'horse'], 2: ['bull']}
        >>> df.groupby("a").cummax()
                b   c
        cow     8   2
        horse   8   2
        bull    6   9
        """
        # 获取是否忽略 NA/null 值的参数，默认为 True
        skipna = kwargs.get("skipna", True)
        # 调用内部方法 _cython_transform，执行累计最大值操作
        return self._cython_transform(
            "cummax", numeric_only=numeric_only, skipna=skipna
        )

    @final
    @Substitution(name="groupby")
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq=None,
        fill_value=lib.no_default,
        suffix: str | None = None,
    ):
        """
        Shift index by desired number of periods.

        Parameters
        ----------
        periods : int or sequence of int, default 1
            Number of periods to shift. Can be positive or negative.
        freq : DateOffset, timedelta, or offset alias string, optional
            Offset to use from the tseries module or time rule (e.g., 'EOM').
        fill_value : scalar, optional
            The scalar value to use for newly introduced missing values.
        suffix : str, optional
            Suffix to add to column names.

        Returns
        -------
        shifted : same type as caller
            Object with index shifted by desired number of periods.
        """
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def diff(
        self,
        periods: int = 1,
        ):
        """
        First discrete difference of element.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.

        Returns
        -------
        same type as caller
            First discrete difference(s) of element.
        """
        pass
    ) -> NDFrameT:
        """
        计算元素的第一离散差分。

        计算每个元素与组中另一个元素（默认为前一行中的元素）的差异。

        Parameters
        ----------
        periods : int, default 1
            计算差异时的移动周期数，可以接受负值。

        Returns
        -------
        Series or DataFrame
            第一差分结果。
        %(see_also)s
        Examples
        --------
        对于 SeriesGroupBy:

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).diff()
        a    NaN
        a   -5.0
        a    6.0
        b    NaN
        b   -1.0
        b    0.0
        dtype: float64

        对于 DataFrameGroupBy:

        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(
        ...     data, index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
        ... )
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).diff()
                 a    b
          dog  NaN  NaN
          dog  2.0  3.0
          dog  2.0  4.0
        mouse  NaN  NaN
        mouse  0.0  0.0
        mouse  1.0 -2.0
        mouse -5.0 -1.0
        """
        obj = self._obj_with_exclusions
        # 将对象进行向前移动以计算差分
        shifted = self.shift(periods=periods)

        # GH45562 - 保持现有行为并匹配 Series.diff() 的行为，
        # int8 和 int16 被强制转换为 float32 而不是 float64。
        dtypes_to_f32 = ["int8", "int16"]
        if obj.ndim == 1:
            # 如果对象是一维的，且数据类型是 int8 或 int16，则转换为 float32
            if obj.dtype in dtypes_to_f32:
                shifted = shifted.astype("float32")
        else:
            # 对于多维对象，找出所有 int8 或 int16 的列，并转换为 float32
            to_coerce = [c for c, dtype in obj.dtypes.items() if dtype in dtypes_to_f32]
            if len(to_coerce):
                shifted = shifted.astype({c: "float32" for c in to_coerce})

        # 返回差分结果
        return obj - shifted
        """
        Calculate pct_change of each value to previous entry in group.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating percentage change. Comparing with
            a period of 1 means adjacent elements are compared, whereas a period
            of 2 compares every other element.

        fill_method : None
            Must be None. This argument will be removed in a future version of pandas.

            .. deprecated:: 2.1
                All options of `fill_method` are deprecated except `fill_method=None`.

        freq : str, pandas offset object, or None, default None
            The frequency increment for time series data (e.g., 'M' for month-end).
            If None, the frequency is inferred from the index. Relevant for time
            series data only.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).pct_change()
        a         NaN
        a    1.000000
        b         NaN
        b    0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data,
        ...     columns=["a", "b", "c"],
        ...     index=["tuna", "salmon", "catfish", "goldfish"],
        ... )
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").pct_change()
                    b  c
            tuna    NaN    NaN
          salmon    1.5  1.000
         catfish    NaN    NaN
        goldfish    0.2  0.125
        """
        # GH#53491
        # 如果 fill_method 不为 None，则抛出异常，要求其必须为 None
        if fill_method is not None:
            raise ValueError(f"fill_method must be None; got {fill_method=}.")

        # TODO(GH#23918): Remove this conditional for SeriesGroupBy when
        #  GH#23918 is fixed
        # 如果 freq 不为 None，则使用 lambda 函数应用 pct_change，并返回结果
        if freq is not None:
            f = lambda x: x.pct_change(
                periods=periods,
                freq=freq,
                axis=0,
            )
            return self._python_apply_general(f, self._selected_obj, is_transform=True)

        # 根据 fill_method 的值确定操作类型，如果为 None，则使用 'ffill'
        if fill_method is None:  # GH30463
            op = "ffill"
        else:
            op = fill_method
        # 使用 getattr 获取填充后的数据，并根据 groupby 进行分组
        filled = getattr(self, op)(limit=0)
        fill_grp = filled.groupby(self._grouper.codes, group_keys=self.group_keys)
        # 根据 periods 和 freq 进行数据的 shift
        shifted = fill_grp.shift(periods=periods, freq=freq)
        # 返回填充后的数据与 shift 后数据的比例变化百分比
        return (filled / shifted) - 1
    def head(self, n: int = 5) -> NDFrameT:
        """
        Return first n rows of each group.

        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Parameters
        ----------
        n : int
            If positive: number of entries to include from start of each group.
            If negative: number of entries to exclude from end of each group.

        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
        >>> df.groupby("A").head(1)
           A  B
        0  1  2
        2  5  6
        >>> df.groupby("A").head(-1)
           A  B
        0  1  2
        """
        # Create a mask to select the first n rows (positive n) or exclude the last n rows (negative n)
        mask = self._make_mask_from_positional_indexer(slice(None, n))
        # Apply the mask to the grouped object and return the subset
        return self._mask_selected_obj(mask)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def tail(self, n: int = 5) -> NDFrameT:
        """
        Return last n rows of each group.

        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Parameters
        ----------
        n : int
            If positive: number of entries to include from end of each group.
            If negative: number of entries to exclude from start of each group.

        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame(
        ...     [["a", 1], ["a", 2], ["b", 1], ["b", 2]], columns=["A", "B"]
        ... )
        >>> df.groupby("A").tail(1)
           A  B
        1  a  2
        3  b  2
        >>> df.groupby("A").tail(-1)
           A  B
        1  a  2
        3  b  2
        """
        # Create a mask to select the last n rows (positive n) or exclude the first n rows (negative n)
        if n:
            mask = self._make_mask_from_positional_indexer(slice(-n, None))
        else:
            mask = self._make_mask_from_positional_indexer([])
        # Apply the mask to the grouped object and return the subset
        return self._mask_selected_obj(mask)

    @final
    def _mask_selected_obj(self, mask: npt.NDArray[np.bool_]) -> NDFrameT:
        """
        Return _selected_obj with mask applied.

        Parameters
        ----------
        mask : np.ndarray[bool]
            Boolean mask to apply.

        Returns
        -------
        Series or DataFrame
            Filtered _selected_obj.
        """
        # Filter the _selected_obj (presumably a Series or DataFrame) using the provided boolean mask
        ids = self._grouper.ids
        mask = mask & (ids != -1)
        return self._selected_obj[mask]

    @final
    def _idxmax_idxmin(
        self,
        how: Literal["idxmax", "idxmin"],
        ignore_unobserved: bool = False,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> NDFrameT:
        """Compute idxmax/idxmin.

        Parameters
        ----------
        how : {'idxmin', 'idxmax'}
            Whether to compute idxmin or idxmax.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ignore_unobserved : bool, default False
            When True and an unobserved group is encountered, do not raise. This is used
            for transform where unobserved groups do not affect the result.

        Returns
        -------
        Series or DataFrame
            idxmax or idxmin for the groupby operation.
        """
        # Check if the object has observed groups and any categorical variables are passed
        if not self.observed and any(
            ping._passed_categorical for ping in self._grouper.groupings
        ):
            # Determine the expected and actual lengths of groups
            expected_len = len(self._grouper.result_index)
            group_sizes = self._grouper.size()
            result_len = group_sizes[group_sizes > 0].shape[0]
            
            # Ensure actual result length does not exceed expected length
            assert result_len <= expected_len

            # Check if there are unobserved groups
            has_unobserved = result_len < expected_len

            # Determine if an error should be raised based on unobserved groups
            raise_err: bool | np.bool_ = not ignore_unobserved and has_unobserved

            # Prepare data for computation
            data = self._obj_with_exclusions

            # If raising error and data is a DataFrame, check for numeric columns
            if raise_err and isinstance(data, DataFrame):
                if numeric_only:
                    data = data._get_numeric_data()
                raise_err = len(data.columns) > 0

            # Raise a ValueError if an error condition is met
            if raise_err:
                raise ValueError(
                    f"Can't get {how} of an empty group due to unobserved categories. "
                    "Specify observed=True in groupby instead."
                )
        
        # If skipna is False, check for NA values in the data
        elif not skipna and self._obj_with_exclusions.isna().any(axis=None):
            raise ValueError(
                f"{type(self).__name__}.{how} with skipna=False encountered an NA "
                f"value."
            )

        # Perform the aggregation operation based on the specified 'how' parameter
        result = self._agg_general(
            numeric_only=numeric_only,
            min_count=1,
            alias=how,
            skipna=skipna,
        )
        # Return the computed result
        return result
    # 定义一个方法来处理索引的最大和最小值位置
    def _wrap_idxmax_idxmin(self, res: NDFrameT) -> NDFrameT:
        # 获取数据框的索引
        index = self.obj.index
        # 如果结果为空，则创建一个类型与索引相同的结果
        if res.size == 0:
            result = res.astype(index.dtype)
        else:
            # 如果索引是多重索引，则转换为扁平索引
            if isinstance(index, MultiIndex):
                index = index.to_flat_index()
            # 获取结果的值数组
            values = res._values
            # 确保值是 NumPy 数组
            assert isinstance(values, np.ndarray)
            # 根据索引的数据类型获取缺失值
            na_value = na_value_for_dtype(index.dtype, compat=False)
            # 如果结果是 Series 类型
            if isinstance(res, Series):
                # 创建一个新的 Series 对象
                # mypy: 表达式的类型为 "Series"，变量的类型为 "NDFrameT"
                result = res._constructor(  # type: ignore[assignment]
                    # 从索引数组中获取指定位置的值，允许填充缺失值
                    index.array.take(values, allow_fill=True, fill_value=na_value),
                    index=res.index,  # 设置新 Series 的索引
                    name=res.name,    # 设置新 Series 的名称
                )
            else:
                # 如果结果不是 Series 类型，创建一个空的数据字典
                data = {}
                # 遍历结果的每一列
                for k, column_values in enumerate(values.T):
                    # 将每一列的值根据索引数组取出，允许填充缺失值
                    data[k] = index.array.take(
                        column_values, allow_fill=True, fill_value=na_value
                    )
                # 使用原始数据框的构造函数创建新的结果数据框
                result = self.obj._constructor(data, index=res.index)
                result.columns = res.columns  # 设置新数据框的列名
        # 返回处理后的结果
        return result
@doc(GroupBy)
# 定义装饰器函数，用于为函数提供文档，其目标是 GroupBy 对象
def get_groupby(
    obj: NDFrame,
    by: _KeysArgType | None = None,
    grouper: ops.BaseGrouper | None = None,
    group_keys: bool = True,
) -> GroupBy:
    klass: type[GroupBy]
    # 检查 obj 是否为 Series 类型
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy

        klass = SeriesGroupBy
    # 检查 obj 是否为 DataFrame 类型
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy

        klass = DataFrameGroupBy
    else:  # pragma: no cover
        # 如果 obj 类型不匹配，则抛出 TypeError 异常
        raise TypeError(f"invalid type: {obj}")

    # 返回根据 obj 类型选择的 GroupBy 类的实例
    return klass(
        obj=obj,
        keys=by,
        grouper=grouper,
        group_keys=group_keys,
    )


def _insert_quantile_level(idx: Index, qs: npt.NDArray[np.float64]) -> MultiIndex:
    """
    Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

    The quantile level in the MultiIndex is a repeated copy of 'qs'.

    Parameters
    ----------
    idx : Index
        索引对象，用于构建 MultiIndex
    qs : np.ndarray[float64]
        包含分位数的 numpy 数组

    Returns
    -------
    MultiIndex
        包含了分位数作为最内层级别的 MultiIndex 对象
    """
    nqs = len(qs)
    # 将分位数数组转换为索引，并获取其唯一标签和标签码
    lev_codes, lev = Index(qs).factorize()
    lev_codes = coerce_indexer_dtype(lev_codes, lev)

    # 如果 idx 是 MultiIndex 对象
    if idx._is_multi:
        idx = cast(MultiIndex, idx)
        # 将分位数级别添加到已存在的 MultiIndex 的级别列表中
        levels = list(idx.levels) + [lev]
        # 构建新的码列表，包含旧的码和重复的分位数码
        codes = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(lev_codes, len(idx))]
        # 创建新的 MultiIndex 对象
        mi = MultiIndex(levels=levels, codes=codes, names=idx.names + [None])
    else:
        nidx = len(idx)
        idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
        # 创建包含 idx 和分位数的新级别列表
        levels = [idx, lev]
        # 构建码列表，包含重复的索引码和分位数码
        codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
        # 创建新的 MultiIndex 对象，指定每个级别的名称
        mi = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])

    # 返回构建好的 MultiIndex 对象
    return mi


# GH#7155
# 用于显示警告信息的字符串模板，指出了操作已弃用的行为
_apply_groupings_depr = (
    "{}.{} operated on the grouping columns. This behavior is deprecated, "
    "and in a future version of pandas the grouping columns will be excluded "
    "from the operation. Either pass `include_groups=False` to exclude the "
    "groupings or explicitly select the grouping columns after groupby to silence "
    "this warning."
)
```