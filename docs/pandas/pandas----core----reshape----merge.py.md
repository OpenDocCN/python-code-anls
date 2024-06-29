# `D:\src\scipysrc\pandas\pandas\core\reshape\merge.py`

```
"""
SQL-style merge routines
"""

# 导入必要的模块和库
from __future__ import annotations  # 导入未来版本的注解支持

from collections.abc import (  # 导入集合抽象基类中的特定模块
    Hashable,  # 可哈希对象
    Sequence,  # 序列类型
)
import datetime  # 导入日期时间模块
from functools import partial  # 导入 partial 函数
from typing import (  # 导入类型提示模块中的特定类型
    TYPE_CHECKING,  # 类型检查标志
    Literal,  # 字面值类型
    cast,  # 强制类型转换函数
    final,  # final 类修饰符
)
import uuid  # 导入 UUID 模块
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库

from pandas._libs import (  # 导入 Pandas 私有 C 扩展库
    Timedelta,  # 时间增量类型
    hashtable as libhashtable,  # 哈希表操作
    join as libjoin,  # 数据合并操作
    lib,  # 核心库
)
from pandas._libs.lib import is_range_indexer  # 判断是否为范围索引器
from pandas._typing import (  # 导入 Pandas 类型提示中的特定类型
    AnyArrayLike,  # 任意类数组类型
    ArrayLike,  # 类数组类型
    IndexLabel,  # 索引标签类型
    JoinHow,  # 连接方式类型
    MergeHow,  # 合并方式类型
    Shape,  # 形状类型
    Suffixes,  # 后缀类型
    npt,  # NumPy 类型
)
from pandas.errors import MergeError  # 导入合并错误异常
from pandas.util._decorators import cache_readonly  # 导入缓存只读装饰器
from pandas.util._exceptions import find_stack_level  # 导入查找堆栈级别异常

from pandas.core.dtypes.base import ExtensionDtype  # 导入扩展类型基类
from pandas.core.dtypes.cast import find_common_type  # 导入查找公共类型函数
from pandas.core.dtypes.common import (  # 导入常见类型判断函数
    ensure_int64,  # 确保为 int64 类型
    ensure_object,  # 确保为对象类型
    is_bool,  # 判断是否为布尔类型
    is_bool_dtype,  # 判断是否为布尔类型的数据框列
    is_float_dtype,  # 判断是否为浮点数类型的数据框列
    is_integer,  # 判断是否为整数
    is_integer_dtype,  # 判断是否为整数类型的数据框列
    is_list_like,  # 判断是否为类列表对象
    is_number,  # 判断是否为数字
    is_numeric_dtype,  # 判断是否为数值类型的数据框列
    is_object_dtype,  # 判断是否为对象类型的数据框列
    is_string_dtype,  # 判断是否为字符串类型的数据框列
    needs_i8_conversion,  # 是否需要进行 int64 转换
)
from pandas.core.dtypes.dtypes import (  # 导入 Pandas 数据类型模块中的特定数据类型
    CategoricalDtype,  # 类别数据类型
    DatetimeTZDtype,  # 带有时区的日期时间数据类型
)
from pandas.core.dtypes.generic import (  # 导入 Pandas 通用数据类型模块中的特定类型
    ABCDataFrame,  # 数据框架抽象基类
    ABCSeries,  # 系列抽象基类
)
from pandas.core.dtypes.missing import (  # 导入 Pandas 缺失值模块中的特定函数和类
    isna,  # 判断是否为缺失值
    na_value_for_dtype,  # 返回指定数据类型的缺失值
)

from pandas import (  # 导入 Pandas 核心对象
    ArrowDtype,  # Arrow 类型
    Categorical,  # 类别数据类型
    Index,  # 索引对象
    MultiIndex,  # 多级索引对象
    Series,  # 系列对象
)
import pandas.core.algorithms as algos  # 导入 Pandas 核心算法模块
from pandas.core.arrays import (  # 导入 Pandas 数组模块中的特定数组类型
    ArrowExtensionArray,  # Arrow 扩展数组类型
    BaseMaskedArray,  # 基本掩码数组类型
    ExtensionArray,  # 扩展数组类型
)
from pandas.core.arrays.string_ import StringDtype  # 导入字符串数据类型
import pandas.core.common as com  # 导入 Pandas 核心公共模块
from pandas.core.construction import (  # 导入 Pandas 构建模块中的特定函数
    ensure_wrapped_if_datetimelike,  # 如果是日期时间对象，则确保包装
    extract_array,  # 提取数组
)
from pandas.core.indexes.api import default_index  # 导入默认索引函数
from pandas.core.sorting import (  # 导入排序模块中的特定函数
    get_group_index,  # 获取分组索引
    is_int64_overflow_possible,  # 判断是否可能出现 int64 溢出
)

if TYPE_CHECKING:  # 如果是类型检查模式
    from pandas import DataFrame  # 导入数据框类型
    from pandas.core import groupby  # 导入分组模块
    from pandas.core.arrays import DatetimeArray  # 导入日期时间数组类型
    from pandas.core.indexes.frozen import FrozenList  # 导入冻结列表类型

# 初始化因子化器字典，将 NumPy 类型映射到 Pandas 的哈希表因子化器
_factorizers = {
    np.int64: libhashtable.Int64Factorizer,
    np.longlong: libhashtable.Int64Factorizer,
    np.int32: libhashtable.Int32Factorizer,
    np.int16: libhashtable.Int16Factorizer,
    np.int8: libhashtable.Int8Factorizer,
    np.uint64: libhashtable.UInt64Factorizer,
    np.uint32: libhashtable.UInt32Factorizer,
    np.uint16: libhashtable.UInt16Factorizer,
    np.uint8: libhashtable.UInt8Factorizer,
    np.bool_: libhashtable.UInt8Factorizer,
    np.float64: libhashtable.Float64Factorizer,
    np.float32: libhashtable.Float32Factorizer,
    np.complex64: libhashtable.Complex64Factorizer,
    np.complex128: libhashtable.Complex128Factorizer,
    np.object_: libhashtable.ObjectFactorizer,
}

# 根据 NumPy 的 intc 类型是否与 int32 相同，决定是否添加到因子化器字典中
if np.intc is not np.int32:
    _factorizers[np.intc] = libhashtable.Int64Factorizer
# 定义一个元组，包含几种特定的类型，用于类型检查
_known = (np.ndarray, ExtensionArray, Index, ABCSeries)

# 定义一个函数merge，用于合并DataFrame或命名Series对象，实现数据库风格的连接操作
def merge(
    left: DataFrame | Series,  # 第一个要合并的pandas对象，可以是DataFrame或命名Series
    right: DataFrame | Series,  # 第二个要合并的pandas对象，可以是DataFrame或命名Series
    how: MergeHow = "inner",  # 合并方式，默认为"inner"，支持'left', 'right', 'outer', 'inner', 'cross'
    on: IndexLabel | AnyArrayLike | None = None,  # 用于连接的列名或索引级别名称，必须在两个DataFrame中都存在
    left_on: IndexLabel | AnyArrayLike | None = None,  # 左边DataFrame中用于连接的列名或索引级别名称，可以是单个标签或列表
    right_on: IndexLabel | AnyArrayLike | None = None,  # 右边DataFrame中用于连接的列名或索引级别名称，可以是单个标签或列表
    left_index: bool = False,  # 是否使用左边DataFrame的索引作为连接键，默认为False
    right_index: bool = False,  # 是否使用右边DataFrame的索引作为连接键，默认为False
    sort: bool = False,  # 是否按连接键对结果进行排序，默认为False
    suffixes: Suffixes = ("_x", "_y"),  # 用于区分重叠列名的后缀，默认为("_x", "_y")
    copy: bool | lib.NoDefault = lib.no_default,  # 是否复制数据，默认根据库的默认行为
    indicator: str | bool = False,  # 是否在结果中添加指示器列，默认为False
    validate: str | None = None,  # 是否检查合并方式的有效性，默认为None
) -> DataFrame:  # 返回一个DataFrame对象作为合并后的结果
    """
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
    ----------
    left : DataFrame or named Series
        First pandas object to merge.
    right : DataFrame or named Series
        Second pandas object to merge.
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
    """
    # 函数主体，实现DataFrame或Series的合并操作
    # left_index : bool, default False
    # 将左侧 DataFrame 的索引用作连接键。如果是 MultiIndex，则右侧 DataFrame（可以是索引或多列）的键数量必须与左侧 DataFrame 的级数匹配。

    # right_index : bool, default False
    # 将右侧 DataFrame 的索引用作连接键。具有与 left_index 相同的注意事项。

    # sort : bool, default False
    # 在结果 DataFrame 中按字典顺序对连接键进行排序。如果为 False，则连接键的顺序取决于连接类型（如何关键字的用法）。

    # suffixes : list-like, default is ("_x", "_y")
    # 长度为 2 的序列，每个元素都是一个字符串，用于指示要添加到左侧和右侧重叠列名中的后缀。如果一个值是 `None`，则表示应保留来自左侧或右侧的列名，不添加后缀。至少一个值必须不是 `None`。

    # copy : bool, default False
    # 如果为 False，则尽可能避免复制。

    # .. note::
    #     `copy` 关键字将在 pandas 3.0 中改变行为。
    #     `Copy-on-Write` 将默认启用，这意味着所有带有 `copy` 关键字的方法将使用延迟复制机制以推迟复制并忽略 `copy` 关键字。`copy` 关键字将在将来的 pandas 版本中移除。
    #     您可以通过启用 copy on write `pd.options.mode.copy_on_write = True` 来获得未来的行为和改进。

    # .. deprecated:: 3.0.0

    # indicator : bool or str, default False
    # 如果为 True，则在输出 DataFrame 中添加一个名为 "_merge" 的列，列中包含每行的源信息。可以通过提供一个字符串参数来指定不同的列名。该列将具有分类类型，对于仅出现在左侧 DataFrame 中的合并键的观察结果将为 "left_only"，对于仅出现在右侧 DataFrame 中的合并键的观察结果将为 "right_only"，如果合并键在两个 DataFrame 中都找到，则为 "both"。

    # validate : str, optional
    # 如果指定，则检查合并的类型。

    # * "one_to_one" 或 "1:1"：检查左侧和右侧数据集中合并键是否在唯一。
    # * "one_to_many" 或 "1:m"：检查左侧数据集中合并键是否唯一。
    # * "many_to_one" 或 "m:1"：检查右侧数据集中合并键是否唯一。
    # * "many_to_many" 或 "m:m"：允许，但不会进行检查。

    # Returns
    # -------
    # DataFrame
    #     两个合并对象的 DataFrame。

    # See Also
    # --------
    # merge_ordered : 带有可选填充/插值的合并。
    # merge_asof : 最近键上的合并。
    # 使用DataFrame的join方法，将两个DataFrame对象按照索引进行连接。
    
    Examples
    --------
    >>> df1 = pd.DataFrame(
    ...     {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 5]}
    ... )
    >>> df2 = pd.DataFrame(
    ...     {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]}
    ... )
    >>> df1
        lkey  value
    0   foo      1
    1   bar      2
    2   baz      3
    3   foo      5
    >>> df2
        rkey  value
    0   foo      5
    1   bar      6
    2   baz      7
    3   foo      8
    
    # 使用merge方法，按照指定的左右列(lkey和rkey)进行连接，将value列默认添加后缀_x和_y。
    >>> df1.merge(df2, left_on="lkey", right_on="rkey")
      lkey  value_x rkey  value_y
    0  foo        1  foo        5
    1  foo        1  foo        8
    2  bar        2  bar        6
    3  baz        3  baz        7
    4  foo        5  foo        5
    5  foo        5  foo        8
    
    # 使用merge方法，按照指定的左右列(lkey和rkey)进行连接，并在重叠列上添加指定的后缀。
    >>> df1.merge(df2, left_on="lkey", right_on="rkey", suffixes=("_left", "_right"))
      lkey  value_left rkey  value_right
    0  foo           1  foo            5
    1  foo           1  foo            8
    2  bar           2  bar            6
    3  baz           3  baz            7
    4  foo           5  foo            5
    5  foo           5  foo            8
    
    # 使用merge方法，按照指定的左右列(lkey和rkey)进行连接，但如果DataFrame有任何重叠列则引发异常。
    >>> df1.merge(df2, left_on="lkey", right_on="rkey", suffixes=(False, False))
    Traceback (most recent call last):
    ...
    ValueError: columns overlap but no suffix specified:
        Index(['value'], dtype='object')
    
    >>> df1 = pd.DataFrame({"a": ["foo", "bar"], "b": [1, 2]})
    >>> df2 = pd.DataFrame({"a": ["foo", "baz"], "c": [3, 4]})
    >>> df1
          a  b
    0   foo  1
    1   bar  2
    >>> df2
          a  c
    0   foo  3
    1   baz  4
    
    # 使用merge方法，按照列'a'进行内连接。
    >>> df1.merge(df2, how="inner", on="a")
          a  b  c
    0   foo  1  3
    
    # 使用merge方法，按照列'a'进行左连接。
    >>> df1.merge(df2, how="left", on="a")
          a  b    c
    0   foo  1  3.0
    1   bar  2  NaN
    
    >>> df1 = pd.DataFrame({"left": ["foo", "bar"]})
    >>> df2 = pd.DataFrame({"right": [7, 8]})
    >>> df1
        left
    0   foo
    1   bar
    >>> df2
        right
    0   7
    1   8
    
    # 使用merge方法，对df1和df2进行交叉连接。
    >>> df1.merge(df2, how="cross")
       left  right
    0   foo      7
    1   foo      8
    2   bar      7
    3   bar      8
    # 如果合并方式为交叉方式（cross），则调用_cross_merge函数进行合并
    if how == "cross":
        return _cross_merge(
            left_df,
            right_df,
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
    # 如果合并方式不是交叉方式，则创建一个_MergeOperation对象进行合并操作
    else:
        op = _MergeOperation(
            left_df,
            right_df,
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
        # 返回_MergeOperation对象的结果
        return op.get_result()
def _cross_merge(
    left: DataFrame,
    right: DataFrame,
    on: IndexLabel | AnyArrayLike | None = None,
    left_on: IndexLabel | AnyArrayLike | None = None,
    right_on: IndexLabel | AnyArrayLike | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    indicator: str | bool = False,
    validate: str | None = None,
) -> DataFrame:
    """
    See merge.__doc__ with how='cross'
    """

    # 检查参数设置是否与交叉合并兼容，若不兼容则引发异常
    if (
        left_index
        or right_index
        or right_on is not None
        or left_on is not None
        or on is not None
    ):
        raise MergeError(
            "Can not pass on, right_on, left_on or set right_index=True or "
            "left_index=True"
        )

    # 创建一个唯一的交叉列名
    cross_col = f"_cross_{uuid.uuid4()}"
    # 在左侧数据框中添加交叉列
    left = left.assign(**{cross_col: 1})
    # 在右侧数据框中添加交叉列
    right = right.assign(**{cross_col: 1})

    # 将交叉列设置为左右数据框的连接键
    left_on = right_on = [cross_col]

    # 执行内连接的数据框合并操作
    res = merge(
        left,
        right,
        how="inner",
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
    # 删除结果数据框中的交叉列
    del res[cross_col]
    # 返回合并后的结果数据框
    return res


def _groupby_and_merge(
    by, left: DataFrame | Series, right: DataFrame | Series, merge_pieces
):
    """
    groupby & merge; we are always performing a left-by type operation

    Parameters
    ----------
    by: field to group
    left: DataFrame
    right: DataFrame
    merge_pieces: function for merging
    """
    pieces = []
    # 如果by不是列表或元组，则转换为列表
    if not isinstance(by, (list, tuple)):
        by = [by]

    # 根据左侧数据框的分组键进行分组
    lby = left.groupby(by, sort=False)
    # 初始化右侧数据框的分组结果变量
    rby: groupby.DataFrameGroupBy | groupby.SeriesGroupBy | None = None

    # 如果右侧数据框的列中包含所有的分组键，则对右侧数据框进行分组
    if all(item in right.columns for item in by):
        rby = right.groupby(by, sort=False)

    # 遍历左侧数据框的每一个分组
    for key, lhs in lby._grouper.get_iterator(lby._selected_obj):
        # 如果右侧数据框未分组，则直接使用右侧数据框
        if rby is None:
            rhs = right
        else:
            try:
                # 根据右侧分组索引获取对应分组的数据
                rhs = right.take(rby.indices[key])
            except KeyError:
                # 如果键在右侧数据框中不存在，处理左侧数据框的列和右侧数据框的不同列，重新索引并添加到结果列表
                lcols = lhs.columns.tolist()
                cols = lcols + [r for r in right.columns if r not in set(lcols)]
                merged = lhs.reindex(columns=cols)
                merged.index = range(len(merged))
                pieces.append(merged)
                continue

        # 使用指定的合并函数合并左右两侧数据
        merged = merge_pieces(lhs, rhs)

        # 确保合并结果中包含连接键
        merged[by] = key

        # 将合并后的结果添加到结果列表中
        pieces.append(merged)

    # 按照原始顺序连接所有结果片段，忽略索引
    from pandas.core.reshape.concat import concat

    result = concat(pieces, ignore_index=True)
    # 重新索引结果，使列顺序与第一个片段保持一致
    result = result.reindex(columns=pieces[0].columns)
    # 返回最终合并和排序后的结果数据框及左侧数据框的分组对象
    return result, lby
    left: DataFrame | Series,
    # 左侧的数据，可以是 DataFrame 或 Series 对象

    right: DataFrame | Series,
    # 右侧的数据，可以是 DataFrame 或 Series 对象

    on: IndexLabel | None = None,
    # 使用的列名或索引，用于连接左右两个数据集；默认为 None 表示使用索引

    left_on: IndexLabel | None = None,
    # 左侧数据集用于连接的列名或索引；默认为 None 表示不指定特定列

    right_on: IndexLabel | None = None,
    # 右侧数据集用于连接的列名或索引；默认为 None 表示不指定特定列

    left_by=None,
    # 左侧数据集用于分组的列名或索引，用于合并多个数据集时；默认为 None 表示不使用分组

    right_by=None,
    # 右侧数据集用于分组的列名或索引，用于合并多个数据集时；默认为 None 表示不使用分组

    fill_method: str | None = None,
    # 如果在连接过程中出现缺失值，用指定方法填充；默认为 None 表示不填充

    suffixes: Suffixes = ("_x", "_y"),
    # 连接时，重叠列名的后缀添加方式；默认为 ("_x", "_y")

    how: JoinHow = "outer",
    # 连接方式，包括 "left", "right", "outer", "inner" 等；默认为 "outer" 表示外连接
# 执行有序数据的合并操作，可选填充/插值。

# 设计用于有序数据（如时间序列数据）的合并。可选择执行分组合并（参见示例）。

# Parameters 参数说明：

# left : DataFrame 或具名 Series
#     第一个要合并的 pandas 对象。
# right : DataFrame 或具名 Series
#     第二个要合并的 pandas 对象。
# on : 标签或列表
#     要加入的字段名。必须在两个 DataFrame 中找到。
# left_on : 标签或列表，或类似数组
#     左侧 DataFrame 中要加入的字段名。可以是长度与 DataFrame 相同的向量或列表，
#     用于使用特定向量作为连接键，而不是列。
# right_on : 标签或列表，或类似数组
#     右侧 DataFrame 中要加入的字段名或向量/列表（见 left_on 文档）。
# left_by : 列名或列名列表
#     按组列对左侧 DataFrame 进行分组，并逐片与右侧 DataFrame 合并。如果左侧或右侧是 Series，则必须为 None。
# right_by : 列名或列名列表
#     按组列对右侧 DataFrame 进行分组，并逐片与左侧 DataFrame 合并。如果左侧或右侧是 Series，则必须为 None。
# fill_method : {'ffill', None}，默认为 None
#     数据的插值方法。
# suffixes : 类似列表，默认为 ("_x", "_y")
#     一个长度为2的序列，每个元素可以选择是字符串，表示要添加到“left”和“right”中重叠列名的后缀。
#     将字符串的值为 None 代替，表示“left”或“right”中的列名应保持不变，不添加后缀。至少有一个值不能为 None。

# how : {'left', 'right', 'outer', 'inner'}，默认为 'outer'
#     * left: 只使用左侧框架的键（SQL: 左外连接）
#     * right: 只使用右侧框架的键（SQL: 右外连接）
#     * outer: 使用两个框架的键的并集（SQL: 完全外连接）
#     * inner: 使用两个框架的键的交集（SQL: 内连接）。

# Returns 返回：

# DataFrame
#     如果 'left' 是 DataFrame 的子类，则合并后的 DataFrame 输出类型将与 'left' 相同。

# See Also 参见：

# merge : 使用类似数据库的连接进行合并。
# merge_asof : 根据最近的键进行合并。

# Examples 示例：

# >>> from pandas import merge_ordered
# >>> df1 = pd.DataFrame(
# ...     {
# ...         "key": ["a", "c", "e", "a", "c", "e"],
# ...         "lvalue": [1, 2, 3, 1, 2, 3],
# ...         "group": ["a", "a", "a", "b", "b", "b"],
# ...     }
# ... )
# >>> df1
#   key  lvalue group
# 0   a       1     a
# 1   c       2     a
# 2   e       3     a
# 3   a       1     b
# 4   c       2     b
# 5   e       3     b

# >>> df2 = pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
# >>> df2
#   key  rvalue
    """
    Perform an ordered merge of two DataFrames based on specified criteria.

    Args:
        x (DataFrame): Left DataFrame for merge operation.
        y (DataFrame): Right DataFrame for merge operation.
        on (str or list): Column(s) name(s) to join on.
        left_on (str or list): Column(s) name(s) from left DataFrame to join on.
        right_on (str or list): Column(s) name(s) from right DataFrame to join on.
        suffixes (tuple): Suffixes to apply to overlapping column names if any.
        fill_method (str): Method to use for filling NaN values during merge.
        how (str): Type of merge to be performed ('left', 'right', 'outer', 'inner').

    Returns:
        DataFrame: Merged DataFrame containing the result of the merge operation.

    Raises:
        ValueError: If both left_by and right_by are provided simultaneously.
        KeyError: If specified columns in left_by or right_by are not found in respective DataFrame.

    Notes:
        This function supports merging based on specific criteria, handling ordered merge operations
        with flexibility in handling missing values and column name conflicts.
    """
    def _merger(x, y) -> DataFrame:
        # perform the ordered merge operation
        op = _OrderedMerge(
            x,
            y,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            fill_method=fill_method,
            how=how,
        )
        return op.get_result()

    if left_by is not None and right_by is not None:
        raise ValueError("Can only group either left or right frames")
    if left_by is not None:
        if isinstance(left_by, str):
            left_by = [left_by]
        # check if specified left_by columns exist in the left DataFrame
        check = set(left_by).difference(left.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in left columns")
        # perform grouping and merging using _groupby_and_merge function
        result, _ = _groupby_and_merge(left_by, left, right, lambda x, y: _merger(x, y))
    elif right_by is not None:
        if isinstance(right_by, str):
            right_by = [right_by]
        # check if specified right_by columns exist in the right DataFrame
        check = set(right_by).difference(right.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in right columns")
        # perform grouping and merging using _groupby_and_merge function, with reversed order of parameters
        result, _ = _groupby_and_merge(
            right_by, right, left, lambda x, y: _merger(y, x)
        )
    else:
        # if neither left_by nor right_by are specified, directly perform the merge using _merger function
        result = _merger(left, right)
    return result
# 定义一个函数，实现按键距离合并操作
def merge_asof(
    left: DataFrame | Series,  # 第一个要合并的 pandas 对象，可以是 DataFrame 或 Series
    right: DataFrame | Series,  # 第二个要合并的 pandas 对象，可以是 DataFrame 或 Series
    on: IndexLabel | None = None,  # 连接键的字段名，必须在两个 DataFrame 中找到，数据必须已排序
    left_on: IndexLabel | None = None,  # 左侧 DataFrame 中用于连接的字段名
    right_on: IndexLabel | None = None,  # 右侧 DataFrame 中用于连接的字段名
    left_index: bool = False,  # 是否使用左侧 DataFrame 的索引作为连接键
    right_index: bool = False,  # 是否使用右侧 DataFrame 的索引作为连接键
    by=None,  # 在执行连接操作之前，匹配这些列
    left_by=None,  # 左侧 DataFrame 中要匹配的字段名
    right_by=None,  # 右侧 DataFrame 中要匹配的字段名
    suffixes: Suffixes = ("_x", "_y"),  # 左右两侧列名重叠时应用的后缀
    tolerance: int | datetime.timedelta | None = None,  # asof 操作的容差范围，必须与合并索引兼容
    allow_exact_matches: bool = True,  # 是否允许与相同的 'on' 值进行匹配
        # True: 允许与相同 'on' 值匹配 (<= 或 >=)
        # False: 不允许与相同 'on' 值匹配 (< 或 >)
    direction: str = "backward",  # 搜索匹配项的方向，可以是 'backward', 'forward' 或 'nearest'
) -> DataFrame:
    """
    Perform a merge by key distance.

    This is similar to a left-join except that we match on nearest
    key rather than equal keys. Both DataFrames must be sorted by the key.

    For each row in the left DataFrame:

      - A "backward" search selects the last row in the right DataFrame whose
        'on' key is less than or equal to the left's key.

      - A "forward" search selects the first row in the right DataFrame whose
        'on' key is greater than or equal to the left's key.

      - A "nearest" search selects the row in the right DataFrame whose 'on'
        key is closest in absolute distance to the left's key.

    Optionally match on equivalent keys with 'by' before searching with 'on'.

    Parameters
    ----------
    left : DataFrame or named Series
        First pandas object to merge.
    right : DataFrame or named Series
        Second pandas object to merge.
    on : label
        Field name to join on. Must be found in both DataFrames.
        The data MUST be ordered. Furthermore this must be a numeric column,
        such as datetimelike, integer, or float. On or left_on/right_on
        must be given.
    left_on : label
        Field name to join on in left DataFrame.
    right_on : label
        Field name to join on in right DataFrame.
    left_index : bool
        Use the index of the left DataFrame as the join key.
    right_index : bool
        Use the index of the right DataFrame as the join key.
    by : column name or list of column names
        Match on these columns before performing merge operation.
    left_by : column name
        Field names to match on in the left DataFrame.
    right_by : column name
        Field names to match on in the right DataFrame.
    suffixes : 2-length sequence (tuple, list, ...)
        Suffix to apply to overlapping column names in the left and right
        side, respectively.
    tolerance : int or timedelta, optional, default None
        Select asof tolerance within this range; must be compatible
        with the merge index.
    allow_exact_matches : bool, default True

        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than).

    direction : 'backward' (default), 'forward', or 'nearest'
        Whether to search for prior, subsequent, or closest matches.

    Returns
    -------
    DataFrame
    """
    DataFrame
        A DataFrame of the two merged objects.

    See Also
    --------
    merge : Merge with a database-style join.
    merge_ordered : Merge with optional filling/interpolation.

    Examples
    --------
    >>> left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    >>> left
        a left_val
    0   1        a
    1   5        b
    2  10        c

    >>> right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})
    >>> right
       a  right_val
    0  1          1
    1  2          2
    2  3          3
    3  6          6
    4  7          7

    >>> pd.merge_asof(left, right, on="a")
        a left_val  right_val
    0   1        a          1
    1   5        b          3
    2  10        c          7

    >>> pd.merge_asof(left, right, on="a", allow_exact_matches=False)
        a left_val  right_val
    0   1        a        NaN
    1   5        b        3.0
    2  10        c        7.0

    >>> pd.merge_asof(left, right, on="a", direction="forward")
        a left_val  right_val
    0   1        a        1.0
    1   5        b        6.0
    2  10        c        NaN

    >>> pd.merge_asof(left, right, on="a", direction="nearest")
        a left_val  right_val
    0   1        a          1
    1   5        b          6
    2  10        c          7

    We can use indexed DataFrames as well.

    >>> left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
    >>> left
       left_val
    1         a
    5         b
    10        c

    >>> right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    >>> right
       right_val
    1          1
    2          2
    3          3
    6          6
    7          7

    >>> pd.merge_asof(left, right, left_index=True, right_index=True)
       left_val  right_val
    1         a          1
    5         b          3
    10        c          7

    Here is a real-world times-series example

    >>> quotes = pd.DataFrame(
    ...     {
    ...         "time": [
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.030"),
    ...             pd.Timestamp("2016-05-25 13:30:00.041"),
    ...             pd.Timestamp("2016-05-25 13:30:00.048"),
    ...             pd.Timestamp("2016-05-25 13:30:00.049"),
    ...             pd.Timestamp("2016-05-25 13:30:00.072"),
    ...             pd.Timestamp("2016-05-25 13:30:00.075"),
    ...         ],
    ...         "ticker": [
    ...             "GOOG",
    ...             "MSFT",
    ...             "MSFT",
    ...             "MSFT",
    ...             "GOOG",
    ...             "AAPL",
    ...             "GOOG",
    ...             "MSFT",
    ...         ],
    ...         "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
    ...         "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
    ...     }
    ... )


注释：
    # 创建一个名为 quotes 的 DataFrame，包含时间、股票代码、买入价和卖出价的数据
    quotes = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.030"),
                pd.Timestamp("2016-05-25 13:30:00.041"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
                pd.Timestamp("2016-05-25 13:30:00.075"),
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
        }
    )
    
    # 展示 quotes 的内容
    quotes
    
    # 创建一个名为 trades 的 DataFrame，包含时间、股票代码、价格和数量的数据
    trades = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.038"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
            ],
            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            "quantity": [75, 155, 100, 100, 100],
        }
    )
    
    # 展示 trades 的内容
    trades
    
    # 使用默认设置对 trades 和 quotes DataFrame 进行按时间合并（asof merge）
    # 按照时间和股票代码匹配，将 quotes 的买入价和卖出价合并到 trades 中
    pd.merge_asof(trades, quotes, on="time", by="ticker")
    
    # 对 trades 和 quotes DataFrame 进行按时间合并（asof merge）
    # 按照时间和股票代码匹配，但只考虑时间在两者之间的差值不超过 2 毫秒（ms）的数据
    pd.merge_asof(
        trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms")
    )
    
    # 对 trades 和 quotes DataFrame 进行按时间合并（asof merge）
    # 按照时间和股票代码匹配，但只考虑时间在两者之间的差值不超过 10 毫秒（ms）的数据
    # 并且排除时间上的精确匹配，确保只向前传播之前的数据
    pd.merge_asof(
        trades,
        quotes,
        on="time",
        by="ticker",
        tolerance=pd.Timedelta("10ms"),
        allow_exact_matches=False,
    )
    """
    使用指定的参数进行按时序合并（asof merge）左右两个数据集。

    op = _AsOfMerge(
        left,
        right,
        on=on,  # 指定连接的列名或列名列表（左数据集）
        left_on=left_on,  # 指定左侧数据集连接的列名或列名列表
        right_on=right_on,  # 指定右侧数据集连接的列名或列名列表
        left_index=left_index,  # 如果为True，使用左侧数据集的索引作为连接键
        right_index=right_index,  # 如果为True，使用右侧数据集的索引作为连接键
        by=by,  # 需要按列连接的列名或列名列表
        left_by=left_by,  # 左侧数据集用于按列连接的列名或列名列表
        right_by=right_by,  # 右侧数据集用于按列连接的列名或列名列表
        suffixes=suffixes,  # 如果列名冲突，用于附加到列名的后缀元组
        how="asof",  # 连接方式，这里指定为按照时间顺序合并
        tolerance=tolerance,  # 允许的时间匹配容差
        allow_exact_matches=allow_exact_matches,  # 是否允许完全匹配
        direction=direction,  # 时间匹配的方向
    )
    返回按时序合并后的结果。
    """
# TODO: transformations??
# 定义一个名为 _MergeOperation 的类，用于执行数据库（SQL）的合并操作，
# 可以在两个 DataFrame 或 Series 对象之间使用列作为键或它们的行索引
class _MergeOperation:
    """
    Perform a database (SQL) merge operation between two DataFrame or Series
    objects using either columns as keys or their row indexes
    """

    # 定义合并操作类型为 "merge"
    _merge_type = "merge"
    # 指定合并方式，可以是 JoinHow 类型或者字符串字面值 "asof"
    how: JoinHow | Literal["asof"]
    # 指定合并时的连接键，可以是索引标签或者 None
    on: IndexLabel | None
    # 在验证规范时，left_on 和 right_on 可能为 None，但在验证过程中会被替换为非 None 值
    left_on: Sequence[Hashable | AnyArrayLike]
    right_on: Sequence[Hashable | AnyArrayLike]
    # 是否使用左侧对象的索引进行合并
    left_index: bool
    # 是否使用右侧对象的索引进行合并
    right_index: bool
    # 是否对合并后的结果进行排序
    sort: bool
    # 合并后重复列名的后缀
    suffixes: Suffixes
    # 指示是否将合并的信息作为新的列添加到结果中
    indicator: str | bool
    # 验证合并的类型，默认为 None
    validate: str | None
    # 合并时参与的连接列的名称列表
    join_names: list[Hashable]
    # 右侧连接键的列表，可以是数组样式
    right_join_keys: list[ArrayLike]
    # 左侧连接键的列表，可以是数组样式
    left_join_keys: list[ArrayLike]

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        how: JoinHow | Literal["asof"] = "inner",
        on: IndexLabel | AnyArrayLike | None = None,
        left_on: IndexLabel | AnyArrayLike | None = None,
        right_on: IndexLabel | AnyArrayLike | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = True,
        suffixes: Suffixes = ("_x", "_y"),
        indicator: str | bool = False,
        validate: str | None = None,
        join_names: list[Hashable],
        right_join_keys: list[ArrayLike],
        left_join_keys: list[ArrayLike]
    ):
        pass  # 初始化方法暂未实现具体逻辑，仅定义参数列表
    ) -> None:
        # 验证并返回经过验证的左操作数
        _left = _validate_operand(left)
        # 验证并返回经过验证的右操作数
        _right = _validate_operand(right)
        # 设置实例的左操作数及原始左操作数
        self.left = self.orig_left = _left
        # 设置实例的右操作数及原始右操作数
        self.right = self.orig_right = _right
        # 设置合并方式
        self.how = how

        # 将 'on' 参数转换为列表（如果不是），并赋给实例属性 self.on
        self.on = com.maybe_make_list(on)

        # 设置合并后的列名后缀
        self.suffixes = suffixes
        # 根据是否指定了排序或合并方式为 "outer"，确定是否排序
        self.sort = sort or how == "outer"

        # 设置左操作数的索引
        self.left_index = left_index
        # 设置右操作数的索引
        self.right_index = right_index

        # 设置指示器，指示是否进行了合并
        self.indicator = indicator

        # 如果 left_index 不是布尔型，抛出 ValueError 异常
        if not is_bool(left_index):
            raise ValueError(
                f"left_index parameter must be of type bool, not {type(left_index)}"
            )
        # 如果 right_index 不是布尔型，抛出 ValueError 异常
        if not is_bool(right_index):
            raise ValueError(
                f"right_index parameter must be of type bool, not {type(right_index)}"
            )

        # 检查左右操作数的列级别数是否相同，不相同则抛出 MergeError 异常
        if _left.columns.nlevels != _right.columns.nlevels:
            msg = (
                "Not allowed to merge between different levels. "
                f"({_left.columns.nlevels} levels on the left, "
                f"{_right.columns.nlevels} on the right)"
            )
            raise MergeError(msg)

        # 验证并返回左右操作数的合并键
        self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)

        # 获取合并的键值对、合并后的列名、需删除的左右操作数的标签或级别
        (
            self.left_join_keys,
            self.right_join_keys,
            self.join_names,
            left_drop,
            right_drop,
        ) = self._get_merge_keys()

        # 如果需删除左操作数的标签或级别，执行删除操作
        if left_drop:
            self.left = self.left._drop_labels_or_levels(left_drop)

        # 如果需删除右操作数的标签或级别，执行删除操作
        if right_drop:
            self.right = self.right._drop_labels_or_levels(right_drop)

        # 检查合并键的数据类型是否匹配，必要时进行强制类型转换
        self._maybe_require_matching_dtypes(self.left_join_keys, self.right_join_keys)
        # 验证合并键的容忍度
        self._validate_tolerance(self.left_join_keys)

        # 如果传入了 validate 参数，验证指定为唯一的列是否确实唯一
        if validate is not None:
            self._validate_validate_kwd(validate)
    ) -> DataFrame:
        """
        重新索引索引并沿列连接。
        """
        # 获取左右数据框的视图，以防修改原始数据
        left = self.left[:]
        right = self.right[:]

        # 根据指定的后缀计算重叠的索引标签
        llabels, rlabels = _items_overlap_with_suffix(
            self.left._info_axis, self.right._info_axis, self.suffixes
        )

        if left_indexer is not None and not is_range_indexer(left_indexer, len(left)):
            # 如果左侧索引器不为空且不是范围索引器，重新索引左侧数据框
            lmgr = left._mgr.reindex_indexer(
                join_index,
                left_indexer,
                axis=1,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            left = left._constructor_from_mgr(lmgr, axes=lmgr.axes)
        left.index = join_index

        if right_indexer is not None and not is_range_indexer(
            right_indexer, len(right)
        ):
            # 如果右侧索引器不为空且不是范围索引器，重新索引右侧数据框
            rmgr = right._mgr.reindex_indexer(
                join_index,
                right_indexer,
                axis=1,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            right = right._constructor_from_mgr(rmgr, axes=rmgr.axes)
        right.index = join_index

        from pandas import concat

        # 设置左右数据框的列标签
        left.columns = llabels
        right.columns = rlabels

        # 沿列方向连接左右数据框
        result = concat([left, right], axis=1)
        return result

    def get_result(self) -> DataFrame:
        if self.indicator:
            # 如果指示器标志为True，进行预合并操作
            self.left, self.right = self._indicator_pre_merge(self.left, self.right)

        # 获取连接信息：连接索引和左右索引器
        join_index, left_indexer, right_indexer = self._get_join_info()

        # 重新索引并连接数据，得到合并结果
        result = self._reindex_and_concat(join_index, left_indexer, right_indexer)
        # 使用指定的方法（_merge_type）对结果进行最终化处理
        result = result.__finalize__(self, method=self._merge_type)

        if self.indicator:
            # 如果指示器标志为True，进行后合并操作
            result = self._indicator_post_merge(result)

        # 可能添加连接键
        self._maybe_add_join_keys(result, left_indexer, right_indexer)

        # 可能恢复索引级别
        self._maybe_restore_index_levels(result)

        # 返回最终化的结果，使用merge方法
        return result.__finalize__(self, method="merge")

    @final
    @cache_readonly
    def _indicator_name(self) -> str | None:
        if isinstance(self.indicator, str):
            # 如果指示器是字符串，直接返回
            return self.indicator
        elif isinstance(self.indicator, bool):
            # 如果指示器是布尔值，根据其值返回相应的标识名
            return "_merge" if self.indicator else None
        else:
            # 抛出错误，指示器选项只能接受布尔值或字符串参数
            raise ValueError(
                "indicator option can only accept boolean or string arguments"
            )

    @final
    def _indicator_pre_merge(
        self, left: DataFrame, right: DataFrame
        ):
        # 在合并之前处理指示器
    ) -> tuple[DataFrame, DataFrame]:
        # 合并两个 DataFrame 的列，并且使用联合后的列名
        columns = left.columns.union(right.columns)

        # 检查是否存在名为 "_left_indicator" 或 "_right_indicator" 的列，若存在则抛出异常
        for i in ["_left_indicator", "_right_indicator"]:
            if i in columns:
                raise ValueError(
                    "Cannot use `indicator=True` option when "
                    f"data contains a column named {i}"
                )
        
        # 检查是否已经存在名为 self._indicator_name 的列，若存在则抛出异常
        if self._indicator_name in columns:
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )

        # 复制 left 和 right DataFrame，避免修改原始数据
        left = left.copy()
        right = right.copy()

        # 在 left DataFrame 中添加名为 "_left_indicator" 的列，值为 1，并转换为 int8 类型
        left["_left_indicator"] = 1
        left["_left_indicator"] = left["_left_indicator"].astype("int8")

        # 在 right DataFrame 中添加名为 "_right_indicator" 的列，值为 2，并转换为 int8 类型
        right["_right_indicator"] = 2
        right["_right_indicator"] = right["_right_indicator"].astype("int8")

        # 返回修改后的 left 和 right DataFrame
        return left, right

    @final
    def _indicator_post_merge(self, result: DataFrame) -> DataFrame:
        # 将结果中的 "_left_indicator" 和 "_right_indicator" 列的缺失值填充为 0
        result["_left_indicator"] = result["_left_indicator"].fillna(0)
        result["_right_indicator"] = result["_right_indicator"].fillna(0)

        # 根据 "_left_indicator" 和 "_right_indicator" 列的和创建分类变量，并赋给 self._indicator_name 列
        result[self._indicator_name] = Categorical(
            (result["_left_indicator"] + result["_right_indicator"]),
            categories=[1, 2, 3],
        )
        
        # 重命名 self._indicator_name 列的分类标签为 ["left_only", "right_only", "both"]
        result[self._indicator_name] = result[
            self._indicator_name
        ].cat.rename_categories(["left_only", "right_only", "both"])

        # 在结果中删除 "_left_indicator" 和 "_right_indicator" 列
        result = result.drop(labels=["_left_indicator", "_right_indicator"], axis=1)
        
        # 返回处理后的结果 DataFrame
        return result

    @final
    def _maybe_restore_index_levels(self, result: DataFrame) -> None:
        """
        Restore index levels specified as `on` parameters

        Here we check for cases where `self.left_on` and `self.right_on` pairs
        each reference an index level in their respective DataFrames. The
        joined columns corresponding to these pairs are then restored to the
        index of `result`.

        **Note:** This method has side effects. It modifies `result` in-place

        Parameters
        ----------
        result: DataFrame
            merge result

        Returns
        -------
        None
        """
        # List to store index names that need to be restored
        names_to_restore = []
        # Iterate over the join names, left_on keys, and right_on keys
        for name, left_key, right_key in zip(
            self.join_names, self.left_on, self.right_on
        ):
            if (
                # Check if left_key references an index level in orig_left DataFrame
                self.orig_left._is_level_reference(left_key)  # type: ignore[arg-type]
                # Check if right_key references an index level in orig_right DataFrame
                and self.orig_right._is_level_reference(
                    right_key  # type: ignore[arg-type]
                )
                # Check if left_key and right_key are equal and name is not already in result's index names
                and left_key == right_key
                and name not in result.index.names
            ):
                # If conditions are met, add name to names_to_restore list
                names_to_restore.append(name)

        # If there are names to restore, set them as the index of the result DataFrame
        if names_to_restore:
            result.set_index(names_to_restore, inplace=True)

    @final
    def _maybe_add_join_keys(
        self,
        result: DataFrame,
        left_indexer: npt.NDArray[np.intp] | None,
        right_indexer: npt.NDArray[np.intp] | None,
    ) -> None:
        """
        Conditionally add join keys to the result DataFrame

        Parameters
        ----------
        result: DataFrame
            merge result
        left_indexer: np.ndarray[np.intp] | None
            left DataFrame indexer array
        right_indexer: np.ndarray[np.intp] | None
            right DataFrame indexer array

        Returns
        -------
        None
        """
        # This method is designed to add join keys based on the specified conditions,
        # and it directly modifies the result DataFrame in-place.

    def _get_join_indexers(
        self,
    ) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """
        Return the join indexers for the merge operation

        Returns
        -------
        tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]
            Tuple containing left and right indexers for the merge operation
        """
        # Ensure that 'asof' joins are not allowed
        assert self.how != "asof"
        # Return the join indexers computed by the get_join_indexers function
        return get_join_indexers(
            self.left_join_keys, self.right_join_keys, sort=self.sort, how=self.how
        )

    @final
    def _get_join_info(
        self,
    ) -> tuple[DataFrame, Series, Series]:
        """
        Return information about the join operation

        Returns
        -------
        tuple[DataFrame, Series, Series]
            Tuple containing DataFrame and Series related to the join operation
        """
        # This method is responsible for returning information about the join operation,
        # specifically returning a tuple of DataFrame, Series, and Series.
        ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # 获取左侧和右侧数据框的索引
        left_ax = self.left.index
        right_ax = self.right.index

        # 如果左侧和右侧都有索引，并且连接方式不是 "asof"
        if self.left_index and self.right_index and self.how != "asof":
            # 执行索引连接，返回连接后的索引以及左右两侧的索引器
            join_index, left_indexer, right_indexer = left_ax.join(
                right_ax, how=self.how, return_indexers=True, sort=self.sort
            )

        # 如果只有右侧有索引，并且连接方式是 "left"
        elif self.right_index and self.how == "left":
            # 执行左连接，基于索引进行连接，返回连接后的索引以及左侧的索引器
            join_index, left_indexer, right_indexer = _left_join_on_index(
                left_ax, right_ax, self.left_join_keys, sort=self.sort
            )

        # 如果只有左侧有索引，并且连接方式是 "right"
        elif self.left_index and self.how == "right":
            # 执行右连接，基于索引进行连接，返回连接后的索引以及右侧的索引器
            join_index, right_indexer, left_indexer = _left_join_on_index(
                right_ax, left_ax, self.right_join_keys, sort=self.sort
            )

        # 如果以上条件都不满足，则执行默认的连接方式
        else:
            # 获取默认的左右索引器
            (left_indexer, right_indexer) = self._get_join_indexers()

            # 如果只有右侧有索引
            if self.right_index:
                if len(self.left) > 0:
                    # 创建右连接的索引
                    join_index = self._create_join_index(
                        left_ax,
                        right_ax,
                        left_indexer,
                        how="right",
                    )
                elif right_indexer is None:
                    join_index = right_ax.copy()
                else:
                    join_index = right_ax.take(right_indexer)

            # 如果只有左侧有索引
            elif self.left_index:
                if self.how == "asof":
                    # 如果连接方式是 "asof"，表现为左连接
                    # GH#33463 asof 应始终表现为左连接
                    join_index = self._create_join_index(
                        left_ax,
                        right_ax,
                        left_indexer,
                        how="left",
                    )
                elif len(self.right) > 0:
                    # 创建左连接的索引
                    join_index = self._create_join_index(
                        right_ax,
                        left_ax,
                        right_indexer,
                        how="left",
                    )
                elif left_indexer is None:
                    join_index = left_ax.copy()
                else:
                    join_index = left_ax.take(left_indexer)
            else:
                # 如果左右两侧都没有索引，根据左侧索引器的情况确定连接索引的长度
                n = len(left_ax) if left_indexer is None else len(left_indexer)
                join_index = default_index(n)

        # 返回最终的连接索引以及左右索引器
        return join_index, left_indexer, right_indexer

    @final
    def _create_join_index(
        self,
        index: Index,
        other_index: Index,
        indexer: npt.NDArray[np.intp] | None,
        how: JoinHow = "left",
    ) -> Index:
        """
        Create a join index by rearranging one index to match another

        Parameters
        ----------
        index : Index
            index being rearranged
        other_index : Index
            used to supply values not found in index
        indexer : np.ndarray[np.intp] or None
            how to rearrange index
        how : str
            Replacement is only necessary if indexer based on other_index.

        Returns
        -------
        Index
        """
        if self.how in (how, "outer") and not isinstance(other_index, MultiIndex):
            # 如果最终索引需要其他索引中的值而目标索引中没有，indexer 可能包含缺失值(-1)，
            # 导致 Index.take 取目标索引中的最后一个值。因此，我们将最后一个元素设置为期望的填充值。
            # 我们不使用 allow_fill 和 fill_value，因为对整数索引会引发 ValueError 异常。
            mask = indexer == -1
            if np.any(mask):
                fill_value = na_value_for_dtype(index.dtype, compat=False)
                index = index.append(Index([fill_value]))
        if indexer is None:
            # 如果 indexer 为 None，则返回 index 的副本
            return index.copy()
        # 否则，根据 indexer 返回重排后的 index
        return index.take(indexer)

    @final
    def _get_merge_keys(
        self,
    ) -> tuple[
        list[ArrayLike],
        list[ArrayLike],
        list[Hashable],
        list[Hashable],
        list[Hashable],
    @final
    @final
    def _validate_validate_kwd(self, validate: str) -> None:
        # 检查每个合并键的唯一性
        if self.left_index:
            # 如果存在左侧索引，则检查原始左侧数据的索引是否唯一
            left_unique = self.orig_left.index.is_unique
        else:
            # 否则，根据左侧连接键创建一个多重索引，并检查其唯一性
            left_unique = MultiIndex.from_arrays(self.left_join_keys).is_unique

        if self.right_index:
            # 如果存在右侧索引，则检查原始右侧数据的索引是否唯一
            right_unique = self.orig_right.index.is_unique
        else:
            # 否则，根据右侧连接键创建一个多重索引，并检查其唯一性
            right_unique = MultiIndex.from_arrays(self.right_join_keys).is_unique

        # 检查数据完整性
        if validate in ["one_to_one", "1:1"]:
            # 对于一对一合并，必须确保左右两侧的合并键都是唯一的
            if not left_unique and not right_unique:
                raise MergeError(
                    "Merge keys are not unique in either left "
                    "or right dataset; not a one-to-one merge"
                )
            if not left_unique:
                raise MergeError(
                    "Merge keys are not unique in left dataset; not a one-to-one merge"
                )
            if not right_unique:
                raise MergeError(
                    "Merge keys are not unique in right dataset; not a one-to-one merge"
                )

        elif validate in ["one_to_many", "1:m"]:
            # 对于一对多合并，只需确保左侧的合并键是唯一的
            if not left_unique:
                raise MergeError(
                    "Merge keys are not unique in left dataset; not a one-to-many merge"
                )

        elif validate in ["many_to_one", "m:1"]:
            # 对于多对一合并，只需确保右侧的合并键是唯一的
            if not right_unique:
                raise MergeError(
                    "Merge keys are not unique in right dataset; "
                    "not a many-to-one merge"
                )

        elif validate in ["many_to_many", "m:m"]:
            # 对于多对多合并，无需检查合并键的唯一性，直接通过
            pass

        else:
            # 如果给定的合并方式参数不在预定义的列表中，抛出值错误
            raise ValueError(
                f'"{validate}" is not a valid argument. '
                "Valid arguments are:\n"
                '- "1:1"\n'
                '- "1:m"\n'
                '- "m:1"\n'
                '- "m:m"\n'
                '- "one_to_one"\n'
                '- "one_to_many"\n'
                '- "many_to_one"\n'
                '- "many_to_many"'
            )
# 定义函数 get_join_indexers，用于获取连接索引器
def get_join_indexers(
    left_keys: list[ArrayLike],
    right_keys: list[ArrayLike],
    sort: bool = False,
    how: JoinHow = "inner",
) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    """
    获取连接索引器。

    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
        左连接键的列表，可以是 ndarray、ExtensionArray、Index 或 Series 类型。
    right_keys : list[ndarray, ExtensionArray, Index, Series]
        右连接键的列表，可以是 ndarray、ExtensionArray、Index 或 Series 类型。
    sort : bool, default False
        是否对键进行排序。
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'
        连接的方式，包括 'inner'、'outer'、'left'、'right'。

    Returns
    -------
    np.ndarray[np.intp] or None
        左连接键的索引器。
    np.ndarray[np.intp] or None
        右连接键的索引器。
    """
    # 检查左右连接键的长度是否相同
    assert len(left_keys) == len(
        right_keys
    ), "left_keys and right_keys must be the same length"

    # 对于空的左/右键的快速路径处理
    left_n = len(left_keys[0])
    right_n = len(right_keys[0])
    if left_n == 0:
        if how in ["left", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["right", "outer"]:
            return _get_no_sort_one_missing_indexer(right_n, True)
    elif right_n == 0:
        if how in ["right", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["left", "outer"]:
            return _get_no_sort_one_missing_indexer(left_n, False)

    # 如果左连接键数量大于1，则获取左右连接标签和每个位置的级别数
    lkey: ArrayLike
    rkey: ArrayLike
    if len(left_keys) > 1:
        mapped = (
            _factorize_keys(left_keys[n], right_keys[n], sort=sort)
            for n in range(len(left_keys))
        )
        zipped = zip(*mapped)
        llab, rlab, shape = (list(x) for x in zipped)

        # 从标签列表获取平坦的 i8 键
        lkey, rkey = _get_join_keys(llab, rlab, tuple(shape), sort)
    else:
        lkey = left_keys[0]
        rkey = right_keys[0]

    # 创建左右 Index 对象
    left = Index(lkey)
    right = Index(rkey)

    # 如果左右索引均为单调递增且其中之一是唯一的，则执行快速路径连接
    if (
        left.is_monotonic_increasing
        and right.is_monotonic_increasing
        and (left.is_unique or right.is_unique)
    ):
        _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
    else:
        # 否则调用非唯一索引的连接处理函数
        lidx, ridx = get_join_indexers_non_unique(
            left._values, right._values, sort, how
        )

    # 如果 lidx 或 ridx 是范围索引器且与左或右的长度相同，则将其设为 None
    if lidx is not None and is_range_indexer(lidx, len(left)):
        lidx = None
    if ridx is not None and is_range_indexer(ridx, len(right)):
        ridx = None
    return lidx, ridx


# 定义函数 get_join_indexers_non_unique，用于获取非唯一索引的连接索引器
def get_join_indexers_non_unique(
    left: ArrayLike,
    right: ArrayLike,
    sort: bool = False,
    how: JoinHow = "inner",
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    获取非唯一索引的连接索引器。

    Parameters
    ----------
    left : ArrayLike
        左连接键。
    right : ArrayLike
        右连接键。
    sort : bool, default False
        是否对键进行排序。
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'
        连接的方式，包括 'inner'、'outer'、'left'、'right'。

    Returns
    -------
    np.ndarray[np.intp]
        左连接键的索引器。
    np.ndarray[np.intp]
        右连接键的索引器。
    """
    # 使用 _factorize_keys 函数对 left 和 right 进行因子化处理，获取左右键、计数值
    lkey, rkey, count = _factorize_keys(left, right, sort=sort, how=how)
    # 如果 count 为 -1，执行哈希连接，直接返回左键和右键
    if count == -1:
        return lkey, rkey
    # 根据连接方式 how 执行相应的连接操作
    if how == "left":
        # 左连接：调用 libjoin.left_outer_join 进行左外连接，获取左右索引
        lidx, ridx = libjoin.left_outer_join(lkey, rkey, count, sort=sort)
    elif how == "right":
        # 右连接：调用 libjoin.left_outer_join 进行左外连接，交换左右键位置，获取右左索引
        ridx, lidx = libjoin.left_outer_join(rkey, lkey, count, sort=sort)
    elif how == "inner":
        # 内连接：调用 libjoin.inner_join 进行内连接，获取左右索引
        lidx, ridx = libjoin.inner_join(lkey, rkey, count, sort=sort)
    elif how == "outer":
        # 外连接：调用 libjoin.full_outer_join 进行全外连接，获取左右索引
        lidx, ridx = libjoin.full_outer_join(lkey, rkey, count)
    # 返回左索引和右索引
    return lidx, ridx
def restore_dropped_levels_multijoin(
    left: MultiIndex,
    right: MultiIndex,
    dropped_level_names,
    join_index: Index,
    lindexer: npt.NDArray[np.intp],
    rindexer: npt.NDArray[np.intp],
) -> tuple[FrozenList, FrozenList, FrozenList]:
    """
    *this is an internal non-public method*

    Returns the levels, labels and names of a multi-index to multi-index join.
    Depending on the type of join, this method restores the appropriate
    dropped levels of the joined multi-index.
    The method relies on lindexer, rindexer which hold the index positions of
    left and right, where a join was feasible

    Parameters
    ----------
    left : MultiIndex
        left index
    right : MultiIndex
        right index
    dropped_level_names : str array
        list of non-common level names
    join_index : Index
        the index of the join between the
        common levels of left and right
    lindexer : np.ndarray[np.intp]
        left indexer
    rindexer : np.ndarray[np.intp]
        right indexer

    Returns
    -------
    levels : list of Index
        levels of combined multiindexes
    labels : np.ndarray[np.intp]
        labels of combined multiindexes
    names : List[Hashable]
        names of combined multiindex levels

    """

    def _convert_to_multiindex(index: Index) -> MultiIndex:
        if isinstance(index, MultiIndex):
            return index
        else:
            return MultiIndex.from_arrays([index._values], names=[index.name])

    # For multi-multi joins with one overlapping level,
    # the returned index if of type Index
    # Assure that join_index is of type MultiIndex
    # so that dropped levels can be appended
    join_index = _convert_to_multiindex(join_index)

    join_levels = join_index.levels
    join_codes = join_index.codes
    join_names = join_index.names

    # Iterate through the levels that must be restored
    for dropped_level_name in dropped_level_names:
        if dropped_level_name in left.names:
            idx = left
            indexer = lindexer
        else:
            idx = right
            indexer = rindexer

        # The index of the level name to be restored
        name_idx = idx.names.index(dropped_level_name)

        restore_levels = idx.levels[name_idx]
        # Inject -1 in the codes list where a join was not possible
        # IOW indexer[i]=-1
        codes = idx.codes[name_idx]
        if indexer is None:
            restore_codes = codes
        else:
            restore_codes = algos.take_nd(codes, indexer, fill_value=-1)

        # error: Cannot determine type of "__add__"
        join_levels = join_levels + [restore_levels]  # type: ignore[has-type]
        join_codes = join_codes + [restore_codes]  # type: ignore[has-type]
        join_names = join_names + [dropped_level_name]

    return join_levels, join_codes, join_names


class _OrderedMerge(_MergeOperation):
    _merge_type = "ordered_merge"
    # 初始化方法，用于初始化一个 MergeOperation 对象
    def __init__(
        self,
        left: DataFrame | Series,       # 左侧数据，可以是 DataFrame 或 Series
        right: DataFrame | Series,      # 右侧数据，可以是 DataFrame 或 Series
        on: IndexLabel | None = None,   # 连接的列或索引标签（默认为 None）
        left_on: IndexLabel | None = None,  # 左侧数据的连接列或索引标签（默认为 None）
        right_on: IndexLabel | None = None,  # 右侧数据的连接列或索引标签（默认为 None）
        left_index: bool = False,       # 是否使用左侧数据的索引作为连接键（默认为 False）
        right_index: bool = False,      # 是否使用右侧数据的索引作为连接键（默认为 False）
        suffixes: Suffixes = ("_x", "_y"),  # 连接重复列时的后缀（默认为 "_x" 和 "_y"）
        fill_method: str | None = None, # 填充缺失数据的方法（默认为 None）
        how: JoinHow | Literal["asof"] = "outer",  # 连接方式（默认为 "outer"）
    ) -> None:
        self.fill_method = fill_method  # 初始化填充缺失数据的方法
        _MergeOperation.__init__(       # 调用父类 _MergeOperation 的初始化方法
            self,
            left,
            right,
            on=on,
            left_on=left_on,
            left_index=left_index,
            right_index=right_index,
            right_on=right_on,
            how=how,
            suffixes=suffixes,
            sort=True,  # 设定排序以进行因子化操作
        )

    # 获取连接结果的方法，返回一个 DataFrame 对象
    def get_result(self) -> DataFrame:
        # 获取连接信息，包括连接的索引和索引器
        join_index, left_indexer, right_indexer = self._get_join_info()

        left_join_indexer: npt.NDArray[np.intp] | None
        right_join_indexer: npt.NDArray[np.intp] | None

        # 根据填充方法处理缺失数据的索引器
        if self.fill_method == "ffill":  # 如果填充方法为向前填充
            if left_indexer is None:
                left_join_indexer = None
            else:
                left_join_indexer = libjoin.ffill_indexer(left_indexer)  # 使用向前填充处理左侧索引器
            if right_indexer is None:
                right_join_indexer = None
            else:
                right_join_indexer = libjoin.ffill_indexer(right_indexer)  # 使用向前填充处理右侧索引器
        elif self.fill_method is None:  # 如果填充方法为 None
            left_join_indexer = left_indexer  # 直接使用左侧索引器
            right_join_indexer = right_indexer  # 直接使用右侧索引器
        else:
            raise ValueError("fill_method must be 'ffill' or None")  # 抛出异常，填充方法必须为 'ffill' 或 None

        # 重新索引和连接操作，返回连接结果
        result = self._reindex_and_concat(
            join_index, left_join_indexer, right_join_indexer
        )
        # 可能添加连接键到结果中
        self._maybe_add_join_keys(result, left_indexer, right_indexer)

        return result  # 返回连接后的结果 DataFrame
def _asof_by_function(direction: str):
    # 根据指定的方向生成对应的函数名
    name = f"asof_join_{direction}_on_X_by_Y"
    # 返回库中对应函数的引用，若不存在则返回 None
    return getattr(libjoin, name, None)


class _AsOfMerge(_OrderedMerge):
    _merge_type = "asof_merge"

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        by=None,
        left_by=None,
        right_by=None,
        suffixes: Suffixes = ("_x", "_y"),
        how: Literal["asof"] = "asof",
        tolerance=None,
        allow_exact_matches: bool = True,
        direction: str = "backward",
    ) -> None:
        # 初始化属性
        self.by = by
        self.left_by = left_by
        self.right_by = right_by
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.direction = direction

        # 检查 'direction' 是否为有效取值
        if self.direction not in ["backward", "forward", "nearest"]:
            raise MergeError(f"direction invalid: {self.direction}")

        # 验证 allow_exact_matches 是否为布尔值
        if not is_bool(self.allow_exact_matches):
            msg = (
                "allow_exact_matches must be boolean, "
                f"passed {self.allow_exact_matches}"
            )
            raise MergeError(msg)

        # 调用父类 _OrderedMerge 的构造方法
        _OrderedMerge.__init__(
            self,
            left,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
            fill_method=None,
        )

    def _maybe_require_matching_dtypes(
        self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]
    ):
        # 这个方法的具体实现将在后续添加，目前是一个占位符
        pass
    ) -> None:
        # TODO: why do we do this for AsOfMerge but not the others?

        def _check_dtype_match(left: ArrayLike, right: ArrayLike, i: int) -> None:
            # 检查左右数据类型是否匹配，如果不匹配则抛出合并错误异常
            if left.dtype != right.dtype:
                if isinstance(left.dtype, CategoricalDtype) and isinstance(
                    right.dtype, CategoricalDtype
                ):
                    # 对于分类数据类型，提供特定的错误信息
                    #
                    # 在这个函数中，连接键包括 merge_asof() 调用的原始键，以及传递给其 by= 参数的键。
                    # 对于前者，不支持无序但相等的类别，但会稍后由 ValueError 报告错误，因此我们在这里不需要检查它们。
                    msg = (
                        f"incompatible merge keys [{i}] {left.dtype!r} and "
                        f"{right.dtype!r}, both sides category, but not equal ones"
                    )
                else:
                    msg = (
                        f"incompatible merge keys [{i}] {left.dtype!r} and "
                        f"{right.dtype!r}, must be the same type"
                    )
                # 抛出合并错误异常，并附带详细的错误信息
                raise MergeError(msg)

        # 验证索引类型是否相同
        for i, (lk, rk) in enumerate(zip(left_join_keys, right_join_keys)):
            # 对每对左右连接键进行类型匹配检查
            _check_dtype_match(lk, rk, i)

        if self.left_index:
            # 如果使用左侧索引，则获取左侧索引的值
            lt = self.left.index._values
        else:
            # 否则，将左连接键的最后一个值作为 lt
            lt = left_join_keys[-1]

        if self.right_index:
            # 如果使用右侧索引，则获取右侧索引的值
            rt = self.right.index._values
        else:
            # 否则，将右连接键的最后一个值作为 rt
            rt = right_join_keys[-1]

        # 检查左右索引值的数据类型是否匹配
        _check_dtype_match(lt, rt, 0)
    # 验证容差值的有效性；如果容差值为 datetime.timedelta 或者 Timedelta（如果我们有 DTI）
    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
        # 如果容差值不为 None
        if self.tolerance is not None:
            # 如果使用左侧索引
            if self.left_index:
                # 获取左侧索引的值
                lt = self.left.index._values
            else:
                # 否则，使用左连接键的最后一个值作为 lt
                lt = left_join_keys[-1]

            # 构造错误消息
            msg = (
                f"incompatible tolerance {self.tolerance}, must be compat "
                f"with type {lt.dtype!r}"
            )

            # 处理 lt 的数据类型为整数并需要转换为 int64 或者是 ArrowExtensionArray 且数据类型为日期时间或时间戳的情况
            if needs_i8_conversion(lt.dtype) or (
                isinstance(lt, ArrowExtensionArray) and lt.dtype.kind in "mM"
            ):
                # 如果容差值不是 datetime.timedelta 类型，则抛出 MergeError 异常
                if not isinstance(self.tolerance, datetime.timedelta):
                    raise MergeError(msg)
                # 如果容差值小于 Timedelta(0)，则抛出 MergeError 异常
                if self.tolerance < Timedelta(0):
                    raise MergeError("tolerance must be positive")

            # 处理 lt 的数据类型为整数的情况
            elif is_integer_dtype(lt.dtype):
                # 如果容差值不是整数类型，则抛出 MergeError 异常
                if not is_integer(self.tolerance):
                    raise MergeError(msg)
                # 如果容差值小于 0，则抛出 MergeError 异常
                if self.tolerance < 0:
                    raise MergeError("tolerance must be positive")

            # 处理 lt 的数据类型为浮点数的情况
            elif is_float_dtype(lt.dtype):
                # 如果容差值不是数字类型，则抛出 MergeError 异常
                if not is_number(self.tolerance):
                    raise MergeError(msg)
                # 如果容差值小于 0，则抛出 MergeError 异常
                if self.tolerance < 0:  # type: ignore[operator]
                    raise MergeError("tolerance must be positive")

            # 处理 lt 的数据类型为其他情况，抛出 MergeError 异常
            else:
                raise MergeError("key must be integer, timestamp or float")

    # 为了进行库级联接操作，将值转换为 NumPy 数组
    def _convert_values_for_libjoin(
        self, values: AnyArrayLike, side: str
    ) -> np.ndarray:
        # 要求连接键值必须是有序且非空的
        if not Index(values).is_monotonic_increasing:
            # 如果值不是单调递增的或者包含空值，则抛出 ValueError 异常
            if isna(values).any():
                raise ValueError(f"Merge keys contain null values on {side} side")
            raise ValueError(f"{side} keys must be sorted")

        # 如果值是 ArrowExtensionArray 类型，则尝试将其转换为日期时间数组
        if isinstance(values, ArrowExtensionArray):
            values = values._maybe_convert_datelike_array()

        # 如果值的数据类型需要转换为 int64
        if needs_i8_conversion(values.dtype):
            values = values.view("i8")

        # 如果值是 BaseMaskedArray 类型，则将其数据部分赋给 values
        elif isinstance(values, BaseMaskedArray):
            # 在上面已经验证过不存在空值
            values = values._data

        # 如果值是 ExtensionArray 类型，则将其转换为 NumPy 数组
        elif isinstance(values, ExtensionArray):
            values = values.to_numpy()

        # 返回转换后的值作为 np.ndarray 类型
        return values  # type: ignore[return-value]
```python`
def _get_multiindex_indexer(
    join_keys: list[ArrayLike], index: MultiIndex, sort: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    # 对给定的 join_keys 和 index 进行 factorize 操作，生成左侧和右侧的标签映射
    mapped = (
        _factorize_keys(index.levels[n]._values, join_keys[n], sort=sort)
        for n in range(index.nlevels)
    )
    # 将映射的结果进行解压，生成行列索引和形状
    zipped = zip(*mapped)
    rcodes, lcodes, shape = (list(x) for x in zipped)
    # 如果需要排序，则根据 index.codes 对 rcodes 进行取值操作
    if sort:
        rcodes = list(map(np.take, rcodes, index.codes))
    else:
        i8copy = lambda a: a.astype("i8", subok=False)
        rcodes = list(map(i8copy, index.codes))

    # 修正右侧标签，处理 null 值的情况
    for i, join_key in enumerate(join_keys):
        mask = index.codes[i] == -1
        if mask.any():
            # 检查当前位置是否已经有 null 值，如果有，将其 factorized 为 shape[i] - 1
            a = join_key[lcodes[i] == shape[i] - 1]
            if a.size == 0 or not a[0] != a[0]:
                shape[i] += 1

            # 将 null 值的位置赋值为 shape[i] - 1
            rcodes[i][mask] = shape[i] - 1

    # 获取平坦的 i8 join keys
    lkey, rkey = _get_join_keys(lcodes, rcodes, tuple(shape), sort)
    return lkey, rkey


def _get_empty_indexer() -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """返回空的 join 索引器。"""
    return (
        np.array([], dtype=np.intp),
        np.array([], dtype=np.intp),
    )


def _get_no_sort_one_missing_indexer(
    n: int, left_missing: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    返回没有排序的情况下，所有一侧被选中，另一侧没有被选中的 join 索引器。

    参数
    ----------
    n : int
        创建索引器的长度。
    left_missing : bool
        如果为 True，左侧索引器将只包含 -1。
        如果为 False，右侧索引器将只包含 -1。

    返回
    -------
    np.ndarray[np.intp]
        左侧索引器
    np.ndarray[np.intp]
        右侧索引器
    """
    idx = np.arange(n, dtype=np.intp)
    idx_missing = np.full(shape=n, fill_value=-1, dtype=np.intp)
    if left_missing:
        return idx_missing, idx
    return idx, idx_missing


def _left_join_on_index(
    left_ax: Index, right_ax: Index, join_keys: list[ArrayLike], sort: bool = False
) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp]]:
    # 判断 right_ax 是否为 MultiIndex 类型
    if isinstance(right_ax, MultiIndex):
        lkey, rkey = _get_multiindex_indexer(join_keys, right_ax, sort=sort)
    else:
        # 错误：类型不兼容的赋值
        lkey = join_keys[0]  # 类型忽略
        # 错误：类型不兼容的赋值
        rkey = right_ax._values  # 类型忽略
    # 使用 _factorize_keys 函数从 lkey 和 rkey 中分解左右键和计数
    left_key, right_key, count = _factorize_keys(lkey, rkey, sort=sort)
    # 使用 libjoin.left_outer_join 函数执行左外连接操作，得到左右索引器
    left_indexer, right_indexer = libjoin.left_outer_join(
        left_key, right_key, count, sort=sort
    )

    # 如果需要排序或左索引器长度不等于左索引长度
    if sort or len(left_ax) != len(left_indexer):
        # 获取按照 left_indexer 重新排序的 left_ax 索引
        join_index = left_ax.take(left_indexer)
        # 返回连接后的索引、左索引器和右索引器
        return join_index, left_indexer, right_indexer

    # 如果不需要排序且左索引长度与 left_ax 相同，则保持左帧的顺序和长度
    return left_ax, None, right_indexer
def _factorize_keys(
    lk: ArrayLike,
    rk: ArrayLike,
    sort: bool = True,
    how: str | None = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    """
    Encode left and right keys as enumerated types.

    This is used to get the join indexers to be used when merging DataFrames.

    Parameters
    ----------
    lk : ndarray, ExtensionArray
        Left key.
    rk : ndarray, ExtensionArray
        Right key.
    sort : bool, defaults to True
        If True, the encoding is done such that the unique elements in the
        keys are sorted.
    how: str, optional
        Used to determine if we can use hash-join. If not given, then just factorize
        keys.

    Returns
    -------
    np.ndarray[np.intp]
        Left (resp. right if called with `key='right'`) labels, as enumerated type.
    np.ndarray[np.intp]
        Right (resp. left if called with `key='right'`) labels, as enumerated type.
    int
        Number of unique elements in union of left and right labels. -1 if we used
        a hash-join.

    See Also
    --------
    merge : Merge DataFrame or named Series objects
        with a database-style join.
    algorithms.factorize : Encode the object as an enumerated type
        or categorical variable.

    Examples
    --------
    >>> lk = np.array(["a", "c", "b"])
    >>> rk = np.array(["a", "c"])

    Here, the unique values are `'a', 'b', 'c'`. With the default
    `sort=True`, the encoding will be `{0: 'a', 1: 'b', 2: 'c'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk)
    (array([0, 2, 1]), array([0, 2]), 3)

    With the `sort=False`, the encoding will correspond to the order
    in which the unique elements first appear: `{0: 'a', 1: 'c', 2: 'b'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk, sort=False)
    (array([0, 1, 2]), array([0, 1]), 3)
    """
    # 如果左右键中有任一为 RangeIndex 类型，可能可以更有效地进行因子化处理？
    if (
        isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype)
    ) or (lib.is_np_dtype(lk.dtype, "M") and lib.is_np_dtype(rk.dtype, "M")):
        # 提取 ndarray（UTC 本地化）的值
        # 注意：我们不需要 dtypes 匹配，因为仍然可以进行比较
        lk, rk = cast("DatetimeArray", lk)._ensure_matching_resos(rk)
        lk = cast("DatetimeArray", lk)._ndarray
        rk = cast("DatetimeArray", rk)._ndarray

    elif (
        isinstance(lk.dtype, CategoricalDtype)
        and isinstance(rk.dtype, CategoricalDtype)
        and lk.dtype == rk.dtype
    ):
        assert isinstance(lk, Categorical)
        assert isinstance(rk, Categorical)
        # 将 rk 转换为编码，以便可以与 lk 的代码进行比较
        rk = lk._encode_with_my_categories(rk)

        lk = ensure_int64(lk.codes)
        rk = ensure_int64(rk.codes)
    # 如果左边（lk）是 ExtensionArray 类型，并且与右边（rk）的 dtype 相同
    elif isinstance(lk, ExtensionArray) and lk.dtype == rk.dtype:
        # 如果左边的 dtype 是 ArrowDtype 并且是字符串类型，或者是 StringDtype 并且存储在 ["pyarrow", "pyarrow_numpy"] 中
        if (isinstance(lk.dtype, ArrowDtype) and is_string_dtype(lk.dtype)) or (
            isinstance(lk.dtype, StringDtype)
            and lk.dtype.storage in ["pyarrow", "pyarrow_numpy"]
        ):
            # 导入必要的 pyarrow 模块
            import pyarrow as pa
            import pyarrow.compute as pc

            # 获取左边数组的长度
            len_lk = len(lk)
            # 将 lk 和 rk 转换为 PyArrow 数组
            lk = lk._pa_array  # type: ignore[attr-defined]
            rk = rk._pa_array  # type: ignore[union-attr]

            # 将左右两个数组合并为一个 chunked array，然后进行字典编码
            dc = (
                pa.chunked_array(lk.chunks + rk.chunks)  # type: ignore[union-attr]
                .combine_chunks()
                .dictionary_encode()
            )

            # 从字典编码的 indices 中提取左标签（llab）、右标签（rlab）和字典的长度（count）
            llab, rlab, count = (
                pc.fill_null(dc.indices[slice(len_lk)], -1)
                .to_numpy()
                .astype(np.intp, copy=False),
                pc.fill_null(dc.indices[slice(len_lk, None)], -1)
                .to_numpy()
                .astype(np.intp, copy=False),
                len(dc.dictionary),
            )

            # 如果需要对结果进行排序
            if sort:
                uniques = dc.dictionary.to_numpy(zero_copy_only=False)
                llab, rlab = _sort_labels(uniques, llab, rlab)

            # 如果存在空值
            if dc.null_count > 0:
                lmask = llab == -1
                lany = lmask.any()
                rmask = rlab == -1
                rany = rmask.any()
                # 如果左边有空值，用 count 替换空标签
                if lany:
                    np.putmask(llab, lmask, count)
                # 如果右边有空值，用 count 替换空标签
                if rany:
                    np.putmask(rlab, rmask, count)
                count += 1
            # 返回左标签、右标签和字典长度
            return llab, rlab, count

        # 如果左边不是 BaseMaskedArray，并且不是数值类型或字符串类型（除非需要排序）
        if not isinstance(lk, BaseMaskedArray) and not (
            isinstance(lk.dtype, ArrowDtype)
            and (
                is_numeric_dtype(lk.dtype.numpy_dtype)
                or is_string_dtype(lk.dtype)
                and not sort
            )
        ):
            # 对左右数组执行 _values_for_factorize 方法
            lk, _ = lk._values_for_factorize()

            # 错误：Item "ndarray" of "Union[Any, ndarray]" has no attribute "_values_for_factorize"
            rk, _ = rk._values_for_factorize()  # type: ignore[union-attr]

    # 如果需要将 lk.dtype 转换为 int64，并且与 rk.dtype 相同
    if needs_i8_conversion(lk.dtype) and lk.dtype == rk.dtype:
        # GH#23917 TODO: 需要针对不匹配的 dtypes 编写测试
        # GH#23917 TODO: 需要测试 lk 是整数类型，而 rk 是日期时间类型的情况
        # 将 lk 和 rk 转换为 int64 类型的 numpy 数组
        lk = np.asarray(lk, dtype=np.int64)
        rk = np.asarray(rk, dtype=np.int64)

    # 调用 _convert_arrays_and_get_rizer_klass 函数，将 lk 和 rk 转换为适当的数组，并获取 rizer 类
    klass, lk, rk = _convert_arrays_and_get_rizer_klass(lk, rk)

    # 创建一个 rizer 对象，用于处理长度为 lk 和 rk 中的较大值的情况
    # 使用_mask 标志表示 rk 是否是 BaseMaskedArray 或 ArrowExtensionArray 的实例
    rizer = klass(
        max(len(lk), len(rk)),
        uses_mask=isinstance(rk, (BaseMaskedArray, ArrowExtensionArray)),
    )

    # 如果 lk 是 BaseMaskedArray 的实例
    if isinstance(lk, BaseMaskedArray):
        # 确保 rk 也是 BaseMaskedArray 的实例
        assert isinstance(rk, BaseMaskedArray)
        # 获取 lk 和 rk 的数据和掩码
        lk_data, lk_mask = lk._data, lk._mask
        rk_data, rk_mask = rk._data, rk._mask
    elif isinstance(lk, ArrowExtensionArray):
        # 检查lk是否为ArrowExtensionArray类型
        assert isinstance(rk, ArrowExtensionArray)
        # 如果rk也是ArrowExtensionArray类型，断言通过
        # 我们只能在这里处理数值类型的数据
        # TODO: 当我们为Arrow创建Factorizer时，删除此部分
        lk_data = lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype)
        # 将lk转换为NumPy数组，使用1作为缺失值，数据类型与lk的NumPy数据类型一致
        rk_data = rk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype)
        # 将rk转换为NumPy数组，使用1作为缺失值，数据类型与lk的NumPy数据类型一致
        lk_mask, rk_mask = lk.isna(), rk.isna()
        # 获取lk和rk的缺失值掩码
    else:
        # 参数1给"factorize"的"ObjectFactorizer"具有不兼容的类型
        # "Union[ndarray[Any, dtype[signedinteger[_64Bit]]],
        # ndarray[Any, dtype[object_]]]"; 期望类型为"ndarray[Any, dtype[object_]]"
        lk_data, rk_data = lk, rk  # type: ignore[assignment]
        # 如果不是ArrowExtensionArray类型，则直接使用lk和rk作为数据
        lk_mask, rk_mask = None, None
        # 没有缺失值掩码

    hash_join_available = how == "inner" and not sort and lk.dtype.kind in "iufb"
    # 判断是否可以使用哈希连接：inner连接、不排序、lk的数据类型是整数、无符号整数、浮点数或布尔型
    if hash_join_available:
        rlab = rizer.factorize(rk_data, mask=rk_mask)
        # 使用rizer对rk_data进行因子化，使用rk_mask作为缺失值掩码
        if rizer.get_count() == len(rlab):
            # 如果rizer中的唯一值数量与rlab的长度相同
            ridx, lidx = rizer.hash_inner_join(lk_data, lk_mask)
            # 执行哈希内连接，获取左表和右表的索引
            return lidx, ridx, -1
            # 返回左表索引、右表索引和-1（表示没有新的组）
        else:
            llab = rizer.factorize(lk_data, mask=lk_mask)
            # 使用rizer对lk_data进行因子化，使用lk_mask作为缺失值掩码
    else:
        llab = rizer.factorize(lk_data, mask=lk_mask)
        # 使用rizer对lk_data进行因子化，使用lk_mask作为缺失值掩码
        rlab = rizer.factorize(rk_data, mask=rk_mask)
        # 使用rizer对rk_data进行因子化，使用rk_mask作为缺失值掩码

    assert llab.dtype == np.dtype(np.intp), llab.dtype
    # 断言llab的数据类型为np.intp（NumPy整数类型）
    assert rlab.dtype == np.dtype(np.intp), rlab.dtype
    # 断言rlab的数据类型为np.intp（NumPy整数类型）

    count = rizer.get_count()
    # 获取rizer中唯一值的数量

    if sort:
        uniques = rizer.uniques.to_array()
        # 如果需要排序，将rizer中的唯一值转换为NumPy数组
        llab, rlab = _sort_labels(uniques, llab, rlab)
        # 对llab和rlab进行排序

    # NA组
    lmask = llab == -1
    # 获取llab中值为-1的掩码
    lany = lmask.any()
    # 检查是否存在llab中的任意值为True
    rmask = rlab == -1
    # 获取rlab中值为-1的掩码
    rany = rmask.any()
    # 检查是否存在rlab中的任意值为True

    if lany or rany:
        # 如果存在NA值组
        if lany:
            np.putmask(llab, lmask, count)
            # 将llab中值为-1的位置替换为count
        if rany:
            np.putmask(rlab, rmask, count)
            # 将rlab中值为-1的位置替换为count
        count += 1
        # count加1，表示增加一个新组

    return llab, rlab, count
    # 返回llab（左表标签）、rlab（右表标签）和count（组的数量）
def _convert_arrays_and_get_rizer_klass(
    lk: ArrayLike, rk: ArrayLike
) -> tuple[type[libhashtable.Factorizer], ArrayLike, ArrayLike]:
    klass: type[libhashtable.Factorizer]
    # 检查左边数组是否是数值类型
    if is_numeric_dtype(lk.dtype):
        # 如果左右数组的数据类型不同，找到它们的共同类型
        if lk.dtype != rk.dtype:
            dtype = find_common_type([lk.dtype, rk.dtype])
            # 如果共同类型是扩展类型
            if isinstance(dtype, ExtensionDtype):
                # 根据扩展类型创建数组类
                cls = dtype.construct_array_type()
                # 如果左数组不是扩展数组，将其转换为指定类型的数组
                if not isinstance(lk, ExtensionArray):
                    lk = cls._from_sequence(lk, dtype=dtype, copy=False)
                else:
                    lk = lk.astype(dtype, copy=False)

                # 如果右数组不是扩展数组，将其转换为指定类型的数组
                if not isinstance(rk, ExtensionArray):
                    rk = cls._from_sequence(rk, dtype=dtype, copy=False)
                else:
                    rk = rk.astype(dtype, copy=False)
            else:
                # 将左右数组转换为共同的数据类型
                lk = lk.astype(dtype, copy=False)
                rk = rk.astype(dtype, copy=False)
        
        # 如果左数组是 BaseMaskedArray 类型
        if isinstance(lk, BaseMaskedArray):
            # 根据左数组的数据类型选择对应的 Factorizer 类型
            klass = _factorizers[lk.dtype.type]  # type: ignore[index]
        elif isinstance(lk.dtype, ArrowDtype):
            # 根据 ArrowDtype 的 numpy 数据类型选择对应的 Factorizer 类型
            klass = _factorizers[lk.dtype.numpy_dtype.type]
        else:
            # 根据左数组的数据类型选择对应的 Factorizer 类型
            klass = _factorizers[lk.dtype.type]

    else:
        # 如果左数组不是数值类型，则选择默认的 ObjectFactorizer 类型
        klass = libhashtable.ObjectFactorizer
        # 确保左右数组是对象类型
        lk = ensure_object(lk)
        rk = ensure_object(rk)
    
    # 返回 Factorizer 类型及转换后的左右数组
    return klass, lk, rk


def _sort_labels(
    uniques: np.ndarray, left: npt.NDArray[np.intp], right: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    # 计算左边数组的长度
    llength = len(left)
    # 将左右数组连接起来
    labels = np.concatenate([left, right])

    # 对唯一值和连接后的标签数组进行安全排序，使用 NA 哨兵值
    _, new_labels = algos.safe_sort(uniques, labels, use_na_sentinel=True)
    # 将排序后的标签数组分割为新的左右数组
    new_left, new_right = new_labels[:llength], new_labels[llength:]

    # 返回新的左右数组
    return new_left, new_right


def _get_join_keys(
    llab: list[npt.NDArray[np.int64 | np.intp]],
    rlab: list[npt.NDArray[np.int64 | np.intp]],
    shape: Shape,
    sort: bool,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    # 计算可以处理而不会发生溢出的级数
    nlev = next(
        lev
        for lev in range(len(shape), 0, -1)
        if not is_int64_overflow_possible(shape[:lev])
    )

    # 获取前 `nlev` 级别的连接键
    stride = np.prod(shape[1:nlev], dtype="i8")
    lkey = stride * llab[0].astype("i8", subok=False, copy=False)
    rkey = stride * rlab[0].astype("i8", subok=False, copy=False)

    # 逐级计算连接键
    for i in range(1, nlev):
        with np.errstate(divide="ignore"):
            stride //= shape[i]
        lkey += llab[i] * stride
        rkey += rlab[i] * stride

    # 如果已经处理完所有级别，则直接返回左右连接键
    if nlev == len(shape):
        return lkey, rkey

    # 否则，将当前键稀疏化以避免溢出
    lkey, rkey, count = _factorize_keys(lkey, rkey, sort=sort)

    # 更新左右标签列表和形状
    llab = [lkey] + llab[nlev:]
    rlab = [rkey] + rlab[nlev:]
    shape = (count,) + shape[nlev:]

    # 返回更新后的左右连接键
    return lkey, rkey
    # 调用函数 _get_join_keys，并返回其结果
    return _get_join_keys(llab, rlab, shape, sort)
# 判断是否应该填充的私有函数，用于比较两个名称是否相同，返回布尔值
def _should_fill(lname, rname) -> bool:
    if not isinstance(lname, str) or not isinstance(rname, str):
        return True  # 如果其中一个参数不是字符串类型，则应该进行填充
    return lname == rname  # 如果两个名称相同，则不需要填充


# 判断给定参数是否不为 None 的私有函数，返回布尔值
def _any(x) -> bool:
    return x is not None and com.any_not_none(*x)


# 验证操作数类型的私有函数，接受 DataFrame 或 Series 对象并返回 DataFrame
def _validate_operand(obj: DataFrame | Series) -> DataFrame:
    if isinstance(obj, ABCDataFrame):
        return obj  # 如果是 DataFrame 则直接返回
    elif isinstance(obj, ABCSeries):
        if obj.name is None:
            raise ValueError("Cannot merge a Series without a name")  # 如果 Series 没有名称则引发错误
        return obj.to_frame()  # 将 Series 转换为 DataFrame 并返回
    else:
        raise TypeError(
            f"Can only merge Series or DataFrame objects, a {type(obj)} was passed"
        )  # 如果参数不是 Series 或 DataFrame 则引发类型错误


# 处理索引重叠情况的私有函数，返回经过重命名处理的两个索引
def _items_overlap_with_suffix(
    left: Index, right: Index, suffixes: Suffixes
) -> tuple[Index, Index]:
    """
    Suffixes type validation.

    If two indices overlap, add suffixes to overlapping entries.

    If corresponding suffix is empty, the entry is simply converted to string.

    """
    if not is_list_like(suffixes, allow_sets=False) or isinstance(suffixes, dict):
        raise TypeError(
            f"Passing 'suffixes' as a {type(suffixes)}, is not supported. "
            "Provide 'suffixes' as a tuple instead."
        )  # 如果 suffixes 不是列表样式或者是字典则引发类型错误

    to_rename = left.intersection(right)  # 找到左右索引的交集

    if len(to_rename) == 0:
        return left, right  # 如果没有重叠的列名，则直接返回原索引

    lsuffix, rsuffix = suffixes  # 解包后缀元组

    if not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {to_rename}")  # 如果存在重叠但未指定后缀则引发值错误

    def renamer(x, suffix: str | None):
        """
        Rename the left and right indices.

        If there is overlap, and suffix is not None, add
        suffix, otherwise, leave it as-is.

        Parameters
        ----------
        x : original column name
        suffix : str or None

        Returns
        -------
        x : renamed column name
        """
        if x in to_rename and suffix is not None:
            return f"{x}{suffix}"  # 如果存在重叠且后缀不为空，则添加后缀
        return x  # 否则返回原名称

    lrenamer = partial(renamer, suffix=lsuffix)  # 对左索引进行重命名函数的偏函数化
    rrenamer = partial(renamer, suffix=rsuffix)  # 对右索引进行重命名函数的偏函数化

    llabels = left._transform_index(lrenamer)  # 通过左索引的重命名函数进行索引转换
    rlabels = right._transform_index(rrenamer)  # 通过右索引的重命名函数进行索引转换

    dups = []
    if not llabels.is_unique:
        # 仅在由于后缀导致重复时发出警告，原始重复列名不应发出警告
        dups = llabels[(llabels.duplicated()) & (~left.duplicated())].tolist()
    if not rlabels.is_unique:
        dups.extend(rlabels[(rlabels.duplicated()) & (~right.duplicated())].tolist())
    if dups:
        raise MergeError(
            f"Passing 'suffixes' which cause duplicate columns {set(dups)} is "
            f"not allowed.",
        )  # 如果存在重复列名则引发合并错误

    return llabels, rlabels  # 返回处理后的左右索引
```