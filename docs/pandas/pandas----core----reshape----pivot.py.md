# `D:\src\scipysrc\pandas\pandas\core\reshape\pivot.py`

```
from __future__ import annotations
# 导入将来版本支持的类型注解功能

import itertools
# 导入用于迭代工具函数的模块

from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
# 导入类型提示相关的模块和函数

import numpy as np
# 导入 NumPy 库，并使用 np 别名

from pandas._libs import lib
# 导入 Pandas 私有 C 库

from pandas.core.dtypes.cast import maybe_downcast_to_dtype
# 导入数据类型转换的函数 maybe_downcast_to_dtype

from pandas.core.dtypes.common import (
    is_list_like,
    is_nested_list_like,
    is_scalar,
)
# 导入用于检查数据类型的函数

from pandas.core.dtypes.dtypes import ExtensionDtype
# 导入 Pandas 扩展数据类型

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
# 导入 Pandas 泛型 DataFrame 和 Series 类

import pandas.core.common as com
# 导入 Pandas 核心通用模块，并使用 com 别名

from pandas.core.groupby import Grouper
# 导入 Pandas 分组器模块 Grouper

from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    get_objs_combined_axis,
)
# 导入 Pandas 索引相关的 API 函数和类

from pandas.core.reshape.concat import concat
# 导入 Pandas 数据合并函数 concat

from pandas.core.reshape.util import cartesian_product
# 导入 Pandas 重塑工具函数 cartesian_product

from pandas.core.series import Series
# 导入 Pandas Series 类

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
    )
    # 如果是类型检查阶段，导入 collections.abc 模块中的 Callable 和 Hashable 类

    from pandas._typing import (
        AggFuncType,
        AggFuncTypeBase,
        AggFuncTypeDict,
        IndexLabel,
        SequenceNotStr,
    )
    # 导入 Pandas 私有 _typing 模块中的类型别名

    from pandas import DataFrame
    # 导入 Pandas 中的 DataFrame 类型

def pivot_table(
    data: DataFrame,
    values=None,
    index=None,
    columns=None,
    aggfunc: AggFuncType = "mean",
    fill_value=None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool = True,
    sort: bool = True,
    **kwargs,
) -> DataFrame:
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    The levels in the pivot table will be stored in MultiIndex objects
    (hierarchical indexes) on the index and columns of the result DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame object.
    values : list-like or scalar, optional
        Column or columns to aggregate.
    index : column, Grouper, array, or list of the previous
        Keys to group by on the pivot table index. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.
    columns : column, Grouper, array, or list of the previous
        Keys to group by on the pivot table column. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.
    aggfunc : function, list of functions, dict, default "mean"
        If a list of functions is passed, the resulting pivot table will have
        hierarchical columns whose top level are the function names
        (inferred from the function objects themselves).
        If a dict is passed, the key is column to aggregate and the value is
        function or list of functions. If ``margins=True``, aggfunc will be
        used to calculate the partial aggregates.
    fill_value : scalar, default None
        Value to replace missing values with.
    margins : bool, default False
        Add all row / columns (e.g. for subtotal / grand totals).
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    margins_name : Hashable, default 'All'
        Name of the row / column that will contain the totals
        when margins is True.
    observed : bool, default True
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.
    sort : bool, default True
        Sort group keys. Get better performance by turning this off.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to func.

    Returns
    -------
    DataFrame
        A DataFrame containing the values of the pivot table.
    """
    # 实现一个类似电子表格样式的透视表，并将其存储为 DataFrame 对象

    # 省略函数体内部的具体实现
    fill_value : scalar, default None
        # 在生成的透视表中用来替换缺失值的值（在聚合之后）。

    margins : bool, default False
        # 如果 ``margins=True``，则在行和列上添加特殊的 ``All`` 列和行，
        # 这些列和行包含跨分类的部分组聚合。

    dropna : bool, default True
        # 是否排除所有条目都是 NaN 的列。如果为 True，
        # 在计算边距之前将省略任何列中存在 NaN 值的行。

    margins_name : str, default 'All'
        # 当 margins=True 时，包含总计的行/列的名称。

    observed : bool, default False
        # 只适用于任何分组者为分类变量的情况。
        # 如果为 True：仅显示分类分组器的观察值。
        # 如果为 False：显示分类分组器的所有值。
        #
        # .. versionchanged:: 3.0.0
        #     默认值现在为 ``True``。

    sort : bool, default True
        # 指定结果是否应该排序。

        # .. versionadded:: 1.3.0

    **kwargs : dict
        # 传递给 ``aggfunc`` 的可选关键字参数。

        # .. versionadded:: 3.0.0

    Returns
    -------
    DataFrame
        # Excel 风格的透视表。

    See Also
    --------
    DataFrame.pivot : 处理非数值数据的透视表，不进行聚合。
    DataFrame.melt : 将宽格式数据框变为长格式，可选择保留标识符。
    wide_to_long : 将宽面板数据转换为长格式。比 melt 不够灵活但更易用。

    Notes
    -----
    参考 :ref:`用户指南 <reshaping.pivot>` 获取更多示例。

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
    ...         "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
    ...         "C": [
    ...             "small",
    ...             "large",
    ...             "large",
    ...             "small",
    ...             "small",
    ...             "large",
    ...             "small",
    ...             "small",
    ...             "large",
    ...         ],
    ...         "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
    ...         "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
    ...     }
    ... )
    >>> df
         A    B      C  D  E
    0  foo  one  small  1  2
    1  foo  one  large  2  4
    2  foo  one  large  2  5
    3  foo  two  small  3  5
    4  foo  two  small  3  6
    5  bar  one  large  4  6
    6  bar  one  small  5  8
    7  bar  two  small  6  9
    8  bar  two  large  7  9

    This first example aggregates values by taking the sum.

    >>> table = pd.pivot_table(
    ...     df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum"
    ... )
    >>> table
    C        large  small
    A   B
    bar one    4.0    5.0
        two    7.0    6.0
    # 将索引参数转换为内部格式
    index = _convert_by(index)
    # 将列参数转换为内部格式
    columns = _convert_by(columns)
    
    # 如果聚合函数是列表形式
    if isinstance(aggfunc, list):
        # 初始化存储 DataFrame 片段和键的列表
        pieces: list[DataFrame] = []
        keys = []
        
        # 遍历每个聚合函数
        for func in aggfunc:
            # 调用内部的透视表函数，生成每个聚合函数对应的透视表
            _table = __internal_pivot_table(
                data,
                values=values,
                index=index,
                columns=columns,
                fill_value=fill_value,
                aggfunc=func,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
                kwargs=kwargs,
            )
            # 将生成的片段和函数名添加到对应列表中
            pieces.append(_table)
            keys.append(getattr(func, "__name__", func))
        
        # 合并所有片段，使用函数名作为键，沿着列轴连接
        table = concat(pieces, keys=keys, axis=1)
        # 返回最终的透视表结果，并使用原始数据来最终化结果
        return table.__finalize__(data, method="pivot_table")
    
    # 如果聚合函数不是列表形式
    # 调用内部的透视表函数，生成单个透视表
    table = __internal_pivot_table(
        data,
        values,
        index,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
        kwargs,
    )
    # 返回最终的透视表结果，并使用原始数据来最终化结果
    return table.__finalize__(data, method="pivot_table")
def __internal_pivot_table(
    data: DataFrame,
    values,
    index,
    columns,
    aggfunc: AggFuncTypeBase | AggFuncTypeDict,
    fill_value,
    margins: bool,
    dropna: bool,
    margins_name: Hashable,
    observed: bool,
    sort: bool,
    kwargs,
) -> DataFrame:
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
    # 将索引和列组合成键
    keys = index + columns

    # 检查是否传入了值
    values_passed = values is not None
    if values_passed:
        # 如果传入的值是一个列表
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            # 如果传入的值不是列表
            values_multi = False
            values = [values]

        # 确保值在数据中存在
        for i in values:
            if i not in data:
                raise KeyError(i)

        # 准备要筛选的列
        to_filter = []
        for x in keys + values:
            # 如果是 Grouper 对象，则获取其键值
            if isinstance(x, Grouper):
                x = x.key
            try:
                # 检查数据中是否存在此列
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        # 如果要筛选的列数小于数据的列数，则筛选数据
        if len(to_filter) < len(data.columns):
            data = data[to_filter]

    else:
        # 如果未传入值，则使用数据的全部列
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)

    # 按照键进行分组
    grouped = data.groupby(keys, observed=observed, sort=sort, dropna=dropna)
    # 应用聚合函数
    agged = grouped.agg(aggfunc, **kwargs)

    # 如果设置了 dropna 且聚合结果是 DataFrame 且列数不为零，则删除空值行
    if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):
        agged = agged.dropna(how="all")

    table = agged

    # 如果设置了 sort 为 True 并且聚合结果是 DataFrame，则按照列名排序
    if sort is True and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)

    # 如果表格的索引层级数大于 1 且索引不为空，则执行以下操作
    if table.index.nlevels > 1 and index:
        # 如果索引名是整数，则确定整数是指层级位置还是名称
        index_names = agged.index.names[: len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        # 使用给定的层级展开表格
        table = agged.unstack(to_unstack, fill_value=fill_value)

    # 如果未设置 dropna，则执行以下操作
    if not dropna:
        # 如果表格的索引是多层索引，则重建索引
        if isinstance(table.index, MultiIndex):
            m = MultiIndex.from_arrays(
                cartesian_product(table.index.levels), names=table.index.names
            )
            table = table.reindex(m, axis=0, fill_value=fill_value)

        # 如果表格的列是多层索引，则重建索引
        if isinstance(table.columns, MultiIndex):
            m = MultiIndex.from_arrays(
                cartesian_product(table.columns.levels), names=table.columns.names
            )
            table = table.reindex(m, axis=1, fill_value=fill_value)

    # 如果设置了 sort 为 True 并且聚合结果是 DataFrame，则按照列名排序
    if sort is True and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)

    # 返回处理后的表格
    return table
    # 如果 fill_value 不是 None，则使用 fillna 方法填充表格中的缺失值
    if fill_value is not None:
        table = table.fillna(fill_value)
        # 如果 aggfunc 是 len，并且 observed 是 False，并且 fill_value 是整数类型，
        # 则将 table 转换为 np.int64 类型。这是为了处理之前由 downcast="infer"
        # 在 fillna 中处理的情况。
        if aggfunc is len and not observed and lib.is_integer(fill_value):
            table = table.astype(np.int64)

    # 如果设置了 margins 标志，则添加边际汇总
    if margins:
        # 如果 dropna 是 True，则从数据中删除所有含有缺失值的行
        if dropna:
            data = data[data.notna().all(axis=1)]
        # 调用 _add_margins 函数来添加行和列的边际汇总
        table = _add_margins(
            table,
            data,
            values,
            rows=index,
            cols=columns,
            aggfunc=aggfunc,
            kwargs=kwargs,
            observed=dropna,
            margins_name=margins_name,
            fill_value=fill_value,
        )

    # 如果 values_passed 为 True，并且 values_multi 为 False，并且 table 的列级别大于 1，
    # 则丢弃顶级列级别，即去掉多级列索引的最外层级别。
    if values_passed and not values_multi and table.columns.nlevels > 1:
        table.columns = table.columns.droplevel(0)
    
    # 如果 index 的长度为 0，并且 columns 的长度大于 0，则对 table 进行转置操作。
    if len(index) == 0 and len(columns) > 0:
        table = table.T

    # 如果 table 是 ABCDataFrame 类型，并且 dropna 是 True，则删除所有列全部为缺失值的列。
    # 这是为了确保在 dropna=True 的情况下移除所有空列。
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how="all", axis=1)

    # 返回处理后的表格
    return table
def _add_margins(
    table: DataFrame | Series,
    data: DataFrame,
    values,
    rows,
    cols,
    aggfunc,
    kwargs,
    observed: bool,
    margins_name: Hashable = "All",
    fill_value=None,
):
    # 检查 margins_name 是否为字符串，若不是则抛出值错误异常
    if not isinstance(margins_name, str):
        raise ValueError("margins_name argument must be a string")

    # 检查在表格索引层级中是否存在与 margins_name 冲突的名称
    msg = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if margins_name in table.index.get_level_values(level):
            raise ValueError(msg)

    # 计算全局边界
    grand_margin = _compute_grand_margin(data, values, aggfunc, kwargs, margins_name)

    # 如果表格是二维的（即 DataFrame）
    if table.ndim == 2:
        # 检查在列索引层级中是否存在与 margins_name 冲突的名称
        for level in table.columns.names[1:]:
            if margins_name in table.columns.get_level_values(level):
                raise ValueError(msg)

    # 初始化 key 变量，用于存储边界名称的元组
    key: str | tuple[str, ...]
    if len(rows) > 1:
        key = (margins_name,) + ("",) * (len(rows) - 1)
    else:
        key = margins_name

    # 如果没有值并且表格是 Series 类型，则直接返回计算的全局边界
    if not values and isinstance(table, ABCSeries):
        return table._append(table._constructor({key: grand_margin[margins_name]}))

    # 如果有值，则生成边界结果集
    elif values:
        marginal_result_set = _generate_marginal_results(
            table,
            data,
            values,
            rows,
            cols,
            aggfunc,
            kwargs,
            observed,
            margins_name,
        )
        # 如果结果集不是元组，则直接返回结果集
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
    else:
        # 没有值，并且表格是 DataFrame 类型
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(
            table, data, rows, cols, aggfunc, kwargs, observed, margins_name
        )
        # 如果结果集不是元组，则直接返回结果集
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set

    # 重新索引行边界并填充缺失值
    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)

    # 填充全局边界
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]

    # 导入 DataFrame 类
    from pandas import DataFrame

    # 创建边界虚拟值 DataFrame
    margin_dummy = DataFrame(row_margin, columns=Index([key])).T

    # 保存结果索引名称
    row_names = result.index.names

    # 遍历结果的数据类型，对浮点数列进行检查
    for dtype in set(result.dtypes):
        if isinstance(dtype, ExtensionDtype):
            # 可以包含 NA
            continue

        # 选择特定数据类型的列
        cols = result.select_dtypes([dtype]).columns

        # 尝试对边界虚拟值 DataFrame 中的列进行数据类型转换
        margin_dummy[cols] = margin_dummy[cols].apply(
            maybe_downcast_to_dtype, args=(dtype,)
        )

    # 将边界虚拟值 DataFrame 追加到结果中
    result = result._append(margin_dummy)

    # 设置结果的索引名称
    result.index.names = row_names

    # 返回最终结果
    return result
    data: DataFrame, values, aggfunc, kwargs, margins_name: Hashable = "All"



# 定义一个函数或方法的参数列表，具体如下：
# - data: DataFrame，表示传入的参数是一个 DataFrame 对象
# - values: 表示传入的参数是一个普通变量，可能是用于聚合的数值列或其他数据
# - aggfunc: 表示传入的参数是一个函数或者字符串，用于指定聚合函数的方式
# - kwargs: 表示传入的参数是一个字典（关键字参数），用于传递额外的参数配置
# - margins_name: 表示传入的参数是一个可散列的对象（如字符串、数字等），默认值为字符串 "All"
# 如果提供了 values 参数，则计算边际汇总结果
def _generate_marginal_results(
    table,
    data: DataFrame,
    values,
    rows,
    cols,
    aggfunc,
    kwargs,
    observed: bool,
    margins_name: Hashable = "All",
):
    # 初始化一个空字典来存储汇总结果
    if values:
        grand_margin = {}
        # 遍历指定的数据集中 values 列的每个项目
        for k, v in data[values].items():
            try:
                # 根据 aggfunc 的类型，应用相应的聚合函数到值 v
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)(**kwargs)
                elif isinstance(aggfunc, dict):
                    # 如果 aggfunc 是字典，则根据键 k 确定使用的聚合函数，并将结果存储到 grand_margin 中
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])(**kwargs)
                    else:
                        grand_margin[k] = aggfunc[k](v, **kwargs)
                else:
                    # 否则，直接应用 aggfunc 到值 v，并将结果存储到 grand_margin 中
                    grand_margin[k] = aggfunc(v, **kwargs)
            except TypeError:
                pass
        # 返回计算得到的边际汇总结果
        return grand_margin
    else:
        # 如果未提供 values 参数，则对整个数据集进行聚合，并以 margins_name 命名
        return {margins_name: aggfunc(data.index, **kwargs)}
    # 如果列数大于0，执行以下操作
    if len(cols) > 0:
        # 需要"交错"边距
        table_pieces = []  # 存放表格片段的列表
        margin_keys = []  # 存放边距键的列表

        def _all_key(key):
            return (key, margins_name) + ("",) * (len(cols) - 1)  # 返回包含键和边距名称的元组

        # 如果行数大于0，执行以下操作
        if len(rows) > 0:
            # 计算边距，将其添加到数据中
            margin = (
                data[rows + values]
                .groupby(rows, observed=observed)
                .agg(aggfunc, **kwargs)
            )
            cat_axis = 1  # 类别轴设置为1

            # 按照第一级索引分组表格，并遍历每个组
            for key, piece in table.T.groupby(level=0, observed=observed):
                piece = piece.T  # 转置组件
                all_key = _all_key(key)  # 获取所有键

                piece[all_key] = margin[key]  # 将边距添加到组件中

                table_pieces.append(piece)  # 将组件添加到表格片段列表中
                margin_keys.append(all_key)  # 将键添加到边距键列表中
        else:
            from pandas import DataFrame

            cat_axis = 0  # 类别轴设置为0
            # 按第一级索引分组表格，并遍历每个组
            for key, piece in table.groupby(level=0, observed=observed):
                if len(cols) > 1:
                    all_key = _all_key(key)  # 获取所有键
                else:
                    all_key = margins_name  # 使用边距名称
                table_pieces.append(piece)  # 将组件添加到表格片段列表中
                # GH31016 用于计算每个组的边距，并将对应的键作为索引
                transformed_piece = DataFrame(piece.apply(aggfunc, **kwargs)).T
                if isinstance(piece.index, MultiIndex):
                    # 添加一个空级别
                    transformed_piece.index = MultiIndex.from_tuples(
                        [all_key],
                        names=piece.index.names
                        + [
                            None,
                        ],
                    )
                else:
                    transformed_piece.index = Index([all_key], name=piece.index.name)

                # 将边距的组件添加到表格片段中
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)  # 将键添加到边距键列表中

        # 如果表格片段为空
        if not table_pieces:
            # GH 49240 返回原表格
            return table
        else:
            # 连接所有表格片段
            result = concat(table_pieces, axis=cat_axis)

        # 如果行数为0，返回结果
        if len(rows) == 0:
            return result
    else:
        result = table  # 结果为原表格
        margin_keys = table.columns  # 边距键为表格的列名

    # 如果列数大于0，执行以下操作
    if len(cols) > 0:
        # 计算行边距，并将其堆叠
        row_margin = (
            data[cols + values].groupby(cols, observed=observed).agg(aggfunc, **kwargs)
        )
        row_margin = row_margin.stack()

        # GH#26568 使用名称而不是索引以避免数字名称的情况
        new_order_indices = itertools.chain([len(cols)], range(len(cols)))
        new_order_names = [row_margin.index.names[i] for i in new_order_indices]
        row_margin.index = row_margin.index.reorder_levels(new_order_names)
    else:
        row_margin = data._constructor_sliced(np.nan, index=result.columns)  # 构建NaN值行边距

    return result, margin_keys, row_margin  # 返回结果、边距键和行边距
def _generate_marginal_results_without_values(
    table: DataFrame,
    data: DataFrame,
    rows,
    cols,
    aggfunc,
    kwargs,
    observed: bool,
    margins_name: Hashable = "All",
):
    margin_keys: list | Index
    if len(cols) > 0:
        # 需要对边际进行 "交错" 处理
        margin_keys = []

        def _all_key():
            if len(cols) == 1:
                return margins_name
            return (margins_name,) + ("",) * (len(cols) - 1)

        if len(rows) > 0:
            # 根据行分组数据，应用聚合函数 aggfunc，并添加到 table 中
            margin = data.groupby(rows, observed=observed)[rows].apply(
                aggfunc, **kwargs
            )
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)

        else:
            # 根据第一级别分组数据，应用聚合函数 aggfunc，并添加到 table 中
            margin = data.groupby(level=0, observed=observed).apply(aggfunc, **kwargs)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        # 如果没有列，则直接返回 table，边际键为表的列名
        result = table
        margin_keys = table.columns

    if len(cols):
        # 根据列分组数据，应用聚合函数 aggfunc，并返回结果、边际键、行边际
        row_margin = data.groupby(cols, observed=observed)[cols].apply(
            aggfunc, **kwargs
        )
    else:
        # 如果没有列，则创建一个具有 NaN 值的 Series 作为行边际
        row_margin = Series(np.nan, index=result.columns)

    return result, margin_keys, row_margin


def _convert_by(by):
    if by is None:
        by = []
    elif (
        is_scalar(by)
        or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper))
        or callable(by)
    ):
        by = [by]
    else:
        by = list(by)
    return by


def pivot(
    data: DataFrame,
    *,
    columns: IndexLabel,
    index: IndexLabel | lib.NoDefault = lib.no_default,
    values: IndexLabel | lib.NoDefault = lib.no_default,
) -> DataFrame:
    """
    返回按给定索引/列值组织的重塑后的 DataFrame。

    基于列值重塑数据（生成“透视”表）。使用指定的 `index` / `columns` 的唯一值来形成结果 DataFrame 的轴。
    此函数不支持数据聚合，多个值将导致列中的 MultiIndex。参见 :ref:`User Guide <reshaping>` 了解更多有关重塑的信息。

    参数
    ----------
    data : DataFrame
        输入的 pandas DataFrame 对象。
    columns : str 或对象 或 str 列表
        用于创建新框架列的列。
    index : str 或对象 或 str 列表, 可选
        用于创建新框架索引的列。如果未提供，则使用现有索引。
    values : str, 对象 或 前述的 str 列表, 可选
        用于填充新框架值的列。如果未指定，则使用所有剩余列，并且结果将具有层次化索引的列。

    返回
    -------
    DataFrame
        返回重塑后的 DataFrame。

    异常
    ------

    """
    ValueError:
        当存在任何重复的 `index` 和 `columns` 组合时，会引发 `ValueError`。在需要聚合的情况下，可以使用 `DataFrame.pivot_table`。

    See Also
    --------
    DataFrame.pivot_table : 通用的 pivot 方法，可以处理一个 index/column 对的重复值。
    DataFrame.unstack : 基于索引值而不是列进行 pivot。
    wide_to_long : 将宽格式面板数据转换为长格式。比 melt 方法更易用但不够灵活。

    Notes
    -----
    如需更精细的控制，请参考层次化索引文档及相关的 stack/unstack 方法。

    Reference :ref:`the user guide <reshaping.pivot>` 了解更多示例。

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "foo": ["one", "one", "one", "two", "two", "two"],
    ...         "bar": ["A", "B", "C", "A", "B", "C"],
    ...         "baz": [1, 2, 3, 4, 5, 6],
    ...         "zoo": ["x", "y", "z", "q", "w", "t"],
    ...     }
    ... )
    >>> df
        foo   bar  baz  zoo
    0   one   A    1    x
    1   one   B    2    y
    2   one   C    3    z
    3   two   A    4    q
    4   two   B    5    w
    5   two   C    6    t

    >>> df.pivot(index="foo", columns="bar", values="baz")
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index="foo", columns="bar")["baz"]
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index="foo", columns="bar", values=["baz", "zoo"])
          baz       zoo
    bar   A  B  C   A  B  C
    foo
    one   1  2  3   x  y  z
    two   4  5  6   q  w  t

    您也可以分配一个列名列表或索引名列表。

    >>> df = pd.DataFrame(
    ...     {
    ...         "lev1": [1, 1, 1, 2, 2, 2],
    ...         "lev2": [1, 1, 2, 1, 1, 2],
    ...         "lev3": [1, 2, 1, 2, 1, 2],
    ...         "lev4": [1, 2, 3, 4, 5, 6],
    ...         "values": [0, 1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> df
        lev1 lev2 lev3 lev4 values
    0   1    1    1    1    0
    1   1    1    2    2    1
    2   1    2    1    3    2
    3   2    1    2    4    3
    4   2    1    1    5    4
    5   2    2    2    6    5

    >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
    lev2    1         2
    lev3    1    2    1    2
    lev1
    1     0.0  1.0  2.0  NaN
    2     4.0  3.0  NaN  5.0

    >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")
          lev3    1    2
    lev1  lev2
       1     1  0.0  1.0
             2  2.0  NaN
       2     1  4.0  3.0
             2  NaN  5.0

    如果存在任何重复，将会引发 ValueError。

    >>> df = pd.DataFrame(
    ...     {
    ...         "foo": ["one", "one", "two", "two"],
    ...         "bar": ["A", "A", "B", "C"],
    ...         "baz": [1, 2, 3, 4],
    ...     }
    ... )
    >>> df
       foo bar  baz
    0  one   A    1
    1  one   A    2
    2  two   B    3
    3  two   C    4

    Notice that the first two rows are the same for our `index`
    and `columns` arguments.

    >>> df.pivot(index="foo", columns="bar", values="baz")
    Traceback (most recent call last):
       ...
    ValueError: Index contains duplicate entries, cannot reshape
    """
    columns_listlike = com.convert_to_list_like(columns)
    # 将列名参数转换为列表形式，以便后续处理

    # If columns is None we will create a MultiIndex level with None as name
    # which might cause duplicated names because None is the default for
    # level names
    if any(name is None for name in data.index.names):
        # 检查索引名中是否存在 None，如果有，则进行处理以避免重复命名
        data = data.copy(deep=False)
        data.index.names = [
            name if name is not None else lib.no_default for name in data.index.names
        ]

    indexed: DataFrame | Series
    if values is lib.no_default:
        if index is not lib.no_default:
            cols = com.convert_to_list_like(index)
        else:
            cols = []

        append = index is lib.no_default
        # error: Unsupported operand types for + ("List[Any]" and "ExtensionArray")
        # error: Unsupported left operand type for + ("ExtensionArray")
        indexed = data.set_index(
            cols + columns_listlike,  # type: ignore[operator]
            append=append,
        )
    else:
        index_list: list[Index] | list[Series]
        if index is lib.no_default:
            if isinstance(data.index, MultiIndex):
                # GH 23955
                # 如果数据的索引是 MultiIndex，则提取每个级别的值
                index_list = [
                    data.index.get_level_values(i) for i in range(data.index.nlevels)
                ]
            else:
                # 否则，将整个索引作为单个级别处理
                index_list = [
                    data._constructor_sliced(data.index, name=data.index.name)
                ]
        else:
            # 如果提供了特定的索引参数，使用参数值作为新的索引级别
            index_list = [data[idx] for idx in com.convert_to_list_like(index)]

        data_columns = [data[col] for col in columns_listlike]
        index_list.extend(data_columns)
        multiindex = MultiIndex.from_arrays(index_list)

        if is_list_like(values) and not isinstance(values, tuple):
            # Exclude tuple because it is seen as a single column name
            # 处理多个数值列作为值的情况，构建新的 DataFrame
            indexed = data._constructor(
                data[values]._values,
                index=multiindex,
                columns=cast("SequenceNotStr", values),
            )
        else:
            # 处理单个数值列作为值的情况，构建新的 Series
            indexed = data._constructor_sliced(data[values]._values, index=multiindex)

    # error: Argument 1 to "unstack" of "DataFrame" has incompatible type "Union
    # [List[Any], ExtensionArray, ndarray[Any, Any], Index, Series]"; expected
    # "Hashable"
    # unstack with a MultiIndex returns a DataFrame
    # 对多级索引的 DataFrame 进行 unstack 操作，返回一个新的 DataFrame
    result = cast("DataFrame", indexed.unstack(columns_listlike))  # type: ignore[arg-type]

    # 设置结果 DataFrame 的索引名，处理未命名的索引级别
    result.index.names = [
        name if name is not lib.no_default else None for name in result.index.names
    ]

    return result
# 定义一个函数用于计算两个或多个因子的交叉制表结果。

By default, computes a frequency table of the factors unless an
array of values and an aggregation function are passed.

# 默认情况下，计算因子的频率表，除非传入值数组和聚合函数。

Parameters
----------
index : array-like, Series, or list of arrays/Series
    # 指定行中要分组的值。
columns : array-like, Series, or list of arrays/Series
    # 指定列中要分组的值。
values : array-like, optional
    # 要根据因子聚合的数值数组。
    # 需要指定 `aggfunc`。
rownames : sequence, default None
    # 如果传递，必须与传递的行数组数量匹配。
colnames : sequence, default None
    # 如果传递，必须与传递的列数组数量匹配。
aggfunc : function, optional
    # 如果指定，需要同时指定 `values`。
margins : bool, default False
    # 添加行/列边际（小计）。
margins_name : str, default 'All'
    # 当 `margins` 为 True 时，用于包含总计的行/列的名称。
dropna : bool, default True
    # 不包括所有条目都为 NaN 的列。
normalize : bool, {'all', 'index', 'columns'}, or {0,1}, default False
    # 根据总和将所有值进行归一化。
    # - 如果传入 'all' 或 `True`，将对所有值进行归一化。
    # - 如果传入 'index'，将对每行进行归一化。
    # - 如果传入 'columns'，将对每列进行归一化。
    # - 如果 `margins` 为 `True`，还将归一化边际值。

Returns
-------
DataFrame
    # 数据的交叉制表结果。

See Also
--------
DataFrame.pivot : 根据列值重塑数据。
pivot_table : 创建一个数据透视表作为 DataFrame。

Notes
-----
Any Series passed will have their name attributes used unless row or column
names for the cross-tabulation are specified.

# 除非指定了交叉制表的行或列名称，否则将使用传递的任何 Series 的名称属性。

Any input passed containing Categorical data will have **all** of its
categories included in the cross-tabulation, even if the actual data does
not contain any instances of a particular category.

# 传递包含分类数据的任何输入都将包含其所有类别在交叉制表中，
即使实际数据不包含特定类别的任何实例。

In the event that there aren't overlapping indexes an empty DataFrame will
be returned.

# 如果没有重叠的索引，将返回一个空的 DataFrame。

Reference :ref:`the user guide <reshaping.crosstabulations>` for more examples.

# 更多示例请参阅用户指南中的 `crosstabulations` 部分。

Examples
--------
>>> a = np.array(
...     [
...         "foo",
...         "foo",
...         "foo",
...         "foo",
...         "bar",
...         "bar",
...         "bar",
...         "bar",
...         "foo",
...         "foo",
...         "foo",
...     ],
...     dtype=object,
... )
>>> b = np.array(
    """
    >>> a = np.array(
    ...     [
    ...         "one",   # 创建一个包含多个重复项的 NumPy 数组 'a'
    ...         "one",
    ...         "one",
    ...         "two",
    ...         "one",
    ...         "one",
    ...         "one",
    ...         "two",
    ...         "two",
    ...         "two",
    ...         "one",
    ...     ],
    ...     dtype=object,   # 指定数组的数据类型为对象类型
    ... )
    >>> c = np.array(
    ...     [
    ...         "dull",   # 创建一个包含多个重复项的 NumPy 数组 'c'
    ...         "dull",
    ...         "shiny",
    ...         "dull",
    ...         "dull",
    ...         "shiny",
    ...         "shiny",
    ...         "dull",
    ...         "shiny",
    ...         "shiny",
    ...         "shiny",
    ...     ],
    ...     dtype=object,   # 指定数组的数据类型为对象类型
    ... )
    >>> pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])   # 根据数组 a, b, c 生成交叉表，指定行和列的名称
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0   # 生成的交叉表数据，显示每个组合的计数
    foo    2     2    1     2

    Here 'c' and 'f' are not represented in the data and will not be
    shown in the output because dropna is True by default. Set
    dropna=False to preserve categories with no data.

    >>> foo = pd.Categorical(["a", "b"], categories=["a", "b", "c"])   # 创建一个分类变量 foo，指定类别包括 a, b, c
    >>> bar = pd.Categorical(["d", "e"], categories=["d", "e", "f"])   # 创建一个分类变量 bar，指定类别包括 d, e, f
    >>> pd.crosstab(foo, bar)   # 根据分类变量 foo, bar 生成交叉表，忽略缺失的类别
    col_0  d  e
    row_0
    a      1  0
    b      0  1
    >>> pd.crosstab(foo, bar, dropna=False)   # 根据分类变量 foo, bar 生成交叉表，保留缺失的类别并填充为 0
    col_0  d  e  f
    row_0
    a      1  0  0
    b      0  1  0
    c      0  0  0
    """
    if values is None and aggfunc is not None:   # 如果未提供 values 参数但提供了 aggfunc 参数，则抛出值错误异常
        raise ValueError("aggfunc cannot be used without values.")

    if values is not None and aggfunc is None:   # 如果提供了 values 参数但未提供 aggfunc 参数，则抛出值错误异常
        raise ValueError("values cannot be used without an aggfunc.")

    if not is_nested_list_like(index):   # 如果 index 不是嵌套列表类型，则将其转换为单元素的列表
        index = [index]
    if not is_nested_list_like(columns):   # 如果 columns 不是嵌套列表类型，则将其转换为单元素的列表
        columns = [columns]

    common_idx = None
    pass_objs = [x for x in index + columns if isinstance(x, (ABCSeries, ABCDataFrame))]   # 从 index 和 columns 中筛选出所有是 Series 或 DataFrame 的对象
    if pass_objs:
        common_idx = get_objs_combined_axis(pass_objs, intersect=True, sort=False)   # 获取这些对象的共同索引轴

    rownames = _get_names(index, rownames, prefix="row")   # 获取行名称，如果未指定则使用默认的名称前缀 'row'
    colnames = _get_names(columns, colnames, prefix="col")   # 获取列名称，如果未指定则使用默认的名称前缀 'col'

    # duplicate names mapped to unique names for pivot op
    (
        rownames_mapper,   # 行名称映射器，用于处理重复名称
        unique_rownames,   # 唯一的行名称列表
        colnames_mapper,   # 列名称映射器，用于处理重复名称
        unique_colnames,   # 唯一的列名称列表
    ) = _build_names_mapper(rownames, colnames)   # 根据行和列名称构建名称映射器，处理重复的名称

    from pandas import DataFrame

    data = {   # 创建一个字典包含数据，包括唯一的行名称和列名称
        **dict(zip(unique_rownames, index)),
        **dict(zip(unique_colnames, columns)),
    }
    df = DataFrame(data, index=common_idx)   # 根据数据和共同索引创建 DataFrame

    if values is None:   # 如果未提供 values 参数，则创建一个名为 '__dummy__' 的列并填充为 0
        df["__dummy__"] = 0
        kwargs = {"aggfunc": len, "fill_value": 0}   # 设置聚合函数为 len（计数），填充值为 0
    else:
        df["__dummy__"] = values   # 如果提供了 values 参数，则使用提供的值填充 '__dummy__' 列
        kwargs = {"aggfunc": aggfunc}   # 设置聚合函数为提供的 aggfunc

    # error: Argument 7 to "pivot_table" of "DataFrame" has incompatible type
    # "**Dict[str, object]"; expected "Union[...]"
    # 使用 pandas 中的 pivot_table 方法创建透视表
    table = df.pivot_table(
        "__dummy__",               # 使用虚拟列 "__dummy__" 进行透视表操作
        index=unique_rownames,      # 指定透视表的行索引为 unique_rownames 中的唯一值
        columns=unique_colnames,    # 指定透视表的列索引为 unique_colnames 中的唯一值
        margins=margins,            # 控制是否显示汇总行和列
        margins_name=margins_name,  # 指定汇总行和列的名称
        dropna=dropna,              # 控制是否丢弃缺失值
        observed=dropna,            # 控制观察值的处理方式
        **kwargs,                   # 允许额外的关键字参数，这里忽略类型为 arg-type 的参数检查
    )
    
    # 如果 normalize 参数不为 False，则调用 _normalize 函数进行表格规范化处理
    if normalize is not False:
        table = _normalize(
            table,                  # 将透视表作为参数传递给 _normalize 函数
            normalize=normalize,    # 传递 normalize 参数
            margins=margins,        # 传递 margins 参数
            margins_name=margins_name  # 传递 margins_name 参数
        )
    
    # 重新命名透视表的行索引和列索引
    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)
    
    # 返回最终的透视表结果
    return table
# 定义一个用于规范化数据框的函数，根据指定的参数进行数据规范化，并可添加行和列的边际总和
def _normalize(
    table: DataFrame, normalize, margins: bool, margins_name: Hashable = "All"
) -> DataFrame:
    # 如果normalize参数不是布尔值或字符串，则抛出异常
    if not isinstance(normalize, (bool, str)):
        axis_subs = {0: "index", 1: "columns"}
        try:
            # 尝试使用axis_subs字典将normalize参数转换为相应的字符串形式
            normalize = axis_subs[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

    # 如果margins为False，则执行数据框的实际规范化处理
    if margins is False:
        # 定义不同类型规范化的函数字典
        normalizers: dict[bool | str, Callable] = {
            "all": lambda x: x / x.sum(axis=1).sum(axis=0),
            "columns": lambda x: x / x.sum(),
            "index": lambda x: x.div(x.sum(axis=1), axis=0),
        }

        # 将True映射为"all"
        normalizers[True] = normalizers["all"]

        try:
            # 根据normalize参数选择相应的规范化函数
            f = normalizers[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

        # 应用选择的规范化函数到数据框
        table = f(table)
        # 将NaN值填充为0
        table = table.fillna(0)

    # 如果margins为True，则执行带有边际的数据框处理
    elif margins is True:
        # 保留数据框的索引和列标签
        table_index = table.index
        table_columns = table.columns
        # 获取数据框最后一行或列的名称
        last_ind_or_col = table.iloc[-1, :].name

        # 检查边际名称是否不在最后一个索引/列中，并且不等于最后一个索引/列名称
        if (margins_name not in last_ind_or_col) & (margins_name != last_ind_or_col):
            raise ValueError(f"{margins_name} not in pivoted DataFrame")
        # 提取列边际和行边际数据
        column_margin = table.iloc[:-1, -1]
        index_margin = table.iloc[-1, :-1]

        # 保留数据框的核心部分（去除边际）
        table = table.iloc[:-1, :-1]

        # 递归调用_normalize函数，规范化数据框的核心部分
        table = _normalize(table, normalize=normalize, margins=False)

        # 修复边际
        if normalize == "columns":
            column_margin = column_margin / column_margin.sum()
            table = concat([table, column_margin], axis=1)
            table = table.fillna(0)
            table.columns = table_columns

        elif normalize == "index":
            index_margin = index_margin / index_margin.sum()
            table = table.append(index_margin, ignore_index=True)
            table = table.fillna(0)
            table.index = table_index

        elif normalize == "all" or normalize is True:
            column_margin = column_margin / column_margin.sum()
            index_margin = index_margin / index_margin.sum()
            index_margin.loc[margins_name] = 1
            table = concat([table, column_margin], axis=1)
            table = table.append(index_margin, ignore_index=True)

            table = table.fillna(0)
            table.index = table_index
            table.columns = table_columns

        else:
            raise ValueError("Not a valid normalize argument")

    else:
        raise ValueError("Not a valid margins argument")

    # 返回规范化后的数据框
    return table
    # 如果传入的 names 参数为 None，则初始化一个空列表 names
    if names is None:
        # 遍历 arrs 列表中的每个元素 arr 和它们的索引 i
        for i, arr in enumerate(arrs):
            # 检查 arr 是否是 Pandas 的 Series 类型，并且其名称不为空
            if isinstance(arr, ABCSeries) and arr.name is not None:
                # 将 arr 的名称添加到 names 列表中
                names.append(arr.name)
            else:
                # 如果 arr 不是符合条件的 Series 类型，则使用带有前缀的默认名称添加到 names 中
                names.append(f"{prefix}_{i}")
    else:
        # 如果传入的 names 不为 None，则检查其长度是否与 arrs 列表长度相等
        if len(names) != len(arrs):
            # 如果长度不相等，则抛出断言错误
            raise AssertionError("arrays and names must have the same length")
        # 如果 names 不是列表类型，则将其转换为列表类型
        if not isinstance(names, list):
            names = list(names)

    # 返回处理后的 names 列表
    return names
# 定义一个函数_build_names_mapper，接收两个参数：rownames和colnames，它们分别代表DataFrame的行名和列名列表
def _build_names_mapper(
    rownames: list[str], colnames: list[str]
) -> tuple[dict[str, str], list[str], dict[str, str], list[str]]:
    """
    Given the names of a DataFrame's rows and columns, returns a set of unique row
    and column names and mappers that convert to original names.

    A row or column name is replaced if it is duplicate among the rows of the inputs,
    among the columns of the inputs or between the rows and the columns.

    Parameters
    ----------
    rownames: list[str]
        输入的DataFrame行名列表
    colnames: list[str]
        输入的DataFrame列名列表

    Returns
    -------
    Tuple(Dict[str, str], List[str], Dict[str, str], List[str])

    rownames_mapper: dict[str, str]
        以新行名为键，原始行名为值的字典
    unique_rownames: list[str]
        替换重复行名后的行名列表
    colnames_mapper: dict[str, str]
        以新列名为键，原始列名为值的字典
    unique_colnames: list[str]
        替换重复列名后的列名列表

    """
    # 合并rownames和colnames，找出所有重复的名称
    dup_names = set(rownames) | set(colnames)

    # 生成行名映射字典，将重复的行名映射到新的名称
    rownames_mapper = {
        f"row_{i}": name for i, name in enumerate(rownames) if name in dup_names
    }
    # 生成唯一的行名列表，将重复的行名替换为新的名称
    unique_rownames = [
        f"row_{i}" if name in dup_names else name for i, name in enumerate(rownames)
    ]

    # 生成列名映射字典，将重复的列名映射到新的名称
    colnames_mapper = {
        f"col_{i}": name for i, name in enumerate(colnames) if name in dup_names
    }
    # 生成唯一的列名列表，将重复的列名替换为新的名称
    unique_colnames = [
        f"col_{i}" if name in dup_names else name for i, name in enumerate(colnames)
    ]

    # 返回四个元素的元组，分别是行名映射字典、唯一行名列表、列名映射字典、唯一列名列表
    return rownames_mapper, unique_rownames, colnames_mapper, unique_colnames
```