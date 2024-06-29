# `D:\src\scipysrc\pandas\pandas\core\reshape\melt.py`

```
from __future__ import annotations
# 引入 future 模块，确保代码在 Python 2/3 兼容情况下使用新特性

import re
# 引入 re 模块，用于正则表达式操作

from typing import TYPE_CHECKING
# 引入 TYPE_CHECKING 类型，用于类型提示检查

import numpy as np
# 引入 numpy 库，用于数值计算

from pandas.core.dtypes.common import (
    is_iterator,
    is_list_like,
)
# 从 pandas 库的 dtypes.common 模块中引入 is_iterator 和 is_list_like 函数

from pandas.core.dtypes.concat import concat_compat
# 从 pandas 库的 dtypes.concat 模块中引入 concat_compat 函数

from pandas.core.dtypes.missing import notna
# 从 pandas 库的 dtypes.missing 模块中引入 notna 函数

import pandas.core.algorithms as algos
# 引入 pandas 库的 core.algorithms 模块，用于算法操作

from pandas.core.indexes.api import MultiIndex
# 从 pandas 库的 indexes.api 模块中引入 MultiIndex 类

from pandas.core.reshape.concat import concat
# 从 pandas 库的 reshape.concat 模块中引入 concat 函数

from pandas.core.reshape.util import tile_compat
# 从 pandas 库的 reshape.util 模块中引入 tile_compat 函数

from pandas.core.tools.numeric import to_numeric
# 从 pandas 库的 tools.numeric 模块中引入 to_numeric 函数

if TYPE_CHECKING:
    from collections.abc import Hashable
    # 如果是类型检查模式，则从 collections.abc 模块中引入 Hashable 类型

    from pandas._typing import AnyArrayLike
    # 如果是类型检查模式，则从 pandas._typing 模块中引入 AnyArrayLike 类型

    from pandas import DataFrame
    # 如果是类型检查模式，则从 pandas 库中引入 DataFrame 类型

def ensure_list_vars(arg_vars, variable: str, columns) -> list:
    # 确保返回一个列表形式的变量集合
    if arg_vars is not None:
        # 如果参数不为空
        if not is_list_like(arg_vars):
            # 如果参数不是列表形式
            return [arg_vars]
            # 将参数转为单元素列表返回
        elif isinstance(columns, MultiIndex) and not isinstance(arg_vars, list):
            # 如果列是 MultiIndex 类型且参数不是列表
            raise ValueError(
                f"{variable} must be a list of tuples when columns are a MultiIndex"
            )
            # 抛出数值错误，要求参数在列是 MultiIndex 时必须是元组列表
        else:
            return list(arg_vars)
            # 返回参数的列表形式
    else:
        return []
        # 如果参数为空，则返回空列表

def melt(
    frame: DataFrame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name: Hashable = "value",
    col_level=None,
    ignore_index: bool = True,
) -> DataFrame:
    """
    Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

    This function is useful to massage a DataFrame into a format where one
    or more columns are identifier variables (`id_vars`), while all other
    columns, considered measured variables (`value_vars`), are "unpivoted" to
    the row axis, leaving just two non-identifier columns, 'variable' and
    'value'.

    Parameters
    ----------
    frame : DataFrame
        The DataFrame to unpivot.
    id_vars : scalar, tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.
    value_vars : scalar, tuple, list, or ndarray, optional
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.
    var_name : scalar, tuple, list, or ndarray, optional
        Name to use for the 'variable' column. If None it uses
        ``frame.columns.name`` or 'variable'. Must be a scalar if columns are a
        MultiIndex.
    value_name : scalar, default 'value'
        Name to use for the 'value' column, can't be an existing column label.
    col_level : scalar, optional
        If columns are a MultiIndex then use this level to melt.
    ignore_index : bool, default True
        If True, original index is ignored. If False, the original index is retained.
        Index labels will be repeated as necessary.

    Returns
    -------
    DataFrame
        Unpivoted DataFrame.

    See Also
    --------
    DataFrame.melt : Identical method.
    pivot_table : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Return reshaped DataFrame organized
        by given index / column values.
    """
    # 从宽格式到长格式的数据框解构，可选地保留标识符设置

    # 检查是否有指定的列作为标识符变量，若没有则使用全部非标识符列
    id_vars = ensure_list_vars(id_vars, 'id_vars', frame.columns)
    # 检查是否有指定的列作为值变量，若没有则使用全部非标识符列
    value_vars = ensure_list_vars(value_vars, 'value_vars', frame.columns)
    # 检查并获取 'var_name' 列的名称，如果为 None 则使用列名或默认 'variable'
    if var_name is None:
        var_name = frame.columns.name or 'variable'
    # 使用 pandas 中的 melt 函数进行数据框解构
    return frame.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        ignore_index=ignore_index,
    )
    # 检查 DataFrame 是否包含指定的 value_name 列，如果是则抛出 ValueError
    if value_name in frame.columns:
        raise ValueError(
            f"value_name ({value_name}) cannot match an element in "
            "the DataFrame columns."
        )
    
    # 确保 id_vars 参数是一个列表，并且其元素都存在于 DataFrame 的列中
    id_vars = ensure_list_vars(id_vars, "id_vars", frame.columns)
    
    # 检查 value_vars 是否在调用前不为 None
    value_vars_was_not_none = value_vars is not None
    
    # 确保 value_vars 参数是一个列表，并且其元素都存在于 DataFrame 的列中
    value_vars = ensure_list_vars(value_vars, "value_vars", frame.columns)
    # 如果 id_vars 或者 value_vars 不为空
    if id_vars or value_vars:
        # 如果指定了 col_level
        if col_level is not None:
            # 获取指定层级的列值
            level = frame.columns.get_level_values(col_level)
        else:
            # 否则使用所有列
            level = frame.columns
        # 合并 id_vars 和 value_vars 到 labels 列表中
        labels = id_vars + value_vars
        # 获取 labels 在 level 中的索引
        idx = level.get_indexer_for(labels)
        # 找出不存在于 level 中的索引
        missing = idx == -1
        # 如果存在不存在的索引
        if missing.any():
            # 找出缺失的标签
            missing_labels = [
                lab for lab, not_found in zip(labels, missing) if not_found
            ]
            # 抛出 KeyError 异常，显示缺失的 id_vars 或 value_vars
            raise KeyError(
                "The following id_vars or value_vars are not present in "
                f"the DataFrame: {missing_labels}"
            )
        # 如果 value_vars 不为空
        if value_vars_was_not_none:
            # 从 frame 中选择唯一的索引列
            frame = frame.iloc[:, algos.unique(idx)]
        else:
            # 否则复制 frame （浅拷贝）
            frame = frame.copy(deep=False)
    else:
        # 否则复制 frame （浅拷贝）
        frame = frame.copy(deep=False)

    # 如果指定了 col_level
    if col_level is not None:  # allow list or other?
        # frame 是一个拷贝
        frame.columns = frame.columns.get_level_values(col_level)

    # 如果 var_name 为 None
    if var_name is None:
        # 如果 frame 的列是 MultiIndex
        if isinstance(frame.columns, MultiIndex):
            # 如果每个级别的名称都不同
            if len(frame.columns.names) == len(set(frame.columns.names)):
                # 使用列级别的名称作为 var_name
                var_name = frame.columns.names
            else:
                # 否则使用默认的 variable_{i} 格式作为 var_name
                var_name = [f"variable_{i}" for i in range(len(frame.columns.names))]
        else:
            # 否则使用列名或者默认的 variable 作为 var_name
            var_name = [
                frame.columns.name if frame.columns.name is not None else "variable"
            ]
    # 否则如果 var_name 是列表类型
    elif is_list_like(var_name):
        # 如果 frame 的列是 MultiIndex
        if isinstance(frame.columns, MultiIndex):
            # 如果 var_name 是迭代器，则转换为列表
            if is_iterator(var_name):
                var_name = list(var_name)
            # 如果 var_name 的长度大于 frame 的列数，抛出 ValueError
            if len(var_name) > len(frame.columns):
                raise ValueError(
                    f"{var_name=} has {len(var_name)} items, "
                    f"but the dataframe columns only have {len(frame.columns)} levels."
                )
        else:
            # 否则 var_name 必须是标量，抛出 ValueError
            raise ValueError(f"{var_name=} must be a scalar.")
    else:
        # 否则将 var_name 转换为列表
        var_name = [var_name]

    # 获取 frame 的行数和列数
    num_rows, K = frame.shape
    # 调整后的列数（排除 id_vars）
    num_cols_adjusted = K - len(id_vars)

    # 创建一个字典 mdata 存储数据
    mdata: dict[Hashable, AnyArrayLike] = {}
    # 遍历 id_vars 列
    for col in id_vars:
        # 弹出 frame 中的 id_vars 列数据
        id_data = frame.pop(col)
        # 如果 id_data 的数据类型不是 np.dtype
        if not isinstance(id_data.dtype, np.dtype):
            # 使用 concat 函数复制 num_cols_adjusted 次 id_data，忽略索引
            if num_cols_adjusted > 0:
                mdata[col] = concat([id_data] * num_cols_adjusted, ignore_index=True)
            else:
                # 如果无法连接空列表，创建一个空的 id_data 类型的对象
                mdata[col] = type(id_data)([], name=id_data.name, dtype=id_data.dtype)
        else:
            # 否则使用 np.tile 复制 id_data._values，复制次数为 num_cols_adjusted
            mdata[col] = np.tile(id_data._values, num_cols_adjusted)

    # 构建 mcolumns 列表，包含 id_vars、var_name 和 value_name
    mcolumns = id_vars + var_name + [value_name]

    # 如果 frame 的列数大于 0 并且 frame 的每列都支持 2D
    if frame.shape[1] > 0 and not any(
        not isinstance(dt, np.dtype) and dt._supports_2d for dt in frame.dtypes
    ):
        # 使用 concat 函数连接 frame 中的所有列，忽略索引，并获取其值
        mdata[value_name] = concat(
            [frame.iloc[:, i] for i in range(frame.shape[1])], ignore_index=True
        ).values
    else:
        # 否则将 frame 的值按 Fortran 风格展平，并赋给 mdata[value_name]
        mdata[value_name] = frame._values.ravel("F")
    # 使用 enumerate() 遍历 var_name 列表，并获取索引 i 和对应的列名 col
    for i, col in enumerate(var_name):
        # 将 frame 对象的列中第 i 层级的值重复 num_rows 次，并赋值给 mdata 的对应列 col
        mdata[col] = frame.columns._get_level_values(i).repeat(num_rows)

    # 使用 mdata 数据创建一个新的 frame 对象，列使用 mcolumns 指定
    result = frame._constructor(mdata, columns=mcolumns)

    # 如果 ignore_index 为 False，则将 result 对象的索引调整为与 frame.index 兼容的值
    if not ignore_index:
        result.index = tile_compat(frame.index, num_cols_adjusted)

    # 返回处理后的结果对象
    return result
# 将宽格式数据重塑为长格式。是 DataFrame.pivot 的广义逆过程。

# 接受一个字典 `groups`，其中每个键是新列名，每个值是要在重塑过程中作为新列名下“融合”的旧列名列表。

# 参数说明：
# data : DataFrame
#     宽格式的 DataFrame。
# groups : dict
#     {新列名 : 列名列表}。
# dropna : bool, 默认 True
#     不包括所有条目为 NaN 的列。

# 返回：
# DataFrame
#     重塑后的 DataFrame。

# 参见：
# melt : 将 DataFrame 从宽格式转换为长格式，可选择保留标识符。
# pivot : 创建类似电子表格的透视表作为 DataFrame。
# DataFrame.pivot : 不进行聚合的透视，可以处理非数值数据。
# DataFrame.pivot_table : pivot 的泛化形式，可以处理一个索引/列对的重复值。
# DataFrame.unstack : 基于索引值而不是列进行透视。
# wide_to_long : 将宽面板数据转换为长格式。比 melt less flexible 但更用户友好。

# 示例：
# >>> data = pd.DataFrame(
# ...     {
# ...         "hr1": [514, 573],
# ...         "hr2": [545, 526],
# ...         "team": ["Red Sox", "Yankees"],
# ...         "year1": [2007, 2007],
# ...         "year2": [2008, 2008],
# ...     }
# ... )
# >>> data
#    hr1  hr2     team  year1  year2
# 0  514  545  Red Sox   2007   2008
# 1  573  526  Yankees   2007   2008
# >>> pd.lreshape(data, {"year": ["year1", "year2"], "hr": ["hr1", "hr2"]})
#       team  year   hr
# 0  Red Sox  2007  514
# 1  Yankees  2007  573
# 2  Red Sox  2008  545
# 3  Yankees  2008  526
def lreshape(data: DataFrame, groups: dict, dropna: bool = True) -> DataFrame:
    mdata = {}  # 创建空字典以存储重塑后的数据
    pivot_cols = []  # 创建空列表以存储要进行透视的列名
    all_cols: set[Hashable] = set()  # 创建空集合以存储所有要处理的列名
    K = len(next(iter(groups.values())))  # 获取第一个值列表的长度作为 K
    for target, names in groups.items():
        if len(names) != K:
            raise ValueError("All column lists must be same length")  # 如果列表长度不同，抛出错误
        to_concat = [data[col]._values for col in names]  # 提取要合并的列的值列表
        mdata[target] = concat_compat(to_concat)  # 将合并后的值存储在目标键下
        pivot_cols.append(target)  # 将目标键添加到透视列列表中
        all_cols = all_cols.union(names)  # 将当前处理的列名添加到所有列名集合中

    id_cols = list(data.columns.difference(all_cols))  # 找到不在所有列名集合中的标识符列名
    for col in id_cols:
        mdata[col] = np.tile(data[col]._values, K)  # 使用 np.tile 复制标识符列的值 K 次

    if dropna:
        mask = np.ones(len(mdata[pivot_cols[0]]), dtype=bool)  # 创建用于掩码的布尔数组
        for c in pivot_cols:
            mask &= notna(mdata[c])  # 更新掩码以剔除所有包含 NaN 的行
        if not mask.all():
            mdata = {k: v[mask] for k, v in mdata.items()}  # 应用掩码以删除对应行

    return data._constructor(mdata, columns=id_cols + pivot_cols)  # 构造并返回新的 DataFrame 对象


def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = "", suffix: str = r"\d+"
) -> DataFrame:
    r"""
    Unpivot a DataFrame from wide to long format.

    Less flexible but more user-friendly than melt.
    """
    # 此函数将不会进行注释，因为它超出了您请求的范围。
    With stubnames ['A', 'B'], this function expects to find one or more
    group of columns with format
    A-suffix1, A-suffix2,..., B-suffix1, B-suffix2,...
    You specify what you want to call this suffix in the resulting long format
    with `j` (for example `j='year'`)

    Each row of these wide variables are assumed to be uniquely identified by
    `i` (can be a single column name or a list of column names)

    All remaining variables in the data frame are left intact.

    Parameters
    ----------
    df : DataFrame
        The wide-format DataFrame.
    stubnames : str or list-like
        The stub name(s). The wide format variables are assumed to
        start with the stub names.
    i : str or list-like
        Column(s) to use as id variable(s).
    j : str
        The name of the sub-observation variable. What you wish to name your
        suffix in the long format.
    sep : str, default ""
        A character indicating the separation of the variable names
        in the wide format, to be stripped from the names in the long format.
        For example, if your column names are A-suffix1, A-suffix2, you
        can strip the hyphen by specifying `sep='-'`.
    suffix : str, default '\\d+'
        A regular expression capturing the wanted suffixes. '\\d+' captures
        numeric suffixes. Suffixes with no numbers could be specified with the
        negated character class '\\D+'. You can also further disambiguate
        suffixes, for example, if your wide variables are of the form A-one,
        B-two,.., and you have an unrelated column A-rating, you can ignore the
        last one by specifying `suffix='(!?one|two)'`. When all suffixes are
        numeric, they are cast to int64/float64.

    Returns
    -------
    DataFrame
        A DataFrame that contains each stub name as a variable, with new index
        (i, j).

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.

    Notes
    -----
    All extra variables are left untouched. This simply uses
    `pandas.melt` under the hood, but is hard-coded to "do the right thing"
    in a typical case.

    Examples
    --------
    >>> np.random.seed(123)
    >>> df = pd.DataFrame(
    ...     {
    ...         "A1970": {0: "a", 1: "b", 2: "c"},
    ...         "A1980": {0: "d", 1: "e", 2: "f"},
    ...         "B1970": {0: 2.5, 1: 1.2, 2: 0.7},
    ...         "B1980": {0: 3.2, 1: 1.3, 2: 0.1},
    ...         "X": dict(zip(range(3), np.random.randn(3))),
    ...     }
    ... )
    >>> df["id"] = df.index
    # 创建一个示例数据框 df，包含了多列数据
    df = pd.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],  # 家庭ID列
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],  # 出生顺序列
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],  # 身高1数据列
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],  # 身高2数据列
        }
    )
    
    # 展示数据框 df
    >>> df
    
    # 使用 `wide_to_long` 函数将宽格式数据转换为长格式数据
    long_format = pd.wide_to_long(df, stubnames="ht", i=["famid", "birth"], j="age")
    
    # 展示转换后的长格式数据
    >>> long_format
    
    # 使用 `unstack` 方法将长格式数据转换回宽格式数据
    wide_format = long_format.unstack()
    
    # 重新设置列名，格式为 "{列名}{年龄}"，以便与初始数据框列名对应
    wide_format.columns = wide_format.columns.map("{0[0]}{0[1]}".format)
    
    # 重置索引，将多层索引转换为单层索引
    wide_format.reset_index()
    
    # 展示重置索引后的数据框
    >>> wide_format.reset_index()
    
    # 使用 NumPy 设置随机种子，生成随机数填充数据框 df
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "A(weekly)-2010": np.random.rand(3),  # 2010年周数据列A
            "A(weekly)-2011": np.random.rand(3),  # 2011年周数据列A
            "B(weekly)-2010": np.random.rand(3),  # 2010年周数据列B
            "B(weekly)-2011": np.random.rand(3),  # 2011年周数据列B
            "X": np.random.randint(3, size=3),  # 随机整数列X
        }
    )
    
    # 添加列 'id'，值为数据框的索引值
    df["id"] = df.index
    
    # 展示填充后的数据框 df
    >>> df
    def get_var_names(df, stub: str, sep: str, suffix: str):
        # 构建用于匹配列名的正则表达式，这里使用了 re.escape 来确保 stub 和 suffix 中的特殊字符被正确转义
        regex = rf"^{re.escape(stub)}{re.escape(sep)}{suffix}$"
        # 使用正则表达式匹配 DataFrame 中符合条件的列名，并返回一个包含匹配列名的数组
        return df.columns[df.columns.str.match(regex)]
    def melt_stub(df, stub: str, i, j, value_vars, sep: str):
        # 使用 pd.melt 函数将 DataFrame 按照指定的列和值变量进行重塑
        newdf = melt(
            df,
            id_vars=i,
            value_vars=value_vars,
            value_name=stub.rstrip(sep),
            var_name=j,
        )
        # 替换变量名列中的前缀和分隔符，使其变得更加整洁
        newdf[j] = newdf[j].str.replace(re.escape(stub + sep), "", regex=True)

        # GH17627 尝试将数值后缀转换为整数或浮点数类型
        try:
            newdf[j] = to_numeric(newdf[j])
        except (TypeError, ValueError, OverflowError):
            # TODO: 是否还需要处理其它异常？
            pass

        # 返回设置了索引的新 DataFrame
        return newdf.set_index(i + [j])

    # 如果 stubnames 不是类列表对象，则将其转换为列表
    if not is_list_like(stubnames):
        stubnames = [stubnames]
    else:
        stubnames = list(stubnames)

    # 如果 DataFrame 的列名与 stubnames 中的任何一个重复，则抛出 ValueError 异常
    if df.columns.isin(stubnames).any():
        raise ValueError("stubname can't be identical to a column name")

    # 如果 i 不是类列表对象，则将其转换为列表
    if not is_list_like(i):
        i = [i]
    else:
        i = list(i)

    # 如果 DataFrame 中的 id 变量有重复的情况，则抛出 ValueError 异常
    if df[i].duplicated().any():
        raise ValueError("the id variables need to uniquely identify each row")

    # 存储每个 stubname 对应的重塑后的 DataFrame，同时扩展 value_vars_flattened 列表
    _melted = []
    value_vars_flattened = []
    for stub in stubnames:
        # 获取每个 stubname 对应的值变量名
        value_var = get_var_names(df, stub, sep, suffix)
        value_vars_flattened.extend(value_var)
        # 对每个 stubname 调用 melt_stub 函数进行重塑，并将结果添加到 _melted 列表中
        _melted.append(melt_stub(df, stub, i, j, value_var, sep))

    # 将所有重塑后的 DataFrame 连接在一起，按列连接（axis=1）
    melted = concat(_melted, axis=1)
    # 从原始 DataFrame 中选择除了 value_vars_flattened 之外的列作为新的 DataFrame
    id_vars = df.columns.difference(value_vars_flattened)
    new = df[id_vars]

    # 根据 i 的长度决定是使用 set_index 和 join 还是 merge，并返回设置了索引的新 DataFrame
    if len(i) == 1:
        return new.set_index(i).join(melted)
    else:
        return new.merge(melted.reset_index(), on=i).set_index(i + [j])
```