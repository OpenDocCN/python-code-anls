# `D:\src\scipysrc\pandas\pandas\core\methods\to_dict.py`

```
from __future__ import annotations
# 导入类型标注所需的模块和特性

from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
# 导入类型检查相关模块和特性

import warnings
# 导入警告模块

import numpy as np
# 导入 NumPy 库，并用 np 别名引用

from pandas._libs import (
    lib,
    missing as libmissing,
)
# 从 pandas 库的 _libs 子模块中导入 lib 和 libmissing

from pandas.util._exceptions import find_stack_level
# 从 pandas.util._exceptions 导入 find_stack_level 函数

from pandas.core.dtypes.cast import maybe_box_native
# 从 pandas.core.dtypes.cast 模块导入 maybe_box_native 函数

from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    ExtensionDtype,
)
# 从 pandas.core.dtypes.dtypes 导入 BaseMaskedDtype 和 ExtensionDtype 类型

from pandas.core import common as com
# 从 pandas.core 导入 common 模块，并用 com 别名引用

if TYPE_CHECKING:
    from collections.abc import Generator
    # 如果处于类型检查模式，导入 Generator 类型

    from pandas._typing import MutableMappingT
    # 导入 pandas._typing 中的 MutableMappingT 类型

    from pandas import DataFrame
    # 导入 pandas 中的 DataFrame 类型

def create_data_for_split(
    df: DataFrame, are_all_object_dtype_cols: bool, object_dtype_indices: list[int]
) -> Generator[list, None, None]:
    """
    Simple helper method to create data for to ``to_dict(orient="split")``
    to create the main output data
    """
    # 根据给定的 DataFrame 创建用于 "split" 方向的数据生成器
    if are_all_object_dtype_cols:
        # 如果所有列都是 object 类型
        for tup in df.itertuples(index=False, name=None):
            # 遍历 DataFrame 的行元组（不包含索引），每个元组不带名称
            yield list(map(maybe_box_native, tup))
            # 使用 maybe_box_native 处理每个元组中的元素，并生成列表
    else:
        # 如果不是所有列都是 object 类型
        for tup in df.itertuples(index=False, name=None):
            # 遍历 DataFrame 的行元组（不包含索引），每个元组不带名称
            data = list(tup)
            # 将元组转换为列表
            if object_dtype_indices:
                # 如果有指定的 object 类型列索引
                # 对这些列应用 maybe_box_native 函数以提高性能
                for i in object_dtype_indices:
                    data[i] = maybe_box_native(data[i])
                    # 处理指定列的数据
            yield data
            # 生成处理后的数据列表

@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> MutableMappingT: ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> list[MutableMappingT]: ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> dict: ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> list[dict]: ...


# error: Incompatible default for argument "into" (default has type "type[dict
# [Any, Any]]", argument has type "type[MutableMappingT] | MutableMappingT")
def to_dict(
    df: DataFrame,
    orient: Literal[
        "dict", "list", "series", "split", "tight", "records", "index"
    ] = "dict",
    *,
    into: type[MutableMappingT] | MutableMappingT = dict,  # type: ignore[assignment]
    index: bool = True,
) -> MutableMappingT | list[MutableMappingT]:
    """
    Convert the DataFrame to a dictionary.

    The type of the key-value pairs can be customized with the parameters
    (see below).

    Parameters
    ----------

    df : DataFrame
        The input DataFrame to be converted.
    orient : Literal["dict", "list", "series", "split", "tight", "records", "index"], optional
        The orientation of the output dictionary, default is "dict".
    into : type[MutableMappingT] | MutableMappingT, optional
        The type into which to convert the DataFrame, default is dict.
    index : bool, default True
        Whether to include the DataFrame's index as part of dict keys.

    Returns
    -------
    MutableMappingT | list[MutableMappingT]
        A dictionary or list of dictionaries representing the DataFrame.

    Notes
    -----
    This function converts a DataFrame into a dictionary format based on the
    specified orientation and type (dict or list of dicts).
    """
    # 将 DataFrame 转换为字典格式的函数
    # orient参数指定返回的数据结构类型，可以是字典、列表、Series等多种形式
    # 默认为'dict'，表示返回字典形式的数据结构，其中每列作为字典的键，对应的数据为值
    # 其他可选项包括'list'、'series'、'split'、'tight'、'records'、'index'
    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        Determines the type of the values of the dictionary.

        - 'dict' (default) : dict like {column -> {index -> value}}
        - 'list' : dict like {column -> [values]}
        - 'series' : dict like {column -> Series(values)}
        - 'split' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
        - 'tight' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
          'index_names' -> [index.names], 'column_names' -> [column.names]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}

        .. versionadded:: 1.4.0
            'tight' as an allowed value for the ``orient`` argument

    # into参数指定返回值的集合类型，通常为一个collections.abc.MutableMapping的子类实例
    # 默认为dict，即返回一个字典类型的对象
    into : class, default dict
        The collections.abc.MutableMapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    # index参数控制是否在返回的字典中包含索引条目及其名称（仅当orient为'split'或'tight'时可为False）
    # 默认为True，表示包含索引条目及其名称
    index : bool, default True
        Whether to include the index item (and index_names item if `orient`
        is 'tight') in the returned dictionary. Can only be ``False``
        when `orient` is 'split' or 'tight'.

        .. versionadded:: 2.0.0

    Returns
    -------
    dict, list or collections.abc.Mapping
        Return a collections.abc.MutableMapping object representing the
        DataFrame. The resulting transformation depends on the `orient` parameter.
    """
    # 如果orient不是'tight'且DataFrame的列不是唯一的，发出警告
    if orient != "tight" and not df.columns.is_unique:
        warnings.warn(
            "DataFrame columns are not unique, some columns will be omitted.",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    # 通过com.standardize_mapping方法标准化into参数指定的集合类型
    into_c = com.standardize_mapping(into)  # type: ignore[no-untyped-call]

    # 将orient参数的值转换为小写
    orient = orient.lower()  # type: ignore[assignment]

    # 如果index为False且orient不在['split', 'tight']中，抛出ValueError异常
    if not index and orient not in ["split", "tight"]:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split' or 'tight'"
        )

    # 如果orient为'series'，快速返回结果，避免创建dtype对象
    if orient == "series":
        # GH46470 Return quickly if orient series to avoid creating dtype objects
        return into_c((k, v) for k, v in df.items())

    # 如果orient为'dict'，返回由每列转换为字典形式的集合类型对象
    if orient == "dict":
        return into_c((k, v.to_dict(into=into)) for k, v in df.items())

    # 获取所有数据类型为object或ExtensionDtype的列的索引
    box_native_indices = [
        i
        for i, col_dtype in enumerate(df.dtypes.values)
        if col_dtype == np.dtype(object) or isinstance(col_dtype, ExtensionDtype)
    ]

    # 判断是否所有列的数据类型都为object
    are_all_object_dtype_cols = len(box_native_indices) == len(df.dtypes)
    # 如果 orient 参数为 "list"，则执行以下逻辑
    if orient == "list":
        # 将 box_native_indices 转换为集合对象
        object_dtype_indices_as_set: set[int] = set(box_native_indices)
        # 如果列的数据类型不是 BaseMaskedDtype 类型，则 box_na_values 为 lib.no_default；否则为 libmissing.NA
        box_na_values = (
            lib.no_default
            if not isinstance(col_dtype, BaseMaskedDtype)
            else libmissing.NA
            for col_dtype in df.dtypes.values
        )
        # 将 DataFrame 转换为嵌套元组的形式，其中可能对某些对象类型列进行了装箱
        return into_c(
            (
                k,
                list(map(maybe_box_native, v.to_numpy(na_value=box_na_value)))
                if i in object_dtype_indices_as_set
                else list(map(maybe_box_native, v.to_numpy())),
            )
            for i, (box_na_value, (k, v)) in enumerate(zip(box_na_values, df.items()))
        )

    # 如果 orient 参数为 "split"，则执行以下逻辑
    elif orient == "split":
        # 创建用于 "split" 方式的数据，返回一个列表
        data = list(
            create_data_for_split(df, are_all_object_dtype_cols, box_native_indices)
        )
        # 将索引、列名和数据组成一个元组列表，转换为嵌套元组的形式
        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                ("data", data),
            )
        )

    # 如果 orient 参数为 "tight"，则执行以下逻辑
    elif orient == "tight":
        # 返回一个紧凑型的表示，包含索引、列名和数据
        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                (
                    "data",
                    [
                        list(map(maybe_box_native, t))
                        for t in df.itertuples(index=False, name=None)
                    ],
                ),
            )
            + ((("index_names", list(df.index.names)),) if index else ())
            + (("column_names", list(df.columns.names)),)
        )

    # 如果 orient 参数为 "records"，则执行以下逻辑
    elif orient == "records":
        # 获取 DataFrame 的列名列表
        columns = df.columns.tolist()
        # 如果所有列都是对象类型，则将每行转换为列名和对应值的字典列表，进行装箱处理
        if are_all_object_dtype_cols:
            return [
                into_c(zip(columns, map(maybe_box_native, row)))
                for row in df.itertuples(index=False, name=None)
            ]
        else:
            # 否则，将每行转换为列名和对应值的字典列表
            data = [
                into_c(zip(columns, t)) for t in df.itertuples(index=False, name=None)
            ]
            # 如果 box_native_indices 不为空，则对数据中指定的对象类型列进行装箱处理
            if box_native_indices:
                object_dtype_indices_as_set = set(box_native_indices)
                object_dtype_cols = {
                    col
                    for i, col in enumerate(df.columns)
                    if i in object_dtype_indices_as_set
                }
                for row in data:
                    for col in object_dtype_cols:
                        row[col] = maybe_box_native(row[col])
            return data  # type: ignore[return-value]
    # 如果 orient 参数为 "index"，则按索引的方式处理数据框
    elif orient == "index":
        # 检查数据框索引是否唯一，若不唯一则抛出数值错误异常
        if not df.index.is_unique:
            raise ValueError("DataFrame index must be unique for orient='index'.")
        
        # 将数据框的列名转换为列表形式
        columns = df.columns.tolist()
        
        # 如果所有列都是对象类型，并且需要盒化本地对象
        if are_all_object_dtype_cols:
            # 使用 into_c 函数转换数据结构，生成元组 (t[0], {列名: 盒化本地对象后的值})
            return into_c(
                (t[0], dict(zip(df.columns, map(maybe_box_native, t[1:]))))
                for t in df.itertuples(name=None)
            )
        
        # 如果需要盒化本地索引值
        elif box_native_indices:
            # 将对象类型的索引转换为集合
            object_dtype_indices_as_set = set(box_native_indices)
            # 使用 into_c 函数转换数据结构，生成元组 (t[0], {列名: 盒化本地对象后的值 或者 原始值})
            return into_c(
                (
                    t[0],
                    {
                        column: maybe_box_native(v)
                        if i in object_dtype_indices_as_set
                        else v
                        for i, (column, v) in enumerate(zip(columns, t[1:]))
                    },
                )
                for t in df.itertuples(name=None)
            )
        
        # 默认情况下，直接使用 into_c 函数转换数据结构，生成元组 (t[0], {列名: 值})
        else:
            return into_c(
                (t[0], dict(zip(columns, t[1:]))) for t in df.itertuples(name=None)
            )
    
    # 如果 orient 参数不是 "index"，抛出数值错误异常
    else:
        raise ValueError(f"orient '{orient}' not understood")
```