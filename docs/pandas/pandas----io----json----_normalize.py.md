# `D:\src\scipysrc\pandas\pandas\io\json\_normalize.py`

```
# ---------------------------------------------------------------------
# JSON normalization routines

# 引入未来支持的注解特性，用于类型注解中的 self 参考
from __future__ import annotations

# 引入 collections 模块的 abc 和 defaultdict 类
from collections import (
    abc,
    defaultdict,
)

# 引入 copy 模块
import copy

# 引入 typing 模块中的类型注解相关内容
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    overload,
)

# 引入 numpy 库，并命名为 np
import numpy as np

# 从 pandas._libs.writers 中引入 convert_json_to_lines 函数
from pandas._libs.writers import convert_json_to_lines

# 引入 pandas 库，并命名为 pd
import pandas as pd

# 从 pandas 中引入 DataFrame 和 Series 类
from pandas import (
    DataFrame,
    Series,
)

# 如果正在类型检查中，则从 collections.abc 中引入 Iterable 类型
if TYPE_CHECKING:
    from collections.abc import Iterable

    # 从 pandas._typing 中引入 IgnoreRaise 和 Scalar 类型
    from pandas._typing import (
        IgnoreRaise,
        Scalar,
    )


def convert_to_line_delimits(s: str) -> str:
    """
    Helper function that converts JSON lists to line delimited JSON.

    Parameters
    ----------
    s : str
        Input string that may represent a JSON list.

    Returns
    -------
    str
        Returns the input string converted to line-delimited JSON format,
        or the original string if not a JSON list.
    """
    # 如果输入的字符串不是以 "[" 开头且以 "]" 结尾，则直接返回原字符串
    if not s[0] == "[" and s[-1] == "]":
        return s
    # 去除首尾的 "[" 和 "]"，得到中间的 JSON 对象
    s = s[1:-1]

    # 调用 convert_json_to_lines 函数将 JSON 对象转换为行分隔的 JSON 格式
    return convert_json_to_lines(s)


@overload
def nested_to_record(
    ds: dict,
    prefix: str = ...,
    sep: str = ...,
    level: int = ...,
    max_level: int | None = ...,
) -> dict[str, Any]: ...


@overload
def nested_to_record(
    ds: list[dict],
    prefix: str = ...,
    sep: str = ...,
    level: int = ...,
    max_level: int | None = ...,
) -> list[dict[str, Any]]: ...


def nested_to_record(
    ds: dict | list[dict],
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    A simplified json_normalize

    Converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.

    Parameters
    ----------
    ds : dict or list of dicts
        The nested dictionary or a list of dictionaries to be flattened.
    prefix : str, optional, default ''
        The prefix to prepend to keys in the flattened dictionary.
    sep : str, optional, default '.'
        Separator to be used between keys in nested records.
    level : int, optional, default 0
        Current level of nesting in the JSON structure.
    max_level : int or None, optional, default None
        Maximum depth to which normalization should occur.

    Returns
    -------
    dict or list of dicts
        Flattened dictionary or list of flattened dictionaries corresponding to `ds`.

    Examples
    --------
    >>> nested_to_record(
    ...     dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))
    ... )
    {
        'flat1': 1,
        'dict1.c': 1,
        'dict1.d': 2,
        'nested.e.c': 1,
        'nested.e.d': 2,
        'nested.d': 2
    }
    """
    # 如果 ds 是 dict 类型，则转换为单元素的列表，标记为 singleton
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True

    # 初始化一个新的空列表 new_ds，用于存储转换后的结果
    new_ds = []
    # 对输入的每个字典进行迭代处理
    for d in ds:
        # 对当前字典进行深拷贝，以避免修改原始数据
        new_d = copy.deepcopy(d)
        # 对当前字典中的每对键值对进行迭代处理
        for k, v in d.items():
            # 如果键不是字符串类型，则转换为字符串类型
            if not isinstance(k, str):
                k = str(k)
            
            # 根据指定的层级确定新的键名
            if level == 0:
                newkey = k  # 在第一层级时保持原始键名不变
            else:
                newkey = prefix + sep + k  # 在后续层级添加前缀和分隔符作为新键名的前缀
            
            # 如果值的类型不是字典，或者已达到最大展开层级（如果有指定的话）
            # 则直接将值放入新的字典中，并跳过递归展开
            if not isinstance(v, dict) or (
                max_level is not None and level >= max_level
            ):
                if level != 0:  # 对于非顶层的情况，跳过复制操作（通常情况）
                    v = new_d.pop(k)  # 从原始字典中移除原键及其对应值
                    new_d[newkey] = v  # 将值与新键名关联并添加到新字典中
                continue

            # 如果值是字典且需要递归展开，则继续递归调用 nested_to_record 函数
            v = new_d.pop(k)  # 从原始字典中移除原键及其对应值
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))  # 递归调用展开字典值，并更新到新字典中

        # 将处理后的新字典添加到结果列表中
        new_ds.append(new_d)

    # 如果 singleton 参数为真，则返回列表中的第一个元素（通常用于处理单一结果）
    if singleton:
        return new_ds[0]
    return new_ds  # 否则返回包含所有处理后字典的列表
# 主递归函数
# 用于将嵌套的 JSON 数据规范化为扁平化的字典，类似于 pd.json_normalize(data)，旨在提升性能，见 issue #15621
def _normalise_json(
    data: Any,
    key_string: str,
    normalized_dict: dict[str, Any],
    separator: str,
) -> dict[str, Any]:
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        # 遍历字典中的键值对
        for key, value in data.items():
            new_key = f"{key_string}{separator}{key}"

            # 如果 key_string 为空，则删除 new_key 的开头分隔符
            if not key_string:
                new_key = new_key.removeprefix(separator)

            # 递归调用自身，处理嵌套的字典值
            _normalise_json(
                data=value,
                key_string=new_key,
                normalized_dict=normalized_dict,
                separator=separator,
            )
    else:
        # 如果数据不是字典，将其加入到规范化后的字典中
        normalized_dict[key_string] = data
    return normalized_dict


# 按顺序排序顶层键，并递归到深度
def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Order the top level keys and then recursively go to depth

    Parameters
    ----------
    data : dict or list of dicts
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    dict or list of dicts, matching `normalised_json_object`
    """
    # 提取顶层不是字典的键值对
    top_dict_ = {k: v for k, v in data.items() if not isinstance(v, dict)}
    # 调用 _normalise_json 处理顶层是字典的键值对
    nested_dict_ = _normalise_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string="",
        normalized_dict={},
        separator=separator,
    )
    # 返回合并后的字典
    return {**top_dict_, **nested_dict_}


# 优化的基本 json_normalize
# 将嵌套的字典转换为扁平化的字典记录，不像 json_normalize 和 nested_to_record 那样做任何智能处理，但对于最基本的用例提升性能
def _simple_json_normalize(
    ds: dict | list[dict],
    sep: str = ".",
) -> dict | list[dict] | Any:
    """
    A optimized basic json_normalize

    Converts a nested dict into a flat dict ("record"), unlike
    json_normalize and nested_to_record it doesn't do anything clever.
    But for the most basic use cases it enhances performance.
    E.g. pd.json_normalize(data)

    Parameters
    ----------
    ds : dict or list of dicts
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    frame : DataFrame
    d - dict or list of dicts, matching `normalised_json_object`

    Examples
    --------
    >>> _simple_json_normalize(
    ...     {
    ...         "flat1": 1,
    ...         "dict1": {"c": 1, "d": 2},
    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},
    ...     }
    ... )
    {
        'flat1': 1,
        'dict1.c': 1,
        'dict1.d': 2,
        'nested.e.c': 1,
        'nested.d': 2
    }
    """
    # 返回调用 _normalise_json_ordered 处理的结果
    return _normalise_json_ordered(ds, sep)
{
'nested.e.d': 2, \  # 定义一个字典项，键为'nested.e.d'，值为2，并使用反斜杠续行
'nested.d': 2\     # 续行后的另一个字典项，键为'nested.d'，值为2
}

"""
normalised_json_object = {}
# 普通化 JSON 对象初始化为空字典
# 预期输入为字典，因为大多数 JSON 是这种格式。但是列表也是完全有效的。
if isinstance(ds, dict):
    normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
elif isinstance(ds, list):
    # 如果输入是列表，则对每行调用 _simple_json_normalize 函数进行普通化
    normalised_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
    return normalised_json_list  # 返回普通化后的列表
return normalised_json_object  # 返回普通化后的字典对象


def json_normalize(
    data: dict | list[dict] | Series,  # 输入数据可以是字典、字典列表或者 Series
    record_path: str | list | None = None,  # 记录路径，用于指定记录的位置
    meta: str | list[str | list[str]] | None = None,  # 元数据，用于指定结果表中每个记录的元数据
    meta_prefix: str | None = None,  # 元数据前缀，如果指定，则在记录名前加上路径前缀
    record_prefix: str | None = None,  # 记录前缀，如果指定，则在记录名前加上路径前缀
    errors: IgnoreRaise = "raise",  # 错误处理选项，可以是'raise'或'ignore'
    sep: str = ".",  # 分隔符，用于嵌套记录的分隔符
    max_level: int | None = None,  # 最大归一化层级深度，如果为 None，则归一化所有层级
) -> DataFrame:
    """
    将半结构化的 JSON 数据规范化为平面表格。

    Parameters
    ----------
    data : dict, list of dicts, or Series of dicts
        未序列化的 JSON 对象。
    record_path : str or list of str, default None
        每个对象中记录列表的路径。如果未传递，则假定数据是记录数组。
    meta : list of paths (str or list of str), default None
        结果表中每个记录的元数据字段。
    meta_prefix : str, default None
        如果为 True，则使用点分隔路径前缀记录，例如如果 meta 是 ['foo'，'bar']，则为 foo.bar.field。
    record_prefix : str, default None
        如果为 True，则使用点分隔路径前缀记录，例如如果路径为 ['foo'，'bar']，则为 foo.bar.field。
    errors : {'raise', 'ignore'}, default 'raise'
        配置错误处理。

        * 'ignore'：如果元数据中列出的键不总是存在，则忽略 KeyError。
        * 'raise'：如果元数据中列出的键不总是存在，则引发 KeyError。
    sep : str, default '.'
        嵌套记录将生成以 sep 分隔的名称。
        例如，对于 sep='.'，{'foo': {'bar': 0}} -> foo.bar。
    max_level : int, default None
        最大归一化层级深度。
        如果为 None，则归一化所有层级。

    Returns
    -------
    frame : DataFrame
        将半结构化的 JSON 数据规范化为平面表格。

    Examples
    --------
    >>> data = [
    ...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    ...     {"name": {"given": "Mark", "family": "Regner"}},
    ...     {"id": 2, "name": "Faye Raker"},
    ... ]
    >>> pd.json_normalize(data)
        id name.first name.last name.given name.family        name
    0  1.0     Coleen      Volk        NaN         NaN         NaN
    1  NaN        NaN       NaN       Mark      Regner         NaN
    2  2.0        NaN       NaN        NaN         NaN  Faye Raker

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    >>> data = [
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    # 对给定的数据进行规范化，最大规范化级别为0（即只规范化顶层键）
    >>> pd.json_normalize(data, max_level=0)
        id        name                        fitness
    0  1.0   Cole Volk  {'height': 130, 'weight': 60}
    1  NaN    Mark Reg  {'height': 130, 'weight': 60}
    2  2.0  Faye Raker  {'height': 130, 'weight': 60}
    
    Normalizes nested data up to level 1.
    
    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    # 对给定的数据进行规范化，最大规范化级别为1（规范化到第一层嵌套）
    >>> pd.json_normalize(data, max_level=1)
        id        name  fitness.height  fitness.weight
    0  1.0   Cole Volk             130              60
    1  NaN    Mark Reg             130              60
    2  2.0  Faye Raker             130              60
    
    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    # 创建一个 Pandas Series 对象，使用指定的索引
    >>> series = pd.Series(data, index=pd.Index(["a", "b", "c"]))
    # 对 Series 中的 JSON 数据进行规范化处理
    >>> pd.json_normalize(series)
        id        name  fitness.height  fitness.weight
    a  1.0   Cole Volk             130              60
    b  NaN    Mark Reg             130              60
    c  2.0  Faye Raker             130              60
    
    >>> data = [
    ...     {
    ...         "state": "Florida",
    ...         "shortname": "FL",
    ...         "info": {"governor": "Rick Scott"},
    ...         "counties": [
    ...             {"name": "Dade", "population": 12345},
    ...             {"name": "Broward", "population": 40000},
    ...             {"name": "Palm Beach", "population": 60000},
    ...         ],
    ...     },
    ...     {
    ...         "state": "Ohio",
    ...         "shortname": "OH",
    ...         "info": {"governor": "John Kasich"},
    ...         "counties": [
    ...             {"name": "Summit", "population": 1234},
    ...             {"name": "Cuyahoga", "population": 1337},
    ...         ],
    ...     },
    ... ]
    # 对数据进行规范化，展开指定的嵌套结构，包括州、简称和州长信息
    >>> result = pd.json_normalize(
    ...     data, "counties", ["state", "shortname", ["info", "governor"]]
    ... )
    # 输出规范化后的数据结果
    >>> result
             name  population    state shortname info.governor
    0        Dade       12345  Florida       FL    Rick Scott
    1     Broward       40000  Florida       FL    Rick Scott
    """
    2  Palm Beach       60000   Florida    FL    Rick Scott
    3      Summit        1234   Ohio       OH    John Kasich
    4    Cuyahoga        1337   Ohio       OH    John Kasich
    """

    """
    >>> data = {"A": [1, 2]}
    >>> pd.json_normalize(data, "A", record_prefix="Prefix.")
        Prefix.0
    0          1
    1          2

    Returns normalized data with columns prefixed with the given string.
    """
    
    def _pull_field(
        js: dict[str, Any], spec: list | str, extract_record: bool = False
    ) -> Scalar | Iterable:
        """Internal function to pull field
        
        Args:
            js (dict[str, Any]): The JSON object to extract fields from.
            spec (list | str): Specification of the field path.
            extract_record (bool, optional): Flag indicating if extracting a record. Defaults to False.
        
        Returns:
            Scalar | Iterable: The extracted field value(s).
        
        Raises:
            KeyError: If a specified key is not found in the JSON object.
            TypeError: If the extracted result is not a list when expecting iterable data.
        """
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(
                    f"Key {e} not found. If specifying a record_path, all elements of "
                    f"data should have the path."
                ) from e
            if errors == "ignore":
                return np.nan
            else:
                raise KeyError(
                    f"Key {e} not found. To replace missing values of {e} with "
                    f"np.nan, pass in errors='ignore'"
                ) from e

        return result

    def _pull_records(js: dict[str, Any], spec: list | str) -> list:
        """
        Internal function to pull field for records, similar to _pull_field,
        but ensures the result is a list. Raises TypeError if the result is
        not iterable or null.
        
        Args:
            js (dict[str, Any]): The JSON object to extract records from.
            spec (list | str): Specification of the record path.
        
        Returns:
            list: The list of extracted records.
        
        Raises:
            KeyError: If a specified key is not found in the JSON object.
            TypeError: If the extracted result is not a list when expecting iterable data.
        """
        result = _pull_field(js, spec, extract_record=True)

        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(
                    f"Path must contain list or null, "
                    f"but got {type(result).__name__} at {spec!r}"
                )
        return result

    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if isinstance(data, list) and not data:
        return DataFrame()
    elif isinstance(data, dict):
        # A bit of a hackjob
        data = [data]
    elif isinstance(data, abc.Iterable) and not isinstance(data, str):
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        data = list(data)
    else:
        raise NotImplementedError

    # check to see if a simple recursive function is possible to
    # improve performance (see #15621) but only for cases such
    # as pd.Dataframe(data) or pd.Dataframe(data, sep)
    # 如果 record_path、meta、meta_prefix、record_prefix 和 max_level 都为 None，则使用 _simple_json_normalize 函数简单规范化数据，并返回一个 DataFrame 对象，使用 index 参数作为索引
    if (
        record_path is None
        and meta is None
        and meta_prefix is None
        and record_prefix is None
        and max_level is None
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep), index=index)

    # 如果 record_path 为 None，并且 data 中至少有一个值是字典类型，则执行简单规范化（nested_to_record），将嵌套的数据转换为扁平的记录形式，并返回一个 DataFrame 对象，使用 index 参数作为索引
    if record_path is None:
        if any([isinstance(x, dict) for x in y.values()] for y in data):
            # naive normalization, this is idempotent for flat records
            # and potentially will inflate the data considerably for
            # deeply nested structures:
            #  {VeryLong: { b: 1,c:2}} -> {VeryLong.b:1 ,VeryLong.c:@}
            #
            # TODO: handle record value which are lists, at least error
            #       reasonably
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data, index=index)

    # 如果 record_path 不为 None，但不是列表类型，则将其转换为单元素的列表
    elif not isinstance(record_path, list):
        record_path = [record_path]

    # 如果 meta 为 None，则将其初始化为空列表；如果 meta 不是列表类型，则将其转换为单元素的列表
    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]

    # 对 meta 中的每个元素，如果不是列表类型，则转换为单元素的列表
    _meta = [m if isinstance(m, list) else [m] for m in meta]

    # 初始化 records 列表和 lengths 列表
    records: list = []
    lengths = []

    # 使用 defaultdict 创建 meta_vals 字典，其中值为列表；meta_keys 是将 _meta 中的值连接起来的列表
    meta_vals: DefaultDict = defaultdict(list)
    meta_keys = [sep.join(val) for val in _meta]

    # 定义递归函数 _recursive_extract，用于从嵌套的数据中提取记录
    def _recursive_extract(data, path, seen_meta, level: int = 0) -> None:
        # 如果 data 是字典类型，则转换为包含该字典的列表
        if isinstance(data, dict):
            data = [data]
        # 如果 path 的长度大于 1，则继续递归提取数据
        if len(path) > 1:
            for obj in data:
                # 对于每个 obj 中的 val 和 meta_keys 进行处理，更新 seen_meta 字典
                for val, key in zip(_meta, meta_keys):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])

                # 递归调用 _recursive_extract，继续处理路径的下一级
                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data:
                # 提取 path[0] 对应的记录
                recs = _pull_records(obj, path[0])
                # 如果记录是字典类型，则使用 nested_to_record 进行简单规范化；否则直接使用该记录
                recs = [
                    nested_to_record(r, sep=sep, max_level=max_level)
                    if isinstance(r, dict)
                    else r
                    for r in recs
                ]

                # 记录长度，用于重复元数据
                lengths.append(len(recs))
                # 对于每个 obj 中的 val 和 meta_keys 进行处理，更新 meta_vals 字典
                for val, key in zip(_meta, meta_keys):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                # 将提取的记录添加到 records 列表中
                records.extend(recs)

    # 调用 _recursive_extract 函数，提取 data 中的记录，并初始化 seen_meta 为空字典
    _recursive_extract(data, record_path, {}, level=0)

    # 创建 DataFrame 对象，并使用 records 初始化
    result = DataFrame(records)

    # 如果指定了 record_prefix，则重命名 DataFrame 的列名，添加前缀
    if record_prefix is not None:
        result = result.rename(columns=lambda x: f"{record_prefix}{x}")

    # 数据类型处理，目前未实现
    `
        # 遍历元数据项字典，其中 k 是键，v 是对应的值
        for k, v in meta_vals.items():
            # 如果有元数据前缀，将键名加上前缀
            if meta_prefix is not None:
                k = meta_prefix + k
    
            # 检查结果字典中是否已存在同名的键，若存在则抛出数值错误异常
            if k in result:
                raise ValueError(
                    f"Conflicting metadata name {k}, need distinguishing prefix "
                )
            # GH 37782
    
            # 将值 v 转换为 NumPy 对象数组
            values = np.array(v, dtype=object)
    
            # 如果值的维度大于 1
            if values.ndim > 1:
                # GH 37782
                # 重新创建一个空的对象数组，长度为原始值 v 的长度
                values = np.empty((len(v),), dtype=object)
                # 遍历原始值 v，将每个值存入新创建的对象数组中
                for i, val in enumerate(v):
                    values[i] = val
    
            # 将处理后的值数组按指定长度重复，存入结果字典中对应的键 k
            result[k] = values.repeat(lengths)
    
        # 如果存在索引，则将索引按指定长度重复，并存入结果字典的索引属性中
        if index is not None:
            result.index = index.repeat(lengths)
    
        # 返回最终的结果字典
        return result
```