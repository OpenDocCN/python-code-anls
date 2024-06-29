# `D:\src\scipysrc\pandas\pandas\io\json\_table_schema.py`

```
"""
Table Schema builders

https://specs.frictionlessdata.io/table-schema/
"""

# 引入必要的模块和库
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

# 引入 pandas 库的内部模块
from pandas._libs import lib
from pandas._libs.json import ujson_loads
from pandas._libs.tslibs import timezones
from pandas.util._exceptions import find_stack_level

# 引入 pandas 核心数据类型相关模块和类
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)

# 引入 pandas DataFrame 类
from pandas import DataFrame
import pandas.core.common as com

# 引入 pandas 时序频率转换函数
from pandas.tseries.frequencies import to_offset

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        JSONSerializable,
    )

    from pandas import Series
    from pandas.core.indexes.multi import MultiIndex


# 定义当前模块的 Table Schema 版本
TABLE_SCHEMA_VERSION = "1.4.0"


# 将 NumPy / pandas 类型转换为对应的 json_table 类型的函数
def as_json_table_type(x: DtypeObj) -> str:
    """
    Convert a NumPy / pandas type to its corresponding json_table.

    Parameters
    ----------
    x : np.dtype or ExtensionDtype

    Returns
    -------
    str
        the Table Schema data types

    Notes
    -----
    This table shows the relationship between NumPy / pandas dtypes,
    and Table Schema dtypes.

    ==============  =================
    Pandas type     Table Schema type
    ==============  =================
    int64           integer
    float64         number
    bool            boolean
    datetime64[ns]  datetime
    timedelta64[ns] duration
    object          str
    categorical     any
    =============== =================
    """
    # 根据输入的数据类型判断并返回对应的 Table Schema 数据类型
    if is_integer_dtype(x):
        return "integer"
    elif is_bool_dtype(x):
        return "boolean"
    elif is_numeric_dtype(x):
        return "number"
    elif lib.is_np_dtype(x, "M") or isinstance(x, (DatetimeTZDtype, PeriodDtype)):
        return "datetime"
    elif lib.is_np_dtype(x, "m"):
        return "duration"
    elif isinstance(x, ExtensionDtype):
        return "any"
    elif is_string_dtype(x):
        return "string"
    else:
        return "any"


# 设置默认的索引名称函数
def set_default_names(data):
    """Sets index names to 'index' for regular, or 'level_x' for Multi"""
    # 如果数据的索引名称都不是 None，则直接返回数据
    if com.all_not_none(*data.index.names):
        nms = data.index.names
        # 如果只有一个索引且名称为 'index'，则发出警告
        if len(nms) == 1 and data.index.name == "index":
            warnings.warn(
                "Index name of 'index' is not round-trippable.",
                stacklevel=find_stack_level(),
            )
        # 如果有多个索引且有以 'level_' 开头的名称，则发出警告
        elif len(nms) > 1 and any(x.startswith("level_") for x in nms):
            warnings.warn(
                "Index names beginning with 'level_' are not round-trippable.",
                stacklevel=find_stack_level(),
            )
        return data

    # 深复制数据，以免修改原始数据
    data = data.copy(deep=False)
    # 如果数据有多层索引，则填充缺失的索引名称
    if data.index.nlevels > 1:
        data.index.names = com.fill_missing_names(data.index.names)
    else:
        # 如果索引名为空或未定义，则将索引名设置为默认值 "index"
        data.index.name = data.index.name or "index"
    # 返回处理后的数据对象
    return data
def convert_pandas_type_to_json_field(arr) -> dict[str, JSONSerializable]:
    # 获取数组的数据类型
    dtype = arr.dtype
    name: JSONSerializable
    # 如果数组没有命名，则使用默认名称 "values"
    if arr.name is None:
        name = "values"
    else:
        name = arr.name
    # 创建字段字典，包含字段名和字段类型
    field: dict[str, JSONSerializable] = {
        "name": name,
        "type": as_json_table_type(dtype),
    }

    # 根据数据类型的不同，添加额外的字段信息
    if isinstance(dtype, CategoricalDtype):
        cats = dtype.categories
        ordered = dtype.ordered

        # 如果是分类数据类型，添加枚举约束和有序信息
        field["constraints"] = {"enum": list(cats)}
        field["ordered"] = ordered
    elif isinstance(dtype, PeriodDtype):
        # 如果是周期数据类型，添加频率信息
        field["freq"] = dtype.freq.freqstr
    elif isinstance(dtype, DatetimeTZDtype):
        # 如果是带时区的日期时间数据类型
        if timezones.is_utc(dtype.tz):
            # 如果时区是 UTC，则设置为 "UTC"
            field["tz"] = "UTC"
        else:
            # 否则设置为具体的时区名称
            field["tz"] = dtype.tz.zone  # type: ignore[attr-defined]
    elif isinstance(dtype, ExtensionDtype):
        # 如果是扩展数据类型，添加扩展类型名称
        field["extDtype"] = dtype.name
    return field


def convert_json_field_to_pandas_type(field) -> str | CategoricalDtype:
    """
    Converts a JSON field descriptor into its corresponding NumPy / pandas type

    Parameters
    ----------
    field
        A JSON field descriptor

    Returns
    -------
    dtype

    Raises
    ------
    ValueError
        If the type of the provided field is unknown or currently unsupported

    Examples
    --------
    >>> convert_json_field_to_pandas_type({"name": "an_int", "type": "integer"})
    'int64'

    >>> convert_json_field_to_pandas_type(
    ...     {
    ...         "name": "a_categorical",
    ...         "type": "any",
    ...         "constraints": {"enum": ["a", "b", "c"]},
    ...         "ordered": True,
    ...     }
    ... )
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=True, categories_dtype=object)

    >>> convert_json_field_to_pandas_type({"name": "a_datetime", "type": "datetime"})
    'datetime64[ns]'

    >>> convert_json_field_to_pandas_type(
    ...     {"name": "a_datetime_with_tz", "type": "datetime", "tz": "US/Central"}
    ... )
    'datetime64[ns, US/Central]'
    """
    typ = field["type"]
    # 根据字段类型返回对应的 NumPy / pandas 数据类型字符串
    if typ == "string":
        return "object"
    elif typ == "integer":
        return field.get("extDtype", "int64")
    elif typ == "number":
        return field.get("extDtype", "float64")
    elif typ == "boolean":
        return field.get("extDtype", "bool")
    elif typ == "duration":
        return "timedelta64"
    elif typ == "datetime":
        # 如果是日期时间类型
        if field.get("tz"):
            # 如果有时区信息，则返回带时区的日期时间类型字符串
            return f"datetime64[ns, {field['tz']}]"
        elif field.get("freq"):
            # 如果有频率信息，则返回周期数据类型字符串
            offset = to_offset(field["freq"])
            freq = PeriodDtype(offset)._freqstr
            return f"period[{freq}]"
        else:
            # 否则返回标准的日期时间类型字符串
            return "datetime64[ns]"
    elif typ == "any":
        # 如果字段类型是 "any"，根据不同的情况返回对应的数据类型
        if "constraints" in field and "ordered" in field:
            # 如果字段包含约束并且是有序的，返回分类数据类型
            return CategoricalDtype(
                categories=field["constraints"]["enum"], ordered=field["ordered"]
            )
        elif "extDtype" in field:
            # 如果字段有扩展数据类型，通过注册表找到对应的数据类型
            return registry.find(field["extDtype"])
        else:
            # 否则返回默认的 "object" 类型
            return "object"

    # 如果字段类型不是 "any"，抛出异常
    raise ValueError(f"Unsupported or invalid field type: {typ}")
def build_table_schema(
    data: DataFrame | Series,
    index: bool = True,
    primary_key: bool | None = None,
    version: bool = True,
) -> dict[str, JSONSerializable]:
    """
    Create a Table schema from ``data``.

    Parameters
    ----------
    data : Series, DataFrame
        输入的数据，可以是 Series 或 DataFrame。
    index : bool, default True
        是否将数据的索引包含在 schema 中。
    primary_key : bool or None, default True
        设定作为主键的列名。
        默认值 `None` 将根据索引的唯一性设置 `'primaryKey'`。
    version : bool, default True
        是否包含字段 `pandas_version`，表示最后修改表结构的 pandas 版本。
        这个版本可能与当前安装的 pandas 版本不同。

    Returns
    -------
    dict
        返回一个包含表结构的字典。

    Notes
    -----
    参见 `Table Schema
    <https://pandas.pydata.org/docs/user_guide/io.html#table-schema>`__ 获取
    转换类型的详细信息。
    时间增量转换为 ISO8601 时长格式，秒字段后保留9位小数以实现纳秒精度。

    类别数据被转换为 `any` 类型，并使用 `enum` 字段约束列出允许的值。
    `ordered` 属性包含在 `ordered` 字段中。

    Examples
    --------
    >>> from pandas.io.json._table_schema import build_table_schema
    >>> df = pd.DataFrame(
    ...     {'A': [1, 2, 3],
    ...      'B': ['a', 'b', 'c'],
    ...      'C': pd.date_range('2016-01-01', freq='d', periods=3),
    ...      }, index=pd.Index(range(3), name='idx'))
    >>> build_table_schema(df)
    {'fields': \
[{'name': 'idx', 'type': 'integer'}, \
{'name': 'A', 'type': 'integer'}, \
{'name': 'B', 'type': 'string'}, \
{'name': 'C', 'type': 'datetime'}], \
'primaryKey': ['idx'], \
'pandas_version': '1.4.0'}
    """
    if index is True:
        # 如果需要包含索引，调用函数将数据的默认名称设置为索引名
        data = set_default_names(data)

    schema: dict[str, Any] = {}
    fields = []

    if index:
        if data.index.nlevels > 1:
            # 如果索引有多个级别，将每个级别转换为 JSON 字段
            data.index = cast("MultiIndex", data.index)
            for level, name in zip(data.index.levels, data.index.names):
                new_field = convert_pandas_type_to_json_field(level)
                new_field["name"] = name
                fields.append(new_field)
        else:
            # 否则，将单个索引转换为 JSON 字段
            fields.append(convert_pandas_type_to_json_field(data.index))

    if data.ndim > 1:
        # 如果数据有多列，遍历每列并将其转换为 JSON 字段
        for column, s in data.items():
            fields.append(convert_pandas_type_to_json_field(s))
    else:
        # 否则，将单列数据转换为 JSON 字段
        fields.append(convert_pandas_type_to_json_field(data))

    schema["fields"] = fields
    if index and data.index.is_unique and primary_key is None:
        if data.index.nlevels == 1:
            # 如果索引是唯一的且 primary_key 为 None，则将索引名作为主键
            schema["primaryKey"] = [data.index.name]
        else:
            # 否则，将索引的名称列表作为主键
            schema["primaryKey"] = data.index.names
    elif primary_key is not None:
        # 如果指定了 primary_key，则将其设置为主键
        schema["primaryKey"] = primary_key

    if version:
        # 如果需要版本信息，则添加 pandas_version 字段
        schema["pandas_version"] = TABLE_SCHEMA_VERSION

    return schema
    # 返回函数中定义的 schema 变量，结束函数并返回这个值
    return schema
def parse_table_schema(json, precise_float: bool) -> DataFrame:
    """
    Builds a DataFrame from a given schema

    Parameters
    ----------
    json :
        A JSON table schema
    precise_float : bool
        Flag controlling precision when decoding string to double values, as
        dictated by ``read_json``

    Returns
    -------
    df : DataFrame

    Raises
    ------
    NotImplementedError
        If the JSON table schema contains either timezone or timedelta data

    Notes
    -----
        Because :func:`DataFrame.to_json` uses the string 'index' to denote a
        name-less :class:`Index`, this function sets the name of the returned
        :class:`DataFrame` to ``None`` when said string is encountered with a
        normal :class:`Index`. For a :class:`MultiIndex`, the same limitation
        applies to any strings beginning with 'level_'. Therefore, an
        :class:`Index` name of 'index'  and :class:`MultiIndex` names starting
        with 'level_' are not supported.

    See Also
    --------
    build_table_schema : Inverse function.
    pandas.read_json
    """
    # Load JSON schema into a Python dictionary using ujson
    table = ujson_loads(json, precise_float=precise_float)
    
    # Extract column order from the schema
    col_order = [field["name"] for field in table["schema"]["fields"]]
    
    # Create a DataFrame from the data part of the JSON schema using extracted column order
    df = DataFrame(table["data"], columns=col_order)[col_order]

    # Determine data types for each column based on the schema
    dtypes = {
        field["name"]: convert_json_field_to_pandas_type(field)
        for field in table["schema"]["fields"]
    }

    # Check if there are columns with 'timedelta64' type, which is not supported
    if "timedelta64" in dtypes.values():
        raise NotImplementedError(
            'table="orient" can not yet read ISO-formatted Timedelta data'
        )

    # Convert DataFrame columns to specified data types
    df = df.astype(dtypes)

    # Set the DataFrame index if specified in the schema
    if "primaryKey" in table["schema"]:
        df = df.set_index(table["schema"]["primaryKey"])
        # Handle special cases where index names need adjustment
        if len(df.index.names) == 1:
            if df.index.name == "index":
                df.index.name = None
        else:
            df.index.names = [
                None if x.startswith("level_") else x for x in df.index.names
            ]

    # Return the constructed DataFrame
    return df
```