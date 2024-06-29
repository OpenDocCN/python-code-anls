# `D:\src\scipysrc\pandas\pandas\core\interchange\from_dataframe.py`

```
# 从未来版本导入类型提示的功能
from __future__ import annotations

# 导入 ctypes 库，用于处理 C 数据类型
import ctypes
# 导入 re 库，用于正则表达式操作
import re
# 导入 typing 库，用于类型提示
from typing import (
    Any,
    overload,
)

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 导入 pandas 的数据交换协议相关模块
from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    Column,
    ColumnNullType,
    DataFrame as DataFrameXchg,
    DtypeKind,
)
# 导入 pandas 的数据交换协议工具类
from pandas.core.interchange.utils import (
    ArrowCTypes,
    Endianness,
)

# 定义一个全局变量 _NP_DTYPES，用于存储不同数据类型及其对应的 numpy 类型映射关系
_NP_DTYPES: dict[DtypeKind, dict[int, Any]] = {
    DtypeKind.INT: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64},
    DtypeKind.UINT: {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64},
    DtypeKind.FLOAT: {32: np.float32, 64: np.float64},
    DtypeKind.BOOL: {1: bool, 8: bool},
}

# 定义函数 from_dataframe，从支持数据交换协议的 DataFrameXchg 对象构建 pd.DataFrame
def from_dataframe(df, allow_copy: bool = True) -> pd.DataFrame:
    """
    Build a ``pd.DataFrame`` from any DataFrame supporting the interchange protocol.

    Parameters
    ----------
    df : DataFrameXchg
        Object supporting the interchange protocol, i.e. `__dataframe__` method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pd.DataFrame

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
    # 如果 df 已经是 pd.DataFrame 类型，则直接返回
    if isinstance(df, pd.DataFrame):
        return df

    # 如果 df 没有 __dataframe__ 方法，则抛出 ValueError
    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    # 调用内部函数 _from_dataframe 进行数据转换，并返回结果
    return _from_dataframe(
        df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy
    )


# 定义内部函数 _from_dataframe，从 DataFrameXchg 对象构建 pd.DataFrame
def _from_dataframe(df: DataFrameXchg, allow_copy: bool = True) -> pd.DataFrame:
    """
    Build a ``pd.DataFrame`` from the DataFrame interchange object.

    Parameters
    ----------
    df : DataFrameXchg
        Object supporting the interchange protocol, i.e. `__dataframe__` method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pd.DataFrame
    """
    # 创建一个空列表 pandas_dfs，用于存储转换后的多个 pd.DataFrame
    pandas_dfs = []
    # 遍历 df 中的块数据，并将每个块转换为 pandas.DataFrame 存入 pandas_dfs
    for chunk in df.get_chunks():
        pandas_df = protocol_df_chunk_to_pandas(chunk)
        pandas_dfs.append(pandas_df)

    # 如果 allow_copy 为 False，且 pandas_dfs 中有多个元素，则抛出 RuntimeError
    if not allow_copy and len(pandas_dfs) > 1:
        raise RuntimeError(
            "To join chunks a copy is required which is forbidden by allow_copy=False"
        )
    # 如果 pandas_dfs 列表为空，则将 protocol_df_chunk_to_pandas 函数应用于 df，并将结果赋给 pandas_df
    if not pandas_dfs:
        pandas_df = protocol_df_chunk_to_pandas(df)
    # 如果 pandas_dfs 列表中只有一个 DataFrame，则直接将该 DataFrame 赋给 pandas_df
    elif len(pandas_dfs) == 1:
        pandas_df = pandas_dfs[0]
    # 如果 pandas_dfs 列表中有多个 DataFrame，则使用 pd.concat 连接它们，并将结果赋给 pandas_df
    else:
        pandas_df = pd.concat(pandas_dfs, axis=0, ignore_index=True, copy=False)

    # 从 df.metadata 中获取 "pandas.index" 键对应的值，并将其赋给 pandas_df 的索引
    index_obj = df.metadata.get("pandas.index", None)
    if index_obj is not None:
        pandas_df.index = index_obj

    # 返回处理后的 pandas_df 对象作为函数的结果
    return pandas_df
def protocol_df_chunk_to_pandas(df: DataFrameXchg) -> pd.DataFrame:
    """
    Convert interchange protocol chunk to ``pd.DataFrame``.

    Parameters
    ----------
    df : DataFrameXchg
        The interchange protocol chunk object.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame constructed from the interchange protocol chunk.
    """
    # We need a dict of columns here, with each column being a NumPy array (at
    # least for now, deal with non-NumPy dtypes later).
    columns: dict[str, Any] = {}
    buffers = []  # hold on to buffers, keeps memory alive
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f"Column {name} is not a string")
        if name in columns:
            raise ValueError(f"Column {name} is not unique")
        col = df.get_column_by_name(name)
        dtype = col.dtype[0]
        if dtype in (
            DtypeKind.INT,
            DtypeKind.UINT,
            DtypeKind.FLOAT,
            DtypeKind.BOOL,
        ):
            columns[name], buf = primitive_column_to_ndarray(col)
        elif dtype == DtypeKind.CATEGORICAL:
            columns[name], buf = categorical_column_to_series(col)
        elif dtype == DtypeKind.STRING:
            columns[name], buf = string_column_to_ndarray(col)
        elif dtype == DtypeKind.DATETIME:
            columns[name], buf = datetime_column_to_ndarray(col)
        else:
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        buffers.append(buf)

    pandas_df = pd.DataFrame(columns)
    pandas_df.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"] = buffers
    return pandas_df


def primitive_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    """
    Convert a column holding one of the primitive dtypes to a NumPy array.

    A primitive type is one of: int, uint, float, bool.

    Parameters
    ----------
    col : Column
        The column object containing the data to be converted.

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    buffers = col.get_buffers()

    data_buff, data_dtype = buffers["data"]
    data = buffer_to_ndarray(
        data_buff, data_dtype, offset=col.offset, length=col.size()
    )

    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
    """
    Convert a column holding categorical data to a pandas Series.

    Parameters
    ----------
    col : Column
        The column object containing the categorical data.

    Returns
    -------
    tuple
        Tuple of pd.Series holding the data and the memory owner object
        that keeps the memory alive.
    """
    categorical = col.describe_categorical

    if not categorical["is_dictionary"]:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")

    cat_column = categorical["categories"]
    # 检查是否具有属性 "_col"
    if hasattr(cat_column, "_col"):
        # 如果有 "_col" 属性，则将其转换为 NumPy 数组，忽略类型检查的联合属性警告
        categories = np.array(cat_column._col)  # type: ignore[union-attr]
    else:
        # 如果没有 "_col" 属性，则抛出 NotImplementedError 异常
        raise NotImplementedError(
            "Interchanging categorical columns isn't supported yet, and our "
            "fallback of using the `col._col` attribute (a ndarray) failed."
        )
    # 获取列的缓冲区
    buffers = col.get_buffers()

    # 获取数据缓冲区和数据类型
    codes_buff, codes_dtype = buffers["data"]
    # 将数据缓冲区转换为 NumPy 数组
    codes = buffer_to_ndarray(
        codes_buff, codes_dtype, offset=col.offset, length=col.size()
    )

    # 对 codes 取模以避免索引错误
    if len(categories) > 0:
        # 使用 categories 数组和 codes 数组生成 values 数组
        values = categories[codes % len(categories)]
    else:
        # 如果 categories 数组为空，则直接使用 codes 数组
        values = codes

    # 创建 Pandas 的分类数据类型
    cat = pd.Categorical(
        values, categories=categories, ordered=categorical["is_ordered"]
    )
    # 创建 Pandas 的系列数据
    data = pd.Series(cat)

    # 将空值设置为 null
    data = set_nulls(data, col, buffers["validity"])
    # 返回处理后的数据和缓冲区
    return data, buffers
def string_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    """
    Convert a column holding string data to a NumPy array.

    Parameters
    ----------
    col : Column
        The input column object containing string data.

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    # Get the description of null values from the column
    null_kind, sentinel_val = col.describe_null

    # Check if the null kind is supported for string columns
    if null_kind not in (
        ColumnNullType.NON_NULLABLE,
        ColumnNullType.USE_BITMASK,
        ColumnNullType.USE_BYTEMASK,
    ):
        raise NotImplementedError(
            f"{null_kind} null kind is not yet supported for string columns."
        )

    # Retrieve data and offsets buffers from the column
    buffers = col.get_buffers()

    # Ensure the offsets buffer is present, as it's necessary for string columns
    assert buffers["offsets"], "String buffers must contain offsets"

    # Retrieve the data buffer containing the UTF-8 code units
    data_buff, _ = buffers["data"]

    # Assert that the data buffer format is compatible with UTF-8
    assert col.dtype[2] in (
        ArrowCTypes.STRING,
        ArrowCTypes.LARGE_STRING,
    )  # format_str == utf-8

    # Convert the data buffer to a NumPy array interpreted as uint8 (byte array)
    data_dtype = (
        DtypeKind.UINT,
        8,
        ArrowCTypes.UINT8,
        Endianness.NATIVE,
    )
    data = buffer_to_ndarray(data_buff, data_dtype, offset=0, length=data_buff.bufsize)

    # Retrieve the offsets buffer containing the start-stop positions of strings
    offset_buff, offset_dtype = buffers["offsets"]

    # Calculate the proper size for the offsets buffer based on the column size
    offsets = buffer_to_ndarray(
        offset_buff, offset_dtype, offset=col.offset, length=col.size() + 1
    )

    # Handle null positions if bitmask or bytemask is used
    null_pos = None
    if null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        validity = buffers["validity"]
        if validity is not None:
            valid_buff, valid_dtype = validity
            null_pos = buffer_to_ndarray(
                valid_buff, valid_dtype, offset=col.offset, length=col.size()
            )
            if sentinel_val == 0:
                null_pos = ~null_pos

    # Initialize a list to store strings after assembly from code units
    str_list: list[None | float | str] = [None] * col.size()
    # 遍历列的索引范围，从0到col.size()-1
    for i in range(col.size()):
        # 检查是否存在缺失值，并且在null_pos中的标志为True
        if null_pos is not None and null_pos[i]:
            # 如果存在缺失值，则将str_list中对应位置设为NaN
            str_list[i] = np.nan
            # 跳过当前循环，继续下一个i的迭代
            continue

        # 从data中提取代码单元的范围，从offsets[i]到offsets[i+1]
        units = data[offsets[i] : offsets[i + 1]]

        # 将代码单元列表转换为字节序列
        str_bytes = bytes(units)

        # 使用utf-8解码字节序列生成字符串
        string = str_bytes.decode(encoding="utf-8")

        # 将生成的字符串添加到str_list中的对应位置
        str_list[i] = string

    # 将字符串列表转换为NumPy数组，数据类型为object，返回结果和buffers
    return np.asarray(str_list, dtype="object"), buffers
def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
    """Parse datetime `format_str` to interpret the `data`."""
    # 解析时间戳格式字符串 'ts{unit}:tz'
    timestamp_meta = re.match(r"ts([smun]):(.*)", format_str)
    if timestamp_meta:
        unit, tz = timestamp_meta.group(1), timestamp_meta.group(2)
        if unit != "s":
            # 格式字符串仅描述单位的首字母，因此添加一个额外字母以转换为numpy风格：
            # 'm' -> 'ms', 'u' -> 'us', 'n' -> 'ns'
            unit += "s"
        data = data.astype(f"datetime64[{unit}]")
        if tz != "":
            data = pd.Series(data).dt.tz_localize("UTC").dt.tz_convert(tz)
        return data

    # 解析日期格式字符串 'td{Days/Ms}'
    date_meta = re.match(r"td([Dm])", format_str)
    if date_meta:
        unit = date_meta.group(1)
        if unit == "D":
            # NumPy不支持DAY单位，因此将天数转换为秒
            # （转换为uint64以避免溢出）
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
        elif unit == "m":
            data = data.astype("datetime64[ms]")
        else:
            raise NotImplementedError(f"不支持的日期单位：{unit}")
        return data

    # 抛出未实现错误，表示不支持该日期时间类型
    raise NotImplementedError(f"不支持的日期时间类型：{format_str}")


def datetime_column_to_ndarray(col: Column) -> tuple[np.ndarray | pd.Series, Any]:
    """
    Convert a column holding DateTime data to a NumPy array.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        包含数据的np.ndarray和保持内存的对象所有者。
    """
    buffers = col.get_buffers()

    _, col_bit_width, format_str, _ = col.dtype
    dbuf, _ = buffers["data"]
    # 考虑dtype为`uint`以获取从1970年1月1日以来传递的单位数

    data = buffer_to_ndarray(
        dbuf,
        (
            DtypeKind.INT,
            col_bit_width,
            getattr(ArrowCTypes, f"INT{col_bit_width}"),
            Endianness.NATIVE,
        ),
        offset=col.offset,
        length=col.size(),
    )

    data = parse_datetime_format_str(format_str, data)  # type: ignore[assignment]
    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def buffer_to_ndarray(
    buffer: Buffer,
    dtype: tuple[DtypeKind, int, str, str],
    *,
    length: int,
    offset: int = 0,
) -> np.ndarray:
    """
    Build a NumPy array from the passed buffer.

    Parameters
    ----------
    buffer : Buffer
        Buffer to build a NumPy array from.
    dtype : tuple
        Data type of the buffer conforming protocol dtypes format.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    length : int, optional
        If the buffer is a bit-mask, specifies a number of bits to read
        from the buffer. Has no effect otherwise.

    Returns
    -------
    np.ndarray
        NumPy array built from the buffer.
    """
    # 从 dtype 中解包出 kind、bit_width 信息
    kind, bit_width, _, _ = dtype

    # 根据 kind 和 bit_width 获取对应的 NumPy 数据类型
    column_dtype = _NP_DTYPES.get(kind, {}).get(bit_width, None)
    # 如果未找到对应的数据类型，抛出未实现错误
    if column_dtype is None:
        raise NotImplementedError(f"Conversion for {dtype} is not yet supported.")

    # 如果数据宽度为 1，处理位掩码缓冲区
    if bit_width == 1:
        # 确保提供了 length 参数
        assert length is not None, "`length` must be specified for a bit-mask buffer."
        # 导入 pyarrow 库作为可选依赖
        pa = import_optional_dependency("pyarrow")
        # 从缓冲区创建布尔数组
        arr = pa.BooleanArray.from_buffers(
            pa.bool_(),
            length,
            [None, pa.foreign_buffer(buffer.ptr, length)],
            offset=offset,
        )
        # 将 pyarrow 数组转换为 NumPy 数组并返回
        return np.asarray(arr)
    else:
        # 计算数据指针的位置
        data_pointer = ctypes.cast(
            buffer.ptr + (offset * bit_width // 8), ctypes.POINTER(ctypes_type)
        )
        # 如果 length 大于 0，将数据指针转换为 NumPy 数组并返回
        if length > 0:
            return np.ctypeslib.as_array(data_pointer, shape=(length,))
        # 如果 length 不大于 0，返回一个空的 NumPy 数组
        return np.array([], dtype=ctypes_type)
@overload
def set_nulls(
    data: np.ndarray,
    col: Column,
    validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None,
    allow_modify_inplace: bool = ...,
) -> np.ndarray: ...

# 设置空值函数的重载定义，用于处理NumPy数组作为输入的情况。

@overload
def set_nulls(
    data: pd.Series,
    col: Column,
    validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None,
    allow_modify_inplace: bool = ...,
) -> pd.Series: ...

# 设置空值函数的重载定义，用于处理Pandas Series作为输入的情况。

@overload
def set_nulls(
    data: np.ndarray | pd.Series,
    col: Column,
    validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None,
    allow_modify_inplace: bool = ...,
) -> np.ndarray | pd.Series: ...

# 设置空值函数的重载定义，用于处理既可以是NumPy数组也可以是Pandas Series作为输入的情况。

def set_nulls(
    data: np.ndarray | pd.Series,
    col: Column,
    validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None,
    allow_modify_inplace: bool = True,
) -> np.ndarray | pd.Series:
    """
    Set null values for the data according to the column null kind.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        数据，需要设置空值。
    col : Column
        描述数据的列对象。
    validity : tuple(Buffer, dtype) or None
        ``col.buffers()`` 的返回值。此处不访问 ``col.buffers()``，
        以避免获取缓冲区对象的内存所有权。
    allow_modify_inplace : bool, default: True
        是否在可能时就地修改 `data`（True），或始终修改 `data` 的副本（False）。

    Returns
    -------
    np.ndarray or pd.Series
        设置了空值的数据。
    """
    if validity is None:
        return data
    null_kind, sentinel_val = col.describe_null
    null_pos = None

    if null_kind == ColumnNullType.USE_SENTINEL:
        null_pos = pd.Series(data) == sentinel_val
    elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        assert validity, "Expected to have a validity buffer for the mask"
        valid_buff, valid_dtype = validity
        null_pos = buffer_to_ndarray(
            valid_buff, valid_dtype, offset=col.offset, length=col.size()
        )
        if sentinel_val == 0:
            null_pos = ~null_pos
    elif null_kind in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_NAN):
        pass
    else:
        raise NotImplementedError(f"Null kind {null_kind} is not yet supported.")

    if null_pos is not None and np.any(null_pos):
        if not allow_modify_inplace:
            data = data.copy()
        try:
            data[null_pos] = None
        except TypeError:
            # 如果 `data` 的数据类型在NumPy标记中似乎是不可空的（bool、int、uint），
            # 则会出现TypeError。如果出现这种情况，将 `data` 转换为可空的浮点数据类型。
            data = data.astype(float)
            data[null_pos] = None

    return data
```