# `D:\src\scipysrc\pandas\pandas\io\sql.py`

```
# -----------------------------------------------------------------------------
# -- Helper functions

# 处理 parse_dates 参数，用于 read_sql 系列函数
def _process_parse_dates_argument(parse_dates):
    # 如果 parse_dates 是 True、None 或 False，则设为空列表
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []

    # 如果 parse_dates 不可迭代，则转为列表
    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    return parse_dates


# 处理日期列，支持处理时区和自定义格式
def _handle_date_column(
    col, utc: bool = False, format: str | dict[str, Any] | None = None
):
    if isinstance(format, dict):
        # 如果 format 是字典，则允许在 parse_dates 参数中使用自定义的 to_datetime 参数值
        # 如 {"errors": "coerce"} 或 {"dayfirst": True}
        return to_datetime(col, **format)


这段代码包含了两个函数的定义和相关的注释。
    else:
        # 如果 format 参数为 None 并且列的数据类型是浮点数或整数
        # GH17855
        if format is None and (
            issubclass(col.dtype.type, np.floating)
            or issubclass(col.dtype.type, np.integer)
        ):
            # 设置 format 为字符串 "s"
            format = "s"
        
        # 如果 format 参数在 ["D", "d", "h", "m", "s", "ms", "us", "ns"] 中
        if format in ["D", "d", "h", "m", "s", "ms", "us", "ns"]:
            # 调用 to_datetime 函数转换列为日期时间对象，设定错误处理为 "coerce"，单位为 format，UTC 时区为 utc
            return to_datetime(col, errors="coerce", unit=format, utc=utc)
        
        # 如果列的数据类型是 DatetimeTZDtype 类型
        elif isinstance(col.dtype, DatetimeTZDtype):
            # 强制转换为 UTC 时区
            # GH11216
            return to_datetime(col, utc=True)
        
        # 对于其他情况
        else:
            # 调用 to_datetime 函数转换列为日期时间对象，设定错误处理为 "coerce"，格式为 format，UTC 时区为 utc
            return to_datetime(col, errors="coerce", format=format, utc=utc)
def _parse_date_columns(data_frame: DataFrame, parse_dates) -> DataFrame:
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    # 处理解析日期参数，确保它是一个字典
    parse_dates = _process_parse_dates_argument(parse_dates)

    # 遍历数据帧的每一列
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        # 如果列的类型是 DatetimeTZDtype 或者列名在 parse_dates 中
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                # 尝试获取指定列的日期格式
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            # 处理日期列，并将处理后的结果设置回数据帧
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))

    # 返回处理后的数据帧
    return data_frame


def _convert_arrays_to_dataframe(
    data,
    columns,
    coerce_float: bool = True,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    # 将数据转换为对象数组的元组形式
    content = lib.to_object_array_tuples(data)
    idx_len = content.shape[0]
    # 转换对象数组为数据帧的数组形式
    arrays = convert_object_array(
        list(content.T),
        dtype=None,
        coerce_float=coerce_float,
        dtype_backend=dtype_backend,
    )
    # 如果使用的是 pyarrow 后端
    if dtype_backend == "pyarrow":
        # 导入 pyarrow 作为可选依赖
        pa = import_optional_dependency("pyarrow")

        result_arrays = []
        # 遍历数组并转换为 ArrowExtensionArray
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                # 如果数组类型是字符串，将其转换为 pyarrow 的字符串类型
                # TODO: Arrow 仍然将字符串数组推断为常规字符串，而不是 large_string，
                # 这在 dtype_backend="pyarrow" 下的处理方式需要重新考虑
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        # 将处理后的结果重新赋值给 arrays
        arrays = result_arrays  # type: ignore[assignment]

    # 如果存在数组，则构造并返回数据帧
    if arrays:
        return DataFrame._from_arrays(
            arrays, columns=columns, index=range(idx_len), verify_integrity=False
        )
    else:
        # 如果数组为空，则返回一个空的数据帧
        return DataFrame(columns=columns)


def _wrap_result(
    data,
    columns,
    index_col=None,
    coerce_float: bool = True,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    # 将结果数据转换为数据帧
    frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)

    # 如果指定了 dtype，则将数据帧转换为指定的 dtype
    if dtype:
        frame = frame.astype(dtype)

    # 处理数据帧中的日期列
    frame = _parse_date_columns(frame, parse_dates)

    # 如果指定了 index_col，则设置数据帧的索引列
    if index_col is not None:
        frame = frame.set_index(index_col)

    # 返回处理后的数据帧
    return frame


def _wrap_result_adbc(
    df: DataFrame,
    *,
    index_col=None,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    # 如果指定了 dtype，则将数据帧转换为指定的 dtype
    if dtype:
        df = df.astype(dtype)

    # 处理数据帧中的日期列
    df = _parse_date_columns(df, parse_dates)

    # 返回处理后的数据帧
    return df
    # 如果指定了 index_col 参数，将 DataFrame 的索引设置为 index_col 列的值
    if index_col is not None:
        df = df.set_index(index_col)

    # 返回处理后的 DataFrame 对象
    return df
# -----------------------------------------------------------------------------
# -- Read and write to DataFrames

# 函数重载，用于从 SQL 表中读取数据到 DataFrame

@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    columns: list[str] | None = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame: ...


@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    columns: list[str] | None = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]: ...


def read_sql_table(
    table_name: str,
    con,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list[str] | dict[str, str] | None = None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
    """
    # dtype_backend 参数用于指定返回的 DataFrame 的数据类型后端，可以是 'numpy_nullable' 或 'pyarrow'
    # 默认为 'numpy_nullable'。这是一个实验性功能，行为如下:
    #   - "numpy_nullable": 返回由可空数据类型支持的 DataFrame（默认）。
    #   - "pyarrow": 返回由 pyarrow 支持的可空 ArrowDtype 的 DataFrame。
    # 这个功能是在 2.0 版本中添加的。

    # 返回一个 DataFrame 或者 DataFrame 的迭代器
    # 表示一个 SQL 表作为带有标签轴的二维数据结构。

    # 查看也可以参考以下方法：
    #   - read_sql_query : 从 SQL 查询中读取数据到 DataFrame。
    #   - read_sql : 从 SQL 查询或数据库表中读取数据到 DataFrame。

    # 注意：
    #   所有带有时区信息的日期时间值将被转换为 UTC。

    # 示例：
    # >>> pd.read_sql_table("table_name", "postgres:///db_name")  # doctest:+SKIP
    ```

    # 检查 dtype_backend 是否有效
    check_dtype_backend(dtype_backend)
    # 如果 dtype_backend 为 lib.no_default，则将其赋值为 "numpy"（类型忽略这个赋值）
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    # 确保 dtype_backend 不为 lib.no_default
    assert dtype_backend is not lib.no_default

    # 使用 pandasSQL_builder 创建一个连接，并且需要在事务中操作
    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        # 如果表不存在于数据库中，抛出 ValueError 异常
        if not pandas_sql.has_table(table_name):
            raise ValueError(f"Table {table_name} not found")

        # 从 pandas_sql 中读取表格数据
        table = pandas_sql.read_table(
            table_name,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    # 如果成功读取表格数据，则返回 table；否则抛出 ValueError 异常
    if table is not None:
        return table
    else:
        raise ValueError(f"Table {table_name} not found", con)
@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame: ...

# 函数重载声明，用于从 SQL 查询读取数据并返回 DataFrame 类型的结果。

@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    chunksize: int = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]: ...

# 函数重载声明，用于从 SQL 查询读取数据并返回 DataFrame 类型的结果的迭代器。

def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list[Any] | Mapping[str, Any] | None = None,
    parse_dates: list[str] | dict[str, str] | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query into a DataFrame.

    将 SQL 查询结果读取为 DataFrame。

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    返回与查询结果对应的 DataFrame。可选地提供 `index_col` 参数来将某一列作为索引，否则将使用默认的整数索引。

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
        
        要执行的 SQL 查询。

    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
        
        使用 SQLAlchemy 可以使用该库支持的任何数据库。如果是 DBAPI2 对象，只支持 sqlite3。

    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
        
        要设置为索引的列（多重索引）。

    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
        
        尝试将非字符串、非数值对象（如 decimal.Decimal）的值转换为浮点数。对 SQL 结果集非常有用。

    params : list, tuple or mapping, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
        
        要传递给 execute 方法的参数列表。传递参数的语法取决于数据库驱动程序。查阅数据库驱动程序文档，了解支持的五种参数传递风格之一（在 PEP 249 的 paramstyle 中描述）。

        例如，对于 psycopg2，使用 %(name)s，所以使用 params={'name' : 'value'}。

    parse_dates : list of str or dict of {column_name: format string}, default: None
        - List of column names to parse as dates.
        - Dict of {column_name: format string} where format string is strftime compatible in case of parsing string times or is one of (D, s, ns, ms, us) in case of parsing integer timestamps.
        
        - 要解析为日期的列名列表。
        - {列名：格式字符串} 的字典，其中格式字符串在解析字符串时间时兼容 strftime，或在解析整数时间戳时为 (D, s, ns, ms, us) 之一。

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number of rows to include in each chunk.
        
        如果指定，返回一个迭代器，其中 chunksize 是每个块中包含的行数。

    dtype : dict or scalar, optional, default: None
        - Dict of column name to dtype. Use a string format for SQL or NoSQL text-based databases.
        - 请参阅 sqlalchemy.engine.json.JSON.JSON . read , specify also used for default converts missing drivers .
        use integers.
    """
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.
    
        .. versionadded:: 1.3.0
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:
    
        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.
    
        .. versionadded:: 2.0
    
    Returns
    -------
    DataFrame or Iterator[DataFrame]
        Returns a DataFrame object that contains the result set of the
        executed SQL query, in relation to the specified database connection.
    
    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.
    
    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    
    Examples
    --------
    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("sqlite:///database.db")  # doctest: +SKIP
    >>> sql_query = "SELECT int_column FROM test_data"  # doctest: +SKIP
    >>> with engine.connect() as conn, conn.begin():  # doctest: +SKIP
    ...     data = pd.read_sql_query(sql_query, conn)  # doctest: +SKIP
    """
    
    check_dtype_backend(dtype_backend)
    # 检查并确保 dtype_backend 参数有效
    if dtype_backend is lib.no_default:
        # 如果 dtype_backend 未设置，默认使用 "numpy"
        dtype_backend = "numpy"  # type: ignore[assignment]
    # 确保 dtype_backend 参数不为 lib.no_default
    assert dtype_backend is not lib.no_default
    
    # 使用 pandasSQL_builder 构建 SQL 查询执行环境
    with pandasSQL_builder(con) as pandas_sql:
        # 调用 pandas_sql 的 read_query 方法执行 SQL 查询并返回结果
        return pandas_sql.read_query(
            sql,
            index_col=index_col,
            params=params,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            chunksize=chunksize,
            dtype=dtype,
            dtype_backend=dtype_backend,
        )
@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> DataFrame: ...

这是一个函数重载的声明，用于从 SQL 查询或数据库表中读取数据到 DataFrame 中。


@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> Iterator[DataFrame]: ...

这是另一个函数重载的声明，返回一个 DataFrame 的迭代器，用于从 SQL 查询或数据库表中批量读取数据。


def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params=None,
    parse_dates=None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    dtype: DtypeArg | None = None,
) -> DataFrame | Iterator[DataFrame]:

实际的函数定义，用于从 SQL 查询或数据库表中读取数据到 DataFrame 中。


"""
Read SQL query or database table into a DataFrame.

This function is a convenience wrapper around ``read_sql_table`` and
``read_sql_query`` (for backward compatibility). It will delegate
to the specific function depending on the provided input. A SQL query
will be routed to ``read_sql_query``, while a database table name will
be routed to ``read_sql_table``. Note that the delegated function might
have more specific notes about their functionality not listed here.

Parameters
----------
sql : str or SQLAlchemy Selectable (select or text object)
    SQL query to be executed or a table name.
con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
    ADBC provides high performance I/O with native type support, where available.
    Using SQLAlchemy makes it possible to use any DB supported by that
    library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
    for engine disposal and connection closure for the ADBC connection and
    SQLAlchemy connectable; str connections are closed automatically. See
    `here <https://docs.sqlalchemy.org/en/20/core/connections.html>`_.
index_col : str or list of str, optional, default: None
    Column(s) to set as index(MultiIndex).
coerce_float : bool, default True
    Attempts to convert values of non-string, non-numeric objects (like
    decimal.Decimal) to floating point, useful for SQL result sets.
params : list, tuple or dict, optional, default: None
    List of parameters to pass to execute method.  The syntax used
    to pass parameters is database driver dependent. Check your
    database driver documentation for which of the five syntax styles,
    described in PEP 249's paramstyle, is supported.
    Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.

这是函数的文档字符串，提供了函数的详细说明、参数说明和使用注意事项。
    parse_dates : list or dict, default: None
        # 指定需要解析为日期的列名列表或字典
        - List of column names to parse as dates.
        # 列名列表，用于解析为日期。
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        # 字典，格式为 ``{列名: 格式字符串}``，格式字符串符合 strftime 规范，用于解析字符串时间，或者是 (D, s, ns, ms, us) 之一，用于解析整数时间戳。
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
        # 字典，``{列名: 参数字典}``，参数字典对应于 :func:`pandas.to_datetime` 的关键字参数
          Especially useful with databases without native Datetime support,
          such as SQLite.
        # 特别适用于不支持原生日期时间的数据库，如 SQLite。
    columns : list, default: None
        # 从 SQL 表中选择的列名列表（仅在读取表时使用）
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        # 如果指定，返回一个迭代器，其中 `chunksize` 是每个块中包含的行数。
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        # 应用于结果 :class:`DataFrame` 的后端数据类型（仍处于实验阶段）。
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0
    dtype : Type name or dict of columns
        # 数据或列的数据类型。例如 np.float64 或 {'a': np.float64, 'b': np.int32, 'c': 'Int64'}。
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.
        The argument is ignored if a table is passed instead of a query.

        .. versionadded:: 2.0.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        # 返回一个包含执行的 SQL 查询结果集或提供的输入关联的 SQL 表的 :class:`DataFrame` 对象。
        Returns a DataFrame object that contains the result set of the
        executed SQL query or an SQL Table based on the provided input,
        in relation to the specified database connection.

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Notes
    -----
    ``pandas`` does not attempt to sanitize SQL statements;
    instead it simply forwards the statement you are executing
    to the underlying driver, which may or may not sanitize from there.
    Please refer to the underlying driver documentation for any details.
    Generally, be wary when accepting statements from arbitrary sources.

    Examples
    --------
    Read data from SQL via either a SQL query or a SQL tablename.
    When using a SQLite database only SQL queries are accepted,
    providing only the SQL tablename will result in an error.

    >>> from sqlite3 import connect
    >>> conn = connect(":memory:")
    >>> df = pd.DataFrame(
    ...     data=[[0, "10/11/12"], [1, "12/11/10"]],
    ...     columns=["int_column", "date_column"],
    ... )
    >>> df.to_sql(name="test_data", con=conn)
    2

    >>> pd.read_sql("SELECT int_column, date_column FROM test_data", conn)
       int_column date_column
    0           0    10/11/12
    1           1    12/11/10
    # 跳过该语句的文档测试，不执行该代码段
    >>> pd.read_sql("test_data", "postgres:///db_name")  # doctest:+SKIP

    # 对于参数化查询，推荐使用 params 而不是字符串插值
    For parameterized query, using ``params`` is recommended over string interpolation.

    # 导入 SQLAlchemy 的 text 类
    >>> from sqlalchemy import text
    # 创建 SQL 查询文本对象
    >>> sql = text(
    ...     "SELECT int_column, date_column FROM test_data WHERE int_column=:int_val"
    ... )
    # 使用参数化查询执行 SQL 查询，并传入参数 {"int_val": 1} 来填充 int_val 的值
    >>> pd.read_sql(sql, conn, params={"int_val": 1})  # doctest:+SKIP
       int_column date_column
    0           1    12/11/10

    # 通过 parse_dates 参数对列应用日期解析
    Apply date parsing to columns through the ``parse_dates`` argument
    # parse_dates 参数调用 pd.to_datetime 对提供的列进行日期解析
    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.
    # 通过字典格式指定在列上应用 pd.to_datetime 的自定义参数值
    Custom argument values for applying ``pd.to_datetime`` on a column are specified
    via a dictionary format:
    >>> pd.read_sql(
    ...     "SELECT int_column, date_column FROM test_data",
    ...     conn,
    ...     parse_dates={"date_column": {"format": "%d/%m/%y"}},
    ... )
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12

    # 版本添加说明：从 2.2.0 版本开始
    .. versionadded:: 2.2.0
    pandas now supports reading via ADBC drivers

    # 导入 ADBC PostgreSQL 驱动的数据库接口
    >>> from adbc_driver_postgresql import dbapi  # doctest:+SKIP
    # 使用 dbapi.connect 连接到名为 "db_name" 的 PostgreSQL 数据库
    >>> with dbapi.connect("postgres:///db_name") as conn:  # doctest:+SKIP
    ...     # 从数据库中读取 int_column 列的数据
    ...     pd.read_sql("SELECT int_column FROM test_data", conn)
       int_column
    0           0
    1           1
    """

    # 检查 dtype_backend 的数据类型后端
    check_dtype_backend(dtype_backend)
    # 如果 dtype_backend 为 lib.no_default，则将其设为 "numpy"
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    # 确保 dtype_backend 不是 lib.no_default
    assert dtype_backend is not lib.no_default

    # 使用 pandasSQL_builder 创建 pandas_sql 对象
    with pandasSQL_builder(con) as pandas_sql:
        # 如果 pandas_sql 是 SQLiteDatabase 类型的实例
        if isinstance(pandas_sql, SQLiteDatabase):
            # 返回从数据库中读取的查询结果
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )

        try:
            # 检查 SQL 查询中是否存在表名
            _is_table_name = pandas_sql.has_table(sql)
        except Exception:
            # 使用通用异常捕获 SQL 驱动程序的错误（GH24988）
            _is_table_name = False

        # 如果存在表名
        if _is_table_name:
            # 从表中读取数据
            return pandas_sql.read_table(
                sql,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
            )
        else:
            # 否则返回查询结果
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )
def to_sql(
    frame,
    name: str,
    con,
    schema: str | None = None,
    if_exists: Literal["fail", "replace", "append"] = "fail",
    index: bool = True,
    index_label: IndexLabel | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    method: Literal["multi"] | Callable | None = None,
    engine: str = "auto",
    **engine_kwargs,
) -> int | None:
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    frame : DataFrame, Series
        要写入 SQL 数据库的数据，可以是 DataFrame 或 Series。
    name : str
        SQL 表的名称。
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        or sqlite3 DBAPI2 connection
        数据库连接，可以是 ADBC 连接，SQLAlchemy 连接对象，字符串或 sqlite3 连接对象。
        ADBC 提供原生类型支持的高性能 I/O，如果使用 SQLAlchemy，可以使用该库支持的任何数据库。
        如果是 DBAPI2 对象，仅支持 sqlite3。
    schema : str, optional
        数据库中要写入的 SQL 模式名称（如果数据库支持）。如果为 None，则使用默认模式（默认值）。
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        - 'fail'：如果表已存在，则什么也不做。
        - 'replace'：如果表已存在，则删除并重新创建表，并插入数据。
        - 'append'：如果表已存在，则插入数据；如果不存在，则创建新表。
    index : bool, default True
        是否将 DataFrame 索引写为列。
    index_label : str or sequence, optional
        索引列的列标签。如果给定 None（默认值），并且 `index` 为 True，则使用索引名称。
        如果 DataFrame 使用 MultiIndex，则应提供一个序列。
    chunksize : int, optional
        指定每批次写入的行数。默认情况下，将一次性写入所有行。
    dtype : dict or scalar, optional
        指定列的数据类型。如果使用字典，则键应为列名，值应为 SQLAlchemy 类型或 sqlite3 回退模式的字符串。
        如果提供标量，则将其应用于所有列。
    method : {None, 'multi', callable}, optional
        控制使用的 SQL 插入子句：
        - None：使用标准的 SQL ``INSERT`` 子句（每行一个）。
        - ``'multi'``：在单个 ``INSERT`` 子句中传递多个值。
        - 可调用对象，签名为 ``(pd_table, conn, keys, data_iter) -> int | None``。
    engine : {'auto', 'sqlalchemy'}, default 'auto'
        要使用的 SQL 引擎库。如果为 'auto'，则使用选项 ``io.sql.engine``。默认 ``io.sql.engine`` 行为是 'sqlalchemy'。
    **engine_kwargs
        传递给引擎的任何额外关键字参数。

    Returns
    -------
    int | None
        返回插入的行数或 None。
    """
    None or int
        Number of rows affected by to_sql. None is returned if the callable
        passed into ``method`` does not return an integer number of rows.

        .. versionadded:: 1.4.0

    Notes
    -----
    The returned rows affected is the sum of the ``rowcount`` attribute of ``sqlite3.Cursor``
    or SQLAlchemy connectable. If using ADBC the returned rows are the result
    of ``Cursor.adbc_ingest``. The returned value may not reflect the exact number of written
    rows as stipulated in the
    `sqlite3 <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.rowcount>`__ or
    `SQLAlchemy <https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.BaseCursorResult.rowcount>`__
    """  # noqa: E501
    # 如果 if_exists 参数不是 "fail", "replace", "append" 中的一个，抛出 ValueError 异常
    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    # 如果 frame 是 Series 类型，则将其转换为 DataFrame 类型
    if isinstance(frame, Series):
        frame = frame.to_frame()
    # 如果 frame 不是 DataFrame 类型，抛出 NotImplementedError 异常
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError(
            "'frame' argument should be either a Series or a DataFrame"
        )

    # 使用 pandasSQL_builder 函数创建 pandas_sql 上下文管理器
    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        # 调用 pandas_sql 的 to_sql 方法执行数据写入数据库操作，并返回结果
        return pandas_sql.to_sql(
            frame,
            name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            engine=engine,
            **engine_kwargs,
        )
def has_table(table_name: str, con, schema: str | None = None) -> bool:
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
        True if the table exists, False otherwise.
    """
    # 使用 pandasSQL_builder 函数创建 pandas_sql 上下文管理器
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        # 调用 pandas_sql 对象的 has_table 方法检查表是否存在
        return pandas_sql.has_table(table_name)


table_exists = has_table


def pandasSQL_builder(
    con,
    schema: str | None = None,
    need_transaction: bool = False,
) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.

    Parameters
    ----------
    con
        Connection object or string URI representing the database connection.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema.
    need_transaction : bool, default False
        Whether a transaction is needed.

    Returns
    -------
    PandasSQL
        Instance of the appropriate PandasSQL subclass based on the connection type.
    """
    import sqlite3

    # 如果 con 是 sqlite3.Connection 类型或者为 None，返回 SQLiteDatabase 对象
    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)

    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")

    # 如果 con 是字符串且未安装 sqlalchemy 报错
    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError("Using URI string without sqlalchemy installed.")

    # 如果 sqlalchemy 存在且 con 是字符串或 sqlalchemy.engine.Connectable 对象，
    # 返回 SQLDatabase 对象
    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)

    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    
    # 如果 adbc 存在且 con 是 adbc.Connection 对象，返回 ADBCDatabase 对象
    if adbc and isinstance(con, adbc.Connection):
        return ADBCDatabase(con)

    # 如果以上条件都不满足，发出警告，并返回 SQLiteDatabase 对象
    warnings.warn(
        "pandas only supports SQLAlchemy connectable (engine/connection) or "
        "database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 "
        "objects are not tested. Please consider using SQLAlchemy.",
        UserWarning,
        stacklevel=find_stack_level(),
    )
    return SQLiteDatabase(con)


class SQLTable(PandasObject):
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """

    # TODO: support for multiIndex

    def __init__(
        self,
        name: str,
        pandas_sql_engine,
        frame=None,
        index: bool | str | list[str] | None = True,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        prefix: str = "pandas",
        index_label=None,
        schema=None,
        keys=None,
        dtype: DtypeArg | None = None,
    ) -> None:
        # 初始化方法，设置对象属性
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype

        if frame is not None:
            # 如果提供了数据框架，则基于数据框架初始化表格设置
            self.table = self._create_table_setup()
        else:
            # 如果没有提供数据，进入只读模式，从数据库中获取表格
            self.table = self.pd_sql.get_table(self.name, self.schema)

        if self.table is None:
            # 如果未能初始化表格，则抛出数值错误异常
            raise ValueError(f"Could not init table '{name}'")

        if not len(self.name):
            # 如果表名为空，则抛出数值错误异常
            raise ValueError("Empty table name specified")

    def exists(self):
        # 检查当前表是否存在于数据库中
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self) -> str:
        # 返回表格的 SQL 架构定义
        from sqlalchemy.schema import CreateTable
        return str(CreateTable(self.table).compile(self.pd_sql.con))

    def _execute_create(self) -> None:
        # 将表格插入数据库，并添加到 MetaData 对象中
        self.table = self.table.to_metadata(self.pd_sql.meta)
        with self.pd_sql.run_transaction():
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        # 创建表格的方法
        if self.exists():
            if self.if_exists == "fail":
                # 如果表格已经存在且指定为失败，则抛出数值错误异常
                raise ValueError(f"Table '{self.name}' already exists.")
            if self.if_exists == "replace":
                # 如果表格已经存在且指定为替换，则先删除旧表格再执行创建
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == "append":
                # 如果表格已经存在且指定为追加，则什么都不做
                pass
            else:
                # 如果 if_exists 参数不合法，则抛出数值错误异常
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            # 如果表格不存在，则执行创建操作
            self._execute_create()

    def _execute_insert(self, conn, keys: list[str], data_iter) -> int:
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
            数据库连接对象
        keys : list of str
            列名列表
        data_iter : generator of list
            每个项包含要插入的值列表
        """
        # 将生成器中的数据转换为字典列表
        data = [dict(zip(keys, row)) for row in data_iter]
        # 执行插入操作并返回影响的行数
        result = conn.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn, keys: list[str], data_iter) -> int:
        """
        Alternative to _execute_insert for DBs support multi-value INSERT.

        Note: multi-value insert is usually faster for analytics DBs
        and tables containing a few columns
        but performance degrades quickly with increase of columns.

        """
        # 使用多值插入的方式将数据插入表格

        from sqlalchemy import insert

        # 将生成器中的数据转换为字典列表
        data = [dict(zip(keys, row)) for row in data_iter]
        # 构建插入语句对象
        stmt = insert(self.table).values(data)
        # 执行插入操作并返回影响的行数
        result = conn.execute(stmt)
        return result.rowcount
    # 定义一个方法，用于插入数据，返回一个元组，包含列名列表和数据列表
    def insert_data(self) -> tuple[list[str], list[np.ndarray]]:
        # 如果存在索引，则创建副本并设置索引名，如果索引名重复则引发异常
        if self.index is not None:
            temp = self.frame.copy(deep=False)
            temp.index.names = self.index
            try:
                # 尝试重置索引，如果有重复的索引名或列名则抛出 ValueError 异常
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f"duplicate name in index/columns: {err}") from err
        else:
            temp = self.frame

        # 获取临时框架中的列名，并转换为字符串列表
        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        # 创建一个列表用于存储数据，初始值为 None，列表长度为列数
        # 错误: 列表项 0 具有不兼容类型 "None"；期望类型 "ndarray"
        data_list: list[np.ndarray] = [None] * ncols  # type: ignore[list-item]

        # 遍历临时框架中的每一列
        for i, (_, ser) in enumerate(temp.items()):
            # 如果列的数据类型是日期时间类型
            if ser.dtype.kind == "M":
                if isinstance(ser._values, ArrowExtensionArray):
                    import pyarrow as pa

                    # 如果是 pyarrow 的日期类型，则转换为 object 类型的 numpy 数组
                    if pa.types.is_date(ser.dtype.pyarrow_dtype):
                        # GH#53854 to_pydatetime 对于 pyarrow 日期类型不支持
                        d = ser._values.to_numpy(dtype=object)
                    else:
                        # 否则转换为 Python datetime 对象数组
                        d = ser.dt.to_pydatetime()._values
                else:
                    # 否则直接转换为 Python datetime 对象数组
                    d = ser._values.to_pydatetime()
            # 如果列的数据类型是时间增量类型
            elif ser.dtype.kind == "m":
                vals = ser._values
                if isinstance(vals, ArrowExtensionArray):
                    # 如果是 Arrow 扩展数组，则转换为 numpy 类型 "m8[ns]" 的数据
                    vals = vals.to_numpy(dtype=np.dtype("m8[ns]"))
                # 将时间增量存储为整数，见 GH#6921, GH#7076
                d = vals.view("i8").astype(object)
            else:
                # 否则将列的值转换为 object 类型的 numpy 数组
                d = ser._values.astype(object)

            # 断言确保 d 是 numpy 数组
            assert isinstance(d, np.ndarray), type(d)

            # 如果列可以包含缺失值（NA），则将缺失值替换为 None
            if ser._can_hold_na:
                # 注意：这会错过时间增量类型，因为它们被转换为整数
                mask = isna(d)
                d[mask] = None

            # 将处理后的数据存入数据列表的相应位置
            data_list[i] = d

        # 返回列名列表和数据列表的元组
        return column_names, data_list

    # 定义一个插入方法，允许设置块大小和插入方法，支持多种参数类型
    def insert(
        self,
        chunksize: int | None = None,
        method: Literal["multi"] | Callable | None = None,
    ) -> int | None:
        # 定义函数签名，指定返回类型为整数或空值

        # 根据传入的插入方法参数，选择对应的执行插入方法
        if method is None:
            exec_insert = self._execute_insert
        elif method == "multi":
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f"Invalid parameter `method`: {method}")

        # 调用对象的插入数据方法，获取数据的键和数据列表
        keys, data_list = self.insert_data()

        # 获取当前数据框的行数
        nrows = len(self.frame)

        # 如果数据框中没有数据，则返回插入行数为0
        if nrows == 0:
            return 0

        # 如果未指定分块大小，则使用数据框的行数作为分块大小
        if chunksize is None:
            chunksize = nrows
        # 如果指定了分块大小为0，则抛出异常
        elif chunksize == 0:
            raise ValueError("chunksize argument should be non-zero")

        # 计算需要分成的块数
        chunks = (nrows // chunksize) + 1
        total_inserted = None

        # 使用对象的数据库连接运行事务
        with self.pd_sql.run_transaction() as conn:
            # 遍历每个数据块
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break

                # 准备数据块的迭代器
                chunk_iter = zip(*(arr[start_i:end_i] for arr in data_list))

                # 执行插入操作，并获取插入的行数
                num_inserted = exec_insert(conn, keys, chunk_iter)

                # 如果成功插入数据，则累加到总插入行数中
                if num_inserted is not None:
                    if total_inserted is None:
                        total_inserted = num_inserted
                    else:
                        total_inserted += num_inserted

        # 返回总共插入的行数
        return total_inserted

    def _query_iterator(
        self,
        result,
        exit_stack: ExitStack,
        chunksize: int | None,
        columns,
        coerce_float: bool = True,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> Generator[DataFrame, None, None]:
        """Return generator through chunked result set."""
        # 初始化标志变量，表示是否已经读取了数据
        has_read_data = False

        # 使用上下文管理器处理数据库连接
        with exit_stack:
            while True:
                # 从查询结果中获取指定大小的数据块
                data = result.fetchmany(chunksize)

                # 如果没有获取到数据块
                if not data:
                    # 如果之前未曾读取过数据，则返回一个空的数据框
                    if not has_read_data:
                        yield DataFrame.from_records(
                            [], columns=columns, coerce_float=coerce_float
                        )
                    break

                # 设置已读取数据的标志为真
                has_read_data = True

                # 将获取的数据块转换为数据框
                self.frame = _convert_arrays_to_dataframe(
                    data, columns, coerce_float, dtype_backend
                )

                # 根据参数调整数据框的列
                self._harmonize_columns(
                    parse_dates=parse_dates, dtype_backend=dtype_backend
                )

                # 如果有指定索引，则设置数据框的索引
                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)

                # 生成当前数据框
                yield self.frame

    def read(
        self,
        exit_stack: ExitStack,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Read data from database using chunked iteration."""
        # 函数用于从数据库中读取数据，以生成器形式逐块返回数据帧
        # 参数详细说明见函数定义
    ) -> DataFrame | Iterator[DataFrame]:
        # 从sqlalchemy库中导入select函数
        from sqlalchemy import select

        # 如果指定了columns并且不为空，则选择相应的列
        if columns is not None and len(columns) > 0:
            # 根据列名构造列对象列表
            cols = [self.table.c[n] for n in columns]
            # 如果指定了index，将index列逆序添加到cols列表的开头
            if self.index is not None:
                for idx in self.index[::-1]:
                    cols.insert(0, self.table.c[idx])
            # 构造SQL的select语句
            sql_select = select(*cols)
        else:
            # 如果未指定columns，则选择整张表
            sql_select = select(self.table)

        # 执行SQL查询
        result = self.pd_sql.execute(sql_select)
        # 获取查询结果的列名
        column_names = result.keys()

        # 如果指定了chunksize，返回查询结果的迭代器
        if chunksize is not None:
            return self._query_iterator(
                result,
                exit_stack,
                chunksize,
                column_names,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype_backend=dtype_backend,
            )
        else:
            # 否则，获取全部数据行
            data = result.fetchall()
            # 将数据转换成DataFrame
            self.frame = _convert_arrays_to_dataframe(
                data, column_names, coerce_float, dtype_backend
            )

            # 调整DataFrame的列属性
            self._harmonize_columns(
                parse_dates=parse_dates, dtype_backend=dtype_backend
            )

            # 如果指定了index，设置DataFrame的索引
            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)

            # 返回最终的DataFrame对象
            return self.frame

    def _index_name(self, index, index_label):
        # 对于写操作：index=True时将索引包含在SQL表中
        if index is True:
            # 获取DataFrame索引的层级数
            nlevels = self.frame.index.nlevels

            # 如果指定了index_label，则将其作为索引名（们）
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError(
                        "Length of 'index_label' should match number of "
                        f"levels, which is {nlevels}"
                    )
                return index_label

            # 如果DataFrame只有一级索引且未命名，并且不存在名为'index'的列，则返回["index"]
            if (
                nlevels == 1
                and "index" not in self.frame.columns
                and self.frame.index.name is None
            ):
                return ["index"]
            else:
                # 否则，返回填充缺失名称的索引列名
                return com.fill_missing_names(self.frame.index.names)

        # 对于读操作：index=(字符串或字符串列表)指定要设置为索引的列
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None
    # 获取列名和类型的列表，根据传入的dtype_mapper映射器进行数据类型转换
    def _get_column_names_and_types(self, dtype_mapper):
        column_names_and_types = []

        # 如果存在索引，遍历索引并根据dtype_mapper映射器获取索引的数据类型，并添加到列表中
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
                column_names_and_types.append((str(idx_label), idx_type, True))

        # 遍历DataFrame的列，根据dtype_mapper映射器获取列的数据类型，并添加到列表中
        column_names_and_types += [
            (str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False)
            for i in range(len(self.frame.columns))
        ]

        # 返回包含列名和类型元组的列表
        return column_names_and_types

    # 创建数据库表的设置，并返回一个SQLAlchemy Table对象
    def _create_table_setup(self):
        from sqlalchemy import (
            Column,
            PrimaryKeyConstraint,
            Table,
        )
        from sqlalchemy.schema import MetaData

        # 获取列名和类型的列表
        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)

        # 根据列名和类型列表创建Column对象列表
        columns: list[Any] = [
            Column(name, typ, index=is_index)
            for name, typ, is_index in column_names_and_types
        ]

        # 如果存在主键键值，创建PrimaryKeyConstraint对象并添加到columns列表中
        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + "_pk")
            columns.append(pkc)

        # 获取表的模式（schema），如果未指定则使用self.schema或self.pd_sql.meta.schema
        schema = self.schema or self.pd_sql.meta.schema

        # 创建一个新的MetaData对象
        meta = MetaData()

        # 返回一个新的Table对象，将之前创建的Column对象列表和schema传入
        return Table(self.name, meta, *columns, schema=schema)

    # 根据参数进行列数据的统一处理
    def _harmonize_columns(
        self,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> None:
        """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
        # 解析 parse_dates 参数，确保格式正确
        parse_dates = _process_parse_dates_argument(parse_dates)

        # 遍历 SQL 表的每一列
        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                # 获取 DataFrame 中的对应列
                df_col = self.frame[col_name]

                # 处理日期解析；避免重复转换列
                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    # 处理日期列的格式
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue

                # 获取 DataFrame 列应有的数据类型
                col_type = self._get_dtype(sql_col.type)

                # 如果数据类型是 datetime, date 或者 DatetimeTZDtype
                if (
                    col_type is datetime
                    or col_type is date
                    or col_type is DatetimeTZDtype
                ):
                    # 将时区感知的 Datetime SQL 列转换为 UTC
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                # 如果 dtype_backend 是 "numpy" 并且数据类型是 float
                elif dtype_backend == "numpy" and col_type is float:
                    # 浮点数支持 NA，可以进行转换
                    self.frame[col_name] = df_col.astype(col_type)

                # 如果 dtype_backend 是 "numpy" 并且列中没有 NA 值
                elif dtype_backend == "numpy" and len(df_col) == df_col.count():
                    # 没有 NA 值，可以转换整数和布尔值
                    if col_type is np.dtype("int64") or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type)
            except KeyError:
                pass  # 如果这一列不在结果中，则跳过
    # 定义一个私有方法，用于根据列的类型推断出对应的SQLAlchemy数据类型
    def _sqlalchemy_type(self, col: Index | Series):
        # 获取数据类型的字典，如果没有则为空字典
        dtype: DtypeArg = self.dtype or {}

        # 如果数据类型是字典形式
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            # 如果列名在数据类型字典中，则返回对应的数据类型
            if col.name in dtype:
                return dtype[col.name]

        # 推断列的数据类型，跳过缺失值
        # 这对于插入包含NULL值的类型化数据是必要的，GH 8778
        col_type = lib.infer_dtype(col, skipna=True)

        # 导入SQLAlchemy库中需要使用的数据类型
        from sqlalchemy.types import (
            TIMESTAMP,
            BigInteger,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            SmallInteger,
            Text,
            Time,
        )

        # 如果列的类型是日期时间或时间戳
        if col_type in ("datetime64", "datetime"):
            # GH 9086: 如果列包含时区信息，则建议使用TIMESTAMP类型
            try:
                # 错误: "Index"的"Union[Index, Series]"项没有"dt"属性
                # 如果列具有时区信息
                if col.dt.tz is not None:  # type: ignore[union-attr]
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                # 列实际上是一个DatetimeIndex或者带有日期数据的Index，例如9999-01-01
                if getattr(col, "tz", None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime

        # 如果列的类型是时间间隔
        if col_type == "timedelta64":
            # 警告：'timedelta'类型不受支持，将写入为整数值（ns频率）到数据库
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return BigInteger

        # 如果列的类型是浮点数
        elif col_type == "floating":
            if col.dtype == "float32":
                return Float(precision=23)
            else:
                return Float(precision=53)

        # 如果列的类型是整数
        elif col_type == "integer":
            # GH35076 将pandas整数映射到最优的SQLAlchemy整数类型
            if col.dtype.name.lower() in ("int8", "uint8", "int16"):
                return SmallInteger
            elif col.dtype.name.lower() in ("uint16", "int32"):
                return Integer
            elif col.dtype.name.lower() == "uint64":
                raise ValueError("Unsigned 64 bit integer datatype is not supported")
            else:
                return BigInteger

        # 如果列的类型是布尔值
        elif col_type == "boolean":
            return Boolean

        # 如果列的类型是日期
        elif col_type == "date":
            return Date

        # 如果列的类型是时间
        elif col_type == "time":
            return Time

        # 如果列的类型是复杂类型
        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        # 默认情况下返回文本类型
        return Text
    # 定义一个方法用于从 SQLAlchemy 的数据类型映射到 Python 数据类型

    from sqlalchemy.types import (
        TIMESTAMP,  # 导入时间戳类型
        Boolean,    # 导入布尔类型
        Date,       # 导入日期类型
        DateTime,   # 导入日期时间类型
        Float,      # 导入浮点数类型
        Integer,    # 导入整数类型
    )

    # 根据给定的 SQLAlchemy 数据类型返回相应的 Python 数据类型
    def _get_dtype(self, sqltype):
        if isinstance(sqltype, Float):
            # 如果是浮点数类型，返回 float 类型
            return float
        elif isinstance(sqltype, Integer):
            # 如果是整数类型，返回 int64 的 NumPy 数据类型
            # TODO: Refine integer size. 待完善整数大小的处理
            return np.dtype("int64")
        elif isinstance(sqltype, TIMESTAMP):
            # 如果是时间戳类型
            if not sqltype.timezone:
                # 如果没有时区信息，返回 datetime 类型
                return datetime
            # 否则返回时区感知的 datetime 数据类型
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            # 如果是日期时间类型
            # 注意: np.datetime64 也是 np.number 的子类
            return datetime
        elif isinstance(sqltype, Date):
            # 如果是日期类型，返回 date 类型
            return date
        elif isinstance(sqltype, Boolean):
            # 如果是布尔类型，返回 bool 类型
            return bool
        # 对于其他未知类型，返回 object 类型
        return object
class PandasSQL(PandasObject, ABC):
    """
    Subclasses Should define read_query and to_sql.
    """

    # 定义 __enter__ 方法，返回当前对象自身
    def __enter__(self) -> Self:
        return self

    # 定义 __exit__ 方法，用于退出上下文管理器，不执行任何操作
    def __exit__(self, *args) -> None:
        pass

    @abstractmethod
    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # 抽象方法，用于从数据源读取数据表，需要在子类中实现
        raise NotImplementedError

    @abstractmethod
    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # 抽象方法，用于执行 SQL 查询并返回数据框或数据框迭代器，需要在子类中实现
        pass

    @abstractmethod
    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        # 抽象方法，用于将数据框写入 SQL 表中，需要在子类中实现
        pass

    @abstractmethod
    def execute(self, sql: str | Select | TextClause, params=None):
        # 抽象方法，用于执行 SQL 语句或对象，需要在子类中实现
        pass

    @abstractmethod
    def has_table(self, name: str, schema: str | None = None) -> bool:
        # 抽象方法，用于检查是否存在指定的表，需要在子类中实现
        pass

    @abstractmethod
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # 抽象方法，用于创建 SQL 表的模式，需要在子类中实现
        pass


class BaseEngine:
    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        """
        Inserts data into already-prepared table
        """
        # 抽象方法，用于向预先准备好的表中插入数据，需要在子类中实现
        raise AbstractMethodError(self)


class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None:
        import_optional_dependency(
            "sqlalchemy", extra="sqlalchemy is required for SQL support."
        )

    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        # 继承自 BaseEngine 的方法，用于向 SQLAlchemy 引擎插入记录
        pass
    # 定义函数，指定返回类型为整数或空值
    def insert_data(
        # 导入 SQL 异常模块
        ) -> int | None:
        # 导入 sqlalchemy 异常模块
        from sqlalchemy import exc

        # 尝试执行数据插入操作，设置批处理大小和方法
        try:
            return table.insert(chunksize=chunksize, method=method)
        # 捕获 SQL 语句错误
        except exc.StatementError as err:
            # 异常消息匹配模式，用于检测 MySQL 中 inf 的使用问题
            # 参考 GH34431 和 Stack Overflow 上的解决方案
            msg = r"""(\(1054, "Unknown column 'inf(e0)?' in 'field list'"\))(?#
            )|inf can not be used with MySQL"""
            # 转换异常对象为字符串形式
            err_text = str(err.orig)
            # 如果异常消息匹配到指定模式，则抛出值错误异常
            if re.search(msg, err_text):
                raise ValueError("inf cannot be used with MySQL") from err
            # 否则继续抛出原始的 SQL 语句错误异常
            raise err
def get_engine(engine: str) -> BaseEngine:
    """返回我们的实现"""
    # 如果 engine 参数为 "auto"，则尝试获取预设的引擎
    if engine == "auto":
        engine = get_option("io.sql.engine")

    # 如果 engine 参数为 "auto"，则按照以下顺序尝试不同的引擎类
    if engine == "auto":
        engine_classes = [SQLAlchemyEngine]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                # 尝试实例化引擎类并返回
                return engine_class()
            except ImportError as err:
                # 捕获 ImportError 并记录错误信息
                error_msgs += "\n - " + str(err)

        # 如果所有引擎类均导入失败，则抛出 ImportError 异常
        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'sqlalchemy'.\n"
            "A suitable version of "
            "sqlalchemy is required for sql I/O "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    # 如果 engine 参数为 "sqlalchemy"，则返回 SQLAlchemyEngine 实例
    if engine == "sqlalchemy":
        return SQLAlchemyEngine()

    # 如果 engine 参数既不是 "auto" 也不是 "sqlalchemy"，则抛出 ValueError 异常
    raise ValueError("engine must be one of 'auto', 'sqlalchemy'")


class SQLDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.
    """

    def __init__(
        self, con, schema: str | None = None, need_transaction: bool = False
    ) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.schema import MetaData

        # self.exit_stack 用于清理 Engine 和 Connection，如果下面创建了这些对象，
        # 并在 read_sql 返回迭代器的情况下，处理事务的提交。
        self.exit_stack = ExitStack()
        
        # 如果 con 参数是字符串，则创建一个 SQLAlchemy Engine 对象
        if isinstance(con, str):
            con = create_engine(con)
            self.exit_stack.callback(con.dispose)
        
        # 如果 con 参数是 Engine 类型，则直接使用它的连接
        if isinstance(con, Engine):
            con = self.exit_stack.enter_context(con.connect())
        
        # 如果需要事务且 con 还未处于事务中，则开始一个新的事务
        if need_transaction and not con.in_transaction():
            self.exit_stack.enter_context(con.begin())
        
        # 设置实例的 con 和 meta 属性
        self.con = con
        self.meta = MetaData(schema=schema)
        self.returns_generator = False

    def __exit__(self, *args) -> None:
        # 如果不是生成器模式，则关闭 exit_stack
        if not self.returns_generator:
            self.exit_stack.close()

    @contextmanager
    def run_transaction(self):
        # 如果 con 尚未处于事务中，则开始一个事务，并返回 con
        if not self.con.in_transaction():
            with self.con.begin():
                yield self.con
        else:
            # 如果 con 已经处于事务中，则直接返回 con
            yield self.con
    # 定义一个方法用于执行 SQL 查询或命令
    def execute(self, sql: str | Select | TextClause, params=None):
        """Simple passthrough to SQLAlchemy connectable"""
        # 如果参数 params 为 None，则将 args 设为一个空列表，否则作为列表中的单个元素
        args = [] if params is None else [params]
        # 如果 sql 参数是字符串类型，则调用 self.con.exec_driver_sql 方法执行 SQL 命令
        if isinstance(sql, str):
            return self.con.exec_driver_sql(sql, *args)
        # 否则调用 self.con.execute 方法执行 SQL 查询
        return self.con.execute(sql, *args)

    # 定义一个方法用于从数据库表中读取数据
    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            SQL数据库中表的名称。
        index_col : string, optional, default: None
            要设置为索引的列。
        coerce_float : bool, default True
            尝试将非字符串、非数值对象（如decimal.Decimal）转换为浮点数。这可能导致精度损失。
        parse_dates : list or dict, default: None
            - 要解析为日期的列名列表。
            - ``{column_name: format string}`` 的字典，其中格式字符串与strftime兼容，用于解析字符串时间，
              或者是(D, s, ns, ms, us)中的一个，用于解析整数时间戳。
            - ``{column_name: arg}`` 的字典，其中arg对应于:func:`pandas.to_datetime`的关键字参数。
              在没有本地日期时间支持的数据库中特别有用，如SQLite。
        columns : list, default: None
            要从SQL表中选择的列名列表。
        schema : string, default None
            要查询的SQL数据库中的模式名称（如果数据库类型支持）。如果指定，则会覆盖SQL数据库对象的默认模式。
        chunksize : int, default None
            如果指定，返回一个迭代器，其中`chunksize`是每个块中包含的行数。
        dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
            应用于结果DataFrame的后端数据类型（仍处于实验阶段）。行为如下：

            * ``"numpy_nullable"``: 返回可空dtype支持的DataFrame（默认）。
            * ``"pyarrow"``: 返回支持pyarrow的可空ArrowDtype DataFrame。

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        # 使用meta数据反射，绑定到给定的表名，从而获取表的元数据信息
        self.meta.reflect(bind=self.con, only=[table_name], views=True)
        # 创建SQLTable对象，用于读取数据库中的表数据
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        # 如果指定了chunksize，将返回生成器
        if chunksize is not None:
            self.returns_generator = True
        # 调用SQLTable对象的read方法，读取数据并返回DataFrame
        return table.read(
            self.exit_stack,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )
    def _query_iterator(
        result,
        exit_stack: ExitStack,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> Generator[DataFrame, None, None]:
        """Return generator through chunked result set"""
        # 初始化一个标志，用来表示是否已经读取了数据
        has_read_data = False
        # 使用 exit_stack 确保退出时资源能够正确关闭
        with
    ) -> DataFrame | Iterator[DataFrame]:
        """
        从 SQL 查询结果读取数据到 DataFrame。

        Parameters
        ----------
        sql : str
            要执行的 SQL 查询语句。
        index_col : string, optional, default: None
            返回的 DataFrame 对象要用作索引的列名。
        coerce_float : bool, default True
            尝试将非字符串、非数值对象（如 decimal.Decimal）的值转换为浮点数，对于 SQL 结果集很有用。
        params : list, tuple or dict, optional, default: None
            要传递给 execute 方法的参数列表。传递参数的语法依赖于数据库驱动程序。
            请查阅您的数据库驱动程序文档，了解支持的参数语法样式。
            例如，对于 psycopg2，使用 %(name)s，因此可以使用 params={'name' : 'value'}。
        parse_dates : list or dict, default: None
            - 要解析为日期的列名列表。
            - 字典形式的 ``{column_name: format string}``，其中格式字符串与 strftime 兼容，用于解析字符串时间；
              或者是 (D, s, ns, ms, us) 中的一种，用于解析整数时间戳。
            - 字典形式的 ``{column_name: arg dict}``，其中 arg dict 对应于 :func:`pandas.to_datetime` 的关键字参数。
              在没有原生日期时间支持的数据库（如 SQLite）中特别有用。
        chunksize : int, default None
            如果指定，则返回一个迭代器，其中 `chunksize` 是每个块中包含的行数。
        dtype : Type name or dict of columns
            数据或列的数据类型。例如 np.float64 或 {'a': np.float64, 'b': np.int32, 'c': 'Int64'}。

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame
            返回一个 DataFrame 对象。

        See Also
        --------
        read_sql_table : 从 SQL 数据库表读取数据到 DataFrame。
        read_sql

        """
        # 执行 SQL 查询，并获取结果集
        result = self.execute(sql, params)
        # 获取结果集的列名
        columns = result.keys()

        # 如果指定了 chunksize，则返回一个迭代器
        if chunksize is not None:
            self.returns_generator = True
            return self._query_iterator(
                result,
                self.exit_stack,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            # 否则，获取所有行数据
            data = result.fetchall()
            # 将结果包装成 DataFrame
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    # read_sql 是 read_query 的别名
    read_sql = read_query
    def prep_table(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool | str | list[str] | None = True,
        index_label=None,
        schema=None,
        dtype: DtypeArg | None = None,
    ) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """

        # 如果指定了数据类型，检查并处理为字典形式以确保兼容性
        if dtype:
            if not is_dict_like(dtype):
                # 如果传入的 dtype 不是字典形式，则构建字典，以列名为键，dtype 为值
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)

            from sqlalchemy.types import TypeEngine

            # 遍历 dtype 字典，检查每个列的数据类型是否为 SQLAlchemy 的有效类型
            for col, my_type in dtype.items():
                if isinstance(my_type, type) and issubclass(my_type, TypeEngine):
                    pass
                elif isinstance(my_type, TypeEngine):
                    pass
                else:
                    # 如果数据类型不符合预期，抛出 ValueError 异常
                    raise ValueError(f"The type of {col} is not a SQLAlchemy type")

        # 创建 SQLTable 对象，用于操作数据库中的表格
        table = SQLTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )
        # 调用 SQLTable 对象的 create 方法，在数据库中创建表格
        table.create()
        # 返回创建的 SQLTable 对象
        return table

    def check_case_sensitive(
        self,
        name: str,
        schema: str | None,
    ) -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """

        # 检查表名是否可能存在大小写敏感性问题
        if not name.isdigit() and not name.islower():
            # 仅当表名不是纯数字且不是全小写时才进行检查

            # 导入 SQLAlchemy 的 inspect 方法
            from sqlalchemy import inspect as sqlalchemy_inspect

            # 使用 inspect 方法检查数据库连接中的表名列表
            insp = sqlalchemy_inspect(self.con)
            table_names = insp.get_table_names(schema=schema or self.meta.schema)

            # 如果指定的表名不在数据库中找到，则发出警告
            if name not in table_names:
                msg = (
                    f"The provided table name '{name}' is not found exactly as "
                    "such in the database after writing the table, possibly "
                    "due to case sensitivity issues. Consider using lower "
                    "case table names."
                )
                # 发出用户警告，提醒可能存在的问题
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
    # 定义一个方法 to_sql，用于将 DataFrame 写入 SQL 数据库中的指定表名
    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ):
        # 此处省略了具体实现，根据传入参数将 DataFrame 写入 SQL 表中
        pass

    # 定义一个属性 tables，返回与当前连接相关的元数据中的表
    @property
    def tables(self):
        return self.meta.tables

    # 定义一个方法 has_table，用于检查指定的表名是否存在于数据库中
    def has_table(self, name: str, schema: str | None = None) -> bool:
        # 导入 SQLAlchemy 的 inspect 函数并使用当前连接进行检查
        from sqlalchemy import inspect as sqlalchemy_inspect

        insp = sqlalchemy_inspect(self.con)
        return insp.has_table(name, schema or self.meta.schema)

    # 定义一个方法 get_table，用于获取指定名称的数据库表对象
    def get_table(self, table_name: str, schema: str | None = None) -> Table:
        # 导入 SQLAlchemy 的 Table 类和 Numeric 类型
        from sqlalchemy import Numeric, Table

        # 如果未指定 schema，则使用默认的 self.meta.schema
        schema = schema or self.meta.schema
        # 使用指定的表名和连接 autoload_with 参数创建表对象 tbl
        tbl = Table(table_name, self.meta, autoload_with=self.con, schema=schema)
        # 遍历表的每一列，如果列的类型是 Numeric，则将 asdecimal 属性设置为 False
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    # 定义一个方法 drop_table，用于删除指定的数据库表
    def drop_table(self, table_name: str, schema: str | None = None) -> None:
        # 如果未指定 schema，则使用默认的 self.meta.schema
        schema = schema or self.meta.schema
        # 如果数据库中存在指定的表，则使用元数据的 reflect 方法加载该表
        if self.has_table(table_name, schema):
            self.meta.reflect(
                bind=self.con, only=[table_name], schema=schema, views=True
            )
            # 在事务中运行获取表对象并删除表的操作
            with self.run_transaction():
                self.get_table(table_name, schema).drop(bind=self.con)
            # 清除元数据缓存
            self.meta.clear()

    # 定义一个方法 _create_sql_schema，用于创建 SQL 表的 SQL 架构语句
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # 创建 SQLTable 对象，用于生成 SQL 表的结构
        table = SQLTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        # 返回 SQLTable 对象生成的 SQL 架构语句的字符串表示
        return str(table.sql_schema())
# ---- SQL without SQLAlchemy ---

# 定义一个从 PandasSQL 继承而来的 ADBCDatabase 类，用于处理 DataFrame 和 SQL 数据库之间的转换
class ADBCDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using ADBC to handle DataBase abstraction.

    Parameters
    ----------
    con : adbc_driver_manager.dbapi.Connection
    """

    def __init__(self, con) -> None:
        # 初始化方法，接收一个连接对象 con，并将其存储在实例变量 self.con 中
        self.con = con

    @contextmanager
    def run_transaction(self):
        # 提供一个事务运行的上下文管理器，使用 self.con 的游标执行事务操作
        with self.con.cursor() as cur:
            try:
                yield cur  # 允许事务代码块执行
            except Exception:
                self.con.rollback()  # 如果出现异常则回滚事务
                raise
            self.con.commit()  # 提交事务

    def execute(self, sql: str | Select | TextClause, params=None):
        # 执行 SQL 查询或命令，接收一个字符串 sql 或 SQLAlchemy 的 Select 或 TextClause 对象
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)  # 执行 SQL 命令
            return cur  # 返回游标以便后续处理结果
        except Exception as exc:
            try:
                self.con.rollback()  # 如果执行失败则回滚事务
            except Exception as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        # 从指定的数据库表中读取数据到 DataFrame 中
        pass  # 这里的方法体尚未实现，只是占位符

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        # 执行给定的 SQL 查询并将结果读取为 DataFrame
        pass  # 这里的方法体尚未实现，只是占位符
    def read_query(
        sql: str,
        index_col: Optional[str] = None,
        coerce_float: bool = True,
        params: Optional[Union[list, tuple, dict]] = None,
        parse_dates: Optional[Union[list, dict]] = None,
        chunksize: Optional[int] = None,
        dtype: Union[type, dict] = None,
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.
    
        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            If not True, raise NotImplementedError because 'coerce_float' is not
            supported for ADBC drivers.
        params : list, tuple or dict, optional, default: None
            If not None, raise NotImplementedError because 'params' is not
            supported for ADBC drivers.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible or one of (D, s, ns, ms, us).
            - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support.
        chunksize : int, default None
            If not None, raise NotImplementedError because 'chunksize' is not
            supported for ADBC drivers.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or {'a': np.float64,
            'b': np.int32, 'c': 'Int64'}.
    
            .. versionadded:: 1.3.0
    
        Returns
        -------
        DataFrame
    
        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql
    
        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if params:
            raise NotImplementedError("'params' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
    
        mapping: type[ArrowDtype] | None | Callable
        if dtype_backend == "pyarrow":
            mapping = ArrowDtype
        elif dtype_backend == "numpy_nullable":
            from pandas.io._util import _arrow_dtype_mapping
    
            mapping = _arrow_dtype_mapping().get
        else:
            mapping = None
    
        with self.con.cursor() as cur:
            cur.execute(sql)
            df = cur.fetch_arrow_table().to_pandas(types_mapper=mapping)
    
        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
            dtype=dtype,
        )
    
    read_sql = read_query
    
    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ):
        """
        Placeholder for 'to_sql' method which is not fully implemented here.
        """
    # 检查数据库中是否存在指定表格
    def has_table(self, name: str, schema: str | None = None) -> bool:
        # 通过调用连接对象的 adbc_get_objects 方法获取数据库对象的元数据
        meta = self.con.adbc_get_objects(
            db_schema_filter=schema, table_name_filter=name
        ).read_all()

        # 遍历元数据中的所有数据库模式
        for catalog_schema in meta["catalog_db_schemas"].to_pylist():
            if not catalog_schema:
                continue
            # 遍历每个模式中的模式记录
            for schema_record in catalog_schema:
                if not schema_record:
                    continue

                # 遍历每个模式记录中的表格记录
                for table_record in schema_record["db_schema_tables"]:
                    # 如果找到与指定表名匹配的表格记录，则返回 True
                    if table_record["table_name"] == name:
                        return True

        # 如果未找到匹配的表格记录，则返回 False
        return False

    # 创建 SQL 模式的字符串表示，但针对 ADBC 抛出未实现异常
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # 抛出未实现异常，提示该方法在 ADBC 中未被实现
        raise NotImplementedError("not implemented for adbc")
# sqlite-specific sql strings and handler class
# dictionary used for readability purposes
# 用于提升可读性的 SQLite 特定 SQL 字符串和处理程序类

_SQL_TYPES = {
    "string": "TEXT",
    "floating": "REAL",
    "integer": "INTEGER",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "INTEGER",
}


def _get_unicode_name(name: object) -> str:
    # Converts the given object `name` to a UTF-8 encoded string, ensuring it's valid UTF-8.
    # Raises a ValueError if conversion fails due to non-UTF-8 characters.
    try:
        uname = str(name).encode("utf-8", "strict").decode("utf-8")
    except UnicodeError as err:
        raise ValueError(f"Cannot convert identifier to UTF-8: '{name}'") from err
    return uname


def _get_valid_sqlite_name(name: object) -> str:
    # See https://stackoverflow.com/questions/6514274/how-do-you-escape-strings\
    # -for-sqlite-table-column-names-in-python
    # Ensure the string can be encoded as UTF-8.
    # Ensure the string does not include any NUL characters.
    # Replace all " with "".
    # Wrap the entire thing in double quotes.

    # Returns a SQLite-valid identifier string for the given `name`.
    uname = _get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        raise ValueError("SQLite identifier cannot contain NULs")
    return '"' + uname.replace('"', '""') + '"'


class SQLiteTable(SQLTable):
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        # Initialize SQLiteTable instance, inheriting from SQLTable.
        super().__init__(*args, **kwargs)
        
        # Register SQLite adapters for date and time types.
        self._register_date_adapters()

    def _register_date_adapters(self) -> None:
        # GH 8341
        # Register an adapter callable for datetime.time object
        import sqlite3

        # This adapter transforms time(12,34,56,789) into '12:34:56.000789'
        def _adapt_time(t) -> str:
            return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"

        # Register adapters for date and datetime
        sqlite3.register_adapter(time, _adapt_time)
        sqlite3.register_adapter(date, lambda val: val.isoformat())
        sqlite3.register_adapter(datetime, lambda val: val.isoformat(" "))

        # Converters for date and timestamp types
        sqlite3.register_converter("date", lambda val: date.fromisoformat(val.decode()))
        sqlite3.register_converter("timestamp", lambda val: datetime.fromisoformat(val.decode()))

    def sql_schema(self) -> str:
        # Return SQL schema as a string, joining all table statements with ';'
        return str(";\n".join(self.table))

    def _execute_create(self) -> None:
        # Execute table creation statements within a transaction
        with self.pd_sql.run_transaction() as conn:
            for stmt in self.table:
                conn.execute(stmt)
    # 生成一个插入语句，用于将数据插入到数据库表中
    def insert_statement(self, *, num_rows: int) -> str:
        # 获取数据框中所有列的名称，并转换为字符串列表
        names = list(map(str, self.frame.columns))
        wld = "?"  # 通配符字符
        escape = _get_valid_sqlite_name  # 获取有效的 SQLite 名称的函数

        # 如果存在索引，则将索引列名倒序插入到列名列表的最前面
        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)

        # 将列名列表中的每个列名转换为适合 SQLite 的格式
        bracketed_names = [escape(column) for column in names]
        col_names = ",".join(bracketed_names)  # 以逗号分隔的列名字符串

        # 生成用于插入指定行数数据的通配符语句
        row_wildcards = ",".join([wld] * len(names))
        wildcards = ",".join([f"({row_wildcards})" for _ in range(num_rows)])
        # 生成最终的 INSERT INTO 语句
        insert_statement = (
            f"INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}"
        )
        return insert_statement

    # 执行单行数据插入操作到数据库表中
    def _execute_insert(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        # 使用连接对象执行多次插入单行数据的 SQL 语句
        conn.executemany(self.insert_statement(num_rows=1), data_list)
        return conn.rowcount

    # 执行多行数据插入操作到数据库表中
    def _execute_insert_multi(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        # 将多行数据扁平化后，执行一次插入多行数据的 SQL 语句
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)
        return conn.rowcount

    # 创建表的 SQL 语句设置
    def _create_table_setup(self):
        """
        返回一个 SQL 语句列表，用于创建反映数据框结构的表。
        第一条语句是 CREATE TABLE 语句，其余是 CREATE INDEX 语句。
        """
        # 获取列名和数据类型的列表，并获取有效的 SQLite 名称函数
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = _get_valid_sqlite_name

        # 生成创建表的 SQL 语句
        create_tbl_stmts = [
            escape(cname) + " " + ctype for cname, ctype, _ in column_names_and_types
        ]

        # 如果存在主键，则生成相应的约束语句
        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ", ".join([escape(c) for c in keys])
            create_tbl_stmts.append(
                f"CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})"
            )

        # 如果存在架构，则设置架构名称；否则为空字符串
        if self.schema:
            schema_name = self.schema + "."
        else:
            schema_name = ""

        # 组装创建表的完整 SQL 语句
        create_stmts = [
            "CREATE TABLE "
            + schema_name
            + escape(self.name)
            + " (\n"
            + ",\n  ".join(create_tbl_stmts)
            + "\n)"
        ]

        # 获取需要创建索引的列，并生成 CREATE INDEX 语句
        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if len(ix_cols):
            cnames = "_".join(ix_cols)
            cnames_br = ",".join([escape(c) for c in ix_cols])
            create_stmts.append(
                "CREATE INDEX "
                + escape("ix_" + self.name + "_" + cnames)
                + "ON "
                + escape(self.name)
                + " ("
                + cnames_br
                + ")"
            )

        return create_stmts
    def _sql_type_name(self, col):
        # 获取传入参数中的 dtype，若未定义则使用空字典
        dtype: DtypeArg = self.dtype or {}

        # 检查 dtype 是否类似字典，如果是则转换为字典类型
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            # 如果列名在 dtype 中已定义，则返回其对应的数据类型
            if col.name in dtype:
                return dtype[col.name]

        # 推断列的数据类型，忽略缺失值
        col_type = lib.infer_dtype(col, skipna=True)

        # 如果推断出的数据类型为 "timedelta64"
        if col_type == "timedelta64":
            # 发出警告，表示 timedelta 类型不受支持，将以整数值（ns 频率）写入数据库
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            col_type = "integer"

        # 如果推断出的数据类型为 "datetime64"
        elif col_type == "datetime64":
            col_type = "datetime"

        # 如果推断出的数据类型为 "empty"
        elif col_type == "empty":
            col_type = "string"

        # 如果推断出的数据类型为 "complex"
        elif col_type == "complex":
            # 抛出异常，复杂数据类型不受支持
            raise ValueError("Complex datatypes not supported")

        # 如果推断出的数据类型不在预定义的 SQL 类型集合中，则默认为 "string"
        if col_type not in _SQL_TYPES:
            col_type = "string"

        # 返回推断出的列数据类型对应的 SQL 类型
        return _SQL_TYPES[col_type]
    def read_query(
        self,
        sql,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """
        Execute a read query on the database and return results in chunks.

        Parameters
        ----------
        sql : str
            SQL query to execute.
        index_col : str or list of str, optional
            Column(s) to set as index of the DataFrame.
        coerce_float : bool, default True
            Attempt to convert values to non-string, non-numeric objects to floating point.
        parse_dates : list or dict, default None
            List of columns to parse as dates.
        params : dict or tuple, optional
            Parameters to pass to the query.
        chunksize : int, optional
            Number of rows to fetch per chunk.
        dtype : dict, optional
            Data type specification for the columns.
        dtype_backend : {'numpy', 'pandas'}, default 'numpy'
            Data type backend to use.

        Returns
        -------
        Generator[DataFrame, None, None]
            Generator yielding DataFrames chunk by chunk from the query result.

        Raises
        ------
        TypeError
            If the provided SQL query is not a string.
        DatabaseError
            If execution of the SQL query fails and rollback also fails.
        """
        # Check if sql is a string; raise error if not
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")

        # Prepare arguments for query execution
        args = [] if params is None else [params]

        # Create a cursor object for executing the query
        cur = self.con.cursor()

        try:
            # Execute the SQL query with provided parameters
            cur.execute(sql, *args)

            # Return the cursor object to fetch data in chunks
            return cur
        except Exception as exc:
            try:
                # Attempt to rollback the transaction if execution fails
                self.con.rollback()
            except Exception as inner_exc:  # pragma: no cover
                # If rollback also fails, raise DatabaseError with details
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            # Raise DatabaseError with details of SQL execution failure
            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc
    ) -> DataFrame | Iterator[DataFrame]:
        # 执行 SQL 查询并获取游标对象
        cursor = self.execute(sql, params)
        # 从游标对象的描述中获取列名列表
        columns = [col_desc[0] for col_desc in cursor.description]

        # 如果指定了分块大小，则返回查询结果的迭代器
        if chunksize is not None:
            return self._query_iterator(
                cursor,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            # 否则，获取所有查询结果并关闭游标
            data = self._fetchall_as_list(cursor)
            cursor.close()

            # 将查询结果包装成 DataFrame 对象
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    def _fetchall_as_list(self, cur):
        # 从游标对象获取所有结果并转换为列表形式
        result = cur.fetchall()
        if not isinstance(result, list):
            result = list(result)
        return result

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: str = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ):
        # 将 DataFrame 对象写入到 SQL 表中
        ...

    def has_table(self, name: str, schema: str | None = None) -> bool:
        # 检查 SQLite 数据库中是否存在指定名称的表或视图
        wld = "?"
        query = f"""
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type IN ('table', 'view')
            AND name={wld};
        """
        return len(self.execute(query, [name]).fetchall()) > 0

    def get_table(self, table_name: str, schema: str | None = None) -> None:
        # 获取指定表在数据库中的信息，SQLite 模式下不支持此方法，返回 None
        return None  # not supported in fallback mode

    def drop_table(self, name: str, schema: str | None = None) -> None:
        # 删除 SQLite 数据库中指定名称的表
        drop_sql = f"DROP TABLE {_get_valid_sqlite_name(name)}"
        self.execute(drop_sql)

    def _create_sql_schema(
        self,
        frame,
        table_name: str,
        keys=None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # 创建 SQLite 表的 SQL 模式
        table = SQLiteTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())
# 定义函数 get_schema，获取给定 DataFrame 的 SQL 数据库表结构

def get_schema(
    frame,                          # 输入参数：DataFrame 对象，表示要获取表结构的数据
    name: str,                      # 输入参数：字符串，指定要创建的 SQL 表名
    keys=None,                      # 输入参数：字符串或序列，指定主键列，可选
    con=None,                       # 输入参数：ADB 连接、SQLAlchemy 可连接对象、sqlite3 连接，可选
                                    # ADB 连接提供原生类型支持的高性能 I/O。
                                    # 使用 SQLAlchemy 可以连接该库支持的任何数据库。
                                    # 如果是 DBAPI2 对象，则只支持 sqlite3 连接。

    dtype: DtypeArg | None = None,   # 输入参数：列名到 SQL 类型的字典，可选
                                    # 指定列的数据类型。SQL 类型应为 SQLAlchemy 类型，或者是 sqlite3 的字符串类型。

    schema: str | None = None,      # 输入参数：字符串，指定用于创建表的模式，可选

) -> str:                           # 返回类型：字符串，表示返回的 SQL 表结构语句

    """
    Get the SQL db table schema for the given frame.

    Parameters
    ----------
    frame : DataFrame               # 输入参数：DataFrame 对象，表示要获取表结构的数据
    name : str                      # 输入参数：字符串，指定要创建的 SQL 表名
        name of SQL table
    keys : string or sequence, default: None
                                    # 输入参数：字符串或序列，指定主键列，可选
                                    # columns to use a primary key
    con: ADBC Connection, SQLAlchemy connectable, sqlite3 connection, default: None
                                    # 输入参数：ADB 连接、SQLAlchemy 可连接对象、sqlite3 连接，可选
                                    # ADBC provides high performance I/O with native type support, where available.
                                    # Using SQLAlchemy makes it possible to use any DB supported by that
                                    # library
                                    # If a DBAPI2 object, only sqlite3 is supported.
    dtype : dict of column name to SQL type, default None
                                    # 输入参数：列名到 SQL 类型的字典，可选
                                    # Optional specifying the datatype for columns. The SQL type should
                                    # be a SQLAlchemy type, or a string for sqlite3 fallback connection.
    schema: str, default: None
                                    # 输入参数：字符串，指定用于创建表的模式，可选
                                    # Optional specifying the schema to be used in creating the table.
    """

    # 使用 pandasSQL_builder 上下文管理器创建 pandas_sql 对象
    with pandasSQL_builder(con=con) as pandas_sql:
        # 调用 pandas_sql 对象的 _create_sql_schema 方法，获取 SQL 表结构语句，并返回
        return pandas_sql._create_sql_schema(
            frame,                      # 参数：DataFrame 对象，表示要创建表的数据
            name,                       # 参数：字符串，指定要创建的 SQL 表名
            keys=keys,                  # 参数：字符串或序列，指定主键列
            dtype=dtype,                # 参数：列名到 SQL 类型的字典，指定列的数据类型
            schema=schema               # 参数：字符串，指定用于创建表的模式
        )
```