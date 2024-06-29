# `D:\src\scipysrc\pandas\pandas\io\formats\info.py`

```
# 从__future__导入annotations，确保使用注解
from __future__ import annotations

# 导入抽象基类相关的模块
from abc import (
    ABC,
    abstractmethod,
)

# 导入sys模块，用于系统相关功能
import sys

# 导入textwrap模块的dedent函数，用于移除文本块的缩进
from textwrap import dedent

# 导入类型检查相关的模块和类型
from typing import TYPE_CHECKING

# 从pandas._config模块导入get_option函数
from pandas._config import get_option

# 从pandas.io.formats导入format别名为fmt
from pandas.io.formats import format as fmt

# 从pandas.io.formats.printing导入pprint_thing函数
from pandas.io.formats.printing import pprint_thing

# 如果正在类型检查阶段，导入以下类型
if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    # 从pandas._typing导入Dtype和WriteBuffer类型
    from pandas._typing import (
        Dtype,
        WriteBuffer,
    )

    # 从pandas导入DataFrame和Series类型
    from pandas import (
        DataFrame,
        Index,
        Series,
    )


# 使用dedent函数定义frame_max_cols_sub变量，包含最大列数说明的文本块
frame_max_cols_sub = dedent(
    """\
    max_cols : int, optional
        When to switch from the verbose to the truncated output. If the
        DataFrame has more than `max_cols` columns, the truncated output
        is used. By default, the setting in
        ``pandas.options.display.max_info_columns`` is used."""
)

# 使用dedent函数定义show_counts_sub变量，包含是否显示非空计数的文本块
show_counts_sub = dedent(
    """\
    show_counts : bool, optional
        Whether to show the non-null counts. By default, this is shown
        only if the DataFrame is smaller than
        ``pandas.options.display.max_info_rows`` and
        ``pandas.options.display.max_info_columns``. A value of True always
        shows the counts, and False never shows the counts."""
)

# 使用dedent函数定义frame_examples_sub变量，包含DataFrame示例的文本块
frame_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
    ...                   "float_col": float_values})
    >>> df
        int_col text_col  float_col
    0        1    alpha       0.00
    1        2     beta       0.25
    2        3    gamma       0.50
    3        4    delta       0.75
    4        5  epsilon       1.00

    Prints information of all columns:

    >>> df.info(verbose=True)
    <class 'pandas.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      object
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Prints a summary of columns count and its dtypes but not per column
    information:

    >>> df.info(verbose=False)
    <class 'pandas.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Columns: 3 entries, int_col to float_col
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Pipe output of DataFrame.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> df.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260
"""
)
    # `memory_usage` 参数允许深度内省模式，特别适用于大型 DataFrame 和精细调整内存优化：
    
    # 创建一个包含随机字符串的大数组，用于生成 DataFrame 的示例数据
    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    
    # 创建一个包含三列的 DataFrame，每列有 100 万条数据，数据随机选择自 ['a', 'b', 'c']
    >>> df = pd.DataFrame({
    ...     'column_1': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_2': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_3': np.random.choice(['a', 'b', 'c'], 10 ** 6)
    ... })
    
    # 打印 DataFrame 的基本信息，包括数据类型和内存使用情况
    >>> df.info()
    <class 'pandas.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype 
    ---  ------    --------------    ----- 
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 22.9+ MB
    
    # 使用 'deep' 参数重新打印 DataFrame 的信息，此时将会包括更详细的内存使用信息
    >>> df.info(memory_usage='deep')
    <class 'pandas.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype 
    ---  ------    --------------    ----- 
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 165.9 MB
# 创建一个多行字符串，描述 DataFrame.describe 和 DataFrame.memory_usage 方法的功能
frame_see_also_sub = dedent(
    """\
    DataFrame.describe: Generate descriptive statistics of DataFrame
        columns.
    DataFrame.memory_usage: Memory usage of DataFrame columns."""
)

# 定义一个字典，包含了用于描述 DataFrame 或 Series 的相关参数和信息
frame_sub_kwargs = {
    "klass": "DataFrame",
    "type_sub": " and columns",
    "max_cols_sub": frame_max_cols_sub,  # DataFrame 的最大列数限制
    "show_counts_sub": show_counts_sub,  # 是否显示计数
    "examples_sub": frame_examples_sub,  # DataFrame 的示例用法
    "see_also_sub": frame_see_also_sub,  # 参见部分，包含了相关的方法描述
    "version_added_sub": "",  # 版本添加信息，此处为空字符串
}

# 创建一个多行字符串，描述 Series 的示例用法
series_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> s = pd.Series(text_values, index=int_values)
    >>> s.info()
    <class 'pandas.core.series.Series'>
    Index: 5 entries, 1 to 5
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    5 non-null      object
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Prints a summary excluding information about its values:

    >>> s.info(verbose=False)
    <class 'pandas.core.series.Series'>
    Index: 5 entries, 1 to 5
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Pipe output of Series.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> s.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big Series and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> s = pd.Series(np.random.choice(['a', 'b', 'c'], 10 ** 6))
    >>> s.info()
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 7.6+ MB

    >>> s.info(memory_usage='deep')
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 55.3 MB"""
)

# 创建一个多行字符串，描述 Series.describe 和 Series.memory_usage 方法的功能
series_see_also_sub = dedent(
    """\
    Series.describe: Generate descriptive statistics of Series.
    Series.memory_usage: Memory usage of Series."""
)

# 定义一个字典，包含了用于描述 Series 的相关参数和信息
series_sub_kwargs = {
    "klass": "Series",
    "type_sub": "",
    "max_cols_sub": "",
    "show_counts_sub": show_counts_sub,  # 是否显示计数
    "examples_sub": series_examples_sub,  # Series 的示例用法
    "see_also_sub": series_see_also_sub,  # 参见部分，包含了相关的方法描述
    "version_added_sub": "\n.. versionadded:: 1.4.0\n",  # 版本添加信息，包含了版本号
}

# 创建一个多行字符串，描述 info 方法的作用和输出格式
INFO_DOCSTRING = dedent(
    """
    Print a concise summary of a {klass}.

    This method prints information about a {klass} including
    the index dtype{type_sub}, non-null values and memory usage.
    {version_added_sub}\
    
    Parameters
    ----------
"""
)
    # verbose : bool, optional
    #     是否打印完整的摘要信息。默认情况下，遵循 ``pandas.options.display.max_info_columns`` 的设置。

    # buf : writable buffer, defaults to sys.stdout
    #     输出的目标位置。默认情况下，输出到 sys.stdout。如果需要进一步处理输出，传递一个可写的缓冲区。

    {max_cols_sub}
    # memory_usage : bool, str, optional
    #     指定是否显示包括索引在内的 {klass} 元素的总内存使用情况。默认情况下，遵循 ``pandas.options.display.memory_usage`` 的设置。
    #
    #     True 总是显示内存使用情况。False 永不显示内存使用情况。
    #     'deep' 的值等同于 "使用深度检查的 True"。
    #     内存使用以人类可读的单位显示（使用二进制表示）。
    #     在没有深度检查的情况下，基于列的 dtype 和行数进行内存估算，假定相应 dtype 的值占用相同的内存量。
    #     使用深度内存检查时，实际执行内存使用情况计算，但会消耗计算资源。
    #     查看 :ref:`常见问题 <df-memory-usage>` 获取更多细节。
    
    {show_counts_sub}

    Returns
    -------
    None
        该方法打印一个 {klass} 的摘要信息并返回 None。

    See Also
    --------
    {see_also_sub}

    Examples
    --------
    {examples_sub}
    """
)

def _put_str(s: str | Dtype, space: int) -> str:
    """
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.info._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)


def _sizeof_fmt(num: float, size_qualifier: str) -> str:
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, "")
    '22.5 KB'

    >>> _sizeof_fmt(23028, "+")
    '22.5+ KB'
    """
    # Iterate through size units and convert bytes to appropriate unit
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f}{size_qualifier} {x}"
        num /= 1024.0
    # If size is extremely large, return in petabytes (PB)
    return f"{num:3.1f}{size_qualifier} PB"


def _initialize_memory_usage(
    memory_usage: bool | str | None = None,
) -> bool | str:
    """Get memory usage based on inputs and display options."""
    # If memory_usage is not specified, retrieve from display options
    if memory_usage is None:
        memory_usage = get_option("display.memory_usage")
    return memory_usage


class _BaseInfo(ABC):
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """

    data: DataFrame | Series
    memory_usage: bool | str

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> list[int] | Series:
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
        # Return memory usage formatted as a human-readable string
        return f"{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}\n"

    @property
    # Property for size qualifier

    def size_qualifier(self) -> str:
        """Size qualifier."""
        # Placeholder method for size qualifier; to be implemented in derived classes
    # 返回一个字符串，用于表示数据结构的内存占用情况的修饰符
    def size_qualifier(self) -> str:
        # 初始化空字符串，用于存储修饰符
        size_qualifier = ""
        # 如果存在内存使用情况
        if self.memory_usage:
            # 如果内存使用情况不是 "deep"
            if self.memory_usage != "deep":
                # size_qualifier 只是尽力而为的修饰符；不能保证捕捉所有情况
                # 例如，它可能会错过包含对象类型数据或者索引有内存使用限制的情况
                if (
                    "object" in self.dtype_counts
                    or self.data.index._is_memory_usage_qualified
                ):
                    # 如果数据包含对象类型或者索引有内存使用限制，修饰符为 "+"
                    size_qualifier = "+"
        # 返回修饰符字符串
        return size_qualifier

    # 抽象方法，用于渲染数据结构
    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        pass
class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(
        self,
        data: DataFrame,
        memory_usage: bool | str | None = None,
    ) -> None:
        # 初始化方法，接受 DataFrame 对象和内存使用选项作为参数
        self.data: DataFrame = data
        # 初始化 self.data 属性为传入的 DataFrame 对象
        self.memory_usage = _initialize_memory_usage(memory_usage)
        # 初始化内存使用选项，并保存在 self.memory_usage 属性中

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        # 返回 DataFrame 中不同数据类型的计数，作为字典
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        # 返回 DataFrame 中每列的数据类型
        return self.data.dtypes

    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        # 返回 DataFrame 的列名索引
        return self.data.columns

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        # 返回 DataFrame 中列的数量
        return len(self.ids)

    @property
    def non_null_counts(self) -> Series:
        """Sequence of non-null counts for all columns or column (if series)."""
        # 返回 DataFrame 中每列或单个列的非空值计数序列
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        # 计算 DataFrame 使用内存的字节数
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        # 渲染方法，将 DataFrameInfo 对象的信息输出到缓冲区
        printer = _DataFrameInfoPrinter(
            info=self,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)


class SeriesInfo(_BaseInfo):
    """
    Class storing series-specific info.
    """

    def __init__(
        self,
        data: Series,
        memory_usage: bool | str | None = None,
    ) -> None:
        # 初始化方法，接受 Series 对象和内存使用选项作为参数
        self.data: Series = data
        # 初始化 self.data 属性为传入的 Series 对象
        self.memory_usage = _initialize_memory_usage(memory_usage)
        # 初始化内存使用选项，并保存在 self.memory_usage 属性中

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        # 渲染方法，将 SeriesInfo 对象的信息输出到缓冲区
        if max_cols is not None:
            raise ValueError(
                "Argument `max_cols` can only be passed "
                "in DataFrame.info, not Series.info"
            )
        printer = _SeriesInfoPrinter(
            info=self,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)

    @property
    def non_null_counts(self) -> list[int]:
        # 返回 Series 中的非空值计数列表
        return [self.data.count()]

    @property
    def dtypes(self) -> Iterable[Dtype]:
        # 返回 Series 的数据类型列表
        return [self.data.dtypes]

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        # 返回 Series 中不同数据类型的计数，作为字典
        from pandas.core.frame import DataFrame
        return _get_dataframe_dtype_counts(DataFrame(self.data))

    @property
    def memory_usage_bytes(self) -> int:
        """Return the memory usage of the object in bytes.

        Returns
        -------
        memory_usage_bytes : int
            The total memory usage of the object in bytes.
        """
        # Determine if deep memory usage calculation is needed based on self.memory_usage
        deep = self.memory_usage == "deep"
        # Call the memory_usage method of self.data to compute memory usage
        return self.data.memory_usage(index=True, deep=deep)
class _InfoPrinterAbstract:
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        """Save dataframe info into buffer."""
        # 创建表格构建器实例
        table_builder = self._create_table_builder()
        # 获取表格行信息
        lines = table_builder.get_lines()
        if buf is None:  # 如果缓冲区为空，则默认输出到标准输出
            buf = sys.stdout
        # 将行信息写入缓冲区
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> _TableBuilderAbstract:
        """Create instance of table builder."""



class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        # 初始化实例变量
        self.info = info
        self.data = info.data
        self.verbose = verbose
        # 初始化最大列数
        self.max_cols = self._initialize_max_cols(max_cols)
        # 初始化是否显示非空计数
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        return get_option("display.max_info_rows")

    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: int | None) -> int:
        # 如果未指定最大列数，使用默认设置
        if max_cols is None:
            return get_option("display.max_info_columns")
        return max_cols

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        # 如果未指定是否显示非空计数，根据数据量和列数判断
        if show_counts is None:
            return bool(not self.exceeds_info_cols and not self.exceeds_info_rows)
        else:
            return show_counts
    def _create_table_builder(self) -> _DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """
        # 如果设置了详细输出（verbose=True），返回详细信息的表格构建器实例
        if self.verbose:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        # 如果明确设置为不详细输出（verbose=False），返回不详细信息的表格构建器实例
        elif self.verbose is False:  # specifically set to False, not necessarily None
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        # 如果没有明确设置详细输出，并且超出了信息列的限制，则返回不详细信息的表格构建器实例
        elif self.exceeds_info_cols:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        # 默认情况下返回详细信息的表格构建器实例
        else:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
# 表示一个用于打印系列信息的类，继承自_InfoPrinterAbstract类
class _SeriesInfoPrinter(_InfoPrinterAbstract):
    
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    # 初始化方法，接受SeriesInfo实例和两个可选参数verbose和show_counts
    def __init__(
        self,
        info: SeriesInfo,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        # 将传入的SeriesInfo实例保存到self.info属性中
        self.info = info
        # 从SeriesInfo实例中获取数据，并保存到self.data属性中
        self.data = info.data
        # 初始化self.show_counts属性，如果show_counts为None，则默认为True
        self.show_counts = self._initialize_show_counts(show_counts)

    # 根据verbosity参数创建相应的table builder实例
    def _create_table_builder(self) -> _SeriesTableBuilder:
        """
        Create instance of table builder based on verbosity.
        """
        if self.verbose or self.verbose is None:
            # 如果verbose为True或者None，返回详细信息的_SeriesTableBuilderVerbose实例
            return _SeriesTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        else:
            # 否则返回简略信息的_SeriesTableBuilderNonVerbose实例
            return _SeriesTableBuilderNonVerbose(info=self.info)

    # 初始化show_counts属性的辅助方法，如果show_counts为None，则返回True
    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return True
        else:
            return show_counts


# 表示信息表构建器的抽象类
class _TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    # 表示信息表的行列表
    _lines: list[str]
    # 表示基本信息的属性
    info: _BaseInfo

    @abstractmethod
    # 抽象方法，获取以行（字符串）列表形式的产品
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""

    @property
    # 获取数据（DataFrame或Series）的属性
    def data(self) -> DataFrame | Series:
        return self.info.data

    @property
    # 获取DataFrame每列的数据类型（Dtype）的可迭代属性
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    # 获取数据类型（Dtype）计数的映射属性
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    # 获取是否显示内存使用情况的属性
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
        return bool(self.info.memory_usage)

    @property
    # 获取包含适当大小限定符的内存使用字符串的属性
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
        return self.info.memory_usage_string

    @property
    # 获取非空计数的属性，可以是整数列表或Series
    def non_null_counts(self) -> list[int] | Series:
        return self.info.non_null_counts

    # 添加对象类型行到表中的方法
    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    # 添加索引范围行到表中的方法
    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
        self._lines.append(self.data.index._summary())

    # 添加包含DataFrame中存在的数据类型的摘要行到表中的方法
    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes = [
            f"{key}({val:d})" for key, val in sorted(self.dtype_counts.items())
        ]
        self._lines.append(f"dtypes: {', '.join(collected_dtypes)}")


# 表示DataFrame信息表构建器的类
class _DataFrameTableBuilder(_TableBuilderAbstract):
    """
    Concrete builder for DataFrame info table.
    """
    Abstract builder for dataframe info table.  # 抽象数据框信息表的构建器。

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.  # DataFrameInfo 的实例。

    """
    def __init__(self, *, info: DataFrameInfo) -> None:
        self.info: DataFrameInfo = info  # 初始化函数，接收一个 DataFrameInfo 的实例并赋值给 self.info。

    def get_lines(self) -> list[str]:
        self._lines = []  # 初始化空列表 _lines 用于存储信息表的行。
        if self.col_count == 0:
            self._fill_empty_info()  # 如果列数为0，填充空数据框的信息。
        else:
            self._fill_non_empty_info()  # 否则填充非空数据框的信息。
        return self._lines  # 返回填充后的信息表行列表。

    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_object_type_line()  # 添加对象类型行。
        self.add_index_range_line()  # 添加索引范围行。
        self._lines.append(f"Empty {type(self.data).__name__}\n")  # 在信息表中添加空数据框的信息行。

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        # 抽象方法，用于填充非空数据框的信息表，具体实现在子类中完成。

    @property
    def data(self) -> DataFrame:
        """DataFrame."""
        return self.info.data  # 返回 DataFrameInfo 对象中的数据框。

    @property
    def ids(self) -> Index:
        """Dataframe columns."""
        return self.info.ids  # 返回 DataFrameInfo 对象中的列索引。

    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        return self.info.col_count  # 返回数据框的列数，即需要总结的列数。

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")  # 添加包含内存使用量的信息行。
class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        # 添加对象类型行到信息表
        self.add_object_type_line()
        # 添加索引范围行到信息表
        self.add_index_range_line()
        # 添加列汇总行到信息表
        self.add_columns_summary_line()
        # 添加数据类型行到信息表
        self.add_dtypes_line()
        # 如果显示内存使用情况，添加内存使用行到信息表
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self) -> None:
        # 将列汇总添加到信息表中
        self._lines.append(self.ids._summary(name="Columns"))


class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        # 返回表头列的宽度列表（只包括标题）
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
        # 获取包含表头和实际内容的列的宽度
        body_column_widths = self._get_body_column_widths()
        return [
            max(*widths)
            for widths in zip(self.header_column_widths, body_column_widths)
        ]

    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
        # 获取表格内容列的宽度
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max(len(x) for x in col) for col in strcols]

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        # 如果包含计数，则生成包含计数的行数据的迭代器，否则生成不包含计数的行数据的迭代器
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        # 构建并添加表头行到信息表
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(self.headers, self.gross_column_widths)
            ]
        )
        self._lines.append(header_line)
    def add_separator_line(self) -> None:
        # 使用 header_column_widths 和 gross_column_widths 中的数据，生成分隔线字符串
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                    self.header_column_widths, self.gross_column_widths
                )
            ]
        )
        # 将生成的分隔线字符串添加到 _lines 列表中
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        # 遍历 strrows 中的每一行数据
        for row in self.strrows:
            # 使用 gross_column_widths 中的数据，格式化每一列数据为字符串，并用 SPACING 连接起来
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(row, self.gross_column_widths)
                ]
            )
            # 将格式化后的行数据添加到 _lines 列表中
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        # 遍历 non_null_counts 中的每一个计数值，生成字符串格式的非空计数描述
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        # 遍历 dtypes 中的每一种数据类型，使用 pprint_thing 函数生成字符串描述
        for dtype in self.dtypes:
            yield pprint_thing(dtype)
class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: DataFrameInfo,
        with_counts: bool,
    ) -> None:
        # 初始化方法，设置对象的属性
        self.info = info  # 存储 DataFrameInfo 对象的引用
        self.with_counts = with_counts  # 标记是否包含计数信息的布尔值
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())  # 生成行数据的字符串表示
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()  # 获取列的总宽度列表

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()  # 添加对象类型行
        self.add_index_range_line()  # 添加索引范围行
        self.add_columns_summary_line()  # 添加列总结行
        self.add_header_line()  # 添加表头行
        self.add_separator_line()  # 添加分隔线
        self.add_body_lines()  # 添加主体行
        self.add_dtypes_line()  # 添加数据类型行
        if self.display_memory_usage:  # 如果显示内存使用信息
            self.add_memory_usage_line()  # 添加内存使用行

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return [" # ", "Column", "Non-Null Count", "Dtype"]  # 返回带计数的表头列表
        return [" # ", "Column", "Dtype"]  # 返回不带计数的表头列表

    def add_columns_summary_line(self) -> None:
        # 添加列总结行，包括总列数信息
        self._lines.append(f"Data columns (total {self.col_count} columns):")

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from zip(
            self._gen_line_numbers(),  # 生成行号
            self._gen_columns(),  # 生成列名
            self._gen_dtypes(),  # 生成数据类型
        )

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_line_numbers(),  # 生成行号
            self._gen_columns(),  # 生成列名
            self._gen_non_null_counts(),  # 生成非空计数
            self._gen_dtypes(),  # 生成数据类型
        )

    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
        for i, _ in enumerate(self.ids):
            yield f" {i}"  # 生成带有序号的字符串表示

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)  # 使用 pprint_thing 格式化输出列名


class _SeriesTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesInfo.
        Instance of SeriesInfo.
    """

    def __init__(self, *, info: SeriesInfo) -> None:
        self.info: SeriesInfo = info  # 存储 SeriesInfo 对象的引用

    def get_lines(self) -> list[str]:
        self._lines = []  # 初始化行列表
        self._fill_non_empty_info()  # 填充非空信息到表格
        return self._lines  # 返回填充后的行列表

    @property
    def data(self) -> Series:
        """Series."""
        return self.info.data  # 返回 SeriesInfo 中的数据对象

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")  # 添加包含内存使用信息的行

    @abstractmethod
    # 抽象方法，子类需实现
    # 定义一个方法 `_fill_non_empty_info`，用于向信息表中添加与非空系列相关的行
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    """
    Series info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        # 添加非空系列相关信息到信息表中
        self.add_object_type_line()
        # 添加对象类型行到信息表中
        self.add_index_range_line()
        # 添加索引范围行到信息表中
        self.add_dtypes_line()
        # 添加数据类型行到信息表中
        if self.display_memory_usage:
            self.add_memory_usage_line()
            # 如果需要显示内存使用情况，则添加内存使用行


class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    """
    Series info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: SeriesInfo,
        with_counts: bool,
    ) -> None:
        self.info = info
        # 存储系列信息对象
        self.with_counts = with_counts
        # 是否包含计数信息
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        # 生成详细行数据的序列
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()
        # 计算总列宽度的序列

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        # 添加非空系列相关信息到信息表中
        self.add_object_type_line()
        # 添加对象类型行到信息表中
        self.add_index_range_line()
        # 添加索引范围行到信息表中
        self.add_series_name_line()
        # 添加系列名称行到信息表中
        self.add_header_line()
        # 添加表头行到信息表中
        self.add_separator_line()
        # 添加分隔线到信息表中
        self.add_body_lines()
        # 添加主体行到信息表中
        self.add_dtypes_line()
        # 添加数据类型行到信息表中
        if self.display_memory_usage:
            self.add_memory_usage_line()
            # 如果需要显示内存使用情况，则添加内存使用行

    def add_series_name_line(self) -> None:
        # 添加系列名称行到信息表中
        self._lines.append(f"Series name: {self.data.name}")

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        # 返回详细表格列的表头名称
        if self.with_counts:
            return ["Non-Null Count", "Dtype"]
        return ["Dtype"]

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        # 生成不带计数的主体数据字符串表示的迭代器
        yield from self._gen_dtypes()

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        # 生成带计数的主体数据字符串表示的迭代器
        yield from zip(
            self._gen_non_null_counts(),
            self._gen_dtypes(),
        )


def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
    # 根据数据帧的数据类型统计出现次数，并按数据类型名称进行分组
    return df.dtypes.value_counts().groupby(lambda x: x.name).sum()
```