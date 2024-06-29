# `D:\src\scipysrc\pandas\pandas\io\parsers\readers.py`

```
"""
Module contains tools for processing files into DataFrames or other objects

GH#48849 provides a convenient way of deprecating keyword arguments
"""

# 从未来导入的语句，允许使用 Python 3.10 中的注解
from __future__ import annotations

# 导入 collections 模块中的 abc 和 defaultdict 类
from collections import (
    abc,
    defaultdict,
)

# 导入 csv、sys 和 textwrap 中的 fill 函数
import csv
import sys
from textwrap import fill

# 导入类型相关的模块和函数
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypedDict,
    overload,
)

# 导入警告相关的模块
import warnings

# 导入 numpy 库
import numpy as np

# 导入 pandas 库的私有函数和异常
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
    AbstractMethodError,
    ParserWarning,
)

# 导入 pandas 库的装饰器和异常处理函数
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

# 导入 pandas 库的数据类型检查函数
from pandas.core.dtypes.common import (
    is_file_like,
    is_float,
    is_integer,
    is_list_like,
    pandas_dtype,
)

# 导入 pandas 库的核心数据结构
from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs

# 导入 pandas 库的 IO 相关模块和函数
from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)

# 导入 pandas 库的 Arrow 解析器包装器
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper

# 导入 pandas 库的基础解析器和默认解析器设置
from pandas.io.parsers.base_parser import (
    ParserBase,
    is_index_col,
    parser_defaults,
)

# 导入 pandas 库的 C 解析器包装器
from pandas.io.parsers.c_parser_wrapper import CParserWrapper

# 导入 pandas 库的 Python 解析器相关类
from pandas.io.parsers.python_parser import (
    FixedWidthFieldParser,
    PythonParser,
)

# 如果是类型检查模式，导入额外的类型
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        CSVEngine,
        DtypeArg,
        DtypeBackend,
        FilePath,
        HashableT,
        IndexLabel,
        ReadCsvBuffer,
        Self,
        StorageOptions,
        Unpack,
        UsecolsArgType,
    )
    class _read_shared(TypedDict, Generic[HashableT], total=False):
        # 定义一个类型字典 _read_shared，用于描述 read_csv/fwf/table 函数的共享注解
        # 注意：需要与实现中的注解保持同步
    
        sep: str | None | lib.NoDefault
        # 分隔符，可以是字符串或者空，或者 lib.NoDefault
    
        delimiter: str | None | lib.NoDefault
        # 分隔符，可以是字符串或者空，或者 lib.NoDefault
    
        header: int | Sequence[int] | None | Literal["infer"]
        # 表头行数，可以是整数、整数序列、空，或者 "infer"
    
        names: Sequence[Hashable] | None | lib.NoDefault
        # 列名列表，可以是哈希可变类型的序列或者空，或者 lib.NoDefault
    
        index_col: IndexLabel | Literal[False] | None
        # 索引列标签，可以是索引标签、False 或者空
    
        usecols: UsecolsArgType
        # 使用的列，类型为 UsecolsArgType
    
        dtype: DtypeArg | None
        # 数据类型，可以是 DtypeArg 或者空
    
        engine: CSVEngine | None
        # CSV 引擎，可以是 CSVEngine 类型或者空
    
        converters: Mapping[HashableT, Callable] | None
        # 转换器，可以是哈希可变类型到可调用对象的映射或者空
    
        true_values: list | None
        # 真值列表，可以是列表或者空
    
        false_values: list | None
        # 假值列表，可以是列表或者空
    
        skipinitialspace: bool
        # 是否跳过起始空格，布尔类型
    
        skiprows: list[int] | int | Callable[[Hashable], bool] | None
        # 跳过行数，可以是整数列表、整数、或者接受哈希类型并返回布尔值的可调用对象，或者空
    
        skipfooter: int
        # 跳过尾部行数，整数类型
    
        nrows: int | None
        # 读取的行数，可以是整数或者空
    
        na_values: (
            Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None
        )
        # 缺失值，可以是哈希类型、哈希类型的可迭代对象、哈希类型到可迭代对象的映射或者空
    
        keep_default_na: bool
        # 保留默认缺失值，布尔类型
    
        na_filter: bool
        # 过滤缺失值，布尔类型
    
        skip_blank_lines: bool
        # 跳过空行，布尔类型
    
        parse_dates: bool | Sequence[Hashable] | None
        # 解析日期，可以是布尔类型、哈希类型序列或者空
    
        date_format: str | dict[Hashable, str] | None
        # 日期格式，可以是字符串、哈希类型到字符串的映射或者空
    
        dayfirst: bool
        # 是否优先日在前，布尔类型
    
        cache_dates: bool
        # 缓存日期，布尔类型
    
        compression: CompressionOptions
        # 压缩选项，CompressionOptions 类型
    
        thousands: str | None
        # 千位分隔符，可以是字符串或者空
    
        decimal: str
        # 十进制符号，字符串类型
    
        lineterminator: str | None
        # 行终止符，可以是字符串或者空
    
        quotechar: str
        # 引用符，字符串类型
    
        quoting: int
        # 引用规则，整数类型
    
        doublequote: bool
        # 双引用符，布尔类型
    
        escapechar: str | None
        # 转义符，可以是字符串或者空
    
        comment: str | None
        # 注释符，可以是字符串或者空
    
        encoding: str | None
        # 编码，可以是字符串或者空
    
        encoding_errors: str | None
        # 编码错误处理，可以是字符串或者空
    
        dialect: str | csv.Dialect | None
        # 方言，可以是字符串、csv.Dialect 类型或者空
    
        on_bad_lines: str
        # 在错误行上的处理方式，字符串类型
    
        low_memory: bool
        # 低内存模式，布尔类型
    
        memory_map: bool
        # 内存映射，布尔类型
    
        float_precision: Literal["high", "legacy", "round_trip"] | None
        # 浮点数精度，可以是 "high"、"legacy"、"round_trip" 或者空
    
        storage_options: StorageOptions | None
        # 存储选项，可以是 StorageOptions 类型或者空
    
        dtype_backend: DtypeBackend | lib.NoDefault
        # 数据类型后端，可以是 DtypeBackend 类型或者 lib.NoDefault
else:
    # 如果条件不满足（即为else分支），将 _read_shared 设置为 dict 类型
    _read_shared = dict


_doc_read_csv_and_table = (
    r"""
{summary}

Also supports optionally iterating or breaking of the file
into chunks.

Additional help can be found in the online docs for
`IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
    expected. A local file could be: file://localhost/path/to/table.csv.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method, such as
    a file handle (e.g. via builtin ``open`` function) or ``StringIO``.
sep : str, default {_default_sep}
    Character or regex pattern to treat as the delimiter. If ``sep=None``, the
    C engine cannot automatically detect
    the separator, but the Python parsing engine can, meaning the latter will
    be used and automatically detect the separator from only the first valid
    row of the file by Python's builtin sniffer tool, ``csv.Sniffer``.
    In addition, separators longer than 1 character and different from
    ``'\s+'`` will be interpreted as regular expressions and will also force
    the use of the Python parsing engine. Note that regex delimiters are prone
    to ignoring quoted data. Regex example: ``'\r\t'``.
delimiter : str, optional
    Alias for ``sep``.
header : int, Sequence of int, 'infer' or None, default 'infer'
    Row number(s) containing column labels and marking the start of the
    data (zero-indexed). Default behavior is to infer the column names: if no ``names``
    are passed the behavior is identical to ``header=0`` and column
    names are inferred from the first line of the file, if column
    names are passed explicitly to ``names`` then the behavior is identical to
    ``header=None``. Explicitly pass ``header=0`` to be able to
    replace existing names. The header can be a list of integers that
    specify row locations for a :class:`~pandas.MultiIndex` on the columns
    e.g. ``[0, 1, 3]``. Intervening rows that are not specified will be
    skipped (e.g. 2 in this example is skipped). Note that this
    parameter ignores commented lines and empty lines if
    ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
    data rather than the first line of the file.

    When inferred from the file contents, headers are kept distinct from
    each other by renaming duplicate names with a numeric suffix of the form
    ``".{{count}}"`` starting from 1, e.g. ``"foo"`` and ``"foo.1"``.
    Empty headers are named ``"Unnamed: {{i}}"`` or ``"Unnamed: {{i}}_level_{{level}}"``
    in the case of MultiIndex columns.
names : Sequence of Hashable, optional
    Sequence of column labels to apply. If the file contains a header row,
    then you should explicitly pass ``header=0`` to override the column names.
    Duplicates in this list are not allowed.


注释：


# 如果你希望显式地指定 `header=0` 来覆盖列名。
# 这个列表中不允许重复项。


这段注释解释了在某个上下文中需要明确传递 `header=0` 参数来覆盖列名，以及关于列表中不允许重复项的限制。
# `index_col`参数指定作为行标签的列，可以是列标签或列索引的序列。如果给定了序列，将为行标签创建MultiIndex。
# 注意：可以使用`index_col=False`来强制pandas不使用第一列作为索引，例如，当文件格式不正确，每行末尾有分隔符时。
index_col : Hashable, Sequence of Hashable or False, optional

# `usecols`参数指定要选择的列的子集，可以通过列标签或列索引指定。如果是列表形式，所有元素必须是位置索引（即文档列的整数索引）或者与用户在`names`参数中提供的列名对应的字符串。
# 如果给定了`names`，则不考虑文档的标题行。例如，有效的列表形式的`usecols`参数可以是`[0, 1, 2]`或`['foo', 'bar', 'baz']`。元素的顺序被忽略，因此`usecols=[0, 1]`等同于`[1, 0]`。
# 若要保留`data`中的元素顺序实例化`pd.DataFrame`，使用`pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]`来指定`['foo', 'bar']`顺序的列，或者`pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]`指定`['bar', 'foo']`顺序的列。
# 如果是可调用对象，将对列名应用该可调用函数，返回在该函数评估为`True`时的列名。一个有效的可调用参数的示例是`lambda x: x.upper() in ['AAA', 'BBB', 'DDD']`。
# 使用此参数可以显著提高解析速度和降低内存使用。
usecols : Sequence of Hashable or Callable, optional

# `dtype`参数指定要应用于整个数据集或单个列的数据类型。例如，`{'a': np.float64, 'b': np.int32, 'c': 'Int64'}`。
# 使用`str`或`object`配合合适的`na_values`设置可以保留并且不解释`dtype`。
# 如果指定了`converters`，它们将被应用于`dtype`转换的替代。
# 添加于1.5.0版本：支持`defaultdict`，指定`defaultdict`作为输入，其默认值决定未显式列出的列的`dtype`。
dtype : dtype or dict of {Hashable: dtype}, optional

# `engine`参数指定要使用的解析引擎。C和pyarrow引擎更快，而python引擎目前更完整支持特性。
# 目前仅支持pyarrow引擎的多线程。
# 添加于1.4.0版本：添加了'pyarrow'引擎作为*实验性*引擎，某些功能可能不受支持或在此引擎下无法正常工作。
engine : {'c', 'python', 'pyarrow'}, optional

# `converters`参数指定要在指定列中转换值的函数。键可以是列标签或列索引。
converters : dict of {Hashable: Callable}, optional
# 定义可作为“True”的值列表，除了大小写不敏感的“True”变体外
true_values : list, optional
# 定义可作为“False”的值列表，除了大小写不敏感的“False”变体外
false_values : list, optional
# 是否跳过分隔符后的空格，默认为False
skipinitialspace : bool, default False
# 要跳过的行数（从0开始计数），可以是int、int列表或可调用对象
# 如果是可调用对象，将针对行索引进行评估，返回True表示应跳过该行，否则为False
# 例如，有效的可调用参数可以是 lambda x: x in [0, 2]
skiprows : int, list of int or Callable, optional
# 要跳过文件底部的行数，默认为0，不支持使用engine='c'
skipfooter : int, default 0
# 要读取的文件行数，对于大文件的片段读取很有用
nrows : int, optional
# 识别为“NA”/“NaN”的附加字符串，可以是单个哈希值、哈希值的可迭代对象或每列特定“NA”值的字典
# 默认情况下，以下值被解释为“NaN”：" ".join(sorted(STR_NA_VALUES))
na_values : Hashable, Iterable of Hashable or dict of {{Hashable : Iterable}}, optional
# 是否保留默认的“NaN”值进行数据解析
# 取决于是否传递了na_values参数，行为如下：
# - 如果keep_default_na为True，并且指定了na_values，则na_values将追加到用于解析的默认“NaN”值。
# - 如果keep_default_na为True，并且未指定na_values，则只使用默认的“NaN”值进行解析。
# - 如果keep_default_na为False，并且指定了na_values，则只使用指定的na_values作为“NaN”值进行解析。
# - 如果keep_default_na为False，并且未指定na_values，则不会解析任何字符串为“NaN”。
# 注意，如果na_filter参数传入False，则将忽略keep_default_na和na_values参数。
keep_default_na : bool, default True
# 是否检测缺失值标记（空字符串和na_values的值）。
# 在没有任何NA值的数据中，传递na_filter=False可以提高读取大文件的性能。
na_filter : bool, default True
# 是否跳过空白行，而不将其解释为“NaN”值
skip_blank_lines : bool, default True
# 是否尝试解析日期。行为如下：
# - 如果为True，则尝试解析索引。
# - 如果为None，并且指定了date_format，则表现为True。
# - 如果为int列表或名称列表，例如[1, 2, 3]，则尝试将每个指定的列解析为单独的日期列。
# 如果无法将列或索引表示为datetime数组，
parse_dates : bool, None, list of Hashable, default None
    say because of an unparsable value or a mixture of timezones, the column
    or index will be returned unaltered as an ``object`` data type. For
    non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime` after
    :func:`~pandas.read_csv`.

    Note: A fast-path exists for iso8601-formatted dates.
# date_format 是一个可选参数，可以是字符串或者列名到日期格式的字典。用于在 parse_dates 启用时解析日期。
# 格式可以是 strftime 格式字符串，例如 "%d/%m/%Y"。详见 strftime 文档了解更多选项，
# 注意 "%f" 可以解析纳秒级别的时间。
# 还可以使用以下特殊字符串：
# - "ISO8601"，用于解析 ISO8601 格式的时间字符串（不一定是完全相同的格式）。
# - "mixed"，用于推断每个元素的格式。这种方式风险较大，通常需要与 dayfirst 一起使用。
# .. versionadded:: 2.0.0

# dayfirst 是一个布尔值，默认为 False。用于解析 DD/MM 格式的日期，国际和欧洲格式。

# cache_dates 是一个布尔值，默认为 True。如果为 True，会使用一个缓存来存储已转换的日期，以应用 datetime 转换。
# 当解析重复的日期字符串（特别是带有时区偏移的字符串）时，可能会显著提高速度。

# iterator 是一个布尔值，默认为 False。如果为 True，则返回一个 TextFileReader 对象，用于迭代或者通过 get_chunk() 获取数据块。

# chunksize 是一个整数，可选参数。表示每个数据块从文件中读取的行数。传递此值将导致函数返回一个 TextFileReader 对象，用于迭代。
# 详见 IO Tools 文档了解 iterator 和 chunksize 的更多信息。

# {decompression_options}
# .. versionchanged:: 1.4.0
# 增加了对 Zstandard 压缩的支持。

# thousands 是一个长度为 1 的字符串，可选参数。表示数字值中的千位分隔符的字符。

# decimal 是一个长度为 1 的字符串，可选参数，默认为 '.'。表示用作小数点的字符（例如，对于欧洲数据可以使用 ','）。

# lineterminator 是一个长度为 1 的字符串，可选参数。表示用于表示换行的字符。仅在使用 C 解析器时有效。

# quotechar 是一个长度为 1 的字符串，可选参数。表示引用项的起始和结束字符。引用项可以包括分隔符，并且分隔符将被忽略。

# quoting 是一个枚举值，可选参数。控制字段引用行为。可以是 0 或者 csv.QUOTE_MINIMAL，1 或者 csv.QUOTE_ALL，2 或者 csv.QUOTE_NONNUMERIC，3 或者 csv.QUOTE_NONE。
# 默认值为 csv.QUOTE_MINIMAL（即 0），表示只有包含特殊字符的字段被引用（例如，在 quotechar、delimiter 或 lineterminator 中定义的字符）。

# doublequote 是一个布尔值，可选参数，默认为 True。当指定 quotechar 并且 quoting 不是 QUOTE_NONE 时，
# 表示是否将两个连续的 quotechar 元素视为一个 quotechar 元素的内容。

# escapechar 是一个长度为 1 的字符串，可选参数。表示用于转义其他字符的字符。

# comment 是一个长度为 1 的字符串，可选参数。表示该行其余部分不应被解析的字符。
    If found at the beginning
    of a line, the line will be ignored altogether. This parameter must be a
    single character. Like empty lines (as long as ``skip_blank_lines=True``),
    fully commented lines are ignored by the parameter ``header`` but not by
    ``skiprows``. For example, if ``comment='#'``, parsing
    ``#empty\\na,b,c\\n1,2,3`` with ``header=0`` will result in ``'a,b,c'`` being
    treated as the header.



    如果发现在行的开头，
    则整行将被完全忽略。此参数必须是
    单个字符。与空行一样（只要 ``skip_blank_lines=True``），
    完全注释的行由参数 ``header`` 忽略，但不受
    ``skiprows`` 影响。例如，如果 ``comment='#'``，解析
    ``#empty\\na,b,c\\n1,2,3`` 时，使用 ``header=0`` 将导致 ``'a,b,c'`` 被
    视为表头。
# 设置编码格式，用于读写 UTF 编码的文件，默认为 'utf-8'
encoding : str, optional, default 'utf-8'
# 编码错误处理方式，默认为 'strict'，即遇到编码错误时引发异常
encoding_errors : str, optional, default 'strict'
# CSV 文件的方言参数，可覆盖以下参数的默认值：delimiter、doublequote、escapechar、skipinitialspace、quotechar 和 quoting
dialect : str or csv.Dialect, optional
# 当遇到错误行（字段过多的行）时的处理方式，可选值有 'error'、'warn'、'skip' 或者可调用的函数
on_bad_lines : {{'error', 'warn', 'skip'}} or Callable, default 'error'
# 内存使用低处理标志，设置为 True 可以在解析时使用更低的内存，但可能导致混合类型推断
low_memory : bool, default True
# 是否在内存中映射文件，适用于提高性能，特别是在处理大文件时
memory_map : bool, default False
# 浮点数精度控制，可选值有 'high'、'legacy'、'round_trip'
float_precision : {{'high', 'legacy', 'round_trip'}}, optional
    # 指定C引擎在处理浮点数值时应该使用的转换器选项。
    # 可选项包括 `None` 表示普通的转换器，`'high'` 表示高精度的转换器，
    # `'legacy'` 表示原始的较低精度的pandas转换器，
    # 以及 `'round_trip'` 表示往返转换器。
{storage_options}
# 存储选项，这里可能包含有关数据存储的额外参数或配置信息

dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
    # 结果DataFrame应用的后端数据类型（仍处于实验阶段）。行为如下：
    # 
    # * ``"numpy_nullable"``: 返回支持可空数据类型的DataFrame（默认）。
    # * ``"pyarrow"``: 返回支持pyarrow的可空ArrowDtype的DataFrame。
    # 
    # .. versionadded:: 2.0
    返回
    -------
    DataFrame或TextFileReader
        返回一个以逗号分隔的值（csv）文件作为带有标记轴的二维数据结构。

    See Also
    --------
    DataFrame.to_csv : 将DataFrame写入逗号分隔的值（csv）文件。
    {see_also_func_name} : {see_also_func_summary}
    read_fwf : 将固定宽度格式的表格行读入DataFrame。

    Examples
    --------
    >>> pd.{func_name}('data.csv')  # doctest: +SKIP
       Name  Value
    0   foo      1
    1   bar      2
    2  #baz      3

    索引和标题可以通过`index_col`和`header`参数指定。

    >>> pd.{func_name}('data.csv', header=None)  # doctest: +SKIP
          0      1
    0  Name  Value
    1   foo      1
    2   bar      2
    3  #baz      3

    >>> pd.{func_name}('data.csv', index_col='Value')  # doctest: +SKIP
           Name
    Value
    1       foo
    2       bar
    3      #baz

    列类型会被推断，但可以使用dtype参数显式指定。

    >>> pd.{func_name}('data.csv', dtype={{'Value': float}})  # doctest: +SKIP
       Name  Value
    0   foo    1.0
    1   bar    2.0
    2  #baz    3.0

    真值、假值和NA值以及千位分隔符具有默认值，但也可以显式指定。提供您想要的值作为字符串或字符串列表！

    >>> pd.{func_name}('data.csv', na_values=['foo', 'bar'])  # doctest: +SKIP
       Name  Value
    0   NaN      1
    1   NaN      2
    2  #baz      3

    使用`comment`参数可以跳过输入文件中的注释行。

    >>> pd.{func_name}('data.csv', comment='#')  # doctest: +SKIP
      Name  Value
    0  foo      1
    1  bar      2

    默认情况下，包含日期的列将被读取为``object``而不是``datetime``。

    >>> df = pd.{func_name}('tmp.csv')  # doctest: +SKIP

    >>> df  # doctest: +SKIP
       col 1       col 2            col 3
    0     10  10/04/2018  Sun 15 Jan 2023
    1     20  15/04/2018  Fri 12 May 2023

    >>> df.dtypes  # doctest: +SKIP
    col 1     int64
    col 2    object
    col 3    object
    dtype: object

    可以通过使用`parse_dates`和`date_format`参数将特定列解析为日期。

    >>> df = pd.{func_name}(
    ...     'tmp.csv',
    ...     parse_dates=[1, 2],
    ...     date_format={{'col 2': '%d/%m/%Y', 'col 3': '%a %d %b %Y'}},
    ... )  # doctest: +SKIP

    >>> df.dtypes  # doctest: +SKIP
    col 1             int64
    col 2    datetime64[ns]
    col 3    datetime64[ns]
    dtype: object
"""
)


class _C_Parser_Defaults(TypedDict):
    na_filter: Literal[True]
    low_memory: Literal[True]
    memory_map: Literal[False]
    float_precision: None


_c_parser_defaults: _C_Parser_Defaults = {
    "na_filter": True,
    "low_memory": True,
    # 默认设置为True的解析器参数
    "memory_map": False,
    # 默认设置为False的解析器参数
    float_precision: None
    # 浮点精度设置为None
}
    # 指定参数 memory_map，表示不使用内存映射文件来加载数据
    "memory_map": False,
    # 指定参数 float_precision，表示没有指定浮点数的精度
    "float_precision": None,
}

# 类型定义，指定了 _Fwf_Defaults 的结构
class _Fwf_Defaults(TypedDict):
    colspecs: Literal["infer"]
    infer_nrows: Literal[100]
    widths: None

# 默认设置 _fwf_defaults，使用 _Fwf_Defaults 结构
_fwf_defaults: _Fwf_Defaults = {"colspecs": "infer", "infer_nrows": 100, "widths": None}

# 不支持的参数集合
_c_unsupported = {"skipfooter"}

# Python 不支持的参数集合
_python_unsupported = {"low_memory", "float_precision"}

# PyArrow 不支持的参数集合
_pyarrow_unsupported = {
    "skipfooter",
    "float_precision",
    "chunksize",
    "comment",
    "nrows",
    "thousands",
    "memory_map",
    "dialect",
    "quoting",
    "lineterminator",
    "converters",
    "iterator",
    "dayfirst",
    "skipinitialspace",
    "low_memory",
}

# 函数重载，验证参数是否为整数或可安全转换为整数
@overload
def validate_integer(name: str, val: None, min_val: int = ...) -> None: ...

@overload
def validate_integer(name: str, val: float, min_val: int = ...) -> int: ...

@overload
def validate_integer(name: str, val: int | None, min_val: int = ...) -> int | None: ...

# 实际的函数定义，用于验证整数参数
def validate_integer(
    name: str, val: int | float | None, min_val: int = 0
) -> int | None:
    """
    Checks whether the 'name' parameter for parsing is either
    an integer OR float that can SAFELY be cast to an integer
    without losing accuracy. Raises a ValueError if that is
    not the case.

    Parameters
    ----------
    name : str
        Parameter name (used for error reporting)
    val : int or float
        The value to check
    min_val : int
        Minimum allowed value (val < min_val will result in a ValueError)
    """
    if val is None:
        return val

    msg = f"'{name:s}' must be an integer >={min_val:d}"
    if is_float(val):  # 检查是否为浮点数
        if int(val) != val:  # 如果浮点数无法精确转换为整数，则抛出 ValueError
            raise ValueError(msg)
        val = int(val)  # 将浮点数转换为整数
    elif not (is_integer(val) and val >= min_val):  # 检查是否为整数且大于等于最小值
        raise ValueError(msg)

    return int(val)  # 返回整数值

# 函数定义，验证参数列表是否包含重复项或无效数据类型
def _validate_names(names: Sequence[Hashable] | None) -> None:
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
    names : array-like or None
        An array containing a list of the names used for the output DataFrame.

    Raises
    ------
    ValueError
        If names are not unique or are not ordered (e.g. set).
    """
    if names is not None:
        if len(names) != len(set(names)):  # 检查是否存在重复的名称
            raise ValueError("Duplicate names are not allowed.")
        if not (
            is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
        ):  # 检查是否为有序集合类型
            raise ValueError("Names should be an ordered collection.")

# 函数定义，通用的文件读取函数
def _read(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], kwds
) -> DataFrame | TextFileReader:
    """Generic reader of line files."""
    # 如果传递了 date_format 但 parse_dates=False，则不应解析日期
    if kwds.get("parse_dates", None) is None:
        if kwds.get("date_format", None) is None:
            kwds["parse_dates"] = False
        else:
            kwds["parse_dates"] = True

    # 提取部分参数（将 chunksize 传递）
    iterator = kwds.get("iterator", False)
    chunksize = kwds.get("chunksize", None)

    # 检查 encoding_errors 的类型
    errors = kwds.get("encoding_errors", "strict")
    if not isinstance(errors, str):
        raise ValueError(
            f"encoding_errors must be a string, got {type(errors).__name__}"
        )

    # 如果 engine 是 'pyarrow'
    if kwds.get("engine") == "pyarrow":
        # 如果 iterator 设置为 True，抛出异常
        if iterator:
            raise ValueError(
                "The 'iterator' option is not supported with the 'pyarrow' engine"
            )

        # 如果 chunksize 不是 None，抛出异常
        if chunksize is not None:
            raise ValueError(
                "The 'chunksize' option is not supported with the 'pyarrow' engine"
            )
    else:
        # 对于非 'pyarrow' 引擎，验证 chunksize 必须是正整数
        chunksize = validate_integer("chunksize", chunksize, 1)

    # 获取 nrows 参数
    nrows = kwds.get("nrows", None)

    # 检查 names 是否有重复
    _validate_names(kwds.get("names", None))

    # 创建 TextFileReader 对象作为解析器
    parser = TextFileReader(filepath_or_buffer, **kwds)

    # 如果 chunksize 或 iterator 任一不为 None，直接返回 parser 对象
    if chunksize or iterator:
        return parser

    # 使用 parser 对象进行上下文管理
    with parser:
        # 调用 parser 的 read 方法读取 nrows 行数据并返回
        return parser.read(nrows)
@overload
# 函数重载定义：读取 CSV 文件内容为 TextFileReader 对象，使用迭代器模式
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
# 函数重载定义：读取 CSV 文件内容为 TextFileReader 对象，指定块大小
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: bool = ...,
    chunksize: int,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
# 函数重载定义：读取 CSV 文件内容为 DataFrame 对象，禁用迭代模式
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame: ...


@overload
# 函数重载定义：根据参数读取 CSV 文件内容，返回 DataFrame 或 TextFileReader 对象
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: bool = ...,
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame | TextFileReader: ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_csv",
        summary="Read a comma-separated values (csv) file into DataFrame.",
        see_also_func_name="read_table",
        see_also_func_summary="Read general delimited file into DataFrame.",
        _default_sep="','",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
# 将文档字符串追加到 read_csv 函数，描述其功能和使用方法
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[HashableT, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    # 定义 CSV 文件中行终止符的字符串表示，默认为 None
    quotechar: str = '"',
    # 定义用于引用字段的字符，默认为双引号
    quoting: int = csv.QUOTE_MINIMAL,
    # 定义引用模式，以处理包含特殊字符的字段，默认为最小化引用
    doublequote: bool = True,
    # 定义是否允许在字段中使用双引号作为引用的一部分，默认为 True
    escapechar: str | None = None,
    # 定义用于转义特殊字符的转义字符，默认为 None
    comment: str | None = None,
    # 定义用于表示注释行的字符，默认为 None
    encoding: str | None = None,
    # 定义 CSV 文件的编码方式，默认为 None
    encoding_errors: str | None = "strict",
    # 定义编码错误处理方式，默认为严格处理
    dialect: str | csv.Dialect | None = None,
    # 定义 CSV 方言，指定不同的分隔符和引用字符，默认为 None
    # Error Handling
    on_bad_lines: str = "error",
    # 定义在遇到错误行时的处理方式，默认为 "error"
    # Internal
    low_memory: bool = _c_parser_defaults["low_memory"],
    # 定义是否以低内存模式处理数据，默认从内部设置获取
    memory_map: bool = False,
    # 定义是否通过内存映射文件处理数据，默认为 False
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    # 定义浮点数精度控制选项，默认为 None
    storage_options: StorageOptions | None = None,
    # 定义存储选项，用于指定文件系统接口，默认为 None
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    # 定义数据类型后端，默认为无默认值
# 返回类型为 DataFrame 或 TextFileReader
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[HashableT, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
):
    # locals() should never be modified
    kwds = locals().copy()
    # 删除不需要传递给 _refine_defaults_read 的参数
    del kwds["filepath_or_buffer"]
    del kwds["sep"]

    # 通过 _refine_defaults_read 函数获取读取函数的默认参数
    kwds_defaults = _refine_defaults_read(
        dialect,
        delimiter,
        engine,
        sep,
        on_bad_lines,
        names,
        defaults={"delimiter": ","},
        dtype_backend=dtype_backend,
    )
    # 更新 kwds 字典，以确保包含了默认参数
    kwds.update(kwds_defaults)

    # 调用 _read 函数，传递 filepath_or_buffer 和已更新的 kwds 字典
    return _read(filepath_or_buffer, kwds)


@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: bool = ...,
    chunksize: int,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame: ...


@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: bool = ...,
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame | TextFileReader: ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_table",
        summary="Read general delimited file into DataFrame.",
        see_also_func_name="read_csv",
        see_also_func_summary=(
            "Read a comma-separated values (csv) file into DataFrame."
        ),
        _default_sep=r"'\\t' (tab-stop)",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[HashableT, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
):
    # 调用 Appender 装饰器，向 read_table 函数添加文档字符串
    pass
    na_filter: bool = True,
    # 是否过滤缺失值，默认为True
    skip_blank_lines: bool = True,
    # 是否跳过空白行，默认为True
    
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    # 解析日期，默认为None；可以是布尔值、可散列序列或None
    date_format: str | dict[Hashable, str] | None = None,
    # 日期格式字符串或格式字典，默认为None
    dayfirst: bool = False,
    # 是否优先考虑日期为日-月形式，默认为False
    cache_dates: bool = True,
    # 是否缓存解析后的日期，默认为True
    
    # Iteration
    iterator: bool = False,
    # 是否返回可迭代对象，默认为False
    chunksize: int | None = None,
    # 每次迭代的行数，默认为None
    
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    # 压缩类型，默认为"infer"
    thousands: str | None = None,
    # 千位分隔符，默认为None
    decimal: str = ".",
    # 小数点字符，默认为"."
    lineterminator: str | None = None,
    # 行终止符，默认为None
    quotechar: str = '"',
    # 引用字符，默认为双引号('"')
    quoting: int = csv.QUOTE_MINIMAL,
    # 引用风格，使用csv.QUOTE_MINIMAL指示最小化引用，默认为csv.QUOTE_MINIMAL
    doublequote: bool = True,
    # 是否双引号转义，默认为True
    escapechar: str | None = None,
    # 转义字符，默认为None
    comment: str | None = None,
    # 注释字符，默认为None
    encoding: str | None = None,
    # 文件编码，默认为None
    encoding_errors: str | None = "strict",
    # 编码错误处理方式，默认为"strict"
    dialect: str | csv.Dialect | None = None,
    # CSV方言，默认为None
    
    # Error Handling
    on_bad_lines: str = "error",
    # 处理错误行的方式，默认为"error"
    
    # Internal
    low_memory: bool = _c_parser_defaults["low_memory"],
    # 是否启用低内存模式，默认从_c_parser_defaults获取
    memory_map: bool = False,
    # 是否启用内存映射，默认为False
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    # 浮点数精度，可以是"high"、"legacy"或"round_trip"，默认为None
    storage_options: StorageOptions | None = None,
    # 存储选项，默认为None
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    # 数据类型后端，默认为lib.no_default
    # locals() should never be modified
    kwds = locals().copy()
    # 删除不需要的关键字参数以避免传递给 _refine_defaults_read 函数
    del kwds["filepath_or_buffer"]
    del kwds["sep"]

    # 使用 _refine_defaults_read 函数处理部分关键字参数的默认值
    kwds_defaults = _refine_defaults_read(
        dialect,
        delimiter,
        engine,
        sep,
        on_bad_lines,
        names,
        # 设置默认的分隔符为制表符
        defaults={"delimiter": "\t"},
        dtype_backend=dtype_backend,
    )
    # 更新 kwds 字典，将 _refine_defaults_read 处理后的默认值合并进去
    kwds.update(kwds_defaults)

    # 调用 _read 函数，传入文件路径或缓冲区和更新后的关键字参数字典
    return _read(filepath_or_buffer, kwds)


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    iterator: bool = ...,
    chunksize: int,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame: ...


# 函数签名：读取固定宽度格式的文本文件，返回 DataFrame 或 TextFileReader 对象
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = "infer",
    widths: Sequence[int] | None = None,
    infer_nrows: int = 100,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame | TextFileReader:
    r"""
    Read a table of fixed-width formatted lines into DataFrame.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Additional help can be found in the `online docs for IO Tools
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a text ``read()`` function.The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.csv``.
    # 检查输入参数。
    if colspecs is None and widths is None:
        # 如果既没有指定 colspecs 也没有指定 widths，则抛出数值错误。
        raise ValueError("Must specify either colspecs or widths")
    if colspecs not in (None, "infer") and widths is not None:
        # 如果 colspecs 不是 None 或 "infer"，并且 widths 不是 None，则抛出数值错误。
        raise ValueError("You must specify only one of 'widths' and 'colspecs'")

    # 如果指定了 widths，则从 widths 计算 colspecs。
    if widths is not None:
        # 初始化 colspecs 列表和列索引 col。
        colspecs, col = [], 0
        # 遍历 widths 中的每个字段宽度 w。
        for w in widths:
            # 添加当前字段的起止位置到 colspecs 中。
            colspecs.append((col, col + w))
            # 更新列索引 col。
            col += w

    # 用于类型检查工具 mypy
    assert colspecs is not None

    # GH#40830
    # 确保 `colspecs` 的长度与 `names` 的长度匹配。
    names = kwds.get("names")
    # 检查是否提供了列名(names)，且不等于特殊值 lib.no_default
    if names is not None and names is not lib.no_default:
        # 如果列名的数量与列规范（colspecs）的数量不一致，并且列规范不是 "infer"，则需要进行下面的检查
        if len(names) != len(colspecs) and colspecs != "infer":
            # 检查 index_col 的长度，因为它可能包含未命名的索引，这种情况下它的名称是不需要的
            len_index = 0
            # 如果关键字参数中有 index_col
            if kwds.get("index_col") is not None:
                # 获取 index_col 的值
                index_col: Any = kwds.get("index_col")
                # 如果 index_col 不为 False
                if index_col is not False:
                    # 检查 index_col 是否是类列表对象
                    if not is_list_like(index_col):
                        len_index = 1
                    else:
                        # 对于类型检查工具 mypy：在 if 分支中已经处理了
                        assert index_col is not lib.no_default
                        len_index = len(index_col)
            # 如果未使用 usecols 参数，并且 names 的长度加上 index_col 的长度不等于 colspecs 的长度
            if kwds.get("usecols") is None and len(names) + len_index != len(colspecs):
                # 抛出数值错误异常，说明 colspecs 的长度必须与 names 的长度相匹配
                raise ValueError("Length of colspecs must match length of names")

    # 检查 dtype_backend 是否已设置，默认使用 lib.no_default
    check_dtype_backend(kwds.setdefault("dtype_backend", lib.no_default))
    # 调用 _read 函数，读取文件或缓冲区内容，并传递相关关键字参数
    return _read(
        filepath_or_buffer,
        kwds
        | {
            "colspecs": colspecs,
            "infer_nrows": infer_nrows,
            "engine": "python-fwf",
            "iterator": iterator,
            "chunksize": chunksize,
        },
    )
class TextFileReader(abc.Iterator):
    """
    文本文件阅读器类，实现了抽象基类 Iterator
    """

    def __init__(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list,
        engine: CSVEngine | None = None,
        **kwds,
    ) -> None:
        # 检查是否指定了引擎，若指定则设置为 True
        if engine is not None:
            engine_specified = True
        else:
            # 否则使用默认的 'python' 引擎，并标记为未指定
            engine = "python"
            engine_specified = False
        self.engine = engine
        self._engine_specified = kwds.get("engine_specified", engine_specified)

        # 验证是否有 skipfooter 选项，并进行处理
        _validate_skipfooter(kwds)

        # 提取并处理 dialect 选项
        dialect = _extract_dialect(kwds)
        if dialect is not None:
            if engine == "pyarrow":
                # 如果使用 'pyarrow' 引擎且指定了 dialect，抛出异常
                raise ValueError(
                    "The 'dialect' option is not supported with the 'pyarrow' engine"
                )
            # 合并 dialect 与其他选项
            kwds = _merge_with_dialect_properties(dialect, kwds)

        # 若 header 选项为 'infer'，则根据是否指定 names 设置其值
        if kwds.get("header", "infer") == "infer":
            kwds["header"] = 0 if kwds.get("names") is None else None

        # 保存原始选项到实例属性
        self.orig_options = kwds

        # 杂项设置，初始化当前行数
        self._currow = 0

        # 获取带有默认值的选项
        options = self._get_options_with_defaults(engine)
        options["storage_options"] = kwds.get("storage_options", None)

        # 设置 chunksize 和 nrows 属性
        self.chunksize = options.pop("chunksize", None)
        self.nrows = options.pop("nrows", None)

        # 检查文件或缓冲区并设置 options 和 engine
        self._check_file_or_buffer(f, engine)
        self.options, self.engine = self._clean_options(options, engine)

        # 如果存在 'has_index_names' 选项，则设置为 True
        if "has_index_names" in kwds:
            self.options["has_index_names"] = kwds["has_index_names"]

        # 初始化 handles 和 _engine 属性
        self.handles: IOHandles | None = None
        self._engine = self._make_engine(f, self.engine)

    def close(self) -> None:
        # 关闭 handles 和 _engine 对象
        if self.handles is not None:
            self.handles.close()
        self._engine.close()
    def _get_options_with_defaults(self, engine: CSVEngine) -> dict[str, Any]:
        # 从原始选项中获取关键字参数的字典
        kwds = self.orig_options

        # 初始化选项字典
        options = {}
        default: object | None

        # 遍历解析器默认选项字典
        for argname, default in parser_defaults.items():
            # 从原始选项中获取参数值，如果不存在则使用默认值
            value = kwds.get(argname, default)

            # 如果引擎为'pyarrow'且该参数不支持，抛出 ValueError 异常
            if (
                engine == "pyarrow"
                and argname in _pyarrow_unsupported
                and value != default
                and value != getattr(value, "value", default)
            ):
                raise ValueError(
                    f"The {argname!r} option is not supported with the "
                    f"'pyarrow' engine"
                )
            # 将参数及其值添加到选项字典中
            options[argname] = value

        # 遍历 C 解析器的默认选项字典
        for argname, default in _c_parser_defaults.items():
            if argname in kwds:
                # 如果原始选项中存在该参数，则使用原始选项中的值
                value = kwds[argname]

                # 如果引擎不是'c'且值不等于默认值，进行进一步检查
                if engine != "c" and value != default:
                    # TODO: 重构此逻辑，它相当复杂
                    if "python" in engine and argname not in _python_unsupported:
                        pass
                    elif "pyarrow" in engine and argname not in _pyarrow_unsupported:
                        pass
                    else:
                        # 如果条件不满足，则抛出 ValueError 异常
                        raise ValueError(
                            f"The {argname!r} option is not supported with the "
                            f"{engine!r} engine"
                        )
            else:
                # 如果原始选项中不存在该参数，则使用默认值
                value = default
            # 将参数及其值添加到选项字典中
            options[argname] = value

        # 如果引擎为'python-fwf'，处理固定宽度格式解析器的默认选项
        if engine == "python-fwf":
            for argname, default in _fwf_defaults.items():
                options[argname] = kwds.get(argname, default)

        # 返回最终的选项字典
        return options

    def _check_file_or_buffer(self, f, engine: CSVEngine) -> None:
        # 查看 GitHub issue #16530
        if is_file_like(f) and engine != "c" and not hasattr(f, "__iter__"):
            # 对于非文件迭代对象，在使用'python'引擎时抛出 ValueError 异常
            raise ValueError(
                "The 'python' engine cannot iterate through this file buffer."
            )
        # 如果文件或缓冲区具有编码属性
        if hasattr(f, "encoding"):
            file_encoding = f.encoding
            orig_reader_enc = self.orig_options.get("encoding", None)
            any_none = file_encoding is None or orig_reader_enc is None
            # 如果文件编码与原始读取器编码不一致且均不为 None，则抛出 ValueError 异常
            if file_encoding != orig_reader_enc and not any_none:
                file_path = getattr(f, "name", None)
                raise ValueError(
                    f"The specified reader encoding {orig_reader_enc} is different "
                    f"from the encoding {file_encoding} of file {file_path}."
                )

    def _clean_options(
        self, options: dict[str, Any], engine: CSVEngine
    ) -> dict[str, Any]:
        # 此方法用于清理和验证选项字典，并返回清理后的选项字典
    # 定义一个方法，返回一个 DataFrame 对象
    def __next__(self) -> DataFrame:
        try:
            # 调用 get_chunk 方法获取数据块并返回
            return self.get_chunk()
        except StopIteration:
            # 如果迭代结束，关闭资源并向上抛出 StopIteration 异常
            self.close()
            raise

    # 定义一个方法，根据给定的文件或缓冲对象类型选择合适的解析引擎，并返回一个 ParserBase 对象
    def _make_engine(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list | IO,
        engine: CSVEngine = "c",
    ) -> ParserBase:
        # 解析引擎类型与实际解析器之间的映射关系
        mapping: dict[str, type[ParserBase]] = {
            "c": CParserWrapper,
            "python": PythonParser,
            "pyarrow": ArrowParserWrapper,
            "python-fwf": FixedWidthFieldParser,
        }

        # 如果给定的引擎不在映射字典中，则抛出 ValueError 异常
        if engine not in mapping:
            raise ValueError(
                f"Unknown engine: {engine} (valid options are {mapping.keys()})"
            )

        # 如果 f 不是 list 类型，则根据引擎类型打开文件或缓冲对象
        if not isinstance(f, list):
            # 在这里打开文件
            is_text = True
            mode = "r"
            if engine == "pyarrow":
                # 对于 pyarrow 引擎，需要以二进制模式打开
                is_text = False
                mode = "rb"
            elif (
                engine == "c"
                and self.options.get("encoding", "utf-8") == "utf-8"
                and isinstance(stringify_path(f), str)
            ):
                # 对于 c 引擎且编码为 utf-8，可以解码为文本模式，但加入 TextIOWrapper 会导致性能下降
                is_text = False
                if "b" not in mode:
                    mode += "b"

            # 根据指定参数获取文件句柄
            self.handles = get_handle(
                f,
                mode,
                encoding=self.options.get("encoding", None),
                compression=self.options.get("compression", None),
                memory_map=self.options.get("memory_map", False),
                is_text=is_text,
                errors=self.options.get("encoding_errors", "strict"),
                storage_options=self.options.get("storage_options", None),
            )
            # 断言文件句柄不为 None
            assert self.handles is not None
            # 使用获取的文件句柄作为新的 f 对象
            f = self.handles.handle

        # 如果 engine 不是 python 并且 f 不是 list 类型，则抛出 ValueError 异常
        elif engine != "python":
            msg = f"Invalid file path or buffer object type: {type(f)}"
            raise ValueError(msg)

        try:
            # 根据映射选择的引擎类型，使用给定的 f 和选项参数实例化对应的解析器对象
            return mapping[engine](f, **self.options)
        except Exception:
            # 如果出现异常，关闭文件句柄（如果存在），然后向上抛出异常
            if self.handles is not None:
                self.handles.close()
            raise

    # 定义一个方法，用于在没有实现的情况下抛出 AbstractMethodError 异常
    def _failover_to_python(self) -> None:
        raise AbstractMethodError(self)
    # 读取数据，返回一个 DataFrame 对象
    def read(self, nrows: int | None = None) -> DataFrame:
        # 如果使用 pyarrow 引擎
        if self.engine == "pyarrow":
            try:
                # 使用 pyarrow 引擎读取数据
                df = self._engine.read()  # type: ignore[attr-defined]
            except Exception:
                # 出现异常时关闭资源并抛出异常
                self.close()
                raise
        else:
            # 验证 nrows 是否为整数
            nrows = validate_integer("nrows", nrows)
            try:
                # 使用其他引擎读取数据
                (
                    index,
                    columns,
                    col_dict,
                ) = self._engine.read(  # type: ignore[attr-defined]
                    nrows
                )
            except Exception:
                # 出现异常时关闭资源并抛出异常
                self.close()
                raise

            # 如果 index 为 None，则根据 col_dict 确定新行数
            if index is None:
                if col_dict:
                    # 如果 col_dict 非空，使用其中任一列的长度作为新行数
                    new_rows = len(next(iter(col_dict.values())))
                    index = RangeIndex(self._currow, self._currow + new_rows)
                else:
                    # 否则新行数为 0
                    new_rows = 0
            else:
                # 否则新行数为 index 的长度
                new_rows = len(index)

            # 获取原始选项中的 dtype 参数
            if hasattr(self, "orig_options"):
                dtype_arg = self.orig_options.get("dtype", None)
            else:
                dtype_arg = None

            # 根据 dtype_arg 的类型设置 dtype
            if isinstance(dtype_arg, dict):
                dtype = defaultdict(lambda: None)  # type: ignore[var-annotated]
                dtype.update(dtype_arg)
            elif dtype_arg is not None and pandas_dtype(dtype_arg) in (
                np.str_,
                np.object_,
            ):
                dtype = defaultdict(lambda: dtype_arg)
            else:
                dtype = None

            # 如果 dtype 不为 None，则根据 dtype 转换列数据
            if dtype is not None:
                new_col_dict = {}
                for k, v in col_dict.items():
                    d = (
                        dtype[k]
                        if pandas_dtype(dtype[k]) in (np.str_, np.object_)
                        else None
                    )
                    new_col_dict[k] = Series(v, index=index, dtype=d, copy=False)
            else:
                # 否则直接使用原始的 col_dict
                new_col_dict = col_dict

            # 创建 DataFrame 对象
            df = DataFrame(
                new_col_dict,
                columns=columns,
                index=index,
                copy=False,
            )

            # 更新当前行数指针
            self._currow += new_rows
        # 返回最终的 DataFrame 对象
        return df

    # 获取指定大小的数据块作为 DataFrame 对象返回
    def get_chunk(self, size: int | None = None) -> DataFrame:
        # 如果 size 为 None，则使用默认的 chunksize
        if size is None:
            size = self.chunksize
        # 如果 nrows 不为 None，并且当前行数大于等于 nrows，则抛出 StopIteration 异常
        if self.nrows is not None:
            if self._currow >= self.nrows:
                raise StopIteration
            # 计算实际读取的数据块大小
            if size is None:
                size = self.nrows - self._currow
            else:
                size = min(size, self.nrows - self._currow)
        # 调用 read 方法读取指定大小的数据块并返回
        return self.read(nrows=size)

    # 上下文管理器方法，返回自身对象
    def __enter__(self) -> Self:
        return self
    # 定义 __exit__ 方法，用于处理资源释放
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # 调用 close 方法关闭资源
        self.close()
# 将输入的文件数据转换为 DataFrame，并进行类型推断和可选的转换（如字符串到日期时间）。
# 还支持对大文件进行分块懒惰迭代。

def TextParser(*args, **kwds) -> TextFileReader:
    """
    Converts lists of lists/tuples into DataFrames with proper type inference
    and optional (e.g. string to datetime) conversion. Also enables iterating
    lazily over chunks of large files

    Parameters
    ----------
    data : file-like object or list
        输入的文件数据或列表对象
    delimiter : separator character to use
        用于分隔的字符
    dialect : str or csv.Dialect instance, optional
        如果分隔符超过1个字符，则此参数被忽略
    names : sequence, default
        列名序列，默认为空
    header : int, default 0
        用于解析列标签的行索引。默认为第一行，之前的行将被丢弃
    index_col : int or list, optional
        用作（可能是分层的）索引的列或列列表
    has_index_names: bool, default False
        如果 index_col 定义的列具有索引名称且不在标题中，则为 True
    na_values : scalar, str, list-like, or dict, optional
        用作NA/NaN的额外字符串
    keep_default_na : bool, default True
        是否保留默认的NA值
    thousands : str, optional
        千位分隔符
    comment : str, optional
        注释行剩余部分的字符
    parse_dates : bool, default False
        是否解析日期
    date_format : str or dict of column -> format, default ``None``
        日期格式字符串或列名到格式字符串的映射

        .. versionadded:: 2.0.0
    skiprows : list of integers
        要跳过的行号列表
    skipfooter : int
        要跳过的文件底部行数
    converters : dict, optional
        用于转换特定列值的函数字典。键可以是整数或列标签，值是接受一个参数（单元格内容）并返回转换后内容的函数
    encoding : str, optional
        读取/写入UTF时使用的编码（例如 'utf-8'）
    float_precision : str, optional
        指定C引擎应该使用的浮点值转换器。选项为 `None` 或 `high` 表示普通转换器，`legacy` 表示原始较低精度的pandas转换器，`round_trip` 表示往返转换器。
    """
    # 将引擎设置为Python引擎，用于处理数据
    kwds["engine"] = "python"
    # 返回一个文本文件读取器，根据传入的参数进行初始化
    return TextFileReader(*args, **kwds)


def _clean_na_values(na_values, keep_default_na: bool = True, floatify: bool = True):
    # na_fvalues 可以是一个集合或字典，用于存储NaN值
    na_fvalues: set | dict
    # 如果 na_values 为None，则根据 keep_default_na 的设置确定默认的NA值集合
    if na_values is None:
        if keep_default_na:
            na_values = STR_NA_VALUES  # 使用默认的NA值集合
        else:
            na_values = set()  # 空集合，表示不处理任何特定的NA值
        na_fvalues = set()  # 初始化空的浮点NA值集合
    elif isinstance(na_values, dict):
        old_na_values = na_values.copy()
        na_values = {}  # 防止别名问题，创建一个空字典作为新的na_values

        # 将na_values字典中的值转换为类数组以便进一步使用。
        # 如果keep_default_na=True，这也是我们添加默认NaN值的地方。
        for k, v in old_na_values.items():
            if not is_list_like(v):  # 检查v是否类数组
                v = [v]  # 如果不是类数组，则转换为列表

            if keep_default_na:
                v = set(v) | STR_NA_VALUES  # 如果保留默认NaN值，将默认NaN值添加到v中

            na_values[k] = v  # 将处理后的v重新赋值给na_values中的k键

        # 将na_values中的每个键值对的值转换为浮点数形式，存入na_fvalues字典
        na_fvalues = {k: _floatify_na_values(v) for k, v in na_values.items()}
    else:
        if not is_list_like(na_values):
            na_values = [na_values]  # 如果na_values不是类数组，则转换为列表
        na_values = _stringify_na_values(na_values, floatify)  # 将na_values中的值转换为字符串形式
        if keep_default_na:
            na_values = na_values | STR_NA_VALUES  # 如果保留默认NaN值，将默认NaN值添加到na_values中

        # 将na_values中的值转换为浮点数形式，存入na_fvalues
        na_fvalues = _floatify_na_values(na_values)

    return na_values, na_fvalues
def _floatify_na_values(na_values):
    # 创建浮点数版本的缺失值集合
    result = set()
    for v in na_values:
        try:
            v = float(v)
            if not np.isnan(v):  # 如果不是 NaN，则添加到结果集合中
                result.add(v)
        except (TypeError, ValueError, OverflowError):
            pass  # 忽略类型错误、数值错误和溢出错误
    return result  # 返回浮点数版本的缺失值集合


def _stringify_na_values(na_values, floatify: bool) -> set[str | float]:
    """将这些值返回为字符串和数值类型"""
    result: list[str | float] = []
    for x in na_values:
        result.append(str(x))  # 将值转换为字符串并添加到结果列表中
        result.append(x)  # 将原始值添加到结果列表中
        try:
            v = float(x)  # 尝试将值转换为浮点数

            # 在这里我们看到像 999 这样的情况
            if v == int(v):  # 如果浮点数等于其整数部分
                v = int(v)
                result.append(f"{v}.0")  # 添加带有小数点的整数字符串到结果列表中
                result.append(str(v))  # 添加整数字符串到结果列表中

            if floatify:
                result.append(v)  # 如果需要浮点化，则添加浮点数到结果列表中
        except (TypeError, ValueError, OverflowError):
            pass  # 忽略类型错误、数值错误和溢出错误
        if floatify:
            try:
                result.append(int(x))  # 如果需要浮点化，尝试将值转换为整数并添加到结果列表中
            except (TypeError, ValueError, OverflowError):
                pass  # 忽略类型错误、数值错误和溢出错误
    return set(result)  # 返回字符串和数值类型的结果集合


def _refine_defaults_read(
    dialect: str | csv.Dialect | None,
    delimiter: str | None | lib.NoDefault,
    engine: CSVEngine | None,
    sep: str | None | lib.NoDefault,
    on_bad_lines: str | Callable,
    names: Sequence[Hashable] | None | lib.NoDefault,
    defaults: dict[str, Any],
    dtype_backend: DtypeBackend | lib.NoDefault,
):
    """验证和优化 read_csv、read_table 的输入参数的默认值。

    Parameters
    ----------
    dialect : str or csv.Dialect or None
        如果提供，此参数将覆盖以下参数的值（默认值或非默认值）：
        `delimiter`、`doublequote`、`escapechar`、`skipinitialspace`、
        `quotechar` 和 `quoting`。如果需要覆盖值，将会发出 ParserWarning。
        更多细节请参考 csv.Dialect 文档。
    delimiter : str or object or lib.NoDefault or None
        sep 的别名。
    engine : {'c', 'python'} or None
        要使用的解析引擎。C 引擎更快，而 Python 引擎目前更加功能完备。
    sep : str or object or lib.NoDefault or None
        用户提供的分隔符（str）或一个特殊值，如 pandas._libs.lib.no_default。
    on_bad_lines : str or callable
        处理错误行的选项或一个特殊值（None）。
    names : array-like or None or lib.NoDefault
        要使用的列名列表。如果文件包含标题行，则应显式传递 `header=0` 来覆盖列名。
        此列表不允许重复项。
    defaults : dict
        输入参数的默认值。
    dtype_backend : DtypeBackend or lib.NoDefault
        dtype 后端或无默认值。

    Returns
    -------
    kwds : dict
        带有正确值的输入参数。
    """
    # 修复 sep、delimiter 类型为 Union[str, Any]
    delim_default = defaults["delimiter"]
    kwds: dict[str, Any] = {}
    # gh-23761
    #
    # 当传递了一个方言时，它会覆盖任何重叠的参数
    # 如果传入了 dialect 参数，则设置 kwds 字典中的 sep_override 标志，
    # 表示需要检查是否同时传入了默认的 delimiter 和 sep 参数，
    # 以避免警告用户。
    if dialect is not None:
        kwds["sep_override"] = delimiter is None and (
            sep is lib.no_default or sep == delim_default
        )

    # 如果同时指定了 delimiter 和 sep 参数，则抛出 ValueError 异常，
    # 因为只能指定其中一个。
    if delimiter and (sep is not lib.no_default):
        raise ValueError("Specified a sep and a delimiter; you can only specify one.")

    # 如果 names 参数未指定，默认将 kwds["names"] 设置为 None。
    kwds["names"] = None if names is lib.no_default else names

    # 如果 delimiter 为 None，则将 sep 的值赋给 delimiter。
    if delimiter is None:
        delimiter = sep

    # 如果 delimiter 为 "\n"，则抛出 ValueError 异常，
    # 因为 "\n" 在 Python 引擎中不被接受作为分隔符。
    if delimiter == "\n":
        raise ValueError(
            r"Specified \n as separator or delimiter. This forces the python engine "
            "which does not accept a line terminator. Hence it is not allowed to use "
            "the line terminator as separator."
        )

    # 如果 delimiter 为 lib.no_default，则将默认的分隔符值 delim_default 赋给 kwds["delimiter"]。
    if delimiter is lib.no_default:
        kwds["delimiter"] = delim_default
    else:
        # 否则将 delimiter 的值赋给 kwds["delimiter"]。
        kwds["delimiter"] = delimiter

    # 如果指定了 engine 参数，则将 kwds["engine_specified"] 设置为 True；
    # 否则将 engine 设置为 "c"，并将 kwds["engine_specified"] 设置为 False。
    if engine is not None:
        kwds["engine_specified"] = True
    else:
        kwds["engine"] = "c"
        kwds["engine_specified"] = False

    # 根据 on_bad_lines 参数的不同取值设置 kwds["on_bad_lines"] 的处理方法，
    # 可能的取值包括 "error"、"warn"、"skip" 和 callable 函数。
    if on_bad_lines == "error":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.ERROR
    elif on_bad_lines == "warn":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.WARN
    elif on_bad_lines == "skip":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.SKIP
    elif callable(on_bad_lines):
        # 如果 on_bad_lines 是一个 callable 函数，则需要检查 engine 参数是否是 "python" 或 "pyarrow"。
        # 否则抛出 ValueError 异常。
        if engine not in ["python", "pyarrow"]:
            raise ValueError(
                "on_bad_line can only be a callable function "
                "if engine='python' or 'pyarrow'"
            )
        kwds["on_bad_lines"] = on_bad_lines
    else:
        # 如果 on_bad_lines 参数值无效，则抛出 ValueError 异常。
        raise ValueError(f"Argument {on_bad_lines} is invalid for on_bad_lines")

    # 检查 dtype_backend 参数的有效性。
    check_dtype_backend(dtype_backend)

    # 将 dtype_backend 参数赋给 kwds["dtype_backend"]。
    kwds["dtype_backend"] = dtype_backend

    # 返回更新后的 kwds 字典。
    return kwds
# 从给定的关键字参数中提取具体的 CSV 方言实例
def _extract_dialect(kwds: dict[str, Any]) -> csv.Dialect | None:
    # 如果关键字参数中没有提供 'dialect'，则返回 None
    if kwds.get("dialect") is None:
        return None

    dialect = kwds["dialect"]
    # 如果提供的方言名称在已知的方言列表中，则获取对应的方言对象
    if dialect in csv.list_dialects():
        dialect = csv.get_dialect(dialect)

    # 验证所提取的方言对象的有效性
    _validate_dialect(dialect)

    return dialect


# 必需的 CSV 方言属性列表
MANDATORY_DIALECT_ATTRS = (
    "delimiter",
    "doublequote",
    "escapechar",
    "skipinitialspace",
    "quotechar",
    "quoting",
)


# 验证 CSV 方言实例的有效性
def _validate_dialect(dialect: csv.Dialect) -> None:
    # 检查方言对象是否具有所有必需的属性
    for param in MANDATORY_DIALECT_ATTRS:
        if not hasattr(dialect, param):
            raise ValueError(f"Invalid dialect {dialect} provided")


# 将默认的关键字参数与 CSV 方言的参数进行合并
def _merge_with_dialect_properties(
    dialect: csv.Dialect,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge default kwargs in TextFileReader with dialect parameters.

    Parameters
    ----------
    dialect : csv.Dialect
        Concrete csv dialect. See csv.Dialect documentation for more details.
    defaults : dict
        Keyword arguments passed to TextFileReader.

    Returns
    -------
    kwds : dict
        Updated keyword arguments, merged with dialect parameters.
    """
    kwds = defaults.copy()

    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)

        # 获取参数的解析器默认值
        parser_default = parser_defaults[param]
        # 获取已提供的参数值或者使用解析器默认值
        provided = kwds.get(param, parser_default)

        # 存储冲突消息的列表
        conflict_msgs = []

        # 如果提供的参数值与方言值不同，生成冲突消息
        if provided not in (parser_default, dialect_val):
            msg = (
                f"Conflicting values for '{param}': '{provided}' was "
                f"provided, but the dialect specifies '{dialect_val}'. "
                "Using the dialect-specified value."
            )

            # 对于分隔符参数和 'sep_override' 冲突的特殊情况，不生成警告
            if not (param == "delimiter" and kwds.pop("sep_override", False)):
                conflict_msgs.append(msg)

        # 如果存在冲突消息，则生成警告
        if conflict_msgs:
            warnings.warn(
                "\n\n".join(conflict_msgs), ParserWarning, stacklevel=find_stack_level()
            )
        
        # 更新关键字参数中的方言值
        kwds[param] = dialect_val

    return kwds


# 检查 skipfooter 参数在 TextFileReader 的其他关键字参数中是否兼容
def _validate_skipfooter(kwds: dict[str, Any]) -> None:
    """
    Check whether skipfooter is compatible with other kwargs in TextFileReader.

    Parameters
    ----------
    kwds : dict
        Keyword arguments passed to TextFileReader.

    Raises
    ------
    # 如果参数字典中存在 "skipfooter"
    if kwds.get("skipfooter"):
        # 如果同时存在 "iterator" 或者 "chunksize" 参数，则抛出数值错误异常
        if kwds.get("iterator") or kwds.get("chunksize"):
            raise ValueError("'skipfooter' not supported for iteration")
        # 如果存在 "nrows" 参数，则抛出数值错误异常
        if kwds.get("nrows"):
            raise ValueError("'skipfooter' not supported with 'nrows'")
```