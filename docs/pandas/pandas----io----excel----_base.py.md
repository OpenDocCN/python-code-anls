# `D:\src\scipysrc\pandas\pandas\io\excel\_base.py`

```
from __future__ import annotations
# 导入用于支持类型注解的特殊模块（Python 3.7 及更早版本）

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
# 导入抽象基类中定义的几种集合类型，用于类型检查和提示

import datetime
# 导入处理日期和时间的模块

from functools import partial
# 导入用于创建偏函数的模块

import os
# 导入用于提供与操作系统交互的功能的模块

from textwrap import fill
# 导入用于文本包装和填充的函数

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)
# 导入用于类型提示的模块和函数

import warnings
# 导入用于处理警告的模块

import zipfile
# 导入用于操作 ZIP 文件的模块

from pandas._config import config
# 导入 Pandas 的配置模块

from pandas._libs import lib
# 导入 Pandas 底层 C 代码库

from pandas._libs.parsers import STR_NA_VALUES
# 导入用于处理解析器的空值的常量

from pandas.compat._optional import (
    get_version,
    import_optional_dependency,
)
# 导入用于支持可选依赖项的模块和函数

from pandas.errors import EmptyDataError
# 导入 Pandas 的空数据错误类

from pandas.util._decorators import (
    Appender,
    doc,
)
# 导入装饰器函数，用于添加文档和其他功能

from pandas.util._exceptions import find_stack_level
# 导入用于找到堆栈级别的异常处理函数

from pandas.util._validators import check_dtype_backend
# 导入用于检查数据类型后端的验证函数

from pandas.core.dtypes.common import (
    is_bool,
    is_file_like,
    is_float,
    is_integer,
    is_list_like,
)
# 导入用于检查常见数据类型的函数

from pandas.core.frame import DataFrame
# 导入 Pandas 的 DataFrame 类

from pandas.core.shared_docs import _shared_docs
# 导入 Pandas 的共享文档

from pandas.util.version import Version
# 导入 Pandas 的版本信息

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
# 导入用于处理 IO 和路径的常用函数

from pandas.io.excel._util import (
    fill_mi_header,
    get_default_engine,
    get_writer,
    maybe_convert_usecols,
    pop_header_name,
)
# 导入用于 Excel 处理的一些实用函数

from pandas.io.parsers import TextParser
# 导入用于文本解析的模块

from pandas.io.parsers.readers import validate_integer
# 导入用于验证整数的函数

if TYPE_CHECKING:
    from types import TracebackType
    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        ExcelWriterIfSheetExists,
        FilePath,
        HashableT,
        IntStrT,
        ReadBuffer,
        Self,
        SequenceNotStr,
        StorageOptions,
        WriteExcelBuffer,
    )
# 如果是类型检查阶段，则导入额外的类型和模块

_read_excel_doc = (
    """
Read an Excel file into a ``pandas`` ``DataFrame``.

Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions
read from a local filesystem or URL. Supports an option to read
a single sheet or a list of sheets.

Parameters
----------
io : str, ExcelFile, xlrd.Book, path object, or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: ``file://localhost/path/to/table.xlsx``.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method,
    such as a file handle (e.g. via builtin ``open`` function)
    or ``StringIO``.

    .. deprecated:: 2.1.0
        Passing byte strings is deprecated. To read from a
        byte string, wrap it in a ``BytesIO`` object.
sheet_name : str, int, list, or None, default 0
    Strings are used for sheet names. Integers are used in zero-indexed
    sheet positions (chart sheets do not count as a sheet position).
    Lists of strings/integers are used to request multiple sheets.
    Specify ``None`` to get all worksheets.
"""
# 定义一个包含读取 Excel 文件函数文档的字符串变量

# 以下是注释部分，根据每行的功能和作用进行解释，但是根据格式要求不能超出代码块范围，因此在这里无法给出注释。
    Available cases:

    * Defaults to ``0``: 1st sheet as a `DataFrame`
    * ``1``: 2nd sheet as a `DataFrame`
    * ``"Sheet1"``: Load sheet with name "Sheet1"
    * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
      as a dict of `DataFrame`
    * ``None``: All worksheets.
header : int, list of int, default 0
    # 指定用于解析DataFrame列标签的行索引（从0开始）。如果传入整数列表，则这些行位置将组合成MultiIndex。如果没有标题行，请使用None。
names : array-like, default None
    # 要使用的列名列表。如果文件不包含标题行，则应显式传递header=None。
index_col : int, str, list of int, default None
    # 用作DataFrame行标签的列（从0开始）。如果没有这样的列，请传递None。如果传递了列表，则这些列将组合成MultiIndex。如果使用usecols选择了数据子集，则index_col将基于该子集。

    # 缺失的值将被前向填充，以允许与``to_excel``和``merged_cells=True``一起往返处理。为了避免前向填充缺失值，请在读取数据后使用``set_index``而不是``index_col``。
usecols : str, list-like, or callable, default None
    # * 如果为None，则解析所有列。
    # * 如果为str，则指示Excel列字母和列范围的逗号分隔列表（例如 "A:E" 或 "A,C,E:F"）。范围包含两端。
    # * 如果为int列表，则指示要解析的列编号（从0开始）。
    # * 如果为字符串列表，则指示要解析的列名。
    # * 如果为callable，则针对每个列名评估它，并且如果callable返回True，则解析该列。

    # 根据上述行为返回列的子集。
dtype : Type name or dict of column -> type, default None
    # 数据或列的数据类型。例如 {'a': np.float64, 'b': np.int32}。
    # 使用``object``来保留数据存储在Excel中的形式，不解释dtype，这将必然导致``object`` dtype。
    # 如果指定了转换器，则将应用它们而不是dtype转换。
    # 如果使用``None``，它将根据数据推断每列的dtype。
engine : {'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}, default None
    # 如果io不是缓冲区或路径，则必须设置此项以识别io。
    # 引擎兼容性：

    # - ``openpyxl`` 支持较新的Excel文件格式。
    # - ``calamine`` 支持Excel (.xls, .xlsx, .xlsm, .xlsb) 和OpenDocument (.ods)文件格式。
    # - ``odf`` 支持OpenDocument文件格式 (.odf, .ods, .odt)。
    # - ``pyxlsb`` 支持二进制Excel文件。
    # - ``xlrd`` 支持旧式Excel文件 (.xls)。

    # 当``engine=None``时，将使用以下逻辑确定引擎：

    # - 如果``path_or_buffer``是OpenDocument格式 (.odf, .ods, .odt)，则使用`odf <https://pypi.org/project/odfpy/>`_。
    # - 否则，如果``path_or_buffer``是xls格式，则使用``xlrd``。
    # - 否则，如果``path_or_buffer``是xlsb格式，则使用``pyxlsb``。
    # - 否则将使用``openpyxl``。
converters : dict, default None
    # 定义一个字典，用于存储将某些列中的值转换的函数集合。
    # 字典的键可以是整数或列标签，对应的值是接受一个输入参数（Excel 单元格内容）并返回转换后内容的函数。
# true_values : list, default None
#    用作真值的值列表，默认为None。

# false_values : list, default None
#    用作假值的值列表，默认为None。

# skiprows : list-like, int, or callable, optional
#    要跳过的行号（从0开始索引），或要跳过的行数（整数），或者可选的可调用对象。
#    如果是可调用的函数，将针对行索引进行评估，返回True表示应该跳过该行，False表示应该保留。
#    一个有效的可调用参数示例可以是 ``lambda x: x in [0, 2]``。

# nrows : int, default None
#    要解析的行数，默认为None。

# na_values : scalar, str, list-like, or dict, default None
#    作为NA/NaN识别的额外字符串。如果传递了字典，则是每列的特定NA值。
#    默认情况下，以下值被解释为NaN：'"""
#    + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent="    ")
#    + """'。

# keep_default_na : bool, default True
#    是否包含默认的NaN值来解析数据。
#    根据是否传递了 ``na_values`` 参数，行为如下：
#    * 如果 ``keep_default_na`` 为True，并且指定了 ``na_values``，则 ``na_values`` 会附加到用于解析的默认NaN值中。
#    * 如果 ``keep_default_na`` 为True，并且未指定 ``na_values``，则仅使用默认的NaN值进行解析。
#    * 如果 ``keep_default_na`` 为False，并且指定了 ``na_values``，则仅使用 ``na_values`` 指定的NaN值进行解析。
#    * 如果 ``keep_default_na`` 为False，并且未指定 ``na_values``，则不会将任何字符串解析为NaN。
#    
#    注意，如果传递了 `na_filter` 参数为False，则会忽略 ``keep_default_na`` 和 ``na_values`` 参数。

# na_filter : bool, default True
#    是否检测缺失值标记（空字符串和na_values的值）。在没有任何NA的数据中，传递 ``na_filter=False`` 可以提高读取大文件的性能。

# verbose : bool, default False
#    指示放置在非数字列中的NA值数量。

# parse_dates : bool, list-like, or dict, default False
#    解析日期的行为如下：
#    * ``bool``. 如果为True -> 尝试解析索引。
#    * 列表形式的int或列名。例如，如果 [1, 2, 3] -> 尝试分别将列1、2、3解析为单独的日期列。
#    * 列表形式的列表。例如，如果 [[1, 3]] -> 结合列1和3并解析为单个日期列。
#    * 字典形式，例如 {'foo' : [1, 3]} -> 解析列1和3作为日期，并命名结果为'foo'。
#    
#    如果列或索引包含无法解析的日期，则整个列或索引将作为对象数据类型返回，不进行更改。
#    如果不想将某些单元格解析为日期，请在Excel中将它们的类型更改为“文本”。
#    对于非标准的日期时间解析，请在 ``pd.read_excel`` 之后使用 ``pd.to_datetime``。
#    
#    注意：ISO8601格式的日期存在快速路径。
# date_format : str or dict of column -> format, default ``None``
#    If used in conjunction with ``parse_dates``, this parameter specifies the date format(s)
#    to parse dates according to. If not specified, dates are not parsed during reading.
#    For more complex date parsing, read the data as 'object' type and then use :func:`to_datetime` as needed.
#
#    .. versionadded:: 2.0.0

# thousands : str, default None
#    Specifies the thousands separator character used when parsing string columns to numeric.
#    This parameter is relevant only for columns stored as text in Excel files.
#    Numeric columns are automatically parsed regardless of their display format.

# decimal : str, default '.'
#    Specifies the character recognized as the decimal point when parsing string columns to numeric.
#    This parameter is necessary only for columns stored as text in Excel files.
#    Numeric columns are automatically parsed regardless of their display format.
#    (e.g., use ',' for European data).
#
#    .. versionadded:: 1.4.0

# comment : str, default None
#    Specifies the character or characters that mark the beginning of a comment in the input file.
#    Any content from the comment string to the end of the line is ignored during parsing.

# skipfooter : int, default 0
#    Specifies the number of rows at the end of the file to skip during parsing.
#    Rows are indexed starting from 0.

# {storage_options}
#    Placeholder for additional storage options that can be passed to the underlying storage system.

# dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
#    Specifies the backend data type to be applied to the resulting DataFrame.
#    This parameter is experimental and controls how null values are handled:
#    - 'numpy_nullable': returns a DataFrame with nullable data types (default).
#    - 'pyarrow': returns a DataFrame with pyarrow-backed nullable data types.
#
#    .. versionadded:: 2.0

# engine_kwargs : dict, optional
#    Additional keyword arguments passed to the Excel engine for customization during reading.

# Returns
# -------
# DataFrame or dict of DataFrames
#    Returns a DataFrame representing the contents of the Excel file. If multiple sheets are read,
#    returns a dictionary of DataFrames.
#    See notes in the sheet_name argument for details on returning a dict of DataFrames.

# See Also
# --------
# DataFrame.to_excel : Write DataFrame to an Excel file.
# DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
# read_csv : Read a comma-separated values (csv) file into DataFrame.
# read_fwf : Read a table of fixed-width formatted lines into DataFrame.

# Notes
# -----
# For detailed information on the methods used by each Excel engine, refer to the pandas
# :ref:`user guide <io.excel_reader>`

# Examples
# --------
# The file can be read using either the file name as a string or an open file object:
#
# >>> pd.read_excel('tmp.xlsx', index_col=0)  # doctest: +SKIP
#        Name  Value
# 0   string1      1
# 1   string2      2
# 2  #Comment      3
#
# >>> pd.read_excel(open('tmp.xlsx', 'rb'), sheet_name='Sheet3')  # doctest: +SKIP
#    Unnamed: 0      Name  Value
# 0           0   string1      1
# 1           1   string2      2
# 2           2  #Comment      3
#
# Index and header can be specified via the `index_col` and `header` arguments:
#
# >>> pd.read_excel('tmp.xlsx', index_col=None, header=None)  # doctest: +SKIP
#      0         1      2
# 使用 pandas 库中的 read_excel 函数读取 Excel 文件内容并返回一个 DataFrame 对象
>>> pd.read_excel('tmp.xlsx', index_col=0,
...               dtype={{'Name': str, 'Value': float}})  # doctest: +SKIP
       Name  Value
0   string1    1.0
1   string2    2.0
2  #Comment    3.0

# 可以通过指定 na_values 参数来将特定的值视为 NaN（缺失值），例如将 'string1' 和 'string2' 视为缺失值
>>> pd.read_excel('tmp.xlsx', index_col=0,
...               na_values=['string1', 'string2'])  # doctest: +SKIP
       Name  Value
0       NaN      1
1       NaN      2
2  #Comment      3

# 通过设置 comment 参数，可以跳过 Excel 文件中以特定字符开头的注释行，这里设置为跳过以 '#' 开头的行
>>> pd.read_excel('tmp.xlsx', index_col=0, comment='#')  # doctest: +SKIP
      Name  Value
0  string1    1.0
1  string2    2.0
2     None    NaN
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,

# na_values=...: 指定用于识别缺失值的值或序列
# keep_default_na: bool = ...: 指定是否保留默认的NA/NaN值集合
# na_filter: bool = ...: 指定是否在数据中识别和过滤缺失值
# verbose: bool = ...: 指定是否输出详细的日志信息
# parse_dates: list | dict | bool = ...: 指定日期解析的规则，可以是列表、字典或布尔值
# date_format: dict[Hashable, str] | str | None = ...: 指定日期格式的规则，可以是字典、字符串或None
# thousands: str | None = ...: 指定千分位分隔符的字符或者None
# decimal: str = ...: 指定小数点分隔符的字符
# comment: str | None = ...: 指定注释标识的字符或者None
# skipfooter: int = ...: 指定在读取时跳过数据尾部的行数
# storage_options: StorageOptions = ...: 指定用于文件存储的选项
# dtype_backend: DtypeBackend | lib.NoDefault = ...: 指定数据类型的后端或者没有默认值
# 定义函数 read_excel，用于读取 Excel 文件内容并返回 DataFrame 或 DataFrame 字典
@doc(storage_options=_shared_docs["storage_options"])
@Appender(_read_excel_doc)
def read_excel(
    io,  # 接收 Excel 文件的输入对象，可以是文件名、文件对象或 ExcelFile 对象
    sheet_name: str | int | list[IntStrT] | None = 0,  # 指定要读取的工作表名或索引列表，默认为第一个工作表
    *,  # 后续参数为关键字参数
    header: int | Sequence[int] | None = 0,  # 指定列标题行的位置，默认为第一行
    names: SequenceNotStr[Hashable] | range | None = None,  # 指定列名列表或范围，默认从数据中推断
    index_col: int | str | Sequence[int] | None = None,  # 指定用作行索引的列位置或名称，默认不使用
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[HashableT], bool]
    | None = None,  # 指定要读取的列，可以是列索引、列名或函数，默认读取所有列
    dtype: DtypeArg | None = None,  # 指定列的数据类型，可以是字典、单个类型或 None，默认推断
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = None,  # 指定使用的 Excel 引擎，默认自动选择
    converters: dict[str, Callable] | dict[int, Callable] | None = None,  # 指定用于转换数据的函数字典，默认无
    true_values: Iterable[Hashable] | None = None,  # 指定为真值的字符串列表，默认无
    false_values: Iterable[Hashable] | None = None,  # 指定为假值的字符串列表，默认无
    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,  # 指定要跳过的行数或条件函数，默认无
    nrows: int | None = None,  # 指定要读取的行数，默认读取所有行
    na_values=None,  # 指定作为缺失值的字符串列表，默认 None
    keep_default_na: bool = True,  # 是否保留默认的缺失值列表，默认 True
    na_filter: bool = True,  # 是否过滤缺失值，默认 True
    verbose: bool = False,  # 是否打印详细信息，默认 False
    parse_dates: list | dict | bool = False,  # 指定是否解析日期，默认不解析
    date_format: dict[Hashable, str] | str | None = None,  # 指定日期解析格式，默认 None
    thousands: str | None = None,  # 指定千位分隔符，默认无
    decimal: str = ".",  # 指定小数点分隔符，默认 "."
    comment: str | None = None,  # 指定注释标记，默认无
    skipfooter: int = 0,  # 指定要跳过的表尾行数，默认为 0
    storage_options: StorageOptions | None = None,  # 存储选项，传递给底层存储库，默认 None
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端，默认为 lib.no_default
    engine_kwargs: dict | None = None,  # 引擎参数字典，默认 None
) -> DataFrame | dict[IntStrT, DataFrame]:  # 返回类型为 DataFrame 或 DataFrame 字典
    check_dtype_backend(dtype_backend)  # 检查数据类型后端是否有效

    should_close = False  # 初始化一个标志，表示是否需要关闭文件

    if engine_kwargs is None:
        engine_kwargs = {}  # 如果引擎参数字典未提供，设置为空字典

    if not isinstance(io, ExcelFile):
        should_close = True  # 如果输入对象不是 ExcelFile，标记需要关闭文件
        io = ExcelFile(
            io,
            storage_options=storage_options,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )  # 创建 ExcelFile 对象，用于处理输入的 Excel 文件

    elif engine and engine != io.engine:
        raise ValueError(
            "Engine should not be specified when passing "
            "an ExcelFile - ExcelFile already has the engine set"
        )  # 如果指定了引擎且与现有的 ExcelFile 引擎不匹配，抛出 ValueError

    try:
        data = io.parse(  # 尝试使用 ExcelFile 对象解析数据
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            parse_dates=parse_dates,
            date_format=date_format,
            thousands=thousands,
            decimal=decimal,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
        )
    finally:
        if should_close:
            io.close()  # 确保关闭已打开的文件句柄

    return data  # 返回解析后的数据

# 定义泛型类 BaseExcelReader，用于表示 Excel 读取器基类
_WorkbookT = TypeVar("_WorkbookT")
class BaseExcelReader(Generic[_WorkbookT]):
    book: _WorkbookT  # book 属性表示 Excel 工作簿对象的类型参数
    # 初始化方法，接收文件路径或缓冲区、存储选项和引擎参数
    def __init__(
        self,
        filepath_or_buffer,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # 如果未提供 engine_kwargs 参数，则设为一个空字典
        if engine_kwargs is None:
            engine_kwargs = {}

        # 初始化 IOHandles 对象，处理文件路径或缓冲区，并设定压缩方法为无
        self.handles = IOHandles(
            handle=filepath_or_buffer, compression={"method": None}
        )
        
        # 如果 filepath_or_buffer 不是 ExcelFile 或者 self._workbook_class 类型的实例
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            # 获取文件句柄，以二进制只读方式打开，支持存储选项，并指定不是文本数据
            self.handles = get_handle(
                filepath_or_buffer, "rb", storage_options=storage_options, is_text=False
            )

        # 如果 handles.handle 是 self._workbook_class 的实例，则将其作为工作簿对象
        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
        # 否则，如果 handles.handle 具有 "read" 方法
        elif hasattr(self.handles.handle, "read"):
            # 将文件指针移动到文件开头
            self.handles.handle.seek(0)
            try:
                # 尝试加载工作簿，使用指定的引擎参数
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                # 加载过程中发生异常，则关闭资源并抛出异常
                self.close()
                raise
        else:
            # 如果不是上述情况，则抛出值错误，要求显式设置引擎参数
            raise ValueError(
                "Must explicitly set engine if not passing in buffer or path for io."
            )

    # 返回工作簿类的属性，子类需要实现该方法
    @property
    def _workbook_class(self) -> type[_WorkbookT]:
        raise NotImplementedError

    # 加载工作簿的抽象方法，子类需要实现该方法
    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> _WorkbookT:
        raise NotImplementedError

    # 关闭资源的方法
    def close(self) -> None:
        # 如果存在 book 属性，并且 book 对象具有 "close" 方法
        if hasattr(self, "book"):
            if hasattr(self.book, "close"):
                # 如果 book 是 pyxlsb 打开的临时文件，则调用其 close 方法
                # 如果是 openpyxl，参考指定的链接进行关闭操作
                self.book.close()
            # 如果 book 对象具有 "release_resources" 方法（例如 xlrd 类型）
            elif hasattr(self.book, "release_resources"):
                # 调用 release_resources 方法释放资源
                self.book.release_resources()
        
        # 关闭 IOHandles 对象，释放其占用的资源
        self.handles.close()

    # 返回工作表名称列表的抽象方法，子类需要实现该方法
    @property
    def sheet_names(self) -> list[str]:
        raise NotImplementedError

    # 根据名称获取工作表的抽象方法，子类需要实现该方法
    def get_sheet_by_name(self, name: str):
        raise NotImplementedError

    # 根据索引获取工作表的抽象方法，子类需要实现该方法
    def get_sheet_by_index(self, index: int):
        raise NotImplementedError

    # 获取工作表数据的抽象方法，子类需要实现该方法
    def get_sheet_data(self, sheet, rows: int | None = None):
        raise NotImplementedError

    # 检查索引是否超出工作表数量的方法，如果超出则抛出值错误
    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        # 获取工作表名称列表的长度
        n_sheets = len(self.sheet_names)
        # 如果索引超出工作表名称列表长度，则抛出值错误
        if index >= n_sheets:
            raise ValueError(
                f"Worksheet index {index} is invalid, {n_sheets} worksheets found"
            )

    # 检查名称是否存在于工作表名称列表中的方法，如果不存在则抛出值错误
    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        # 如果名称不在工作表名称列表中，则抛出值错误
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    # 检查跳过行函数的方法，子类需要实现该方法
    def _check_skiprows_func(
        self,
        skiprows: Callable,
        rows_to_use: int,
    def _calc_rows(
        self,
        header: int | Sequence[int] | None,
        index_col: int | Sequence[int] | None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None,
        nrows: int | None,
    ) -> int | None:
        """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.

        Parameters
        ----------
        header : int, list of int, or None
            Specifies the number of rows or list of row indices that make up the header.
        index_col : int, str, list of int, or None
            Specifies which column(s) to use as the row labels of the DataFrame.
        skiprows : list-like, int, callable, or None
            Specifies rows to skip, can be a list of row numbers, single row number, or function.
        nrows : int or None
            Specifies the number of rows to read from the file.

        Returns
        -------
        int or None
            Returns the total number of rows to be read considering header, index, and skipped rows,
            or None if nrows is not specified.
        """
        if nrows is None:
            return None
        if header is None:
            header_rows = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(Sequence, header)
            header_rows = 1 + header[-1]

        # If there is a MultiIndex header and an index then there is also
        # a row containing just the index name(s)
        if is_list_like(header) and index_col is not None:
            header = cast(Sequence, header)
            if len(header) > 1:
                header_rows += 1

        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):

            def f(skiprows: Sequence, x: int) -> bool:
                return x in skiprows

            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)

        if callable(skiprows):
            return self._check_skiprows_func(
                skiprows,
                header_rows + nrows,
            )

        # else unexpected skiprows type: read_excel will not optimize
        # the number of rows read from file
        return None
    # 定义一个方法 parse，用于解析数据文件，具体参数如下：

    self,
        # self 表示类的实例本身，这个方法是一个类方法

    sheet_name: str | int | list[int] | list[str] | None = 0,
        # sheet_name：指定要读取的工作表名称或索引，可以是字符串、整数、整数列表、字符串列表或 None，默认为 0

    header: int | Sequence[int] | None = 0,
        # header：指定作为列名的行号，默认为 0，可以是整数、整数序列或 None

    names: SequenceNotStr[Hashable] | range | None = None,
        # names：指定列名的列表，要求列名不能是字符串，可以是序列（不包括字符串）、range 对象或 None

    index_col: int | Sequence[int] | None = None,
        # index_col：指定哪些列作为索引列，可以是整数、整数序列或 None

    usecols=None,
        # usecols：指定要读取的列，可以是 None 或者要读取的列的列表

    dtype: DtypeArg | None = None,
        # dtype：指定每列的数据类型，可以是数据类型、字典或 None

    true_values: Iterable[Hashable] | None = None,
        # true_values：指定哪些值应该被解释为 True，可以是可迭代对象或 None

    false_values: Iterable[Hashable] | None = None,
        # false_values：指定哪些值应该被解释为 False，可以是可迭代对象或 None

    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        # skiprows：指定要跳过的行号，可以是整数、整数序列、函数或 None

    nrows: int | None = None,
        # nrows：指定要读取的行数，可以是整数或 None

    na_values=None,
        # na_values：指定哪些值应该被视为缺失值，可以是 None 或者要被视为缺失值的值的列表

    verbose: bool = False,
        # verbose：指定是否显示详细信息，默认为 False

    parse_dates: list | dict | bool = False,
        # parse_dates：指定哪些列需要解析为日期，可以是列表、字典或布尔值，默认为 False

    date_format: dict[Hashable, str] | str | None = None,
        # date_format：指定日期的格式，可以是字典、字符串或 None

    thousands: str | None = None,
        # thousands：指定千位分隔符的字符，默认为 None

    decimal: str = ".",
        # decimal：指定小数点的字符，默认为 "."

    comment: str | None = None,
        # comment：指定注释字符，默认为 None

    skipfooter: int = 0,
        # skipfooter：指定要跳过文件末尾的行数，默认为 0

    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        # dtype_backend：指定后端数据类型处理，默认为 lib.no_default

    **kwds,
        # **kwds：接收额外的关键字参数，用于灵活处理其他未明确定义的参数
        ):
            # 验证表头参数的有效性
            validate_header_arg(header)
            # 验证 nrows 参数为整数
            validate_integer("nrows", nrows)

            ret_dict = False

            # 保留 sheet_name 以保持向后兼容性。
            sheets: list[int] | list[str]
            if isinstance(sheet_name, list):
                # 如果 sheet_name 是列表，则直接使用它作为 sheets
                sheets = sheet_name
                ret_dict = True
            elif sheet_name is None:
                # 如果 sheet_name 为 None，则使用 self.sheet_names 作为 sheets
                sheets = self.sheet_names
                ret_dict = True
            elif isinstance(sheet_name, str):
                # 如果 sheet_name 是字符串，则将其作为单元素列表
                sheets = [sheet_name]
            else:
                # 否则将 sheet_name 转换为单元素列表
                sheets = [sheet_name]

            # 处理重复类型的表单名
            sheets = cast(Union[list[int], list[str]], list(dict.fromkeys(sheets).keys()))

            output = {}

            last_sheetname = None
            for asheetname in sheets:
                last_sheetname = asheetname
                if verbose:
                    # 如果 verbose 为 True，则打印读取的表单名
                    print(f"Reading sheet {asheetname}")

                if isinstance(asheetname, str):
                    # 如果表单名是字符串，则通过名称获取表单对象
                    sheet = self.get_sheet_by_name(asheetname)
                else:  # assume an integer if not a string
                    # 如果不是字符串，则假设为整数，并通过索引获取表单对象
                    sheet = self.get_sheet_by_index(asheetname)

                # 计算需要的文件行数
                file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
                # 获取表单数据
                data = self.get_sheet_data(sheet, file_rows_needed)
                if hasattr(sheet, "close"):
                    # 如果 sheet 具有 close 方法（例如 pyxlsb），则关闭表单对象
                    sheet.close()
                # 可能转换 usecols 到适当的格式
                usecols = maybe_convert_usecols(usecols)

                if not data:
                    # 如果数据为空，则将空 DataFrame 放入输出中，并继续下一个表单
                    output[asheetname] = DataFrame()
                    continue

                # 解析表单数据并更新输出字典
                output = self._parse_sheet(
                    data=data,
                    output=output,
                    asheetname=asheetname,
                    header=header,
                    names=names,
                    index_col=index_col,
                    usecols=usecols,
                    dtype=dtype,
                    skiprows=skiprows,
                    nrows=nrows,
                    true_values=true_values,
                    false_values=false_values,
                    na_values=na_values,
                    parse_dates=parse_dates,
                    date_format=date_format,
                    thousands=thousands,
                    decimal=decimal,
                    comment=comment,
                    skipfooter=skipfooter,
                    dtype_backend=dtype_backend,
                    **kwds,
                )

            if last_sheetname is None:
                # 如果最后的表单名为 None，则抛出值错误异常
                raise ValueError("Sheet name is an empty list")

            if ret_dict:
                # 如果 ret_dict 为 True，则返回整个输出字典
                return output
            else:
                # 否则返回最后一个表单名对应的数据
                return output[last_sheetname]
    # 定义一个方法 `_parse_sheet`，用于解析数据表格中的数据
    self,
        # 数据表格的数据，通常为二维列表形式
        data: list,
        # 输出结果的字典，用于存储解析后的数据
        output: dict,
        # 指定要解析的表格名称或索引，可以是字符串、整数或空值
        asheetname: str | int | None = None,
        # 指定表格中作为标题的行号或行号序列，或为空表示默认为第一行作为标题
        header: int | Sequence[int] | None = 0,
        # 指定作为列名的序列或范围，如果不是字符串的可散列类型则用于命名列
        names: SequenceNotStr[Hashable] | range | None = None,
        # 指定作为索引的列号或列号序列，如果为空则不使用索引列
        index_col: int | Sequence[int] | None = None,
        usecols=None,  # 未提供具体信息，可能用于指定要解析的列
        dtype: DtypeArg | None = None,  # 指定列的数据类型，或者为 None 表示自动推断
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        # 跳过的行号或行号的判断函数，用于跳过读取数据时的部分行
        nrows: int | None = None,  # 指定要读取的行数，或为空表示读取全部行
        true_values: Iterable[Hashable] | None = None,  # 真值的可能值列表，用于解析布尔类型数据
        false_values: Iterable[Hashable] | None = None,  # 假值的可能值列表，用于解析布尔类型数据
        na_values=None,  # 缺失值的可能值列表，用于解析数据时的缺失值标识
        parse_dates: list | dict | bool = False,  # 是否解析日期数据，可以是列表、字典或布尔值
        date_format: dict[Hashable, str] | str | None = None,  # 日期格式的指定方式
        thousands: str | None = None,  # 数字中的千位分隔符
        decimal: str = ".",  # 数字中的小数点符号
        comment: str | None = None,  # 注释标识符，用于忽略数据文件中的注释行
        skipfooter: int = 0,  # 要跳过文件尾部的行数
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型的后端处理方式
        **kwds,  # 其余关键字参数，可能用于传递给具体的解析器
# 使用装饰器 @doc，指定 storage_options 参数的文档字符串来修饰该类
@doc(storage_options=_shared_docs["storage_options"])
# 定义 ExcelWriter 类，用于将 DataFrame 对象写入 Excel 表格
class ExcelWriter(Generic[_WorkbookT]):
    """
    Class for writing DataFrame objects into excel sheets.

    Default is to use:

    * `xlsxwriter <https://pypi.org/project/XlsxWriter/>`__ for xlsx files if xlsxwriter
      is installed otherwise `openpyxl <https://pypi.org/project/openpyxl/>`__
    * `odswriter <https://pypi.org/project/odswriter/>`__ for ods files

    See :meth:`DataFrame.to_excel` for typical usage.

    The writer should be used as a context manager. Otherwise, call `close()` to save
    and close any opened file handles.

    Parameters
    ----------
    path : str or typing.BinaryIO
        Path to xls or xlsx or ods file.
    engine : str (optional)
        Engine to use for writing. If None, defaults to
        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword
        argument.
    date_format : str, default None
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
    datetime_format : str, default None
        Format string for datetime objects written into Excel files.
        (e.g. 'YYYY-MM-DD HH:MM:SS').
    mode : {{'w', 'a'}}, default 'w'
        File mode to use (write or append). Append does not work with fsspec URLs.
    {storage_options}
    
    if_sheet_exists : {{'error', 'new', 'replace', 'overlay'}}, default 'error'
        How to behave when trying to write to a sheet that already
        exists (append mode only).

        * error: raise a ValueError.
        * new: Create a new sheet, with a name determined by the engine.
        * replace: Delete the contents of the sheet before writing to it.
        * overlay: Write contents to the existing sheet without first removing,
          but possibly over top of, the existing contents.

        .. versionadded:: 1.3.0

        .. versionchanged:: 1.4.0

           Added ``overlay`` option

    engine_kwargs : dict, optional
        Keyword arguments to be passed into the engine. These will be passed to
        the following functions of the respective engines:

        * xlsxwriter: ``xlsxwriter.Workbook(file, **engine_kwargs)``
        * openpyxl (write mode): ``openpyxl.Workbook(**engine_kwargs)``
        * openpyxl (append mode): ``openpyxl.load_workbook(file, **engine_kwargs)``
        * odswriter: ``odf.opendocument.OpenDocumentSpreadsheet(**engine_kwargs)``

        .. versionadded:: 1.3.0

    See Also
    --------
    read_excel : Read an Excel sheet values (xlsx) file into DataFrame.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Notes
    -----
    For compatibility with CSV writers, ExcelWriter serializes lists
    and dicts to strings before writing.

    Examples
    --------
    Default usage:

    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    # 将数据框写入 Excel 文件，但跳过此行的执行（仅用于文档测试）
    df.to_excel(writer)  # doctest: +SKIP

    # 将两个数据框写入同一个文件的不同工作表中
    df1 = pd.DataFrame([["AAA", "BBB"]], columns=["Spam", "Egg"])  # doctest: +SKIP
    df2 = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    with pd.ExcelWriter("path_to_file.xlsx") as writer:
        df1.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP
        df2.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP

    # 可以设置日期或日期时间格式
    from datetime import date, datetime  # doctest: +SKIP
    df = pd.DataFrame(
        [
            [date(2014, 1, 31), date(1999, 9, 24)],
            [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
        ],
        index=["Date", "Datetime"],
        columns=["X", "Y"],
    )  # doctest: +SKIP
    with pd.ExcelWriter(
        "path_to_file.xlsx",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        df.to_excel(writer)  # doctest: +SKIP

    # 可以将数据框追加到现有的 Excel 文件中的新工作表
    with pd.ExcelWriter("path_to_file.xlsx", mode="a", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet3")  # doctest: +SKIP

    # 可以替换已存在的工作表，根据 if_sheet_exists 参数的设定
    with pd.ExcelWriter(
        "path_to_file.xlsx",
        mode="a",
        engine="openpyxl",
        if_sheet_exists="replace",
    ) as writer:
        df.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP

    # 可以将多个数据框写入单个工作表，需要将 if_sheet_exists 参数设为 overlay
    with pd.ExcelWriter(
        "path_to_file.xlsx",
        mode="a",
        engine="openpyxl",
        if_sheet_exists="overlay",
    ) as writer:
        df1.to_excel(writer, sheet_name="Sheet1")
        df2.to_excel(writer, sheet_name="Sheet1", startcol=3)  # doctest: +SKIP

    # 可以将 Excel 文件存储在内存中
    import io
    df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer)

    # 可以将 Excel 文件打包成 zip 压缩文件
    import zipfile  # doctest: +SKIP
    df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    with zipfile.ZipFile("path_to_file.zip", "w") as zf:
        with zf.open("filename.xlsx", "w") as buffer:
            with pd.ExcelWriter(buffer) as writer:
                df.to_excel(writer)  # doctest: +SKIP

    # 可以为底层引擎指定额外的参数
    with pd.ExcelWriter(
        "path_to_file.xlsx",
        engine="xlsxwriter",
    # 定义一个 ExcelWriter 实现（查看更多抽象方法...）
    
    # - 必需的方法
    #   - ``write_cells(self, cells, sheet_name=None, startrow=0, startcol=0)``
    #     --> 用于将额外的 DataFrame 写入磁盘
    #   - ``_supported_extensions``（支持的扩展名元组），用于检查引擎是否支持给定的扩展名。
    #   - ``_engine`` - 字符串，表示引擎名称。直接实例化类并绕过 ``ExcelWriterMeta`` 引擎查找时必需。
    #   - ``save(self)`` --> 用于将文件保存到磁盘
    # - 大部分情况下是必需的（至少应该存在）
    #   - book, cur_sheet, path
    
    # - 可选的方法：
    #   - ``__init__(self, path, engine=None, **kwargs)`` --> 始终以路径作为第一个参数调用。
    
    # 还需要使用 ``register_writer()`` 注册该类。
    # 从技术上讲，ExcelWriter 的实现不需要是 ExcelWriter 的子类。
    
    _engine: str
    _supported_extensions: tuple[str, ...]
    
    def __new__(
        cls,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict | None = None,
    ) -> Self:
        # 只有在泛型 ExcelWriter 时才切换类
        if cls is ExcelWriter:
            if engine is None or (isinstance(engine, str) and engine == "auto"):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[-1][1:]
                else:
                    ext = "xlsx"
    
                try:
                    engine = config.get_option(f"io.excel.{ext}.writer")
                    if engine == "auto":
                        engine = get_default_engine(ext, mode="writer")
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err
    
            # for mypy
            assert engine is not None
            # error: Incompatible types in assignment (expression has type
            # "type[ExcelWriter[Any]]", variable has type "type[Self]")
            cls = get_writer(engine)  # type: ignore[assignment]
    
        return object.__new__(cls)
    
    # 声明可以依赖的外部属性
    _path = None
    
    @property
    # 返回支持的文件扩展名的元组
    def supported_extensions(self) -> tuple[str, ...]:
        """Extensions that writer engine supports."""
        return self._supported_extensions

    # 返回引擎的名称
    @property
    def engine(self) -> str:
        """Name of engine."""
        return self._engine

    # 返回一个字典，将工作表名称映射到工作表对象
    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        raise NotImplementedError

    # 返回工作簿实例，具体的类类型取决于所使用的引擎
    @property
    def book(self) -> _WorkbookT:
        """
        Book instance. Class type will depend on the engine used.

        This attribute can be used to access engine-specific features.
        """
        raise NotImplementedError

    # 将格式化的单元格数据写入到 Excel 工作表中
    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        """
        Write given formatted cells into Excel an excel sheet

        Parameters
        ----------
        cells : generator
            cell of formatted data to save to Excel sheet
        sheet_name : str, default None
            Name of Excel sheet, if None, then use self.cur_sheet
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame
        freeze_panes: int tuple of length 2
            contains the bottom-most row and right-most column to freeze
        """
        raise NotImplementedError

    # 将工作簿保存到磁盘
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        raise NotImplementedError

    # 初始化方法，接受路径、引擎类型、日期和时间格式、模式等参数
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ):
        # 定义一个类型提示，指定该方法没有返回值
        ) -> None:
        # 验证引擎是否能处理给定路径的文件扩展名
        if isinstance(path, str):
            # 获取文件路径的扩展名
            ext = os.path.splitext(path)[-1]
            # 调用函数检查文件扩展名是否有效
            self.check_extension(ext)

        # 根据模式打开文件
        if "b" not in mode:
            mode += "b"
        # 如果模式包含 "a"，则替换为 "r+"，以便让 Excel 后端先读取现有文件，然后追加数据
        mode = mode.replace("a", "r+")

        # 如果 if_sheet_exists 不是 None、"error"、"new"、"replace"、"overlay" 中的一个，抛出 ValueError 异常
        if if_sheet_exists not in (None, "error", "new", "replace", "overlay"):
            raise ValueError(
                f"'{if_sheet_exists}' is not valid for if_sheet_exists. "
                "Valid options are 'error', 'new', 'replace' and 'overlay'."
            )
        # 如果 if_sheet_exists 不是 None 并且模式不包含 "r+"，抛出 ValueError 异常
        if if_sheet_exists and "r+" not in mode:
            raise ValueError("if_sheet_exists is only valid in append mode (mode='a')")
        # 如果 if_sheet_exists 是 None，则将其设为 "error"
        if if_sheet_exists is None:
            if_sheet_exists = "error"
        # 设置对象的 _if_sheet_exists 属性为 if_sheet_exists 的值
        self._if_sheet_exists = if_sheet_exists

        # 将 ExcelWriter 强制转换以避免添加 'if self._handles is not None'
        self._handles = IOHandles(
            cast(IO[bytes], path), compression={"compression": None}
        )
        # 如果 path 不是 ExcelWriter 类型，则调用 get_handle 获取文件句柄
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(
                path, mode, storage_options=storage_options, is_text=False
            )
        # 将当前工作表设置为 None
        self._cur_sheet = None

        # 如果 date_format 为 None，则设置默认日期格式为 "YYYY-MM-DD"
        if date_format is None:
            self._date_format = "YYYY-MM-DD"
        else:
            self._date_format = date_format
        # 如果 datetime_format 为 None，则设置默认日期时间格式为 "YYYY-MM-DD HH:MM:SS"
        if datetime_format is None:
            self._datetime_format = "YYYY-MM-DD HH:MM:SS"
        else:
            self._datetime_format = datetime_format

        # 设置对象的 _mode 属性为传入的 mode
        self._mode = mode

    @property
    def date_format(self) -> str:
        """
        返回写入 Excel 文件的日期格式字符串（例如 'YYYY-MM-DD'）。
        """
        return self._date_format

    @property
    def datetime_format(self) -> str:
        """
        返回写入 Excel 文件的日期时间格式字符串（例如 'YYYY-MM-DD HH:MM:SS'）。
        """
        return self._datetime_format

    @property
    def if_sheet_exists(self) -> str:
        """
        返回在追加模式下写入已存在工作表时的行为设置。
        """
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        """
        返回句柄的文件路径名，用于文件系统路径操作。
        """
        return getattr(self._handles.handle, "name", "")

    def _get_sheet_name(self, sheet_name: str | None) -> str:
        """
        根据给定的工作表名或当前工作表名返回有效的工作表名。

        如果 sheet_name 是 None，则使用当前的 _cur_sheet 属性；
        如果 _cur_sheet 也是 None，则抛出 ValueError 异常。
        """
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:  # pragma: no cover
            raise ValueError("Must pass explicit sheet_name or set _cur_sheet property")
        return sheet_name

    def _value_with_fmt(
        self, val
    ) -> tuple[
        int | float | bool | str | datetime.datetime | datetime.date, str | None
    ```
    def _write_cell(self, val: object) -> Tuple[Any, Optional[str]]:
        """
        Convert numpy types to Python types for the Excel writers.

        Parameters
        ----------
        val : object
            Value to be written into cells

        Returns
        -------
        Tuple with the first element being the converted value and the second
        being an optional format
        """
        fmt = None  # 初始化 fmt 变量为 None

        if is_integer(val):  # 如果 val 是整数类型
            val = int(val)  # 转换为整数
        elif is_float(val):  # 如果 val 是浮点数类型
            val = float(val)  # 转换为浮点数
        elif is_bool(val):  # 如果 val 是布尔类型
            val = bool(val)  # 转换为布尔值
        elif isinstance(val, datetime.datetime):  # 如果 val 是 datetime.datetime 类型
            fmt = self._datetime_format  # 设置格式为类中定义的日期时间格式
        elif isinstance(val, datetime.date):  # 如果 val 是 datetime.date 类型
            fmt = self._date_format  # 设置格式为类中定义的日期格式
        elif isinstance(val, datetime.timedelta):  # 如果 val 是 datetime.timedelta 类型
            val = val.total_seconds() / 86400  # 转换为 Excel 时间格式
            fmt = "0"  # 设置格式为数字
        else:  # 对于其他类型的 val
            val = str(val)  # 转换为字符串
            # GH#56954
            # Excel 对单元格内容的限制是 32767 个字符
            # 参考链接 https://support.microsoft.com/en-au/office/excel-specifications-and-limits-1672b34d-7043-467e-8e27-269d656771c3
            if len(val) > 32767:  # 如果字符串长度超过 32767
                warnings.warn(
                    f"Cell contents too long ({len(val)}), "
                    "truncated to 32767 characters",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
        return val, fmt  # 返回转换后的值和格式

    @classmethod
    def check_extension(cls, ext: str) -> Literal[True]:
        """
        检查文件扩展名是否在该写入器支持的扩展名列表中，如果不支持，则引发 UnsupportedFiletypeError。
        """
        if ext.startswith("."):  # 如果扩展名以点号开头
            ext = ext[1:]  # 去掉点号
        if not any(ext in extension for extension in cls._supported_extensions):  # 如果扩展名不在支持的扩展名列表中
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")  # 抛出异常
        return True  # 扩展名有效，返回 True

    # 允许作为上下文管理器使用
    def __enter__(self) -> Self:
        return self  # 返回自身对象

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()  # 调用 close 方法关闭资源

    def close(self) -> None:
        """synonym for save, to make it more file-like"""
        self._save()  # 调用 _save 方法保存数据
        self._handles.close()  # 关闭文件句柄
XLS_SIGNATURES = (
    b"\x09\x00\x04\x00\x07\x00\x10\x00",  # BIFF2，Excel二进制文件格式版本2的签名
    b"\x09\x02\x06\x00\x00\x00\x10\x00",  # BIFF3，Excel二进制文件格式版本3的签名
    b"\x09\x04\x06\x00\x00\x00\x10\x00",  # BIFF4，Excel二进制文件格式版本4的签名
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # Compound File Binary，复合文件二进制格式的签名
)
ZIP_SIGNATURE = b"PK\x03\x04"  # ZIP文件的签名
PEEK_SIZE = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))  # 获取所有签名的最大长度作为预览大小


@doc(storage_options=_shared_docs["storage_options"])
def inspect_excel_format(
    content_or_path: FilePath | ReadBuffer[bytes],
    storage_options: StorageOptions | None = None,
) -> str | None:
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    {storage_options}

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    with get_handle(
        content_or_path, "rb", storage_options=storage_options, is_text=False
    ) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)  # 从流中读取预览大小的数据
        if buf is None:
            raise ValueError("stream is empty")
        assert isinstance(buf, bytes)
        peek = buf
        stream.seek(0)

        if any(peek.startswith(sig) for sig in XLS_SIGNATURES):
            return "xls"  # 如果流的开头匹配任意一种Excel文件格式的签名，则返回"xls"
        elif not peek.startswith(ZIP_SIGNATURE):
            return None  # 如果流的开头不是ZIP文件的签名，则返回None

        with zipfile.ZipFile(stream) as zf:
            # Workaround for some third party files that use forward slashes and
            # lower case names.
            component_names = {
                name.replace("\\", "/").lower() for name in zf.namelist()
            }  # 将ZIP文件中所有文件名转换为小写，并替换反斜杠为正斜杠

        if "xl/workbook.xml" in component_names:
            return "xlsx"  # 如果ZIP文件包含"xl/workbook.xml"，则为xlsx格式
        if "xl/workbook.bin" in component_names:
            return "xlsb"  # 如果ZIP文件包含"xl/workbook.bin"，则为xlsb格式
        if "content.xml" in component_names:
            return "ods"  # 如果ZIP文件包含"content.xml"，则为ods格式
        return "zip"  # 默认返回zip格式


@doc(storage_options=_shared_docs["storage_options"])
class ExcelFile:
    """
    Class for parsing tabular Excel sheets into DataFrame objects.

    See read_excel for more documentation.

    Parameters
    ----------
    path_or_buffer : str, bytes, pathlib.Path,
        A file-like object, xlrd workbook or openpyxl workbook.
        If a string or path object, expected to be a path to a
        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.
    """
    """
    engine : str, default None
        如果 io 不是缓冲区或路径，则必须设置此参数以识别 io。
        支持的引擎有：``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``, ``calamine``
        引擎兼容性：
    
        - ``xlrd`` 支持旧版 Excel 文件（.xls）。
        - ``openpyxl`` 支持较新的 Excel 文件格式。
        - ``odf`` 支持 OpenDocument 文件格式（.odf, .ods, .odt）。
        - ``pyxlsb`` 支持二进制 Excel 文件。
        - ``calamine`` 支持 Excel （.xls, .xlsx, .xlsm, .xlsb）
          和 OpenDocument （.ods）文件格式。
    
        .. versionchanged:: 1.2.0
    
           引擎 `xlrd <https://xlrd.readthedocs.io/en/latest/>`_
           现在仅支持旧版 ``.xls`` 文件。
           当 ``engine=None`` 时，将使用以下逻辑确定引擎：
    
           - 如果 ``path_or_buffer`` 是 OpenDocument 格式（.odf, .ods, .odt），
             则使用 `odf <https://pypi.org/project/odfpy/>`_。
           - 否则，如果 ``path_or_buffer`` 是 xls 格式，
             则使用 ``xlrd``。
           - 否则，如果 ``path_or_buffer`` 是 xlsb 格式，
             则使用 `pyxlsb <https://pypi.org/project/pyxlsb/>`_。
    
        .. versionadded:: 1.3.0
    
           - 否则，如果安装了 `openpyxl <https://pypi.org/project/openpyxl/>`_，
             则使用 ``openpyxl``。
           - 否则，如果安装了 ``xlrd >= 2.0``，将引发 ``ValueError``。
    
        .. warning::
    
           使用 ``xlrd`` 读取 ``.xlsx`` 文件时请不要报告问题。
           这是不支持的，请转而使用 ``openpyxl``。
    
    {storage_options}
    engine_kwargs : dict, optional
        传递给 Excel 引擎的任意关键字参数。
    
    See Also
    --------
    DataFrame.to_excel : 将 DataFrame 写入 Excel 文件。
    DataFrame.to_csv : 将 DataFrame 写入逗号分隔值（csv）文件。
    read_csv : 从逗号分隔值（csv）文件读取数据到 DataFrame。
    read_fwf : 从固定宽度格式的行读取表格数据到 DataFrame。
    
    Examples
    --------
    >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
    >>> with pd.ExcelFile("myfile.xls") as xls:  # doctest: +SKIP
    ...     df1 = pd.read_excel(xls, "Sheet1")  # doctest: +SKIP
    """
    
    from pandas.io.excel._calamine import CalamineReader
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader
    
    _engines: Mapping[str, Any] = {
        "xlrd": XlrdReader,
        "openpyxl": OpenpyxlReader,
        "odf": ODFReader,
        "pyxlsb": PyxlsbReader,
        "calamine": CalamineReader,
    }
    ) -> None:
        # 如果 engine_kwargs 为 None，则设为一个空字典
        if engine_kwargs is None:
            engine_kwargs = {}

        # 如果指定了 engine，并且它不在已知的引擎列表中，则抛出异常
        if engine is not None and engine not in self._engines:
            raise ValueError(f"Unknown engine: {engine}")

        # 将 path_or_buffer 转换为字符串并赋值给 self._io
        # 总是得到一个字符串类型的路径
        self._io = stringify_path(path_or_buffer)

        # 如果未指定 engine
        if engine is None:
            # 只有在需要的时候才确定文件扩展名
            ext: str | None = None

            # 如果 path_or_buffer 不是字符串、PathLike 对象或 ExcelFile 对象，并且不是文件对象
            if not isinstance(
                path_or_buffer, (str, os.PathLike, ExcelFile)
            ) and not is_file_like(path_or_buffer):
                # GH#56692 - 尽量避免导入 xlrd 库
                if import_optional_dependency("xlrd", errors="ignore") is None:
                    xlrd_version = None
                else:
                    import xlrd

                    xlrd_version = Version(get_version(xlrd))

                # 如果 path_or_buffer 是 xlrd.Book 类型，则设定 ext 为 "xls"
                if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                    ext = "xls"

            # 如果 ext 仍然是 None，则调用 inspect_excel_format 函数确定扩展名
            if ext is None:
                ext = inspect_excel_format(
                    content_or_path=path_or_buffer, storage_options=storage_options
                )
                # 如果无法确定扩展名，则抛出 ValueError 异常
                if ext is None:
                    raise ValueError(
                        "Excel file format cannot be determined, you must specify "
                        "an engine manually."
                    )

            # 根据 ext 从配置中获取 engine
            engine = config.get_option(f"io.excel.{ext}.reader")
            # 如果 engine 是 "auto"，则调用 get_default_engine 函数获取默认的引擎
            if engine == "auto":
                engine = get_default_engine(ext, mode="reader")

        # 断言 engine 不为 None
        assert engine is not None
        # 将 engine 和 storage_options 赋值给实例变量
        self.engine = engine
        self.storage_options = storage_options

        # 使用选定的引擎创建 self._reader 对象
        self._reader = self._engines[engine](
            self._io,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    # 实现 __fspath__ 方法，返回 self._io 的值
    def __fspath__(self):
        return self._io

    # 定义 parse 方法，解析 Excel 文件内容
    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        converters=None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        parse_dates: list | dict | bool = False,
        date_format: str | dict[Hashable, str] | None = None,
        thousands: str | None = None,
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ):
        # 这里应该是方法定义的开始，缺失了类名，可能是因为代码被截断了
    def book(self):
        """
        Gets the Excel workbook.

        Workbook is the top-level container for all document information.

        Returns
        -------
        Excel Workbook
            The workbook object of the type defined by the engine being used.

        See Also
        --------
        read_excel : Read an Excel file into a pandas DataFrame.

        Examples
        --------
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.book  # doctest: +SKIP
        <openpyxl.workbook.workbook.Workbook object at 0x11eb5ad70>
        >>> file.book.path  # doctest: +SKIP
        '/xl/workbook.xml'
        >>> file.book.active  # doctest: +SKIP
        <openpyxl.worksheet._read_only.ReadOnlyWorksheet object at 0x11eb5b370>
        >>> file.book.sheetnames  # doctest: +SKIP
        ['Sheet1', 'Sheet2']
        """
        return self._reader.book

    @property
    def sheet_names(self):
        """
        Names of the sheets in the document.

        This is particularly useful for loading a specific sheet into a DataFrame when
        you do not know the sheet names beforehand.

        Returns
        -------
        list of str
            List of sheet names in the document.

        See Also
        --------
        ExcelFile.parse : Parse a sheet into a DataFrame.
        read_excel : Read an Excel file into a pandas DataFrame. If you know the sheet
            names, it may be easier to specify them directly to read_excel.

        Examples
        --------
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.sheet_names  # doctest: +SKIP
        ["Sheet1", "Sheet2"]
        """
        return self._reader.sheet_names

    def close(self) -> None:
        """Close any open resources associated with the reader."""
        self._reader.close()

    def __enter__(self) -> Self:
        """Enter the context manager and return self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager and close resources if an exception occurred."""
        self.close()
```