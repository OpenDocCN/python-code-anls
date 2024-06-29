# `D:\src\scipysrc\pandas\pandas\io\json\_json.py`

```
# 从未来版本导入注解功能，允许使用类型注解中的“|”操作符
from __future__ import annotations

# 导入抽象基类ABC和abstractmethod装饰器
from abc import (
    ABC,
    abstractmethod,
)

# 导入集合类的抽象基类
from collections import abc

# 导入迭代工具函数islice
from itertools import islice

# 导入类型注解相关的模块和类型
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    final,
    overload,
)

# 导入第三方数值计算库numpy并简写为np
import numpy as np

# 导入pandas内部库
from pandas._libs import lib

# 导入pandas内部库中的JSON处理模块ujson_dumps和ujson_loads
from pandas._libs.json import (
    ujson_dumps,
    ujson_loads,
)

# 导入pandas时间序列库tslibs中的iNaT
from pandas._libs.tslibs import iNaT

# 导入pandas兼容性库中的依赖导入函数
from pandas.compat._optional import import_optional_dependency

# 导入pandas错误处理模块中的抽象方法错误异常
from pandas.errors import AbstractMethodError

# 导入pandas工具模块中的文档装饰器
from pandas.util._decorators import doc

# 导入pandas工具模块中的数据类型验证函数
from pandas.util._validators import check_dtype_backend

# 导入pandas核心数据类型中的常用函数和类
from pandas.core.dtypes.common import (
    ensure_str,
    is_string_dtype,
)

# 导入pandas核心数据类型中的时间周期数据类型
from pandas.core.dtypes.dtypes import PeriodDtype

# 导入pandas核心模块
from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
    to_datetime,
)

# 导入pandas核心数据重塑模块中的连接函数concat
from pandas.core.reshape.concat import concat

# 导入pandas核心共享文档模块中的共享文档函数和类
from pandas.core.shared_docs import _shared_docs

# 导入pandas输入输出模块中的公共函数和类
from pandas.io.common import (
    IOHandles,
    dedup_names,
    get_handle,
    is_potential_multi_index,
    stringify_path,
)

# 导入pandas JSON数据处理模块中的行分隔处理函数
from pandas.io.json._normalize import convert_to_line_delimits

# 导入pandas JSON数据处理模块中的表模式构建和解析函数
from pandas.io.json._table_schema import (
    build_table_schema,
    parse_table_schema,
)

# 导入pandas解析器模块中的整数验证函数
from pandas.io.parsers.readers import validate_integer

# 如果是类型检查阶段，则导入额外的类型注解
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Mapping,
    )
    from types import TracebackType

    # 导入pandas类型注解中的多种类型
    from pandas._typing import (
        CompressionOptions,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        JSONEngine,
        JSONSerializable,
        ReadBuffer,
        Self,
        StorageOptions,
        WriteBuffer,
    )

    # 导入pandas核心泛型模块中的数据框架类型NDFrame
    from pandas.core.generic import NDFrame

# 定义类型变量FrameSeriesStrT，表示是数据框架还是序列
FrameSeriesStrT = TypeVar("FrameSeriesStrT", bound=Literal["frame", "series"])

# JSON序列化函数to_json的重载定义，支持多种参数组合
@overload
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> None: ...


@overload
def to_json(
    path_or_buf: None,
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> str: ...


# JSON序列化函数to_json的最终实现定义，支持多种参数组合
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] | None,
    obj: NDFrame,
    # obj 参数：表示一个 NDFrame 对象，可能是 pandas 中的 DataFrame 或 Series

    orient: str | None = None,
    # orient 参数：指定序列化 JSON 时的方向，可选值为字符串或 None，默认为 None

    date_format: str = "epoch",
    # date_format 参数：指定日期序列化的格式，默认为 "epoch"

    double_precision: int = 10,
    # double_precision 参数：控制浮点数的精度，默认为 10

    force_ascii: bool = True,
    # force_ascii 参数：指定是否强制使用 ASCII 编码，默认为 True

    date_unit: str = "ms",
    # date_unit 参数：指定日期的单位，默认为 "ms"（毫秒）

    default_handler: Callable[[Any], JSONSerializable] | None = None,
    # default_handler 参数：用于定制对象序列化的处理函数，可以是一个可调用对象或者 None

    lines: bool = False,
    # lines 参数：指定是否按行序列化 JSON，默认为 False

    compression: CompressionOptions = "infer",
    # compression 参数：指定写入时的压缩选项，可以是 "infer" 或具体的压缩算法

    index: bool | None = None,
    # index 参数：指定是否包含索引，默认为 None

    indent: int = 0,
    # indent 参数：指定输出 JSON 时的缩进空格数，默认为 0

    storage_options: StorageOptions | None = None,
    # storage_options 参数：存储选项，通常用于文件存储时的额外参数，可以是一个字典或 None

    mode: Literal["a", "w"] = "w",
    # mode 参数：指定文件打开模式，可以是 "a"（追加）或 "w"（覆盖），默认为 "w"
# 定义一个方法签名，接收 orient、index、mode、lines、obj、date_format、double_precision、
# force_ascii、date_unit、default_handler、index 和 indent 参数，返回一个字符串或 None
def to_json(
    orient: str | None,
    index: bool | None = None,
    mode: str = "w",
    lines: bool = False,
    obj: NDFrame,
    date_format: str,
    double_precision: int,
    force_ascii: bool,
    date_unit: str,
    default_handler: Callable[[Any], JSONSerializable] | None = None,
    index: int = 0,
) -> str | None:
    # 如果 orient 是 "records" 或 "values"，并且 index 是 True，则抛出 ValueError 异常
    if orient in ["records", "values"] and index is True:
        raise ValueError(
            "'index=True' is only valid when 'orient' is 'split', 'table', "
            "'index', or 'columns'."
        )
    # 如果 orient 是 "index" 或 "columns"，并且 index 是 False，则抛出 ValueError 异常
    elif orient in ["index", "columns"] and index is False:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split', 'table', "
            "'records', or 'values'."
        )
    # 如果 index 是 None，并且 orient 不是 "records" 和 "values"，则将 index 设置为 True
    elif index is None:
        index = True  # will be ignored for orient='records' and 'values'

    # 如果 lines 为 True，并且 orient 不是 "records"，则抛出 ValueError 异常
    if lines and orient != "records":
        raise ValueError("'lines' keyword only valid when 'orient' is records")

    # 如果 mode 不在 ["a", "w"] 中，则抛出 ValueError 异常
    if mode not in ["a", "w"]:
        msg = (
            f"mode={mode} is not a valid option."
            "Only 'w' and 'a' are currently supported."
        )
        raise ValueError(msg)

    # 如果 mode 是 "a" 并且 lines 不为 True 或 orient 不是 "records"，则抛出 ValueError 异常
    if mode == "a" and (not lines or orient != "records"):
        msg = (
            "mode='a' (append) is only supported when "
            "lines is True and orient is 'records'"
        )
        raise ValueError(msg)

    # 如果 orient 是 "table" 并且 obj 是 Series 类型，则将 obj 转换为 DataFrame
    if orient == "table" and isinstance(obj, Series):
        obj = obj.to_frame(name=obj.name or "values")

    # 根据 obj 的类型选择合适的 Writer 类型
    writer: type[Writer]
    if orient == "table" and isinstance(obj, DataFrame):
        writer = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")

    # 使用选定的 Writer 类型实例化 writer 对象，并调用其 write 方法，返回结果给 s
    s = writer(
        obj,
        orient=orient,
        date_format=date_format,
        double_precision=double_precision,
        ensure_ascii=force_ascii,
        date_unit=date_unit,
        default_handler=default_handler,
        index=index,
        indent=indent,
    ).write()

    # 如果 lines 为 True，则将 s 转换为以行分隔的字符串
    if lines:
        s = convert_to_line_delimits(s)

    # 如果 path_or_buf 不为 None，则使用 get_handle 方法获取文件处理对象 handles，
    # 并将 s 写入到 handles.handle 中，应用压缩和字节/文本转换
    if path_or_buf is not None:
        with get_handle(
            path_or_buf, mode, compression=compression, storage_options=storage_options
        ) as handles:
            handles.handle.write(s)
    else:
        # 如果 path_or_buf 为 None，则返回 s
        return s
    # 返回 None
    return None
    # 定义一个抽象方法，子类需要实现该方法来格式化图表的轴
    def _format_axes(self) -> None:
        # 抛出抽象方法错误，提示子类必须实现该方法
        raise AbstractMethodError(self)
    
    # 定义一个写入方法，返回一个JSON格式的字符串
    def write(self) -> str:
        # 检查日期格式是否为ISO格式
        iso_dates = self.date_format == "iso"
        # 将对象转换为JSON字符串，使用ujson进行序列化
        return ujson.dumps(
            self.obj_to_write,
            orient=self.orient,
            double_precision=self.double_precision,
            ensure_ascii=self.ensure_ascii,
            date_unit=self.date_unit,
            iso_dates=iso_dates,
            default_handler=self.default_handler,
            indent=self.indent,
        )
    
    # 定义一个属性方法，用于子类实现，将数据对象转换为JSON格式写入
    @property
    @abstractmethod
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        """Object to write in JSON format."""
class SeriesWriter(Writer):
    _default_orient = "index"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # 如果不需要索引且 orient 为 "split"，返回一个包含对象名称和数值的字典
        if not self.index and self.orient == "split":
            return {"name": self.obj.name, "data": self.obj.values}
        else:
            return self.obj

    def _format_axes(self) -> None:
        # 如果对象的索引不唯一且 orient 为 "index"，抛出数值错误
        if not self.obj.index.is_unique and self.orient == "index":
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")


class FrameWriter(Writer):
    _default_orient = "columns"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # 如果不需要索引且 orient 为 "split"，将对象转换为以 split 方式组织的字典，去除索引信息
        if not self.index and self.orient == "split":
            obj_to_write = self.obj.to_dict(orient="split")
            del obj_to_write["index"]
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self) -> None:
        """
        Try to format axes if they are datelike.
        """
        # 如果对象的索引不唯一且 orient 在 "index", "columns" 中，抛出数值错误
        if not self.obj.index.is_unique and self.orient in ("index", "columns"):
            raise ValueError(
                f"DataFrame index must be unique for orient='{self.orient}'."
            )
        # 如果对象的列名不唯一且 orient 在 "index", "columns", "records" 中，抛出数值错误
        if not self.obj.columns.is_unique and self.orient in (
            "index",
            "columns",
            "records",
        ):
            raise ValueError(
                f"DataFrame columns must be unique for orient='{self.orient}'."
            )


class JSONTableWriter(FrameWriter):
    _default_orient = "records"

    def __init__(
        self,
        obj,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ):
        # 这里应添加初始化的参数和其作用的注释，但是由于截断的原因，这里不展示具体内容。
    ) -> None:
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        # 调用父类构造函数，初始化对象属性
        super().__init__(
            obj,
            orient,
            date_format,
            double_precision,
            ensure_ascii,
            date_unit,
            index,
            default_handler=default_handler,
            indent=indent,
        )

        # 检查日期格式是否为'iso'，否则抛出异常
        if date_format != "iso":
            msg = (
                "Trying to write with `orient='table'` and "
                f"`date_format='{date_format}'`. Table Schema requires dates "
                "to be formatted with `date_format='iso'`"
            )
            raise ValueError(msg)

        # 根据传入的数据对象 obj 构建数据表模式 schema
        self.schema = build_table_schema(obj, index=self.index)

        # 如果数据对象是二维且列为 MultiIndex，则不支持 orient='table'
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError(
                "orient='table' is not supported for MultiIndex columns"
            )

        # 检查索引和列之间是否存在重叠的名称，抛出异常
        if (
            (obj.ndim == 1)
            and (obj.name in set(obj.index.names))
            or len(obj.columns.intersection(obj.index.names))
        ):
            msg = "Overlapping names between the index and columns"
            raise ValueError(msg)

        # 选择数据中包含 'timedelta' 类型的列，并将其转换为 ISO 格式字符串
        timedeltas = obj.select_dtypes(include=["timedelta"]).columns
        copied = False
        if len(timedeltas):
            obj = obj.copy()
            copied = True
            obj[timedeltas] = obj[timedeltas].map(lambda x: x.isoformat())

        # 如果不包含索引，则重置对象的索引
        if not self.index:
            self.obj = obj.reset_index(drop=True)
        else:
            # 将 PeriodIndex 转换为日期时间类型
            if isinstance(obj.index.dtype, PeriodDtype):
                if not copied:
                    obj = obj.copy(deep=False)
                obj.index = obj.index.to_timestamp()
            self.obj = obj.reset_index(drop=False)
        self.date_format = "iso"
        self.orient = "records"
        self.index = index

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # 返回一个字典，包含要写入的对象的模式和数据
        return {"schema": self.schema, "data": self.obj}
# read_json 函数的重载：接受不同参数并返回不同类型的 JSONReader 对象

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,  # 读取数据的块大小
    compression: CompressionOptions = ...,  # 压缩选项，指定数据是否压缩
    nrows: int | None = ...,  # 读取的行数限制
    storage_options: StorageOptions = ...,  # 存储选项，关于数据存储的配置信息
    dtype_backend: DtypeBackend | lib.NoDefault = ...,  # 数据类型后端配置
    engine: JSONEngine = ...,  # JSON 解析引擎
) -> JsonReader[Literal["frame"]]: ...  # 返回一个 DataFrame 的 JSONReader 对象

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],  # 读取类型为 series
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,  # 读取数据的块大小
    compression: CompressionOptions = ...,  # 压缩选项，指定数据是否压缩
    nrows: int | None = ...,  # 读取的行数限制
    storage_options: StorageOptions = ...,  # 存储选项，关于数据存储的配置信息
    dtype_backend: DtypeBackend | lib.NoDefault = ...,  # 数据类型后端配置
    engine: JSONEngine = ...,  # JSON 解析引擎
) -> JsonReader[Literal["series"]]: ...  # 返回一个 Series 的 JSONReader 对象

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],  # 读取类型为 series
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,  # 不使用块大小，可能为单块读取
    compression: CompressionOptions = ...,  # 压缩选项，指定数据是否压缩
    nrows: int | None = ...,  # 读取的行数限制
    storage_options: StorageOptions = ...,  # 存储选项，关于数据存储的配置信息
    dtype_backend: DtypeBackend | lib.NoDefault = ...,  # 数据类型后端配置
    engine: JSONEngine = ...,  # JSON 解析引擎
) -> Series: ...  # 返回一个 Series 对象

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,  # 读取类型为 frame
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,  # 不使用块大小，可能为单块读取
    compression: CompressionOptions = ...,  # 压缩选项，指定数据是否压缩
    nrows: int | None = ...,  # 读取的行数限制
    storage_options: StorageOptions = ...,  # 存储选项，关于数据存储的配置信息
    dtype_backend: DtypeBackend | lib.NoDefault = ...,  # 数据类型后端配置
    engine: JSONEngine = ...,  # JSON 解析引擎
) -> DataFrame: ...  # 返回一个 DataFrame 对象

@doc(
    storage_options=_shared_docs["storage_options"],
    # 使用共享文档中的"decompression_options"条目，根据提供的"path_or_buf"参数进行格式化
    decompression_options=_shared_docs["decompression_options"] % "path_or_buf",
# 定义 read_json 函数，用于将 JSON 字符串转换为 pandas 对象
def read_json(
    # path_or_buf 参数接受文件路径、路径对象或类文件对象（具有 read() 方法）
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    # orient 参数指定预期的 JSON 字符串格式，默认为 None
    *,
    orient: str | None = None,
    # typ 参数指定返回的对象类型，默认为 "frame"，可选 "frame" 或 "series"
    typ: Literal["frame", "series"] = "frame",
    # dtype 参数指定数据类型，默认为 None
    dtype: DtypeArg | None = None,
    # convert_axes 参数指定是否转换轴，默认为 None
    convert_axes: bool | None = None,
    # convert_dates 参数指定是否转换日期，默认为 True 或日期列表
    convert_dates: bool | list[str] = True,
    # keep_default_dates 参数指定是否保留默认日期，默认为 True
    keep_default_dates: bool = True,
    # precise_float 参数指定是否精确表示浮点数，默认为 False
    precise_float: bool = False,
    # date_unit 参数指定日期单位，默认为 None
    date_unit: str | None = None,
    # encoding 参数指定编码格式，默认为 None
    encoding: str | None = None,
    # encoding_errors 参数指定编码错误处理方式，默认为 "strict"
    encoding_errors: str | None = "strict",
    # lines 参数指定是否处理每行为 JSON，默认为 False
    lines: bool = False,
    # chunksize 参数指定块大小，默认为 None
    chunksize: int | None = None,
    # compression 参数指定压缩格式，默认为 "infer"
    compression: CompressionOptions = "infer",
    # nrows 参数指定读取的行数，默认为 None
    nrows: int | None = None,
    # storage_options 参数指定存储选项，默认为 None
    storage_options: StorageOptions | None = None,
    # dtype_backend 参数指定后端数据类型，默认为 lib.no_default
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    # engine 参数指定 JSON 解析引擎，默认为 "ujson"
    engine: JSONEngine = "ujson",
) -> DataFrame | Series | JsonReader:
    """
    Convert a JSON string to pandas object.

    Parameters
    ----------
    path_or_buf : a str path, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.json``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.

        .. deprecated:: 2.1.0
            Passing json literal strings is deprecated.

    orient : str, optional
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{{index -> [index], columns -> [columns], data -> [values]}}``
        - ``'records'`` : list like
          ``[{{column -> value}}, ... , {{column -> value}}]``
        - ``'index'`` : dict like ``{{index -> {{column -> value}}}}``
        - ``'columns'`` : dict like ``{{column -> {{index -> value}}}}``
        - ``'values'`` : just the values array
        - ``'table'`` : dict like ``{{'schema': {{schema}}, 'data': {{data}}}}``

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'``,

          - allowed orients are ``{{'split','records','index'}}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{{'split','records','index',
            'columns','values', 'table'}}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    Returns
    -------
    DataFrame or Series or JsonReader
        The type of object returned depends on the `typ` parameter.
    """
    # 设置数据恢复的目标类型，可以是 'frame' 或 'series'
    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.

    # 控制是否推断数据类型。如果是 True，将推断数据类型；如果是一个列到数据类型的字典，则使用这些数据类型；
    # 如果是 False，则不进行数据类型推断，仅适用于数据本身。
    # 对于除了 'table' 以外的所有 orient 值，默认值为 True。
    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.

        For all ``orient`` values except ``'table'``, default is True.

    # 尝试将轴转换为正确的数据类型。
    # 对于除了 'table' 以外的所有 orient 值，默认值为 True。
    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.

        For all ``orient`` values except ``'table'``, default is True.

    # 控制是否转换日期类型列。如果是 True，则默认的日期类列将被转换（取决于 keep_default_dates）。
    # 如果是 False，则不会转换日期。如果是列名的列表，则这些列将被转换，并且默认的日期类列也可能会被转换（取决于 keep_default_dates）。
    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on
        keep_default_dates).
        If False, no dates will be converted.
        If a list of column names, then those columns will be converted and
        default datelike columns may also be converted (depending on
        keep_default_dates).

    # 如果解析日期（convert_dates 不是 False），则尝试解析默认的日期类列。
    # 如果列标签是日期类的话，比如

    # * 以 ``'_at'`` 结尾，
    # * 以 ``'_time'`` 结尾，
    # * 以 ``'timestamp'`` 开头，
    # * 是 ``'modified'``，
    # * 是 ``'date'``。

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the
        default datelike columns.
        A column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``.

    # 设置是否使用更高精度的 strtod 函数来解码字符串为双精度值。
    # 默认（False）是使用快速但精度较低的内置功能。
    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.

    # 在转换日期时检测时间戳的单位。
    # 默认行为是尝试检测正确的精度，但如果不需要，则传入 's'、'ms'、'us' 或 'ns' 中的一个来强制解析秒、毫秒、微秒或纳秒。
    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    # 解码 py3 字节时使用的编码。
    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    # 编码错误如何处理的设置。查看 `可能的值列表 <https://docs.python.org/3/library/codecs.html#error-handlers>`_ 。
    encoding_errors : str, optional, default "strict"
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

        .. versionadded:: 1.3.0

    # 将文件作为每行一个 json 对象读取。
    lines : bool, default False
        Read the file as a json object per line.

    # 返回一个 JsonReader 对象以进行迭代。查看 `line-delimited json docs
    # <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
    # 获取关于 ``chunksize`` 的更多信息。
    # 只有在 `lines=True` 时才能传递这个参数。
    # 如果为 None，则文件将一次性全部读入内存。
    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `line-delimited json docs
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.

    # {decompression_options}
    # 
    # .. versionchanged:: 1.4.0 Zstandard support.

    # 从行分隔的 json 文件中读取的行数。
    # 只有在 `lines=True` 时才能传递这个参数。
    # 如果为 None，则返回所有行。
    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read.
        This can only be passed if `lines=True`.

    # {storage_options}
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        # 结果DataFrame使用的后端数据类型（仍处于实验阶段）。行为如下：
        #   - "numpy_nullable": 返回支持可空数据类型的DataFrame（默认）。
        #   - "pyarrow": 返回支持pyarrow的可空ArrowDtype的DataFrame。
        # 
        # .. versionadded:: 2.0

    engine : {'ujson', 'pyarrow'}, default 'ujson'
        # 使用的解析引擎。当`lines=True`时，只有"pyarrow"引擎可用。
        # 
        # .. versionadded:: 2.0

    Returns
    -------
    Series, DataFrame, or pandas.api.typing.JsonReader
        # 当`chunksize`不为`0`或`None`时返回JsonReader。
        # 否则，返回类型取决于`typ`的值。

    See Also
    --------
    DataFrame.to_json : 将DataFrame转换为JSON字符串。
    Series.to_json : 将Series转换为JSON字符串。
    json_normalize : 将半结构化JSON数据规范化为平面表格。

    Notes
    -----
    对于`orient='table'`，如果DataFrame具有名为`index`的文字索引名称，
    使用`to_json`写入后，后续的读取操作将错误地将索引名称设置为`None`。
    这是因为`index`在`DataFrame.to_json`中也用于表示缺少的索引名称，
    而后续的`read_json`操作无法区分这两种情况。对于`MultiIndex`和任何以
    `'level_'`开头的名称，也会遇到相同的限制。

    Examples
    --------
    >>> from io import StringIO
    >>> df = pd.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   index=['row 1', 'row 2'],
    ...                   columns=['col 1', 'col 2'])

    使用"split"格式化JSON进行编码/解码DataFrame：

    >>> df.to_json(orient='split')
        '\
"""
Encoding/decoding a DataFrame using ``'split'`` formatted JSON:

>>> pd.read_json(StringIO(_), orient='split')  # noqa: F821
      col 1 col 2
row 1     a     b
row 2     c     d

Encoding/decoding a DataFrame using ``'index'`` formatted JSON:

>>> df.to_json(orient='index')
'{{"row 1":{{"col 1":"a","col 2":"b"}},"row 2":{{"col 1":"c","col 2":"d"}}}}'

>>> pd.read_json(StringIO(_), orient='index')  # noqa: F821
      col 1 col 2
row 1     a     b
row 2     c     d

Encoding/decoding a DataFrame using ``'records'`` formatted JSON.
Note that index labels are not preserved with this encoding.

>>> df.to_json(orient='records')
'[{{"col 1":"a","col 2":"b"}},{{"col 1":"c","col 2":"d"}}]'
>>> pd.read_json(StringIO(_), orient='records')  # noqa: F821
  col 1 col 2
0     a     b
1     c     d

Encoding with Table Schema

>>> df.to_json(orient='table')
'\
{{"schema":{{"fields":[\
{{"name":"index","type":"string"}},\
{{"name":"col 1","type":"string"}},\
{{"name":"col 2","type":"string"}}],\
"primaryKey":["index"],\
"pandas_version":"1.4.0"}},\
"data":[\
{{"index":"row 1","col 1":"a","col 2":"b"}},\
{{"index":"row 2","col 1":"c","col 2":"d"}}]\
}}\
'

The following example uses ``dtype_backend="numpy_nullable"``

>>> data = '''{{"index": {{"0": 0, "1": 1}},
...        "a": {{"0": 1, "1": null}},
...        "b": {{"0": 2.5, "1": 4.5}},
...        "c": {{"0": true, "1": false}},
...        "d": {{"0": "a", "1": "b"}},
...        "e": {{"0": 1577.2, "1": 1577.1}}}}'''
>>> pd.read_json(StringIO(data), dtype_backend="numpy_nullable")
   index     a    b      c  d       e
0      0     1  2.5   True  a  1577.2
1      1  <NA>  4.5  False  b  1577.1
"""

if orient == "table" and dtype:
    raise ValueError("cannot pass both dtype and orient='table'")
if orient == "table" and convert_axes:
    raise ValueError("cannot pass both convert_axes and orient='table'")

check_dtype_backend(dtype_backend)

if dtype is None and orient != "table":
    # error: Incompatible types in assignment (expression has type "bool", variable
    # has type "Union[ExtensionDtype, str, dtype[Any], Type[str], Type[float],
    # Type[int], Type[complex], Type[bool], Type[object], Dict[Hashable,
    # Union[ExtensionDtype, Union[str, dtype[Any]], Type[str], Type[float],
    # Type[int], Type[complex], Type[bool], Type[object]]], None]")
    dtype = True  # type: ignore[assignment]
if convert_axes is None and orient != "table":
    convert_axes = True
"""
    # 创建一个 JsonReader 对象，用于读取和解析 JSON 数据
    json_reader = JsonReader(
        path_or_buf,               # JSON 文件路径或缓冲区
        orient=orient,             # 数据方向（如记录导向、列导向等）
        typ=typ,                   # 数据类型（如Series或DataFrame）
        dtype=dtype,               # 指定列的数据类型
        convert_axes=convert_axes, # 是否转换轴
        convert_dates=convert_dates, # 是否转换日期
        keep_default_dates=keep_default_dates, # 是否保留默认日期格式
        precise_float=precise_float, # 是否精确表示浮点数
        date_unit=date_unit,       # 日期单位
        encoding=encoding,         # 文件编码格式
        lines=lines,               # 是否按行读取
        chunksize=chunksize,       # 指定每个块的大小（行数）
        compression=compression,   # 压缩类型（如gzip、bz2等）
        nrows=nrows,               # 读取的行数限制
        storage_options=storage_options, # 存储选项（如S3的认证信息等）
        encoding_errors=encoding_errors, # 编码错误处理方式
        dtype_backend=dtype_backend, # 数据类型的后端
        engine=engine,             # JSON 解析引擎（如C、Python）
    )
    
    # 如果设置了 chunksize，则直接返回 JsonReader 对象，否则执行读取操作并返回结果
    if chunksize:
        return json_reader    # 返回 JsonReader 对象
    else:
        return json_reader.read()  # 读取 JSON 数据并返回
class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """

    def __init__(
        self,
        filepath_or_buffer,  # JSON 文件路径或缓冲区
        orient,              # JSON 文件的方向（例如 records, columns）
        typ: FrameSeriesStrT,  # JSON 数据的类型参数（泛型）
        dtype,               # 数据类型
        convert_axes: bool | None,  # 是否转换轴
        convert_dates,       # 是否转换日期
        keep_default_dates: bool,  # 是否保留默认日期
        precise_float: bool,  # 是否使用精确浮点数
        date_unit,           # 日期单位
        encoding,            # 编码格式
        lines: bool,         # 是否逐行读取
        chunksize: int | None,  # 分块大小，如果逐行读取为 True
        compression: CompressionOptions,  # 压缩选项（例如 gzip, bz2）
        nrows: int | None,   # 读取的行数限制
        storage_options: StorageOptions | None = None,  # 存储选项（例如 AWS S3）
        encoding_errors: str | None = "strict",  # 编码错误处理方式
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端
        engine: JSONEngine = "ujson",  # 使用的 JSON 引擎（例如 ujson, pandas）
        self.orient = orient
        # 设置对象的数据方向
        self.typ = typ
        # 设置对象的类型
        self.dtype = dtype
        # 设置对象的数据类型
        self.convert_axes = convert_axes
        # 设置是否转换轴
        self.convert_dates = convert_dates
        # 设置是否转换日期
        self.keep_default_dates = keep_default_dates
        # 设置是否保留默认日期
        self.precise_float = precise_float
        # 设置是否使用精确浮点数
        self.date_unit = date_unit
        # 设置日期单位
        self.encoding = encoding
        # 设置编码方式
        self.engine = engine
        # 设置数据处理引擎
        self.compression = compression
        # 设置压缩方式
        self.storage_options = storage_options
        # 设置存储选项
        self.lines = lines
        # 设置是否处理每行数据
        self.chunksize = chunksize
        # 设置每块数据的大小
        self.nrows_seen = 0
        # 初始化已处理行数为0
        self.nrows = nrows
        # 设置处理的行数限制
        self.encoding_errors = encoding_errors
        # 设置编码错误处理方式
        self.handles: IOHandles[str] | None = None
        # 初始化句柄为None，用于文件IO操作
        self.dtype_backend = dtype_backend
        # 设置数据类型后端处理方式

        # 如果使用的引擎不在支持列表中，抛出异常
        if self.engine not in {"pyarrow", "ujson"}:
            raise ValueError(
                f"The engine type {self.engine} is currently not supported."
            )
        
        # 如果设置了块大小，验证并确保大于0
        if self.chunksize is not None:
            self.chunksize = validate_integer("chunksize", self.chunksize, 1)
            # 如果不是按行处理，则抛出异常
            if not self.lines:
                raise ValueError("chunksize can only be passed if lines=True")
            # 如果使用了不支持块大小的引擎，抛出异常
            if self.engine == "pyarrow":
                raise ValueError(
                    "currently pyarrow engine doesn't support chunksize parameter"
                )
        
        # 如果设置了行数限制，验证并确保大于等于0
        if self.nrows is not None:
            self.nrows = validate_integer("nrows", self.nrows, 0)
            # 如果不是按行处理，则抛出异常
            if not self.lines:
                raise ValueError("nrows can only be passed if lines=True")
        
        # 如果使用了pyarrow引擎，但不是按行处理，则抛出异常
        if self.engine == "pyarrow":
            if not self.lines:
                raise ValueError(
                    "currently pyarrow engine only supports "
                    "the line-delimited JSON format"
                )
            # 用文件路径或缓冲区初始化数据
            self.data = filepath_or_buffer
        
        # 如果使用了ujson引擎，从文件路径或缓冲区获取数据
        elif self.engine == "ujson":
            data = self._get_data_from_filepath(filepath_or_buffer)
            # 如果没有设置块大小和行数限制，则将数据读入内存
            if not (self.chunksize or self.nrows):
                with self:
                    self.data = data.read()
            else:
                # 否则准备数据以供`__next__`方法使用
                self.data = data
    # 定义一个私有方法，用于从文件路径或缓冲区中获取数据
    def _get_data_from_filepath(self, filepath_or_buffer):
        """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
        """
        # 将输入的文件路径或缓冲区转换为字符串表示形式
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        try:
            # 获取处理后的文件句柄，以便后续读取数据
            self.handles = get_handle(
                filepath_or_buffer,
                "r",
                encoding=self.encoding,
                compression=self.compression,
                storage_options=self.storage_options,
                errors=self.encoding_errors,
            )
        except OSError as err:
            # 如果文件不存在，则抛出文件未找到的异常
            raise FileNotFoundError(
                f"File {filepath_or_buffer} does not exist"
            ) from err
        # 更新文件路径或缓冲区为最终处理后的句柄
        filepath_or_buffer = self.handles.handle
        # 返回最终处理后的文件路径或缓冲区句柄
        return filepath_or_buffer

    # 定义一个私有方法，将多行的 JSON 对象合并为一个 JSON 对象字符串
    def _combine_lines(self, lines) -> str:
        """
        Combines a list of JSON objects into one JSON object.
        """
        # 将每行 JSON 对象字符串进行清洗和合并成一个有效的 JSON 数组字符串
        return (
            f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'
        )

    # 以下为 read 方法的类型重载声明，返回值可能为 DataFrame、Series 或它们的组合
    # 定义一个方法 read，用于从 JSON 输入中读取数据并转换为 pandas 对象（DataFrame 或 Series）
    def read(self) -> DataFrame | Series:
        """
        Read the whole JSON input into a pandas object.
        """
        obj: DataFrame | Series  # 声明一个变量 obj，类型可以是 DataFrame 或 Series

        # 使用上下文管理器 self 打开资源
        with self:
            # 根据引擎类型选择不同的处理方式
            if self.engine == "pyarrow":
                # 导入可选依赖库 pyarrow.json，并使用其读取 JSON 数据
                pyarrow_json = import_optional_dependency("pyarrow.json")
                pa_table = pyarrow_json.read_json(self.data)

                mapping: type[ArrowDtype] | None | Callable
                # 根据 dtype_backend 设置映射类型
                if self.dtype_backend == "pyarrow":
                    mapping = ArrowDtype
                elif self.dtype_backend == "numpy_nullable":
                    # 从 pandas.io._util 导入 _arrow_dtype_mapping 函数，并获取映射
                    from pandas.io._util import _arrow_dtype_mapping
                    mapping = _arrow_dtype_mapping().get
                else:
                    mapping = None

                # 将 pyarrow 表格转换为 pandas 对象，使用指定的类型映射
                return pa_table.to_pandas(types_mapper=mapping)
            elif self.engine == "ujson":
                if self.lines:
                    if self.chunksize:
                        # 如果存在 chunksize，则使用 concat 方法合并数据
                        obj = concat(self)
                    elif self.nrows:
                        # 如果指定了 nrows，则从数据中获取指定行数，并解析为对象
                        lines = list(islice(self.data, self.nrows))
                        lines_json = self._combine_lines(lines)
                        obj = self._get_object_parser(lines_json)
                    else:
                        # 否则，将数据转换为字符串，并按行拆分后进行解析
                        data = ensure_str(self.data)
                        data_lines = data.split("\n")
                        obj = self._get_object_parser(self._combine_lines(data_lines))
                else:
                    # 如果没有明确指定 lines，则直接解析数据为对象
                    obj = self._get_object_parser(self.data)
                
                # 如果存在 dtype_backend 设置，则进行数据类型转换
                if self.dtype_backend is not lib.no_default:
                    return obj.convert_dtypes(
                        infer_objects=False, dtype_backend=self.dtype_backend
                    )
                else:
                    return obj

    # 定义一个方法 _get_object_parser，用于将 JSON 文档解析为 pandas 对象（DataFrame 或 Series）
    def _get_object_parser(self, json) -> DataFrame | Series:
        """
        Parses a json document into a pandas object.
        """
        typ = self.typ  # 获取对象的类型信息
        dtype = self.dtype  # 获取数据类型信息
        kwargs = {
            "orient": self.orient,
            "dtype": self.dtype,
            "convert_axes": self.convert_axes,
            "convert_dates": self.convert_dates,
            "keep_default_dates": self.keep_default_dates,
            "precise_float": self.precise_float,
            "date_unit": self.date_unit,
            "dtype_backend": self.dtype_backend,
        }

        obj = None  # 初始化对象为 None

        # 根据对象的类型选择不同的解析器进行解析
        if typ == "frame":
            obj = FrameParser(json, **kwargs).parse()  # 解析为 DataFrame 对象
        elif typ == "series" or obj is None:
            if not isinstance(dtype, bool):
                kwargs["dtype"] = dtype
            obj = SeriesParser(json, **kwargs).parse()  # 解析为 Series 对象

        return obj  # 返回解析后的对象

    # 定义一个方法 close，用于关闭可能打开的资源
    def close(self) -> None:
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
        if self.handles is not None:
            self.handles.close()  # 关闭资源的句柄

    # 实现迭代器接口，返回自身对象
    def __iter__(self) -> Self:
        return self
    # 使用类型注解定义函数签名重载，返回值为DataFrame
    @overload
    def __next__(self: JsonReader[Literal["frame"]]) -> DataFrame: ...

    # 使用类型注解定义函数签名重载，返回值为Series
    @overload
    def __next__(self: JsonReader[Literal["series"]]) -> Series: ...

    # 使用类型注解定义函数签名重载，返回值为DataFrame或Series
    @overload
    def __next__(
        self: JsonReader[Literal["frame", "series"]],
    ) -> DataFrame | Series: ...

    # 实现迭代器的核心方法，返回值为DataFrame或Series
    def __next__(self) -> DataFrame | Series:
        # 如果指定了读取的行数并且已经超过了设定的行数，则关闭资源并抛出迭代结束的异常
        if self.nrows and self.nrows_seen >= self.nrows:
            self.close()
            raise StopIteration

        # 从数据源中获取指定数量的行数据
        lines = list(islice(self.data, self.chunksize))
        # 如果没有获取到行数据，则关闭资源并抛出迭代结束的异常
        if not lines:
            self.close()
            raise StopIteration

        try:
            # 将获取的行数据组合成完整的 JSON 对象
            lines_json = self._combine_lines(lines)
            # 解析 JSON 对象得到数据对象
            obj = self._get_object_parser(lines_json)

            # 确保返回的对象具有正确的索引
            obj.index = range(self.nrows_seen, self.nrows_seen + len(obj))
            self.nrows_seen += len(obj)
        except Exception as ex:
            # 如果出现异常，则关闭资源并抛出该异常
            self.close()
            raise ex

        # 根据后端数据类型设置，对返回的对象进行数据类型转换
        if self.dtype_backend is not lib.no_default:
            return obj.convert_dtypes(
                infer_objects=False, dtype_backend=self.dtype_backend
            )
        else:
            return obj

    # 实现上下文管理器的 __enter__ 方法，返回自身实例
    def __enter__(self) -> Self:
        return self

    # 实现上下文管理器的 __exit__ 方法，用于处理异常时的资源释放
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
class Parser:
    _split_keys: tuple[str, ...]  # 元组类型的私有类变量，存储了预定义的键名集合
    _default_orient: str  # 字符串类型的私有类变量，存储了默认的方向值

    _STAMP_UNITS = ("s", "ms", "us", "ns")  # 元组类型的类变量，存储了时间戳单位
    _MIN_STAMPS = {
        "s": 31536000,
        "ms": 31536000000,
        "us": 31536000000000,
        "ns": 31536000000000000,
    }  # 字典类型的类变量，存储了最小的时间戳数值

    json: str  # 字符串类型的类变量，存储了 JSON 数据

    def __init__(
        self,
        json: str,
        orient,  # 参数：JSON 字符串和数据方向
        dtype: DtypeArg | None = None,  # 可选参数：数据类型
        convert_axes: bool = True,  # 布尔类型的可选参数：是否转换轴
        convert_dates: bool | list[str] = True,  # 可选参数：是否转换日期
        keep_default_dates: bool = False,  # 布尔类型的可选参数：是否保留默认日期
        precise_float: bool = False,  # 布尔类型的可选参数：是否精确浮点数
        date_unit=None,  # 可选参数：日期单位
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 可选参数：数据类型后端
    ) -> None:
        self.json = json  # 初始化 JSON 数据

        if orient is None:
            orient = self._default_orient  # 如果方向未指定，则使用默认方向

        self.orient = orient  # 初始化方向

        self.dtype = dtype  # 初始化数据类型

        if date_unit is not None:
            date_unit = date_unit.lower()  # 将日期单位转换为小写
            if date_unit not in self._STAMP_UNITS:
                raise ValueError(f"date_unit must be one of {self._STAMP_UNITS}")  # 如果日期单位不在预定义的单位中，则引发值错误异常
            self.min_stamp = self._MIN_STAMPS[date_unit]  # 根据日期单位设置最小时间戳数值
        else:
            self.min_stamp = self._MIN_STAMPS["s"]  # 默认使用秒作为最小时间戳数值

        self.precise_float = precise_float  # 初始化是否精确浮点数
        self.convert_axes = convert_axes  # 初始化是否转换轴
        self.convert_dates = convert_dates  # 初始化是否转换日期
        self.date_unit = date_unit  # 初始化日期单位
        self.keep_default_dates = keep_default_dates  # 初始化是否保留默认日期
        self.obj: DataFrame | Series | None = None  # 初始化对象为空
        self.dtype_backend = dtype_backend  # 初始化数据类型后端

    @final
    def check_keys_split(self, decoded: dict) -> None:
        """
        Checks that dict has only the appropriate keys for orient='split'.
        检查字典是否仅包含适用于 orient='split' 的合适键名。
        """
        bad_keys = set(decoded.keys()).difference(set(self._split_keys))  # 获取不适合的键名集合
        if bad_keys:
            bad_keys_joined = ", ".join(bad_keys)
            raise ValueError(f"JSON data had unexpected key(s): {bad_keys_joined}")  # 如果存在不适合的键名，引发值错误异常

    @final
    def parse(self):
        """
        Parse the JSON data.
        解析 JSON 数据。
        """
        self._parse()  # 调用内部方法进行解析

        if self.obj is None:
            return None
        if self.convert_axes:
            self._convert_axes()  # 如果需要转换轴，调用内部方法进行转换
        self._try_convert_types()  # 尝试转换数据类型
        return self.obj  # 返回解析后的对象

    def _parse(self) -> None:
        """
        Placeholder method for parsing JSON data.
        解析 JSON 数据的占位方法。
        """
        raise AbstractMethodError(self)  # 抽象方法，由子类实现具体的解析逻辑

    @final
    def _convert_axes(self) -> None:
        """
        Try to convert axes.
        尝试转换轴。
        """
        obj = self.obj
        assert obj is not None  # 断言确保对象不为空，用于类型检查
        for axis_name in obj._AXIS_ORDERS:
            ax = obj._get_axis(axis_name)  # 获取轴对象
            ser = Series(ax, dtype=ax.dtype, copy=False)  # 创建系列对象
            new_ser, result = self._try_convert_data(
                name=axis_name,
                data=ser,
                use_dtypes=False,
                convert_dates=True,
                is_axis=True,
            )  # 尝试转换数据类型
            if result:
                new_axis = Index(new_ser, dtype=new_ser.dtype, copy=False)  # 创建新的索引对象
                setattr(self.obj, axis_name, new_axis)  # 设置对象的新轴属性

    def _try_convert_types(self) -> None:
        """
        Placeholder method for trying to convert types.
        尝试转换数据类型的占位方法。
        """
        raise AbstractMethodError(self)  # 抽象方法，由子类实现具体的数据类型转换逻辑
    def _try_convert_data(
        self,
        name: Hashable,
        data: Series,
        use_dtypes: bool = True,
        convert_dates: bool | list[str] = True,
        is_axis: bool = False,
    ) -> tuple[Series, bool]:
        """
        Try to parse a Series into a column by inferring dtype.
        """
        # 如果 use_dtypes 为真，则尝试按照指定的数据类型进行转换
        if use_dtypes:
            # 如果 self.dtype 未定义
            if not self.dtype:
                # 如果数据中没有空值，则直接返回原始数据和 False
                if all(notna(data)):
                    return data, False

                # 填充空值为 NaN
                filled = data.fillna(np.nan)
                return filled, True

            elif self.dtype is True:
                pass
            else:
                # 如果指定了具体的数据类型
                # 如果 self.dtype 是字典，则从中获取相应列名对应的数据类型
                dtype = (
                    self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype
                )
                if dtype is not None:
                    # 尝试将数据转换为指定的数据类型
                    try:
                        return data.astype(dtype), True
                    except (TypeError, ValueError):
                        return data, False

        # 如果 convert_dates 为真，则尝试将数据转换为日期类型
        if convert_dates:
            new_data, result = self._try_convert_to_date(data)
            if result:
                return new_data, True

        converted = False
        # 如果 self.dtype_backend 存在且不是轴（axis）数据，则继续转换
        if self.dtype_backend is not lib.no_default and not is_axis:
            # 转换暂时不进行，后续处理
            return data, True
        elif is_string_dtype(data.dtype):
            # 尝试将字符串数据转换为浮点数
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        # 如果数据类型是浮点数且不是 float64，则将其转换为 float64
        if data.dtype.kind == "f" and data.dtype != "float64":
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        # 如果数据长度不为零且数据类型是 float 或 object
        if len(data) and data.dtype in ("float", "object"):
            # 尝试将数据转换为 int64（如果可能）
            try:
                new_data = data.astype("int64")
                if (new_data == data).all():
                    data = new_data
                    converted = True
            except (TypeError, ValueError, OverflowError):
                pass

        # 如果数据类型是 int 且不是 int64，则将其转换为 int64
        if data.dtype == "int" and data.dtype != "int64":
            try:
                data = data.astype("int64")
                converted = True
            except (TypeError, ValueError):
                pass

        # 如果是索引，并且数据长度不为零，根据 orient 参数来处理数据类型
        if name == "index" and len(data):
            if self.orient == "split":
                return data, False

        return data, converted

    @final
    def _try_convert_to_date(self, data: Series) -> tuple[Series, bool]:
        """
        Try to parse a ndarray like into a date column.

        Try to coerce object in epoch/iso formats and integer/float in epoch
        formats. Return a boolean if parsing was successful.
        """
        # 如果数据为空，则直接返回原数据和 False
        if not len(data):
            return data, False

        new_data = data  # 将传入的数据赋值给新变量 new_data

        # 如果数据类型是字符串，则将其转换为对象类型
        if new_data.dtype == "string":
            new_data = new_data.astype(object)

        # 如果数据类型是对象类型
        if new_data.dtype == "object":
            try:
                new_data = data.astype("int64")  # 尝试将数据转换为 int64 类型
            except OverflowError:
                return data, False  # 如果转换过程中发生溢出错误，则返回原数据和 False
            except (TypeError, ValueError):
                pass  # 捕获类型错误或数值错误，继续执行后续操作

        # 对于数值类型的数据，检查是否在范围内
        if issubclass(new_data.dtype.type, np.number):
            in_range = (
                isna(new_data._values)  # 检查数据是否为缺失值
                | (new_data > self.min_stamp)  # 检查数据是否大于最小时间戳
                | (new_data._values == iNaT)  # 检查数据是否为 NaT（时间戳的缺失值）
            )
            if not in_range.all():
                return data, False  # 如果不是所有数据都在指定范围内，则返回原数据和 False

        # 确定日期单位，默认为类中指定的单位或时间戳的默认单位
        date_units = (self.date_unit,) if self.date_unit else self._STAMP_UNITS
        for date_unit in date_units:
            try:
                new_data = to_datetime(new_data, errors="raise", unit=date_unit)
                # 尝试将数据转换为日期时间格式，如果成功则返回转换后的数据和 True
            except (ValueError, OverflowError, TypeError):
                continue  # 捕获可能的错误类型并继续尝试下一个日期单位
            return new_data, True  # 如果成功转换，则返回转换后的数据和 True
        return data, False  # 如果所有日期单位尝试都失败，则返回原数据和 False
class SeriesParser(Parser):
    _default_orient = "index"
    _split_keys = ("name", "index", "data")
    obj: Series | None

    def _parse(self) -> None:
        # 解析 JSON 数据为 Python 字典
        data = ujson_loads(self.json, precise_float=self.precise_float)

        # 根据 orient 参数决定解析方式
        if self.orient == "split":
            # 将字典键转换为字符串类型
            decoded = {str(k): v for k, v in data.items()}
            # 检查解析后的键是否满足预期，调用相应的方法
            self.check_keys_split(decoded)
            # 创建 Series 对象，并传入解析后的数据
            self.obj = Series(**decoded)
        else:
            # 创建 Series 对象，直接传入解析后的数据
            self.obj = Series(data)

    def _try_convert_types(self) -> None:
        # 如果 self.obj 为 None，直接返回
        if self.obj is None:
            return
        # 尝试将数据转换为指定类型，包括日期转换
        obj, result = self._try_convert_data(
            "data", self.obj, convert_dates=self.convert_dates
        )
        # 如果成功转换，更新 self.obj
        if result:
            self.obj = obj


class FrameParser(Parser):
    _default_orient = "columns"
    _split_keys = ("columns", "index", "data")
    obj: DataFrame | None

    def _parse(self) -> None:
        # 获取 JSON 数据和 orient 参数
        json = self.json
        orient = self.orient

        # 根据 orient 参数决定 DataFrame 的创建方式
        if orient == "columns":
            # 根据列向导入数据创建 DataFrame 对象
            self.obj = DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )
        elif orient == "split":
            # 将解析后的字典键转换为字符串类型
            decoded = {
                str(k): v
                for k, v in ujson_loads(json, precise_float=self.precise_float).items()
            }
            # 检查解析后的键是否符合预期，调整列名格式
            self.check_keys_split(decoded)
            # 处理原始列名，确保唯一性
            orig_names = [
                (tuple(col) if isinstance(col, list) else col)
                for col in decoded["columns"]
            ]
            decoded["columns"] = dedup_names(
                orig_names,
                is_potential_multi_index(orig_names, None),
            )
            # 使用解析后的数据创建 DataFrame 对象
            self.obj = DataFrame(dtype=None, **decoded)
        elif orient == "index":
            # 使用 orient="index" 参数将解析后的字典数据导入 DataFrame
            self.obj = DataFrame.from_dict(
                ujson_loads(json, precise_float=self.precise_float),
                dtype=None,
                orient="index",
            )
        elif orient == "table":
            # 解析表格模式的 JSON 数据为 DataFrame 对象
            self.obj = parse_table_schema(json, precise_float=self.precise_float)
        else:
            # 默认情况下，直接导入 JSON 数据创建 DataFrame
            self.obj = DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )

    def _process_converter(
        self,
        f: Callable[[Hashable, Series], tuple[Series, bool]],
        filt: Callable[[Hashable], bool] | None = None,
    ) -> None:
        """
        Take a conversion function and possibly recreate the frame.
        """
        # 如果未提供过滤函数，则默认为True的lambda函数
        if filt is None:
            filt = lambda col: True

        # 获取self对象的引用
        obj = self.obj
        # 确保obj不为None，用于类型检查（例如mypy）
        assert obj is not None  # for mypy

        # 初始化标记变量，用于判断是否需要创建新的对象
        needs_new_obj = False
        # 初始化一个新的空字典，用于存储处理后的列数据
        new_obj = {}
        # 遍历self.obj.items()的结果，即DataFrame的每一列及其对应的数据
        for i, (col, c) in enumerate(obj.items()):
            # 根据过滤函数判断是否对当前列进行处理
            if filt(col):
                # 调用转换函数f对当前列c进行处理，返回处理后的数据new_data和处理结果result
                new_data, result = f(col, c)
                # 如果处理结果为True，则更新列c为处理后的数据new_data，并标记需要创建新对象
                if result:
                    c = new_data
                    needs_new_obj = True
            # 将处理后的列数据c存入新的对象new_obj中，以索引i作为键
            new_obj[i] = c

        # 如果需要创建新的对象
        if needs_new_obj:
            # 创建一个新的DataFrame对象new_frame，使用new_obj作为数据，obj.index作为索引
            new_frame = DataFrame(new_obj, index=obj.index)
            # 将原DataFrame的列名称赋给新DataFrame
            new_frame.columns = obj.columns
            # 更新self.obj为新创建的DataFrame对象new_frame
            self.obj = new_frame

    def _try_convert_types(self) -> None:
        # 如果self.obj为None，则直接返回
        if self.obj is None:
            return
        # 如果需要转换日期类型，则调用_try_convert_dates方法
        if self.convert_dates:
            self._try_convert_dates()

        # 调用_process_converter方法，传入lambda函数，尝试对数据进行类型转换
        self._process_converter(
            lambda col, c: self._try_convert_data(col, c, convert_dates=False)
        )

    def _try_convert_dates(self) -> None:
        # 如果self.obj为None，则直接返回
        if self.obj is None:
            return

        # 初始化要转换的日期列列表
        convert_dates_list_bool = self.convert_dates
        # 如果convert_dates_list_bool是布尔类型，则初始化为空列表
        if isinstance(convert_dates_list_bool, bool):
            convert_dates_list_bool = []
        # 转换成集合类型，存储需要转换为日期的列名
        convert_dates = set(convert_dates_list_bool)

        def is_ok(col) -> bool:
            """
            Return if this col is ok to try for a date parse.
            """
            # 如果列名col在convert_dates集合中，则返回True
            if col in convert_dates:
                return True
            # 如果不保留默认日期类型，且列名不是字符串类型，则返回False
            if not self.keep_default_dates:
                return False
            # 如果列名是字符串类型
            if not isinstance(col, str):
                return False

            # 将列名转换为小写
            col_lower = col.lower()
            # 判断列名是否以指定字符串结尾或者与指定字符串匹配，如果是则返回True
            if (
                col_lower.endswith(("_at", "_time"))
                or col_lower == "modified"
                or col_lower == "date"
                or col_lower == "datetime"
                or col_lower.startswith("timestamp")
            ):
                return True
            # 否则返回False
            return False

        # 调用_process_converter方法，传入is_ok函数作为过滤条件，尝试对数据进行日期转换
        self._process_converter(lambda col, c: self._try_convert_to_date(c), filt=is_ok)
```