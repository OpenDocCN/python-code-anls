# `D:\src\scipysrc\pandas\pandas\io\common.py`

```
"""Common IO api utilities"""

from __future__ import annotations  # 允许在类型注释中使用字符串形式的类型

from abc import (  # 导入抽象基类模块中的相关功能
    ABC,  # Python中的抽象基类
    abstractmethod,  # 定义抽象方法的装饰器
)
import codecs  # 提供编解码器注册和处理的接口函数
from collections import defaultdict  # 创建默认字典的类
from collections.abc import (  # 导入集合抽象基类模块中的相关功能
    Hashable,  # 可散列对象的抽象基类
    Mapping,  # 映射类型的抽象基类
    Sequence,  # 序列类型的抽象基类
)
import dataclasses  # 提供装饰器和函数，用于创建和操作数据类
import functools  # 提供用于高阶函数操作的工具，如缓存和部分函数
import gzip  # 实现gzip文件压缩和解压缩的模块
from io import (  # 导入输入输出操作的抽象基类和工具
    BufferedIOBase,  # 带缓冲的IO操作的基类
    BytesIO,  # 在内存中操作二进制数据的类
    RawIOBase,  # 未经过缓冲的原始IO操作的基类
    StringIO,  # 在内存中操作字符串数据的类
    TextIOBase,  # 文本IO操作的基类
    TextIOWrapper,  # 将字节流封装为文本IO对象的类
)
import mmap  # 内存映射文件的支持
import os  # 提供与操作系统交互的功能
from pathlib import Path  # 提供处理路径的面向对象的API
import re  # 提供正则表达式操作的支持
import tarfile  # 提供操作tar文件的工具
from typing import (  # 导入类型提示相关的工具
    IO,  # 定义输入输出流对象的抽象基类
    TYPE_CHECKING,  # 类型检查时用于避免循环导入的常量
    Any,  # 表示可以是任何类型的对象
    AnyStr,  # 表示可以是任何字符串类型的对象
    DefaultDict,  # 默认字典类型的泛型
    Generic,  # 泛型类型的基类
    Literal,  # 字面值类型的泛型
    TypeVar,  # 类型变量的泛型
    cast,  # 用于类型强制转换的函数
    overload,  # 定义函数的重载
)
from urllib.parse import (  # 导入URL解析相关的功能
    urljoin,  # 合并URL的函数
    urlparse as parse_url,  # 解析URL的函数，并重命名为parse_url
    uses_netloc,  # URL解析时需要使用网络位置的常量列表
    uses_params,  # URL解析时需要使用参数的常量列表
    uses_relative,  # URL解析时需要使用相对路径的常量列表
)
import warnings  # 用于管理警告的模块
import zipfile  # 提供操作zip文件的工具

from pandas._typing import (  # 导入Pandas中的类型提示相关功能
    BaseBuffer,  # 基本缓冲区类型的类型变量
    ReadCsvBuffer,  # 读取CSV缓冲区的类型变量
)
from pandas.compat._optional import import_optional_dependency  # 导入可选依赖项的导入功能
from pandas.util._decorators import doc  # 导入用于文档装饰的装饰器
from pandas.util._exceptions import find_stack_level  # 导入用于查找堆栈级别的异常处理函数

from pandas.core.dtypes.common import (  # 导入Pandas中通用数据类型相关的功能
    is_bool,  # 判断对象是否为布尔类型的函数
    is_file_like,  # 判断对象是否类似文件的函数
    is_integer,  # 判断对象是否为整数类型的函数
    is_list_like,  # 判断对象是否类似列表的函数
)
from pandas.core.dtypes.generic import ABCMultiIndex  # 导入Pandas中多级索引抽象基类

from pandas.core.shared_docs import _shared_docs  # 导入Pandas中共享文档相关功能

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)  # 合法的URL集合，排除空字符串
_RFC_3986_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+\-+.]*://")  # 匹配RFC 3986规范的正则表达式模式

BaseBufferT = TypeVar("BaseBufferT", bound=BaseBuffer)  # 定义泛型类型BaseBufferT，限制为BaseBuffer的子类


if TYPE_CHECKING:
    from types import TracebackType  # 导入用于追踪异常回溯的类型

    from pandas._typing import (  # 导入Pandas中的类型提示相关功能
        CompressionDict,  # 压缩字典的类型变量
        CompressionOptions,  # 压缩选项的类型变量
        FilePath,  # 文件路径的类型变量
        ReadBuffer,  # 读取缓冲区的类型变量
        StorageOptions,  # 存储选项的类型变量
        WriteBuffer,  # 写入缓冲区的类型变量
    )

    from pandas import MultiIndex  # 导入Pandas中的多级索引类型


@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.

    Represents the arguments for IO operations.
    """

    filepath_or_buffer: str | BaseBuffer  # 文件路径或缓冲区对象
    encoding: str  # 编码方式
    mode: str  # 打开文件的模式
    compression: CompressionDict  # 压缩字典
    should_close: bool = False  # 是否应该关闭


@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    # handle might not implement the IO-interface
    handle: IO[AnyStr]  # IO操作的文件句柄
    compression: CompressionDict  # 压缩字典
    created_handles: list[IO[bytes] | IO[str]] = dataclasses.field(default_factory=list)  # 创建的所有文件句柄列表，默认为空
    is_wrapped: bool = False  # 是否需要分离TextIOWrapper
    # 关闭所有已创建的文件句柄。

    # 如果包装了 TextIOWrapper，需要先刷新并分离，以避免关闭可能是用户创建的缓冲区。
    if self.is_wrapped:
        assert isinstance(self.handle, TextIOWrapper)
        self.handle.flush()  # 刷新文本包装器
        self.handle.detach()  # 分离文本包装器
        self.created_handles.remove(self.handle)  # 从已创建句柄列表中移除该包装器

    # 关闭所有已创建的文件句柄
    for handle in self.created_handles:
        handle.close()

    # 清空已创建句柄列表
    self.created_handles = []

    # 将包装标志设为 False，表示没有包装的文件句柄
    self.is_wrapped = False


    # 实现上下文管理器的 __exit__ 方法，用于处理退出时的清理工作。
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # 调用 close 方法关闭所有已创建的文件句柄
        self.close()
def is_url(url: object) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode
        The URL string to be validated.

    Returns
    -------
    isurl : bool
        True if `url` has a valid protocol, otherwise False.
    """
    # Check if `url` is a string
    if not isinstance(url, str):
        return False
    # Parse the URL and check if its scheme (protocol) is in _VALID_URLS
    return parse_url(url).scheme in _VALID_URLS


@overload
def _expand_user(filepath_or_buffer: str) -> str: ...


@overload
def _expand_user(filepath_or_buffer: BaseBufferT) -> BaseBufferT: ...


def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT:
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : str or object
        File path or buffer object to be expanded.

    Returns
    -------
    expanded_filepath_or_buffer : str or object
        An expanded file path or the original object if not expandable.
    """
    # Check if filepath_or_buffer is a string, then expand user's home directory
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    # If filepath_or_buffer is not a string, return it unchanged
    return filepath_or_buffer


def validate_header_arg(header: object) -> None:
    """
    Validate the header argument for DataFrame operations.

    Parameters
    ----------
    header : object
        The header argument to be validated.

    Raises
    ------
    ValueError
        If header is invalid based on specified conditions.
    TypeError
        If header is of an invalid type.

    Notes
    -----
    This function validates header against known rules for DataFrame headers.
    """
    # If header is None, return immediately
    if header is None:
        return
    # If header is an integer, ensure it's non-negative
    if is_integer(header):
        header = cast(int, header)
        if header < 0:
            # Raise ValueError for negative integer header
            raise ValueError(
                "Passing negative integer to header is invalid. "
                "For no header, use header=None instead"
            )
        return
    # If header is a list-like object, validate each element
    if is_list_like(header, allow_sets=False):
        header = cast(Sequence, header)
        # Check if all elements are integers
        if not all(map(is_integer, header)):
            raise ValueError("header must be integer or list of integers")
        # Check if any element is negative
        if any(i < 0 for i in header):
            raise ValueError("cannot specify multi-index header with negative integers")
        return
    # If header is a boolean, raise TypeError
    if is_bool(header):
        raise TypeError(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify "
            "the row(s) making up the column names"
        )
    # If header is of an unexpected type, raise ValueError
    raise ValueError("header must be integer or list of integers")


@overload
def stringify_path(
    filepath_or_buffer: FilePath, convert_file_like: bool = ...
) -> str: ...


@overload
def stringify_path(
    filepath_or_buffer: BaseBufferT, convert_file_like: bool = ...
) -> BaseBufferT: ...


def stringify_path(
    filepath_or_buffer: FilePath | BaseBufferT,
    convert_file_like: bool = False,
) -> str | BaseBufferT:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : FilePath or BaseBufferT
        Object to be converted to a string representation.

    convert_file_like : bool, optional
        Whether to convert file-like objects, by default False.

    Returns
    -------
    str_filepath_or_buffer : str or BaseBufferT
        String representation of the object if converted successfully,
        otherwise returns the original object.

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not path-like.
    """
    # Return the string representation of filepath_or_buffer if possible
    # If convert_file_like is False, only convert FilePath objects
    # If convert_file_like is True, convert both FilePath and BaseBufferT
    pass  # Placeholder comment: Actual implementation not provided in the snippet
    """
    如果 convert_file_like 为假且 filepath_or_buffer 是类文件对象（file-like）：
        # GH 38125：某些 fsspec 对象实现了 os.PathLike 接口，但已经打开了一个文件。
        # 这样可以防止再次打开文件。infer_compression 调用此函数并传入 convert_file_like=True 来推断压缩方式。
        返回类型转换后的 BaseBufferT 对象，即 filepath_or_buffer 的强制类型转换结果

    如果 filepath_or_buffer 是 os.PathLike 的实例：
        filepath_or_buffer 被转换成其路径字符串表示 __fspath__()

    返回 _expand_user 处理后的 filepath_or_buffer
    """
def urlopen(*args: Any, **kwargs: Any) -> Any:
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request  # 导入标准库中的 urllib.request 模块

    return urllib.request.urlopen(*args, **kwargs)  # 调用 urllib.request.urlopen 函数并返回结果


def is_fsspec_url(url: FilePath | BaseBuffer) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    return (
        isinstance(url, str)  # 检查 url 是否为字符串类型
        and bool(_RFC_3986_PATTERN.match(url))  # 检查 url 是否符合 RFC 3986 规范
        and not url.startswith(("http://", "https://"))  # 检查 url 是否不以 "http://" 或 "https://" 开头
    )


@doc(
    storage_options=_shared_docs["storage_options"],  # 使用共享文档中的 storage_options 来注释
    compression_options=_shared_docs["compression_options"] % "filepath_or_buffer",  # 使用共享文档中的 compression_options 来注释
)
def _get_filepath_or_buffer(
    filepath_or_buffer: FilePath | BaseBuffer,
    encoding: str = "utf-8",
    compression: CompressionOptions | None = None,
    mode: str = "r",
    storage_options: StorageOptions | None = None,
) -> IOArgs:
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str or pathlib.Path),
                         or buffer
                         如果 filepath_or_buffer 是一个 URL，则翻译并返回缓冲区；否则直接通过。

    {compression_options}
        .. versionchanged:: 1.4.0 Zstandard support.
        .. 版本更改：1.4.0 支持 Zstandard。

    encoding : the encoding to use to decode bytes, default is 'utf-8'
               用于解码字节的编码方式，默认为 'utf-8'
    mode : str, optional
           可选的字符串模式

    {storage_options}
        使用共享文档中的 storage_options 来注释

    Returns the dataclass IOArgs.
    返回数据类 IOArgs。
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)  # 转换 filepath_or_buffer 为字符串路径

    # 处理压缩字典
    compression_method, compression = get_compression_method(compression)  # 获取压缩方法和选项
    compression_method = infer_compression(filepath_or_buffer, compression_method)  # 推断压缩方法

    # GH21227 internal compression is not used for non-binary handles.
    if compression_method and hasattr(filepath_or_buffer, "write") and "b" not in mode:
        warnings.warn(
            "compression has no effect when passing a non-binary object as input.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )
        compression_method = None  # 如果压缩方法存在但是传入的对象不是二进制模式，发出警告并设置 compression_method 为 None

    compression = dict(compression, method=compression_method)  # 将压缩方法添加到压缩选项中

    # bz2 and xz do not write the byte order mark for utf-16 and utf-32
    # print a warning when writing such files
    if (
        "w" in mode
        and compression_method in ["bz2", "xz"]
        and encoding in ["utf-16", "utf-32"]
    ):
        warnings.warn(
            f"{compression} will not write the byte order mark for {encoding}",
            UnicodeWarning,
            stacklevel=find_stack_level(),
        )  # 如果是写入模式并且是 bz2 或 xz 压缩，且编码是 utf-16 或 utf-32，发出警告

    if "a" in mode and compression_method in ["zip", "tar"]:
        # GH56778
        warnings.warn(
            "zip and tar do not support mode 'a' properly. "
            "This combination will result in multiple files with same name "
            "being added to the archive.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )  # 如果是追加模式且是 zip 或 tar 压缩，发出警告

    # Use binary mode when converting path-like objects to file-like objects (fsspec)
    # 当将类似路径的对象转换为类似文件的对象时，使用二进制模式（fsspec）
    # 将 fsspec_mode 设定为与 mode 相同的初始值
    fsspec_mode = mode
    # 如果 fsspec_mode 中既不包含 "t" 也不包含 "b"
    if "t" not in fsspec_mode and "b" not in fsspec_mode:
        # 向 fsspec_mode 添加 "b"，表示二进制模式
        fsspec_mode += "b"

    # 如果 filepath_or_buffer 是字符串且是 URL 地址
    if isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer):
        # TODO: fsspec 也可以通过 requests 处理 HTTP，但暂时保持不变。
        # 使用 fsspec 似乎会破坏推断服务器是否响应 gzip 数据的能力。

        # 如果 storage_options 未提供，则设为空字典
        storage_options = storage_options or {}

        # 等待直到这里导入，以匹配在本模块的其他地方定义的 urlopen 函数的懒加载逻辑
        import urllib.request

        # 假设 storage_options 应被解释为请求头
        # 创建用于请求的 urllib.request.Request 对象
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        
        # 使用 urlopen 打开请求
        with urlopen(req_info) as req:
            # 获取响应的 Content-Encoding 头部信息
            content_encoding = req.headers.get("Content-Encoding", None)
            # 如果内容编码为 "gzip"
            if content_encoding == "gzip":
                # 根据 Content-Encoding 头部覆盖压缩设置为 gzip 方法
                compression = {"method": "gzip"}
            # 将响应的内容读入 BytesIO 对象中
            reader = BytesIO(req.read())
        
        # 返回 IOArgs 对象，包含读取的文件内容、编码方式、压缩方法、关闭标志和模式信息
        return IOArgs(
            filepath_or_buffer=reader,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )
    # 如果路径是一个 fsspec URL
    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(
            filepath_or_buffer, str
        )  # 仅为了在这个分支中满足 mypy 的要求
        # 两种特殊情况的 s3-like 协议；这些在 Hadoop 中有特殊含义，
        # 但从 fsspec 的角度来看等同于 "s3"
        # cc #11071
        if filepath_or_buffer.startswith("s3a://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3a://", "s3://")
        if filepath_or_buffer.startswith("s3n://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3n://", "s3://")
        
        # 导入 fsspec 库
        fsspec = import_optional_dependency("fsspec")

        # 如果安装了 botocore，则以 anon=True 读取以允许从公共存储桶读取
        err_types_to_retry_with_anon: list[Any] = []
        try:
            import_optional_dependency("botocore")
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
            )

            # 定义可能需要使用 anon=True 重试的错误类型列表
            err_types_to_retry_with_anon = [
                ClientError,
                NoCredentialsError,
                PermissionError,
            ]
        except ImportError:
            pass

        try:
            # 尝试以指定的 mode 和 storage_options 打开文件对象
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()
        # GH 34626 如果从公共存储桶读取需要 anon=True
        except tuple(err_types_to_retry_with_anon):
            if storage_options is None:
                storage_options = {"anon": True}
            else:
                # 不改变用户输入，复制一份 storage_options
                storage_options = dict(storage_options)
                storage_options["anon"] = True
            # 以 anon=True 打开文件对象
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()

        # 返回 IOArgs 对象，包含打开的文件对象及相关参数
        return IOArgs(
            filepath_or_buffer=file_obj,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )
    
    # 如果传入了 storage_options，但不是 fsspec URL，则抛出 ValueError
    elif storage_options:
        raise ValueError(
            "storage_options passed with file object or non-fsspec file path"
        )

    # 如果 filepath_or_buffer 是字符串、字节串或 mmap.mmap 对象，则扩展用户路径
    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(
            filepath_or_buffer=_expand_user(filepath_or_buffer),
            encoding=encoding,
            compression=compression,
            should_close=False,
            mode=mode,
        )

    # 如果 filepath_or_buffer 不具有 read 或 write 属性，则抛出 ValueError
    # is_file_like 要求具有 (read | write) 和 __iter__ 方法，但对于 read_csv(engine=python) 仅需要 __iter__ 方法
    if not (
        hasattr(filepath_or_buffer, "read") or hasattr(filepath_or_buffer, "write")
    ):
        msg = f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
        raise ValueError(msg)
    # 创建并返回一个IOArgs对象，用于处理输入文件路径或缓冲区、编码、压缩方式等参数
    return IOArgs(
        # 文件路径或缓冲区的参数，传入的值为filepath_or_buffer
        filepath_or_buffer=filepath_or_buffer,
        # 编码参数，传入的值为encoding
        encoding=encoding,
        # 压缩方式参数，传入的值为compression
        compression=compression,
        # 指示是否应该关闭文件或缓冲区的标志，传入的值为False
        should_close=False,
        # 模式参数，传入的值为mode
        mode=mode,
    )
def file_path_to_url(path: str) -> str:
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    # 导入路径转换函数，延迟加载以提高性能（大约30毫秒）
    from urllib.request import pathname2url

    # 使用pathname2url函数将本地路径转换为FILE URL格式
    return urljoin("file:", pathname2url(path))


extension_to_compression = {
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tar.bz2": "tar",
    ".tar.xz": "tar",
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd",
}
_supported_compressions = set(extension_to_compression.values())


def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of (str or None, dict)
        The compression method and additional compression arguments.

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
    compression_method: str | None
    if isinstance(compression, Mapping):
        # 如果compression是一个映射，则将其转换为字典，并尝试获取'method'键对应的压缩方法
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop("method")
        except KeyError as err:
            # 如果没有找到'method'键，则抛出ValueError异常
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        # 如果compression是字符串，则无需额外参数，直接使用该字符串作为压缩方法
        compression_args = {}
        compression_method = compression
    return compression_method, compression_args


@doc(compression_options=_shared_docs["compression_options"] % "filepath_or_buffer")
def infer_compression(
    filepath_or_buffer: FilePath | BaseBuffer, compression: str | None
) -> str | None:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    compression : str or None
        Compression method to infer or validate.

    Returns
    -------
    str or None
        Inferred or validated compression method.

    Raises
    ------
    ValueError
        If an invalid compression method is specified.
    """
    if compression is None:
        return None

    # 推断压缩方法
    # 如果 compression 参数为 "infer"，则尝试推断压缩类型
    if compression == "infer":
        # 将所有路径类型（如 pathlib.Path）转换为字符串
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        # 如果 filepath_or_buffer 不是字符串，则无法推断其压缩类型，假设无压缩
        if not isinstance(filepath_or_buffer, str):
            return None

        # 从文件名或 URL 扩展名推断压缩类型
        for extension, compression in extension_to_compression.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None

    # 如果指定了 compression，检查其是否有效
    if compression in _supported_compressions:
        return compression

    # 如果 compression 不在支持的压缩类型列表中，抛出 ValueError 异常
    valid = ["infer", None] + sorted(_supported_compressions)
    msg = (
        f"Unrecognized compression type: {compression}\n"
        f"Valid compression types are {valid}"
    )
    raise ValueError(msg)
# 检查文件路径的父目录是否存在，若不存在则引发 OSError 异常
def check_parent_directory(path: Path | str) -> None:
    # 获取文件路径对象的父目录路径
    parent = Path(path).parent
    # 如果父目录不存在，则抛出异常
    if not parent.is_dir():
        raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")


# 获取文件处理句柄的重载函数，返回字节流的 IOHandles 对象
@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[False],
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[bytes]: ...


# 获取文件处理句柄的重载函数，返回字符串的 IOHandles 对象
@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[True] = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str]: ...


# 获取文件处理句柄的重载函数，返回字符串或字节流的 IOHandles 对象
@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str] | IOHandles[bytes]: ...


# 根据参数生成文件处理句柄的函数，根据参数返回不同类型的 IOHandles 对象
@doc(compression_options=_shared_docs["compression_options"] % "path_or_buf")
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = None,
    compression: CompressionOptions | None = None,
    memory_map: bool = False,
    is_text: bool = True,
    errors: str | None = None,
    storage_options: StorageOptions | None = None,
) -> IOHandles[str] | IOHandles[bytes]:
    """
    根据给定的路径/缓冲区和模式获取文件句柄。

    Parameters
    ----------
    path_or_buf : str 或文件句柄
        文件路径或对象。
    mode : str
        打开 path_or_buf 的模式。
    encoding : str 或 None
        使用的编码。
    {compression_options}
        如果压缩模式为 'zip'，则可以是具有 'method' 键作为压缩模式，其他键作为压缩选项的字典。

        .. versionchanged:: 1.4.0 支持 Zstandard。

    memory_map : bool，默认为 False
        查看 parsers._parser_params 以获取更多信息，仅被 read_csv 使用。
    is_text : bool，默认为 True
        内容传递给文件/缓冲区的类型是字符串还是字节流。这不同于 `"b" not in mode`。
        如果将字符串内容传递给二进制文件/缓冲区，则会插入包装器。
    errors : str，默认为 'strict'
        指定如何处理编码和解码错误。
        有关选项的完整列表，请参见 :func:`open` 的 errors 参数。

    """
    storage_options: StorageOptions = None
        # 存储选项，传递给 _get_filepath_or_buffer 函数

    Returns the dataclass IOHandles
    """
    # Windows 默认不使用 utf-8 编码，为了保持一致性，设置为 utf-8
    encoding = encoding or "utf-8"

    # 错误处理方式，默认为严格模式
    errors = errors or "strict"

    # read_csv 函数无法判断缓冲区是以二进制还是文本模式打开，若以二进制打开则添加 'b' 标记
    if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
        mode += "b"

    # 校验指定的编码是否有效
    codecs.lookup(encoding)
    if isinstance(errors, str):
        # 校验指定的错误处理方式是否有效
        codecs.lookup_error(errors)

    # 处理 URL 打开的情况
    ioargs = _get_filepath_or_buffer(
        path_or_buf,
        encoding=encoding,
        compression=compression,
        mode=mode,
        storage_options=storage_options,
    )

    # 获取文件路径或缓冲区对象
    handle = ioargs.filepath_or_buffer
    handles: list[BaseBuffer]

    # 内存映射需要作为第一步进行，仅在 read_csv 中使用
    handle, memory_map, handles = _maybe_memory_map(handle, memory_map)

    # 判断 handle 是否为路径字符串
    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop("method")

    # 仅针对写入方法
    if "r" not in mode and is_path:
        # 检查父目录是否存在，用于写入操作
        check_parent_directory(str(handle))

    elif isinstance(handle, str):
        # 检查是否以二进制模式打开文件，二进制模式不支持 'encoding' 和 'newline' 参数
        if ioargs.encoding and "b" not in ioargs.mode:
            # 以指定编码方式打开文件
            handle = open(
                handle,
                ioargs.mode,
                encoding=ioargs.encoding,
                errors=errors,
                newline="",
            )
        else:
            # 以二进制模式打开文件
            handle = open(handle, ioargs.mode)
        handles.append(handle)

    # 处理转换 BytesIO 或文件对象传递时的编码情况
    is_wrapped = False
    if not is_text and ioargs.mode == "rb" and isinstance(handle, TextIOBase):
        # 如果不是文本模式，并且以二进制模式打开且是文本IO对象，则进行包装处理
        handle = _BytesIOWrapper(
            handle,
            encoding=ioargs.encoding,
        )
    elif is_text and (
        compression or memory_map or _is_binary_mode(handle, ioargs.mode)
    ):
        if (
            not hasattr(handle, "readable")
            or not hasattr(handle, "writable")
            or not hasattr(handle, "seekable")
        ):
            # 若 handle 不具备预期的文件对象方法，则进行包装处理
            handle = _IOWrapper(handle)
        # 以指定编码方式打开文件对象
        handle = TextIOWrapper(
            handle,  # type: ignore[arg-type]
            encoding=ioargs.encoding,
            errors=errors,
            newline="",
        )
        handles.append(handle)
        # 当调用者提供了 handle 时，标记为已包装
        is_wrapped = not (
            isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close
        )
    # 检查 ioargs.mode 中是否包含 "r" 并且 handle 没有 "read" 方法
    if "r" in ioargs.mode and not hasattr(handle, "read"):
        # 如果条件成立，抛出 TypeError 异常，指示预期的文件路径名或类文件对象类型不匹配
        raise TypeError(
            "Expected file path name or file-like object, "
            f"got {type(ioargs.filepath_or_buffer)} type"
        )

    # 将 handles 列表反转，以便先关闭最近添加的缓冲区
    handles.reverse()  # close the most recently added buffer first
    
    # 如果 ioargs.should_close 为真，则断言 ioargs.filepath_or_buffer 不是字符串
    if ioargs.should_close:
        assert not isinstance(ioargs.filepath_or_buffer, str)
        # 将 ioargs.filepath_or_buffer 添加到 handles 列表中
        handles.append(ioargs.filepath_or_buffer)

    # 返回一个 IOHandles 对象，封装了处理过的文件对象和相关信息
    return IOHandles(
        # 设置 handle 参数为当前处理的文件对象
        handle=handle,  # type: ignore[arg-type]
        # 设置 created_handles 参数为处理过的文件对象列表 handles
        created_handles=handles,  # type: ignore[arg-type]
        # 设置 is_wrapped 参数为是否进行了包装的标志
        is_wrapped=is_wrapped,
        # 设置 compression 参数为 ioargs 中指定的压缩算法
        compression=ioargs.compression,
    )
# error: Definition of "__enter__" in base class "IOBase" is incompatible
# with definition in base class "BinaryIO"
class _BufferedWriter(BytesIO, ABC):  # type: ignore[misc]
    """
    Some objects do not support multiple .write() calls (TarFile and ZipFile).
    This wrapper writes to the underlying buffer on close.
    """

    buffer = BytesIO()  # 创建一个字节流作为缓冲区

    @abstractmethod
    def write_to_buffer(self) -> None:  # 抽象方法，子类需实现将数据写入缓冲区的操作
        ...

    def close(self) -> None:
        if self.closed:
            # 如果已经关闭，则直接返回
            return
        if self.getbuffer().nbytes:
            # 如果缓冲区不为空
            self.seek(0)  # 将当前指针位置移动到文件开头
            with self.buffer:
                self.write_to_buffer()  # 调用子类实现的方法，将数据写入缓冲区
        else:
            self.buffer.close()  # 关闭缓冲区
        super().close()  # 调用父类的关闭方法


class _BytesTarFile(_BufferedWriter):
    def __init__(
        self,
        name: str | None = None,
        mode: Literal["r", "a", "w", "x"] = "r",
        fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None = None,
        archive_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # 调用父类的构造方法
        self.archive_name = archive_name  # 设置归档文件名
        self.name = name  # 设置文件名
        # error: Incompatible types in assignment (expression has type "TarFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: tarfile.TarFile = tarfile.TarFile.open(  # type: ignore[assignment]
            name=name,
            mode=self.extend_mode(mode),  # 扩展文件打开模式
            fileobj=fileobj,
            **kwargs,
        )

    def extend_mode(self, mode: str) -> str:
        mode = mode.replace("b", "")  # 移除模式中的 'b' 字符
        if mode != "w":
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in (".gz", ".xz", ".bz2"):
                mode = f"{mode}:{suffix[1:]}"  # 如果有压缩后缀，则追加到模式中
        return mode

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
        if self.name is None:
            return None

        filename = Path(self.name)
        if filename.suffix == ".tar":
            return filename.with_suffix("").name  # 移除 '.tar' 后缀
        elif filename.suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            return filename.with_suffix("").with_suffix("").name  # 移除 '.tar.gz' 等双后缀
        return filename.name  # 返回文件名

    def write_to_buffer(self) -> None:
        # TarFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "tar"  # 推断归档文件名
        tarinfo = tarfile.TarInfo(name=archive_name)  # 创建 TarInfo 对象
        tarinfo.size = len(self.getvalue())  # 设置 TarInfo 的文件大小
        self.buffer.addfile(tarinfo, self)  # 向 TarFile 写入数据


class _BytesZipFile(_BufferedWriter):
    def __init__(
        self,
        file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
        mode: str,
        archive_name: str | None = None,
        **kwargs: Any,
        ) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 替换模式中的 'b'，如果有的话
        mode = mode.replace("b", "")
        # 设置存档名称
        self.archive_name = archive_name

        # 设置压缩参数，默认为 ZIP_DEFLATED 压缩
        kwargs.setdefault("compression", zipfile.ZIP_DEFLATED)
        # 使用给定的文件、模式和参数创建 ZipFile 对象，存储在 self.buffer 中
        self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
            file, mode, **kwargs
        )

    def infer_filename(self) -> str | None:
        """
        如果没有显式提供存档名称，我们希望 ZIP 文件中的文件不要以 .zip 结尾，
        因为这会引起混淆 (GH39465)。
        """
        # 检查 self.buffer.filename 是否为字符串或类似路径的实例
        if isinstance(self.buffer.filename, (os.PathLike, str)):
            # 将文件名转换为 Path 对象
            filename = Path(self.buffer.filename)
            # 如果文件名以 .zip 结尾，则返回去掉后缀的文件名
            if filename.suffix == ".zip":
                return filename.with_suffix("").name
            # 否则返回文件名本身
            return filename.name
        # 如果 self.buffer.filename 不是字符串或路径实例，返回 None
        return None

    def write_to_buffer(self) -> None:
        # 确定存档名称，优先使用 self.archive_name，其次根据推断得到的文件名，最后默认为 "zip"
        archive_name = self.archive_name or self.infer_filename() or "zip"
        # 将当前缓冲区的内容以 archive_name 为文件名写入到 ZipFile 对象中
        self.buffer.writestr(archive_name, self.getvalue())
class _IOWrapper:
    # TextIOWrapper is overly strict: it request that the buffer has seekable, readable,
    # and writable. If we have a read-only buffer, we shouldn't need writable and vice
    # versa. Some buffers, are seek/read/writ-able but they do not have the "-able"
    # methods, e.g., tempfile.SpooledTemporaryFile.
    # If a buffer does not have the above "-able" methods, we simple assume they are
    # seek/read/writ-able.
    
    # 初始化方法，接受一个 BaseBuffer 对象作为参数
    def __init__(self, buffer: BaseBuffer) -> None:
        self.buffer = buffer

    # 当访问对象的属性时，委托给被包装的 buffer 对象
    def __getattr__(self, name: str) -> Any:
        return getattr(self.buffer, name)

    # 判断包装的 buffer 是否可读
    def readable(self) -> bool:
        if hasattr(self.buffer, "readable"):
            return self.buffer.readable()
        return True

    # 判断包装的 buffer 是否可寻址（可定位读取位置）
    def seekable(self) -> bool:
        if hasattr(self.buffer, "seekable"):
            return self.buffer.seekable()
        return True

    # 判断包装的 buffer 是否可写
    def writable(self) -> bool:
        if hasattr(self.buffer, "writable"):
            return self.buffer.writable()
        return True


class _BytesIOWrapper:
    # Wrapper that wraps a StringIO buffer and reads bytes from it
    # Created for compat with pyarrow read_csv
    
    # 初始化方法，接受一个 StringIO 或 TextIOBase 对象和一个编码参数作为输入
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = "utf-8") -> None:
        self.buffer = buffer
        self.encoding = encoding
        # 因为一个字符可能由多个字节表示，因此可能读取的字节比实际的 n 更多
        # 我们将额外的字节存储在 overflow 变量中，下次读取时将其添加到字节流的前面
        self.overflow = b""

    # 当访问对象的属性时，委托给被包装的 buffer 对象
    def __getattr__(self, attr: str) -> Any:
        return getattr(self.buffer, attr)

    # 从包装的 buffer 中读取指定数量的字节
    def read(self, n: int | None = -1) -> bytes:
        assert self.buffer is not None
        # 读取 n 个字符，并使用指定的编码转换为字节串
        bytestring = self.buffer.read(n).encode(self.encoding)
        # 处理可能存在的字节溢出情况
        combined_bytestring = self.overflow + bytestring
        if n is None or n < 0 or n >= len(combined_bytestring):
            self.overflow = b""
            return combined_bytestring
        else:
            to_return = combined_bytestring[:n]
            self.overflow = combined_bytestring[n:]
            return to_return


def _maybe_memory_map(
    handle: str | BaseBuffer, memory_map: bool
) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
    """Try to memory map file/buffer."""
    handles: list[BaseBuffer] = []
    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
    if not memory_map:
        return handle, memory_map, handles

    # 如果需要内存映射，转换为 ReadCsvBuffer 类型
    handle = cast(ReadCsvBuffer, handle)

    # 如果 handle 是字符串类型，则打开文件并添加到 handles 列表中
    if isinstance(handle, str):
        handle = open(handle, "rb")
        handles.append(handle)

    # 返回处理后的 handle，内存映射标志以及打开的文件句柄列表
    return handle, memory_map, handles
    try:
        # 尝试打开 mmap 并将其包装为可读的 _IOWrapper 对象
        # 错误："_IOWrapper" 的第一个参数类型为 "mmap"，但期望的是 "BaseBuffer"
        wrapped = _IOWrapper(
            mmap.mmap(
                handle.fileno(),
                0,
                access=mmap.ACCESS_READ,  # 忽略类型检查：arg-type
            )
        )
    finally:
        # 反向处理所有的 handles
        for handle in reversed(handles):
            # 错误： "BaseBuffer" 对象没有 "close" 属性
            handle.close()  # 忽略属性定义的类型检查

    # 返回封装后的对象 wrapped，memory_map 变量和包含 wrapped 的列表
    return wrapped, memory_map, [wrapped]
# 测试文件是否存在
def file_exists(filepath_or_buffer: FilePath | BaseBuffer) -> bool:
    """Test whether file exists."""
    exists = False
    # 将文件路径或缓冲区转换为字符串路径
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    # 如果不是字符串类型则直接返回 False
    if not isinstance(filepath_or_buffer, str):
        return exists
    try:
        # 检查文件或路径是否存在
        exists = os.path.exists(filepath_or_buffer)
        # 如果文件路径过长会在此处引发异常（见 GitHub 问题 #5874）
        # gh-5874: if the filepath is too long will raise here
    except (TypeError, ValueError):
        pass
    return exists


# 检查句柄是否以二进制模式打开
def _is_binary_mode(handle: FilePath | BaseBuffer, mode: str) -> bool:
    """Whether the handle is opened in binary mode"""
    # 如果用户指定了 't' 或 'b' 模式，则返回是否包含 'b'
    if "t" in mode or "b" in mode:
        return "b" in mode

    # 处理一些例外情况
    text_classes = (
        # 预期字符串但模式包含 'b' 的类
        codecs.StreamWriter,
        codecs.StreamReader,
        codecs.StreamReaderWriter,
    )
    # 如果句柄是文本类流的子类，则不是二进制模式
    if issubclass(type(handle), text_classes):
        return False

    # 否则，根据句柄的类型判断是否为二进制 IO 类
    return isinstance(handle, _get_binary_io_classes()) or "b" in getattr(
        handle, "mode", mode
    )


# 获取预期接受字节的 IO 类
@functools.lru_cache
def _get_binary_io_classes() -> tuple[type, ...]:
    """IO classes that that expect bytes"""
    # 定义预期接受字节的 IO 类
    binary_classes: tuple[type, ...] = (BufferedIOBase, RawIOBase)

    # 导入可选依赖项 'zstandard'，并在可能的情况下添加额外的二进制 IO 类
    zstd = import_optional_dependency("zstandard", errors="ignore")
    if zstd is not None:
        with zstd.ZstdDecompressor().stream_reader(b"") as reader:
            binary_classes += (type(reader),)

    return binary_classes


# 检查列是否有潜在成为 MultiIndex 的可能性
def is_potential_multi_index(
    columns: Sequence[Hashable] | MultiIndex,
    index_col: bool | Sequence[int] | None = None,
) -> bool:
    """
    Check whether or not the `columns` parameter
    could be converted into a MultiIndex.

    Parameters
    ----------
    columns : array-like
        Object which may or may not be convertible into a MultiIndex
    index_col : None, bool or list, optional
        Column or columns to use as the (possibly hierarchical) index

    Returns
    -------
    bool : Whether or not columns could become a MultiIndex
    """
    # 如果 index_col 为 None 或布尔值，则不指定索引列
    if index_col is None or isinstance(index_col, bool):
        index_columns = set()
    else:
        # 否则将 index_col 转换为集合
        index_columns = set(index_col)

    # 检查列是否可以成为 MultiIndex
    return bool(
        len(columns)
        and not isinstance(columns, ABCMultiIndex)
        and all(isinstance(c, tuple) for c in columns if c not in index_columns)
    )


# 如果存在重复列名，则重命名列名以避免冲突
def dedup_names(
    names: Sequence[Hashable], is_potential_multiindex: bool
) -> Sequence[Hashable]:
    """
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    # 将输入的names转换为列表，以便可以索引操作
    names = list(names)  # so we can index
    # 创建一个默认字典，用于记录每个名称出现的次数
    counts: DefaultDict[Hashable, int] = defaultdict(int)

    # 遍历names列表的索引和元素
    for i, col in enumerate(names):
        # 获取当前名称col已经出现的次数
        cur_count = counts[col]

        # 如果当前名称已经出现过，进入循环直到找到一个未被使用的新名称
        while cur_count > 0:
            # 将当前名称的计数加1
            counts[col] = cur_count + 1

            # 如果允许潜在的多级索引（is_potential_multiindex为True）
            if is_potential_multiindex:
                # 确保col是元组类型
                assert isinstance(col, tuple)
                # 修改col的最后一个元素，添加一个新的编号
                col = col[:-1] + (f"{col[-1]}.{cur_count}",)
            else:
                # 修改col，添加一个新的编号
                col = f"{col}.{cur_count}"
            
            # 更新当前名称的计数
            cur_count = counts[col]

        # 将修改后的名称保存回names列表中的原位置
        names[i] = col
        # 更新当前名称的计数
        counts[col] = cur_count + 1

    # 返回更新后的names列表，其中重复的名称已经添加了编号
    return names
```