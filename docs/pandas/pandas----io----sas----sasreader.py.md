# `D:\src\scipysrc\pandas\pandas\io\sas\sasreader.py`

```
"""
Read SAS sas7bdat or xport files.
"""

from __future__ import annotations  # 导入未来版本的类型注解支持

from abc import (  # 导入抽象基类相关模块
    ABC,  # Python 中的抽象基类
    abstractmethod,  # 定义抽象方法的装饰器
)
from collections.abc import Iterator  # 导入迭代器抽象基类
from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,  # 类型检查标记
    overload,  # 函数重载装饰器
)

from pandas.util._decorators import doc  # 导入文档装饰器

from pandas.core.shared_docs import _shared_docs  # 导入共享文档
from pandas.io.common import stringify_path  # 导入路径字符串化函数

if TYPE_CHECKING:  # 如果是类型检查模式
    from collections.abc import Hashable  # 导入可哈希类型抽象基类
    from types import TracebackType  # 导入追踪类型

    from pandas._typing import (  # 导入 Pandas 类型定义
        CompressionOptions,  # 压缩选项类型
        FilePath,  # 文件路径类型
        ReadBuffer,  # 读取缓冲区类型
        Self,  # 自引用类型
    )

    from pandas import DataFrame  # 导入 DataFrame 类型


class SASReader(Iterator["DataFrame"], ABC):  # 定义 SASReader 类，继承自 Iterator 和 ABC
    """
    Abstract class for XportReader and SAS7BDATReader.
    """

    @abstractmethod
    def read(self, nrows: int | None = None) -> DataFrame: ...
    # 抽象方法：读取数据，返回 DataFrame 对象

    @abstractmethod
    def close(self) -> None: ...
    # 抽象方法：关闭资源，无返回值

    def __enter__(self) -> Self:
        return self
    # 进入上下文管理器时返回自身对象

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
    # 退出上下文管理器时调用 close 方法关闭资源


@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
) -> SASReader: ...
# 函数重载：返回 SASReader 对象的函数定义

@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: None = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
) -> DataFrame | SASReader: ...
# 函数重载：返回 DataFrame 或 SASReader 对象的函数定义

@doc(decompression_options=_shared_docs["decompression_options"] % "filepath_or_buffer")
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> DataFrame | SASReader:
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.sas7bdat``.
    format : str {{'xport', 'sas7bdat'}} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    """
    # 函数文档字符串：读取存储为 XPORT 或 SAS7BDAT 格式的 SAS 文件
    # encoding : str, default is None
    # 文本数据的编码方式，默认为 None，即文本数据存储为原始字节。
    chunksize : int
    # 一次读取文件的行数大小，返回一个迭代器。
    iterator : bool, defaults to False
    # 如果为 True，返回一个用于增量读取文件的迭代器。
    {decompression_options}

    Returns
    -------
    DataFrame, SAS7BDATReader, or XportReader
    # 如果 iterator=False 且 chunksize=None，则返回 DataFrame；否则根据文件扩展名推断返回 SAS7BDATReader 或 XportReader。

    See Also
    --------
    read_csv : 将逗号分隔值（CSV）文件读取到 pandas DataFrame 中。
    read_excel : 将 Excel 文件读取到 pandas DataFrame 中。
    read_spss : 将 SPSS 文件读取到 pandas DataFrame 中。
    read_orc : 将 ORC 对象加载到 pandas DataFrame 中。
    read_feather : 将 feather 格式对象加载到 pandas DataFrame 中。

    Examples
    --------
    >>> df = pd.read_sas("sas_data.sas7bdat")  # doctest: +SKIP
    """
    if format is None:
        buffer_error_msg = (
            "If this is a buffer object rather "
            "than a string name, you must specify a format string"
        )
        # 将 filepath_or_buffer 转换为字符串路径
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        # 如果 filepath_or_buffer 不是字符串，则抛出 ValueError 异常
        if not isinstance(filepath_or_buffer, str):
            raise ValueError(buffer_error_msg)
        # 将文件路径或缓冲区名转换为小写
        fname = filepath_or_buffer.lower()
        # 根据文件名推断文件格式
        if ".xpt" in fname:
            format = "xport"
        elif ".sas7bdat" in fname:
            format = "sas7bdat"
        else:
            # 如果无法推断 SAS 文件的格式，则抛出 ValueError 异常
            raise ValueError(
                f"unable to infer format of SAS file from filename: {fname!r}"
            )

    # 创建一个 SASReader 类型的 reader 变量
    reader: SASReader
    # 根据 format 的值确定读取器的具体类型
    if format.lower() == "xport":
        from pandas.io.sas.sas_xport import XportReader

        # 使用 XportReader 类创建读取器对象
        reader = XportReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    elif format.lower() == "sas7bdat":
        from pandas.io.sas.sas7bdat import SAS7BDATReader

        # 使用 SAS7BDATReader 类创建读取器对象
        reader = SAS7BDATReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    else:
        # 如果 format 不是 "xport" 或 "sas7bdat"，则抛出 ValueError 异常
        raise ValueError("unknown SAS format")

    # 如果 iterator 或 chunksize 为真，则直接返回 reader 对象
    if iterator or chunksize:
        return reader

    # 使用 reader 对象进行上下文管理，读取文件数据并返回
    with reader:
        return reader.read()
```