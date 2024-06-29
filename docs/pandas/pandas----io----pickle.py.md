# `D:\src\scipysrc\pandas\pandas\io\pickle.py`

```
# 导入pickle模块，用于对象的序列化和反序列化操作
import pickle
# 引入类型检查标记，用于静态类型检查
from typing import (
    TYPE_CHECKING,
    Any,
)
# 引入警告模块，用于处理警告信息
import warnings

# 从pandas.compat模块导入pickle_compat
from pandas.compat import pickle_compat
# 从pandas.util._decorators模块导入doc装饰器
from pandas.util._decorators import doc

# 从pandas.core.shared_docs模块导入_shared_docs
from pandas.core.shared_docs import _shared_docs

# 从pandas.io.common模块导入get_handle函数
from pandas.io.common import get_handle

# 如果TYPE_CHECKING为True，从pandas._typing中导入相关类型
if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadPickleBuffer,
        StorageOptions,
        WriteBuffer,
    )
    # 从pandas中导入DataFrame和Series类型
    from pandas import (
        DataFrame,
        Series,
    )

# 使用doc装饰器注释to_pickle函数，添加storage_options和compression_options参数的文档说明
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePath | WriteBuffer[bytes],
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions | None = None,
) -> None:
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : any object
        Any python object.
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function.
        Also accepts URL. URL has to be of S3 or GCS.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible
        values for this parameter depend on the version of Python. For Python
        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.
        For Python >= 3.4, 4 is a valid value. A negative value for the
        protocol parameter is equivalent to setting its value to
        HIGHEST_PROTOCOL.

    {storage_options}

        .. [1] https://docs.python.org/3/library/pickle.html

    See Also
    --------
    read_pickle : Load pickled pandas object (or any object) from file.
    DataFrame.to_hdf : Write DataFrame to an HDF5 file.
    DataFrame.to_sql : Write DataFrame to a SQL database.
    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {{"foo": range(5), "bar": range(5, 10)}}
    ... )  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """
    # 如果protocol参数小于0，则将其设为最高协议版本号
    if protocol < 0:
        protocol = pickle.HIGHEST_PROTOCOL
    # 使用 `get_handle` 函数获取文件句柄，以写入二进制模式打开文件（"wb"）
    # 根据提供的压缩选项选择是否压缩数据，is_text 参数表明数据是否为文本而非二进制
    # storage_options 可能包含其他存储选项
    with get_handle(
        filepath_or_buffer,
        "wb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # 直接让 pickle 将对象 obj 直接写入句柄，这样更节省内存
        pickle.dump(obj, handles.handle, protocol=protocol)
@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "filepath_or_buffer",
)
# 定义函数 read_pickle，用于从文件中加载序列化的 pandas 对象（或任何对象），并返回反序列化后的对象
def read_pickle(
    filepath_or_buffer: FilePath | ReadPickleBuffer,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
) -> DataFrame | Series:
    """
    Load pickled pandas object (or any object) from file and return unpickled object.

    .. warning::

       Loading pickled data received from untrusted sources can be
       unsafe. See `here <https://docs.python.org/3/library/pickle.html>`__.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``readlines()`` function.
        Also accepts URL. URL is not limited to S3 and GCS.

    {decompression_options}
        解压选项，参见共享文档中的解压选项说明

        .. versionchanged:: 1.4.0 Zstandard support.
            版本变更说明：1.4.0 版本添加了对 Zstandard 的支持

    {storage_options}
        存储选项，参见共享文档中的存储选项说明

    Returns
    -------
    object
        反序列化后的 pandas 对象（或任何对象）

    See Also
    --------
    DataFrame.to_pickle : 将 DataFrame 对象序列化到文件中。
    Series.to_pickle : 将 Series 对象序列化到文件中。
    read_hdf : 从 HDF5 文件中读取到 DataFrame 中。
    read_sql : 从 SQL 查询或数据库表中读取到 DataFrame 中。
    read_parquet : 加载 Parquet 对象，返回一个 DataFrame。

    Notes
    -----
    read_pickle 只能保证与 pandas 1.0 向后兼容，前提是对象是用 to_pickle 序列化的。

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {{"foo": range(5), "bar": range(5, 10)}}
    ... )  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """
    # 定义可能会捕获的异常类型列表，用于捕获 Cython 关于对象新建的异常
    excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
    # 使用 get_handle 函数获取文件句柄，并以二进制只读模式打开文件
    with get_handle(
        filepath_or_buffer,
        "rb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # 使用上下文管理器打开文件句柄 handles

        # 1) 尝试使用标准库 Pickle 加载数据
        # 2) 尝试使用 pickle_compat（用于较旧的 pandas 版本）处理子类变更
        try:
            with warnings.catch_warnings(record=True):
                # 忽略所有警告，例如有关移动模块等的警告
                warnings.simplefilter("ignore", Warning)
                # 从文件句柄 handles 中加载 Pickle 数据
                return pickle.load(handles.handle)
        except excs_to_catch:
            # 捕获特定的异常类型 excs_to_catch
            # 例如：
            # "No module named 'pandas.core.sparse.series'"
            # "Can't get attribute '_nat_unpickle' on <module 'pandas._libs.tslib"
            # 将文件句柄 handles 的位置重置到开头
            handles.handle.seek(0)
            # 使用 pickle_compat.Unpickler 从 handles 中加载数据
            return pickle_compat.Unpickler(handles.handle).load()
```