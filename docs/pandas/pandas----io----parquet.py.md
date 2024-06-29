# `D:\src\scipysrc\pandas\pandas\io\parquet.py`

```
# 导入必要的模块和函数，包括将来版本兼容性的声明
"""parquet compat"""
from __future__ import annotations

import io  # 导入用于处理文件流的模块
import json  # 导入用于 JSON 操作的模块
import os  # 导入用于操作操作系统功能的模块
from typing import (  # 导入类型提示相关的模块
    TYPE_CHECKING,
    Any,
    Literal,
)
from warnings import (  # 导入警告相关的模块
    catch_warnings,
    filterwarnings,
)

from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 内部配置相关模块

from pandas._libs import lib  # 导入 pandas 核心 C 库相关模块
from pandas.compat._optional import import_optional_dependency  # 导入 pandas 兼容性依赖模块
from pandas.errors import AbstractMethodError  # 导入 pandas 错误处理相关模块
from pandas.util._decorators import doc  # 导入 pandas 工具装饰器相关模块
from pandas.util._validators import check_dtype_backend  # 导入 pandas 数据类型验证相关模块

import pandas as pd  # 导入 pandas 库并命名为 pd
from pandas import (  # 从 pandas 中导入 DataFrame 和 get_option 函数
    DataFrame,
    get_option,
)
from pandas.core.shared_docs import _shared_docs  # 导入 pandas 共享文档相关模块

from pandas.io._util import arrow_string_types_mapper  # 导入 pandas I/O 工具模块
from pandas.io.common import (  # 导入 pandas I/O 公共工具函数
    IOHandles,
    get_handle,
    is_fsspec_url,
    is_url,
    stringify_path,
)

if TYPE_CHECKING:
    from pandas._typing import (  # 根据类型检查导入 pandas 类型提示
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )


def get_engine(engine: str) -> BaseImpl:
    """根据指定的引擎名称返回对应的实现对象"""
    if engine == "auto":
        engine = get_option("io.parquet.engine")  # 如果引擎名称为 "auto"，则从配置中获取默认的引擎名

    if engine == "auto":
        # 尝试按照以下顺序使用不同的引擎类
        engine_classes = [PyArrowImpl, FastParquetImpl]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()  # 尝试实例化每个引擎类对象并返回
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "无法找到可用的引擎；已尝试使用：'pyarrow'、'fastparquet'。\n"
            "parquet 格式需要合适版本的 pyarrow 或 fastparquet 支持。\n"
            "尝试导入上述模块时出现以下错误："
            f"{error_msgs}"
        )

    if engine == "pyarrow":
        return PyArrowImpl()  # 返回 pyarrow 引擎的实现对象
    elif engine == "fastparquet":
        return FastParquetImpl()  # 返回 fastparquet 引擎的实现对象

    raise ValueError("引擎名称必须是 'pyarrow' 或 'fastparquet'")  # 抛出值错误，引擎名称不合法


def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],  # 定义函数 _get_path_or_handle，接收文件路径或字节流等参数
    fs: Any,  # 文件系统相关参数
    storage_options: StorageOptions | None = None,  # 存储选项，默认为 None
    mode: str = "rb",  # 文件打开模式，默认为只读二进制模式
    is_dir: bool = False,  # 是否为目录，默认为 False
) -> tuple[
    FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any
]:
    """处理 PyArrow 所需的文件操作"""
    path_or_handle = stringify_path(path)  # 将输入的路径参数转换为字符串表示
    # 如果文件系统对象不为空
    if fs is not None:
        # 尝试导入 pyarrow.fs 库，忽略导入错误
        pa_fs = import_optional_dependency("pyarrow.fs", errors="ignore")
        # 尝试导入 fsspec 库，忽略导入错误
        fsspec = import_optional_dependency("fsspec", errors="ignore")
        
        # 如果成功导入了 pyarrow.fs 并且 fs 是 pyarrow.fs.FileSystem 类型的实例
        if pa_fs is not None and isinstance(fs, pa_fs.FileSystem):
            # 如果提供了 storage_options，抛出未实现错误，pyarrow FileSystem 不支持 storage_options
            if storage_options:
                raise NotImplementedError(
                    "storage_options not supported with a pyarrow FileSystem."
                )
        
        # 如果导入了 fsspec 并且 fs 是 fsspec.spec.AbstractFileSystem 类型的实例
        elif fsspec is not None and isinstance(fs, fsspec.spec.AbstractFileSystem):
            pass
        
        # 如果 fs 不是 pyarrow 或 fsspec 文件系统类型的实例，则抛出值错误
        else:
            raise ValueError(
                f"filesystem must be a pyarrow or fsspec FileSystem, "
                f"not a {type(fs).__name__}"
            )
    
    # 如果 path_or_handle 是 fsspec URL 且 fs 为空
    if is_fsspec_url(path_or_handle) and fs is None:
        # 如果未提供 storage_options
        if storage_options is None:
            # 尝试导入 pyarrow 库和 pyarrow.fs 库
            pa = import_optional_dependency("pyarrow")
            pa_fs = import_optional_dependency("pyarrow.fs")

            try:
                # 使用 pyarrow.fs.FileSystem.from_uri 从路径获取文件系统对象和路径
                fs, path_or_handle = pa_fs.FileSystem.from_uri(path)
            except (TypeError, pa.ArrowInvalid):
                pass
        
        # 如果仍然没有有效的文件系统对象 fs
        if fs is None:
            # 尝试导入 fsspec 库
            fsspec = import_optional_dependency("fsspec")
            # 使用 fsspec.core.url_to_fs 从 URL 获取文件系统对象 fs 和路径
            fs, path_or_handle = fsspec.core.url_to_fs(
                path_or_handle, **(storage_options or {})
            )
    
    # 否则，如果提供了 storage_options 且 path_or_handle 不是 URL 或者 mode 不是 "rb"
    elif storage_options and (not is_url(path_or_handle) or mode != "rb"):
        # 不能对远程 URL 进行写入操作，此时需要使用 fsspec
        raise ValueError("storage_options passed with buffer, or non-supported URL")
    
    # 初始化句柄为 None
    handles = None
    
    # 如果没有有效的文件系统对象 fs，并且 path_or_handle 不是目录，是一个字符串路径且不是目录
    if (
        not fs
        and not is_dir
        and isinstance(path_or_handle, str)
        and not os.path.isdir(path_or_handle)
    ):
        # 调用 get_handle 函数获取文件句柄
        handles = get_handle(
            path_or_handle, mode, is_text=False, storage_options=storage_options
        )
        # 重置 fs 为 None，路径为句柄的路径
        fs = None
        path_or_handle = handles.handle
    
    # 返回路径或句柄 path_or_handle，句柄 handles，文件系统对象 fs
    return path_or_handle, handles, fs
class BaseImpl:
    @staticmethod
    # 静态方法：验证输入是否为 DataFrame 类型，否则抛出数值错误异常
    def validate_dataframe(df: DataFrame) -> None:
        if not isinstance(df, DataFrame):
            raise ValueError("to_parquet only supports IO with DataFrames")

    # 抽象方法：用于子类实现，写入 DataFrame 数据到指定路径
    def write(self, df: DataFrame, path, compression, **kwargs) -> None:
        raise AbstractMethodError(self)

    # 抽象方法：用于子类实现，从指定路径读取数据并返回 DataFrame 对象
    def read(self, path, columns=None, **kwargs) -> DataFrame:
        raise AbstractMethodError(self)


class PyArrowImpl(BaseImpl):
    # 构造方法：初始化 PyArrowImpl 类，引入必要的依赖并注册扩展类型
    def __init__(self) -> None:
        import_optional_dependency(
            "pyarrow", extra="pyarrow is required for parquet support."
        )
        import pyarrow.parquet

        # 导入 utils 以注册 pyarrow 扩展类型
        import pandas.core.arrays.arrow.extension_types  # pyright: ignore[reportUnusedImport] # noqa: F401

        # 设置 pyarrow API 对象
        self.api = pyarrow

    # 实现方法：将 DataFrame 写入指定路径，支持压缩和其他选项
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: str | None = "snappy",
        index: bool | None = None,
        storage_options: StorageOptions | None = None,
        partition_cols: list[str] | None = None,
        filesystem=None,
        **kwargs,
    ) -> None:
        # 调用内部方法验证数据框的有效性
        self.validate_dataframe(df)

        # 准备从 Pandas 数据框创建 Table 对象的参数字典
        from_pandas_kwargs: dict[str, Any] = {"schema": kwargs.pop("schema", None)}
        if index is not None:
            from_pandas_kwargs["preserve_index"] = index

        # 使用 API 创建 Table 对象，从 Pandas 数据框转换而来
        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)

        # 如果 Pandas 数据框具有元数据属性
        if df.attrs:
            # 将 Pandas 数据框的 attrs 属性转换为 JSON 字符串，并包装成字典
            df_metadata = {"PANDAS_ATTRS": json.dumps(df.attrs)}
            existing_metadata = table.schema.metadata
            # 合并现有的表结构元数据和 Pandas 数据框的元数据
            merged_metadata = {**existing_metadata, **df_metadata}
            # 替换 Table 对象的结构元数据
            table = table.replace_schema_metadata(merged_metadata)

        # 获取路径或文件句柄以及相关的文件系统和处理程序
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path,
            filesystem,
            storage_options=storage_options,
            mode="wb",
            is_dir=partition_cols is not None,
        )

        # 如果路径或句柄是 io.BufferedWriter 类型且具有 name 属性，则获取其名称
        if (
            isinstance(path_or_handle, io.BufferedWriter)
            and hasattr(path_or_handle, "name")
            and isinstance(path_or_handle.name, (str, bytes))
        ):
            if isinstance(path_or_handle.name, bytes):
                # 如果名称是字节流，则解码为字符串
                path_or_handle = path_or_handle.name.decode()
            else:
                path_or_handle = path_or_handle.name

        try:
            # 如果存在分区列，则将数据写入给定路径下的多个文件
            if partition_cols is not None:
                # 使用 API 将 Table 对象以 Parquet 格式写入数据集
                self.api.parquet.write_to_dataset(
                    table,
                    path_or_handle,
                    compression=compression,
                    partition_cols=partition_cols,
                    filesystem=filesystem,
                    **kwargs,
                )
            else:
                # 否则，将 Table 对象以 Parquet 格式写入单个输出文件
                self.api.parquet.write_table(
                    table,
                    path_or_handle,
                    compression=compression,
                    filesystem=filesystem,
                    **kwargs,
                )
        finally:
            # 最终操作：如果存在文件句柄，则关闭句柄
            if handles is not None:
                handles.close()

    def read(
        self,
        path,
        columns=None,
        filters=None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ) -> DataFrame:
        # 将 use_pandas_metadata 参数设置为 True，使得读取的数据包含 Pandas 元数据
        kwargs["use_pandas_metadata"] = True

        # 初始化空字典，用于存储将要传递给 to_pandas 方法的参数
        to_pandas_kwargs = {}
        
        # 根据 dtype_backend 的不同值，配置 types_mapper 参数
        if dtype_backend == "numpy_nullable":
            # 导入 _arrow_dtype_mapping 函数并调用，获取 Arrow 类型映射
            from pandas.io._util import _arrow_dtype_mapping
            mapping = _arrow_dtype_mapping()
            # 将获取的映射函数设置为 types_mapper
            to_pandas_kwargs["types_mapper"] = mapping.get
        elif dtype_backend == "pyarrow":
            # 将 pd.ArrowDtype 函数设置为 types_mapper，用于 PyArrow 数据类型映射
            to_pandas_kwargs["types_mapper"] = pd.ArrowDtype  # type: ignore[assignment]
        elif using_pyarrow_string_dtype():
            # 将 arrow_string_types_mapper 函数设置为 types_mapper，处理 PyArrow 字符串类型映射
            to_pandas_kwargs["types_mapper"] = arrow_string_types_mapper()

        # 调用 _get_path_or_handle 函数获取路径、处理器和文件系统对象
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path,
            filesystem,
            storage_options=storage_options,
            mode="rb",
        )
        try:
            # 使用 self.api.parquet.read_table 方法读取 Parquet 表数据
            pa_table = self.api.parquet.read_table(
                path_or_handle,
                columns=columns,
                filesystem=filesystem,
                filters=filters,
                **kwargs,
            )
            with catch_warnings():
                # 忽略特定警告类型，这里是对 "make_block is deprecated" 的警告
                filterwarnings(
                    "ignore",
                    "make_block is deprecated",
                    DeprecationWarning,
                )
                # 将 Parquet 表数据转换为 Pandas DataFrame，并传递额外的参数
                result = pa_table.to_pandas(**to_pandas_kwargs)

            # 如果 Parquet 表的 schema 包含 metadata 信息
            if pa_table.schema.metadata:
                # 检查 metadata 中是否包含特定键 "PANDAS_ATTRS"
                if b"PANDAS_ATTRS" in pa_table.schema.metadata:
                    # 将 metadata 中的 "PANDAS_ATTRS" 解析为 JSON 格式，并设置为 DataFrame 的 attrs 属性
                    df_metadata = pa_table.schema.metadata[b"PANDAS_ATTRS"]
                    result.attrs = json.loads(df_metadata)
            # 返回转换后的 Pandas DataFrame
            return result
        finally:
            # 如果 handles 不为空，则关闭它
            if handles is not None:
                handles.close()
class FastParquetImpl(BaseImpl):
    def __init__(self) -> None:
        # 导入 fastparquet 库作为可选依赖项
        # fastparquet 是 parquet 支持所需的库之一
        fastparquet = import_optional_dependency(
            "fastparquet", extra="fastparquet is required for parquet support."
        )
        self.api = fastparquet  # 将导入的 fastparquet 赋值给实例变量 self.api

    def write(
        self,
        df: DataFrame,
        path,
        compression: Literal["snappy", "gzip", "brotli"] | None = "snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ) -> None:
        self.validate_dataframe(df)  # 调用 validate_dataframe 方法验证 DataFrame 的有效性

        if "partition_on" in kwargs and partition_cols is not None:
            raise ValueError(
                "Cannot use both partition_on and "
                "partition_cols. Use partition_cols for partitioning data"
            )
        if "partition_on" in kwargs:
            partition_cols = kwargs.pop("partition_on")  # 如果 partition_on 在 kwargs 中，将其移除并赋给 partition_cols

        if partition_cols is not None:
            kwargs["file_scheme"] = "hive"  # 如果存在 partition_cols，则设置文件方案为 "hive"

        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )  # 抛出未实现的错误，因为 fastparquet 引擎不支持 filesystem 参数

        # 不能使用 get_handle，因为 write() 不接受文件缓冲区
        path = stringify_path(path)  # 调用 stringify_path 将 path 转换为字符串表示
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")  # 导入 fsspec 库作为可选依赖项

            # 如果 path 是 fsspec URL，使用 fsspec 打开文件，并设置为 'wb' 模式
            kwargs["open_with"] = lambda path, _: fsspec.open(
                path, "wb", **(storage_options or {})
            ).open()
        elif storage_options:
            raise ValueError(
                "storage_options passed with file object or non-fsspec file path"
            )  # 如果提供了 storage_options 参数但不是 fsspec 文件路径，则抛出 ValueError

        with catch_warnings(record=True):
            self.api.write(
                path,
                df,
                compression=compression,
                write_index=index,
                partition_on=partition_cols,
                **kwargs,
            )  # 使用 fastparquet 的 write 方法写入数据到指定路径

    def read(
        self,
        path,
        columns=None,
        filters=None,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ):
        # 略，read 方法未完整提供，无法添加注释
        ) -> DataFrame:
        # 定义函数签名，声明返回类型为 DataFrame
        
        parquet_kwargs: dict[str, Any] = {}
        # 初始化 Parquet 文件读取的关键字参数字典
        
        dtype_backend = kwargs.pop("dtype_backend", lib.no_default)
        # 从 kwargs 中弹出 dtype_backend 参数，若不存在则使用 lib.no_default
        
        # 我们在此禁用可空数据类型以提升 fastparquet 的性能，但需进一步讨论确认
        parquet_kwargs["pandas_nulls"] = False
        
        if dtype_backend is not lib.no_default:
            # 如果传入了不支持的 dtype_backend 参数，则抛出 ValueError 异常
            raise ValueError(
                "The 'dtype_backend' argument is not supported for the "
                "fastparquet engine"
            )
        
        if filesystem is not None:
            # 如果指定了 filesystem 参数，则抛出 NotImplementedError
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )
        
        path = stringify_path(path)
        # 将 path 转换为字符串形式
        
        handles = None
        # 初始化 handles 变量为 None
        
        if is_fsspec_url(path):
            # 如果 path 是一个 fsspec URL
            fsspec = import_optional_dependency("fsspec")
            
            # 使用 fsspec 打开 path 对应的文件对象，并设置到 parquet_kwargs 中的 fs 参数
            parquet_kwargs["fs"] = fsspec.open(path, "rb", **(storage_options or {})).fs
        
        elif isinstance(path, str) and not os.path.isdir(path):
            # 如果 path 是字符串且不是目录
            # 当我们确定它不是目录时，才使用 get_handle
            # fsspec 资源也可能指向目录，这个分支例如用于从非 fsspec URL 读取
            
            # 获取文件句柄并将其设置为 path
            handles = get_handle(
                path, "rb", is_text=False, storage_options=storage_options
            )
            path = handles.handle
        
        try:
            # 尝试创建 ParquetFile 对象读取 Parquet 文件
            parquet_file = self.api.ParquetFile(path, **parquet_kwargs)
            
            # 忽略特定警告信息
            with catch_warnings():
                filterwarnings(
                    "ignore",
                    "make_block is deprecated",
                    DeprecationWarning,
                )
                
                # 将 Parquet 文件内容转换为 pandas DataFrame 并返回
                return parquet_file.to_pandas(
                    columns=columns, filters=filters, **kwargs
                )
        
        finally:
            # 最终执行的清理操作
            if handles is not None:
                handles.close()
# 定义一个函数，将 DataFrame 写入 Parquet 格式
@doc(storage_options=_shared_docs["storage_options"])
def to_parquet(
    df: DataFrame,
    # 文件路径或写入缓冲区的对象，可以为 None，默认为 None
    path: FilePath | WriteBuffer[bytes] | None = None,
    # 使用的 Parquet 库，默认为 'auto'，自动选择 'pyarrow' 或 'fastparquet'
    engine: str = "auto",
    # 压缩算法，默认为 'snappy'，可选 'snappy', 'gzip', 'brotli', 'lz4', 'zstd', None
    compression: str | None = "snappy",
    # 是否包含索引，默认为 None，若为 True，将包含 DataFrame 的索引
    index: bool | None = None,
    # 存储选项，用于指定存储相关的配置
    storage_options: StorageOptions | None = None,
    # 分区列的列表，用于分区数据集
    partition_cols: list[str] | None = None,
    # 文件系统对象，用于读取 Parquet 文件
    filesystem: Any = None,
    **kwargs,
) -> bytes | None:
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, file-like object, or None, default None
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function. If None, the result is
        returned as bytes. If a string, it will be used as Root Directory path
        when writing a partitioned dataset. The engine fastparquet does not
        accept file-like objects.
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    compression : {{'snappy', 'gzip', 'brotli', 'lz4', 'zstd', None}},
        default 'snappy'. Name of the compression to use. Use ``None``
        for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    {storage_options}

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    kwargs
        Additional keyword arguments passed to the engine

    Returns
    -------
    bytes if no path argument is provided else None
    """
    # 如果 partition_cols 是字符串，则转换为包含该字符串的列表
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]
    # 根据指定的引擎获取实现对象
    impl = get_engine(engine)
    # 如果未提供路径，则创建一个 BytesIO 对象作为默认路径或缓冲区
    path_or_buf: FilePath | WriteBuffer[bytes] = io.BytesIO() if path is None else path
    
    # 调用特定的实现函数来写入 DataFrame 数据
    impl.write(
        df,
        path_or_buf,
        compression=compression,
        index=index,
        partition_cols=partition_cols,
        storage_options=storage_options,
        filesystem=filesystem,
        **kwargs,
    )
    
    # 如果未提供路径，则确保 path_or_buf 是 BytesIO 对象，并返回其内容的字节表示
    if path is None:
        assert isinstance(path_or_buf, io.BytesIO)
        return path_or_buf.getvalue()
    else:
        # 如果提供了路径，则返回 None
        return None
# 根据指定路径读取 Parquet 文件内容，并返回一个 DataFrame 对象
@doc(storage_options=_shared_docs["storage_options"])
def read_parquet(
    # 文件路径，可以是字符串、路径对象（实现了 os.PathLike[str] 接口）、或者具有二进制 read() 函数的文件对象
    path: FilePath | ReadBuffer[bytes],
    # 指定使用的 Parquet 库，默认为 'auto'，根据可用性选择 'pyarrow' 或 'fastparquet'
    engine: str = "auto",
    # 要读取的列名列表，如果为 None，则读取所有列
    columns: list[str] | None = None,
    # 存储选项，用于指定特定的存储设置，例如 compression 等
    storage_options: StorageOptions | None = None,
    # 指定结果 DataFrame 的数据类型后端，默认为 'numpy_nullable'
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    # 文件系统对象，用于读取 Parquet 文件，仅当 engine="pyarrow" 时有效
    filesystem: Any = None,
    # 筛选条件，用于筛选要读取的数据行
    filters: list[tuple] | list[list[tuple]] | None = None,
    # 其他关键字参数，传递给底层的 Parquet 读取引擎
    **kwargs,
) -> DataFrame:
    """
    从文件路径中加载 Parquet 对象，返回一个 DataFrame。

    该函数自动处理从 Parquet 文件中读取数据，并创建具有适当结构的 DataFrame。

    Parameters
    ----------
    path : str, path object or file-like object
        文件路径，可以是字符串、路径对象（实现了 os.PathLike[str] 接口）、或者具有二进制 read() 函数的文件对象。
        字符串可以是 URL。有效的 URL 方案包括 http、ftp、s3、gs 和 file。对于 file URL，需要一个主机名。
        本地文件可以是: "file://localhost/path/to/table.parquet"。
        文件 URL 也可以是指向包含多个分区 Parquet 文件的目录路径。pyarrow 和 fastparquet 都支持目录路径和文件 URL。
        目录路径可以是: "file://localhost/path/to/tables" 或 "s3://bucket/partition_dir"。
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet 库的选择。如果为 'auto'，则使用选项 "io.parquet.engine"。
        默认 "io.parquet.engine" 的行为是尝试使用 'pyarrow'，如果 'pyarrow' 不可用，则使用 'fastparquet'。
        
        当使用 'pyarrow' 引擎并且未提供存储选项且文件系统同时由 pyarrow.fs 和 fsspec 实现时（例如 "s3://"），则首先尝试使用 "pyarrow.fs" 文件系统。
        如果希望使用 fsspec 文件系统的实现，请使用带有已实例化的 fsspec 文件系统的 filesystem 关键字。
    columns : list, default=None
        如果不为 None，则只读取指定的列。
    {storage_options}
        存储选项，用于指定特定的存储设置，例如 compression 等。
        
        .. versionadded:: 1.3.0
    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        应用于结果 DataFrame 的后端数据类型（仍处于实验阶段）的选择。
        
        - "numpy_nullable": 返回支持可空 dtype 的 DataFrame（默认）。
        - "pyarrow": 返回基于 pyarrow 的可空 ArrowDtype DataFrame。
        
        .. versionadded:: 2.0
    filesystem : fsspec or pyarrow filesystem, default None
        用于读取 Parquet 文件的文件系统对象。仅在 engine="pyarrow" 时有效。
        
        .. versionadded:: 2.1.0
    """
    # 获取特定引擎的实现对象，该引擎由参数 `engine` 指定
    impl = get_engine(engine)
    # 检查指定的数据类型后端是否符合要求
    check_dtype_backend(dtype_backend)

    # 调用引擎对象的读取方法，读取指定路径下的数据
    return impl.read(
        path,  # 文件路径或数据源路径
        columns=columns,  # 需要读取的列名列表
        filters=filters,  # 数据过滤条件，用于选择性加载数据
        storage_options=storage_options,  # 存储选项，用于特定存储引擎的配置
        dtype_backend=dtype_backend,  # 数据类型后端，用于数据类型处理的特定选项
        filesystem=filesystem,  # 文件系统对象，用于特定文件系统的配置
        **kwargs,  # 其他传递给引擎的参数
    )
```