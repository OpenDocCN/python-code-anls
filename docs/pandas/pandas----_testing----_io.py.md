# `D:\src\scipysrc\pandas\pandas\_testing\_io.py`

```
# 导入必要的模块和函数
from __future__ import annotations  # 导入用于支持类型注解的特性

import gzip  # 导入 gzip 模块，用于处理 gzip 压缩文件
import io  # 导入 io 模块，提供核心的 IO 功能
import pathlib  # 导入 pathlib 模块，用于处理文件路径
import tarfile  # 导入 tarfile 模块，用于处理 tar 压缩文件
from typing import (  # 导入类型提示相关的模块和类
    TYPE_CHECKING,
    Any,
)
import uuid  # 导入 uuid 模块，用于生成唯一标识符
import zipfile  # 导入 zipfile 模块，用于处理 zip 压缩文件

from pandas.compat._optional import import_optional_dependency  # 导入辅助函数

import pandas as pd  # 导入 pandas 库，并使用 pd 作为别名
from pandas._testing.contexts import ensure_clean  # 导入上下文管理函数

if TYPE_CHECKING:
    from collections.abc import Callable  # 导入 Callable 类型提示

    from pandas._typing import (  # 导入 pandas 库中的类型提示
        FilePath,
        ReadPickleBuffer,
    )

    from pandas import (  # 导入 pandas 库中的 DataFrame 和 Series 类型
        DataFrame,
        Series,
    )

# ------------------------------------------------------------------
# File-IO

# 定义一个函数，将对象序列化到 pickle 文件并读取回来
def round_trip_pickle(
    obj: Any, path: FilePath | ReadPickleBuffer | None = None
) -> DataFrame | Series:
    """
    Pickle an object and then read it again.

    Parameters
    ----------
    obj : any object
        The object to pickle and then re-read.
    path : str, path object or file-like object, default None
        The path where the pickled object is written and then read.

    Returns
    -------
    pandas object
        The original object that was pickled and then re-read.
    """
    _path = path  # 保存传入的路径参数
    if _path is None:
        _path = f"__{uuid.uuid4()}__.pickle"  # 如果路径为空，使用随机生成的 pickle 文件名
    with ensure_clean(_path) as temp_path:  # 使用上下文管理确保文件操作后清理临时文件
        pd.to_pickle(obj, temp_path)  # 将对象序列化到指定路径
        return pd.read_pickle(temp_path)  # 读取并返回序列化后的对象

# 定义一个函数，使用 pathlib.Path 将对象写入文件并读取回来
def round_trip_pathlib(writer, reader, path: str | None = None):
    """
    Write an object to file specified by a pathlib.Path and read it back

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
    Path = pathlib.Path  # 设置别名 Path 来引用 pathlib.Path 类
    if path is None:
        path = "___pathlib___"  # 如果路径为空，使用默认路径名
    with ensure_clean(path) as path:  # 使用上下文管理确保文件操作后清理临时文件
        writer(Path(path))  # 调用写入函数将对象写入指定路径
        obj = reader(Path(path))  # 调用读取函数读取指定路径的对象
    return obj  # 返回读取的对象

# 定义一个函数，将数据写入压缩文件中
def write_to_compressed(compression, path, data, dest: str = "test") -> None:
    """
    Write data to a compressed file.

    Parameters
    ----------
    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd'}
        The compression type to use.
    path : str
        The file path to write the data.
    data : str
        The data to write.
    dest : str, default "test"
        The destination file (for ZIP only)

    Raises
    ------
    ValueError : An invalid compression value was passed in.
    """
    args: tuple[Any, ...] = (data,)  # 初始化参数元组，包含待写入的数据
    mode = "wb"  # 设置默认写入模式为二进制
    method = "write"  # 设置默认的写入方法为写入字符串
    compress_method: Callable  # 声明压缩方法的类型提示变量

    if compression == "zip":
        compress_method = zipfile.ZipFile  # 如果压缩类型是 zip，则使用 zipfile.ZipFile 类
        mode = "w"  # 设置 zip 文件写入模式为写入
        args = (dest, data)  # 设置参数元组为目标文件名和待写入数据
        method = "writestr"  # 设置写入方法为写入字符串操作
    # 如果压缩类型为 "tar"，选择 tarfile 库进行操作
    elif compression == "tar":
        # 设定压缩方法为 TarFile
        compress_method = tarfile.TarFile
        # 设定模式为写入模式
        mode = "w"
        # 创建一个 TarInfo 对象，表示要写入的文件信息
        file = tarfile.TarInfo(name=dest)
        # 将数据封装为字节流对象
        bytes = io.BytesIO(data)
        # 设置文件大小为数据的长度
        file.size = len(data)
        # 准备参数，用于后续调用压缩方法
        args = (file, bytes)
        # 设定方法名称为 "addfile"，表示添加文件到 tar 压缩包中
        method = "addfile"
    
    # 如果压缩类型为 "gzip"，选择 gzip 库进行操作
    elif compression == "gzip":
        # 设定压缩方法为 GzipFile
        compress_method = gzip.GzipFile
    
    # 如果压缩类型为 "bz2"，导入 bz2 库进行操作
    elif compression == "bz2":
        import bz2
        # 设定压缩方法为 BZ2File
        compress_method = bz2.BZ2File
    
    # 如果压缩类型为 "zstd"，使用 import_optional_dependency 函数导入 zstandard 库，并使用 open 方法
    elif compression == "zstd":
        # 设定压缩方法为 zstandard 库的 open 方法
        compress_method = import_optional_dependency("zstandard").open
    
    # 如果压缩类型为 "xz"，导入 lzma 库进行操作
    elif compression == "xz":
        import lzma
        # 设定压缩方法为 LZMAFile
        compress_method = lzma.LZMAFile
    
    # 如果压缩类型不在已知类型中，则抛出 ValueError 异常
    else:
        raise ValueError(f"Unrecognized compression type: {compression}")
    
    # 使用选定的压缩方法和参数进行文件压缩操作
    with compress_method(path, mode=mode) as f:
        # 利用 getattr 函数动态调用压缩方法对象的 method 方法，并传入 args 参数
        getattr(f, method)(*args)
```