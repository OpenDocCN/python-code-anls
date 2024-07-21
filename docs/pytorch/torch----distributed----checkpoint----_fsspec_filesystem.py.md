# `.\pytorch\torch\distributed\checkpoint\_fsspec_filesystem.py`

```py
# Mypy 将不会尝试推断任何已安装的第三方库的类型。
# mypy: ignore-errors

import io  # 导入用于处理 IO 操作的模块
import os  # 导入操作系统相关功能的模块
from contextlib import contextmanager  # 导入上下文管理器相关的功能
from pathlib import Path  # 导入处理路径的模块
from typing import Generator, Optional, Union  # 导入类型提示相关的功能

import fsspec  # 导入 fsspec 库，用于文件系统操作
from fsspec import AbstractFileSystem  # 导入抽象文件系统类
from fsspec.core import url_to_fs  # 导入将 URL 转换为文件系统对象的函数

from torch.distributed.checkpoint.filesystem import (  # 导入分布式检查点文件系统相关的功能
    FileSystemBase,  # 文件系统基类
    FileSystemReader,  # 文件系统读取器
    FileSystemWriter,  # 文件系统写入器
)

__all__ = [  # 定义模块对外公开的接口列表
    "FsspecWriter",  # 文件系统写入器
    "FsspecReader",  # 文件系统读取器（在代码中未定义）
]


class FileSystem(FileSystemBase):
    def __init__(self) -> None:
        self.fs: Optional[AbstractFileSystem] = None  # 初始化文件系统对象为 None

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        assert self.fs is not None  # 断言文件系统对象不为空
        with self.fs.transaction:  # 在文件系统事务中执行操作
            with fsspec.open(str(path), mode) as stream:  # 使用 fsspec 打开指定路径的文件流
                yield stream  # 返回文件流对象

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        return os.path.join(path, suffix)  # 将路径和后缀连接成完整的路径

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        self.fs, _ = url_to_fs(path)  # 初始化文件系统对象
        return path  # 返回路径本身

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        self.fs.rename(path, new_path)  # 重命名路径

    def mkdir(self, path: [str, os.PathLike]) -> None:
        self.fs.makedirs(path, exist_ok=True)  # 创建目录，如果目录已存在则忽略

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        if isinstance(checkpoint_id, Path):
            return False  # 如果检查点 ID 是路径对象，则返回 False

        try:
            url_to_fs(checkpoint_id)  # 尝试将检查点 ID 转换为文件系统对象
        except ValueError:
            return False  # 转换失败，返回 False

        return True  # 转换成功，返回 True

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return self.fs.exists(path)  # 检查路径是否存在

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        self.fs.rm(path)  # 删除文件


# TODO: add the dcp.async_save mixin
class FsspecWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using FFspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],  # 检查点路径，可以是字符串或路径对象
        single_file_per_rank: bool = True,  # 每个排名是否使用单个文件，默认为 True
        sync_files: bool = True,  # 是否同步文件，默认为 True
        thread_count: int = 1,  # 线程数，默认为 1
        per_thread_copy_ahead: int = 10_000_000,  # 每个线程的复制数据量，默认为 10,000,000
        overwrite: bool = True,  # 是否覆盖已存在文件，默认为 True

        path: Union[str, os.PathLike],  # 检查点路径，可以是字符串或路径对象
        single_file_per_rank: bool = True,  # 每个排名是否使用单个文件，默认为 True
        sync_files: bool = True,  # 是否同步文件，默认为 True
        thread_count: int = 1,  # 线程数，默认为 1
        per_thread_copy_ahead: int = 10_000_000,  # 每个线程的复制数据量，默认为 10,000,000
        overwrite: bool = True,  # 是否覆盖已存在文件，默认为 True



    ):
        super().__init__(path, single_file_per_rank, sync_files, thread_count)  # 调用父类构造函数初始化基本属性

    def _save(
        self, obj: object, filename: Union[str, os.PathLike], rank: Optional[int] = None
    ) -> None:
        if rank is not None and not self.single_file_per_rank:
            filename = f"{filename}.{rank}"  # 根据排名生成文件名

        with self.fs.transaction:  # 在文件系统事务中执行操作
            with fsspec.open(
                str(filename), "wb", **self._open_args
            ) as stream:  # 使用 fsspec 打开指定路径的二进制写入流
                self._save_stream(obj, stream)  # 调用保存数据到流的方法

    def _save_stream(self, obj: object, stream: io.IOBase) -> None:
        if self.sync_files:
            stream.write(self.serialize(obj))  # 将序列化后的数据写入流
            stream.flush()  # 刷新流，确保数据写入磁盘
        else:
            raise NotImplementedError("Asynchronous save is not yet implemented.")  # 异步保存功能尚未实现

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return self.fs.exists(path)  # 检查路径是否存在

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        self.fs.rm(path)  # 删除文件



class FsspecReader(FileSystemReader):
    """
    Basic implementation of StorageReader using FFspec.

    This implementation assumes:
    * Each rank saves data to a unique file or set of files.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],  # 检查点路径，可以是字符串或路径对象
        thread_count: int = 1,  # 线程数，默认为 1
    ):
        super().__init__(path, thread_count)  # 调用父类构造函数初始化基本属性

    def _read(
        self, filename: Union[str, os.PathLike], rank: Optional[int] = None
    ) -> object:
        if rank is not None:
            filename = f"{filename}.{rank}"  # 根据排名生成文件名

        with fsspec.open(
            str(filename), "rb", **self._open_args
        ) as stream:  # 使用 fsspec 打开指定路径的二进制读取流
            return self._read_stream(stream)  # 返回从流中读取的数据

    def _read_stream(self, stream: io.IOBase) -> object:
        return self.deserialize(stream.read())  # 从流中读取数据并反序列化

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return self.fs.exists(path)  # 检查路径是否存在

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        self.fs.rm(path)  # 删除文件
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        # 调用父类的初始化方法来设置写入器的基本参数
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            overwrite=overwrite,
        )
        # 创建一个文件系统对象用于处理路径
        self.fs = FileSystem()
        # 初始化路径并保存到实例属性中
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        # 使用文件系统的方法来验证检查点ID的有效性
        return FileSystem.validate_checkpoint_id(checkpoint_id)
# 定义一个继承自 FileSystemReader 的 FsspecReader 类，用于读取文件系统中的内容
class FsspecReader(FileSystemReader):
    # 初始化方法，接受路径参数（可以是字符串或 os.PathLike 对象）
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        # 调用父类的初始化方法
        super().__init__(path)
        # 创建一个文件系统对象
        self.fs = FileSystem()
        # 初始化路径属性，使用文件系统对象初始化给定的路径
        self.path = self.fs.init_path(path)

    # 类方法：验证检查点 ID 是否有效
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        # 调用 FileSystem 类的方法来验证给定的检查点 ID 是否有效
        return FileSystem.validate_checkpoint_id(checkpoint_id)
```