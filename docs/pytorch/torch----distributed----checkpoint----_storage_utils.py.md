# `.\pytorch\torch\distributed\checkpoint\_storage_utils.py`

```
# 导入所需的模块
import os
from typing import List, Type, Union

# 从本地文件系统模块导入读写器类
from .filesystem import FileSystemReader, FileSystemWriter
# 从存储模块导入读写器类
from .storage import StorageReader, StorageWriter

# 定义私有函数 `_storage_setup`，用于设置存储器对象
def _storage_setup(
    storage: Union[StorageReader, StorageWriter, None],  # 参数 storage，可以是 StorageReader、StorageWriter 或 None
    checkpoint_id: Union[str, os.PathLike, None],  # 参数 checkpoint_id，可以是字符串、路径对象或 None
    reader: bool = False,  # 布尔型参数 reader，默认为 False
) -> Union[None, StorageReader, StorageWriter]:  # 函数返回值可以是 None、StorageReader 或 StorageWriter

    # 如果 storage 参数不为空
    if storage:
        # 如果 checkpoint_id 不为 None，则重置存储器对象的状态
        if checkpoint_id is not None:
            storage.reset(checkpoint_id)
        return storage  # 返回已配置的存储器对象

    # 如果 storage 参数为空，并且 checkpoint_id 也为空，则抛出运行时错误
    if not checkpoint_id:
        raise RuntimeError(
            "`checkpoint_id` must be specificed if "
            "storage_reader/storage_writer is None."
        )

    # 初始化目标存储器类型列表
    targets: List[Type[Union[StorageReader, StorageWriter]]] = []

    # 如果指定了 reader 参数为 True，则将文件系统读取器作为目标
    if reader:
        targets = [
            FileSystemReader,
        ]
    else:
        targets = [
            FileSystemWriter,
        ]

    # 尝试导入 Fsspec 的文件系统读写器类，并根据 reader 参数添加到目标列表中
    try:
        from ._fsspec_filesystem import FsspecReader, FsspecWriter

        targets.append(FsspecReader if reader else FsspecWriter)
    except Exception:
        pass  # 如果导入失败，则什么也不做继续执行

    # 遍历目标列表中的每个存储器类型类
    for target in targets:
        # 调用目标类的静态方法 validate_checkpoint_id 检查给定的 checkpoint_id 是否有效
        if target.validate_checkpoint_id(checkpoint_id):
            # 如果有效，则使用该存储器类型创建存储器对象，并重置其状态
            storage = target(checkpoint_id)  # type: ignore[call-arg]
            storage.reset(checkpoint_id)
            return storage  # 返回已配置的存储器对象

    # 如果无法确定使用哪个存储器读取器或写入器，则抛出运行时错误
    raise RuntimeError(
        "Cannot detect which StorageReader or StorageWriter to use. "
        "Please specify the storage_reader/storage_writer."
    )
```