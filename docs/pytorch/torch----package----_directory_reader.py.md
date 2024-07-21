# `.\pytorch\torch\package\_directory_reader.py`

```py
# mypy: allow-untyped-defs
# 引入操作系统路径模块和文件通配模块
import os.path
from glob import glob
# 引入类型提示相关模块
from typing import cast

# 引入 PyTorch 模块
import torch
from torch.types import Storage

# 指定序列化 ID 记录的文件名
__serialization_id_record_name__ = ".data/serialization_id"


# 定义一个类，用于包装具有存储功能的对象
class _HasStorage:
    def __init__(self, storage):
        self._storage = storage

    # 返回对象的存储
    def storage(self):
        return self._storage


# 定义一个目录读取器类，用于在未压缩的包中操作
class DirectoryReader:
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        # 初始化目录读取器，指定工作目录
        self.directory = directory

    # 根据名称获取记录的内容
    def get_record(self, name):
        filename = f"{self.directory}/{name}"
        with open(filename, "rb") as f:
            return f.read()

    # 根据记录名称、元素数目和数据类型获取存储对象
    def get_storage_from_record(self, name, numel, dtype):
        filename = f"{self.directory}/{name}"
        # 计算存储字节大小
        nbytes = torch._utils._element_size(dtype) * numel
        # 从文件中读取存储对象，并将其转换为指定类型的存储
        storage = cast(Storage, torch.UntypedStorage)
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))

    # 检查是否存在指定路径的记录
    def has_record(self, path):
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    # 获取目录中所有的记录文件名列表
    def get_all_records(
        self,
    ):
        files = []
        # 使用通配符获取目录下所有文件（包括子目录）
        for filename in glob(f"{self.directory}/**", recursive=True):
            if not os.path.isdir(filename):  # 排除子目录
                # 添加相对路径到文件名列表中
                files.append(filename[len(self.directory) + 1 :])
        return files

    # 获取序列化 ID 记录的内容
    def serialization_id(
        self,
    ):
        if self.has_record(__serialization_id_record_name__):
            # 如果存在序列化 ID 记录，返回其内容
            return self.get_record(__serialization_id_record_name__)
        else:
            # 否则返回空字符串
            return ""
```