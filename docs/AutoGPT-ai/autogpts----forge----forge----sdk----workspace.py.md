# `.\AutoGPT\autogpts\forge\forge\sdk\workspace.py`

```py
# 导入必要的模块
import abc
import os
import typing
from pathlib import Path

from google.cloud import storage

# 定义一个抽象基类 Workspace
class Workspace(abc.ABC):
    # 初始化方法，接受基础路径参数
    @abc.abstractclassmethod
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    # 抽象方法，用于读取数据
    @abc.abstractclassmethod
    def read(self, task_id: str, path: str) -> bytes:
        pass

    # 抽象方法，用于写入数据
    @abc.abstractclassmethod
    def write(self, task_id: str, path: str, data: bytes) -> None:
        pass

    # 抽象方法，用于删除数据
    @abc.abstractclassmethod
    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        pass

    # 抽象方法，用于检查数据是否存在
    @abc.abstractclassmethod
    def exists(self, task_id: str, path: str) -> bool:
        pass

    # 抽象方法，用于列出数据
    @abc.abstractclassmethod
    def list(self, task_id: str, path: str) -> typing.List[str]:
        pass

# 定义一个本地工作空间类，继承自 Workspace
class LocalWorkspace(Workspace):
    # 初始化方法，接受基础路径参数
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()

    # 内部方法，用于解析路径
    def _resolve_path(self, task_id: str, path: str) -> Path:
        path = str(path)
        path = path if not path.startswith("/") else path[1:]
        abs_path = (self.base_path / task_id / path).resolve()
        if not str(abs_path).startswith(str(self.base_path)):
            print("Error")
            raise ValueError(f"Directory traversal is not allowed! - {abs_path}")
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        return abs_path

    # 读取数据的方法
    def read(self, task_id: str, path: str) -> bytes:
        with open(self._resolve_path(task_id, path), "rb") as f:
            return f.read()

    # 写入数据的方法
    def write(self, task_id: str, path: str, data: bytes) -> None:
        file_path = self._resolve_path(task_id, path)
        with open(file_path, "wb") as f:
            f.write(data)

    # 删除数据的方法
    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    # 删除指定路径的文件或目录
    def delete(self, task_id: str, path: str, directory: bool = False, recursive: bool = False) -> None:
        # 构建完整的路径
        path = self.base_path / task_id / path
        # 解析路径
        resolved_path = self._resolve_path(task_id, path)
        # 如果是目录
        if directory:
            # 如果需要递归删除
            if recursive:
                # 递归删除目录
                os.rmdir(resolved_path)
            else:
                # 删除目录及其父目录，直到遇到非空目录
                os.removedirs(resolved_path)
        else:
            # 删除文件
            os.remove(resolved_path)

    # 检查指定路径是否存在
    def exists(self, task_id: str, path: str) -> bool:
        # 构建完整的路径
        path = self.base_path / task_id / path
        # 返回解析后的路径是否存在
        return self._resolve_path(task_id, path).exists()

    # 列出指定路径下的文件和目录
    def list(self, task_id: str, path: str) -> typing.List[str]:
        # 构建完整的路径
        path = self.base_path / task_id / path
        # 解析路径
        base = self._resolve_path(task_id, path)
        # 如果路径不存在或者不是目录，则返回空列表
        if not base.exists() or not base.is_dir():
            return []
        # 返回相对于基本路径的所有文件和目录的列表
        return [str(p.relative_to(self.base_path / task_id)) for p in base.iterdir()]
# 定义一个 GCSWorkspace 类，继承自 Workspace 类
class GCSWorkspace(Workspace):
    # 初始化方法，接受一个 bucket_name 字符串和一个可选的 base_path 字符串作为参数
    def __init__(self, bucket_name: str, base_path: str = ""):
        # 将传入的 bucket_name 赋值给实例变量 bucket_name
        self.bucket_name = bucket_name
        # 如果传入了 base_path，则将其转换为 Path 对象并解析路径，赋值给实例变量 base_path；否则为空字符串
        self.base_path = Path(base_path).resolve() if base_path else ""
        # 创建一个 Google Cloud Storage 客户端对象
        self.storage_client = storage.Client()
        # 获取指定名称的存储桶对象
        self.bucket = self.storage_client.get_bucket(self.bucket_name)

    # 内部方法，用于解析路径，接受一个 task_id 字符串和一个 path 字符串作为参数，返回一个 Path 对象
    def _resolve_path(self, task_id: str, path: str) -> Path:
        # 将 path 转换为字符串
        path = str(path)
        # 如果 path 以斜杠开头，则去掉开头的斜杠
        path = path if not path.startswith("/") else path[1:]
        # 将 base_path、task_id 和 path 拼接成绝对路径，并解析路径
        abs_path = (self.base_path / task_id / path).resolve()
        # 如果绝对路径不以 base_path 开头，则抛出异常
        if not str(abs_path).startswith(str(self.base_path)):
            print("Error")
            raise ValueError(f"Directory traversal is not allowed! - {abs_path}")
        return abs_path

    # 读取指定任务和路径下的数据，接受一个 task_id 字符串和一个 path 字符串作为参数，返回一个字节串
    def read(self, task_id: str, path: str) -> bytes:
        # 获取指定路径的 Blob 对象
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        # 如果 Blob 对象不存在，则抛出文件未找到异常
        if not blob.exists():
            raise FileNotFoundError()
        # 下载 Blob 对象的内容并返回字节串
        return blob.download_as_bytes()

    # 写入数据到指定任务和路径下，接受一个 task_id 字符串、一个 path 字符串和一个 data 字节串作为参数，无返回值
    def write(self, task_id: str, path: str, data: bytes) -> None:
        # 获取指定路径的 Blob 对象
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        # 从字节串中上传数据到 Blob 对象
        blob.upload_from_string(data)

    # 删除指定任务和路径下的数据，接受一个 task_id 字符串、一个 path 字符串和两个可选的布尔值参数 directory 和 recursive，无返回值
    def delete(self, task_id: str, path: str, directory=False, recursive=False):
        # 如果要删除目录但未设置递归标志，则抛出异常
        if directory and not recursive:
            raise ValueError("recursive must be True when deleting a directory")
        # 获取指定路径的 Blob 对象
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        # 如果 Blob 对象不存在，则直接返回
        if not blob.exists():
            return
        # 如果要删除目录，则遍历目录下的所有 Blob 对象并删除；否则直接删除指定 Blob 对象
        if directory:
            for b in list(self.bucket.list_blobs(prefix=blob.name)):
                b.delete()
        else:
            blob.delete()

    # 检查指定任务和路径下的数据是否存在，接受一个 task_id 字符串和一个 path 字符串作为参数，返回一个布尔值
    def exists(self, task_id: str, path: str) -> bool:
        # 获取指定路径的 Blob 对象
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        # 返回 Blob 对象是否存在的布尔值
        return blob.exists()
    # 定义一个方法，用于列出指定路径下的所有文件名
    def list(self, task_id: str, path: str) -> typing.List[str]:
        # 拼接路径前缀，将路径中的反斜杠替换为斜杠，并添加"/"作为结尾
        prefix = os.path.join(task_id, self.base_path, path).replace("\\", "/") + "/"
        # 获取指定前缀下的所有对象列表
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        # 遍历对象列表，获取每个对象的相对路径，并添加到列表中
        return [str(Path(b.name).relative_to(prefix[:-1])) for b in blobs]
```