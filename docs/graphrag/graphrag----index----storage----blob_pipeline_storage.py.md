# `.\graphrag\graphrag\index\storage\blob_pipeline_storage.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Azure Blob Storage implementation of PipelineStorage."""

import logging                      # 导入日志模块，用于记录日志信息
import re                           # 导入正则表达式模块，用于处理字符串匹配
from collections.abc import Iterator  # 从 collections.abc 模块导入迭代器抽象基类
from pathlib import Path           # 导入路径操作模块 Path
from typing import Any             # 导入 Any 类型提示

from azure.identity import DefaultAzureCredential  # 导入 Azure 身份验证模块
from azure.storage.blob import BlobServiceClient  # 导入 Azure 存储 Blob 服务客户端
from datashaper import Progress   # 导入进度条显示模块

from graphrag.index.progress import ProgressReporter  # 导入进度报告模块

from .typing import PipelineStorage  # 从当前包中导入 PipelineStorage 类型

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class BlobPipelineStorage(PipelineStorage):
    """The Blob-Storage implementation."""

    _connection_string: str | None  # 存储连接字符串或者为 None
    _container_name: str             # 存储容器名称
    _path_prefix: str                # 存储路径前缀
    _encoding: str                   # 编码方式，默认为 utf-8
    _storage_account_blob_url: str | None  # 存储账户 Blob URL 或者为 None

    def __init__(
        self,
        connection_string: str | None,
        container_name: str,
        encoding: str | None = None,
        path_prefix: str | None = None,
        storage_account_blob_url: str | None = None,
    ):
        """Create a new BlobStorage instance."""
        if connection_string:
            self._blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
        else:
            if storage_account_blob_url is None:
                msg = "Either connection_string or storage_account_blob_url must be provided."
                raise ValueError(msg)

            self._blob_service_client = BlobServiceClient(
                account_url=storage_account_blob_url,
                credential=DefaultAzureCredential(),
            )
        self._encoding = encoding or "utf-8"  # 设置编码方式，若未指定则使用默认 utf-8
        self._container_name = container_name  # 设置存储容器名称
        self._connection_string = connection_string  # 设置连接字符串
        self._path_prefix = path_prefix or ""  # 设置路径前缀，若未指定则为空字符串
        self._storage_account_blob_url = storage_account_blob_url  # 设置存储账户 Blob URL
        self._storage_account_name = (
            storage_account_blob_url.split("//")[1].split(".")[0]  # 提取存储账户名称
            if storage_account_blob_url
            else None
        )
        log.info(
            "creating blob storage at container=%s, path=%s",
            self._container_name,
            self._path_prefix,
        )
        self.create_container()  # 创建存储容器

    def create_container(self) -> None:
        """Create the container if it does not exist."""
        if not self.container_exists():  # 如果容器不存在
            container_name = self._container_name  # 获取容器名称
            container_names = [
                container.name
                for container in self._blob_service_client.list_containers()
            ]  # 获取已存在的容器名称列表
            if container_name not in container_names:  # 如果指定容器名不在已存在的容器列表中
                self._blob_service_client.create_container(container_name)  # 创建容器

    def delete_container(self) -> None:
        """Delete the container."""
        if self.container_exists():  # 如果容器存在
            self._blob_service_client.delete_container(self._container_name)  # 删除容器
    def container_exists(self) -> bool:
        """Check if the container exists."""
        # 获取容器名
        container_name = self._container_name
        # 获取所有容器的名称列表
        container_names = [
            container.name for container in self._blob_service_client.list_containers()
        ]
        # 判断容器名是否在容器名称列表中
        return container_name in container_names

    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        progress: ProgressReporter | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ):
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find blobs in a container using a file pattern, as well as a custom filter function.

        Params:
            base_dir: The name of the base container.
            file_pattern: The regular expression pattern object used to match blob names.
            file_filter: A dictionary specifying key-value pairs to filter the blobs.
            max_count: The maximum number of blobs to return. If -1, all blobs are returned.

        Returns
        -------
                An iterator yielding tuples of blob names and their corresponding regex matches.
        """
        base_dir = base_dir or ""

        log.info(
            "search container %s for files matching %s",
            self._container_name,
            file_pattern.pattern,
        )

        def blobname(blob_name: str) -> str:
            """Converts a blob name by removing the prefix and leading slash if present."""
            if blob_name.startswith(self._path_prefix):
                blob_name = blob_name.replace(self._path_prefix, "", 1)
            if blob_name.startswith("/"):
                blob_name = blob_name[1:]
            return blob_name

        def item_filter(item: dict[str, Any]) -> bool:
            """Checks if an item matches all filters specified in file_filter."""
            if file_filter is None:
                return True

            return all(re.match(value, item[key]) for key, value in file_filter.items())

        try:
            container_client = self._blob_service_client.get_container_client(
                self._container_name
            )
            all_blobs = list(container_client.list_blobs())

            num_loaded = 0
            num_total = len(list(all_blobs))
            num_filtered = 0
            for blob in all_blobs:
                match = file_pattern.match(blob.name)
                if match and blob.name.startswith(base_dir):
                    group = match.groupdict()
                    if item_filter(group):
                        yield (blobname(blob.name), group)
                        num_loaded += 1
                        if max_count > 0 and num_loaded >= max_count:
                            break
                    else:
                        num_filtered += 1
                else:
                    num_filtered += 1
                if progress is not None:
                    progress(
                        _create_progress_status(num_loaded, num_filtered, num_total)
                    )
        except Exception:
            log.exception(
                "Error finding blobs: base_dir=%s, file_pattern=%s, file_filter=%s",
                base_dir,
                file_pattern,
                file_filter,
            )
            raise

    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    async def get(self, key: str, as_bytes: bool = False, encoding: str | None = None) -> Any:
        """从缓存中获取一个值。"""
        try:
            # 根据传入的键名生成实际的存储键名
            key = self._keyname(key)
            # 获取容器客户端对象
            container_client = self._blob_service_client.get_container_client(
                self._container_name
            )
            # 获取具体 blob 客户端对象
            blob_client = container_client.get_blob_client(key)
            # 下载 blob 数据并读取全部内容
            blob_data = blob_client.download_blob().readall()
            # 如果不需要返回字节形式的数据，则尝试根据指定的编码解码为字符串
            if not as_bytes:
                coding = encoding or "utf-8"
                blob_data = blob_data.decode(coding)
        except Exception:
            # 如果发生异常，记录错误信息并返回 None
            log.exception("Error getting key %s", key)
            return None
        else:
            # 返回获取到的 blob 数据
            return blob_data

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """向缓存中设置一个值。"""
        try:
            # 根据传入的键名生成实际的存储键名
            key = self._keyname(key)
            # 获取容器客户端对象
            container_client = self._blob_service_client.get_container_client(
                self._container_name
            )
            # 获取具体 blob 客户端对象
            blob_client = container_client.get_blob_client(key)
            # 根据值的类型选择上传方式，如果是 bytes，则直接上传；否则根据指定的编码转换为 bytes 后上传
            if isinstance(value, bytes):
                blob_client.upload_blob(value, overwrite=True)
            else:
                coding = encoding or "utf-8"
                blob_client.upload_blob(value.encode(coding), overwrite=True)
        except Exception:
            # 如果发生异常，记录错误信息
            log.exception("Error setting key %s: %s", key)

    def set_df_json(self, key: str, dataframe: Any) -> None:
        """将一个 JSON 格式的 DataFrame 存储到缓存中。"""
        if self._connection_string is None and self._storage_account_name:
            # 如果没有指定连接字符串但指定了存储账户名，则使用默认 Azure 凭证进行存储
            dataframe.to_json(
                self._abfs_url(key),
                storage_options={
                    "account_name": self._storage_account_name,
                    "credential": DefaultAzureCredential(),
                },
                orient="records",
                lines=True,
                force_ascii=False,
            )
        else:
            # 否则使用指定的连接字符串进行存储
            dataframe.to_json(
                self._abfs_url(key),
                storage_options={"connection_string": self._connection_string},
                orient="records",
                lines=True,
                force_ascii=False,
            )

    def set_df_parquet(self, key: str, dataframe: Any) -> None:
        """将一个 Parquet 格式的 DataFrame 存储到缓存中。"""
        if self._connection_string is None and self._storage_account_name:
            # 如果没有指定连接字符串但指定了存储账户名，则使用默认 Azure 凭证进行存储
            dataframe.to_parquet(
                self._abfs_url(key),
                storage_options={
                    "account_name": self._storage_account_name,
                    "credential": DefaultAzureCredential(),
                },
            )
        else:
            # 否则使用指定的连接字符串进行存储
            dataframe.to_parquet(
                self._abfs_url(key),
                storage_options={"connection_string": self._connection_string},
            )
    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        # 构造实际存储的键名
        key = self._keyname(key)
        # 获取 Blob 容器客户端
        container_client = self._blob_service_client.get_container_client(
            self._container_name
        )
        # 获取指定键名对应的 Blob 客户端
        blob_client = container_client.get_blob_client(key)
        # 检查指定的 Blob 是否存在
        return blob_client.exists()

    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        # 构造实际存储的键名
        key = self._keyname(key)
        # 获取 Blob 容器客户端
        container_client = self._blob_service_client.get_container_client(
            self._container_name
        )
        # 获取指定键名对应的 Blob 客户端
        blob_client = container_client.get_blob_client(key)
        # 删除指定的 Blob
        blob_client.delete_blob()

    async def clear(self) -> None:
        """Clear the cache."""
        # 此方法尚未实现，留空

    def child(self, name: str | None) -> "PipelineStorage":
        """Create a child storage instance."""
        if name is None:
            return self
        # 构造新的路径
        path = str(Path(self._path_prefix) / name)
        # 返回一个新的 BlobPipelineStorage 实例
        return BlobPipelineStorage(
            self._connection_string,
            self._container_name,
            self._encoding,
            path,
            self._storage_account_blob_url,
        )

    def _keyname(self, key: str) -> str:
        """Get the key name."""
        # 返回完整的存储键名
        return str(Path(self._path_prefix) / key)

    def _abfs_url(self, key: str) -> str:
        """Get the ABFS URL."""
        # 构造 ABFS URL
        path = str(Path(self._container_name) / self._path_prefix / key)
        return f"abfs://{path}"
# 创建基于 Blob 的存储对象，并返回 PipelineStorage 对象
def create_blob_storage(
    connection_string: str | None,
    storage_account_blob_url: str | None,
    container_name: str,
    base_dir: str | None,
) -> PipelineStorage:
    """Create a blob based storage."""
    # 记录日志，指示正在创建 Blob 存储容器
    log.info("Creating blob storage at %s", container_name)
    
    # 检查容器名称是否为 None，若是则引发 ValueError 异常
    if container_name is None:
        msg = "No container name provided for blob storage."
        raise ValueError(msg)
    
    # 检查连接字符串和存储账户 Blob URL 是否都未提供，若是则引发 ValueError 异常
    if connection_string is None and storage_account_blob_url is None:
        msg = "No storage account blob url provided for blob storage."
        raise ValueError(msg)
    
    # 返回 BlobPipelineStorage 对象，用给定的参数初始化
    return BlobPipelineStorage(
        connection_string,
        container_name,
        path_prefix=base_dir,
        storage_account_blob_url=storage_account_blob_url,
    )


def validate_blob_container_name(container_name: str):
    """
    Check if the provided blob container name is valid based on Azure rules.

        - A blob container name must be between 3 and 63 characters in length.
        - Start with a letter or number
        - All letters used in blob container names must be lowercase.
        - Contain only letters, numbers, or the hyphen.
        - Consecutive hyphens are not permitted.
        - Cannot end with a hyphen.

    Args:
    -----
    container_name (str)
        The blob container name to be validated.

    Returns
    -------
        bool: True if valid, False otherwise.
    """
    # 检查容器名称的长度是否符合要求
    if len(container_name) < 3 or len(container_name) > 63:
        return ValueError(
            f"Container name must be between 3 and 63 characters in length. Name provided was {len(container_name)} characters long."
        )

    # 检查容器名称是否以字母或数字开头
    if not container_name[0].isalnum():
        return ValueError(
            f"Container name must start with a letter or number. Starting character was {container_name[0]}."
        )

    # 使用正则表达式检查容器名称是否仅包含小写字母、数字和连字符
    if not re.match("^[a-z0-9-]+$", container_name):
        return ValueError(
            f"Container name must only contain:\n- lowercase letters\n- numbers\n- or hyphens\nName provided was {container_name}."
        )

    # 检查容器名称中是否存在连续的连字符
    if "--" in container_name:
        return ValueError(
            f"Container name cannot contain consecutive hyphens. Name provided was {container_name}."
        )

    # 检查容器名称是否以连字符结尾
    if container_name[-1] == "-":
        return ValueError(
            f"Container name cannot end with a hyphen. Name provided was {container_name}."
        )

    # 如果通过所有验证条件，返回 True 表示容器名称有效
    return True


def _create_progress_status(
    num_loaded: int, num_filtered: int, num_total: int
) -> Progress:
    # 创建并返回进度对象，描述已加载和已过滤的文件数量
    return Progress(
        total_items=num_total,
        completed_items=num_loaded + num_filtered,
        description=f"{num_loaded} files loaded ({num_filtered} filtered)",
    )
```