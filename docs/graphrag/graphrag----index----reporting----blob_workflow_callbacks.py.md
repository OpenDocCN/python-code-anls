# `.\graphrag\graphrag\index\reporting\blob_workflow_callbacks.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A reporter that writes to a blob storage."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from datashaper import NoopWorkflowCallbacks


class BlobWorkflowCallbacks(NoopWorkflowCallbacks):
    """A reporter that writes to a blob storage."""

    _blob_service_client: BlobServiceClient  # 声明 Azure Blob Service 客户端对象
    _container_name: str  # 声明存储容器名称
    _max_block_count: int = 25000  # 每个 Blob 的最大数据块数量设定为 25,000

    def __init__(
        self,
        connection_string: str | None,
        container_name: str,
        blob_name: str = "",
        base_dir: str | None = None,
        storage_account_blob_url: str | None = None,
    ):  # type: ignore
        """Create a new instance of the BlobStorageReporter class."""
        if container_name is None:
            # 如果未提供存储容器名称，则抛出值错误异常
            msg = "No container name provided for blob storage."
            raise ValueError(msg)
        if connection_string is None and storage_account_blob_url is None:
            # 如果未提供连接字符串或存储账户 Blob URL，则抛出值错误异常
            msg = "No storage account blob url provided for blob storage."
            raise ValueError(msg)
        
        # 初始化连接字符串和 Blob URL
        self._connection_string = connection_string
        self._storage_account_blob_url = storage_account_blob_url
        
        # 根据连接字符串或 Blob URL 初始化 BlobServiceClient 对象
        if self._connection_string:
            self._blob_service_client = BlobServiceClient.from_connection_string(
                self._connection_string
            )
        else:
            if storage_account_blob_url is None:
                # 如果未提供 Blob URL，则抛出值错误异常
                msg = "Either connection_string or storage_account_blob_url must be provided."
                raise ValueError(msg)

            self._blob_service_client = BlobServiceClient(
                storage_account_blob_url,
                credential=DefaultAzureCredential(),
            )
        
        # 如果未提供 Blob 名称，则默认生成以当前时间戳命名的日志文件名
        if blob_name == "":
            blob_name = f"report/{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')}.logs.json"

        # 设置 Blob 名称，包括基本目录
        self._blob_name = str(Path(base_dir or "") / blob_name)
        self._container_name = container_name
        
        # 获取 Blob 客户端对象，并创建 Blob 如果不存在
        self._blob_client = self._blob_service_client.get_blob_client(
            self._container_name, self._blob_name
        )
        if not self._blob_client.exists():
            self._blob_client.create_append_blob()

        self._num_blocks = 0  # 初始块计数器置零，用于跟踪已写入的数据块数量
    # 写入日志方法，接收一个字典作为日志内容
    def _write_log(self, log: dict[str, Any]):
        # 当块计数达到25k时，创建新文件
        if (
            self._num_blocks >= self._max_block_count
        ):  # 检查块计数是否超过25k
            # 重新初始化对象，创建一个新的日志文件
            self.__init__(
                self._connection_string,
                self._container_name,
                storage_account_blob_url=self._storage_account_blob_url,
            )

        # 获取 blob 客户端，用于写入日志数据
        blob_client = self._blob_service_client.get_blob_client(
            self._container_name, self._blob_name
        )
        # 向 blob 中追加写入日志内容（以 JSON 格式），并换行
        blob_client.append_block(json.dumps(log) + "\n")

        # 更新块计数
        self._num_blocks += 1

    # 报告错误的方法，接收错误消息、异常原因、堆栈信息和详细信息字典（可选）
    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ):
        """Report an error."""
        # 调用 _write_log 方法，记录错误日志
        self._write_log({
            "type": "error",
            "data": message,
            "cause": str(cause),
            "stack": stack,
            "details": details,
        })

    # 报告警告的方法，接收警告消息和详细信息字典（可选）
    def on_warning(self, message: str, details: dict | None = None):
        """Report a warning."""
        # 调用 _write_log 方法，记录警告日志
        self._write_log({"type": "warning", "data": message, "details": details})

    # 报告一般日志消息的方法，接收日志消息和详细信息字典（可选）
    def on_log(self, message: str, details: dict | None = None):
        """Report a generic log message."""
        # 调用 _write_log 方法，记录一般日志
        self._write_log({"type": "log", "data": message, "details": details})
```