# `.\graphrag\graphrag\index\storage\memory_pipeline_storage.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'InMemoryStorage' model."""

from typing import Any  # 导入类型提示模块 Any

from .file_pipeline_storage import FilePipelineStorage  # 导入自定义模块 FilePipelineStorage
from .typing import PipelineStorage  # 导入自定义模块 PipelineStorage


class MemoryPipelineStorage(FilePipelineStorage):  # 定义 MemoryPipelineStorage 类，继承自 FilePipelineStorage 类
    """In memory storage class definition."""

    _storage: dict[str, Any]  # 类属性 _storage，类型为字典，键为 str，值为 Any 类型

    def __init__(self):
        """Init method definition."""
        super().__init__(root_dir=".output")  # 调用父类初始化方法，设置 root_dir 为 ".output"
        self._storage = {}  # 初始化 _storage 为空字典

    async def get(
        self, key: str, as_bytes: bool | None = None, encoding: str | None = None
    ) -> Any:
        """Get the value for the given key.

        Args:
            - key - The key to get the value for.
            - as_bytes - Whether or not to return the value as bytes.

        Returns
        -------
            - output - The value for the given key.
        """
        return self._storage.get(key) or await super().get(key, as_bytes, encoding)
        # 返回 _storage 中对应 key 的值，如果不存在则调用父类的 get 方法获取值

    async def set(
        self, key: str, value: str | bytes | None, encoding: str | None = None
    ) -> None:
        """Set the value for the given key.

        Args:
            - key - The key to set the value for.
            - value - The value to set.
        """
        self._storage[key] = value  # 将给定 key 的值设置为指定的 value

    async def has(self, key: str) -> bool:
        """Return True if the given key exists in the storage.

        Args:
            - key - The key to check for.

        Returns
        -------
            - output - True if the key exists in the storage, False otherwise.
        """
        return key in self._storage or await super().has(key)
        # 返回 _storage 是否包含指定 key 的布尔值，如果不存在则调用父类的 has 方法检查

    async def delete(self, key: str) -> None:
        """Delete the given key from the storage.

        Args:
            - key - The key to delete.
        """
        del self._storage[key]  # 从 _storage 中删除指定 key 的条目

    async def clear(self) -> None:
        """Clear the storage."""
        self._storage.clear()  # 清空 _storage 中所有的条目

    def child(self, name: str | None) -> "PipelineStorage":
        """Create a child storage instance."""
        return self  # 返回当前对象作为子存储实例的模拟

def create_memory_storage() -> PipelineStorage:
    """Create memory storage."""
    return MemoryPipelineStorage()  # 返回 MemoryPipelineStorage 的实例作为内存存储
```