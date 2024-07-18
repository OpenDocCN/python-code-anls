# `.\graphrag\graphrag\index\cache\memory_pipeline_cache.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'InMemoryCache' model."""

from typing import Any

from .pipeline_cache import PipelineCache


class InMemoryCache(PipelineCache):
    """In memory cache class definition."""

    _cache: dict[str, Any]  # 定义一个私有属性 _cache，用于存储缓存数据
    _name: str  # 定义一个私有属性 _name，用于存储缓存的名称

    def __init__(self, name: str | None = None):
        """Init method definition."""
        self._cache = {}  # 初始化 _cache 为空字典，用于存储缓存数据
        self._name = name or ""  # 初始化 _name，如果传入 name 参数为 None，则设为空字符串

    async def get(self, key: str) -> Any:
        """Get the value for the given key.

        Args:
            - key - The key to get the value for.
        
        Returns:
            - output - The value for the given key, or None if the key does not exist.
        """
        key = self._create_cache_key(key)  # 调用 _create_cache_key 方法生成实际使用的缓存键
        return self._cache.get(key)  # 返回给定键对应的值，如果不存在返回 None

    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Set the value for the given key.

        Args:
            - key - The key to set the value for.
            - value - The value to set.
        """
        key = self._create_cache_key(key)  # 调用 _create_cache_key 方法生成实际使用的缓存键
        self._cache[key] = value  # 设置给定键对应的值为传入的 value

    async def has(self, key: str) -> bool:
        """Return True if the given key exists in the storage.

        Args:
            - key - The key to check for.
        
        Returns:
            - output - True if the key exists in the storage, False otherwise.
        """
        key = self._create_cache_key(key)  # 调用 _create_cache_key 方法生成实际使用的缓存键
        return key in self._cache  # 返回该键是否存在于缓存中的布尔值

    async def delete(self, key: str) -> None:
        """Delete the given key from the storage.

        Args:
            - key - The key to delete.
        """
        key = self._create_cache_key(key)  # 调用 _create_cache_key 方法生成实际使用的缓存键
        del self._cache[key]  # 从缓存中删除指定键及其对应的值

    async def clear(self) -> None:
        """Clear the storage by emptying the cache."""
        self._cache.clear()  # 清空缓存，即清空 _cache 中的所有项

    def child(self, name: str) -> PipelineCache:
        """Create a sub cache with the given name."""
        return InMemoryCache(name)  # 创建一个名为 name 的子缓存，并返回对应的 InMemoryCache 实例

    def _create_cache_key(self, key: str) -> str:
        """Create a cache key for the given key.

        Args:
            - key - The original key to generate a cache key for.
        
        Returns:
            - output - The generated cache key combining the cache instance name and the original key.
        """
        return f"{self._name}{key}"  # 生成缓存键，将实例的 _name 和传入的 key 进行组合

def create_memory_cache() -> PipelineCache:
    """Create a memory cache instance.

    Returns:
        - output - An instance of InMemoryCache.
    """
    return InMemoryCache()  # 创建并返回一个 InMemoryCache 的实例作为内存缓存
```