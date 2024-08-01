# `.\DB-GPT-src\dbgpt\core\interface\cache.py`

```py
"""The cache interface.

The cache interface is used to cache LLM results and embedding results.

Maybe we can cache more server results in the future.
"""

from abc import ABC, abstractmethod               # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass                 # 导入数据类装饰器
from enum import Enum                             # 导入枚举类
from typing import Any, Generic, Optional, TypeVar # 导入类型提示相关的模块

from dbgpt.core.interface.serialization import Serializable  # 导入可序列化接口


K = TypeVar("K")   # 定义类型变量 K
V = TypeVar("V")   # 定义类型变量 V


class RetrievalPolicy(str, Enum):
    """The retrieval policy of the cache."""

    EXACT_MATCH = "exact_match"         # 精确匹配策略
    SIMILARITY_MATCH = "similarity_match"   # 相似度匹配策略


class CachePolicy(str, Enum):
    """The cache policy of the cache."""

    LRU = "lru"     # 最近最少使用策略
    FIFO = "fifo"   # 先进先出策略


@dataclass
class CacheConfig:
    """The cache config."""

    retrieval_policy: Optional[RetrievalPolicy] = RetrievalPolicy.EXACT_MATCH   # 缓存配置的检索策略，默认为精确匹配
    cache_policy: Optional[CachePolicy] = CachePolicy.LRU   # 缓存配置的缓存策略，默认为最近最少使用策略


class CacheKey(Serializable, ABC, Generic[K]):
    """The key of the cache. Must be hashable and comparable.

    Supported cache keys:
    - The LLM cache key: Include user prompt and the parameters to LLM.
    - The embedding model cache key: Include the texts to embedding and the parameters
    to embedding model.
    """

    @abstractmethod
    def __hash__(self) -> int:
        """Return the hash value of the key."""

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Check equality with another key."""

    @abstractmethod
    def get_hash_bytes(self) -> bytes:
        """Return the byte array of hash value."""

    @abstractmethod
    def get_value(self) -> K:
        """Get the underlying value of the cache key.

        Returns:
            K: The real object of current cache key
        """


class CacheValue(Serializable, ABC, Generic[V]):
    """Cache value abstract class."""

    @abstractmethod
    def get_value(self) -> V:
        """Get the underlying real value."""


class CacheClient(ABC, Generic[K, V]):
    """The cache client interface."""

    @abstractmethod
    async def get(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> Optional[CacheValue[V]]:
        """Retrieve a value from the cache using the provided key.

        Args:
            key (CacheKey[K]): The key to get cache
            cache_config (Optional[CacheConfig]): Cache config

        Returns:
            Optional[CacheValue[V]]: The value retrieved according to key. If cache key
                not exist, return None.
        """

    @abstractmethod
    async def set(
        self,
        key: CacheKey[K],
        value: CacheValue[V],
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Set a value in the cache for the provided key.

        Args:
            key (CacheKey[K]): The key to set to cache
            value (CacheValue[V]): The value to set to cache
            cache_config (Optional[CacheConfig]): Cache config
        """
    # 异步方法：检查缓存中是否存在指定的键
    async def exists(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> bool:
        """Check if a key exists in the cache.

        Args:
            key (CacheKey[K]): The key to set to cache
            cache_config (Optional[CacheConfig]): Cache config

        Return:
            bool: True if the key in the cache, otherwise is False
        """

    # 抽象方法：根据参数创建一个缓存键
    @abstractmethod
    def new_key(self, **kwargs) -> CacheKey[K]:
        """Create a cache key with params."""

    # 抽象方法：根据参数创建一个缓存值
    @abstractmethod
    def new_value(self, **kwargs) -> CacheValue[K]:
        """Create a cache key with params."""
```