# `.\DB-GPT-src\dbgpt\storage\cache\storage\base.py`

```py
"""Base cache storage class."""
# 导入日志模块
import logging
# 导入抽象基类模块
from abc import ABC, abstractmethod
# 导入有序字典模块
from collections import OrderedDict
# 导入数据类模块
from dataclasses import dataclass
# 导入类型提示模块
from typing import Optional

# 导入消息包模块
import msgpack

# 导入缓存相关接口
from dbgpt.core.interface.cache import (
    CacheConfig,
    CacheKey,
    CachePolicy,
    CacheValue,
    K,
    RetrievalPolicy,
    V,
)
# 导入内存工具模块
from dbgpt.util.memory_utils import _get_object_bytes

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


@dataclass
class StorageItem:
    """A class representing a storage item.

    This class encapsulates data related to a storage item, such as its length,
    the hash of the key, and the data for both the key and value.

    Parameters:
        length (int): The bytes length of the storage item.
        key_hash (bytes): The hash value of the storage item's key.
        key_data (bytes): The data of the storage item's key, represented in bytes.
        value_data (bytes): The data of the storage item's value, also in bytes.
    """

    length: int  # The bytes length of the storage item
    key_hash: bytes  # The hash value of the storage item's key
    key_data: bytes  # The data of the storage item's key
    value_data: bytes  # The data of the storage item's value

    @staticmethod
    def build_from(
        key_hash: bytes, key_data: bytes, value_data: bytes
    ) -> "StorageItem":
        """Build a StorageItem from the provided key and value data."""
        # 计算存储项的长度，包括固定的32字节和各数据部分的字节长度
        length = (
            32
            + _get_object_bytes(key_hash)
            + _get_object_bytes(key_data)
            + _get_object_bytes(value_data)
        )
        # 返回创建的 StorageItem 对象
        return StorageItem(
            length=length, key_hash=key_hash, key_data=key_data, value_data=value_data
        )

    @staticmethod
    def build_from_kv(key: CacheKey[K], value: CacheValue[V]) -> "StorageItem":
        """Build a StorageItem from the provided key and value."""
        # 获取键的哈希值和序列化后的键数据，以及序列化后的值数据，然后调用 build_from 方法创建 StorageItem 对象
        key_hash = key.get_hash_bytes()
        key_data = key.serialize()
        value_data = value.serialize()
        return StorageItem.build_from(key_hash, key_data, value_data)

    def serialize(self) -> bytes:
        """Serialize the StorageItem into a byte stream using MessagePack.

        This method packs the object data into a dictionary, marking the
        key_data and value_data fields as raw binary data to avoid re-serialization.

        Returns:
            bytes: The serialized bytes.
        """
        # 将 StorageItem 对象序列化为 MessagePack 格式的字节流
        obj = {
            "length": self.length,
            "key_hash": msgpack.ExtType(1, self.key_hash),
            "key_data": msgpack.ExtType(2, self.key_data),
            "value_data": msgpack.ExtType(3, self.value_data),
        }
        return msgpack.packb(obj)

    @staticmethod
    def deserialize(data: bytes) -> "StorageItem":
        """Deserialize bytes back into a StorageItem using MessagePack.

        This extracts the fields from the MessagePack dict back into
        a StorageItem object.

        Args:
            data (bytes): Serialized bytes

        Returns:
            StorageItem: Deserialized StorageItem object.
        """
        # 使用 MessagePack 解析传入的字节数据，得到 Python 对象
        obj = msgpack.unpackb(data)
        # 从解析后的对象中提取 key_hash 字段的数据
        key_hash = obj["key_hash"].data
        # 从解析后的对象中提取 key_data 字段的数据
        key_data = obj["key_data"].data
        # 从解析后的对象中提取 value_data 字段的数据
        value_data = obj["value_data"].data

        # 创建并返回一个新的 StorageItem 对象，使用提取的字段作为参数
        return StorageItem(
            length=obj["length"],
            key_hash=key_hash,
            key_data=key_data,
            value_data=value_data,
        )
class CacheStorage(ABC):
    """Base class for cache storage."""

    @abstractmethod
    def check_config(
        self,
        cache_config: Optional[CacheConfig] = None,
        raise_error: Optional[bool] = True,
    ) -> bool:
        """Check whether the CacheConfig is legal.

        Args:
            cache_config (Optional[CacheConfig]): Cache config.
            raise_error (Optional[bool]): Whether raise error if illegal.

        Returns:
            bool: True if the cache configuration is legal, otherwise False.
            Raises ValueError if raise_error is True and config is illegal.
        """

    def support_async(self) -> bool:
        """Check whether the storage supports async operation.

        Returns:
            bool: Always returns False for base class, indicating no async support.
        """
        return False

    @abstractmethod
    def get(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> Optional[StorageItem]:
        """Retrieve a storage item from the cache using the provided key.

        Args:
            key (CacheKey[K]): The key to get cache
            cache_config (Optional[CacheConfig]): Cache config

        Returns:
            Optional[StorageItem]: The storage item retrieved according to key. If
                cache key does not exist, return None.
        """

    async def aget(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> Optional[StorageItem]:
        """Retrieve a storage item from the cache using the provided key asynchronously.

        Args:
            key (CacheKey[K]): The key to get cache
            cache_config (Optional[CacheConfig]): Cache config

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Returns:
            Optional[StorageItem]: The storage item retrieved according to key. If
                cache key does not exist, return None.
        """
        raise NotImplementedError

    @abstractmethod
    def set(
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

    async def aset(
        self,
        key: CacheKey[K],
        value: CacheValue[V],
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Set a value in the cache for the provided key asynchronously.

        Args:
            key (CacheKey[K]): The key to set to cache
            value (CacheValue[V]): The value to set to cache
            cache_config (Optional[CacheConfig]): Cache config

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class MemoryCacheStorage(CacheStorage):
    """A simple in-memory cache storage implementation."""
    # 初始化 MemoryCacheStorage 实例，设置最大内存限制并初始化当前内存使用量为 0
    def __init__(self, max_memory_mb: int = 256):
        """Create a new instance of MemoryCacheStorage."""
        self.cache: OrderedDict = OrderedDict()  # 使用有序字典作为缓存存储结构
        self.max_memory = max_memory_mb * 1024 * 1024  # 将最大内存限制从 MB 转换为字节
        self.current_memory_usage = 0  # 当前缓存使用的内存大小

    # 检查 CacheConfig 是否合法，可选是否抛出错误
    def check_config(
        self,
        cache_config: Optional[CacheConfig] = None,
        raise_error: Optional[bool] = True,
    ) -> bool:
        """Check whether the CacheConfig is legal."""
        if (
            cache_config
            and cache_config.retrieval_policy != RetrievalPolicy.EXACT_MATCH
        ):
            if raise_error:
                raise ValueError(
                    "MemoryCacheStorage only supports 'EXACT_MATCH' retrieval policy"
                )
            return False
        return True

    # 根据提供的键从缓存中检索存储项
    def get(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> Optional[StorageItem]:
        """Retrieve a storage item from the cache using the provided key."""
        self.check_config(cache_config, raise_error=True)
        # 计算键的哈希值
        key_hash = hash(key)
        item: Optional[StorageItem] = self.cache.get(key_hash)  # 获取缓存中的存储项
        logger.debug(f"MemoryCacheStorage get key {key}, hash {key_hash}, item: {item}")  # 记录调试信息

        if not item:
            return None
        # 将使用过的存储项移到有序字典的末尾，表示最近使用
        self.cache.move_to_end(key_hash)
        return item

    # 将键值对存入缓存中
    def set(
        self,
        key: CacheKey[K],
        value: CacheValue[V],
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Set a value in the cache for the provided key."""
        key_hash = hash(key)  # 计算键的哈希值
        item = StorageItem.build_from_kv(key, value)  # 使用键值构建存储项
        # 计算新条目的内存大小
        new_entry_size = _get_object_bytes(item)
        # 如果必要，根据缓存策略驱逐条目
        while self.current_memory_usage + new_entry_size > self.max_memory:
            self._apply_cache_policy(cache_config)

        # 将条目存入缓存中
        self.cache[key_hash] = item
        self.current_memory_usage += new_entry_size  # 更新当前内存使用量
        logger.debug(f"MemoryCacheStorage set key {key}, hash {key_hash}, item: {item}")  # 记录调试信息

    # 检查缓存中是否存在指定键的条目
    def exists(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> bool:
        """Check if the key exists in the cache."""
        return self.get(key, cache_config) is not None

    # 根据缓存策略（FIFO 或 LRU）移除最老或最新的条目
    def _apply_cache_policy(self, cache_config: Optional[CacheConfig] = None):
        # 根据缓存策略决定移除最老或最新的条目
        if cache_config and cache_config.cache_policy == CachePolicy.FIFO:
            self.cache.popitem(last=False)  # 移除最老的条目
        else:  # 默认使用 LRU 策略
            self.cache.popitem(last=True)  # 移除最新的条目
```