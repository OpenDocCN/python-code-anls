# `.\DB-GPT-src\dbgpt\storage\cache\storage\disk\disk_storage.py`

```py
"""Disk storage for cache.

Implement the cache storage using rocksdb.
"""
import logging
from typing import Optional

from rocksdict import Options, Rdict  # 导入 rocksdb 的 Options 和 Rdict 类

from dbgpt.core.interface.cache import (  # 导入缓存相关的接口和类型
    CacheConfig,
    CacheKey,
    CacheValue,
    K,
    RetrievalPolicy,
    V,
)

from ..base import CacheStorage, StorageItem  # 导入基础缓存存储和存储项

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 对象


def db_options(mem_table_buffer_mb: int = 256, background_threads: int = 2):
    """Create rocksdb options."""
    opt = Options()  # 创建 rocksdb 的 Options 对象
    opt.create_if_missing(True)  # 如果数据库不存在则创建
    opt.set_max_background_jobs(background_threads)  # 设置后台工作线程数
    opt.set_write_buffer_size(mem_table_buffer_mb * 1024 * 1024)  # 设置写缓冲区大小
    return opt  # 返回配置好的 Options 对象


class DiskCacheStorage(CacheStorage):
    """Disk cache storage using rocksdb."""

    def __init__(self, persist_dir: str, mem_table_buffer_mb: int = 256) -> None:
        """Create a new instance of DiskCacheStorage."""
        super().__init__()  # 调用父类的构造方法
        self.db: Rdict = Rdict(  # 使用 rocksdb 的 Rdict 类初始化数据库对象
            persist_dir, db_options(mem_table_buffer_mb=mem_table_buffer_mb)
        )

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
                    "DiskCacheStorage only supports 'EXACT_MATCH' retrieval policy"
                )
            return False
        return True

    def get(
        self, key: CacheKey[K], cache_config: Optional[CacheConfig] = None
    ) -> Optional[StorageItem]:
        """Retrieve a storage item from the cache using the provided key."""
        self.check_config(cache_config, raise_error=True)  # 检查缓存配置是否合法

        key_hash = key.get_hash_bytes()  # 获取键的哈希值字节表示
        item_bytes = self.db.get(key_hash)  # 从数据库中获取键对应的数据字节
        if not item_bytes:
            return None
        item = StorageItem.deserialize(item_bytes)  # 反序列化存储项
        logger.debug(f"Read file cache, key: {key}, storage item: {item}")  # 记录调试信息
        return item

    def set(
        self,
        key: CacheKey[K],
        value: CacheValue[V],
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Store a value in the cache using the provided key."""
        self.check_config(cache_config, raise_error=True)  # 检查缓存配置是否合法

        key_hash = key.get_hash_bytes()  # 获取键的哈希值字节表示
        item_bytes = value.serialize()  # 序列化值为字节表示
        self.db.put(key_hash, item_bytes)  # 将键值对存入数据库
        logger.debug(f"Write file cache, key: {key}, value: {value}")  # 记录调试信息
    ) -> None:
        """在缓存中设置给定键的值。"""
        # 使用提供的键和值构建一个 StorageItem 对象
        item = StorageItem.build_from_kv(key, value)
        # 计算键的哈希值
        key_hash = item.key_hash
        # 将序列化后的 StorageItem 对象存储在数据库中
        self.db[key_hash] = item.serialize()
        # 记录调试信息，显示缓存保存的键和值
        logger.debug(f"Save file cache, key: {key}, value: {value}")
```