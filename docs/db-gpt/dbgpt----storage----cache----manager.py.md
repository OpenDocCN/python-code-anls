# `.\DB-GPT-src\dbgpt\storage\cache\manager.py`

```py
# 缓存管理器模块的导入和定义
"""Cache manager."""

import logging  # 导入日志记录模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from concurrent.futures import Executor  # 导入并发执行器接口
from typing import Optional, Type, cast  # 导入类型提示模块

# 从 dbgpt.component 模块中导入基础组件、组件类型和系统应用类
from dbgpt.component import BaseComponent, ComponentType, SystemApp
# 从 dbgpt.core 模块导入缓存配置、缓存键、缓存值、可序列化接口和序列化器类
from dbgpt.core import CacheConfig, CacheKey, CacheValue, Serializable, Serializer
# 从 dbgpt.core.interface.cache 模块导入泛型类型 K 和 V
from dbgpt.core.interface.cache import K, V
# 从 dbgpt.util.executor_utils 模块导入执行器工厂和将阻塞函数转换为异步函数的工具函数
from dbgpt.util.executor_utils import ExecutorFactory, blocking_func_to_async

# 从当前包中的 storage.base 模块导入缓存存储类 CacheStorage
from .storage.base import CacheStorage

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class CacheManager(BaseComponent, ABC):
    """The cache manager interface."""

    name = ComponentType.MODEL_CACHE_MANAGER  # 设置组件类型为模型缓存管理器

    def __init__(self, system_app: SystemApp | None = None):
        """Create cache manager."""
        super().__init__(system_app)

    def init_app(self, system_app: SystemApp):
        """Initialize cache manager."""
        self.system_app = system_app  # 初始化系统应用实例

    @abstractmethod
    async def set(
        self,
        key: CacheKey[K],
        value: CacheValue[V],
        cache_config: Optional[CacheConfig] = None,
    ):
        """Set cache with key."""  # 抽象方法：根据键设置缓存

    @abstractmethod
    async def get(
        self,
        key: CacheKey[K],
        cls: Type[Serializable],
        cache_config: Optional[CacheConfig] = None,
    ) -> Optional[CacheValue[V]]:
        """Retrieve cache with key."""  # 抽象方法：根据键检索缓存

    @property
    @abstractmethod
    def serializer(self) -> Serializer:
        """Return serializer to serialize/deserialize cache value."""  # 抽象属性：返回用于序列化/反序列化缓存值的序列化器


class LocalCacheManager(CacheManager):
    """Local cache manager."""

    def __init__(
        self, system_app: SystemApp, serializer: Serializer, storage: CacheStorage
    ) -> None:
        """Create local cache manager."""
        super().__init__(system_app)
        self._serializer = serializer  # 设置序列化器实例
        self._storage = storage  # 设置缓存存储实例

    @property
    def executor(self) -> Executor:
        """Return executor."""
        return self.system_app.get_component(  # 获取系统应用中的默认执行器工厂，创建执行器
            ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
        ).create()

    async def set(
        self,
        key: CacheKey[K],
        value: CacheValue[V],
        cache_config: Optional[CacheConfig] = None,
    ):
        """Set cache with key."""
        if self._storage.support_async():  # 如果缓存存储支持异步操作
            await self._storage.aset(key, value, cache_config)  # 使用异步方法设置缓存
        else:
            await blocking_func_to_async(  # 否则，将阻塞的设置缓存方法转换为异步执行
                self.executor, self._storage.set, key, value, cache_config
            )

    async def get(
        self,
        key: CacheKey[K],
        cls: Type[Serializable],
        cache_config: Optional[CacheConfig] = None,
    ) -> Optional[CacheValue[V]]:
        """Retrieve cache with key."""  # 使用键检索缓存数据
        # 实现依赖缓存存储方式的异步数据检索操作
        raise NotImplementedError("Method not implemented")
    ) -> Optional[CacheValue[V]]:
        """定义一个方法，用于从缓存中获取带有指定键的缓存值。

        如果缓存支持异步操作，则使用异步方法获取缓存值。
        否则，使用阻塞函数将同步获取的结果转换为异步结果。

        Args:
            key: 缓存键值。
            cache_config: 缓存配置对象。

        Returns:
            如果找到缓存值，则返回该值，否则返回 None。
        """
        if self._storage.support_async():
            # 如果存储支持异步操作，直接使用异步方法获取缓存值
            item_bytes = await self._storage.aget(key, cache_config)
        else:
            # 否则，使用阻塞函数将同步获取的缓存值转换为异步操作
            item_bytes = await blocking_func_to_async(
                self.executor, self._storage.get, key, cache_config
            )
        if not item_bytes:
            # 如果未获取到缓存值，则返回 None
            return None
        # 使用序列化器反序列化缓存值的字节数据，转换为指定类型的缓存值
        return cast(
            CacheValue[V], self._serializer.deserialize(item_bytes.value_data, cls)
        )

    @property
    def serializer(self) -> Serializer:
        """返回用于序列化和反序列化缓存值的序列化器对象。"""
        return self._serializer
def initialize_cache(
    system_app: SystemApp, storage_type: str, max_memory_mb: int, persist_dir: str
):
    """Initialize cache manager.

    Args:
        system_app (SystemApp): The system app instance to manage caching for.
        storage_type (str): Type of storage to be used ('disk' or other).
        max_memory_mb (int): Maximum memory allowance for caching in megabytes.
        persist_dir (str): Directory path for persistent storage if applicable.
    """
    from dbgpt.util.serialization.json_serialization import JsonSerializer

    from .storage.base import MemoryCacheStorage

    # Check if the specified storage type is 'disk'
    if storage_type == "disk":
        try:
            from .storage.disk.disk_storage import DiskCacheStorage

            # Attempt to instantiate DiskCacheStorage with specified parameters
            cache_storage: CacheStorage = DiskCacheStorage(
                persist_dir, mem_table_buffer_mb=max_memory_mb
            )
        except ImportError as e:
            # Log a warning and fall back to MemoryCacheStorage if DiskCacheStorage cannot be imported
            logger.warn(
                f"Can't import DiskCacheStorage, falling back to MemoryCacheStorage. Import error: {str(e)}"
            )
            cache_storage = MemoryCacheStorage(max_memory_mb=max_memory_mb)
    else:
        # Use MemoryCacheStorage if storage_type is not 'disk'
        cache_storage = MemoryCacheStorage(max_memory_mb=max_memory_mb)

    # Register LocalCacheManager with the system_app instance
    system_app.register(
        LocalCacheManager, serializer=JsonSerializer(), storage=cache_storage
    )
```