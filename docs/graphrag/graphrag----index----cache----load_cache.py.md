# `.\graphrag\graphrag\index\cache\load_cache.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load_cache method definition."""

# 导入未来版本的类型注解
from __future__ import annotations

# 导入类型检查相关模块
from typing import TYPE_CHECKING, cast

# 导入枚举类型 CacheType
from graphrag.config.enums import CacheType

# 导入缓存配置相关模块
from graphrag.index.config.cache import (
    PipelineBlobCacheConfig,
    PipelineFileCacheConfig,
)

# 导入存储相关模块
from graphrag.index.storage import BlobPipelineStorage, FilePipelineStorage

# 如果是类型检查状态，导入 PipelineCacheConfig 类型
if TYPE_CHECKING:
    from graphrag.index.config import (
        PipelineCacheConfig,
    )

# 导入具体的缓存实现类
from .json_pipeline_cache import JsonPipelineCache
from .memory_pipeline_cache import create_memory_cache
from .noop_pipeline_cache import NoopPipelineCache


# 定义函数 load_cache，负责根据配置加载缓存
def load_cache(config: PipelineCacheConfig | None, root_dir: str | None):
    """Load the cache from the given config."""
    # 如果配置为空，则返回一个空的 NoopPipelineCache 实例
    if config is None:
        return NoopPipelineCache()

    # 根据配置的缓存类型进行匹配
    match config.type:
        # 如果缓存类型为 none，则返回一个空的 NoopPipelineCache 实例
        case CacheType.none:
            return NoopPipelineCache()
        # 如果缓存类型为 memory，则返回一个内存缓存实例
        case CacheType.memory:
            return create_memory_cache()
        # 如果缓存类型为 file，则创建文件存储并返回对应的 JsonPipelineCache 实例
        case CacheType.file:
            # 将 config 强制转换为 PipelineFileCacheConfig 类型
            config = cast(PipelineFileCacheConfig, config)
            # 创建文件存储，并设定基本目录
            storage = FilePipelineStorage(root_dir).child(config.base_dir)
            # 使用该存储创建 JsonPipelineCache 实例并返回
            return JsonPipelineCache(storage)
        # 如果缓存类型为 blob，则创建 Blob 存储并返回对应的 JsonPipelineCache 实例
        case CacheType.blob:
            # 将 config 强制转换为 PipelineBlobCacheConfig 类型
            config = cast(PipelineBlobCacheConfig, config)
            # 创建 Blob 存储，并设定相关连接信息和基本目录
            storage = BlobPipelineStorage(
                config.connection_string,
                config.container_name,
                storage_account_blob_url=config.storage_account_blob_url,
            ).child(config.base_dir)
            # 使用该存储创建 JsonPipelineCache 实例并返回
            return JsonPipelineCache(storage)
        # 如果缓存类型不在已知类型中，则抛出 ValueError 异常
        case _:
            msg = f"Unknown cache type: {config.type}"
            raise ValueError(msg)
```