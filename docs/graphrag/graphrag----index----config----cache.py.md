# `.\graphrag\graphrag\index\config\cache.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'PipelineCacheConfig', 'PipelineFileCacheConfig' and 'PipelineMemoryCacheConfig' models."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from graphrag.config.enums import CacheType

T = TypeVar("T")

# 定义一个泛型模型 PipelineCacheConfig，表示管道的缓存配置
class PipelineCacheConfig(BaseModel, Generic[T]):
    """Represent the cache configuration for the pipeline."""
    
    type: T  # 缓存的类型

# 继承自 PipelineCacheConfig 的子类，表示文件缓存的配置
class PipelineFileCacheConfig(PipelineCacheConfig[Literal[CacheType.file]]):
    """Represent the file cache configuration for the pipeline."""
    
    type: Literal[CacheType.file] = CacheType.file  # 缓存的类型为文件类型
    
    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.", default=None
    )
    """The base directory for the cache."""

# 继承自 PipelineCacheConfig 的子类，表示内存缓存的配置
class PipelineMemoryCacheConfig(PipelineCacheConfig[Literal[CacheType.memory]]):
    """Represent the memory cache configuration for the pipeline."""
    
    type: Literal[CacheType.memory] = CacheType.memory  # 缓存的类型为内存类型

# 继承自 PipelineCacheConfig 的子类，表示无缓存的配置
class PipelineNoneCacheConfig(PipelineCacheConfig[Literal[CacheType.none]]):
    """Represent the none cache configuration for the pipeline."""
    
    type: Literal[CacheType.none] = CacheType.none  # 缓存的类型为无缓存类型

# 继承自 PipelineCacheConfig 的子类，表示 Blob 存储缓存的配置
class PipelineBlobCacheConfig(PipelineCacheConfig[Literal[CacheType.blob]]):
    """Represents the blob cache configuration for the pipeline."""
    
    type: Literal[CacheType.blob] = CacheType.blob  # 缓存的类型为 Blob 存储类型
    
    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.", default=None
    )
    """The base directory for the cache."""
    
    connection_string: str | None = pydantic_Field(
        description="The blob cache connection string for the cache.", default=None
    )
    """The blob cache connection string for the cache."""
    
    container_name: str = pydantic_Field(
        description="The container name for cache", default=None
    )
    """The container name for cache"""
    
    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for cache", default=None
    )
    """The storage account blob url for cache"""

# 定义一个联合类型，包括所有不同类型的 PipelineCacheConfig 子类
PipelineCacheConfigTypes = (
    PipelineFileCacheConfig
    | PipelineMemoryCacheConfig
    | PipelineBlobCacheConfig
    | PipelineNoneCacheConfig
)
```