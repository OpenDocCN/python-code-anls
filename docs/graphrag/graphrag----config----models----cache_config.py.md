# `.\graphrag\graphrag\config\models\cache_config.py`

```py
# Parameterization settings for the default configuration.
# This module defines configuration settings using Pydantic for handling cache-related parameters.

from pydantic import BaseModel, Field  # 引入 Pydantic 库中的 BaseModel 和 Field 类

import graphrag.config.defaults as defs  # 导入默认配置模块
from graphrag.config.enums import CacheType  # 导入枚举类型 CacheType

class CacheConfig(BaseModel):
    """The default configuration section for Cache."""
    # 定义缓存配置的基础模型 CacheConfig，继承自 Pydantic 的 BaseModel

    type: CacheType = Field(
        description="The cache type to use.", default=defs.CACHE_TYPE
    )
    # type 字段表示要使用的缓存类型，使用 Field 指定了其描述和默认值，默认值来自 defs.CACHE_TYPE

    base_dir: str = Field(
        description="The base directory for the cache.", default=defs.CACHE_BASE_DIR
    )
    # base_dir 字段表示缓存的基础目录，使用 Field 指定了其描述和默认值，默认值来自 defs.CACHE_BASE_DIR

    connection_string: str | None = Field(
        description="The cache connection string to use.", default=None
    )
    # connection_string 字段表示要使用的缓存连接字符串，使用 Field 指定了其描述和默认值，默认为 None

    container_name: str | None = Field(
        description="The cache container name to use.", default=None
    )
    # container_name 字段表示要使用的缓存容器名称，使用 Field 指定了其描述和默认值，默认为 None

    storage_account_blob_url: str | None = Field(
        description="The storage account blob url to use.", default=None
    )
    # storage_account_blob_url 字段表示要使用的存储账户 Blob URL，使用 Field 指定了其描述和默认值，默认为 None
```