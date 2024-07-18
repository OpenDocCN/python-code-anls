# `.\graphrag\graphrag\config\models\storage_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 导入必要的模块和库
from pydantic import BaseModel, Field

# 导入默认配置和枚举类型
import graphrag.config.defaults as defs
from graphrag.config.enums import StorageType

# 定义存储配置类，继承自BaseModel
class StorageConfig(BaseModel):
    """The default configuration section for Storage."""

    # 存储类型，使用StorageType枚举，使用默认值defs.STORAGE_TYPE
    type: StorageType = Field(
        description="The storage type to use.", default=defs.STORAGE_TYPE
    )
    # 存储的基础目录，使用默认值defs.STORAGE_BASE_DIR
    base_dir: str = Field(
        description="The base directory for the storage.",
        default=defs.STORAGE_BASE_DIR,
    )
    # 存储连接字符串，可以为None，默认值为None
    connection_string: str | None = Field(
        description="The storage connection string to use.", default=None
    )
    # 存储容器名称，可以为None，默认值为None
    container_name: str | None = Field(
        description="The storage container name to use.", default=None
    )
    # 存储账户的 Blob URL，可以为None，默认值为None
    storage_account_blob_url: str | None = Field(
        description="The storage account blob url to use.", default=None
    )
```