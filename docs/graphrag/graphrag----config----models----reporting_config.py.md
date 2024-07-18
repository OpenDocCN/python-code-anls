# `.\graphrag\graphrag\config\models\reporting_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 导入必要的模块和库
from pydantic import BaseModel, Field

# 导入默认配置和枚举类型
import graphrag.config.defaults as defs
from graphrag.config.enums import ReportingType

# 定义报告配置类，继承自BaseModel
class ReportingConfig(BaseModel):
    """The default configuration section for Reporting."""

    # 报告类型，使用Field指定了描述和默认值
    type: ReportingType = Field(
        description="The reporting type to use.", default=defs.REPORTING_TYPE
    )
    # 报告基础目录，使用Field指定了描述和默认值
    base_dir: str = Field(
        description="The base directory for reporting.",
        default=defs.REPORTING_BASE_DIR,
    )
    # 报告连接字符串，可以为空，使用Field指定了描述和默认值
    connection_string: str | None = Field(
        description="The reporting connection string to use.", default=None
    )
    # 报告容器名称，可以为空，使用Field指定了描述和默认值
    container_name: str | None = Field(
        description="The reporting container name to use.", default=None
    )
    # 存储账户 Blob URL，可以为空，使用Field指定了描述和默认值
    storage_account_blob_url: str | None = Field(
        description="The storage account blob url to use.", default=None
    )
```