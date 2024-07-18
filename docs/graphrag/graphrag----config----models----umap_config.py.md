# `.\graphrag\graphrag\config\models\umap_config.py`

```py
# 版权声明及许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的库和模块
"""Parameterization settings for the default configuration."""
from pydantic import BaseModel, Field  # 导入基础模型和字段定义

# 导入默认配置模块
import graphrag.config.defaults as defs

# 定义 UmapConfig 类，用于配置 UMAP 相关参数
class UmapConfig(BaseModel):
    """Configuration section for UMAP."""

    # 是否启用 UMAP 的标志位，默认值为从默认配置中获取
    enabled: bool = Field(
        description="A flag indicating whether to enable UMAP.",  # 字段描述信息
        default=defs.UMAP_ENABLED,  # 默认值从默认配置中获取
    )
```