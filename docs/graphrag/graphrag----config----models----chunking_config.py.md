# `.\graphrag\graphrag\config\models\chunking_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 导入基于 Pydantic 的 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入默认配置模块
import graphrag.config.defaults as defs

# 定义 ChunkingConfig 类，继承自 BaseModel，用于配置分块处理相关参数
class ChunkingConfig(BaseModel):
    """Configuration section for chunking."""

    # 定义分块大小的配置项，使用 Field 来描述并设置默认值为 defs.CHUNK_SIZE
    size: int = Field(description="The chunk size to use.", default=defs.CHUNK_SIZE)
    
    # 定义分块重叠大小的配置项，使用 Field 来描述并设置默认值为 defs.CHUNK_OVERLAP
    overlap: int = Field(
        description="The chunk overlap to use.", default=defs.CHUNK_OVERLAP
    )
    
    # 定义分块依据的列的配置项，使用 Field 来描述并设置默认值为 defs.CHUNK_GROUP_BY_COLUMNS
    group_by_columns: list[str] = Field(
        description="The chunk by columns to use.",
        default=defs.CHUNK_GROUP_BY_COLUMNS,
    )
    
    # 定义分块策略的配置项，使用 Field 来描述并设置默认值为 None，可以是 dict 或 None 类型
    strategy: dict | None = Field(
        description="The chunk strategy to use, overriding the default tokenization strategy",
        default=None,
    )

    # 获取已解析的分块策略的方法
    def resolved_strategy(self) -> dict:
        """Get the resolved chunking strategy."""
        # 导入需要的模块和类
        from graphrag.index.verbs.text.chunk import ChunkStrategyType
        
        # 如果 strategy 已经设置，则直接返回；否则返回默认的分块策略字典
        return self.strategy or {
            "type": ChunkStrategyType.tokens,
            "chunk_size": self.size,
            "chunk_overlap": self.overlap,
            "group_by_columns": self.group_by_columns,
        }
```