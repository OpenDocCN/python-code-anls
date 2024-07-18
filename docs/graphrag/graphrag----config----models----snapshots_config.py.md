# `.\graphrag\graphrag\config\models\snapshots_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 导入必要的库和模块
from pydantic import BaseModel, Field

# 导入默认配置模块
import graphrag.config.defaults as defs

# 定义快照配置类，继承自基础模型BaseModel
class SnapshotsConfig(BaseModel):
    """Configuration section for snapshots."""

    # 是否对GraphML进行快照的标志，使用默认值defs.SNAPSHOTS_GRAPHML
    graphml: bool = Field(
        description="A flag indicating whether to take snapshots of GraphML.",
        default=defs.SNAPSHOTS_GRAPHML,
    )
    # 是否对原始实体进行快照的标志，使用默认值defs.SNAPSHOTS_RAW_ENTITIES
    raw_entities: bool = Field(
        description="A flag indicating whether to take snapshots of raw entities.",
        default=defs.SNAPSHOTS_RAW_ENTITIES,
    )
    # 是否对顶层节点进行快照的标志，使用默认值defs.SNAPSHOTS_TOP_LEVEL_NODES
    top_level_nodes: bool = Field(
        description="A flag indicating whether to take snapshots of top-level nodes.",
        default=defs.SNAPSHOTS_TOP_LEVEL_NODES,
    )
```