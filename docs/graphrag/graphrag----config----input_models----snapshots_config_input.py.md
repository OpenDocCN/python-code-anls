# `.\graphrag\graphrag\config\input_models\snapshots_config_input.py`

```py
# 引入必要的模块和类型定义
"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired, TypedDict

# 定义一个 TypedDict 类型 SnapshotsConfigInput，用于描述快照配置的输入参数
class SnapshotsConfigInput(TypedDict):
    """Configuration section for snapshots."""

    # graphml 字段，可选的布尔值、字符串或空值
    graphml: NotRequired[bool | str | None]
    # raw_entities 字段，可选的布尔值、字符串或空值
    raw_entities: NotRequired[bool | str | None]
    # top_level_nodes 字段，可选的布尔值、字符串或空值
    top_level_nodes: NotRequired[bool | str | None]
```