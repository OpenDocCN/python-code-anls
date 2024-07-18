# `.\graphrag\graphrag\config\input_models\cluster_graph_config_input.py`

```py
# 导入必要的模块和类
"""Parameterization settings for the default configuration."""

# 从 typing_extensions 模块导入 NotRequired 和 TypedDict 类
from typing_extensions import NotRequired, TypedDict

# 定义一个新的 TypedDict 类型 ClusterGraphConfigInput，用于描述集群图的配置信息
class ClusterGraphConfigInput(TypedDict):
    """Configuration section for clustering graphs."""
    
    # 定义 max_cluster_size 键，其值可以是 int 或 None 类型，并且是可选的
    max_cluster_size: NotRequired[int | None]
    
    # 定义 strategy 键，其值可以是 dict 或 None 类型，并且是可选的
    strategy: NotRequired[dict | None]
```