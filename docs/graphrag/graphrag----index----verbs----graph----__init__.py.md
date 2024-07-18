# `.\graphrag\graphrag\index\verbs\graph\__init__.py`

```py
# 版权声明，2024年由Microsoft Corporation拥有
# 根据MIT许可证授权

"""索引引擎图包的根目录"""

# 从相对路径导入模块和函数
from .clustering import cluster_graph
from .compute_edge_combined_degree import compute_edge_combined_degree
from .create import DEFAULT_EDGE_ATTRIBUTES, DEFAULT_NODE_ATTRIBUTES, create_graph
from .embed import embed_graph
from .layout import layout_graph
from .merge import merge_graphs
from .report import (
    create_community_reports,
    prepare_community_reports,
    prepare_community_reports_claims,
    prepare_community_reports_edges,
    restore_community_hierarchy,
)
from .unpack import unpack_graph

# 声明该模块中可以导出的所有公共接口
__all__ = [
    "DEFAULT_EDGE_ATTRIBUTES",                    # 默认边属性
    "DEFAULT_NODE_ATTRIBUTES",                    # 默认节点属性
    "cluster_graph",                              # 图聚类函数
    "compute_edge_combined_degree",               # 计算边的组合度
    "create_community_reports",                   # 创建社区报告
    "create_graph",                               # 创建图函数
    "embed_graph",                                # 嵌入图函数
    "layout_graph",                               # 布局图函数
    "merge_graphs",                               # 合并图函数
    "prepare_community_reports",                  # 准备社区报告
    "prepare_community_reports_claims",           # 准备社区声明报告
    "prepare_community_reports_edges",            # 准备社区边缘报告
    "restore_community_hierarchy",                # 恢复社区层次结构函数
    "unpack_graph",                               # 解包图函数
]
```