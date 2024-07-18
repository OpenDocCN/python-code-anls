# `.\graphrag\graphrag\index\verbs\graph\clustering\__init__.py`

```py
# 版权声明和许可信息，表明该模块代码版权归 2024 年微软公司所有，采用 MIT 许可证授权
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 模块的顶层注释，指出这是索引引擎的图聚类包的根目录
"""The Indexing Engine graph clustering package root."""

# 从当前目录导入模块 cluster_graph 中的 GraphCommunityStrategyType 和 cluster_graph 函数
from .cluster_graph import GraphCommunityStrategyType, cluster_graph

# 定义该模块的公开接口，即外部可以访问的对象列表，包括 GraphCommunityStrategyType 和 cluster_graph 函数
__all__ = ["GraphCommunityStrategyType", "cluster_graph"]
```