# `.\graphrag\graphrag\index\graph\utils\normalize_node_names.py`

```py
# 版权声明和许可证信息，指明代码的版权和许可条件
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入用于 HTML 解码的模块
import html

# 导入网络图库 NetworkX
import networkx as nx


# 定义函数 normalize_node_names，用于规范化图中节点的名称
def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    
    # 创建一个节点映射字典，将每个节点名转换为 HTML 解码后的大写形式并去除空白字符
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    
    # 使用节点映射字典重新标记图中的节点名称
    return nx.relabel_nodes(graph, node_mapping)
```