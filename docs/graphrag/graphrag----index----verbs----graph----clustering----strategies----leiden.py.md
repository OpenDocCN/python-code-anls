# `.\graphrag\graphrag\index\verbs\graph\clustering\strategies\leiden.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and _compute_leiden_communities methods definitions."""

import logging
from typing import Any

import networkx as nx
from graspologic.partition import hierarchical_leiden

# 导入自定义模块中的函数
from graphrag.index.graph.utils import stable_largest_connected_component

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


def run(graph: nx.Graph, args: dict[str, Any]) -> dict[int, dict[str, list[str]]]:
    """Run method definition."""
    # 从参数中获取最大聚类大小，默认为10
    max_cluster_size = args.get("max_cluster_size", 10)
    # 从参数中获取是否使用最大连通组件标志，默认为True
    use_lcc = args.get("use_lcc", True)
    # 如果参数中设置了 verbose 为 True，则记录运行日志
    if args.get("verbose", False):
        log.info(
            "Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc
        )

    # 调用内部方法计算 Leiden 社区划分
    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    # 获取参数中的 levels，若未指定则使用所有的 levels
    levels = args.get("levels")

    # 若 levels 未指定，则使用所有的 level
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    # 存储不同层级下的社区划分结果
    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        # 将节点按社区划分整理到结果中
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """Return Leiden root communities."""
    # 如果指定使用最大连通组件，则进行处理
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    # 使用 hierarchical_leiden 函数计算 Leiden 社区划分
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    # 整理计算结果到指定格式的字典中
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results
```