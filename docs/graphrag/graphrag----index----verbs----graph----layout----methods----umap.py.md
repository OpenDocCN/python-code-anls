# `.\graphrag\graphrag\index\verbs\graph\layout\methods\umap.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and _create_node_position methods definitions."""

# 导入日志和异常追踪模块
import logging
import traceback
# 导入类型提示相关模块
from typing import Any

# 导入第三方库
import networkx as nx
import numpy as np

# 导入本地模块
from graphrag.index.graph.visualization import (
    GraphLayout,
    NodePosition,
    compute_umap_positions,
)
from graphrag.index.typing import ErrorHandlerFn
from graphrag.index.verbs.graph.embed.typing import NodeEmbeddings

# 设置日志记录器
log = logging.getLogger(__name__)


def run(
    graph: nx.Graph,
    embeddings: NodeEmbeddings,
    args: dict[str, Any],
    on_error: ErrorHandlerFn,
) -> GraphLayout:
    """Run method definition."""
    # 初始化节点聚类和节点大小的空列表
    node_clusters = []
    node_sizes = []

    # 过滤原始嵌入，去除空值
    embeddings = _filter_raw_embeddings(embeddings)
    # 提取节点列表和其对应的嵌入向量
    nodes = list(embeddings.keys())
    embedding_vectors = [embeddings[node_id] for node_id in nodes]

    # 遍历节点列表，获取节点的聚类和大小信息
    for node_id in nodes:
        node = graph.nodes[node_id]
        # 获取节点的聚类信息，默认为-1
        cluster = node.get("cluster", node.get("community", -1))
        node_clusters.append(cluster)
        # 获取节点的大小信息，默认为0
        size = node.get("degree", node.get("size", 0))
        node_sizes.append(size)

    # 初始化额外的参数字典
    additional_args = {}
    # 如果节点聚类列表不为空，添加到额外参数中
    if len(node_clusters) > 0:
        additional_args["node_categories"] = node_clusters
    # 如果节点大小列表不为空，添加到额外参数中
    if len(node_sizes) > 0:
        additional_args["node_sizes"] = node_sizes

    try:
        # 调用计算 UMAP 位置的函数，传入嵌入向量和其他参数
        return compute_umap_positions(
            embedding_vectors=np.array(embedding_vectors),
            node_labels=nodes,
            **additional_args,
            min_dist=args.get("min_dist", 0.75),
            n_neighbors=args.get("n_neighbors", 5),
        )
    except Exception as e:
        # 记录异常信息到日志
        log.exception("Error running UMAP")
        # 调用错误处理函数，传入异常对象和堆栈信息
        on_error(e, traceback.format_exc(), None)
        # Umap 可能因输入稀疏或内存压力而失败。
        # 在这些情况下，返回所有节点位置都为 (0, 0) 的布局
        result = []
        for i in range(len(nodes)):
            cluster = node_clusters[i] if len(node_clusters) > 0 else 1
            # 构建节点位置对象，并添加到结果列表中
            result.append(
                NodePosition(x=0, y=0, label=nodes[i], size=0, cluster=str(cluster))
            )
        return result


def _filter_raw_embeddings(embeddings: NodeEmbeddings) -> NodeEmbeddings:
    """Filter out None values from raw embeddings."""
    return {
        node_id: embedding
        for node_id, embedding in embeddings.items()
        if embedding is not None
    }
```