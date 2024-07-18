# `.\graphrag\graphrag\index\verbs\graph\embed\strategies\node_2_vec.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run method definition."""

from typing import Any  # 导入 Any 类型，用于参数类型的灵活性

import networkx as nx  # 导入网络图库 networkx

from graphrag.index.graph.embedding import embed_nod2vec  # 导入节点嵌入函数 embed_nod2vec
from graphrag.index.graph.utils import stable_largest_connected_component  # 导入稳定的最大连通组件函数
from graphrag.index.verbs.graph.embed.typing import NodeEmbeddings  # 导入节点嵌入类型 NodeEmbeddings


def run(graph: nx.Graph, args: dict[str, Any]) -> NodeEmbeddings:
    """Run method definition."""
    if args.get("use_lcc", True):  # 如果参数中设置了 use_lcc 为 True 或未设置，默认为 True
        graph = stable_largest_connected_component(graph)  # 调用函数，获取最大稳定的连通组件的图

    # create graph embedding using node2vec
    embeddings = embed_nod2vec(
        graph=graph,
        dimensions=args.get("dimensions", 1536),  # 获取嵌入维度，默认为 1536
        num_walks=args.get("num_walks", 10),  # 获取每个节点的随机游走数量，默认为 10
        walk_length=args.get("walk_length", 40),  # 获取每次随机游走的长度，默认为 40
        window_size=args.get("window_size", 2),  # 获取 Word2Vec 窗口大小，默认为 2
        iterations=args.get("iterations", 3),  # 获取迭代次数，默认为 3
        random_seed=args.get("random_seed", 86),  # 获取随机数种子，默认为 86
    )

    pairs = zip(embeddings.nodes, embeddings.embeddings.tolist(), strict=True)  # 将节点和嵌入向量组合成元组，strict=True 严格匹配长度
    sorted_pairs = sorted(pairs, key=lambda x: x[0])  # 按节点排序元组列表

    return dict(sorted_pairs)  # 返回排序后的节点和嵌入向量的字典
```