# `.\graphrag\graphrag\index\graph\embedding\embedding.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""用于生成图嵌入的实用工具。"""

# 导入必要的库
from dataclasses import dataclass
import graspologic as gc
import networkx as nx
import numpy as np

@dataclass
class NodeEmbeddings:
    """节点嵌入类的定义。"""
    nodes: list[str]   # 节点列表
    embeddings: np.ndarray   # 嵌入向量数组

def embed_nod2vec(
    graph: nx.Graph | nx.DiGraph,   # 图或有向图对象
    dimensions: int = 1536,   # 嵌入向量的维度，默认为1536
    num_walks: int = 10,   # 每个节点的随机游走次数，默认为10
    walk_length: int = 40,   # 每次随机游走的长度，默认为40
    window_size: int = 2,   # 用于生成Skip-gram模型的窗口大小，默认为2
    iterations: int = 3,   # 迭代次数，默认为3
    random_seed: int = 86,   # 随机种子，默认为86
) -> NodeEmbeddings:
    """使用Node2Vec生成节点嵌入。"""
    # 生成嵌入向量
    lcc_tensors = gc.embed.node2vec_embed(  # type: ignore
        graph=graph,
        dimensions=dimensions,
        window_size=window_size,
        iterations=iterations,
        num_walks=num_walks,
        walk_length=walk_length,
        random_seed=random_seed,
    )
    # 返回节点嵌入对象
    return NodeEmbeddings(embeddings=lcc_tensors[0], nodes=lcc_tensors[1])
```