# `.\DB-GPT-src\dbgpt\storage\graph_store\memgraph_store.py`

```py
"""Graph store base class."""
import json  # 导入 JSON 库，用于处理 JSON 数据
import logging  # 导入 logging 模块，用于记录日志
from typing import List, Optional, Tuple  # 引入类型提示，用于静态类型检查

from dbgpt._private.pydantic import ConfigDict, Field  # 导入 ConfigDict 和 Field 类
from dbgpt.storage.graph_store.base import GraphStoreBase, GraphStoreConfig  # 导入图存储基类和配置类
from dbgpt.storage.graph_store.graph import Direction, Edge, Graph, MemoryGraph  # 导入图相关的类

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 对象


class MemoryGraphStoreConfig(GraphStoreConfig):
    """Memory graph store config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义模型配置字典，允许任意类型

    edge_name_key: str = Field(
        default="label",
        description="The label of edge name, `label` by default.",
    )  # 定义边名称键，默认为"label"


class MemoryGraphStore(GraphStoreBase):
    """Memory graph store."""

    def __init__(self, graph_store_config: MemoryGraphStoreConfig):
        """Initialize MemoryGraphStore with a memory graph."""
        self._edge_name_key = graph_store_config.edge_name_key  # 初始化边名称键
        self._graph = MemoryGraph(edge_label=self._edge_name_key)  # 使用边名称键初始化内存图对象

    def insert_triplet(self, sub: str, rel: str, obj: str):
        """Insert a triplet into the graph."""
        self._graph.append_edge(Edge(sub, obj, **{self._edge_name_key: rel}))  # 向图中插入三元组

    def get_triplets(self, sub: str) -> List[Tuple[str, str]]:
        """Retrieve triplets originating from a subject."""
        subgraph = self.explore([sub], direct=Direction.OUT, depth=1)  # 探索以给定主题为起点的子图
        return [(e.get_prop(self._edge_name_key), e.tid) for e in subgraph.edges()]  # 返回三元组列表

    def delete_triplet(self, sub: str, rel: str, obj: str):
        """Delete a specific triplet from the graph."""
        self._graph.del_edges(sub, obj, **{self._edge_name_key: rel})  # 从图中删除特定的三元组

    def drop(self):
        """Drop graph."""
        self._graph = None  # 释放图对象

    def get_schema(self, refresh: bool = False) -> str:
        """Return the graph schema as a JSON string."""
        return json.dumps(self._graph.schema())  # 返回图的模式作为 JSON 字符串

    def get_full_graph(self, limit: Optional[int] = None) -> MemoryGraph:
        """Return self."""
        if not limit:
            return self._graph  # 如果未指定限制，则返回完整的图对象

        subgraph = MemoryGraph()
        for count, edge in enumerate(self._graph.edges()):
            if count >= limit:
                break
            subgraph.upsert_vertex(self._graph.get_vertex(edge.sid))
            subgraph.upsert_vertex(self._graph.get_vertex(edge.tid))
            subgraph.append_edge(edge)
            count += 1
        return subgraph  # 返回限制大小的子图对象

    def explore(
        self,
        subs: List[str],
        direct: Direction = Direction.BOTH,
        depth: Optional[int] = None,
        fan: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> MemoryGraph:
        """Explore the graph from given subjects up to a depth."""
        return self._graph.search(subs, direct, depth, fan, limit)  # 从给定主题探索图，直到指定深度

    def query(self, query: str, **args) -> Graph:
        """Execute a query on graph."""
        raise NotImplementedError("Query memory graph not allowed")  # 抛出未实现错误，不允许在内存图上执行查询
```