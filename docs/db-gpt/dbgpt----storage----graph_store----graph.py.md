# `.\DB-GPT-src\dbgpt\storage\graph_store\graph.py`

```py
"""Graph store base class."""
import itertools  # 导入 itertools 库，用于高效迭代操作
import json  # 导入 json 库，用于处理 JSON 格式数据
import logging  # 导入 logging 库，用于日志记录
import re  # 导入 re 库，用于正则表达式匹配
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 类和 abstractmethod 装饰器
from collections import defaultdict  # 导入 defaultdict 类，创建默认值为列表的字典
from enum import Enum  # 导入 Enum 枚举类
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple  # 导入类型提示相关的类和接口

import networkx as nx  # 导入 networkx 库，用于处理图结构数据

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class Direction(Enum):
    """Direction class."""
    
    OUT = 0  # 枚举类型 OUT，代表出边方向
    IN = 1   # 枚举类型 IN，代表入边方向
    BOTH = 2  # 枚举类型 BOTH，代表双向边方向


class Elem(ABC):
    """Elem class."""
    
    def __init__(self):
        """Initialize Elem."""
        self._props = {}  # 初始化属性字典为空字典

    @property
    def props(self) -> Dict[str, Any]:
        """Get all the properties of Elem."""
        return self._props  # 返回元素的所有属性字典

    def set_prop(self, key: str, value: Any):
        """Set a property of ELem."""
        self._props[key] = value  # 设置元素的一个属性

    def get_prop(self, key: str):
        """Get one of the properties of Elem."""
        return self._props.get(key)  # 获取元素的一个属性值，如果属性不存在返回 None

    def del_prop(self, key: str):
        """Delete a property of ELem."""
        self._props.pop(key, None)  # 删除元素的一个属性，如果属性不存在不抛出异常

    def has_props(self, **props):
        """Check if the element has the specified properties with the given values."""
        return all(self._props.get(k) == v for k, v in props.items())  # 检查元素是否具有指定属性及其给定值

    @abstractmethod
    def format(self, label_key: Optional[str] = None):
        """Format properties into a string."""
        formatted_props = [
            f"{k}:{json.dumps(v)}" for k, v in self._props.items() if k != label_key
        ]  # 将属性格式化为字符串列表，排除标签键
        return f"{{{';'.join(formatted_props)}}}"  # 返回格式化后的字符串


class Vertex(Elem):
    """Vertex class."""
    
    def __init__(self, vid: str, **props):
        """Initialize Vertex."""
        super().__init__()  # 调用父类初始化方法
        self._vid = vid  # 设置顶点 ID
        for k, v in props.items():
            self.set_prop(k, v)  # 设置顶点的属性

    @property
    def vid(self) -> str:
        """Return the vertex ID."""
        return self._vid  # 返回顶点 ID

    def format(self, label_key: Optional[str] = None):
        """Format vertex properties into a string."""
        label = self.get_prop(label_key) if label_key else self._vid  # 获取顶点的标签
        props_str = super().format(label_key)  # 调用父类的格式化方法
        if props_str == "{}":
            return f"({label})"  # 如果属性为空，则返回仅包含标签的字符串
        else:
            return f"({label}:{props_str})"  # 否则返回包含标签和属性的字符串

    def __str__(self):
        """Return the vertex ID as its string representation."""
        return f"({self._vid})"  # 返回顶点 ID 的字符串表示


class Edge(Elem):
    """Edge class."""
    
    def __init__(self, sid: str, tid: str, **props):
        """Initialize Edge."""
        super().__init__()  # 调用父类初始化方法
        self._sid = sid  # 设置边的源顶点 ID
        self._tid = tid  # 设置边的目标顶点 ID
        for k, v in props.items():
            self.set_prop(k, v)  # 设置边的属性

    @property
    def sid(self) -> str:
        """Return the source vertex ID of the edge."""
        return self._sid  # 返回边的源顶点 ID

    @property
    def tid(self) -> str:
        """Return the target vertex ID of the edge."""
        return self._tid  # 返回边的目标顶点 ID
    # 返回相邻节点的 ID
    def nid(self, vid):
        """Return neighbor id."""
        # 如果 vid 等于起始节点 ID，则返回目标节点 ID
        if vid == self._sid:
            return self._tid
        # 如果 vid 等于目标节点 ID，则返回起始节点 ID
        elif vid == self._tid:
            return self._sid
        else:
            # 如果 vid 既不是起始节点 ID 也不是目标节点 ID，则抛出数值错误异常
            raise ValueError(f"Get nid of {vid} on {self} failed")

    # 将边属性格式化成字符串
    def format(self, label_key: Optional[str] = None):
        """Format the edge properties into a string."""
        # 获取指定标签键的标签值
        label = self.get_prop(label_key) if label_key else ""
        # 调用父类方法格式化所有属性成字符串
        props_str = super().format(label_key)
        # 如果格式化后的属性字符串为空字典，则返回简化格式的边描述
        if props_str == "{}":
            return f"-[{label}]->" if label else "->"
        else:
            # 否则返回包含属性的边描述
            return f"-[{label}:{props_str}]->" if label else f"-[{props_str}]->"

    # 返回包含指定标签键的三元组
    def triplet(self, label_key: str) -> Tuple[str, str, str]:
        """Return a triplet."""
        # 断言标签键非空，若为空则抛出断言错误
        assert label_key, "label key is needed"
        # 返回起始节点 ID、标签键对应的属性字符串、目标节点 ID 的三元组
        return self._sid, str(self.get_prop(label_key)), self._tid

    # 返回边的字符串表示形式 '(起始节点 ID)->(目标节点 ID)'
    def __str__(self):
        """Return the edge '(sid)->(tid)'."""
        return f"({self._sid})->({self._tid})"
    def upsert_vertex(self, vertex: Vertex):
        """Add or update a vertex in the graph."""



    def append_edge(self, edge: Edge):
        """Add an edge to the graph."""



    def has_vertex(self, vid: str) -> bool:
        """Check if a vertex with the given ID exists in the graph."""



    def get_vertex(self, vid: str) -> Vertex:
        """Retrieve a vertex from the graph by its ID."""



    def get_neighbor_edges(
        self,
        vid: str,
        direction: Direction = Direction.OUT,
        limit: Optional[int] = None,
    ) -> Iterator[Edge]:
        """Retrieve neighbor edges of a vertex based on direction and optional limit."""



    def vertices(self) -> Iterator[Vertex]:
        """Return an iterator over all vertices in the graph."""



    def edges(self) -> Iterator[Edge]:
        """Return an iterator over all edges in the graph."""



    def del_vertices(self, *vids: str):
        """Delete vertices and their incident edges from the graph."""



    def del_edges(self, sid: str, tid: str, **props):
        """Delete edges from vertex sid to vertex tid based on properties."""



    def del_neighbor_edges(self, vid: str, direction: Direction = Direction.OUT):
        """Delete neighbor edges of a vertex based on direction."""



    def search(
        self,
        vids: List[str],
        direct: Direction = Direction.OUT,
        depth: Optional[int] = None,
        fan: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> "Graph":
        """Search the graph starting from specified vertices with optional parameters."""



    def schema(self) -> Dict[str, Any]:
        """Retrieve the schema of the graph, describing its structure."""



    def format(self) -> str:
        """Format the graph data into a string representation."""



    def __init__(self, vertex_label: Optional[str] = None, edge_label: str = "label"):
        """Initialize MemoryGraph with optional vertex label and required edge label."""



        assert edge_label, "Edge label is needed"



        # metadata



        self._vertex_label = vertex_label



        self._edge_label = edge_label



        self._vertex_prop_keys = {vertex_label} if vertex_label else set()



        self._edge_prop_keys = {edge_label}



        self._edge_count = 0



        # init vertices, out edges, in edges index



        self._vs: Any = defaultdict()



        self._oes: Any = defaultdict(lambda: defaultdict(set))



        self._ies: Any = defaultdict(lambda: defaultdict(set))



    @property
    def vertex_label(self):
        """Return the label for vertices."""



        return self._vertex_label



    @property
    def edge_label(self):
        """Return the label for edges."""



        return self._edge_label



    @property
    def vertex_prop_keys(self):
        """Return a set of property keys for vertices."""



        return self._vertex_prop_keys



    @property
    def edge_prop_keys(self):
        """Return a set of property keys for edges."""



        return self._edge_prop_keys
    def vertex_count(self):
        """Return the number of vertices in the graph."""
        # 返回图中顶点的数量，即顶点字典 `_vs` 的长度
        return len(self._vs)

    @property
    def edge_count(self):
        """Return the count of edges in the graph."""
        # 返回图中边的数量，即 `_edge_count` 的当前值
        return self._edge_count

    def upsert_vertex(self, vertex: Vertex):
        """Insert or update a vertex based on its ID."""
        # 如果顶点已存在，则更新其属性；否则插入新顶点
        if vertex.vid in self._vs:
            self._vs[vertex.vid].props.update(vertex.props)
        else:
            self._vs[vertex.vid] = vertex

        # 更新顶点属性键集合
        self._vertex_prop_keys.update(vertex.props.keys())

    def append_edge(self, edge: Edge):
        """Append an edge if it doesn't exist; requires edge label."""
        # 检查边的属性中是否包含必需的边标签
        if self.edge_label not in edge.props.keys():
            raise ValueError(f"Edge prop '{self.edge_label}' is needed")

        sid = edge.sid
        tid = edge.tid

        # 如果边已存在，则返回 False
        if edge in self._oes[sid][tid]:
            return False

        # 初始化顶点索引，如果顶点不存在则创建
        self._vs.setdefault(sid, Vertex(sid))
        self._vs.setdefault(tid, Vertex(tid))

        # 更新边的索引
        self._oes[sid][tid].add(edge)
        self._ies[tid][sid].add(edge)

        # 更新边的属性键集合和边计数
        self._edge_prop_keys.update(edge.props.keys())
        self._edge_count += 1
        return True

    def has_vertex(self, vid: str) -> bool:
        """Retrieve a vertex by ID."""
        # 检查顶点是否存在于 `_vs` 字典中
        return vid in self._vs

    def get_vertex(self, vid: str) -> Vertex:
        """Retrieve a vertex by ID."""
        # 根据顶点 ID 返回对应的顶点对象
        return self._vs[vid]

    def get_neighbor_edges(
        self,
        vid: str,
        direction: Direction = Direction.OUT,
        limit: Optional[int] = None,
    ) -> Iterator[Edge]:
        """Get edges connected to a vertex by direction."""
        # 根据给定的方向获取连接到顶点的边的迭代器
        if direction == Direction.OUT:
            es = (e for es in self._oes[vid].values() for e in es)

        elif direction == Direction.IN:
            es = iter(e for es in self._ies[vid].values() for e in es)

        elif direction == Direction.BOTH:
            # 获取出边和入边的迭代器，并合并为一个迭代器
            oes = (e for es in self._oes[vid].values() for e in es)
            ies = (e for es in self._ies[vid].values() for e in es)
            tuples = itertools.zip_longest(oes, ies)
            es = (e for t in tuples for e in t if e is not None)

            # 去重，确保每条边只出现一次
            seen = set()
            def unique_elements(elements):
                for element in elements:
                    if element not in seen:
                        seen.add(element)
                        yield element
            es = unique_elements(es)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # 如果指定了限制数量，则返回限制后的迭代器
        return itertools.islice(es, limit) if limit else es

    def vertices(self) -> Iterator[Vertex]:
        """Return vertices."""
        # 返回图中所有顶点的迭代器
        return iter(self._vs.values())
    # 返回图中所有边的迭代器
    def edges(self) -> Iterator[Edge]:
        """Return edges."""
        return iter(e for nbs in self._oes.values() for es in nbs.values() for e in es)

    # 删除指定的顶点及其关联边
    def del_vertices(self, *vids: str):
        """Delete specified vertices."""
        for vid in vids:
            self.del_neighbor_edges(vid, Direction.BOTH)  # 删除顶点的所有邻接边
            self._vs.pop(vid, None)  # 删除顶点本身

    # 删除指定边或根据属性删除边
    def del_edges(self, sid: str, tid: str, **props):
        """Delete edges."""
        old_edge_cnt = len(self._oes[sid][tid])  # 记录原始边的数量

        if not props:
            self._edge_count -= old_edge_cnt  # 更新总边数
            self._oes[sid].pop(tid, None)  # 删除指定边的出边
            self._ies[tid].pop(sid, None)  # 删除指定边的入边
            return

        # 根据属性过滤并删除边
        def remove_matches(es):
            return set(filter(lambda e: not e.has_props(**props), es))

        self._oes[sid][tid] = remove_matches(self._oes[sid][tid])  # 更新出边集合
        self._ies[tid][sid] = remove_matches(self._ies[tid][sid])  # 更新入边集合

        self._edge_count -= old_edge_cnt - len(self._oes[sid][tid])  # 更新总边数

    # 删除指定顶点的所有邻接边
    def del_neighbor_edges(self, vid: str, direction: Direction = Direction.OUT):
        """Delete all neighbor edges."""

        # 辅助函数：删除指定顶点在某个方向上的邻接边
        def del_index(idx, i_idx):
            for nid in idx[vid].keys():
                self._edge_count -= len(i_idx[nid][vid])  # 更新总边数
                i_idx[nid].pop(vid, None)  # 删除邻接边
            idx.pop(vid, None)  # 删除顶点的索引

        if direction in [Direction.OUT, Direction.BOTH]:
            del_index(self._oes, self._ies)  # 删除出边

        if direction in [Direction.IN, Direction.BOTH]:
            del_index(self._ies, self._oes)  # 删除入边

    # 从指定顶点开始搜索图中的子图
    def search(
        self,
        vids: List[str],
        direct: Direction = Direction.OUT,
        depth: Optional[int] = None,
        fan: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> "MemoryGraph":
        """Search the graph from a vertex with specified parameters."""
        subgraph = MemoryGraph()  # 创建用于存储子图的对象

        for vid in vids:
            self.__search(vid, direct, depth, fan, limit, 0, set(), subgraph)  # 调用搜索函数

        return subgraph  # 返回子图对象

    # 递归搜索子图的内部实现函数
    def __search(
        self,
        vid: str,
        direct: Direction,
        depth: Optional[int],
        fan: Optional[int],
        limit: Optional[int],
        _depth: int,
        _visited: Set,
        _subgraph: "MemoryGraph",
    ):
        # 实际的搜索逻辑在调用时详细说明
    ):
        # 如果顶点已经被访问过或者达到了指定的深度而且深度不为零，则返回
        if vid in _visited or depth and _depth >= depth:
            return

        # 访问顶点
        # 如果图中不包含该顶点，则返回
        if not self.has_vertex(vid):
            return
        # 向子图中插入该顶点
        _subgraph.upsert_vertex(self.get_vertex(vid))
        # 将该顶点标记为已访问
        _visited.add(vid)

        # 访问边
        nids = set()
        for edge in self.get_neighbor_edges(vid, direct, fan):
            # 如果已经达到了限制的边数，则返回
            if limit and _subgraph.edge_count >= limit:
                return

            # 如果成功添加边，则访问新的顶点
            if _subgraph.append_edge(edge):
                nid = edge.nid(vid)
                # 如果新顶点未被访问过，则将其加入待访问列表
                if nid not in _visited:
                    nids.add(nid)

        # 进行下一跳搜索
        for nid in nids:
            self.__search(
                nid, direct, depth, fan, limit, _depth + 1, _visited, _subgraph
            )

    # 返回图的模式
    def schema(self) -> Dict[str, Any]:
        """Return schema."""
        return {
            "schema": [
                {
                    "type": "VERTEX",
                    "label": f"{self._vertex_label}",
                    "properties": [{"name": k} for k in self._vertex_prop_keys],
                },
                {
                    "type": "EDGE",
                    "label": f"{self._edge_label}",
                    "properties": [{"name": k} for k in self._edge_prop_keys],
                },
            ]
        }

    # 将图格式化为字符串
    def format(self) -> str:
        """Format graph to string."""
        # 格式化顶点列表
        vs_str = "\n".join(v.format(self.vertex_label) for v in self.vertices())
        # 格式化边列表
        es_str = "\n".join(
            f"{self.get_vertex(e.sid).format(self.vertex_label)}"
            f"{e.format(self.edge_label)}"
            f"{self.get_vertex(e.tid).format(self.vertex_label)}"
            for e in self.edges()
        )
        return f"Vertices:\n{vs_str}\n\nEdges:\n{es_str}"

    # 生成图的Graphviz表示
    def graphviz(self, name="g"):
        """View graphviz graph: https://dreampuf.github.io/GraphvizOnline."""
        # 创建一个多重有向图
        g = nx.MultiDiGraph()
        # 将所有顶点添加到图中
        for vertex in self.vertices():
            g.add_node(vertex.vid)

        # 将所有边添加到图中
        for edge in self.edges():
            triplet = edge.triplet(self.edge_label)
            g.add_edge(triplet[0], triplet[2], label=triplet[1])

        # 将图转换为Graphviz格式的字符串
        digraph = nx.nx_agraph.to_agraph(g).to_string()
        # 替换默认的图名称为指定名称
        digraph = digraph.replace('digraph ""', f"digraph {name}")
        # 移除Graphviz格式中的键信息
        digraph = re.sub(r"key=\d+,?\s*", "", digraph)
        return digraph
```