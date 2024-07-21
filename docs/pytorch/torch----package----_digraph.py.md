# `.\pytorch\torch\package\_digraph.py`

```
# mypy: allow-untyped-defs
# 从 collections 模块导入 deque 类
from collections import deque
# 从 typing 模块导入 List 和 Set 类型
from typing import List, Set

# 定义有向图类 DiGraph
class DiGraph:
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    # 初始化方法
    def __init__(self):
        # 字典，存储节点 -> 节点的属性字典
        self._node = {}
        # 嵌套字典，存储节点 -> 后继节点 -> 空值（未实现边的数据）
        self._succ = {}
        # 嵌套字典，存储节点 -> 前驱节点 -> 空值
        self._pred = {}

        # 记录节点添加的顺序
        self._node_order = {}
        self._insertion_idx = 0

    # 添加节点方法
    def add_node(self, n, **kwargs):
        """Add a node to the graph.

        Args:
            n: the node. Can we any object that is a valid dict key.
            **kwargs: any attributes you want to attach to the node.
        """
        # 如果节点 n 不在 _node 字典中
        if n not in self._node:
            # 添加节点 n 和其属性字典到 _node 字典中
            self._node[n] = kwargs
            # 初始化节点 n 的后继节点字典和前驱节点字典为空字典
            self._succ[n] = {}
            self._pred[n] = {}
            # 记录节点 n 的插入顺序和索引
            self._node_order[n] = self._insertion_idx
            self._insertion_idx += 1
        else:
            # 如果节点 n 已存在，更新其属性字典
            self._node[n].update(kwargs)

    # 添加边方法
    def add_edge(self, u, v):
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
        # 添加节点 u 和 v 到图中（如果它们不存在的话）
        self.add_node(u)
        self.add_node(v)

        # 添加边 (u, v)
        self._succ[u][v] = True  # u 的后继节点中添加 v
        self._pred[v][u] = True  # v 的前驱节点中添加 u

    # 返回节点 n 的后继节点迭代器方法
    def successors(self, n):
        """Returns an iterator over successor nodes of n."""
        try:
            return iter(self._succ[n])
        except KeyError as e:
            # 如果节点 n 不在图中，抛出 ValueError 异常
            raise ValueError(f"The node {n} is not in the digraph.") from e

    # 返回节点 n 的前驱节点迭代器方法
    def predecessors(self, n):
        """Returns an iterator over predecessors nodes of n."""
        try:
            return iter(self._pred[n])
        except KeyError as e:
            # 如果节点 n 不在图中，抛出 ValueError 异常
            raise ValueError(f"The node {n} is not in the digraph.") from e

    # 返回所有边的迭代器方法
    @property
    def edges(self):
        """Returns an iterator over all edges (u, v) in the graph"""
        for n, successors in self._succ.items():
            for succ in successors:
                yield n, succ

    # 返回所有节点及其属性字典的方法
    @property
    def nodes(self):
        """Returns a dictionary of all nodes to their attributes."""
        return self._node

    # 迭代器方法，返回图中所有节点的迭代器
    def __iter__(self):
        """Iterate over the nodes."""
        return iter(self._node)

    # 成员检查方法，判断节点 n 是否在图中
    def __contains__(self, n):
        """Returns True if ``n`` is a node in the graph, False otherwise."""
        try:
            return n in self._node
        except TypeError:
            return False
    def forward_transitive_closure(self, src: str) -> Set[str]:
        """Returns a set of nodes that are reachable from src"""

        # 初始化结果集合，包含起始节点
        result = set(src)
        # 初始化工作集合，起始节点作为初始值
        working_set = deque(src)
        # 使用广度优先搜索计算从 src 出发可达的所有节点
        while len(working_set) > 0:
            cur = working_set.popleft()
            # 对当前节点的后继节点进行遍历
            for n in self.successors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        # 返回结果集合
        return result

    def backward_transitive_closure(self, src: str) -> Set[str]:
        """Returns a set of nodes that are reachable from src in reverse direction"""

        # 初始化结果集合，包含起始节点
        result = set(src)
        # 初始化工作集合，起始节点作为初始值
        working_set = deque(src)
        # 使用广度优先搜索计算从 src 出发的反向可达节点
        while len(working_set) > 0:
            cur = working_set.popleft()
            # 对当前节点的前驱节点进行遍历
            for n in self.predecessors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        # 返回结果集合
        return result

    def all_paths(self, src: str, dst: str):
        """Returns a subgraph rooted at src that shows all the paths to dst."""

        # 创建一个空的有向图对象
        result_graph = DiGraph()
        # 计算从 src 出发的正向传递闭包（所有可达的节点）
        forward_reachable_from_src = self.forward_transitive_closure(src)

        # 如果目标节点不在从 src 可达的节点集合中，则返回空图
        if dst not in forward_reachable_from_src:
            return result_graph

        # 初始化工作集合，目标节点作为初始值
        working_set = deque(dst)
        # 遍历目标节点的反向依赖关系，将可达于从 src 的节点添加到输出图中
        while len(working_set) > 0:
            cur = working_set.popleft()
            # 遍历当前节点的前驱节点
            for n in self.predecessors(cur):
                if n in forward_reachable_from_src:
                    # 在结果图中添加边，仅限于从 src 可达的节点
                    result_graph.add_edge(n, cur)
                    # 只有当节点从 src 可达时，才继续探索
                    working_set.append(n)

        # 返回输出图的 DOT 表示形式
        return result_graph.to_dot()

    def first_path(self, dst: str) -> List[str]:
        """Returns a list of nodes that show the first path that resulted in dst being added to the graph."""
        
        # 初始化路径列表
        path = []

        # 从目标节点开始，沿着路径反向遍历，直到起始节点
        while dst:
            path.append(dst)
            # 获取导致目标节点添加到图中的候选节点集合
            candidates = self._pred[dst].keys()
            dst, min_idx = "", None
            # 在候选节点中选择最小索引的节点作为下一个目标节点
            for candidate in candidates:
                idx = self._node_order.get(candidate, None)
                if idx is None:
                    break
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                    dst = candidate

        # 返回反转后的路径列表
        return list(reversed(path))

    def to_dot(self) -> str:
        """Returns the dot representation of the graph.

        Returns:
            A dot representation of the graph.
        """
        # 构建图的 DOT 表示形式的边列表
        edges = "\n".join(f'"{f}" -> "{t}";' for f, t in self.edges)
        # 返回包含所有边的 DOT 表示形式
        return f"""
        digraph {{
            {edges}
        }}
        """
# 构建一个 DOT 语言格式的字符串，用于描述有向图的结构
digraph G {{
    # 设置图的排列方向为从左到右（LR表示从左到右，TB表示从上到下）
    rankdir = LR;
    # 设置所有节点的形状为矩形
    node [shape=box];
    # 插入具体的边（由外部传入的参数替换{edges}）
    {edges}
}}
"""
```