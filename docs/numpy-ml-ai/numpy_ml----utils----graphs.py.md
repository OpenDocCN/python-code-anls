# `numpy-ml\numpy_ml\utils\graphs.py`

```
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations, permutations
import numpy as np

#######################################################################
#                          Graph Components                           #
#######################################################################

# 定义边对象
class Edge(object):
    def __init__(self, fr, to, w=None):
        """
        A generic directed edge object.

        Parameters
        ----------
        fr: int
            The id of the vertex the edge goes from
        to: int
            The id of the vertex the edge goes to
        w: float, :class:`Object` instance, or None
            The edge weight, if applicable. If weight is an arbitrary Object it
            must have a method called 'sample' which takes no arguments and
            returns a random sample from the weight distribution. If `w` is
            None, no weight is assumed. Default is None.
        """
        self.fr = fr
        self.to = to
        self._w = w

    def __repr__(self):
        return "{} -> {}, weight: {}".format(self.fr, self.to, self._w)

    @property
    def weight(self):
        return self._w.sample() if hasattr(self._w, "sample") else self._w

    def reverse(self):
        """Reverse the edge direction"""
        return Edge(self.t, self.f, self.w)

#######################################################################
#                             Graph Types                             #
#######################################################################

# 定义图的抽象基类
class Graph(ABC):
    def __init__(self, V, E):
        self._I2V = {i: v for i, v in zip(range(len(V)), V)}
        self._V2I = {v: i for i, v in zip(range(len(V)), V)}
        self._G = {i: set() for i in range(len(V))}
        self._V = V
        self._E = E

        self._build_adjacency_list()

    def __getitem__(self, v_i):
        return self.get_neighbors(v_i)
    # 获取给定顶点的内部索引
    def get_index(self, v):
        """Get the internal index for a given vetex"""
        return self._V2I[v]

    # 根据给定内部索引获取原始顶点
    def get_vertex(self, v_i):
        """Get the original vertex from a given internal index"""
        return self._I2V[v_i]

    # 返回图的顶点集合
    @property
    def vertices(self):
        return self._V

    # 返回图的顶点索引集合
    @property
    def indices(self):
        return list(range(len(self.vertices)))

    # 返回图的边集合
    @property
    def edges(self):
        return self._E

    # 返回从具有索引 `v_i` 的顶点可达的顶点的内部索引列表
    def get_neighbors(self, v_i):
        """
        Return the internal indices of the vertices reachable from the vertex
        with index `v_i`.
        """
        return [self._V2I[e.to] for e in self._G[v_i]]

    # 返回图的邻接矩阵表示
    def to_matrix(self):
        """Return an adjacency matrix representation of the graph"""
        adj_mat = np.zeros((len(self._V), len(self._V)))
        for e in self.edges:
            fr, to = self._V2I[e.fr], self._V2I[e.to]
            adj_mat[fr, to] = 1 if e.weight is None else e.weight
        return adj_mat

    # 返回图的邻接字典表示
    def to_adj_dict(self):
        """Return an adjacency dictionary representation of the graph"""
        adj_dict = defaultdict(lambda: list())
        for e in self.edges:
            adj_dict[e.fr].append(e)
        return adj_dict
    def path_exists(self, s_i, e_i):
        """
        Check whether a path exists from vertex index `s_i` to `e_i`.

        Parameters
        ----------
        s_i: Int
            The interal index of the start vertex
        e_i: Int
            The internal index of the end vertex

        Returns
        -------
        path_exists : Boolean
            Whether or not a valid path exists between `s_i` and `e_i`.
        """
        # 初始化队列，起始点为s_i，路径为[s_i]
        queue = [(s_i, [s_i])]
        # 循环直到队列为空
        while len(queue):
            # 取出队列中的第一个元素
            c_i, path = queue.pop(0)
            # 获取当前节点的邻居节点，但不包括已经在路径中的节点
            nbrs_not_on_path = set(self.get_neighbors(c_i)) - set(path)

            # 遍历邻居节点
            for n_i in nbrs_not_on_path:
                # 将邻居节点加入路径，并加入队列
                queue.append((n_i, path + [n_i]))
                # 如果找到终点e_i，则返回True
                if n_i == e_i:
                    return True
        # 如果循环结束仍未找到路径，则返回False
        return False

    def all_paths(self, s_i, e_i):
        """
        Find all simple paths between `s_i` and `e_i` in the graph.

        Notes
        -----
        Uses breadth-first search. Ignores all paths with repeated vertices.

        Parameters
        ----------
        s_i: Int
            The interal index of the start vertex
        e_i: Int
            The internal index of the end vertex

        Returns
        -------
        complete_paths : list of lists
            A list of all paths from `s_i` to `e_i`. Each path is represented
            as a list of interal vertex indices.
        """
        # 初始化存储所有路径的列表
        complete_paths = []
        # 初始化队列，起始点为s_i，路径为[s_i]
        queue = [(s_i, [s_i])]

        # 循环直到队列为空
        while len(queue):
            # 取出队列中的第一个元素
            c_i, path = queue.pop(0)
            # 获取当前节点的邻居节点，但不包括已经在路径中的节点
            nbrs_not_on_path = set(self.get_neighbors(c_i)) - set(path)

            # 遍历邻居节点
            for n_i in nbrs_not_on_path:
                # 如果找到终点e_i，则将完整路径加入complete_paths
                if n_i == e_i:
                    complete_paths.append(path + [n_i])
                else:
                    # 将邻居节点加入路径，并加入队列
                    queue.append((n_i, path + [n_i]))

        # 返回所有找到的路径
        return complete_paths

    @abstractmethod
    def _build_adjacency_list(self):
        pass
class DiGraph(Graph):
    def __init__(self, V, E):
        """
        A generic directed graph object.

        Parameters
        ----------
        V : list
            A list of vertex IDs.
        E : list of :class:`Edge <numpy_ml.utils.graphs.Edge>` objects
            A list of directed edges connecting pairs of vertices in ``V``.
        """
        # 调用父类的构造函数，传入顶点列表和边列表
        super().__init__(V, E)
        # 设置图为有向图
        self.is_directed = True
        # 初始化拓扑排序列表为空
        self._topological_ordering = []

    def _build_adjacency_list(self):
        """Encode directed graph as an adjancency list"""
        # 假设没有平行边
        for e in self.edges:
            # 获取起始顶点在顶点列表中的索引
            fr_i = self._V2I[e.fr]
            # 将边添加到起始顶点的邻接表中
            self._G[fr_i].add(e)

    def reverse(self):
        """Reverse the direction of all edges in the graph"""
        # 返回一个新的有向图对象，边的方向取反
        return DiGraph(self.vertices, [e.reverse() for e in self.edges])

    def is_acyclic(self):
        """Check whether the graph contains cycles"""
        # 检查图是否包含环，通过判断拓扑排序结果是否为None来判断
        return self.topological_ordering() is not None


class UndirectedGraph(Graph):
    def __init__(self, V, E):
        """
        A generic undirected graph object.

        Parameters
        ----------
        V : list
            A list of vertex IDs.
        E : list of :class:`Edge <numpy_ml.utils.graphs.Edge>` objects
            A list of edges connecting pairs of vertices in ``V``. For any edge
            connecting vertex `u` to vertex `v`, :class:`UndirectedGraph
            <numpy_ml.utils.graphs.UndirectedGraph>` will assume that there
            exists a corresponding edge connecting `v` to `u`, even if this is
            not present in `E`.
        """
        # 调用父类的构造函数，传入顶点列表和边列表
        super().__init__(V, E)
        # 设置图为无向图
        self.is_directed = False
    # 构建邻接表来表示无向、无权重图
    def _build_adjacency_list(self):
        """Encode undirected, unweighted graph as an adjancency list"""
        # 假设没有平行边
        # 每条边出现两次，分别为 (u,v) 和 (v,u)
        for e in self.edges:
            # 获取起始节点和目标节点在节点列表中的索引
            fr_i = self._V2I[e.fr]
            to_i = self._V2I[e.to]

            # 将边添加到起始节点的邻接表中
            self._G[fr_i].add(e)
            # 将边的反向边添加到目标节点的邻接表中
            self._G[to_i].add(e.reverse())
#######################################################################
#                          Graph Generators                           #
#######################################################################

# 生成一个无权 Erdős-Rényi 随机图
def random_unweighted_graph(n_vertices, edge_prob=0.5, directed=False):
    """
    Generate an unweighted Erdős-Rényi random graph [*]_.

    References
    ----------
    .. [*] Erdős, P. and Rényi, A. (1959). On Random Graphs, *Publ. Math. 6*, 290.

    Parameters
    ----------
    n_vertices : int
        The number of vertices in the graph.
    edge_prob : float in [0, 1]
        The probability of forming an edge between two vertices. Default is
        0.5.
    directed : bool
        Whether the edges in the graph should be directed. Default is False.

    Returns
    -------
    G : :class:`Graph` instance
        The resulting random graph.
    """
    vertices = list(range(n_vertices))
    candidates = permutations(vertices, 2) if directed else combinations(vertices, 2)

    edges = []
    # 遍历所有可能的边
    for (fr, to) in candidates:
        # 根据概率确定是否生成边
        if np.random.rand() <= edge_prob:
            edges.append(Edge(fr, to))

    # 如果是有向图，则返回有向图对象；否则返回无向图对象
    return DiGraph(vertices, edges) if directed else UndirectedGraph(vertices, edges)


# 创建一个 '随机' 无权有向无环图，通过修剪所有反向连接来生成
def random_DAG(n_vertices, edge_prob=0.5):
    """
    Create a 'random' unweighted directed acyclic graph by pruning all the
    backward connections from a random graph.

    Parameters
    ----------
    n_vertices : int
        The number of vertices in the graph.
    edge_prob : float in [0, 1]
        The probability of forming an edge between two vertices in the
        underlying random graph, before edge pruning. Default is 0.5.

    Returns
    -------
    G : :class:`Graph` instance
        The resulting DAG.
    """
    # 生成一个有向随机图
    G = random_unweighted_graph(n_vertices, edge_prob, directed=True)

    # 修剪边以删除顶点之间的反向连接
    G = DiGraph(G.vertices, [e for e in G.edges if e.fr < e.to])
    # 如果我们删除了所有边，则生成一个新的图
    while not len(G.edges):
        # 生成一个具有随机权重的有向图
        G = random_unweighted_graph(n_vertices, edge_prob, directed=True)
        # 创建一个新的有向图，只包含起点小于终点的边
        G = DiGraph(G.vertices, [e for e in G.edges if e.fr < e.to])
    # 返回生成的图
    return G
```