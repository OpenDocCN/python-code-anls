# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\__init__.py`

```
r"""
Compressed sparse graph routines (:mod:`scipy.sparse.csgraph`)
==============================================================

.. currentmodule:: scipy.sparse.csgraph

Fast graph algorithms based on sparse matrix representations.

Contents
--------

.. autosummary::
   :toctree: generated/

   connected_components -- determine connected components of a graph
   laplacian -- compute the laplacian of a graph
   shortest_path -- compute the shortest path between points on a positive graph
   dijkstra -- use Dijkstra's algorithm for shortest path
   floyd_warshall -- use the Floyd-Warshall algorithm for shortest path
   bellman_ford -- use the Bellman-Ford algorithm for shortest path
   johnson -- use Johnson's algorithm for shortest path
   yen -- use Yen's algorithm for K-shortest paths between to nodes.
   breadth_first_order -- compute a breadth-first order of nodes
   depth_first_order -- compute a depth-first order of nodes
   breadth_first_tree -- construct the breadth-first tree from a given node
   depth_first_tree -- construct a depth-first tree from a given node
   minimum_spanning_tree -- construct the minimum spanning tree of a graph
   reverse_cuthill_mckee -- compute permutation for reverse Cuthill-McKee ordering
   maximum_flow -- solve the maximum flow problem for a graph
   maximum_bipartite_matching -- compute a maximum matching of a bipartite graph
   min_weight_full_bipartite_matching - compute a minimum weight full matching of a bipartite graph
   structural_rank -- compute the structural rank of a graph
   NegativeCycleError

.. autosummary::
   :toctree: generated/

   construct_dist_matrix
   csgraph_from_dense
   csgraph_from_masked
   csgraph_masked_from_dense
   csgraph_to_dense
   csgraph_to_masked
   reconstruct_path

Graph Representations
---------------------
This module uses graphs which are stored in a matrix format. A
graph with N nodes can be represented by an (N x N) adjacency matrix G.
If there is a connection from node i to node j, then G[i, j] = w, where
w is the weight of the connection. For nodes i and j which are
not connected, the value depends on the representation:

- for dense array representations, non-edges are represented by
  G[i, j] = 0, infinity, or NaN.

- for dense masked representations (of type np.ma.MaskedArray), non-edges
  are represented by masked values. This can be useful when graphs with
  zero-weight edges are desired.

- for sparse array representations, non-edges are represented by
  non-entries in the matrix. This sort of sparse representation also
  allows for edges with zero weights.

As a concrete example, imagine that you would like to represent the following
undirected graph::

              G

             (0)
            /   \
           1     2
          /       \
        (2)       (1)

This graph has three nodes, where node 0 and 1 are connected by an edge of
weight 2, and nodes 0 and 2 are connected by an edge of weight 1.

"""
# 导入 NumPy 库
import numpy as np
# 创建一个密集表示的图 G_dense，这里是一个 3x3 的对称矩阵
G_dense = np.array([[0, 2, 1],
                    [2, 0, 0],
                    [1, 0, 0]])
# 使用 G_dense 创建一个掩码表示的图 G_masked，将值为 0 的元素掩盖起来
G_masked = np.ma.masked_values(G_dense, 0)
# 导入 scipy 库中的 csr_matrix 类
from scipy.sparse import csr_matrix
# 使用 G_dense 创建一个稀疏表示的图 G_sparse，将其转换为 CSR 格式的稀疏矩阵
G_sparse = csr_matrix(G_dense)

# 当零权重边很重要时，使用掩码或稀疏表示来消除歧义是更为困难的。例如，考虑稍作修改的图 G2：
#
#          G2
#
#          (0)
#         /   \
#        0     2
#       /       \
#     (2)       (1)
#
# 这个图与上面的图相同，除了节点 0 和 2 之间连接了一个权重为零的边。在这种情况下，
# 上面的密集表示会导致歧义：如果零是一个有意义的值，那么如何表示不存在的边？
# 在这种情况下，必须使用掩码或稀疏表示来消除歧义：
#
# 导入 NumPy 库
import numpy as np
# 创建一个包含无穷大的数据数组 G2_data
G2_data = np.array([[np.inf, 2,      0     ],
                    [2,      np.inf, np.inf],
                    [0,      np.inf, np.inf]])
# 使用掩码将无效值掩盖起来，生成 G2_masked
G2_masked = np.ma.masked_invalid(G2_data)
# 从 scipy.sparse.csgraph 模块中导入 csgraph_from_dense 函数
from scipy.sparse.csgraph import csgraph_from_dense
# 使用 csgraph_from_dense 将密集表示的图 G2_data 转换为稀疏表示的图 G2_sparse，
# null_value 指定了在图中表示空值的标志
G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
# 查看 G2_sparse 中的数据数组，显示了图中明确编码的零值
G2_sparse.data
array([ 2.,  0.,  2.,  0.])

# 这里使用 csgraph 子模块中的实用程序例程，将密集表示转换为稀疏表示，
# 以便 csgraph 子模块中的算法可以理解。通过查看数据数组，我们可以看到
# 零值在图中是显式编码的。

# 有向 vs. 无向
# ^^^^^^^^^^^^^^
# 矩阵可以表示有向或无向图。这在整个 csgraph 模块中由布尔关键字指定。
# 默认情况下假定图是有向的。在有向图中，从节点 i 到节点 j 的遍历可以通过
# 边 G[i, j] 完成，但不能通过边 G[j, i] 完成。考虑以下密集图：
#
# 导入 NumPy 库
import numpy as np
# 创建一个密集表示的图 G_dense
G_dense = np.array([[0, 1, 0],
                    [2, 0, 3],
                    [0, 4, 0]])

# 当 directed=True 时，我们得到如下的图：
#
#    ---1--> ---3-->
#  (0)     (1)     (2)
#    <--2--- <--4---
#
# 在非有向图中，从节点 i 到节点 j 的遍历可以通过 G[i, j] 或 G[j, i] 中的任一边完成。
# 如果两条边都不是空值，并且它们的权重不相等，则选择较小的那个。

# 因此，对于同一图，当 directed=False 时，我们得到如下的图：
#
# (0)--1--(1)--3--(2)
#
# 注意，对称矩阵将表示一个无向图，无论 'directed' 关键字设置为 True 还是 False。
# 在这种情况下，通常使用 directed=True 可以带来更高效的计算。
"""
The routines in this module accept as input either scipy.sparse representations
(csr, csc, or lil format), masked representations, or dense representations
with non-edges indicated by zeros, infinities, and NaN entries.
"""  # noqa: E501

__docformat__ = "restructuredtext en"

__all__ = ['connected_components',
           'laplacian',
           'shortest_path',
           'floyd_warshall',
           'dijkstra',
           'bellman_ford',
           'johnson',
           'yen',
           'breadth_first_order',
           'depth_first_order',
           'breadth_first_tree',
           'depth_first_tree',
           'minimum_spanning_tree',
           'reverse_cuthill_mckee',
           'maximum_flow',
           'maximum_bipartite_matching',
           'min_weight_full_bipartite_matching',
           'structural_rank',
           'construct_dist_matrix',
           'reconstruct_path',
           'csgraph_masked_from_dense',
           'csgraph_from_dense',
           'csgraph_from_masked',
           'csgraph_to_dense',
           'csgraph_to_masked',
           'NegativeCycleError']

# 导入 laplacian 函数，用于计算图的拉普拉斯矩阵
from ._laplacian import laplacian
# 导入 shortest_path 系列函数，包括多种最短路径算法
from ._shortest_path import (
    shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, yen,
    NegativeCycleError
)
# 导入遍历算法函数，用于图的广度优先和深度优先遍历
from ._traversal import (
    breadth_first_order, depth_first_order, breadth_first_tree,
    depth_first_tree, connected_components
)
# 导入最小生成树算法函数
from ._min_spanning_tree import minimum_spanning_tree
# 导入最大流算法函数
from ._flow import maximum_flow
# 导入二部图匹配算法函数
from ._matching import (
    maximum_bipartite_matching, min_weight_full_bipartite_matching
)
# 导入重新排序算法函数
from ._reordering import reverse_cuthill_mckee, structural_rank
# 导入工具函数，用于稀疏矩阵和密集矩阵的转换及路径重构
from ._tools import (
    construct_dist_matrix, reconstruct_path, csgraph_from_dense,
    csgraph_to_dense, csgraph_masked_from_dense, csgraph_from_masked,
    csgraph_to_masked
)

# 导入测试工具类并创建测试实例
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 删除不再需要的测试类引用，以免污染命名空间
del PytestTester
```