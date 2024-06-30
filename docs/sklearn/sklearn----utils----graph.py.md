# `D:\src\scipysrc\scikit-learn\sklearn\utils\graph.py`

```
"""Graph utilities and algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..metrics.pairwise import pairwise_distances
from ._param_validation import Integral, Interval, validate_params


###############################################################################
# Path and connected component analysis.
# Code adapted from networkx
@validate_params(
    {
        "graph": ["array-like", "sparse matrix"],
        "source": [Interval(Integral, 0, None, closed="left")],
        "cutoff": [Interval(Integral, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def single_source_shortest_path_length(graph, source, *, cutoff=None):
    """Return the length of the shortest path from source to all reachable nodes.

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_nodes, n_nodes)
        Adjacency matrix of the graph. Sparse matrix of format LIL is
        preferred.

    source : int
       Start node for path.

    cutoff : int, default=None
        Depth to stop the search - only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dict
        Reachable end nodes mapped to length of path from source,
        i.e. `{end: path_length}`.

    Examples
    --------
    >>> from sklearn.utils.graph import single_source_shortest_path_length
    >>> import numpy as np
    >>> graph = np.array([[ 0, 1, 0, 0],
    ...                   [ 1, 0, 1, 0],
    ...                   [ 0, 1, 0, 0],
    ...                   [ 0, 0, 0, 0]])
    >>> single_source_shortest_path_length(graph, 0)
    {0: 0, 1: 1, 2: 2}
    >>> graph = np.ones((6, 6))
    >>> sorted(single_source_shortest_path_length(graph, 2).items())
    [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
    """
    if sparse.issparse(graph):
        # 转换稀疏矩阵为LIL格式以便进行修改
        graph = graph.tolil()
    else:
        # 将稠密矩阵转换为LIL格式
        graph = sparse.lil_matrix(graph)
    seen = {}  # level (number of hops) when seen in BFS
    level = 0  # the current level
    next_level = [source]  # dict of nodes to check at next level
    while next_level:
        this_level = next_level  # advance to next level
        next_level = set()  # and start a new list (fringe)
        for v in this_level:
            if v not in seen:
                seen[v] = level  # set the level of vertex v
                next_level.update(graph.rows[v])
        if cutoff is not None and cutoff <= level:
            break
        level += 1
    return seen  # return all path lengths as dictionary


def _fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    """Add connections to sparse graph to connect unconnected components.

    For each pair of unconnected components, compute all pairwise distances
    from one component to the other, and add a connection on the closest pair
    # 如果使用预先计算的距离矩阵并且 X 是稀疏矩阵，抛出运行时错误
    if metric == "precomputed" and sparse.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    # 遍历每个连通分量
    for i in range(n_connected_components):
        # 获取第 i 个连通分量的索引
        idx_i = np.flatnonzero(component_labels == i)
        # 提取第 i 个连通分量对应的样本特征
        Xi = X[idx_i]

        # 遍历前面的连通分量 j
        for j in range(i):
            # 获取第 j 个连通分量的索引
            idx_j = np.flatnonzero(component_labels == j)
            # 提取第 j 个连通分量对应的样本特征
            Xj = X[idx_j]

            # 根据指定的度量计算样本 Xi 和 Xj 之间的距离矩阵 D
            if metric == "precomputed":
                # 如果度量为预先计算的矩阵，则直接使用预先计算的距离矩阵
                D = X[np.ix_(idx_i, idx_j)]
            else:
                # 否则根据指定的度量和参数计算样本 Xi 和 Xj 之间的距离矩阵
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            # 在图中设置连接信息，根据 mode 参数设置不同的方式
            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                # 如果模式为 'connectivity'，则将连接设置为 1
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                # 如果模式为 'distance'，则将连接设置为样本之间的距离
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                # 如果模式不是 'connectivity' 或 'distance'，抛出错误
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    # 返回具有单个连接分量的样本连接图
    return graph
```