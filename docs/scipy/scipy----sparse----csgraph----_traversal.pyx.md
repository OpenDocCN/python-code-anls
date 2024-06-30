# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_traversal.pyx`

```
"""
Routines for traversing graphs in compressed sparse format
"""

# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD, (C) 2012

import numpy as np
cimport numpy as np  # 导入并使用Cython定义的NumPy接口

from scipy.sparse.csgraph._validation import validate_graph
from scipy.sparse.csgraph._tools import reconstruct_path

cimport cython  # 导入Cython扩展模块

np.import_array()  # 调用NumPy的import_array函数

include 'parameters.pxi'  # 包含参数文件

def connected_components(csgraph, directed=True, connection='weak',
                         return_labels=True):
    """
    connected_components(csgraph, directed=True, connection='weak',
                         return_labels=True)

    Analyze the connected components of a sparse graph

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like or sparse matrix
        The N x N matrix representing the compressed sparse graph.  The input
        csgraph will be converted to csr format for the calculation.
    directed : bool, optional
        If True (default), then operate on a directed graph: only
        move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].
    connection : str, optional
        ['weak'|'strong'].  For directed graphs, the type of connection to
        use.  Nodes i and j are strongly connected if a path exists both
        from i to j and from j to i. A directed graph is weakly connected
        if replacing all of its directed edges with undirected edges produces
        a connected (undirected) graph. If directed == False, this keyword
        is not referenced.
    return_labels : bool, optional
        If True (default), then return the labels for each of the connected
        components.

    Returns
    -------
    n_components: int
        The number of connected components.
    labels: ndarray
        The length-N array of labels of the connected components.

    References
    ----------
    .. [1] D. J. Pearce, "An Improved Algorithm for Finding the Strongly
           Connected Components of a Directed Graph", Technical Report, 2005

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import connected_components

    >>> graph = [
    ... [0, 1, 1, 0, 0],
    ... [0, 0, 1, 0, 0],
    ... [0, 0, 0, 0, 0],
    ... [0, 0, 0, 0, 1],
    ... [0, 0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
      (np.int32(0), np.int32(1))    1
      (np.int32(0), np.int32(2))    1
      (np.int32(1), np.int32(2))    1
      (np.int32(3), np.int32(4))    1

    >>> n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    >>> n_components
    2
    >>> labels
    array([0, 0, 0, 1, 1], dtype=int32)

    """
    if connection.lower() not in ['weak', 'strong']:
        raise ValueError("connection must be 'weak' or 'strong'")
    # 如果连接类型为弱连接（即无向图的组件）
    if connection.lower() == 'weak':
        # 设置图的方向为无向
        directed = False

    # 验证图的有效性，并根据需要进行稠密输出
    csgraph = validate_graph(csgraph, directed,
                             dtype=csgraph.dtype,
                             dense_output=False)

    # 创建一个标签数组，用于存储每个节点的组件标识符
    labels = np.empty(csgraph.shape[0], dtype=ITYPE)
    labels.fill(NULL_IDX)

    # 如果图是有向的，则计算有向图的连通组件数
    if directed:
        n_components = _connected_components_directed(csgraph.indices,
                                                      csgraph.indptr,
                                                      labels)
    else:
        # 如果图是无向的，转置图并计算其连通组件数
        csgraph_T = csgraph.T.tocsr()
        n_components = _connected_components_undirected(csgraph.indices,
                                                        csgraph.indptr,
                                                        csgraph_T.indices,
                                                        csgraph_T.indptr,
                                                        labels)

    # 如果需要返回节点标签，则返回连通组件数和标签数组
    if return_labels:
        return n_components, labels
    else:
        # 否则，只返回连通组件数
        return n_components
# 定义一个函数，使用广度优先搜索生成一个树
def breadth_first_tree(csgraph, i_start, directed=True):
    r"""
    breadth_first_tree(csgraph, i_start, directed=True)

    返回由广度优先搜索生成的树

    注意，从指定节点生成的广度优先树是唯一的。

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like or sparse matrix
        表示压缩稀疏图的 N x N 矩阵。输入的 csgraph 将被转换为 CSR 格式进行计算。
    i_start : int
        起始节点的索引。
    directed : bool, optional
        如果为 True（默认），则操作在有向图上：只能沿着 csgraph[i, j] 的路径从节点 i 移动到节点 j。
        如果为 False，则在无向图上查找最短路径：算法可以沿着 csgraph[i, j] 或 csgraph[j, i] 前进。

    Returns
    -------
    cstree : csr matrix
        从 csgraph 开始的广度优先树的 N x N 有向压缩稀疏表示。

    Notes
    -----
    如果存在多个有效解决方案，则输出可能随着 SciPy 和 Python 版本而变化。

    Examples
    --------
    下面的示例展示了从节点 0 开始计算简单四部件图的广度优先树的过程：

         输入图           从 (0) 开始的广度优先树

         (0)                        (0)
        /   \                      /   \
       3     8                    3     8
      /       \                  /       \
    (3)---5---(1)             (3)       (1)
      \       /                          /
       6     2                          2
        \   /                          /
         (2)                        (2)

    在压缩稀疏表示中，解决方案看起来像这样：

    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import breadth_first_tree
    >>> X = csr_matrix([[0, 8, 0, 3],
    ...                 [0, 0, 2, 5],
    ...                 [0, 0, 0, 6],
    ...                 [0, 0, 0, 0]])
    >>> Tcsr = breadth_first_tree(X, 0, directed=False)
    >>> Tcsr.toarray().astype(int)
    array([[0, 8, 0, 3],
           [0, 0, 2, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])

    结果图是一个跨越整个图的有向无环图。从给定节点开始的广度优先树是唯一的。
    """
    # 调用广度优先搜索函数，获取节点列表和前驱节点信息
    _node_list, predecessors = breadth_first_order(csgraph, i_start,
                                                   directed, True)
    # 使用前驱节点信息重构路径并返回结果
    return reconstruct_path(csgraph, predecessors, directed)
    # 依赖于节点搜索时子节点的顺序。

    .. versionadded:: 0.11.0
    # 添加版本说明，表明此函数是在 0.11.0 版本中添加的。

    Parameters
    ----------
    csgraph : array_like or sparse matrix
        压缩稀疏图的 N x N 矩阵。输入的 csgraph 将被转换为 CSR 格式进行计算。
    i_start : int
        起始节点的索引。
    directed : bool, optional
        如果为 True（默认），则操作的是有向图：只能沿着 csgraph[i, j] 的路径从点 i 移动到点 j。
        如果为 False，则在无向图上找到最短路径：算法可以沿着 csgraph[i, j] 或 csgraph[j, i] 进行进展。

    Returns
    -------
    cstree : csr matrix
        从指定节点开始，从 csgraph 绘制的深度优先树的 N x N 有向压缩稀疏表示。

    Notes
    -----
    如果存在多个有效的解决方案，则输出可能会随着 SciPy 和 Python 版本的不同而有所不同。

    Examples
    --------
    下面的示例展示了在一个简单的四部件图上计算从节点 0 开始的深度优先树的过程::

         输入图                   从 (0) 开始的深度优先树

             (0)                         (0)
            /   \                           \
           3     8                           8
          /       \                           \
        (3)---5---(1)               (3)       (1)
          \       /                   \       /
           6     2                     6     2
            \   /                       \   /
             (2)                         (2)

    在压缩稀疏表示中，解决方案如下：

    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import depth_first_tree
    >>> X = csr_matrix([[0, 8, 0, 3],
    ...                 [0, 0, 2, 5],
    ...                 [0, 0, 0, 6],
    ...                 [0, 0, 0, 0]])
    >>> Tcsr = depth_first_tree(X, 0, directed=False)
    >>> Tcsr.toarray().astype(int)
    array([[0, 8, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 6],
           [0, 0, 0, 0]])

    注意，结果图是一个跨越整个图的有向无环图（DAG）。与广度优先树不同，给定图的深度优先树在图中包含循环时并不唯一。如果上面的解决方案从连接节点 0 和 3 的边开始，则结果将不同。
    """
    _node_list, predecessors = depth_first_order(csgraph, i_start,
                                                 directed, True)
    # 使用深度优先搜索算法获取节点列表和前驱节点信息
    return reconstruct_path(csgraph, predecessors, directed)
    # 返回重构的路径，这里是基于前驱节点信息和图的方向性。
# 以广度优先顺序遍历图结构，从指定节点开始
cpdef breadth_first_order(csgraph, i_start,
                          directed=True, return_predecessors=True):
    """
    breadth_first_order(csgraph, i_start, directed=True, return_predecessors=True)

    Return a breadth-first ordering starting with specified node.

    Note that a breadth-first order is not unique, but the tree which it
    generates is unique.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like or sparse matrix
        The N x N compressed sparse graph.  The input csgraph will be
        converted to csr format for the calculation.
    i_start : int
        The index of starting node.
    directed : bool, optional
        If True (default), then operate on a directed graph: only
        move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].
    return_predecessors : bool, optional
        If True (default), then return the predecessor array (see below).

    Returns
    -------
    node_array : ndarray, one dimension
        The breadth-first list of nodes, starting with specified node.  The
        length of node_array is the number of nodes reachable from the
        specified node.
    predecessors : ndarray, one dimension
        Returned only if return_predecessors is True.
        The length-N list of predecessors of each node in a breadth-first
        tree.  If node i is in the tree, then its parent is given by
        predecessors[i]. If node i is not in the tree (and for the parent
        node) then predecessors[i] = -9999.

    Notes
    -----
    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import breadth_first_order

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [2, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
      (np.int32(0), np.int32(1))    1
      (np.int32(0), np.int32(2))    2
      (np.int32(1), np.int32(3))    1
      (np.int32(2), np.int32(0))    2
      (np.int32(2), np.int32(3))    3

    >>> breadth_first_order(graph,0)
    (array([0, 1, 2, 3], dtype=int32), array([-9999,     0,     0,     1], dtype=int32))

    """
    # 验证图结构的有效性，将其转换为 CSR 格式以进行计算
    csgraph = validate_graph(csgraph, directed, dense_output=False)
    # 获取图结构中节点的数量
    cdef int N = csgraph.shape[0]

    # 初始化存储节点顺序的数组和前驱节点数组
    cdef np.ndarray node_list = np.empty(N, dtype=ITYPE)
    cdef np.ndarray predecessors = np.empty(N, dtype=ITYPE)
    node_list.fill(NULL_IDX)
    predecessors.fill(NULL_IDX)

    # 如果是有向图，调用有向图的广度优先搜索函数
    if directed:
        length = _breadth_first_directed(i_start,
                                csgraph.indices, csgraph.indptr,
                                node_list, predecessors)
    # 如果设置了 return_predecessors 参数为 True，则返回节点列表和前驱节点字典
    if return_predecessors:
        # 返回节点列表的前 length 个元素和前驱节点字典
        return node_list[:length], predecessors
    else:
        # 否则只返回节点列表的前 length 个元素
        return node_list[:length]
# 使用广度优先搜索算法遍历有向图，生成节点的广度优先列表和节点的前驱列表
def _breadth_first_directed(
        unsigned int head_node,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors):
    # 调用内部函数进行实际的广度优先搜索
    return _breadth_first_directed2(head_node, indices, indptr,
                                    node_list, predecessors)


cdef unsigned int _breadth_first_directed2(
        unsigned int head_node,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors) noexcept:
    # Inputs:
    #  head_node: (input) 开始遍历的节点索引
    #  indices: (input) 图的 CSR 索引
    #  indptr:  (input) 图的 CSR indptr
    #  node_list: (output) 节点的广度优先遍历列表
    #  predecessors: (output) 节点在广度优先树中的前驱列表，应初始化为 NULL_IDX
    # Returns:
    #  n_nodes: 广度优先树中节点的数量
    cdef unsigned int i, pnode, cnode
    cdef unsigned int i_nl, i_nl_end

    node_list[0] = head_node
    i_nl = 0
    i_nl_end = 1

    while i_nl < i_nl_end:
        pnode = node_list[i_nl]

        for i in range(indptr[pnode], indptr[pnode + 1]):
            cnode = indices[i]
            if (cnode == head_node):
                continue
            elif (predecessors[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                predecessors[cnode] = pnode
                i_nl_end += 1

        i_nl += 1

    return i_nl


# 使用广度优先搜索算法遍历无向图，生成节点的广度优先列表和节点的前驱列表
def _breadth_first_undirected(
        unsigned int head_node,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices1,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr1,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indices2,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indptr2,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors):
    # 调用内部函数进行实际的广度优先搜索
    return _breadth_first_undirected2(head_node, indices1, indptr1, indices2,
                                      indptr2, node_list, predecessors)


cdef unsigned int _breadth_first_undirected2(
        unsigned int head_node,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices1,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr1,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indices2,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indptr2,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors) noexcept:
    # Inputs:
    #  head_node: (input) 开始遍历的节点索引
    #  indices1: (input) 图的第一组 CSR 索引
    #  indptr1:  (input) 图的第一组 CSR indptr
    #  indices2: (input) 图的第二组 CSR 索引
    #  indptr2:  (input) 图的第二组 CSR indptr
    #  node_list: (output) 节点的广度优先遍历列表
    #  predecessors: (output) 节点在广度优先树中的前驱列表，应初始化为 NULL_IDX
    #  indices2: (input) CSR indices of transposed graph
    #  indptr2:  (input) CSR indptr of transposed graph
    #  node_list: (output) breadth-first list of nodes
    #  predecessors: (output) list of predecessors of nodes in breadth-first
    #                tree.  Should be initialized to NULL_IDX
    # Returns:
    #  n_nodes: the number of nodes in the breadth-first tree
    cdef unsigned int i, pnode, cnode  # 声明无符号整数变量 i, pnode, cnode
    cdef unsigned int i_nl, i_nl_end   # 声明无符号整数变量 i_nl, i_nl_end
    
    node_list[0] = head_node  # 将起始节点 head_node 放入节点列表 node_list 的第一个位置
    i_nl = 0  # 初始化广度优先遍历的节点列表索引
    i_nl_end = 1  # 初始化广度优先遍历的节点列表末尾索引
    
    while i_nl < i_nl_end:  # 开始广度优先遍历的循环，直到索引 i_nl 达到末尾索引 i_nl_end
        pnode = node_list[i_nl]  # 获取当前广度优先遍历的父节点 pnode
    
        for i in range(indptr1[pnode], indptr1[pnode + 1]):  # 遍历父节点 pnode 的邻接节点
            cnode = indices1[i]  # 获取当前邻接节点 cnode
            if (cnode == head_node):  # 如果邻接节点是起始节点 head_node，则跳过
                continue
            elif (predecessors[cnode] == NULL_IDX):  # 如果邻接节点的前驱节点为 NULL_IDX，则更新前驱节点信息
                node_list[i_nl_end] = cnode  # 将邻接节点 cnode 加入节点列表 node_list
                predecessors[cnode] = pnode  # 更新邻接节点 cnode 的前驱节点为 pnode
                i_nl_end += 1  # 更新广度优先遍历的末尾索引
    
        for i in range(indptr2[pnode], indptr2[pnode + 1]):  # 遍历父节点 pnode 在转置图中的邻接节点
            cnode = indices2[i]  # 获取当前转置图中的邻接节点 cnode
            if (cnode == head_node):  # 如果邻接节点是起始节点 head_node，则跳过
                continue
            elif (predecessors[cnode] == NULL_IDX):  # 如果邻接节点的前驱节点为 NULL_IDX，则更新前驱节点信息
                node_list[i_nl_end] = cnode  # 将邻接节点 cnode 加入节点列表 node_list
                predecessors[cnode] = pnode  # 更新邻接节点 cnode 的前驱节点为 pnode
                i_nl_end += 1  # 更新广度优先遍历的末尾索引
    
        i_nl += 1  # 更新广度优先遍历的当前节点索引
    
    return i_nl  # 返回广度优先遍历树中的节点数目
    # 验证并转换输入的图形数据，确保符合要求的稀疏矩阵格式
    csgraph = validate_graph(csgraph, directed, dense_output=False)
    # 获取图的节点数目
    cdef int N = csgraph.shape[0]

    # 创建一个空的节点列表，用于存储深度优先遍历的结果，数据类型为ITYPE
    node_list = np.empty(N, dtype=ITYPE)
    # 创建一个空的前驱节点列表，用于存储深度优先树中每个节点的父节点，数据类型为ITYPE
    predecessors = np.empty(N, dtype=ITYPE)
    # 创建一个空的根节点列表，用于存储深度优先遍历的起始节点，数据类型为ITYPE
    root_list = np.empty(N, dtype=ITYPE)
    # 创建一个标志数组，用于标记节点是否已经被访问过，数据类型为ITYPE
    flag = np.zeros(N, dtype=ITYPE)
    # 将节点列表、前驱节点列表和根节点列表初始化为NULL_IDX
    node_list.fill(NULL_IDX)
    predecessors.fill(NULL_IDX)
    root_list.fill(NULL_IDX)
    # 如果图是有向的，则执行深度优先搜索
    if directed:
        # 调用深度优先搜索函数，返回从起始节点开始的深度优先遍历路径长度
        length = _depth_first_directed(i_start,
                              csgraph.indices, csgraph.indptr,
                              node_list, predecessors,
                              root_list, flag)
    else:
        # 如果图是无向的，先获取其转置图的 CSR 表示
        csgraph_T = csgraph.T.tocsr()
        # 调用无向图的深度优先搜索函数，返回从起始节点开始的深度优先遍历路径长度
        length = _depth_first_undirected(i_start,
                                         csgraph.indices, csgraph.indptr,
                                         csgraph_T.indices, csgraph_T.indptr,
                                         node_list, predecessors,
                                         root_list, flag)

    # 如果需要返回前驱节点列表，则返回节点列表和前驱节点字典
    if return_predecessors:
        return node_list[:length], predecessors
    else:
        # 否则，只返回节点列表
        return node_list[:length]
# 深度优先搜索（DFS）算法，针对有向图进行遍历
def _depth_first_directed(
        unsigned int head_node,  # 起始节点
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices,  # 边的目标节点数组
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr,  # 节点到边索引的指针数组
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,  # 存储访问顺序的节点列表
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors,  # 存储每个节点的前驱节点
        np.ndarray[ITYPE_t, ndim=1, mode='c'] root_list,  # 当前处理节点的根节点列表
        np.ndarray[ITYPE_t, ndim=1, mode='c'] flag):  # 节点访问标志数组
    # 调用内部的有向图深度优先搜索函数
    return _depth_first_directed2(head_node, indices, indptr,
                                  node_list, predecessors,
                                  root_list, flag)


# 内部函数：深度优先搜索（DFS）算法，有向图版本
cdef unsigned int _depth_first_directed2(
        unsigned int head_node,  # 起始节点
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices,  # 边的目标节点数组
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr,  # 节点到边索引的指针数组
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,  # 存储访问顺序的节点列表
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors,  # 存储每个节点的前驱节点
        np.ndarray[ITYPE_t, ndim=1, mode='c'] root_list,  # 当前处理节点的根节点列表
        np.ndarray[ITYPE_t, ndim=1, mode='c'] flag) noexcept:  # 节点访问标志数组，不抛出异常
    cdef unsigned int i, i_nl_end, cnode, pnode
    cdef unsigned int N = node_list.shape[0]
    cdef int no_children, i_root

    node_list[0] = head_node  # 将起始节点存入节点列表
    root_list[0] = head_node  # 将起始节点存入根节点列表
    i_root = 0  # 初始化根节点列表的索引
    i_nl_end = 1  # 初始化节点列表的末尾索引
    flag[head_node] = 1  # 标记起始节点已访问

    while i_root >= 0:  # 当根节点列表不为空时进行循环
        pnode = root_list[i_root]  # 获取当前处理的根节点
        no_children = True  # 假设当前节点没有未访问的子节点
        for i in range(indptr[pnode], indptr[pnode + 1]):  # 遍历当前节点的所有边
            cnode = indices[i]  # 获取目标节点
            if flag[cnode]:  # 如果目标节点已访问过，则跳过
                continue
            else:
                i_root += 1  # 将目标节点加入根节点列表
                root_list[i_root] = cnode
                node_list[i_nl_end] = cnode  # 将目标节点加入节点列表
                predecessors[cnode] = pnode  # 设置目标节点的前驱节点为当前节点
                flag[cnode] = 1  # 标记目标节点已访问
                i_nl_end += 1  # 更新节点列表末尾索引
                no_children = False  # 更新标记，说明当前节点有未访问的子节点
                break

        if i_nl_end == N:  # 如果节点列表已满，结束循环
            break

        if no_children:  # 如果当前节点没有未访问的子节点
            i_root -= 1  # 从根节点列表中移除当前节点

    return i_nl_end  # 返回节点列表的末尾索引，表示遍历完成


# 深度优先搜索（DFS）算法，针对无向图进行遍历
def _depth_first_undirected(
        unsigned int head_node,  # 起始节点
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices1,  # 边的目标节点数组1
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr1,  # 节点到边索引的指针数组1
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indices2,  # 边的目标节点数组2
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indptr2,  # 节点到边索引的指针数组2
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,  # 存储访问顺序的节点列表
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors,  # 存储每个节点的前驱节点
        np.ndarray[ITYPE_t, ndim=1, mode='c'] root_list,  # 当前处理节点的根节点列表
        np.ndarray[ITYPE_t, ndim=1, mode='c'] flag):  # 节点访问标志数组
    # 调用内部的无向图深度优先搜索函数
    return _depth_first_undirected2(head_node, indices1, indptr1,
                                    indices2, indptr2,
                                    node_list, predecessors,
                                    root_list, flag)
cdef unsigned int _depth_first_undirected2(
        unsigned int head_node,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices1,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr1,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indices2,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indptr2,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] root_list,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] flag) noexcept:
    """
    Implements a depth-first search (DFS) algorithm for an undirected graph.

    Args:
    - head_node: Starting node for the DFS traversal.
    - indices1, indptr1: Compressed sparse row (CSR) representation of graph adjacency matrix for component 1.
    - indices2, indptr2: CSR representation of graph adjacency matrix for component 2.
    - node_list: Array to store nodes visited during DFS.
    - predecessors: Array to store predecessor nodes during DFS.
    - root_list: Array to store the current nodes being processed in the DFS.
    - flag: Array to mark nodes as visited.

    Returns:
    - i_nl_end: Number of nodes visited during the DFS traversal.
    """
    cdef unsigned int i, i_nl_end, cnode, pnode
    cdef unsigned int N = node_list.shape[0]
    cdef int no_children, i_root

    # Initialize the DFS with the head node
    node_list[0] = head_node
    root_list[0] = head_node
    i_root = 0
    i_nl_end = 1
    flag[head_node] = 1

    # Perform DFS traversal
    while i_root >= 0:
        pnode = root_list[i_root]
        no_children = True

        # Traverse through adjacency list 1 of the current node
        for i in range(indptr1[pnode], indptr1[pnode + 1]):
            cnode = indices1[i]
            if flag[cnode]:
                continue
            else:
                # Process a new child node found
                i_root += 1
                root_list[i_root] = cnode
                node_list[i_nl_end] = cnode
                predecessors[cnode] = pnode
                flag[cnode] = 1
                i_nl_end += 1
                no_children = False
                break

        # If no unvisited children found, traverse through adjacency list 2
        if no_children:
            for i in range(indptr2[pnode], indptr2[pnode + 1]):
                cnode = indices2[i]
                if flag[cnode]:
                    continue
                else:
                    # Process a new child node found
                    i_root += 1
                    root_list[i_root] = cnode
                    node_list[i_nl_end] = cnode
                    predecessors[cnode] = pnode
                    flag[cnode] = 1
                    i_nl_end += 1
                    no_children = False
                    break

        # If all nodes are visited, exit the loop
        if i_nl_end == N:
            break

        # If no unvisited children found, backtrack
        if no_children:
            i_root -= 1

    return i_nl_end


def _connected_components_directed(
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] labels):
    """
    Wrapper function to call _connected_components_directed2.

    Args:
    - indices: CSR representation of graph adjacency matrix.
    - indptr: CSR representation of graph adjacency matrix.
    - labels: Array to store component labels.

    Returns:
    - Result from _connected_components_directed2 function call.
    """
    return _connected_components_directed2(indices, indptr, labels)


cdef int _connected_components_directed2(
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] labels) noexcept:
    """
    Implements an iterative version of Tarjan's algorithm to find strongly connected components
    in a directed graph represented as a sparse matrix.

    Args:
    - indices: CSR representation of graph adjacency matrix.
    - indptr: CSR representation of graph adjacency matrix.
    - labels: Array to store component labels.

    Returns:
    - Number of strongly connected components found.
    """
    # Function implementation is provided in a separate file or documentation.
    pass
    # 定义一些变量和常量，用于算法中的节点处理和索引
    cdef int v, w, index, low_v, low_w, label, j
    cdef int SS_head, root, stack_head, f, b
    DEF VOID = -1  # 表示未定义状态的常量
    DEF END = -2   # 表示结束状态的常量

    # 从输入的数组中获取节点数量
    cdef int N = labels.shape[0]

    # 定义几个数组，存储算法中需要用到的状态和信息
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] SS, lowlinks, stack_f, stack_b

    # 将低链接数组初始化为输入的标签数组
    lowlinks = labels

    # 创建节点状态数组 SS 和两个栈数组 stack_f 和 stack_b
    SS = np.ndarray((N,), dtype=ITYPE)
    stack_b = np.ndarray((N,), dtype=ITYPE)
    stack_f = SS

    # 将 SS 数组填充为 VOID（未定义状态）
    SS.fill(VOID)
    SS_head = END  # 设置 SS 头部指针为 END

    # 将低链接数组填充为 VOID（未定义状态）
    lowlinks.fill(VOID)

    # 初始化 DFS 栈的头部指针和两个栈数组的填充状态
    stack_head = END
    stack_f.fill(VOID)
    stack_b.fill(VOID)

    # 初始化索引和标签，用于追踪 SCC 标签
    index = 0
    label = N - 1  # 标签从 N-1 开始递减，避免与低链接值冲突
    for v in range(N):
        if lowlinks[v] == VOID:
            # 如果 lowlinks[v] 是 VOID，则表示节点 v 尚未访问过，执行以下操作
            stack_head = v
            stack_f[v] = END
            stack_b[v] = END
            while stack_head != END:
                v = stack_head
                if lowlinks[v] == VOID:
                    # 如果 lowlinks[v] 是 VOID，表示节点 v 第一次访问
                    lowlinks[v] = index
                    index += 1

                    # 添加后继节点
                    for j in range(indptr[v], indptr[v+1]):
                        w = indices[j]
                        if lowlinks[w] == VOID:
                            # 如果 lowlinks[w] 是 VOID，表示节点 w 还未访问
                            with cython.boundscheck(False):
                                # 禁用边界检查，提高性能
                                if stack_f[w] != VOID:
                                    # 如果 w 已在栈中，将其移除
                                    f = stack_f[w]
                                    b = stack_b[w]
                                    if b != END:
                                        stack_f[b] = f
                                    if f != END:
                                        stack_b[f] = b

                                # 将 w 入栈
                                stack_f[w] = stack_head
                                stack_b[w] = END
                                stack_b[stack_head] = w
                                stack_head = w

                else:
                    # 如果 lowlinks[v] 不是 VOID，表示节点 v 已经访问过，执行栈的出栈操作
                    stack_head = stack_f[v]
                    if stack_head >= 0:
                        stack_b[stack_head] = END
                    stack_f[v] = VOID
                    stack_b[v] = VOID

                    root = 1  # True
                    low_v = lowlinks[v]
                    for j in range(indptr[v], indptr[v+1]):
                        # 找到 v 的所有邻居节点的最小 lowlinks 值
                        low_w = lowlinks[indices[j]]
                        if low_w < low_v:
                            low_v = low_w
                            root = 0  # False
                    lowlinks[v] = low_v

                    if root:  # 找到一个根节点
                        index -= 1
                        # 将所有在栈 SS 中的节点弹出，直到遇到一个 lowlinks[v] <= lowlinks[SS_head] 的节点
                        while SS_head != END and lowlinks[v] <= lowlinks[SS_head]:
                            w = SS_head  # w = pop(S)
                            SS_head = SS[w]
                            SS[w] = VOID

                            labels[w] = label  # 设置节点 w 的 label 值为当前 label
                            index -= 1         # index 减一
                        labels[v] = label  # 设置节点 v 的 label 值为当前 label
                        label -= 1         # label 减一
                    else:
                        # 将节点 v 入栈 SS
                        SS[v] = SS_head
                        SS_head = v

    # labels 值从 N-1 到 0 递减，修改它们使其从 0 开始递增
    labels *= -1
    labels += (N - 1)
    # 返回节点标签的数组，其长度为 N，最大值为 N-1
    return (N - 1) - label
def _connected_components_undirected(
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices1,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr1,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indices2,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indptr2,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] labels):
    # 调用底层函数进行无向图的连通组件查找
    return _connected_components_undirected2(indices1, indptr1,
                                             indices2, indptr2,
                                             labels)


cdef int _connected_components_undirected2(
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indices1,
        np.ndarray[int32_or_int64, ndim=1, mode='c'] indptr1,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indices2,
        np.ndarray[int32_or_int64_b, ndim=1, mode='c'] indptr2,
        np.ndarray[ITYPE_t, ndim=1, mode='c'] labels) noexcept:

    cdef int v, w, j, label, SS_head
    cdef int N = labels.shape[0]
    DEF VOID = -1
    DEF END = -2

    # 将标签数组填充为 VOID
    labels.fill(VOID)
    label = 0

    # 共享内存用于堆栈和标签，因为标签只有在节点从堆栈中弹出时才被应用。
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] SS = labels
    SS_head = END

    # 遍历所有节点
    for v in range(N):
        if labels[v] == VOID:
            # 初始化堆栈头部为当前节点 v
            SS_head = v
            # 将 SS[v] 标记为 END，表示当前节点 v 已经被访问
            SS[v] = END

            # 使用深度优先搜索（DFS）遍历连通分量
            while SS_head != END:
                # 弹出堆栈顶部节点 v
                v = SS_head
                SS_head = SS[v]

                # 将当前节点标记为当前连通分量的标签
                labels[v] = label

                # 将未访问过的子节点压入堆栈
                # 处理 indices1 中 v 节点的邻接节点
                for j in range(indptr1[v], indptr1[v+1]):
                    w = indices1[j]
                    if SS[w] == VOID:
                        SS[w] = SS_head
                        SS_head = w
                # 处理 indices2 中 v 节点的邻接节点
                for j in range(indptr2[v], indptr2[v+1]):
                    w = indices2[j]
                    if SS[w] == VOID:
                        SS[w] = SS_head
                        SS_head = w

            # 连通分量标签加一
            label += 1

    # 返回连通分量的数量
    return label
```