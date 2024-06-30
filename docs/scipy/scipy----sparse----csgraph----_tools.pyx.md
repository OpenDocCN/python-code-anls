# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_tools.pyx`

```
"""
Tools and utilities for working with compressed sparse graphs
"""

# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD, (C) 2012

import numpy as np
cimport numpy as np  # Importing C-level numpy functionality

from scipy.sparse import csr_matrix, issparse
from scipy.sparse._sputils import is_pydata_spmatrix

np.import_array()  # Importing numpy array interface

include 'parameters.pxi'  # Including Cython header file 'parameters.pxi'

def csgraph_from_masked(graph):
    """
    csgraph_from_masked(graph)

    Construct a CSR-format graph from a masked array.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    graph : MaskedArray
        Input graph.  Shape should be (n_nodes, n_nodes).

    Returns
    -------
    csgraph : csr_matrix
        Compressed sparse representation of graph,

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.csgraph import csgraph_from_masked

    >>> graph_masked = np.ma.masked_array(data =[
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [0, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ],
    ... mask=[[ True, False, False,  True],
    ...       [ True,  True,  True, False],
    ...       [ True,  True,  True, False],
    ...       [ True,  True,  True,  True]],
    ... fill_value = 0)

    >>> csgraph_from_masked(graph_masked)
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 4 stored elements and shape (4, 4)>

    """
    # check that graph is a square matrix
    graph = np.ma.asarray(graph)  # Convert graph to a masked array

    if graph.ndim != 2:
        raise ValueError("graph should have two dimensions")
    N = graph.shape[0]
    if graph.shape[1] != N:
        raise ValueError("graph should be a square array")

    # construct the csr matrix using graph and mask
    data = graph.compressed()  # Retrieve non-masked data from the graph
    mask = ~graph.mask  # Invert mask to get valid data points

    data = np.asarray(data, dtype=DTYPE, order='c')  # Convert data to specified dtype

    idx_grid = np.empty((N, N), dtype=ITYPE)  # Create an empty array for indices
    idx_grid[:] = np.arange(N, dtype=ITYPE)  # Fill indices with row indices
    indices = np.asarray(idx_grid[mask], dtype=ITYPE, order='c')  # Retrieve valid indices

    indptr = np.zeros(N + 1, dtype=ITYPE)  # Initialize indptr array
    indptr[1:] = mask.sum(1).cumsum()  # Compute cumulative sum of valid entries per row

    return csr_matrix((data, indices, indptr), (N, N))  # Return csr_matrix constructed from data, indices, and indptr


def csgraph_masked_from_dense(graph,
                              null_value=0,
                              nan_null=True,
                              infinity_null=True,
                              copy=True):
    """
    csgraph_masked_from_dense(graph, null_value=0, nan_null=True,
                              infinity_null=True, copy=True)

    Construct a masked array graph representation from a dense matrix.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    graph : array_like
        Input graph.  Shape should be (n_nodes, n_nodes).
    null_value : float or None (optional)
        Value that denotes non-edges in the graph.  Default is zero.
    infinity_null : bool
        If True (default), then infinite entries (both positive and negative)
        are treated as null edges.
    nan_null : bool
        If True (default), then NaN entries are treated as non-edges

    Returns
    -------
    """
    
# 从稠密矩阵构建 CSR 格式的稀疏图
def csgraph_from_dense(graph,
                       null_value=0,
                       nan_null=True,
                       infinity_null=True):
    """
    csgraph_from_dense(graph, null_value=0, nan_null=True, infinity_null=True)

    Construct a CSR-format sparse graph from a dense matrix.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    graph : array_like
        Input graph.  Shape should be (n_nodes, n_nodes).
    null_value : float or None (optional)
        Value that denotes non-edges in the graph.  Default is zero.
    infinity_null : bool
        If True (default), then infinite entries (both positive and negative)
        are treated as null edges.
    nan_null : bool
        If True (default), then NaN entries are treated as non-edges

    Returns
    -------
    csgraph : csr_matrix
        Compressed sparse representation of graph,

    Examples
    --------
    >>> from scipy.sparse.csgraph import csgraph_from_dense

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [0, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]

    >>> csgraph_from_dense(graph)
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 4 stored elements and shape (4, 4)>

    """
    # 调用 csgraph_masked_from_dense 函数，将稠密图转换为 CSR 格式的稀疏图
    return csgraph_from_masked(csgraph_masked_from_dense(graph,
                                                         null_value,
                                                         nan_null,
                                                         infinity_null))


def csgraph_to_dense(csgraph, null_value=0):
    """
    csgraph_to_dense(csgraph, null_value=0)

    Convert a sparse graph representation to a dense representation

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : csr_matrix, csc_matrix, or lil_matrix
        Sparse representation of a graph.
    null_value : float, optional
        The value used to indicate null edges in the dense representation.
        Default is 0.

    Returns
    -------
    graph : ndarray
        The dense representation of the sparse graph.

    Notes
    -----
    For normal sparse graph representations, calling csgraph_to_dense with
    null_value=0 produces an equivalent result to using dense format
    conversions in the main sparse package.  When the sparse representations
    have repeated values, however, the results will differ.  The tools in
    scipy.sparse will add repeating values to obtain a final value.  This
    function will select the minimum among repeating values to obtain a
    final value.  For example, here we'll create a two-node directed sparse
    graph with multiple edges from node 0 to node 1, of weights 2 and 3.
    This illustrates the difference in behavior:

    >>> from scipy.sparse import csr_matrix, csgraph
    >>> import numpy as np
    >>> data = np.array([2, 3])
    >>> indices = np.array([1, 1])
    >>> indptr = np.array([0, 2, 2])
    >>> M = csr_matrix((data, indices, indptr), shape=(2, 2))
    >>> M.toarray()

    """
    # 将稀疏图的 CSR 格式转换为稠密矩阵表示
    graph = csgraph.toarray()
    return graph
    # Allow only csr, lil and csc matrices: other formats when converted to csr
    # combine duplicated edges: we don't want this to happen in the background.
    if not issparse(csgraph):
        raise ValueError("csgraph must be sparse")
    if csgraph.format not in ("lil", "csc", "csr"):
        raise ValueError("csgraph must be lil, csr, or csc format")
    csgraph = csgraph.tocsr()
    
    N = csgraph.shape[0]
    if csgraph.shape[1] != N:
        raise ValueError('csgraph should be a square matrix')
    
    # get attribute arrays
    data = np.asarray(csgraph.data, dtype=DTYPE, order='C')
    indices = np.asarray(csgraph.indices, dtype=ITYPE, order='C')
    indptr = np.asarray(csgraph.indptr, dtype=ITYPE, order='C')
    
    # create the output array
    graph = np.empty(csgraph.shape, dtype=DTYPE)
    graph.fill(np.inf)
    _populate_graph(data, indices, indptr, graph, null_value)
    return graph
# 将稀疏图表示转换为掩码数组表示
def csgraph_to_masked(csgraph):
    # 返回将稀疏图 csgraph 转换为稠密掩码数组表示的结果
    return np.ma.masked_invalid(csgraph_to_dense(csgraph, np.nan))


# 使用 Cython 声明定义的函数，用于填充图的稀疏表示
cdef void _populate_graph(np.ndarray[DTYPE_t, ndim=1, mode='c'] data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr,
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] graph,
                          DTYPE_t null_value) noexcept:
    # data, indices, indptr 是稀疏输入的 csr 属性
    # 在输入时，graph 应该被填充为无穷大，大小为 [N, N]，N 是稀疏矩阵的大小
    cdef unsigned int N = graph.shape[0]
    # 创建一个布尔类型的 N x N 矩阵，用于标记空值
    cdef np.ndarray null_flag = np.ones((N, N), dtype=bool, order='C')
    cdef np.npy_bool* null_ptr = <np.npy_bool*> null_flag.data
    cdef unsigned int row, col, i

    for row in range(N):
        for i in range(indptr[row], indptr[row + 1]):
            col = indices[i]
            null_ptr[col] = 0
            # 对于多条边的情况，选择最小的值
            if data[i] < graph[row, col]:
                graph[row, col] = data[i]
        null_ptr += N

    # 将标记为 True 的位置（空值）设为 null_value
    graph[null_flag] = null_value


# 从图和前导节点列表构造树
def reconstruct_path(csgraph, predecessors, directed=True):
    # 返回从图 csgraph 和前导节点 predecessors 构造的树的结果
    pass  # 这里的函数体还未实现，暂时使用 pass 占位符
    # 从 `_validation` 模块导入 `validate_graph` 函数
    from ._validation import validate_graph

    # 检查 `csgraph` 是否为 PyData 稀疏矩阵类型
    is_pydata_sparse = is_pydata_spmatrix(csgraph)

    # 如果 `csgraph` 是 PyData 稀疏矩阵类型，保存其类和填充值
    if is_pydata_sparse:
        pydata_sparse_cls = csgraph.__class__
        pydata_sparse_fill_value = csgraph.fill_value

    # 调用 `validate_graph` 函数，验证 `csgraph` 的合法性，并根据参数 `directed` 进行处理
    csgraph = validate_graph(csgraph, directed, dense_output=False)

    # 获取矩阵的大小 N
    N = csgraph.shape[0]

    # 计算 `predecessors` 中小于 0 的元素个数
    nnull = (predecessors < 0).sum()

    # 对 `predecessors` 进行排序，并去掉小于 0 的部分，转换为 `ITYPE` 类型
    indices = np.argsort(predecessors)[nnull:].astype(ITYPE)

    # 根据 `indices` 生成 `indptr` 数组，以便构建稀疏矩阵
    pind = predecessors[indices]
    indptr = pind.searchsorted(np.arange(N + 1)).astype(ITYPE)

    # 从 `csgraph` 中提取与 `indices` 和 `pind` 相对应的数据，用于构建稀疏矩阵
    data = csgraph[pind, indices]

    # 解决问题 #4018：
    # 如果 `data` 是稀疏矩阵，则将其转换为密集矩阵
    if issparse(data):
        data = data.todense()
    data = data.getA1()

    # 如果图是无向图，需要处理对称性
    if not directed:
        # 从 `csgraph` 中提取 `indices` 和 `pind` 对应的另一部分数据
        data2 = csgraph[indices, pind]

        # 如果 `data2` 是稀疏矩阵，则转换为密集矩阵
        if issparse(data2):
            data2 = data2.todense()
        data2 = data2.getA1()

        # 将 `data` 和 `data2` 中的零元素替换为无穷大
        data[data == 0] = np.inf
        data2[data2 == 0] = np.inf

        # 取 `data` 和 `data2` 中的元素的最小值
        data = np.minimum(data, data2)

    # 根据 `data`, `indices` 和 `indptr` 构建稀疏矩阵 `sctree`
    sctree = csr_matrix((data, indices, indptr), shape=(N, N))

    # 如果输入的 `csgraph` 是 PyData 稀疏矩阵类型，将 `sctree` 转换为相应类型
    if is_pydata_sparse:
        try:
            # 使用 PyData Sparse 0.15.4 新增的 `fill_value` 关键字构建稀疏矩阵
            sctree = pydata_sparse_cls.from_scipy_sparse(
                sctree, fill_value=pydata_sparse_fill_value
            )
        except TypeError:
            # 兼容处理，对于不支持 `fill_value` 的版本，直接构建稀疏矩阵
            sctree = pydata_sparse_cls.from_scipy_sparse(sctree)

    # 返回构建好的稀疏矩阵 `sctree`
    return sctree
# 从前趋矩阵构建距离矩阵
def construct_dist_matrix(graph,
                          predecessors,
                          directed=True,
                          null_value=np.inf):
    """
    construct_dist_matrix(graph, predecessors, directed=True, null_value=np.inf)

    Construct distance matrix from a predecessor matrix

    .. versionadded:: 0.11.0

    Parameters
    ----------
    graph : array_like or sparse
        The N x N matrix representation of a directed or undirected graph.
        If dense, then non-edges are indicated by zeros or infinities.
    predecessors : array_like
        The N x N matrix of predecessors of each node (see Notes below).
    directed : bool, optional
        If True (default), then operate on a directed graph: only move from
        point i to point j along paths csgraph[i, j].
        If False, then operate on an undirected graph: the algorithm can
        progress from point i to j along csgraph[i, j] or csgraph[j, i].
    null_value : bool, optional
        value to use for distances between unconnected nodes.  Default is
        np.inf

    Returns
    -------
    dist_matrix : ndarray
        The N x N matrix of distances between nodes along the path specified
        by the predecessor matrix.  If no path exists, the distance is zero.

    Notes
    -----
    The predecessor matrix is of the form returned by
    `shortest_path`.  Row i of the predecessor matrix contains
    information on the shortest paths from point i: each entry
    predecessors[i, j] gives the index of the previous node in the path from
    point i to point j.  If no path exists between point i and j, then
    predecessors[i, j] = -9999

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import construct_dist_matrix

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [0, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
       (np.int32(0), np.int32(1))    1
       (np.int32(0), np.int32(2))    2
       (np.int32(1), np.int32(3))    1
       (np.int32(2), np.int32(3))    3

    >>> pred = np.array([[-9999, 0, 0, 2],
    ...                  [1, -9999, 0, 1],
    ...                  [2, 0, -9999, 2],
    ...                  [1, 3, 3, -9999]], dtype=np.int32)

    >>> construct_dist_matrix(graph=graph, predecessors=pred, directed=False)
    array([[0., 1., 2., 5.],
           [1., 0., 3., 1.],
           [2., 3., 0., 3.],
           [2., 1., 3., 0.]])

    """
    from ._validation import validate_graph
    # 验证并处理图形表示，确保适当的数据类型和格式
    graph = validate_graph(graph, directed, dtype=DTYPE,
                           csr_output=False,
                           copy_if_dense=not directed)
    predecessors = np.asarray(predecessors)

    if predecessors.shape != graph.shape:
        # 如果前趋矩阵形状与图形矩阵形状不匹配，则引发值错误
        raise ValueError("graph and predecessors must have the same shape")

    # 初始化距离矩阵，形状与输入图形矩阵相同，数据类型为 DTYPE
    dist_matrix = np.zeros(graph.shape, dtype=DTYPE)
    # 使用给定的图(graph)、前驱(predecessors)、距离矩阵(dist_matrix)、是否有向(directed)和空值(null_value)来构建距离矩阵
    _construct_dist_matrix(graph, predecessors, dist_matrix,
                           directed, null_value)
    
    # 返回构建好的距离矩阵作为结果
    return dist_matrix
    # 执行构造距离矩阵的函数，使用传入的图形矩阵、前驱矩阵和距离矩阵
    # 如果 directed == False，则可能修改图形矩阵以对称化
    # 距离矩阵在进入函数时应全部为零
    cdef void _construct_dist_matrix(np.ndarray[DTYPE_t, ndim=2] graph,
                                     np.ndarray[ITYPE_t, ndim=2] pred,
                                     np.ndarray[DTYPE_t, ndim=2] dist,
                                     int directed,
                                     DTYPE_t null_value) noexcept:
        # 定义全局变量 NULL_IDX
        global NULL_IDX

        cdef int i, j, k1, k2, N, null_path
        # 获取图形矩阵的大小 N x N
        N = graph.shape[0]

        #------------------------------------------
        # 如果不是有向图，则对图形矩阵进行对称化处理
        if not directed:
            # 将图形矩阵中的零元素替换为正无穷
            graph[graph == 0] = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    # 根据对称性，选择较小的值进行赋值，确保对称性
                    if graph[j, i] <= graph[i, j]:
                        graph[i, j] = graph[j, i]
                    else:
                        graph[j, i] = graph[i, j]
        #------------------------------------------

        # 遍历所有节点对 i, j
        for i in range(N):
            for j in range(N):
                # 初始化 null_path 为 True，表示尚未找到路径
                null_path = True
                k2 = j
                # 从节点 j 开始向前追溯路径，直到回到节点 i
                while k2 != i:
                    k1 = pred[i, k2]
                    # 如果 k1 是 NULL_IDX，则表示路径中断
                    if k1 == NULL_IDX:
                        break
                    # 将路径上的权重累加到距离矩阵中
                    dist[i, j] += graph[k1, k2]
                    # 将 null_path 设为 False，表示找到了有效路径
                    null_path = False
                    k2 = k1
                # 如果 null_path 仍为 True 且 i 不等于 j，则将距离矩阵中的值设置为 null_value
                if null_path and i != j:
                    dist[i, j] = null_value
```