# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_min_spanning_tree.pyx`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库
cimport numpy as np  # 在 Cython 中导入 NumPy 库
cimport cython  # 导入 Cython 库

from scipy.sparse import csr_matrix  # 从 SciPy 稀疏矩阵模块中导入 csr_matrix 类
from scipy.sparse.csgraph._validation import validate_graph  # 从 SciPy 稀疏矩阵图验证模块中导入 validate_graph 函数
from scipy.sparse._sputils import is_pydata_spmatrix  # 从 SciPy 稀疏矩阵实用工具模块中导入 is_pydata_spmatrix 函数

np.import_array()  # 调用 NumPy 的 import_array 函数

include 'parameters.pxi'  # 包含 parameters.pxi 文件中的定义

# 定义最小生成树函数
def minimum_spanning_tree(csgraph, overwrite=False):
    r"""
    minimum_spanning_tree(csgraph, overwrite=False)

    Return a minimum spanning tree of an undirected graph

    A minimum spanning tree is a graph consisting of the subset of edges
    which together connect all connected nodes, while minimizing the total
    sum of weights on the edges.  This is computed using the Kruskal algorithm.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        The N x N matrix representing an undirected graph over N nodes
        (see notes below).
    overwrite : bool, optional
        If true, then parts of the input graph will be overwritten for
        efficiency. Default is False.

    Returns
    -------
    span_tree : csr matrix
        The N x N compressed-sparse representation of the undirected minimum
        spanning tree over the input (see notes below).

    Notes
    -----
    This routine uses undirected graphs as input and output.  That is, if
    graph[i, j] and graph[j, i] are both zero, then nodes i and j do not
    have an edge connecting them.  If either is nonzero, then the two are
    connected by the minimum nonzero value of the two.

    This routine loses precision when users input a dense matrix.
    Small elements < 1E-8 of the dense matrix are rounded to zero.
    All users should input sparse matrices if possible to avoid it.

    If the graph is not connected, this routine returns the minimum spanning
    forest, i.e. the union of the minimum spanning trees on each connected
    component.

    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    The following example shows the computation of a minimum spanning tree
    over a simple four-component graph::

         input graph             minimum spanning tree

             (0)                         (0)
            /   \                       /
           3     8                     3
          /       \                   /
        (3)---5---(1)               (3)---5---(1)
          \       /                           /
           6     2                           2
            \   /                           /
             (2)                         (2)

    It is easy to see from inspection that the minimum spanning tree involves
    removing the edges with weights 8 and 6.  In compressed sparse
    representation, the solution looks like this:

    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import minimum_spanning_tree
    >>> X = csr_matrix([[0, 8, 0, 3],
    ...                 [0, 0, 2, 5],
    ...                 [0, 0, 0, 6],
    ...                 [0, 0, 0, 0]])
    >>> Tcsr = minimum_spanning_tree(X)
    >>> Tcsr.toarray().astype(int)
    array([[0, 0, 0, 3],
           [0, 0, 2, 5],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    global NULL_IDX  # 声明全局变量 NULL_IDX

    is_pydata_sparse = is_pydata_spmatrix(csgraph)  # 检查 csgraph 是否为 PyData Sparse 矩阵
    if is_pydata_sparse:
        pydata_sparse_cls = csgraph.__class__  # 获取 csgraph 的类
        pydata_sparse_fill_value = csgraph.fill_value  # 获取 csgraph 的填充值
    csgraph = validate_graph(csgraph, True, DTYPE, dense_output=False,
                             copy_if_sparse=not overwrite)  # 验证图结构，并根据需要复制

    cdef int N = csgraph.shape[0]  # 获取 csgraph 的顶点数

    data = csgraph.data  # 获取 csgraph 的数据数组
    indices = csgraph.indices  # 获取 csgraph 的列索引数组
    indptr = csgraph.indptr  # 获取 csgraph 的行指针数组

    rank = np.zeros(N, dtype=ITYPE)  # 创建长度为 N 的零数组，用于记录节点的秩
    predecessors = np.arange(N, dtype=ITYPE)  # 创建长度为 N 的数组，记录每个节点的前驱节点

    # Stable sort is a necessary but not sufficient operation
    # to get to a canonical representation of solutions.
    # 对数据进行稳定排序，以便得到规范的解表示
    i_sort = np.argsort(data, kind='stable').astype(ITYPE)
    row_indices = np.zeros(len(data), dtype=ITYPE)  # 创建与数据长度相同的零数组，用于行索引

    _min_spanning_tree(data, indices, indptr, i_sort,
                       row_indices, predecessors, rank)  # 执行最小生成树算法

    sp_tree = csr_matrix((data, indices, indptr), (N, N))  # 根据数据、列索引和行指针创建稀疏矩阵
    sp_tree.eliminate_zeros()  # 去除稀疏矩阵中的零元素

    if is_pydata_sparse:
        # 如果是 PyData Sparse 格式，尝试使用新版本支持的 fill_value 关键字进行转换
        try:
            sp_tree = pydata_sparse_cls.from_scipy_sparse(
                sp_tree, fill_value=pydata_sparse_fill_value
            )
        except TypeError:
            # 如果版本不支持 fill_value 关键字，则直接转换为 PyData Sparse 格式
            sp_tree = pydata_sparse_cls.from_scipy_sparse(sp_tree)
    return sp_tree  # 返回最小生成树的稀疏表示
# 使用 Cython 的装饰器禁用边界检查和包装，以提高性能
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 C 函数 _min_spanning_tree，计算最小生成树的核心算法
cdef void _min_spanning_tree(DTYPE_t[::1] data,
                             const ITYPE_t[::1] col_indices,
                             const ITYPE_t[::1] indptr,
                             const ITYPE_t[::1] i_sort,
                             ITYPE_t[::1] row_indices,
                             ITYPE_t[::1] predecessors,
                             ITYPE_t[::1] rank) noexcept nogil:
    # 使用 Kruskal 算法计算最小生成树的工作函数。通过将此部分代码分离，可以实现更高效的索引操作。

    # 定义变量
    cdef unsigned int i, j, V1, V2, R1, R2, n_edges_in_mst, n_verts, n_data
    n_verts = predecessors.shape[0]  # 获取顶点数
    n_data = i_sort.shape[0]          # 获取排序后数据的数量

    # 将 `row_indices` 数组设置为每个数据在 `data` 中对应的行索引
    for i in range(n_verts):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i

    # 从最小到最大步进遍历边
    n_edges_in_mst = 0
    i = 0
    while i < n_data and n_edges_in_mst < n_verts - 1:
        j = i_sort[i]
        V1 = row_indices[j]         # 边的一个顶点
        V2 = col_indices[j]         # 边的另一个顶点

        # 向上找到每个子树的头结点
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # 压缩路径
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2

        # 如果两棵子树不同，则连接它们并保留边；否则，移除边以避免生成树中的重复边
        if R1 != R2:
            n_edges_in_mst += 1

            # 使用近似的秩（由于路径压缩）来尽量保持平衡树
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
            elif rank[R1] < rank[R2]:
                predecessors[R1] = R2
            else:
                predecessors[R2] = R1
                rank[R1] += 1
        else:
            data[j] = 0  # 移除重复边

        i += 1

    # 如果找到完整的最小生成树，则可能会提前停止，这时需要将剩余的边设置为零
    while i < n_data:
        j = i_sort[i]
        data[j] = 0
        i += 1
```