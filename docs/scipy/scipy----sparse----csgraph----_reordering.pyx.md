# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_reordering.pyx`

```
# Author: Paul Nation  -- <nonhermitian@gmail.com>
# Original Source: QuTiP: Quantum Toolbox in Python (qutip.org)
# License: New BSD, (C) 2014

# 导入必要的库和模块
import numpy as np
# 导入Cython生成的NumPy模块
cimport numpy as np
# 导入警告模块中的warn函数
from warnings import warn
# 导入稀疏矩阵相关的类和函数
from scipy.sparse import csr_matrix, issparse, SparseEfficiencyWarning
# 导入将Python数据转换为SciPy稀疏矩阵的函数
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy
# 导入最大二分匹配模块
from . import maximum_bipartite_matching

# 调用NumPy的C API
np.import_array()

# 包含外部参数文件parameters.pxi中的定义

# 定义函数reverse_cuthill_mckee，用于计算稀疏CSR或CSC矩阵的Reverse-Cuthill McKee排序的排列数组
def reverse_cuthill_mckee(graph, symmetric_mode=False):
    """
    reverse_cuthill_mckee(graph, symmetric_mode=False)
    
    Returns the permutation array that orders a sparse CSR or CSC matrix
    in Reverse-Cuthill McKee ordering.  
    
    It is assumed by default, ``symmetric_mode=False``, that the input matrix 
    is not symmetric and works on the matrix ``A+A.T``. If you are 
    guaranteed that the matrix is symmetric in structure (values of matrix 
    elements do not matter) then set ``symmetric_mode=True``.
    
    Parameters
    ----------
    graph : sparse matrix
        Input sparse in CSC or CSR sparse matrix format.
    symmetric_mode : bool, optional
        Is input matrix guaranteed to be symmetric.

    Returns
    -------
    perm : ndarray
        Array of permuted row and column indices.
 
    Notes
    -----
    .. versionadded:: 0.15.0

    References
    ----------
    E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric Matrices",
    ACM '69 Proceedings of the 1969 24th national conference, (1969).

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import reverse_cuthill_mckee

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

    >>> reverse_cuthill_mckee(graph)
    array([3, 2, 1, 0], dtype=int32)
    
    """
    # 将Python数据转换为SciPy稀疏矩阵
    graph = convert_pydata_sparse_to_scipy(graph)
    # 检查输入矩阵是否为稀疏矩阵
    if not issparse(graph):
        raise TypeError("Input graph must be sparse")
    # 检查输入矩阵是否为CSC或CSR格式
    if graph.format not in ("csc", "csr"):
        raise TypeError('Input must be in CSC or CSR sparse matrix format.')
    # 获取矩阵的行数
    nrows = graph.shape[0]
    # 如果不是对称模式，则处理输入矩阵为A+A.T
    if not symmetric_mode:
        graph = graph + graph.transpose()
    # 调用底层Cython函数计算Reverse-Cuthill McKee排序的排列数组
    return _reverse_cuthill_mckee(graph.indices, graph.indptr, nrows)


# 定义Cython函数_node_degrees，用于计算稀疏CSR或CSC矩阵中每个节点（行）的度数
cdef _node_degrees(
        np.ndarray[int32_or_int64, ndim=1, mode="c"] ind,
        np.ndarray[int32_or_int64, ndim=1, mode="c"] ptr,
        np.npy_intp num_rows):
    """
    Find the degree of each node (matrix row) in a graph represented
    by a sparse CSR or CSC matrix.
    """
    cdef np.npy_intp ii, jj
    # 创建一个与节点数相同的零数组，用于存储每个节点的度数
    cdef np.ndarray[int32_or_int64] degree = np.zeros(num_rows, dtype=ind.dtype)
    # 对于每一行 ii，计算其对应的度数
    for ii in range(num_rows):
        # 计算当前行 ii 的度数，即指针 ptr[ii+1] 减去 ptr[ii] 的差值
        degree[ii] = ptr[ii + 1] - ptr[ii]
        # 遍历从 ptr[ii] 到 ptr[ii+1]-1 的列索引 jj
        for jj in range(ptr[ii], ptr[ii + 1]):
            # 如果列索引 jj 对应的行索引 ind[jj] 等于 ii
            if ind[jj] == ii:
                # 如果找到对角线元素在行 ii 上，将度数 degree[ii] 加一
                degree[ii] += 1
                # 因为找到了对角线元素，所以可以终止内部循环
                break
    # 返回每一行的度数组成的列表 degree
    return degree
# 计算稀疏对称 CSR 或 CSC 矩阵的反向 Cuthill-McKee 排序
# 我们遵循原始的 Cuthill-McKee 论文，始终从每个连通分量的度最低的节点开始执行此算法

def _reverse_cuthill_mckee(np.ndarray[int32_or_int64, ndim=1, mode="c"] ind,
        np.ndarray[int32_or_int64, ndim=1, mode="c"] ptr,
        np.npy_intp num_rows):
    """
    Reverse Cuthill-McKee ordering of a sparse symmetric CSR or CSC matrix.  
    We follow the original Cuthill-McKee paper and always start the routine
    at a node of lowest degree for each connected component.
    """
    # 初始化变量
    cdef np.npy_intp N = 0, N_old, level_start, level_end, temp
    cdef np.npy_intp zz, ii, jj, kk, ll, level_len
    # 创建用于存储节点顺序的数组
    cdef np.ndarray[int32_or_int64] order = np.zeros(num_rows, dtype=ind.dtype)
    # 计算每个节点的度数
    cdef np.ndarray[int32_or_int64] degree = _node_degrees(ind, ptr, num_rows)
    # 按度数对节点进行排序，并得到其索引
    cdef np.ndarray[np.npy_intp] inds = np.argsort(degree)
    # 根据排序索引再次排序以获得逆序索引
    cdef np.ndarray[np.npy_intp] rev_inds = np.argsort(inds)
    # 用于临时存储节点度数的数组，大小为最大度数
    cdef np.ndarray[ITYPE_t] temp_degrees = np.zeros(np.max(degree), dtype=ITYPE)
    cdef int32_or_int64 i, j, seed, temp2
    
    # 循环遍历可能的不连通图
    for zz in range(num_rows):
        if inds[zz] != -1:   # 以 inds[zz] 为种子进行 BFS
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = N - 1
            level_end = N

            while level_start < level_end:
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N

                    # 添加未访问的邻居节点
                    for jj in range(ptr[i], ptr[i + 1]):
                        # j 是与 i 连接的节点编号
                        j = ind[jj]
                        if inds[rev_inds[j]] != -1:
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            N += 1

                    # 将节点插入排序的临时度数数组中
                    level_len = 0
                    for kk in range(N_old, N):
                        temp_degrees[level_len] = degree[order[kk]]
                        level_len += 1
                
                    # 对节点按照度数从低到高进行插入排序
                    for kk in range(1, level_len):
                        temp = temp_degrees[kk]
                        temp2 = order[N_old+kk]
                        ll = kk
                        while (ll > 0) and (temp < temp_degrees[ll-1]):
                            temp_degrees[ll] = temp_degrees[ll-1]
                            order[N_old+ll] = order[N_old+ll-1]
                            ll -= 1
                        temp_degrees[ll] = temp
                        order[N_old+ll] = temp2
                
                # 设置下一级别的起始和结束范围
                level_start = level_end
                level_end = N

        if N == num_rows:
            break

    # 返回反向排序后的顺序，即 RCM 排序
    return order[::-1]


def structural_rank(graph):
    """
    structural_rank(graph)
    """
    Compute the structural rank of a graph (matrix) with a given 
    sparsity pattern.

    The structural rank of a matrix is the number of entries in the maximum 
    transversal of the corresponding bipartite graph, and is an upper bound 
    on the numerical rank of the matrix. A graph has full structural rank 
    if it is possible to permute the elements to make the diagonal zero-free.

    .. versionadded:: 0.19.0

    Parameters
    ----------
    graph : sparse matrix
        Input sparse matrix.

    Returns
    -------
    rank : int
        The structural rank of the sparse graph.
    
    References
    ----------
    .. [1] I. S. Duff, "Computing the Structural Index", SIAM J. Alg. Disc. 
            Meth., Vol. 7, 594 (1986).
    
    .. [2] http://www.cise.ufl.edu/research/sparse/matrices/legend.html

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import structural_rank

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [1, 0, 0, 1],
    ... [2, 0, 0, 3],
    ... [0, 1, 3, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
      (np.int32(0), np.int32(1))    1
      (np.int32(0), np.int32(2))    2
      (np.int32(1), np.int32(0))    1
      (np.int32(1), np.int32(3))    1
      (np.int32(2), np.int32(0))    2
      (np.int32(2), np.int32(3))    3
      (np.int32(3), np.int32(1))    1
      (np.int32(3), np.int32(2))    3

    >>> structural_rank(graph)
    4
    """

# 将 Python 的稀疏数据转换为 scipy 稀疏矩阵
graph = convert_pydata_sparse_to_scipy(graph)

# 如果输入不是稀疏矩阵，则抛出类型错误
if not issparse(graph):
    raise TypeError('Input must be a sparse matrix')

# 如果稀疏矩阵的格式不是 CSR，则将其转换为 CSR 格式
if graph.format != "csr":
    if graph.format not in ("csc", "coo"):
        # 如果不是 CSR、CSC 或 COO 格式，则发出警告
        warn('Input matrix should be in CSC, CSR, or COO matrix format',
                SparseEfficiencyWarning)
    graph = csr_matrix(graph)

# 如果矩阵的行数大于列数，则将其转置为 CSR 格式
if graph.shape[0] > graph.shape[1]:
    graph = graph.T.tocsr()

# 计算稀疏矩阵的结构秩，即最大二分图匹配的非负数个数
rank = np.sum(maximum_bipartite_matching(graph) >= 0)
return rank
```