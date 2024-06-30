# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_matching.pyx`

```
# 引入警告模块，用于在程序执行过程中输出警告信息
import warnings

# 使用Cython的cimport语句，导入Cython模块
cimport cython
# 导入NumPy库，并将其命名为np，用于处理数组和数值计算
import numpy as np
# 使用Cython的cimport语句，导入NumPy库的C语言API接口
cimport numpy as np
# 从C标准库libc.math中导入INFINITY常量，表示正无穷大
from libc.math cimport INFINITY

# 导入SciPy稀疏矩阵处理模块中的issparse函数，用于判断一个对象是否为稀疏矩阵
from scipy.sparse import issparse
# 导入SciPy稀疏矩阵处理模块中的convert_pydata_sparse_to_scipy函数，用于将Python数据结构转换为SciPy稀疏矩阵
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy

# 调用NumPy的import_array函数，用于导入NumPy库的C语言API接口
np.import_array()

# 包含一个名为"parameters.pxi"的Cython预处理文件，用于导入其中定义的内容

# 定义一个名为maximum_bipartite_matching的函数，用于求解二分图的最大匹配问题
def maximum_bipartite_matching(graph, perm_type='row'):
    r"""
    maximum_bipartite_matching(graph, perm_type='row')

    Returns a matching of a bipartite graph whose cardinality is as least that
    of any given matching of the graph.

    Parameters
    ----------
    graph : sparse matrix
        Input sparse in CSR format whose rows represent one partition of the
        graph and whose columns represent the other partition. An edge between
        two vertices is indicated by the corresponding entry in the matrix
        existing in its sparse representation.
    perm_type : str, {'row', 'column'}
        Which partition to return the matching in terms of: If ``'row'``, the
        function produces an array whose length is the number of columns in the
        input, and whose :math:`j`'th element is the row matched to the
        :math:`j`'th column. Conversely, if ``perm_type`` is ``'column'``, this
        returns the columns matched to each row.

    Returns
    -------
    perm : ndarray
        A matching of the vertices in one of the two partitions. Unmatched
        vertices are represented by a ``-1`` in the result.

    Notes
    -----
    This function implements the Hopcroft--Karp algorithm [1]_. Its time
    complexity is :math:`O(\lvert E \rvert \sqrt{\lvert V \rvert})`, and its
    space complexity is linear in the number of rows. In practice, this
    asymmetry between rows and columns means that it can be more efficient to
    transpose the input if it contains more columns than rows.

    By Konig's theorem, the cardinality of the matching is also the number of
    vertices appearing in a minimum vertex cover of the graph.

    Note that if the sparse representation contains explicit zeros, these are
    still counted as edges.

    The implementation was changed in SciPy 1.4.0 to allow matching of general
    bipartite graphs, where previous versions would assume that a perfect
    matching existed. As such, code written against 1.4.0 will not necessarily
    work on older versions.

    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    References
    ----------
    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for
           Maximum Matchings in Bipartite Graphs" In: SIAM Journal of Computing
           2.4 (1973), pp. 225--231. :doi:`10.1137/0202019`

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import maximum_bipartite_matching

    As a simple example, consider a bipartite graph in which the partitions
    contain 2 and 3 elements respectively. Suppose that one partition contains
    ...
    """
    # 函数体中的具体实现内容在这里省略，涉及Hopcroft-Karp算法的二分图最大匹配问题
    # 将 Python 数据结构的稀疏矩阵转换为 scipy 的 CSR 格式
    graph = convert_pydata_sparse_to_scipy(graph)

    # 检查 graph 是否为稀疏矩阵，如果不是则抛出类型错误异常
    if not issparse(graph):
        raise TypeError("graph must be sparse")

    # 检查 graph 的格式是否为 CSR、CSC 或 COO，如果不是则抛出类型错误异常
    if graph.format not in ("csr", "csc", "coo"):
        raise TypeError("graph must be in CSC, CSR, or COO format.")

    # 将 graph 转换为 CSR 格式
    graph = graph.tocsr()

    # 获取 CSR 矩阵的形状
    i, j = graph.shape

    # 调用 _hopcroft_karp 函数执行 Hopcroft-Karp 最大二分匹配算法，获取匹配结果 x, y
    x, y = _hopcroft_karp(graph.indices, graph.indptr, i, j)

    # 根据 perm_type 参数选择返回结果 x 或 y，并转换为 NumPy 数组返回
    return np.asarray(x if perm_type == 'column' else y)
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 Cython 函数 _hopcroft_karp，用于执行 Hopcroft-Karp 最大匹配算法
cdef tuple _hopcroft_karp(const ITYPE_t[:] indices, const ITYPE_t[:] indptr,
                          const ITYPE_t i, const ITYPE_t j):
    # 定义变量 INF，表示整型的最大值
    cdef ITYPE_t INF = np.iinfo(ITYPE).max
    
    # x 数组将存储行与列的匹配，y 数组将存储列与行的匹配
    cdef ITYPE_t[:] x = np.empty(i, dtype=ITYPE)
    cdef ITYPE_t[:] y = np.empty(j, dtype=ITYPE)

    # BFS 步骤中，dist 数组将跟踪搜索的层级，仅在左侧分区中跟踪。数组长度比左侧分区的基数多一个，
    # 使用一个额外的顶点 i，语义上未匹配的列将与此顶点匹配。
    cdef ITYPE_t[:] dist = np.empty(i + 1, dtype=ITYPE)

    cdef ITYPE_t k, v, w, up, u, yu, u_old

    # 初始化未匹配的顶点，将 x 数组中所有值设为 -1
    for k in range(i):
        x[k] = -1

    # 将 y 数组中所有值设为 i，表示初始状态下所有列均未匹配
    for k in range(j):
        y[k] = i

    # 设置三个数据结构用于搜索：q 表示 BFS 步骤中的顶点队列，仅保留左侧分区的顶点及辅助顶点 i；
    # 使用两个指针 head 和 tail 表示队列的头尾。
    cdef ITYPE_t[:] q = np.empty(i + 1, dtype=ITYPE)
    cdef ITYPE_t head, tail

    # 使用栈进行深度优先搜索，同样仅保留左侧分区的顶点；
    # stack_head 表示栈顶的指针。
    cdef ITYPE_t[:] stack = np.empty(i + 1, dtype=ITYPE)
    cdef ITYPE_t stack_head

    # parents 数组用于在深度优先搜索过程中记录路径，简化找到增广路径后的匹配更新操作。
    cdef ITYPE_t[:] parents = np.empty(i, dtype=ITYPE)

    # 在算法的 BFS 部分，将所有未匹配的列的 y 值设置为 -1，表示它们未匹配到任何行。
    for k in range(j):
        if y[k] == i:
            y[k] = -1

    # 返回 x 和 y 数组作为结果
    return x, y
# 定义一个函数，用于计算二分图的最小权重全匹配
def min_weight_full_bipartite_matching(biadjacency_matrix, maximize=False):
    """
    min_weight_full_bipartite_matching(biadjacency_matrix, maximize=False)

    返回二分图的最小权重全匹配。

    .. versionadded:: 1.6.0

    Parameters
    ----------
    biadjacency_matrix : sparse matrix
        二分图的双向邻接矩阵：以CSR、CSC或COO格式表示的稀疏矩阵，其行代表图的一个分区，列代表另一个分区。
        矩阵中的条目表示两个顶点之间的边，其权重由条目的值给出。这与图的完整邻接矩阵不同，我们只需要定义二分图结构的子矩阵。

    maximize : bool (default: False)
        如果为真，则计算最大权重匹配。

    Returns
    -------
    row_ind, col_ind : array
        一个包含行索引和列索引的数组，给出最优匹配。匹配的总权重可以通过 ``graph[row_ind, col_ind].sum()`` 计算。
        行索引将会排序；在方阵的情况下，它们将等于 ``numpy.arange(graph.shape[0])``。

    Notes
    -----
    假设 :math:`G = ((U, V), E)` 是一个带有非零权重 :math:`w : E \to \mathbb{R} \setminus \{0\}` 的加权二分图。
    该函数返回一个匹配 :math:`M \subseteq E`，其基数为

    .. math::
       \lvert M \rvert = \min(\lvert U \rvert, \lvert V \rvert),

    它最小化匹配中包含的边的权重总和 :math:`\sum_{e \in M} w(e)`，如果不存在这样的匹配，则会引发错误。

    当 :math:`\lvert U \rvert = \lvert V \rvert` 时，通常称为完美匹配；在这里，我们允许 :math:`\lvert U \rvert` 和 :math:`\lvert V \rvert` 不同，
    我们遵循 Karp [1]_ 的术语，将匹配称为“全匹配”。

    该函数实现了LAPJVsp算法 [2]_，即“线性分配问题，Jonker--Volgenant，稀疏”。

    它解决的问题等同于矩形线性分配问题。[3]_ 因此，该函数可以用于解决与 :func:`scipy.optimize.linear_sum_assignment` 相同的问题。
    当输入稠密或特定类型的输入时（例如， :math:`(i, j)` 处的条目是欧几里得空间中两点之间的距离时），该函数可能表现更好。

    如果不存在全匹配，则该函数会引发 ``ValueError``。要确定图中最大匹配的大小，请参阅 :func:`maximum_bipartite_matching`。

    我们要求权重仅为非零，以避免在不同稀疏表示之间转换时出现显式零的处理问题。
    """
    Zero weights can be handled by adding a constant to all weights, so that
    the resulting matrix contains no zeros.

    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    References
    ----------
    .. [1] Richard Manning Karp:
       An algorithm to Solve the m x n Assignment Problem in Expected Time
       O(mn log n).
       Networks, 10(2):143-152, 1980.
    .. [2] Roy Jonker and Anton Volgenant:
       A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
       Assignment Problems.
       Computing 38:325-340, 1987.
    .. [3] https://en.wikipedia.org/wiki/Assignment_problem

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    Let us first consider an example in which all weights are equal:

    >>> biadjacency_matrix = csr_matrix([[1, 1, 1], [1, 0, 0], [0, 1, 0]])

    Here, all we get is a perfect matching of the graph:

    >>> print(min_weight_full_bipartite_matching(biadjacency_matrix)[1])
    [2 0 1]
    # 输出结果表示的是匹配的列索引，即第一、第二、第三行分别与第三、第一、第二列匹配。注意，输入矩阵中的 0 并不表示权重为 0 的边，而是未通过边连接的顶点对。

    Note also that in this case, the output matches the result of applying
    :func:`maximum_bipartite_matching`:

    >>> from scipy.sparse.csgraph import maximum_bipartite_matching
    >>> biadjacency = csr_matrix([[1, 1, 1], [1, 0, 0], [0, 1, 0]])
    >>> print(maximum_bipartite_matching(biadjacency, perm_type='column'))
    [2 0 1]
    # 对于多个可能的匹配方案，输出与 maximum_bipartite_matching 函数的结果匹配。

    When multiple edges are available, the ones with lowest weights are
    preferred:

    >>> biadjacency = csr_matrix([[3, 3, 6], [4, 3, 5], [10, 1, 8]])
    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)
    >>> print(col_ind)
    [0 2 1]
    # 输出的 col_ind 表示最小权重匹配的列索引。

    The total weight in this case is :math:`3 + 5 + 1 = 9`:

    >>> print(biadjacency[row_ind, col_ind].sum())
    9
    # 在这种情况下，总权重为 9。

    When the matrix is not square, i.e. when the two partitions have different
    cardinalities, the matching is as large as the smaller of the two
    partitions:

    >>> biadjacency = csr_matrix([[0, 1, 1], [0, 2, 3]])
    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)
    >>> print(row_ind, col_ind)
    [0 1] [2 1]
    >>> biadjacency = csr_matrix([[0, 1], [3, 1], [1, 4]])
    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)
    >>> print(row_ind, col_ind)
    [0 2] [1 0]
    # 当矩阵不是方阵时，即两个分区的基数不同时，匹配的大小等于两个分区中较小的一个。

    When one or both of the partitions are empty, the matching is empty as
    well:

    >>> biadjacency = csr_matrix((2, 0))
    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)
    >>> print(row_ind, col_ind)
    [] []
    # 当一个或两个分区为空时，匹配也为空。

    In general, we will always reach the same sum of weights as if we had used
    """
    Convert the input `biadjacency_matrix` to a scipy sparse matrix if it's not already in that format,
    ensuring it meets necessary criteria for the bipartite matching algorithm.

    Parameters:
    ----------
    biadjacency_matrix : scipy.sparse matrix or array-like
        The input graph in sparse format (CSR, CSC, or COO).
    
    Returns:
    -------
    biadjacency_matrix : scipy.sparse.csr_matrix
        The converted and validated biadjacency matrix in CSR format.

    Raises:
    ------
    TypeError:
        If `biadjacency_matrix` is not sparse or in an unsupported format.
    ValueError:
        If `biadjacency_matrix` contains non-numerical entries or unexpected types.

    Notes:
    ------
    - Converts all matrix entries to double precision.
    - If `maximize` flag is set, converts the matrix entries to their negatives.
    - Handles removal of explicit zero weights with a warning if present.
    - Ensures the matrix has more columns than rows, necessary for the matching algorithm.
    """
    biadjacency_matrix = convert_pydata_sparse_to_scipy(biadjacency_matrix)
    if not issparse(biadjacency_matrix):
        raise TypeError("graph must be sparse")
    if biadjacency_matrix.format not in ("csr", "csc", "coo"):
        raise TypeError("graph must be in CSC, CSR, or COO format.")

    if not (np.issubdtype(biadjacency_matrix.dtype, np.number) or
            biadjacency_matrix.dtype == np.dtype(np.bool_)):
        raise ValueError("expected a matrix containing numerical entries, " +
                         "got %s" % (biadjacency_matrix.dtype,))

    biadjacency_matrix = biadjacency_matrix.astype(np.double)

    if maximize:
        biadjacency_matrix = -biadjacency_matrix

    # Change all infinities to zeros, then remove those zeros, but warn the
    # user if any zeros were present in the first place.
    if not np.all(biadjacency_matrix.data):
        warnings.warn('explicit zero weights are removed before matching')

    biadjacency_matrix.data[np.isposinf(biadjacency_matrix.data)] = 0
    biadjacency_matrix.eliminate_zeros()

    i, j = biadjacency_matrix.shape

    a = np.arange(np.min(biadjacency_matrix.shape))

    # The algorithm expects more columns than rows in the graph, so
    # we use the transpose if that is not already the case. We also
    # ensure that we have a full matching. In principle, it should be
    # possible to avoid this check for a performance improvement, by
    # checking for infeasibility during the execution of _lapjvsp below
    # instead, but some cases are not yet handled there.
    # 如果 j 小于 i，则执行以下操作
    if j < i:
        # 转置二分邻接矩阵
        biadjacency_matrix_t = biadjacency_matrix.T
        # 如果转置后的格式不是 "csr"，则转换为 CSR 格式
        if biadjacency_matrix_t.format != "csr":
            biadjacency_matrix_t = biadjacency_matrix_t.tocsr()
        # 使用 Hopcroft-Karp 算法求解最大匹配
        matching, _ = _hopcroft_karp(biadjacency_matrix_t.indices,
                                     biadjacency_matrix_t.indptr,
                                     j, i)
        matching = np.asarray(matching)
        # 检查匹配结果是否完整
        if np.sum(matching != -1) != min(i, j):
            raise ValueError('no full matching exists')
        # 计算与 j 对应的度量 b
        b = np.asarray(_lapjvsp(biadjacency_matrix_t.indptr,
                                biadjacency_matrix_t.indices,
                                biadjacency_matrix_t.data,
                                j, i))
        # 按 b 的值对索引进行排序，并返回排序后的 b 和原始的 a
        indices = np.argsort(b)
        return (b[indices], a[indices])
    else:
        # 如果 j 大于等于 i，则执行以下操作
        # 如果原始二分邻接矩阵格式不是 "csr"，则转换为 CSR 格式
        if biadjacency_matrix.format != "csr":
            biadjacency_matrix = biadjacency_matrix.tocsr()
        # 使用 Hopcroft-Karp 算法求解最大匹配
        matching, _ = _hopcroft_karp(biadjacency_matrix.indices,
                                     biadjacency_matrix.indptr,
                                     i, j)
        matching = np.asarray(matching)
        # 检查匹配结果是否完整
        if np.sum(matching != -1) != min(i, j):
            raise ValueError('no full matching exists')
        # 计算与 i 对应的度量 b
        b = np.asarray(_lapjvsp(biadjacency_matrix.indptr,
                                biadjacency_matrix.indices,
                                biadjacency_matrix.data,
                                i, j))
        # 返回原始的 a 和计算得到的 b
        return (a, b)
# 我们使用 uint8 类型来表示布尔值，以简化下面布尔数组的处理。
BTYPE = np.uint8
# 为了在 Cython 中定义一个新的类型别名 BTYPE_t，我们使用 ctypedef 来声明其底层类型为 np.uint8_t。
ctypedef np.uint8_t BTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 C 函数 _lapjvsp，用于解决最小权重二分匹配问题，采用 LAPJVsp 算法。
cdef ITYPE_t[:] _lapjvsp(ITYPE_t[:] first,
                         ITYPE_t[:] kk,
                         DTYPE_t[:] cc,
                         const ITYPE_t nr,
                         const ITYPE_t nc) noexcept:
    """使用 LAPJVsp 解决最小权重二分匹配问题。

    这个实现是对 Anton Volgenant 的原始 Pascal 实现的直接移植。原始代码作为 LAPJVS.P
    在 [1]_ 上公开，并且使用 BSD-3 许可证，版权属于 A. Volgenant/阿姆斯特丹经济学院，
    阿姆斯特丹大学。

    原始代码未经注释，通过参考 [2]_ 要比直接查看代码容易理解。在我们的 Cython 移植中，我们
    尽可能地保持与原始实现的一致性，更喜欢易于从 Pascal 版本识别出来的代码，而非优化后的代码。
    当我们在非显而易见的方式上偏离原始实现时，我们会明确地描述如何以及为什么这样做。通常，我们
    更新所有的索引（因为 Pascal 使用 1 作为起始索引），引入了更多的变量来突出某些变量的作用域，
    将 goto 语句转换为函数调用，并使用双无穷大替代大整数。

    Parameters
    ----------
    first : 长度为 :math:`|U| + 1` 的 memoryview
        对应于 CSR 格式图的 ``indptr`` 属性。
    kk : memoryview
        对应于 CSR 格式图的 ``indices`` 属性。
    cc : memoryview
        对应于 CSR 格式图的 ``data`` 属性。
    nr : int
        CSR 格式图的行数。
    nc : int
        CSR 格式图的列数。必须至少与 ``nr`` 一样大。

    Returns
    -------
    row_ind : 长度为 :math:`|U|` 的 memoryview
        每行所匹配的列索引。

    References
    ----------
    .. [1] http://www.assignmentproblems.com/LAPJV.htm
    .. [2] Roy Jonker and Anton Volgenant:
       A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
       Assignment Problems.
       Computing 38:325-340, 1987.

    """
    cdef ITYPE_t l0, jp, t, i, lp, j1, tp, j0p, j1p, l0p, h, i0, td1
    cdef DTYPE_t min_diff, v0, vj, dj
    # 创建长度为 nc 的零数组 v，用于存储临时变量。
    cdef DTYPE_t[:] v = np.zeros(nc, dtype=DTYPE)
    # 创建长度为 nr 的空数组 x，用于存储行指派。
    cdef ITYPE_t[:] x = np.empty(nr, dtype=ITYPE)
    # 初始化 x 数组为 -1。
    for i in range(nr):
        x[i] = -1
    # 创建长度为 nc 的空数组 y，用于存储列指派。
    cdef ITYPE_t[:] y = np.empty(nc, dtype=ITYPE)
    # 初始化 y 数组为 -1。
    for j in range(nc):
        y[j] = -1
    # 创建长度为 nr 的零数组 u，用于存储行偏差。
    cdef DTYPE_t[:] u = np.zeros(nr, dtype=DTYPE)
    # 创建长度为 nc 的零数组 d，用于存储列偏差。
    cdef DTYPE_t[:] d = np.zeros(nc, dtype=DTYPE)
    # 创建长度为 nc 的零数组 ok，用于存储列状态。
    cdef BTYPE_t[:] ok = np.zeros(nc, dtype=BTYPE)
    # 创建长度为 nr 的零数组 xinv，用于存储行状态。
    cdef BTYPE_t[:] xinv = np.zeros(nr, dtype=BTYPE)
    # 创建长度为 nr 的空数组 free，用于存储自由链表。
    cdef ITYPE_t[:] free = np.empty(nr, dtype=ITYPE)
    # 初始化数组 `free` 中的每个元素为 -1
    for i in range(nr):
        free[i] = -1
    
    # 创建一个 NumPy 数组 `todo`，用于存储 `nc` 个元素，每个元素类型为 `ITYPE_t`，并初始化为 -1
    cdef ITYPE_t[:] todo = np.empty(nc, dtype=ITYPE)
    for j in range(nc):
        todo[j] = -1
    
    # 创建一个 NumPy 数组 `lab`，用于存储 `nc` 个元素，每个元素类型为 `ITYPE_t`，初始化为 0
    cdef ITYPE_t[:] lab = np.zeros(nc, dtype=ITYPE)

    # 如果不是方阵的情况，完全跳过初始化过程，而是直接将 `free` 数组填充为 [0, 1, ..., nr-1]
    else:
        l0 = nr
        for i in range(nr):
            free[i] = i

    # 根据原始 Pascal 代码，为了避免使用 goto 语句和复杂的逻辑来跳出嵌套循环，
    # 我们提取了循环体内容到单独的函数 `_lapjvsp_single_l` 中，该函数对应于 Pascal 代码中 109 至 154 行，
    # 155 和 156 行则在下面的两个单独函数中分开处理。
    # 对于每个 l 在范围 l0 内，调用 `_lapjvsp_single_l` 函数处理，并将结果存储在 `td1` 中
    td1 = -1
    for l in range(l0):
        td1 = _lapjvsp_single_l(l, nc, d, ok, free, first, kk, cc, v, lab,
                                todo, y, x, td1)
    
    # 返回结果数组 `x`
    return x
# 定义一个 C 函数，使用 Cython 进行优化，禁用边界检查和负数索引检查
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 _lapjvsp_single_l，接受多个参数，返回 ITYPE_t 类型结果或抛出异常
cdef ITYPE_t _lapjvsp_single_l(ITYPE_t l, ITYPE_t nc, DTYPE_t[:] d,
                               BTYPE_t[:] ok, ITYPE_t[:] free,
                               ITYPE_t[:] first, ITYPE_t[:] kk,
                               DTYPE_t[:] cc, DTYPE_t[:] v, ITYPE_t[:] lab,
                               ITYPE_t[:] todo, ITYPE_t[:] y, ITYPE_t[:] x,
                               ITYPE_t td1) except -2:
    # 声明变量
    cdef ITYPE_t jp, i0, j, t, td2, hp, last, j0, i, tp
    cdef DTYPE_t min_diff, dj, h, vj

    # 初始化 d 数组和 ok 数组
    for jp in range(nc):
        d[jp] = INFINITY
        ok[jp] = 0

    # 初始化 min_diff
    min_diff = INFINITY
    i0 = free[l]

    # 遍历范围 [first[i0], first[i0 + 1]) 的元素
    for t in range(first[i0], first[i0 + 1]):
        j = kk[t]
        dj = cc[t] - v[j]
        d[j] = dj
        lab[j] = i0
        # 更新 min_diff 和 todo 数组
        if dj <= min_diff:
            if dj < min_diff:
                td1 = -1
                min_diff = dj
            td1 += 1
            todo[td1] = j

    # 遍历 todo 数组
    for hp in range(td1 + 1):
        j = todo[hp]
        # 如果 y[j] 等于 -1，则更新 lab、y 和 x 数组，返回 td1
        if y[j] == -1:
            _lapjvsp_update_assignments(lab, y, x, j, i0)
            return td1
        ok[j] = 1

    # 设置 td2 和 last 的值
    td2 = nc - 1
    last = nc
    while True:
        # 如果此时 td1 是负数，表示前一次运行中没有分配，因此没有完全匹配存在，会抛出异常
        if td1 < 0:
            raise ValueError('no full matching exists')
        # 取出 todo 中索引为 td1 的元素作为 j0
        j0 = todo[td1]
        # td1 减一
        td1 -= 1
        # 取出 y 中索引为 j0 的元素作为 i
        i = y[j0]
        # 将 j0 放入 todo 中索引为 td2 的位置
        todo[td2] = j0
        # td2 减一
        td2 -= 1
        # 取出 first 中索引为 i 的值，赋给 tp
        tp = first[i]
        # 找到 kk 中第一个等于 j0 的索引位置，赋给 tp，即确定 tp 的值使得 kk[tp] == j0
        while kk[tp] != j0:
            tp += 1
        # 计算 h 的值，h = cc[tp] - v[j0] - min_diff
        h = cc[tp] - v[j0] - min_diff

        # 遍历 first[i] 到 first[i + 1] 之间的所有值
        for t in range(first[i], first[i + 1]):
            # 取出 kk 中索引为 t 的值作为 j
            j = kk[t]
            # 如果 ok[j] 等于 0
            if ok[j] == 0:
                # 计算 vj 的值，vj = cc[t] - v[j] - h
                vj = cc[t] - v[j] - h
                # 如果 vj 小于 d[j]
                if vj < d[j]:
                    # 更新 d[j] 的值为 vj
                    d[j] = vj
                    # 更新 lab[j] 的值为 i
                    lab[j] = i
                    # 如果 vj 等于 min_diff
                    if vj == min_diff:
                        # 如果 y[j] 等于 -1
                        if y[j] == -1:
                            # 调用 _lapjvsp_update_dual 函数更新双重变量
                            _lapjvsp_update_dual(nc, d, v, todo,
                                                 last, min_diff)
                            # 调用 _lapjvsp_update_assignments 更新分配
                            _lapjvsp_update_assignments(lab, y, x, j, i0)
                            # 返回 td1
                            return td1
                        # td1 加一
                        td1 += 1
                        # 将 j 放入 todo 中索引为 td1 的位置
                        todo[td1] = j
                        # 将 ok[j] 设置为 1
                        ok[j] = 1

        # 如果 td1 等于 -1
        if td1 == -1:
            # 原始的 Pascal 代码使用有限的数值代替 INFINITY，因此这里需要稍作调整
            min_diff = INFINITY
            # last 被设置为 td2 + 1
            last = td2 + 1

            # 遍历 nc 范围内的所有值
            for jp in range(nc):
                # 如果 d[jp] 不等于 INFINITY 且 d[jp] 小于等于 min_diff 并且 ok[jp] 等于 0
                if d[jp] != INFINITY and d[jp] <= min_diff and ok[jp] == 0:
                    # 如果 d[jp] 小于 min_diff
                    if d[jp] < min_diff:
                        # td1 设为 -1
                        td1 = -1
                        # min_diff 设为 d[jp]
                        min_diff = d[jp]
                    # td1 加一
                    td1 += 1
                    # 将 jp 放入 todo 中索引为 td1 的位置
                    todo[td1] = jp

            # 遍历 td1 + 1 范围内的所有值
            for hp in range(td1 + 1):
                # 取出 todo 中索引为 hp 的值作为 j
                j = todo[hp]
                # 如果 y[j] 等于 -1
                if y[j] == -1:
                    # 调用 _lapjvsp_update_dual 函数更新双重变量
                    _lapjvsp_update_dual(nc, d, v, todo, last, min_diff)
                    # 调用 _lapjvsp_update_assignments 更新分配
                    _lapjvsp_update_assignments(lab, y, x, j, i0)
                    # 返回 td1
                    return td1
                # 将 ok[j] 设置为 1
                ok[j] = 1
# 禁用 Cython 的边界检查和负数索引包装功能，用于优化性能
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 Cython 编译的函数，用于更新拉普拉斯算法的对偶变量
cdef _lapjvsp_update_dual(ITYPE_t nc, DTYPE_t[:] d, DTYPE_t[:] v,
                          ITYPE_t[:] todo, ITYPE_t last, DTYPE_t min_diff):
    # 声明整型变量 j0
    cdef ITYPE_t j0
    # 循环遍历 todo 数组，更新 v 数组中的值
    for k in range(last, nc):
        # 获取当前待处理的索引值 j0
        j0 = todo[k]
        # 更新 v[j0] 的值，根据给定公式计算
        v[j0] += d[j0] - min_diff


# 禁用 Cython 的边界检查和负数索引包装功能，用于优化性能
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 Cython 编译的函数，用于更新拉普拉斯算法的分配情况
cdef _lapjvsp_update_assignments(ITYPE_t[:] lab, ITYPE_t[:] y, ITYPE_t[:] x,
                                 ITYPE_t j, ITYPE_t i0):
    # 声明整型变量 i, k
    cdef ITYPE_t i, k
    # 无限循环，更新 y[j] 和 x[i] 的值，直到 i 等于 i0
    while True:
        # 获取 lab[j] 的值赋给 i
        i = lab[j]
        # 将 i 赋给 y[j]
        y[j] = i
        # 将 j 赋给 k
        k = j
        # 将 x[i] 的值赋给 j
        j = x[i]
        # 将 k 赋给 x[i]
        x[i] = k
        # 如果 i 等于 i0，则退出循环
        if i == i0:
            return
```