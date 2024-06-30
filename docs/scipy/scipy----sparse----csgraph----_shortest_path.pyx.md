# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_shortest_path.pyx`

```
        The N x N array of distances representing the input graph.
    method : {'auto', 'FW', 'D', 'BF', 'J', 'Yen'}, optional
        The algorithm to use for finding shortest paths:
        - 'auto': selects the best algorithm based on the input graph and other parameters.
        - 'FW': Floyd-Warshall algorithm for all-pairs shortest paths.
        - 'D': Dijkstra's algorithm with Fibonacci Heaps.
        - 'BF': Bellman-Ford algorithm for single-source shortest paths.
        - 'J': Johnson's algorithm for all-pairs shortest paths.
        - 'Yen': Yen's k-Shortest Paths algorithm for finding k-shortest paths.
    directed : bool, optional
        If True, treat the graph as directed (default). If False, treat it as undirected.
    return_predecessors : bool, optional
        If True, return predecessors along with the shortest distances.
    unweighted : bool, optional
        If True, interpret the graph as unweighted.
    overwrite : bool, optional
        If True, overwrite input data in-place for optimization.
    indices : tuple, optional
        Tuple of indices (i, j) specifying the submatrix of the shortest path.

    Raises
    ------
    ValueError
        If the input graph is not square (N != M).

    Returns
    -------
    dist_matrix : ndarray
        The matrix of shortest path distances between nodes.
    predecessors : ndarray, optional
        The matrix of predecessors, only returned if `return_predecessors` is True.

    Notes
    -----
    This function computes the shortest path between nodes in a graph using
    various algorithms depending on the method chosen. It supports both dense
    and sparse representations of the graph using NumPy arrays or SciPy sparse matrices.
    """
    # Validate the input graph and convert it to a CSR matrix if necessary
    csgraph = validate_graph(csgraph, directed=directed, unweighted=unweighted)
    
    # Determine the number of nodes (N) in the graph
    N = csgraph.shape[0]

    # If indices are specified, extract the submatrix of interest
    if indices is not None:
        i, j = indices
        csgraph = csgraph[i:j, i:j]

    # Select the appropriate shortest path algorithm based on the method parameter
    if method == 'auto':
        # Choose the best algorithm based on the characteristics of the graph
        if issparse(csgraph):
            method = 'D' if csgraph.nnz > N * np.log2(N) else 'BF'
        else:
            method = 'FW' if N < 1000 else 'D'
    
    # Call the appropriate Cython function for computing shortest paths
    if method == 'FW':
        return _shortest_path(csgraph, overwrite=overwrite)
    elif method == 'D':
        return _dijkstra(csgraph, return_predecessors=return_predecessors,
                         overwrite=overwrite)
    elif method == 'BF':
        return _bellman_ford(csgraph, return_predecessors=return_predecessors,
                             overwrite=overwrite)
    elif method == 'J':
        return _johnson(csgraph, return_predecessors=return_predecessors,
                        overwrite=overwrite)
    elif method == 'Yen':
        raise NotImplementedError("Yen's algorithm is not yet implemented.")

    # Raise an error if an unsupported method is requested
    else:
        raise ValueError(f"Unknown method '{method}' for computing shortest paths.")
    method : string ['auto'|'FW'|'D'], optional
        # 定义参数 method，用于选择最短路径算法
        Algorithm to use for shortest paths.  Options are:

           'auto' -- (default) select the best among 'FW', 'D', 'BF', or 'J'
                     based on the input data.

           'FW'   -- Floyd-Warshall algorithm.
                     Computational cost is approximately ``O[N^3]``.
                     The input csgraph will be converted to a dense representation.

           'D'    -- Dijkstra's algorithm with Fibonacci heaps.
                     Computational cost is approximately ``O[N(N*k + N*log(N))]``,
                     where ``k`` is the average number of connected edges per node.
                     The input csgraph will be converted to a csr representation.

           'BF'   -- Bellman-Ford algorithm.
                     This algorithm can be used when weights are negative.
                     If a negative cycle is encountered, an error will be raised.
                     Computational cost is approximately ``O[N(N^2 k)]``, where
                     ``k`` is the average number of connected edges per node.
                     The input csgraph will be converted to a csr representation.

           'J'    -- Johnson's algorithm.
                     Like the Bellman-Ford algorithm, Johnson's algorithm is
                     designed for use when the weights are negative. It combines
                     the Bellman-Ford algorithm with Dijkstra's algorithm for
                     faster computation.

    directed : bool, optional
        # 定义参数 directed，指定图的方向性
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    return_predecessors : bool, optional
        # 定义参数 return_predecessors，指定是否返回前驱矩阵
        If True, return the size (N, N) predecessor matrix.
    unweighted : bool, optional
        # 定义参数 unweighted，指定是否使用无权重距离
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    overwrite : bool, optional
        # 定义参数 overwrite，指定是否覆盖原始图数据
        If True, overwrite csgraph with the result.  This applies only if
        method == 'FW' and csgraph is a dense, c-ordered array with
        dtype=float64.
    indices : array_like or int, optional
        # 定义参数 indices，指定计算路径的起始点索引
        If specified, only compute the paths from the points at the given
        indices. Incompatible with method == 'FW'.

    Returns
    -------
    dist_matrix : ndarray
        # 返回值为距离矩阵 dist_matrix，描述节点之间的最短路径距离
        The N x N matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.
    """
    csgraph = convert_pydata_sparse_to_scipy(csgraph, accept_fv=[0, np.inf, np.nan])

    # validate here to catch errors early but don't store the result;
    # we'll validate again later
    validate_graph(csgraph, directed, DTYPE,
                   copy_if_dense=(not overwrite),
                   copy_if_sparse=(not overwrite))
    """

    # 将输入的稀疏图数据结构转换为 SciPy 的稀疏矩阵表示，并接受特定的缺失值定义
    csgraph = convert_pydata_sparse_to_scipy(csgraph, accept_fv=[0, np.inf, np.nan])

    # 在这里进行图的验证，以便尽早捕获错误，但不存储结果；
    # 我们稍后会再次进行验证
    validate_graph(csgraph, directed, DTYPE,
                   copy_if_dense=(not overwrite),
                   copy_if_sparse=(not overwrite))

    # 声明变量，用于标记稀疏性
    cdef bint is_sparse
    # 声明变量，用于存储节点数量
    cdef ssize_t N      # XXX cdef ssize_t Nk fails in Python 3 (?)
    # 如果方法是 'auto'，则根据节点数和边数猜测最快的方法
    if method == 'auto':
        # 获取图的节点数 N
        N = csgraph.shape[0]
        # 将 Python 数据稀疏矩阵转换为 scipy 稀疏矩阵格式
        csgraph = convert_pydata_sparse_to_scipy(csgraph)
        # 检查是否为稀疏矩阵
        is_sparse = issparse(csgraph)
        if is_sparse:
            # 如果是稀疏矩阵，获取非零元素的个数 Nk
            Nk = csgraph.nnz
            # 根据格式获取边的数据
            if csgraph.format in ('csr', 'csc', 'coo'):
                edges = csgraph.data
            else:
                edges = csgraph.tocoo().data
        elif np.ma.isMaskedArray(csgraph):
            # 如果是掩码数组，获取非掩码元素的个数 Nk
            Nk = csgraph.count()
            edges = csgraph.compressed()
        else:
            # 否则，获取有限元素并移除零元素，得到边的数据
            edges = csgraph[np.isfinite(csgraph)]
            edges = edges[edges != 0]
            Nk = edges.size

        # 如果指定了 indices 或者 Nk 小于 N*N/4
        if indices is not None or Nk < N * N / 4:
            # 如果边中有负数，选择方法 'J'
            if np.any(edges < 0):
                method = 'J'
            else:
                # 否则选择方法 'D'
                method = 'D'
        else:
            # 否则选择方法 'FW'
            method = 'FW'

    # 如果方法是 'FW'
    if method == 'FW':
        # 如果指定了 indices，抛出异常
        if indices is not None:
            raise ValueError("Cannot specify indices with method == 'FW'.")
        # 调用 Floyd-Warshall 算法计算最短路径
        return floyd_warshall(csgraph, directed,
                              return_predecessors=return_predecessors,
                              unweighted=unweighted,
                              overwrite=overwrite)

    # 如果方法是 'D'
    elif method == 'D':
        # 调用 Dijkstra 算法计算最短路径
        return dijkstra(csgraph, directed,
                        return_predecessors=return_predecessors,
                        unweighted=unweighted, indices=indices)

    # 如果方法是 'BF'
    elif method == 'BF':
        # 调用 Bellman-Ford 算法计算最短路径
        return bellman_ford(csgraph, directed,
                            return_predecessors=return_predecessors,
                            unweighted=unweighted, indices=indices)

    # 如果方法是 'J'
    elif method == 'J':
        # 调用 Johnson 算法计算最短路径
        return johnson(csgraph, directed,
                       return_predecessors=return_predecessors,
                       unweighted=unweighted, indices=indices)

    # 如果方法未被识别，抛出异常
    else:
        raise ValueError("unrecognized method '%s'" % method)
# 定义 Floyd-Warshall 算法函数，计算最短路径长度
def floyd_warshall(csgraph, directed=True,
                   return_predecessors=False,
                   unweighted=False,
                   overwrite=False):
    """
    floyd_warshall(csgraph, directed=True, return_predecessors=False,
                   unweighted=False, overwrite=False)

    Compute the shortest path lengths using the Floyd-Warshall algorithm

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array, matrix, or sparse matrix, 2 dimensions
        The N x N array of distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    return_predecessors : bool, optional
        If True, return the size (N, N) predecessor matrix.
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    overwrite : bool, optional
        If True, overwrite csgraph with the result.  This applies only if
        csgraph is a dense, c-ordered array with dtype=float64.

    Returns
    -------
    dist_matrix : ndarray
        The N x N matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.

    predecessors : ndarray
        Returned only if return_predecessors == True.
        The N x N matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999

    Raises
    ------
    NegativeCycleError:
        if there are negative cycles in the graph

    Notes
    -----
    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import floyd_warshall

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

    >>> dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    >>> dist_matrix
    # 验证并处理输入的图数据，返回距离矩阵
    dist_matrix = validate_graph(csgraph, directed, DTYPE,
                                 csr_output=False,
                                 copy_if_dense=not overwrite)
    
    if not issparse(csgraph):
        # 对于密集数组输入，零条目表示非边
        dist_matrix[dist_matrix == 0] = INFINITY

    if unweighted:
        # 如果是无权图，将非无穷大的距离设为1
        dist_matrix[~np.isinf(dist_matrix)] = 1

    if return_predecessors:
        # 如果需要返回前驱矩阵，创建一个与距离矩阵相同形状的空矩阵
        predecessor_matrix = np.empty(dist_matrix.shape,
                                      dtype=ITYPE, order='C')
    else:
        # 否则，创建一个空的零维矩阵
        predecessor_matrix = np.empty((0, 0), dtype=ITYPE)

    # 调用 Floyd-Warshall 算法计算最短路径和前驱矩阵
    _floyd_warshall(dist_matrix,
                    predecessor_matrix,
                    int(directed))

    # 检查是否存在负权重环路，如果存在则抛出异常
    if np.any(dist_matrix.diagonal() < 0):
        raise NegativeCycleError("Negative cycle in nodes %s"
                                 % np.where(dist_matrix.diagonal() < 0)[0])

    if return_predecessors:
        # 如果需要返回前驱矩阵，则返回距离矩阵和前驱矩阵
        return dist_matrix, predecessor_matrix
    else:
        # 否则，只返回距离矩阵
        return dist_matrix
@cython.boundscheck(False)
# 声明函数 _floyd_warshall，用于执行 Floyd-Warshall 算法，计算最短路径
cdef void _floyd_warshall(
               np.ndarray[DTYPE_t, ndim=2, mode='c'] dist_matrix,
               np.ndarray[ITYPE_t, ndim=2, mode='c'] predecessor_matrix,
               int directed=0) noexcept:
    # dist_matrix : in/out
    #    on input, the graph
    #    on output, the matrix of shortest paths
    # dist_matrix should be a [N,N] matrix, such that dist_matrix[i, j]
    # is the distance from point i to point j.  Zero-distances imply that
    # the points are not connected.
    # 输入参数和输出结果都是距离矩阵，表示图的结构和最短路径的矩阵
    cdef int N = dist_matrix.shape[0]
    assert dist_matrix.shape[1] == N

    cdef unsigned int i, j, k

    cdef DTYPE_t d_ijk

    # ----------------------------------------------------------------------
    #  Initialize distance matrix
    #   - set diagonal to zero
    #   - symmetrize matrix if non-directed graph is desired
    # 初始化距离矩阵
    # - 将对角线元素设为零
    # - 如果需要非定向图，则使矩阵对称化
    dist_matrix.flat[::N + 1] = 0
    if not directed:
        for i in range(N):
            for j in range(i + 1, N):
                if dist_matrix[j, i] <= dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    dist_matrix[j, i] = dist_matrix[i, j]

    #----------------------------------------------------------------------
    #  Initialize predecessor matrix
    #   - check matrix size
    #   - initialize diagonal and all non-edges to NULL
    #   - initialize all edges to the row index
    # 初始化前驱矩阵
    # - 检查矩阵大小
    # - 将对角线和所有非边设为 NULL
    # - 将所有边初始化为其行索引
    cdef int store_predecessors = False

    if predecessor_matrix.size > 0:
        store_predecessors = True
        assert predecessor_matrix.shape[0] == N
        assert predecessor_matrix.shape[1] == N
        predecessor_matrix.fill(NULL_IDX)
        i_edge = np.where(~np.isinf(dist_matrix))
        predecessor_matrix[i_edge] = i_edge[0]
        predecessor_matrix.flat[::N + 1] = NULL_IDX

    # Now perform the Floyd-Warshall algorithm.
    # In each loop, this finds the shortest path from point i
    #  to point j using intermediate nodes 0 ... k
    # 执行 Floyd-Warshall 算法
    # 在每次循环中，使用中间节点 0 ... k 找到点 i 到点 j 的最短路径
    if store_predecessors:
        for k in range(N):
            for i in range(N):
                if dist_matrix[i, k] == INFINITY:
                    continue
                for j in range(N):
                    d_ijk = dist_matrix[i, k] + dist_matrix[k, j]
                    if d_ijk < dist_matrix[i, j]:
                        dist_matrix[i, j] = d_ijk
                        predecessor_matrix[i, j] = predecessor_matrix[k, j]
    else:
        for k in range(N):
            for i in range(N):
                if dist_matrix[i, k] == INFINITY:
                    continue
                for j in range(N):
                    d_ijk = dist_matrix[i, k] + dist_matrix[k, j]
                    if d_ijk < dist_matrix[i, j]:
                        dist_matrix[i, j] = d_ijk


def dijkstra(csgraph, directed=True, indices=None,
             return_predecessors=False,
             unweighted=False, limit=np.inf,
             bint min_only=False):
    """
    Implementing Dijkstra's algorithm to find the shortest paths from a starting node
    in a graph represented by a compressed sparse graph (csgraph).

    Parameters:
    - csgraph: Compressed sparse graph representation
    - directed: Whether the graph is directed (default=True)
    - indices: Indices of the starting nodes (default=None)
    - return_predecessors: Whether to return predecessor information (default=False)
    - unweighted: Whether the graph is unweighted (default=False)
    - limit: Maximum value for the algorithm to search for (default=np.inf)
    - min_only: Whether to return only the minimum value (default=False)
    """
    # 使用 Dijkstra 算法计算最短路径距离矩阵
    
    Dijkstra algorithm using Fibonacci Heaps
    
    .. versionadded:: 0.11.0
    
    Parameters
    ----------
    csgraph : array, matrix, or sparse matrix, 2 dimensions
        表示输入图的 N x N 非负距离数组，用于计算最短路径。
    directed : bool, optional
        如果为 True（默认值），在有向图上查找最短路径：
        只能沿着 csgraph[i, j] 的路径从点 i 移动到点 j，以及沿着 csgraph[j, i] 的路径从点 j 移动到点 i。
        如果为 False，在无向图上查找最短路径：算法可以沿着 csgraph[i, j] 或 csgraph[j, i] 从点 i 移动到点 j 或从点 j 移动到点 i。
        注意：在使用 `directed=False` 时，请参考下面的注意事项。
    indices : array_like or int, optional
        如果指定，则仅计算从给定索引处的点开始的路径。
    return_predecessors : bool, optional
        如果为 True，则返回大小为 (N, N) 的前驱矩阵。
    unweighted : bool, optional
        如果为 True，则计算无权距离。即，不是找到使权重和最小化的每个点之间的路径，而是找到使边数最小化的路径。
    limit : float, optional
        要计算的最大距离，必须 >= 0。使用较小的限制将通过中止在距离 > limit 的对之间的计算来减少计算时间。
        对于这样的对，距离将等于 np.inf（即，不连接）。
        .. versionadded:: 0.14.0
    min_only : bool, optional
        如果为 False（默认值），则对于图中的每个节点，找到从索引中的每个节点到每个节点的最短路径。
        如果为 True，则对于图中的每个节点，找到从索引中任何节点到该节点的最短路径（这可能更快）。
        .. versionadded:: 1.3.0
    
    Returns
    -------
    dist_matrix : ndarray, shape ([n_indices, ]n_nodes,)
        图节点之间的距离矩阵。如果 min_only=False，则 dist_matrix 的形状为 (n_indices, n_nodes)，dist_matrix[i, j]
        给出了从点 i 到点 j 沿图的最短距离。如果 min_only=True，则 dist_matrix 的形状为 (n_nodes,)，并且对于给定节点，该节点到索引中任何节点的最短路径。
    #------------------------------
    # 验证 csgraph 并将其转换为 CSR 矩阵
    csgraph = validate_graph(csgraph, directed, DTYPE,
                             dense_output=False)

    # 如果图中存在负权重，则发出警告，因为 Dijkstra 算法无法处理包含负循环的图
    if np.any(csgraph.data < 0):
        warnings.warn("Graph has negative weights: dijkstra will give "
                      "inaccurate results if the graph contains negative "
                      "cycles. Consider johnson or bellman_ford.")

    # 获取图的节点数目
    N = csgraph.shape[0]

    #------------------------------
    # 初始化/验证索引
    # 如果未提供 indices 参数，则创建一个从 0 到 N-1 的整数数组作为默认值
    if indices is None:
        indices = np.arange(N, dtype=ITYPE)
        # 如果 min_only 为 True，则返回一个形状为 (N,) 的数组
        if min_only:
            return_shape = (N,)
        else:
            # 否则返回一个形状为 (len(indices), N) 的数组
            return_shape = indices.shape + (N,)
    else:
        # 将输入的 indices 转换为指定的数据类型 ITYPE 的数组，并确保是 C 风格的连续数组
        indices = np.array(indices, order='C', dtype=ITYPE, copy=True)
        # 如果 min_only 为 True，则返回一个形状为 (N,) 的数组
        if min_only:
            return_shape = (N,)
        else:
            # 否则返回一个形状为 (len(indices), N) 的数组
            return_shape = indices.shape + (N,)
        
        # 将 indices 至少转换为一维数组，并处理负值索引，使其在 0 到 N-1 范围内
        indices = np.atleast_1d(indices).reshape(-1)
        indices[indices < 0] += N
        # 检查是否有任何索引超出了 0 到 N-1 的范围，如果有则抛出 ValueError 异常
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError("indices out of range 0...N")

    # 将 limit 转换为本地变量 limitf，确保其值大于等于 0
    cdef DTYPE_t limitf = limit
    if limitf < 0:
        raise ValueError('limit must be >= 0')

    #------------------------------
    # 初始化用于输出的距离矩阵 dist_matrix
    if min_only:
        # 如果 min_only 为 True，则初始化一个形状为 (N,) 的数组，所有值为无穷大，索引 indices 处为 0
        dist_matrix = np.full(N, np.inf, dtype=DTYPE)
        dist_matrix[indices] = 0
    else:
        # 否则初始化一个形状为 (len(indices), N) 的数组，所有值为无穷大，主对角线上的值为 0
        dist_matrix = np.full((len(indices), N), np.inf, dtype=DTYPE)
        dist_matrix[np.arange(len(indices)), indices] = 0

    #------------------------------
    # 初始化用于输出的前驱矩阵 predecessor_matrix
    if return_predecessors:
        if min_only:
            # 如果 min_only 为 True，则初始化形状为 (N,) 的空数组，填充为 NULL_IDX
            predecessor_matrix = np.empty((N), dtype=ITYPE)
            predecessor_matrix.fill(NULL_IDX)
            # 同样初始化形状为 (N,) 的空数组，填充为 NULL_IDX，用于存储源节点信息
            source_matrix = np.empty((N), dtype=ITYPE)
            source_matrix.fill(NULL_IDX)
        else:
            # 否则初始化形状为 (len(indices), N) 的空数组，填充为 NULL_IDX
            predecessor_matrix = np.empty((len(indices), N), dtype=ITYPE)
            predecessor_matrix.fill(NULL_IDX)
    else:
        if min_only:
            # 如果 min_only 为 True 且不需要前驱矩阵，则初始化空数组
            predecessor_matrix = np.empty(0, dtype=ITYPE)
            source_matrix = np.empty(0, dtype=ITYPE)
        else:
            # 否则初始化形状为 (0, N) 的空数组
            predecessor_matrix = np.empty((0, N), dtype=ITYPE)

    # 根据 unweighted 参数确定使用的 csr_data
    if unweighted:
        csr_data = np.ones(csgraph.data.shape)
    else:
        csr_data = csgraph.data

    # 获取 csgraph 的索引和指针数组
    csgraph_indices = csgraph.indices
    csgraph_indptr = csgraph.indptr
    # 如果索引或指针数组的数据类型不是 ITYPE，则将其转换为 ITYPE
    if csgraph_indices.dtype != ITYPE:
        csgraph_indices = csgraph_indices.astype(ITYPE)
    if csgraph_indptr.dtype != ITYPE:
        csgraph_indptr = csgraph_indptr.astype(ITYPE)
    
    # 如果是有向图，根据 min_only 调用不同的 Dijkstra 算法函数
    if directed:
        if min_only:
            _dijkstra_directed_multi(indices,
                                     csr_data, csgraph_indices,
                                     csgraph_indptr,
                                     dist_matrix, predecessor_matrix,
                                     source_matrix, limitf)
        else:
            _dijkstra_directed(indices,
                               csr_data, csgraph_indices, csgraph_indptr,
                               dist_matrix, predecessor_matrix, limitf)
    # 如果不是有向图，则转置图的 CSR 表示用于计算
    else:
        # 将转置图表示为 CSR 格式
        csgraphT = csgraph.T.tocsr()
        
        # 根据是否为无权图选择合适的数据结构
        if unweighted:
            csrT_data = csr_data
        else:
            csrT_data = csgraphT.data
        
        # 如果只需要最短路径，调用对应的 Dijkstra 算法函数
        if min_only:
            _dijkstra_undirected_multi(indices,
                                       csr_data, csgraph_indices,
                                       csgraph_indptr,
                                       csrT_data, csgraphT.indices,
                                       csgraphT.indptr,
                                       dist_matrix, predecessor_matrix,
                                       source_matrix, limitf)
        else:
            # 否则调用普通的 Dijkstra 算法函数
            _dijkstra_undirected(indices,
                                 csr_data, csgraph_indices, csgraph_indptr,
                                 csrT_data, csgraphT.indices, csgraphT.indptr,
                                 dist_matrix, predecessor_matrix, limitf)

    # 如果需要返回前驱节点信息
    if return_predecessors:
        # 根据需求选择返回的数据结构形状
        if min_only:
            return (dist_matrix.reshape(return_shape),
                    predecessor_matrix.reshape(return_shape),
                    source_matrix.reshape(return_shape))
        else:
            return (dist_matrix.reshape(return_shape),
                    predecessor_matrix.reshape(return_shape))
    else:
        # 否则只返回距离矩阵
        return dist_matrix.reshape(return_shape)
# 禁用 Cython 中的边界检查优化
@cython.boundscheck(False)
# 定义一个 Cython 函数，用于多源 Dijkstra 算法的堆初始化设置
cdef _dijkstra_setup_heap_multi(FibonacciHeap *heap,
                                FibonacciNode* nodes,
                                const int[:] source_indices,
                                int[:] sources,
                                double[:] dist_matrix,
                                int return_pred):
    cdef:
        # 源索引数组的长度
        unsigned int Nind = source_indices.shape[0]
        # 节点数
        unsigned int N = dist_matrix.shape[0]
        # 循环中的计数变量
        unsigned int i, k, j_source
        # 当前处理的 FibonacciNode 指针
        FibonacciNode *current_node

    # 初始化所有节点
    for k in range(N):
        initialize_node(&nodes[k], k)

    # 将堆的最小节点指针设为 NULL
    heap.min_node = NULL
    # 对每一个源索引执行以下操作
    for i in range(Nind):
        # 获取源节点的索引
        j_source = source_indices[i]
        # 获取当前节点指针
        current_node = &nodes[j_source]
        # 如果当前节点已经被扫描过，则跳过
        if current_node.state == SCANNED:
            continue
        # 将源节点到自身的距离设置为 0
        dist_matrix[j_source] = 0
        # 如果需要返回前驱节点，则将源节点自身作为其前驱
        if return_pred:
            sources[j_source] = j_source
        # 标记当前节点为已扫描状态
        current_node.state = SCANNED
        # 设置当前节点的源索引
        current_node.source = j_source
        # 将当前节点插入堆中
        insert_node(heap, &nodes[j_source])

# 禁用 Cython 中的边界检查优化
@cython.boundscheck(False)
# 定义一个 Cython 函数，用于多源 Dijkstra 算法的堆扫描操作
cdef _dijkstra_scan_heap_multi(FibonacciHeap *heap,
                               FibonacciNode *v,
                               FibonacciNode* nodes,
                               const double[:] csr_weights,
                               const int[:] csr_indices,
                               const int[:] csr_indptr,
                               int[:] pred,
                               int[:] sources,
                               int return_pred,
                               DTYPE_t limit):
    cdef:
        # 当前邻接节点在 csr_indices 中的索引
        unsigned int j_current
        # 当前邻接节点的全局索引
        ITYPE_t j
        # 下一个值
        DTYPE_t next_val
        # 当前处理的 FibonacciNode 指针
        FibonacciNode *current_node

    # 遍历节点 v 的邻接节点
    for j in range(csr_indptr[v.index], csr_indptr[v.index + 1]):
        # 获取当前邻接节点的全局索引
        j_current = csr_indices[j]
        # 获取当前邻接节点的指针
        current_node = &nodes[j_current]
        # 如果当前邻接节点尚未被扫描过
        if current_node.state != SCANNED:
            # 计算从节点 v 到当前邻接节点的新路径值
            next_val = v.val + csr_weights[j]
            # 如果新路径值小于等于限制值
            if next_val <= limit:
                # 如果当前节点尚未在堆中，则将其插入堆中
                if current_node.state == NOT_IN_HEAP:
                    current_node.state = IN_HEAP
                    current_node.val = next_val
                    current_node.source = v.source
                    insert_node(heap, current_node)
                    # 如果需要返回前驱节点，则设置当前邻接节点的前驱和源节点
                    if return_pred:
                        pred[j_current] = v.index
                        sources[j_current] = v.source
                # 如果当前节点已经在堆中，并且新路径值更优，则更新堆中节点的值
                elif current_node.val > next_val:
                    current_node.source = v.source
                    decrease_val(heap, current_node,
                                 next_val)
                    # 如果需要返回前驱节点，则更新当前邻接节点的前驱和源节点
                    if return_pred:
                        pred[j_current] = v.index
                        sources[j_current] = v.source
# 定义一个 Cython 函数，使用 Fibonacci 堆优化的 Dijkstra 算法来处理有向图的单源最短路径问题
cdef _dijkstra_scan_heap(FibonacciHeap *heap,
                         FibonacciNode *v,
                         FibonacciNode* nodes,
                         const double[:] csr_weights,
                         const int[:] csr_indices,
                         const int[:] csr_indptr,
                         int[:, :] pred,
                         int return_pred,
                         DTYPE_t limit,
                         int i):
    cdef:
        unsigned int j_current  # 当前处理的邻接节点在 CSR 结构中的索引
        ITYPE_t j               # 当前处理的邻接节点的索引
        DTYPE_t next_val        # 下一个候选路径的权重值
        FibonacciNode *current_node  # 当前处理的邻接节点对象

    # 遍历节点 v 的所有邻接节点
    for j in range(csr_indptr[v.index], csr_indptr[v.index + 1]):
        j_current = csr_indices[j]
        current_node = &nodes[j_current]
        # 如果邻接节点尚未被处理过
        if current_node.state != SCANNED:
            next_val = v.val + csr_weights[j]
            # 如果从当前节点 v 经由邻接节点到达该节点的路径权重小于等于限制值
            if next_val <= limit:
                # 如果邻接节点尚未加入堆中
                if current_node.state == NOT_IN_HEAP:
                    current_node.state = IN_HEAP
                    current_node.val = next_val
                    insert_node(heap, current_node)
                    # 如果需要记录前驱节点，则记录
                    if return_pred:
                        pred[i, j_current] = v.index
                # 如果邻接节点已在堆中但是路径权重更优
                elif current_node.val > next_val:
                    decrease_val(heap, current_node, next_val)
                    # 如果需要记录前驱节点，则记录
                    if return_pred:
                        pred[i, j_current] = v.index

# 设置 Cython 函数的边界检查为关闭
@cython.boundscheck(False)
cdef int _dijkstra_directed(
            const int[:] source_indices,
            const double[:] csr_weights,
            const int[:] csr_indices,
            const int[:] csr_indptr,
            double[:, :] dist_matrix,
            int[:, :] pred,
            DTYPE_t limit) except -1:
    cdef:
        unsigned int Nind = dist_matrix.shape[0]  # dist_matrix 的行数
        unsigned int N = dist_matrix.shape[1]     # dist_matrix 的列数
        unsigned int i, k, j_source               # 循环变量和源节点索引
        int return_pred = (pred.size > 0)         # 是否需要记录前驱节点
        FibonacciHeap heap                        # Fibonacci 堆对象
        FibonacciNode *v                         # 当前处理的节点指针
        FibonacciNode* nodes = <FibonacciNode*> malloc(N *
                                                       sizeof(FibonacciNode))  # 节点数组的内存分配

    # 检查内存分配是否成功
    if nodes == NULL:
        raise MemoryError("Failed to allocate memory in _dijkstra_directed")

    # 对每个源节点进行 Dijkstra 算法的初始化和处理
    for i in range(Nind):
        j_source = source_indices[i]

        # 初始化所有节点对象
        for k in range(N):
            initialize_node(&nodes[k], k)

        # 设置源节点到自身的距离为 0，并将源节点加入堆中
        dist_matrix[i, j_source] = 0
        heap.min_node = NULL
        insert_node(&heap, &nodes[j_source])

        # 在堆非空的情况下执行 Dijkstra 算法
        while heap.min_node:
            v = remove_min(&heap)
            v.state = SCANNED

            # 使用 Fibonacci 堆优化的方法处理当前节点的邻接节点
            _dijkstra_scan_heap(&heap, v, nodes,
                                csr_weights, csr_indices, csr_indptr,
                                pred, return_pred, limit, i)

            # 将当前节点的最短路径距离记录到结果矩阵中
            dist_matrix[i, v.index] = v.val

    # 释放节点数组的内存
    free(nodes)
    return 0

# 设置 Cython 函数的边界检查为关闭
@cython.boundscheck(False)
cdef int _dijkstra_directed_multi(
            const int[:] source_indices,
            const double[:] csr_weights,
            const int[:] csr_indices,
            const int[:] csr_indptr,
            double[:] dist_matrix,
            int[:] pred,
            int[:] sources,
            DTYPE_t limit) except -1:
    cdef:
        unsigned int N = dist_matrix.shape[0]  # 获取距离矩阵的行数，即节点数目

        int return_pred = (pred.size > 0)  # 如果pred数组非空，则设置return_pred为真

        FibonacciHeap heap  # 创建一个Fibonacci堆
        FibonacciNode *v  # 声明一个Fibonacci节点指针v
        FibonacciNode* nodes = <FibonacciNode*> malloc(N * sizeof(FibonacciNode))  # 分配N个Fibonacci节点的内存空间
    if nodes == NULL:
        raise MemoryError("Failed to allocate memory in "
                          "_dijkstra_directed_multi")  # 如果分配内存失败，则抛出MemoryError异常

    # 使用起始节点初始化堆，将它们放入堆中并标记为已扫描状态，距离矩阵中的入口值为0
    # pred数组将指向起始索引中的一个
    _dijkstra_setup_heap_multi(&heap, nodes, source_indices,
                               sources, dist_matrix, return_pred)

    while heap.min_node:
        v = remove_min(&heap)  # 从堆中移除最小节点v
        v.state = SCANNED  # 将节点v标记为已扫描

        _dijkstra_scan_heap_multi(&heap, v, nodes,
                                  csr_weights, csr_indices, csr_indptr,
                                  pred, sources, return_pred, limit)
        # 节点v现在已经被扫描过：将距离添加到结果中
        dist_matrix[v.index] = v.val  # 将节点v的值写入距离矩阵中

    free(nodes)  # 释放分配的节点内存空间
    return 0  # 返回0表示函数执行成功

@cython.boundscheck(False)
cdef int _dijkstra_undirected(
            const int[:] source_indices,
            const double[:] csr_weights,
            const int[:] csr_indices,
            const int[:] csr_indptr,
            const double[:] csrT_weights,
            const int[:] csrT_indices,
            const int[:] csrT_indptr,
            double[:, :] dist_matrix,
            int[:, :] pred,
            DTYPE_t limit) except -1:
    cdef:
        unsigned int Nind = dist_matrix.shape[0]  # 获取距离矩阵的行数
        unsigned int N = dist_matrix.shape[1]  # 获取距离矩阵的列数
        unsigned int i, k, j_source
        int return_pred = (pred.size > 0)  # 如果pred数组非空，则设置return_pred为真

        FibonacciHeap heap  # 创建一个Fibonacci堆
        FibonacciNode *v  # 声明一个Fibonacci节点指针v
        FibonacciNode* nodes = <FibonacciNode*> malloc(N * sizeof(FibonacciNode))  # 分配N个Fibonacci节点的内存空间
    if nodes == NULL:
        raise MemoryError("Failed to allocate memory in _dijkstra_undirected")  # 如果分配内存失败，则抛出MemoryError异常
    # 遍历每个源节点索引，共循环 Nind 次
    for i in range(Nind):
        # 获取当前源节点的索引
        j_source = source_indices[i]

        # 对每个节点进行初始化操作，共循环 N 次
        for k in range(N):
            initialize_node(&nodes[k], k)

        # 将距离矩阵中从源节点 j_source 到自身的距离设为 0
        dist_matrix[i, j_source] = 0

        # 初始化堆的最小节点为空
        heap.min_node = NULL

        # 将源节点 j_source 插入堆中
        insert_node(&heap, &nodes[j_source])

        # 当堆不为空时进行循环
        while heap.min_node:
            # 从堆中移除最小节点 v
            v = remove_min(&heap)
            # 将节点 v 的状态标记为 SCANNED，表示已扫描过
            v.state = SCANNED

            # 使用正向 CSR 图扫描堆，更新节点距离和路径信息
            _dijkstra_scan_heap(&heap, v, nodes,
                                csr_weights, csr_indices, csr_indptr,
                                pred, return_pred, limit, i)

            # 使用反向 CSR 图扫描堆，更新节点距离和路径信息
            _dijkstra_scan_heap(&heap, v, nodes,
                                csrT_weights, csrT_indices, csrT_indptr,
                                pred, return_pred, limit, i)

            # 节点 v 现在已被扫描完成，将其距离值加入距离矩阵中
            dist_matrix[i, v.index] = v.val

    # 释放节点数组的内存空间
    free(nodes)

    # 返回执行成功状态码
    return 0
@cython.boundscheck(False)
# 使用 Cython 提供的装饰器来关闭边界检查，提升代码执行效率

cdef int _dijkstra_undirected_multi(
            const int[:] source_indices,
            const double[:] csr_weights,
            const int[:] csr_indices,
            const int[:] csr_indptr,
            const double[:] csrT_weights,
            const int[:] csrT_indices,
            const int[:] csrT_indptr,
            double[:] dist_matrix,
            int[:] pred,
            int[:] sources,
            DTYPE_t limit) except -1:
    # 定义 Cython 函数，用于执行多源无向图的 Dijkstra 算法
    cdef:
        unsigned int N = dist_matrix.shape[0]  # 获取距离矩阵的大小
        int return_pred = (pred.size > 0)  # 检查是否需要返回前驱节点信息
        FibonacciHeap heap  # 创建 Fibonacci 堆，用于存储节点和距离信息
        FibonacciNode *v  # 定义 Fibonacci 节点指针
        FibonacciNode* nodes = <FibonacciNode*> malloc(N *
                                                       sizeof(FibonacciNode))
        # 分配内存用于存储 Fibonacci 节点数组

    if nodes == NULL:
        # 如果内存分配失败，则抛出内存错误异常
        raise MemoryError("Failed to allocate memory in "
                          "_dijkstra_undirected_multi")

    _dijkstra_setup_heap_multi(&heap, nodes, source_indices,
                               sources, dist_matrix, return_pred)
    # 初始化堆数据结构，设置初始节点和距离信息

    while heap.min_node:
        # 当堆中存在节点时循环执行以下操作
        v = remove_min(&heap)
        # 移除堆中最小节点，并将其赋给变量 v
        v.state = SCANNED
        # 标记节点状态为已扫描

        _dijkstra_scan_heap_multi(&heap, v, nodes,
                                  csr_weights, csr_indices, csr_indptr,
                                  pred, sources, return_pred, limit)
        # 使用 Dijkstra 算法扫描堆，更新距离和前驱信息

        _dijkstra_scan_heap_multi(&heap, v, nodes,
                                  csrT_weights, csrT_indices, csrT_indptr,
                                  pred, sources, return_pred, limit)
        # 使用 Dijkstra 算法扫描堆（转置图），更新距离和前驱信息

        # v 现在已被扫描完成：将距离添加到结果中
        dist_matrix[v.index] = v.val
        # 将节点 v 的距离值存入距离矩阵中对应位置

    free(nodes)
    # 释放分配的内存空间
    return 0
    # 返回执行成功的标志


def bellman_ford(csgraph, directed=True, indices=None,
                 return_predecessors=False,
                 unweighted=False):
    """
    bellman_ford(csgraph, directed=True, indices=None, return_predecessors=False,
                 unweighted=False)

    Compute the shortest path lengths using the Bellman-Ford algorithm.

    The Bellman-Ford algorithm can robustly deal with graphs with negative
    weights.  If a negative cycle is detected, an error is raised.  For
    graphs without negative edge weights, Dijkstra's algorithm may be faster.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array, matrix, or sparse matrix, 2 dimensions
        The N x N array of distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    indices : array_like or int, optional
        if specified, only compute the paths from the points at the given
        indices.
    return_predecessors : bool, optional
        If True, return the size (N, N) predecessor matrix.
    """
    # 使用 Bellman-Ford 算法计算最短路径长度的函数，可以处理带有负权边的图

    # 函数体内容未提供，未包含在注释范围内
    """
    # ------------------------------
    # 验证 csgraph 并将其转换为 CSR 矩阵
    csgraph = validate_graph(csgraph, directed, DTYPE,
                             dense_output=False)
    N = csgraph.shape[0]

    # ------------------------------
    # 初始化/验证 indices
    if indices is None:
        indices = np.arange(N, dtype=ITYPE)
    else:
        indices = np.array(indices, order='C', dtype=ITYPE)
        indices[indices < 0] += N
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError("indices out of range 0...N")
    return_shape = indices.shape + (N,)
    indices = np.atleast_1d(indices).reshape(-1)

    # ------------------------------
    # 为输出初始化 dist_matrix
    dist_matrix = np.empty((len(indices), N), dtype=DTYPE)
    dist_matrix.fill(np.inf)
    dist_matrix[np.arange(len(indices)), indices] = 0

    # ------------------------------
    """
    # 如果需要返回前驱节点矩阵，则初始化一个空的前驱节点矩阵，形状为 (len(indices), N)，元素类型为 ITYPE
    if return_predecessors:
        predecessor_matrix = np.empty((len(indices), N), dtype=ITYPE)
        predecessor_matrix.fill(NULL_IDX)
    else:
        # 否则初始化一个空的前驱节点矩阵，形状为 (0, N)，元素类型为 ITYPE
        predecessor_matrix = np.empty((0, N), dtype=ITYPE)

    # 如果是无权图，将 csr_data 数组设为全为 1
    if unweighted:
        csr_data = np.ones(csgraph.data.shape)
    else:
        # 否则将 csr_data 数组设为 csgraph.data 的内容
        csr_data = csgraph.data

    # 如果是有向图，调用 _bellman_ford_directed 函数计算最短路径和前驱节点
    if directed:
        ret = _bellman_ford_directed(indices,
                                     csr_data, csgraph.indices,
                                     csgraph.indptr,
                                     dist_matrix, predecessor_matrix)
    else:
        # 否则调用 _bellman_ford_undirected 函数计算最短路径和前驱节点
        ret = _bellman_ford_undirected(indices,
                                       csr_data, csgraph.indices,
                                       csgraph.indptr,
                                       dist_matrix, predecessor_matrix)

    # 如果 ret 大于等于 0，表示检测到负权重环，抛出 NegativeCycleError 异常
    if ret >= 0:
        raise NegativeCycleError("Negative cycle detected on node %i" % ret)

    # 如果需要返回前驱节点矩阵，则返回重塑后的 dist_matrix 和 predecessor_matrix
    if return_predecessors:
        return (dist_matrix.reshape(return_shape),
                predecessor_matrix.reshape(return_shape))
    else:
        # 否则只返回重塑后的 dist_matrix
        return dist_matrix.reshape(return_shape)
# 定义一个 Cython 函数用于执行有向图的 Bellman-Ford 算法
cdef int _bellman_ford_directed(
            const int[:] source_indices,           # 源节点的索引数组
            const double[:] csr_weights,           # CSR 格式的权重数组
            const int[:] csr_indices,              # CSR 格式的列索引数组
            const int[:] csr_indptr,               # CSR 格式的行指针数组
            double[:, :] dist_matrix,              # 距离矩阵
            int[:, :] pred) noexcept:              # 前驱节点矩阵，用于记录路径

    cdef:
        unsigned int Nind = dist_matrix.shape[0]  # 距离矩阵的行数（索引数组长度）
        unsigned int N = dist_matrix.shape[1]      # 距离矩阵的列数
        unsigned int i, j, k, j_source, count      # 循环变量和计数器
        DTYPE_t d1, d2, w12                        # 临时变量，存储距离和权重
        int return_pred = (pred.size > 0)          # 是否返回前驱节点矩阵的标志

    # 对每个源节点执行 Bellman-Ford 算法
    for i in range(Nind):
        j_source = source_indices[i]

        # 对所有边进行 N-1 次松弛操作
        for count in range(N - 1):
            for j in range(N):
                d1 = dist_matrix[i, j]
                # 遍历从节点 j 出发的所有边
                for k in range(csr_indptr[j], csr_indptr[j + 1]):
                    w12 = csr_weights[k]
                    d2 = dist_matrix[i, csr_indices[k]]
                    # 如果通过节点 j 可以获得更短的路径，则更新距离和前驱节点
                    if d1 + w12 < d2:
                        dist_matrix[i, csr_indices[k]] = d1 + w12
                        if return_pred:
                            pred[i, csr_indices[k]] = j

        # 检查是否存在负权重环
        for j in range(N):
            d1 = dist_matrix[i, j]
            for k in range(csr_indptr[j], csr_indptr[j + 1]):
                w12 = csr_weights[k]
                d2 = dist_matrix[i, csr_indices[k]]
                # 如果存在负权重环，则返回源节点索引
                if d1 + w12 + DTYPE_EPS < d2:
                    return j_source

    # 如果没有负权重环，则返回 -1
    return -1


# 定义一个 Cython 函数用于执行无向图的 Bellman-Ford 算法
cdef int _bellman_ford_undirected(
            const int[:] source_indices,           # 源节点的索引数组
            const double[:] csr_weights,           # CSR 格式的权重数组
            const int[:] csr_indices,              # CSR 格式的列索引数组
            const int[:] csr_indptr,               # CSR 格式的行指针数组
            double[:, :] dist_matrix,              # 距离矩阵
            int[:, :] pred) noexcept:              # 前驱节点矩阵，用于记录路径

    cdef:
        unsigned int Nind = dist_matrix.shape[0]  # 距离矩阵的行数（索引数组长度）
        unsigned int N = dist_matrix.shape[1]      # 距离矩阵的列数
        unsigned int i, j, k, j_source, ind_k, count  # 循环变量和计数器
        DTYPE_t d1, d2, w12                        # 临时变量，存储距离和权重
        int return_pred = (pred.size > 0)          # 是否返回前驱节点矩阵的标志
    # 对每个源节点 i 进行松弛操作 N-1 次，其中 N 是节点总数
    for i in range(Nind):
        j_source = source_indices[i]

        # 松弛所有边 N-1 次
        for count in range(N - 1):
            # 遍历所有节点 j
            for j in range(N):
                d1 = dist_matrix[i, j]
                # 遍历节点 j 的邻接边
                for k in range(csr_indptr[j], csr_indptr[j + 1]):
                    w12 = csr_weights[k]
                    ind_k = csr_indices[k]
                    d2 = dist_matrix[i, ind_k]
                    # 如果通过节点 j 可以获得更短路径到 ind_k，则更新距离矩阵
                    if d1 + w12 < d2:
                        dist_matrix[i, ind_k] = d2 = d1 + w12
                        # 如果需要返回前驱矩阵，则更新前驱矩阵
                        if return_pred:
                            pred[i, ind_k] = j
                    # 如果通过 ind_k 可以获得更短路径到 j，则更新距离矩阵
                    if d2 + w12 < d1:
                        dist_matrix[i, j] = d1 = d2 + w12
                        # 如果需要返回前驱矩阵，则更新前驱矩阵
                        if return_pred:
                            pred[i, j] = ind_k

        # 检查是否存在负权重环
        for j in range(N):
            d1 = dist_matrix[i, j]
            # 遍历节点 j 的邻接边
            for k in range(csr_indptr[j], csr_indptr[j + 1]):
                w12 = csr_weights[k]
                d2 = dist_matrix[i, csr_indices[k]]
                # 如果存在负权重环，则返回源节点 j_source
                if abs(d2 - d1) > w12 + DTYPE_EPS:
                    return j_source

    # 如果没有负权重环，则返回 -1
    return -1
# 定义函数 johnson，用于计算使用 Johnson 算法得到的最短路径长度
def johnson(csgraph, directed=True, indices=None,
            return_predecessors=False,
            unweighted=False):
    """
    johnson(csgraph, directed=True, indices=None, return_predecessors=False,
            unweighted=False)

    Compute the shortest path lengths using Johnson's algorithm.

    Johnson's algorithm combines the Bellman-Ford algorithm and Dijkstra's
    algorithm to quickly find shortest paths in a way that is robust to
    the presence of negative cycles.  If a negative cycle is detected,
    an error is raised.  For graphs without negative edge weights,
    dijkstra may be faster.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array, matrix, or sparse matrix, 2 dimensions
        The N x N array of distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    indices : array_like or int, optional
        if specified, only compute the paths from the points at the given
        indices.
    return_predecessors : bool, optional
        If True, return the size (N, N) predecessor matrix.
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.

    Returns
    -------
    dist_matrix : ndarray
        The N x N matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.

    predecessors : ndarray
        Returned only if return_predecessors == True.
        The N x N matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999

    Raises
    ------
    NegativeCycleError:
        if there are negative cycles in the graph

    Notes
    -----
    This routine is specially designed for graphs with negative edge weights.
    If all edge weights are positive, then Dijkstra's algorithm is a better
    choice.

    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import johnson

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [2, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    """
    # ------------------------------
    # 如果是无权图，则没有负权重：直接使用 Dijkstra 算法
    if unweighted:
        return dijkstra(csgraph, directed, indices,
                        return_predecessors, unweighted)

    # ------------------------------
    # 验证 csgraph 并转换为 CSR 矩阵格式
    csgraph = validate_graph(csgraph, directed, DTYPE,
                             dense_output=False)
    N = csgraph.shape[0]

    # ------------------------------
    # 初始化/验证 indices
    if indices is None:
        indices = np.arange(N, dtype=ITYPE)
        return_shape = indices.shape + (N,)
    else:
        indices = np.array(indices, order='C', dtype=ITYPE)
        return_shape = indices.shape + (N,)
        indices = np.atleast_1d(indices).reshape(-1)
        indices[indices < 0] += N
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError("indices out of range 0...N")

    #------------------------------
    # 初始化输出用的 dist_matrix
    dist_matrix = np.empty((len(indices), N), dtype=DTYPE)
    dist_matrix.fill(np.inf)
    dist_matrix[np.arange(len(indices)), indices] = 0

    #------------------------------
    # 初始化输出用的 predecessors
    if return_predecessors:
        predecessor_matrix = np.empty((len(indices), N), dtype=ITYPE)
        predecessor_matrix.fill(NULL_IDX)
    else:
        predecessor_matrix = np.empty((0, N), dtype=ITYPE)

    #------------------------------
    # 初始化距离数组
    dist_array = np.zeros(N, dtype=DTYPE)

    csr_data = csgraph.data.copy()

    #------------------------------
    # 在这里，我们首先向图中添加一个单独的节点，通过权重为零的有向边连接到每个节点，并执行贝尔曼-福特算法
    if directed:
        ret = _johnson_directed(csr_data, csgraph.indices,
                                csgraph.indptr, dist_array)
    else:
        ret = _johnson_undirected(csr_data, csgraph.indices,
                                  csgraph.indptr, dist_array)

    if ret >= 0:
        raise NegativeCycleError("Negative cycle detected on node %i" % ret)

    #------------------------------
    # 将贝尔曼-福特算法计算得到的权重添加到数据中
    _johnson_add_weights(csr_data, csgraph.indices,
                         csgraph.indptr, dist_array)

    if directed:
        _dijkstra_directed(indices,
                           csr_data, csgraph.indices, csgraph.indptr,
                           dist_matrix, predecessor_matrix, np.inf)
    # 如果不是稀疏图模式，则转置稀疏图对象csgraph，生成csgraph的转置矩阵csgraphT
    csgraphT = csr_matrix((csr_data, csgraph.indices, csgraph.indptr),
                           csgraph.shape).T.tocsr()
    # 使用Johnson算法为csgraphT添加权重，权重存储在dist_array中
    _johnson_add_weights(csgraphT.data, csgraphT.indices,
                         csgraphT.indptr, dist_array)
    # 使用Dijkstra算法计算无向图的最短路径
    _dijkstra_undirected(indices,
                         csr_data, csgraph.indices, csgraph.indptr,
                         csgraphT.data, csgraphT.indices, csgraphT.indptr,
                         dist_matrix, predecessor_matrix, np.inf)

# ------------------------------
# 校正距离矩阵以适应Bellman-Ford算法的权重
dist_matrix += dist_array  # 将距离矩阵加上dist_array的值
dist_matrix -= dist_array[:, None][indices]  # 从距离矩阵中减去dist_array与indices相关的部分

if return_predecessors:
    # 如果需要返回前驱矩阵，则返回形状调整后的距离矩阵和前驱矩阵
    return (dist_matrix.reshape(return_shape),
            predecessor_matrix.reshape(return_shape))
else:
    # 否则，只返回形状调整后的距离矩阵
    return dist_matrix.reshape(return_shape)
# 定义一个 C 函数，用于计算 Johnson 算法中权重更新的操作
cdef void _johnson_add_weights(
            double[:] csr_weights,  # CSR 格式的权重数组
            const int[:] csr_indices,  # CSR 格式的列索引数组
            const int[:] csr_indptr,  # CSR 格式的指针数组
            const double[:] dist_array) noexcept:
    # 计算节点数 N
    cdef unsigned int j, k, N = dist_array.shape[0]

    # 遍历所有节点 j
    for j in range(N):
        # 遍历节点 j 的所有邻接边 k
        for k in range(csr_indptr[j], csr_indptr[j + 1]):
            # 更新边的权重 w(u, v) = w(u, v) + h(u) - h(v)
            csr_weights[k] += dist_array[j]
            csr_weights[k] -= dist_array[csr_indices[k]]


# 定义一个 C 函数，实现 Johnson 算法中的有向图最短路径计算
cdef int _johnson_directed(
            const double[:] csr_weights,  # CSR 格式的权重数组
            const int[:] csr_indices,  # CSR 格式的列索引数组
            const int[:] csr_indptr,  # CSR 格式的指针数组
            double[:] dist_array) noexcept:
    # 注意：dist_array 的内容必须在进入函数时初始化为零
    cdef:
        unsigned int N = dist_array.shape[0]  # 计算节点数 N
        unsigned int j, k, count  # 循环计数器和节点索引
        DTYPE_t d1, d2, w12  # 路径长度和权重

    # 松弛所有边 (N+1) - 1 次
    for count in range(N):
        # 遍历所有节点 j
        for j in range(N):
            d1 = dist_array[j]
            # 遍历节点 j 的所有邻接边 k
            for k in range(csr_indptr[j], csr_indptr[j + 1]):
                w12 = csr_weights[k]
                d2 = dist_array[csr_indices[k]]
                # 如果可以通过节点 j 松弛到节点 csr_indices[k]，则更新路径长度
                if d1 + w12 < d2:
                    dist_array[csr_indices[k]] = d1 + w12

    # 检查是否存在负权重环
    for j in range(N):
        d1 = dist_array[j]
        # 再次遍历所有节点 j 的邻接边 k
        for k in range(csr_indptr[j], csr_indptr[j + 1]):
            w12 = csr_weights[k]
            d2 = dist_array[csr_indices[k]]
            # 如果发现负权重环，则返回环的起始节点索引
            if d1 + w12 + DTYPE_EPS < d2:
                return j

    # 如果没有负权重环，则返回 -1
    return -1


# 定义一个 C 函数，实现 Johnson 算法中的无向图最短路径计算
cdef int _johnson_undirected(
            const double[:] csr_weights,  # CSR 格式的权重数组
            const int[:] csr_indices,  # CSR 格式的列索引数组
            const int[:] csr_indptr,  # CSR 格式的指针数组
            double[:] dist_array) noexcept:
    # 注意：dist_array 的内容必须在进入函数时初始化为零
    cdef:
        unsigned int N = dist_array.shape[0]  # 计算节点数 N
        unsigned int j, k, ind_k, count  # 循环计数器和节点索引
        DTYPE_t d1, d2, w12  # 路径长度和权重

    # 松弛所有边 (N+1) - 1 次
    for count in range(N):
        # 遍历所有节点 j
        for j in range(N):
            d1 = dist_array[j]
            # 遍历节点 j 的所有邻接边 k
            for k in range(csr_indptr[j], csr_indptr[j + 1]):
                w12 = csr_weights[k]
                ind_k = csr_indices[k]
                d2 = dist_array[ind_k]
                # 如果可以通过节点 j 松弛到节点 ind_k，更新路径长度
                if d1 + w12 < d2:
                    dist_array[ind_k] = d1 + w12
                # 如果可以通过节点 k 松弛到节点 j，也更新路径长度
                if d2 + w12 < d1:
                    dist_array[j] = d1 = d2 + w12

    # 检查是否存在负权重环
    for j in range(N):
        d1 = dist_array[j]
        # 再次遍历所有节点 j 的邻接边 k
        for k in range(csr_indptr[j], csr_indptr[j + 1]):
            w12 = csr_weights[k]
            d2 = dist_array[csr_indices[k]]
            # 如果发现负权重环，则返回环的起始节点索引
            if abs(d2 - d1) > w12 + DTYPE_EPS:
                return j

    # 如果没有负权重环，则返回 -1
    return -1


######################################################################
# FibonacciNode 结构体
# 这个结构体及其操作是 Fibonacci 堆的节点
#
cdef enum FibonacciState:
    SCANNED  # 节点状态：已扫描
    NOT_IN_HEAP  # 节点状态：不在堆中
    IN_HEAP  # 节点状态：在堆中


cdef struct FibonacciNode:
    unsigned int index  # 节点的索引
    # 声明无符号整数变量 rank，用于表示斐波那契堆中节点的排名或者优先级
    unsigned int rank
    # 声明无符号整数变量 source，用于表示节点的来源或标识
    unsigned int source
    # 声明 FibonacciState 类型的变量 state，用于存储节点的状态信息，可能包括活跃状态等
    FibonacciState state
    # 声明 DTYPE_t 类型的变量 val，用于存储节点的值或者关联的数据
    DTYPE_t val
    # 声明 FibonacciNode* 类型的指针变量 parent，指向当前节点的父节点
    FibonacciNode* parent
    # 声明 FibonacciNode* 类型的指针变量 left_sibling，指向当前节点的左兄弟节点
    FibonacciNode* left_sibling
    # 声明 FibonacciNode* 类型的指针变量 right_sibling，指向当前节点的右兄弟节点
    FibonacciNode* right_sibling
    # 声明 FibonacciNode* 类型的指针变量 children，指向当前节点的孩子节点的链表头
    FibonacciNode* children
cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=0) noexcept:
    # 初始化 Fibonacci 节点的各个字段
    # 假设：- node 是一个有效的指针
    #       - node 当前不属于任何堆
    node.index = index
    node.source = -9999
    node.val = val
    node.rank = 0
    node.state = NOT_IN_HEAP

    node.parent = NULL
    node.left_sibling = NULL
    node.right_sibling = NULL
    node.children = NULL


cdef FibonacciNode* leftmost_sibling(FibonacciNode* node) noexcept:
    # 查找给定节点的最左边的兄弟节点
    # 假设：- node 是一个有效的指针
    cdef FibonacciNode* temp = node
    while(temp.left_sibling):
        temp = temp.left_sibling
    return temp


cdef void add_child(FibonacciNode* node, FibonacciNode* new_child) noexcept:
    # 将一个节点添加为另一个节点的子节点
    # 假设：- node 是一个有效的指针
    #       - new_child 是一个有效的指针
    #       - new_child 目前不是任何其他节点的兄弟或子节点
    new_child.parent = node

    if node.children:
        add_sibling(node.children, new_child)
    else:
        node.children = new_child
        new_child.right_sibling = NULL
        new_child.left_sibling = NULL
        node.rank = 1


cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling) noexcept:
    # 将一个节点添加为另一个节点的兄弟节点
    # 假设：- node 是一个有效的指针
    #       - new_sibling 是一个有效的指针
    #       - new_sibling 目前不是任何其他节点的子节点或兄弟节点

    # 在 node 和 node.right_sibling 之间插入 new_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = new_sibling
    new_sibling.right_sibling = node.right_sibling
    new_sibling.left_sibling = node
    node.right_sibling = new_sibling

    new_sibling.parent = node.parent
    if new_sibling.parent:
        new_sibling.parent.rank += 1


cdef void remove(FibonacciNode* node) noexcept:
    # 移除一个节点
    # 假设：- node 是一个有效的指针
    if node.parent:
        node.parent.rank -= 1
        if node.parent.children == node:  # node 是最左边的兄弟节点
            node.parent.children = node.right_sibling

    if node.left_sibling:
        node.left_sibling.right_sibling = node.right_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = node.left_sibling

    node.left_sibling = NULL
    node.right_sibling = NULL
    node.parent = NULL


######################################################################
# FibonacciHeap 结构
#  这个结构及其操作使用 FibonacciNode 来实现 Fibonacci 堆

ctypedef FibonacciNode* pFibonacciNode


cdef struct FibonacciHeap:
    # 在这种表示中，min_node 始终位于链表的最左边，因此 min_node.left_sibling 总是 NULL.
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # 最大节点数大约为 ~2^100.


cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node) noexcept:
    # 如果 heap.min_node 存在，则进入条件判断
    if heap.min_node:
        # 如果当前节点 node 的值小于 heap.min_node 的值
        if node.val < heap.min_node.val:
            # 将当前节点 node 插入到根链表的最左侧，取代 heap.min_node 的位置
            node.left_sibling = NULL
            node.right_sibling = heap.min_node
            heap.min_node.left_sibling = node
            heap.min_node = node
        else:
            # 如果当前节点 node 的值不小于 heap.min_node 的值，则将当前节点作为 heap.min_node 的兄弟节点添加
            add_sibling(heap.min_node, node)
    else:
        # 如果 heap.min_node 不存在，则将当前节点 node 设为 heap.min_node
        heap.min_node = node
cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval) noexcept:
    # 减小节点值的操作，假设以下前提成立：
    # - heap 是一个有效的指针
    # - newval 小于等于 node.val
    # - node 是一个有效的指针
    # - node 不是任何其他节点的子节点或兄弟节点
    # - node 在堆中

    # 更新节点的值为新值
    node.val = newval

    # 如果节点有父节点并且其父节点的值大于等于新值
    if node.parent and (node.parent.val >= newval):
        # 将节点从当前位置移除
        remove(node)
        # 将节点插入到堆中适当位置
        insert_node(heap, node)
    
    # 否则，如果堆的最小节点的值大于节点的值
    elif heap.min_node.val > node.val:
        # 替换堆的最小节点为当前节点，该节点总是根链表中最左边的节点
        remove(node)
        node.right_sibling = heap.min_node
        heap.min_node.left_sibling = node
        heap.min_node = node


cdef void link(FibonacciHeap* heap, FibonacciNode* node) noexcept:
    # 将一个节点链接到斐波那契堆中的操作，假设以下前提成立：
    # - heap 是一个有效的指针
    # - node 是一个有效的指针
    # - node 已经在堆中

    cdef FibonacciNode *linknode

    # 如果对应 rank 的根节点为空，则将当前节点作为根节点
    if heap.roots_by_rank[node.rank] == NULL:
        heap.roots_by_rank[node.rank] = node
    else:
        # 否则，将当前 rank 的根节点与当前节点链接起来
        linknode = heap.roots_by_rank[node.rank]
        heap.roots_by_rank[node.rank] = NULL

        # 根据节点值的大小或当前节点是否为最小节点来决定链接方式
        if node.val < linknode.val or node == heap.min_node:
            remove(linknode)
            add_child(node, linknode)
            link(heap, node)
        else:
            remove(node)
            add_child(linknode, node)
            link(heap, linknode)


cdef FibonacciNode* remove_min(FibonacciHeap* heap) noexcept:
    # 从斐波那契堆中移除最小节点的操作，假设以下前提成立：
    # - heap 是一个有效的指针
    # - heap.min_node 是一个有效的指针

    cdef:
        FibonacciNode *temp
        FibonacciNode *temp_right
        FibonacciNode *out
        unsigned int i

    # 将所有最小节点的子节点变为根节点
    temp = heap.min_node.children

    while temp:
        temp_right = temp.right_sibling
        remove(temp)
        add_sibling(heap.min_node, temp)
        temp = temp_right

    # 移除最小根节点，并选择另一个根节点作为暂时的最小根节点
    out = heap.min_node
    temp = heap.min_node.right_sibling
    remove(heap.min_node)
    heap.min_node = temp

    # 如果当前没有其他根节点
    if temp == NULL:
        # 树中只有一个根节点，因此唯一的最小节点就是我们要返回的节点
        return out

    # 重新连接堆
    for i in range(100):
        heap.roots_by_rank[i] = NULL

    while temp:
        if temp.val < heap.min_node.val:
            heap.min_node = temp
        temp_right = temp.right_sibling
        link(heap, temp)
        temp = temp_right

    # 将 heap.min_node 移动到根链表的最左端
    temp = leftmost_sibling(heap.min_node)
    if heap.min_node != temp:
        remove(heap.min_node)
        heap.min_node.right_sibling = temp
        temp.left_sibling = heap.min_node

    return out
# Debugging: Functions for printing the Fibonacci heap
#
#cdef void print_node(FibonacciNode* node, int level=0) noexcept:
#    print('%s(%i,%i) %i' % (level*' ', node.index, node.val, node.rank))
#    if node.children:
#        print_node(node.children, level+1)
#    if node.right_sibling:
#        print_node(node.right_sibling, level)
#
#
#cdef void print_heap(FibonacciHeap* heap) noexcept:
#    print("---------------------------------")
#    if heap.min_node:
#        print("min node: (%i, %i)" % (heap.min_node.index, heap.min_node.val))
#        print_node(heap.min_node)
#    else:
#        print("[empty heap]")

######################################################################

# Author: Tomer Sery  -- <tomersery28@gmail.com>
# License: BSD 3-clause ("New BSD License"), (C) 2024

def yen(
    csgraph,
    source,
    sink,
    K,
    *,
    directed=True,
    return_predecessors=False,
    unweighted=False,
):
    """
    yen(csgraph, source, sink, K, *, directed=True, return_predecessors=False,
        unweighted=False)

    Yen's K-Shortest Paths algorithm on a directed or undirected graph.

    .. versionadded:: 1.14.0

    Parameters
    ----------
    csgraph : array or sparse array, 2 dimensions
        The N x N array of distances representing the input graph.
    source : int
        The index of the starting node for the paths.
    sink : int
        The index of the ending node for the paths.
    K : int
        The number of shortest paths to find.
    directed : bool, optional
        If ``True`` (default), then find the shortest path on a directed graph:
        only move from point ``i`` to point ``j`` along paths ``csgraph[i, j]``.
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along ``csgraph[i, j]`` or
        ``csgraph[j, i]``.
    return_predecessors : bool, optional
        If ``True``, return the size ``(M, N)`` predecessor matrix. Default: ``False``.
    unweighted : bool, optional
        If ``True``, then find unweighted distances. That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized. Default: ``False``.

    Returns
    -------
    dist_array : ndarray
        Array of size ``M`` of shortest distances between the source and sink nodes.
        ``dist_array[i]`` gives the i-th shortest distance from the source to the sink
        along the graph. ``M`` is the number of shortest paths found, which is less than or
        equal to `K`.
    """
    # 使用 validate_graph 函数验证输入的图 csgraph，并确保其符合要求
    csgraph = validate_graph(csgraph, directed, DTYPE, dense_output=False)

    # 设置节点数量 N 为图 csgraph 的行数
    cdef int N = csgraph.shape[0]
    
    # 初始化是否有负权重的标志，默认为 False
    cdef int has_negative_weights = False
    
    # 初始化距离数组 dist_array，大小为 K，所有元素初始值为无穷大
    dist_array = np.full(K, INFINITY, dtype=DTYPE)

    # 初始化前驱矩阵 predecessor_matrix，大小为 K x N，所有元素初始值为 NULL_IDX
    predecessor_matrix = np.full((K, N), NULL_IDX, dtype=ITYPE)

    # 如果图是无权图，则将 csr_data 设置为全为 1 的数组
    if unweighted:
        csr_data = np.ones(csgraph.data.shape)
    else:
        # 复制 CSR 图的数据
        csr_data = csgraph.data.copy()
        # 检查是否存在负权重，如果存在，则需要使用 Johnson 算法处理
        if np.any(csr_data < 0):
            # 标记存在负权重
            has_negative_weights = True
            # 初始化 Johnson 算法所需的距离数组
            johnson_dist_array = np.zeros(N, dtype=DTYPE)
            # 根据图的方向选择适当的 Johnson 算法函数
            if directed:
                ret = _johnson_directed(csr_data, csgraph.indices,
                                        csgraph.indptr, johnson_dist_array)
            else:
                ret = _johnson_undirected(csr_data, csgraph.indices,
                                          csgraph.indptr, johnson_dist_array)
            # 如果返回值表明检测到负环，则抛出异常
            if ret >= 0:
                raise NegativeCycleError("Negative cycle detected on node %i" % ret)
    
    # 如果图中存在负权重，则调整 CSR 数据的权重
    if has_negative_weights:
        _johnson_add_weights(csr_data, csgraph.indices, csgraph.indptr,
                             johnson_dist_array)

    # 如果图是有向图，则转置图为原始图
    if directed:
        csgraphT = csgraph
        # 初始化转置图的 CSR 数据
        csrT_data = np.empty(0, dtype=DTYPE)
    else:
        # 获取原始图的转置图，并将其转换为 CSR 格式
        csgraphT = csgraph.T.tocsr()
        # 如果是无权重图，则转置图的 CSR 数据与原始图相同
        if unweighted:
            csrT_data = csr_data
        else:
            # 如果原始图存在负权重，则将转置图的权重进行调整
            if has_negative_weights:
                _johnson_add_weights(csgraphT.data, csgraphT.indices,
                                     csgraphT.indptr, johnson_dist_array)
            csrT_data = csgraphT.data

    # 调用 Yen's K-Shortest Paths 算法计算最短路径
    _yen(
        source, sink,
        csr_data, csgraph.indices, csgraph.indptr,
        csrT_data, csgraphT.indices, csgraphT.indptr,
        dist_array, predecessor_matrix,
    )

    # 如果存在负权重，则调整距离数组以反映 Johnson 算法的影响
    if has_negative_weights:
        dist_array += johnson_dist_array[sink] - johnson_dist_array[source]

    # 计算找到的路径数量
    num_paths_found = sum(dist_array < INFINITY)
    # 设置返回结果的形状
    return_shape = (num_paths_found, N)
    # 如果需要返回前驱矩阵，则同时返回距离数组和前驱矩阵
    if return_predecessors:
        return (dist_array[:num_paths_found].reshape((num_paths_found,)),
                predecessor_matrix[:num_paths_found].reshape(return_shape))
    # 否则，只返回距离数组
    return dist_array[:num_paths_found].reshape((num_paths_found,))
# 禁用边界检查以提高性能
@cython.boundscheck(False)
# 定义一个私有函数 _yen，用于计算 Yen's K-Shortest Paths 算法的核心部分
cdef void _yen(
    # 源节点和汇节点的索引
    const int source,
    const int sink,
    # 原始权重数组及其 CSR 格式的索引数组
    const double[:] original_weights, const int[:] csr_indices, const int[:] csr_indptr,
    # 转置权重数组及其 CSR 格式的索引数组
    const double[:] originalT_weights, const int[:] csrT_indices, const int[:] csrT_indptr,
    # 存储最短距离的数组
    double[:] shortest_distances,
    # 存储最短路径的前驱矩阵
    int[:, :] shortest_paths_predecessors,
):
    cdef:
        # 要找到的路径数目 K
        int K = shortest_paths_predecessors.shape[0]
        # 图中节点的总数 N
        int N = shortest_paths_predecessors.shape[1]
        # 判断是否是有向图
        bint directed = originalT_weights.size == 0

        # Dijkstra 算法中使用的操作数和结果数组
        # 源节点到目标节点的索引数组
        int[:] indice_node_arr = np.array([source], dtype=ITYPE)
        # 存储前驱节点的矩阵，初始化为 NULL_IDX
        int[:, :] predecessor_matrix = np.full((1, N), NULL_IDX, dtype=ITYPE)
        # 存储距离的矩阵，初始化为无穷大
        double[:, :] dist_matrix = np.full((1, N), np.inf, dtype=DTYPE)
    # 将源节点到自身的距离设为 0
    dist_matrix[0, source] = 0

    # ---------------------------------------------------
    # 计算并存储最短路径
    if directed:
        # 如果是有向图，使用 _dijkstra_directed 函数计算最短路径
        _dijkstra_directed(
            indice_node_arr,
            original_weights, csr_indices, csr_indptr,
            dist_matrix, predecessor_matrix, INFINITY,
        )
    else:
        # 如果是无向图，使用 _dijkstra_undirected 函数计算最短路径
        _dijkstra_undirected(
            indice_node_arr,
            original_weights, csr_indices, csr_indptr,
            originalT_weights, csrT_indices, csrT_indptr,
            dist_matrix, predecessor_matrix, INFINITY,
        )

    # 将计算得到的最短距离存入 shortest_distances 数组
    shortest_distances[0] = dist_matrix[0, sink]
    # 如果最短距离为无穷大，表示源节点和汇节点之间不存在路径
    if shortest_distances[0] == INFINITY:
        # 返回，表示没有路径连接源节点和汇节点
        return
    # 如果是有向图，避免复制大小为 0 的内存视图
    if directed:
        # 将原始转置权重数组设为原始权重数组
        originalT_weights = original_weights

    cdef:
        # 初始化候选数组
        # 对于索引 'i'，candidate_distances[i] 存储在 candidate_predecessors[i, :] 中路径的距离
        double[:] candidate_distances = np.full(K, INFINITY, dtype=DTYPE)
        # 候选前驱矩阵，初始化为 NULL_IDX
        int[:, :] candidate_predecessors = np.full((K, N), NULL_IDX, dtype=ITYPE)
        # 存储原始图权重，用于恢复图
        double[:] csr_weights = original_weights.copy()
        double[:] csrT_weights = originalT_weights.copy()

        # 初始化变量
        int k, i, spur_node, node, short_path_idx, tmp_i
        double root_path_distance, total_distance, tmp_d

    # 将最短路径复制到 shortest_paths_predecessors 数组中
    node = sink
    while node != NULL_IDX:
        shortest_paths_predecessors[0, node] = predecessor_matrix[0, node]
        node = predecessor_matrix[0, node]


# ---------------------------------------------------
# 检查路径是否在候选路径中存在的函数
@cython.boundscheck(False)
cdef bint _yen_is_path_in_candidates(
    # 候选前驱矩阵，存储候选路径的前驱节点
    const int[:, :] candidate_predecessors,
    # 原始路径数组和骨刺路径数组
    const int[:] orig_path, const int[:] spur_path,
    # 骨刺节点和汇节点的索引
    const int spur_node, const int sink
):
    """
    如果由 orig_path 和 spur_path 合并形成的路径存在于 candidate_predecessors 中，则返回 1。
    如果不存在，则返回 0。
    """
    cdef int i
    cdef int node
    cdef bint break_flag = 0
    for i in range(candidate_predecessors.shape[0]):
        node = sink
        break_flag = 0
        while node != spur_node:
            # 从汇点向候选前驱节点反向检查路径
            if candidate_predecessors[i, node] != spur_path[node]:
                # 如果路径不匹配，设置中断标志并退出循环
                break_flag = 1
                break
            node = candidate_predecessors[i, node]
        if break_flag:
            # 如果中断标志为真，表示路径不匹配，继续下一个候选前驱节点的检查
            continue
        while node != NULL_IDX:
            # 检查从候选前驱节点到源节点的路径
            if candidate_predecessors[i, node] != orig_path[node]:
                # 如果路径不匹配，设置中断标志并退出循环
                break_flag = 1
                break
            node = candidate_predecessors[i, node]
        if break_flag == 0:
            # 如果中断标志为假，表示路径匹配，返回结果为1
            return 1
    # 所有候选路径均未匹配，返回结果为0
    return 0
```