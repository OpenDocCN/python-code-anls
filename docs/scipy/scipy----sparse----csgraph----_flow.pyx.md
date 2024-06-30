# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_flow.pyx`

```
# cython: wraparound=False, boundscheck=False
# 设置 Cython 编译选项，禁用数组边界检查和负数索引的包装

import numpy as np
# 导入 NumPy 库

from scipy.sparse import csr_matrix, issparse
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy, is_pydata_spmatrix
# 导入 SciPy 库中稀疏矩阵相关模块和函数

cimport numpy as np
# 使用 Cython 导入 NumPy 库

include 'parameters.pxi'
# 包含预编译的 Cython 参数文件

np.import_array()
# 调用 NumPy 的 import_array() 函数，导入 NumPy 的 C API

class MaximumFlowResult:
    """Represents the result of a maximum flow calculation.

    Attributes
    ----------
    flow_value : int
        The value of the maximum flow.
    flow : csr_matrix
        The maximum flow.
    """

    def __init__(self, flow_value, flow):
        self.flow_value = flow_value
        self.flow = flow

    def __repr__(self):
        return 'MaximumFlowResult with value of %d' % self.flow_value
    # 返回描述对象的字符串表示形式

def maximum_flow(csgraph, source, sink, *, method='dinic'):
    r"""
    maximum_flow(csgraph, source, sink)

    Maximize the flow between two vertices in a graph.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    csgraph : csr_matrix
        The square matrix representing a directed graph whose (i, j)'th entry
        is an integer representing the capacity of the edge between
        vertices i and j.
    source : int
        The source vertex from which the flow flows.
    sink : int
        The sink vertex to which the flow flows.
    method: {'edmonds_karp', 'dinic'}, optional
        The method/algorithm to be used for computing the maximum flow.
        Following methods are supported,

            * 'edmonds_karp': Edmonds Karp algorithm in [1]_.
            * 'dinic': Dinic's algorithm in [4]_.

        Default is 'dinic'.

        .. versionadded:: 1.8.0

    Returns
    -------
    res : MaximumFlowResult
        A maximum flow represented by a ``MaximumFlowResult``
        which includes the value of the flow in ``flow_value``,
        and the flow graph in ``flow``.

    Raises
    ------
    TypeError:
        if the input graph is not in CSR format.

    ValueError:
        if the capacity values are not integers, or the source or sink are out
        of bounds.

    Notes
    -----
    This solves the maximum flow problem on a given directed weighted graph:
    A flow associates to every edge a value, also called a flow, less than the
    capacity of the edge, so that for every vertex (apart from the source and
    the sink vertices), the total incoming flow is equal to the total outgoing
    flow. The value of a flow is the sum of the flow of all edges leaving the
    source vertex, and the maximum flow problem consists of finding a flow
    whose value is maximal.

    By the max-flow min-cut theorem, the maximal value of the flow is also the
    total weight of the edges in a minimum cut.

    To solve the problem, we provide Edmonds--Karp [1]_ and Dinic's algorithm
    [4]_. The implementation of both algorithms strive to exploit sparsity.
    The time complexity of the former :math:`O(|V|\,|E|^2)` and its space
    complexity is :math:`O(|E|)`. The latter achieves its performance by
    """

    # 根据给定的方法选择相应的最大流算法
    building level graphs and finding blocking flows in them. Its time
    complexity is :math:`O(|V|^2\,|E|)` and its space complexity is
    :math:`O(|E|)`.


    # 构建级别图并在其中寻找阻塞流。其时间复杂度为 :math:`O(|V|^2\,|E|)`，空间复杂度为 :math:`O(|E|)`。



    The maximum flow problem is usually defined with real valued capacities,
    but we require that all capacities are integral to ensure convergence. When
    dealing with rational capacities, or capacities belonging to
    :math:`x\mathbb{Q}` for some fixed :math:`x \in \mathbb{R}`, it is possible
    to reduce the problem to the integral case by scaling all capacities
    accordingly.


    # 最大流问题通常定义在具有实值容量的情况下，但我们要求所有容量都是整数，以确保收敛性。当处理有理数容量或属于 :math:`x\mathbb{Q}`（其中 :math:`x \in \mathbb{R}`）的容量时，可以通过按比例缩放所有容量来将问题简化为整数情况。



    Solving a maximum-flow problem can be used for example for graph cuts
    optimization in computer vision [3]_.


    # 解决最大流问题可以用于例如在计算机视觉中的图像分割优化 [3]_。



    References
    ----------
    .. [1] Edmonds, J. and Karp, R. M.
           Theoretical improvements in algorithmic efficiency for network flow
           problems. 1972. Journal of the ACM. 19 (2): pp. 248-264
    .. [2] Cormen, T. H. and Leiserson, C. E. and Rivest, R. L. and Stein C.
           Introduction to Algorithms. Second Edition. 2001. MIT Press.
    .. [3] https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision
    .. [4] Dinic, Efim A.
           Algorithm for solution of a problem of maximum flow in networks with
           power estimation. In Soviet Math. Doklady, vol. 11, pp. 1277-1280.
           1970.


    # 参考文献
    # [1] Edmonds, J. and Karp, R. M.
    #        网络流问题中算法效率的理论改进。1972年。Journal of the ACM。19 (2): pp. 248-264
    # [2] Cormen, T. H. and Leiserson, C. E. and Rivest, R. L. and Stein C.
    #        算法导论。第二版。2001年。MIT出版社。
    # [3] https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision
    # [4] Dinic, Efim A.
    #        解决带功率估算网络最大流问题的算法。在Soviet Math. Doklady，卷11，pp. 1277-1280中。1970年。



    Examples
    --------
    Perhaps the simplest flow problem is that of a graph of only two vertices
    with an edge from source (0) to sink (1)::

        (0) --5--> (1)

    Here, the maximum flow is simply the capacity of the edge:

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import maximum_flow
    >>> graph = csr_matrix([[0, 5], [0, 0]])
    >>> maximum_flow(graph, 0, 1).flow_value
    5
    >>> maximum_flow(graph, 0, 1, method='edmonds_karp').flow_value
    5


    # 示例
    # 或许最简单的流问题是仅有两个顶点的图，从源点 (0) 到汇点 (1) 有一条边::
    #
    #     (0) --5--> (1)
    #
    # 在这里，最大流量简单地是边的容量：
    #
    # >>> import numpy as np
    # >>> from scipy.sparse import csr_matrix
    # >>> from scipy.sparse.csgraph import maximum_flow
    # >>> graph = csr_matrix([[0, 5], [0, 0]])
    # >>> maximum_flow(graph, 0, 1).flow_value
    # 5
    # >>> maximum_flow(graph, 0, 1, method='edmonds_karp').flow_value
    # 5



    If, on the other hand, there is a bottleneck between source and sink, that
    can reduce the maximum flow::

        (0) --5--> (1) --3--> (2)

    >>> graph = csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
    >>> maximum_flow(graph, 0, 2).flow_value
    3


    # 另一方面，如果源点和汇点之间存在瓶颈，这可能会减少最大流量::
    #
    #     (0) --5--> (1) --3--> (2)
    #
    # >>> graph = csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
    # >>> maximum_flow(graph, 0, 2).flow_value
    # 3



    A less trivial example is given in [2]_, Chapter 26.1:

    >>> graph = csr_matrix([[0, 16, 13,  0,  0,  0],
    ...                     [0,  0, 10, 12,  0,  0],
    ...                     [0,  4,  0,  0, 14,  0],
    ...                     [0,  0,  9,  0,  0, 20],
    ...                     [0,  0,  0,  7,  0,  4],
    ...                     [0,  0,  0,  0,  0,  0]])
    >>> maximum_flow(graph, 0, 5).flow_value
    23


    # 一个不那么简单的例子见于 [2]_，第26.1章节:
    #
    # >>> graph = csr_matrix([[0, 16, 13,  0,  0,  0],
    # ...                     [0,  0, 10, 12,  0,  0],
    # ...                     [0,  4,  0,  0, 14,  0],
    # ...                     [0,  0,  9,  0,  0, 20],
    # ...                     [0,  0,  0,  7,  0,  4],
    # ...                     [0,  0,  0,  0,  0,  0]])
    # >>> maximum_flow(graph, 0, 5).flow_value
    # 23



    It is possible to reduce the problem of finding a maximum matching in a
    bipartite graph to a maximum flow problem: Let :math:`G = ((U, V), E)` be a
    bipartite graph. Then, add to the graph a source vertex with edges to every
    vertex in :math:`U` and a sink vertex with edges from every vertex in
    :math:`V`. Finally, give every edge in the resulting graph a capacity of 1.


    # 可以将在二分图中找到最大匹配的问题简化为最大流问题：设 :math:`G = ((U, V), E)` 是一个二分图。然后，向图中添加一个源顶点，它与 :math:`U` 中的每个顶点连接边，以及一个汇顶点，它从 :math:`V` 中的每个顶点连接边。最后，将结果图中的每条边容量设为1。
    # 检查输入的图结构是否为 PyData Sparse 矩阵
    is_pydata_sparse = is_pydata_spmatrix(csgraph)
    # 如果是 PyData Sparse 矩阵，则记录其类别
    if is_pydata_sparse:
        pydata_sparse_cls = csgraph.__class__
    # 将 PyData Sparse 矩阵转换为 Scipy 的 CSR 格式
    csgraph = convert_pydata_sparse_to_scipy(csgraph, target_format="csr")
    # 检查转换后的结果是否为 Scipy 的 CSR 格式稀疏矩阵，否则抛出类型错误
    if not (issparse(csgraph) and csgraph.format == "csr"):
        raise TypeError("graph must be in CSR format")
    # 检查图的容量数据类型是否为整数类型的子类，如果不是则引发数值错误异常
    if not issubclass(csgraph.dtype.type, np.integer):
        raise ValueError("graph capacities must be integers")
    # 如果图的数据类型不是指定的整数类型，则将其转换为指定的整数类型
    elif csgraph.dtype != ITYPE:
        csgraph = csgraph.astype(ITYPE)
    # 检查源点和汇点是否相同，如果相同则引发数值错误异常
    if source == sink:
        raise ValueError("source and sink vertices must differ")
    # 检查图的形状是否为方阵，如果不是则引发数值错误异常
    if csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError("graph must be specified as a square matrix.")
    # 检查源点的值是否在合理范围内，如果不是则引发数值错误异常
    if source < 0 or source >= csgraph.shape[0]:
        raise ValueError('source value ({}) must be between '.format(source) +
                         '0 and {}'.format(csgraph.shape[0] - 1))
    # 检查汇点的值是否在合理范围内，如果不是则引发数值错误异常
    if sink < 0 or sink >= csgraph.shape[0]:
        raise ValueError('sink value ({}) must be between '.format(sink) +
                         '0 and {}'.format(csgraph.shape[0] - 1))

    # 如果图未按排序索引排列，则进行排序
    if not csgraph.has_sorted_indices:
        csgraph = csgraph.sorted_indices()

    # 添加缺失的反向边，这是最大流算法的先决条件之一
    m = _add_reverse_edges(csgraph)
    # 创建反向边指针
    rev_edge_ptr = _make_edge_pointers(m)
    
    # 根据选择的方法调用相应的最大流算法函数
    if method == 'edmonds_karp':
        # 使用 Edmonds-Karp 算法计算最大流
        tails = _make_tails(m)
        flow = _edmonds_karp(m.indptr, tails, m.indices,
                             m.data, rev_edge_ptr, source, sink)
    elif method == 'dinic':
        # 使用 Dinic 算法计算最大流
        flow = _dinic(m.indptr, m.indices, m.data, rev_edge_ptr,
                      source, sink)
    else:
        # 如果选择的方法不支持，则引发数值错误异常
        raise ValueError('{} method is not supported yet.'.format(method))
    
    # 将计算得到的最大流转换为 NumPy 数组
    flow_array = np.asarray(flow)
    # 根据最大流数据创建稀疏矩阵
    flow_matrix = csr_matrix((flow_array, m.indices, m.indptr),
                             shape=m.shape)
    
    # 如果是 PyData 稀疏矩阵，将其转换为 PyData 稀疏矩阵的实例
    if is_pydata_sparse:
        flow_matrix = pydata_sparse_cls.from_scipy_sparse(flow_matrix)
    
    # 计算源点的流出量并返回最大流结果对象
    source_flow = flow_array[m.indptr[source]:m.indptr[source + 1]]
    return MaximumFlowResult(source_flow.sum(), flow_matrix)
# 定义一个函数，用于在图中的所有边上添加反向边
def _add_reverse_edges(a):
    """Add reversed edges to all edges in a graph.

    This adds to a given directed weighted graph all edges in the reverse
    direction and give them weight 0, unless they already exist.

    Parameters
    ----------
    a : csr_matrix
        The square matrix in CSR format representing a directed graph

    Returns
    -------
    res : csr_matrix
        A new matrix in CSR format in which the missing edges are represented
        by explicit zeros.

    """
    # 引用输入矩阵的相关数组。
    cdef ITYPE_t n = a.shape[0]  # 获取矩阵的维度作为节点数量
    cdef ITYPE_t[:] a_data_view = a.data  # 获取矩阵数据的视图
    cdef ITYPE_t[:] a_indices_view = a.indices  # 获取矩阵列索引的视图
    cdef ITYPE_t[:] a_indptr_view = a.indptr  # 获取矩阵行指针的视图

    # 创建转置矩阵，以便使用其索引数组来添加零容量的反向边。
    # 实际上我们不使用转置矩阵的值，只用它的索引存在的事实。
    at = csr_matrix(a.transpose())
    cdef ITYPE_t[:] at_indices_view = at.indices  # 获取转置矩阵的列索引视图
    cdef ITYPE_t[:] at_indptr_view = at.indptr  # 获取转置矩阵的行指针视图

    # 为结果矩阵创建数组，添加反向边。
    # 为数据分配两倍于 `a` 非零元素数量的空间，这通常足够了。
    # 如果 `a` 中已经有一些反向边，这可能会分配过多的空间，但csr_matrix会隐式截断超出indptr给定的索引的data和indices元素。
    res_data = np.zeros(2 * a.nnz, ITYPE)
    cdef ITYPE_t[:] res_data_view = res_data  # 结果矩阵数据的视图
    res_indices = np.zeros(2 * a.nnz, ITYPE)
    cdef ITYPE_t[:] res_indices_view = res_indices  # 结果矩阵列索引的视图
    res_indptr = np.zeros(n + 1, ITYPE)
    cdef ITYPE_t[:] res_indptr_view = res_indptr  # 结果矩阵行指针的视图

    cdef ITYPE_t i = 0  # 初始化行索引
    cdef ITYPE_t res_ptr = 0  # 初始化结果矩阵指针
    cdef ITYPE_t a_ptr, a_end, at_ptr, at_end  # 定义其他循环变量
    cdef bint move_a, move_at  # 定义布尔值变量来判断是否移动指针
    # 循环遍历所有行
    # 当 i 不等于 n 时，执行以下循环
    while i != n:
        # 对于每一行，确保结果矩阵具有排序后的索引，
        # 同时同时遍历 a 和 a 的转置 a.T 的第 i 行，
        # 只有在不会破坏排序的情况下，才会在一个矩阵中移动指针。
        
        # 获取当前行 i 在 a 和 a.T 的行指针范围
        a_ptr, a_end = a_indptr_view[i], a_indptr_view[i + 1]
        at_ptr, at_end = at_indptr_view[i], at_indptr_view[i + 1]
        
        # 当 a_ptr 和 at_ptr 没有遍历完当前行的所有元素时执行循环
        while a_ptr != a_end or at_ptr != at_end:
            # 判断是否应该移动 a_ptr 指针
            move_a = a_ptr != a_end \
                     and (at_ptr == at_end
                          or a_indices_view[a_ptr] <= at_indices_view[at_ptr])
            # 判断是否应该移动 at_ptr 指针
            move_at = at_ptr != at_end \
                      and (a_ptr == a_end
                           or at_indices_view[at_ptr] <= a_indices_view[a_ptr])
            
            # 如果应该移动 a_ptr 指针，则将对应的索引和数据放入结果视图中
            if move_a:
                # 注意，有可能同时移动两个指针的情况。
                # 在这种情况下，我们显式地希望使用原始矩阵的值。
                res_indices_view[res_ptr] = a_indices_view[a_ptr]
                res_data_view[res_ptr] = a_data_view[a_ptr]
                a_ptr += 1
            
            # 如果应该移动 at_ptr 指针，则将对应的索引放入结果视图中
            if move_at:
                res_indices_view[res_ptr] = at_indices_view[at_ptr]
                at_ptr += 1
            
            # 移动结果指针到下一个位置
            res_ptr += 1
        
        # 将结果指针位置记录到结果行指针视图中
        res_indptr_view[i] = res_ptr
        
        # 更新行索引 i
        i += 1
    
    # 返回 Compressed Sparse Row (CSR) 格式的矩阵，由结果数据、结果索引和结果行指针构成，
    # 形状为 (n, n)
    return csr_matrix((res_data, res_indices, res_indptr), shape=(n, n))
# 定义一个函数来创建每条边的指向其反向边的指针数组
def _make_edge_pointers(a):
    """Create for each edge pointers to its reverse."""
    # 获取数组 a 的行数
    cdef int n = a.shape[0]
    # 创建一个与数组 a 的数据长度相同的整数数组 b_data，用来表示每条边的索引
    b_data = np.arange(a.data.shape[0], dtype=ITYPE)
    # 使用 csr_matrix 构造函数创建稀疏矩阵 b，其中数据为 b_data，列索引为 a.indices，行指针为 a.indptr
    b = csr_matrix(
        (b_data, a.indices, a.indptr), shape=(n, n), dtype=ITYPE)
    # 将 b 转置为 CSR 格式
    b = csr_matrix(b.transpose())
    # 返回矩阵 b 中的数据数组
    return b.data


# 定义一个函数来创建每条边的指向其尾部的指针数组
def _make_tails(a):
    """Create for each edge pointers to its tail."""
    # 获取数组 a 的行数
    cdef int n = a.shape[0]
    # 创建一个长度为数组 a 的数据长度的整数数组 tails，用于存储每条边的尾部顶点
    cdef ITYPE_t[:] tails = np.empty(a.data.shape[0], dtype=ITYPE)
    # 获取数组 a 的行指针视图
    cdef ITYPE_t[:] a_indptr_view = a.indptr
    cdef ITYPE_t i, j
    # 遍历数组 a 的每一行
    for i in range(n):
        # 遍历当前行 i 的每条边
        for j in range(a_indptr_view[i], a_indptr_view[i + 1]):
            # 将当前边 j 的尾部顶点设置为 i
            tails[j] = i
    # 返回尾部顶点数组 tails
    return tails


# 定义 C 语言扩展函数 _edmonds_karp，用于解决最大流问题的 Edmonds-Karp 算法
cdef ITYPE_t[:] _edmonds_karp(
        const ITYPE_t[:] edge_ptr,
        const ITYPE_t[:] tails,
        const ITYPE_t[:] heads,
        const ITYPE_t[:] capacities,
        const ITYPE_t[:] rev_edge_ptr,
        const ITYPE_t source,
        const ITYPE_t sink) noexcept:
    """Solves the maximum flow problem using the Edmonds--Karp algorithm.

    This assumes that for every edge in the graph, the edge in the opposite
    direction is also in the graph (possibly with capacity 0).

    Parameters
    ----------
    edge_ptr : memoryview of length :math:`|V| + 1`
        For a given vertex v, the edges whose tail is ``v`` are those between
        ``edge_ptr[v]`` and ``edge_ptr[v + 1] - 1``.
    tails : memoryview of length :math:`|E|`
        For a given edge ``e``, ``tails[e]`` is the tail vertex of ``e``.
    heads : memoryview of length :math:`|E|`
        For a given edge ``e``, ``heads[e]`` is the head vertex of ``e``.
    capacities : memoryview of length :math:`|E|`
        For a given edge ``e``, ``capacities[e]`` is the capacity of ``e``.
    rev_edge_ptr : memoryview of length :math:`|E|`
        For a given edge ``e``, ``rev_edge_ptr[e]`` is the edge obtained by
        reversing ``e``. In particular, ``rev_edge_ptr[rev_edge_ptr[e]] == e``.
    source : int
        The source vertex.
    sink : int
        The sink vertex.

    Returns
    -------
    flow : memoryview of length :math:`|E|`
        The flow graph with respect to a maximum flow.

    """
    # 获取顶点数目
    cdef ITYPE_t n_verts = edge_ptr.shape[0] - 1
    # 获取边数目
    cdef ITYPE_t n_edges = capacities.shape[0]
    # 获取 ITYPE 类型的最大值
    cdef ITYPE_t ITYPE_MAX = np.iinfo(ITYPE).max

    # 创建一个数组 flow，用于记录每条边的流量
    cdef ITYPE_t[:] flow = np.zeros(n_edges, dtype=ITYPE)

    # 创建一个循环队列 q 用于广度优先搜索，队列元素的弹出位置为 start，插入位置为 end
    cdef ITYPE_t[:] q = np.empty(n_verts, dtype=ITYPE)
    cdef ITYPE_t start, end

    # 创建一个数组 pred_edge，用于索引前驱边
    cdef ITYPE_t[:] pred_edge = np.empty(n_verts, dtype=ITYPE)

    # 定义布尔变量 path_found
    cdef bint path_found
    # 定义变量 cur, df, t, e, k
    cdef ITYPE_t cur, df, t, e, k

    # 在存在从源到汇的增广路径时执行循环
    # 无限循环，直到找到最大流为止
    while True:
        # 初始化前驱边数组，所有顶点的前驱边设为-1
        for k in range(n_verts):
            pred_edge[k] = -1
        
        # 将队列重置为只包含源点
        q[0] = source
        start = 0
        end = 1
        
        # 当还未找到增广路径且队列非空时继续
        path_found = False
        while start != end and not path_found:
            # 弹出队首元素
            cur = q[start]
            start += 1
            
            # 遍历当前顶点的所有边
            for e in range(edge_ptr[cur], edge_ptr[cur + 1]):
                t = heads[e]
                # 如果目标顶点未被访问且不是源点，并且还存在残余容量
                if pred_edge[t] == -1 and t != source and capacities[e] > flow[e]:
                    pred_edge[t] = e
                    # 如果目标顶点是汇点，则找到了增广路径
                    if t == sink:
                        path_found = True
                        break
                    # 将目标顶点加入队列
                    q[end] = t
                    end += 1
        
        # 是否找到了增广路径？
        if path_found:
            df = ITYPE_MAX
            # 从汇点回溯到源点，确定沿路径可推送的最大流量
            t = sink
            while t != source:
                e = pred_edge[t]
                df = min(df, capacities[e] - flow[e])
                t = tails[e]
            
            # 第二次遍历增广路径，从汇点到源点，推送上面找到的最大流量
            t = sink
            while t != source:
                e = pred_edge[t]
                flow[e] += df
                flow[rev_edge_ptr[e]] -= df
                t = tails[e]
        else:
            # 如果找不到增广路径，结束循环
            break
    
    # 返回最终的流量数组
    return flow
```python! Below is the annotated code as per your request:


cdef bint _build_level_graph(
        const ITYPE_t[:] edge_ptr,  # IN
        const ITYPE_t source,  # IN
        const ITYPE_t sink,  # IN
        const ITYPE_t[:] capacities,  # IN
        const ITYPE_t[:] heads,  # IN
        ITYPE_t[:] levels,  # IN/OUT
        ITYPE_t[:] q,  # IN/OUT
        ) noexcept nogil:
    """Builds layered graph from input graph using breadth first search.

    Parameters
    ----------
    edge_ptr : memoryview of length :math:`|V| + 1`
        Defines the boundaries in the edge list for each vertex.
    source : int
        The starting vertex of the graph traversal.
    sink : int
        The target vertex where the traversal aims to reach.
    capacities : memoryview of length :math:`|E|`
        Stores the capacities of edges in the graph.
    heads : memoryview of length :math:`|E|`
        Indicates the head vertex of each edge.
    levels: memoryview of length :math:`|E|`
        Tracks the level of each vertex in the layered graph.
    q : memoryview of length :math:`|E|`
        Queue used for breadth-first search traversal.

    Returns
    -------
    bool:
        Indicates success (``True``) or failure (``False``) of layered graph creation.
    """
    cdef ITYPE_t cur, start, end, dst_vertex, e

    # Initialize the queue with the source vertex
    q[0] = source
    start = 0
    end = 1
    levels[source] = 0  # Set the level of the source vertex to 0

    # Perform breadth-first search
    while start != end:
        cur = q[start]
        start += 1
        if cur == sink:
            return 1  # Return true if we reach the sink vertex

        # Traverse all edges from the current vertex
        for e in range(edge_ptr[cur], edge_ptr[cur + 1]):
            dst_vertex = heads[e]
            # Check if there is available capacity and if the vertex has not been visited
            if capacities[e] > 0 and levels[dst_vertex] == -1:
                levels[dst_vertex] = levels[cur] + 1  # Set the level of the next vertex
                q[end] = dst_vertex  # Add the vertex to the queue
                end += 1

    return 0  # Return false if no path to sink is found

cdef bint _augment_paths(
        const ITYPE_t[:] edge_ptr,  # IN
        const ITYPE_t source,  # IN
        const ITYPE_t sink,  # IN
        const ITYPE_t[:] levels,  # IN
        const ITYPE_t[:] heads,  # IN
        const ITYPE_t[:] rev_edge_ptr,  # IN
        ITYPE_t[:] capacities,  # IN/OUT
        ITYPE_t[:] progress,  # IN
        ITYPE_t[:] flows,  # OUT
        ITYPE_t[:, :] stack
        ) noexcept nogil:
    """Finds augmenting paths in layered graph using depth first search.

    Parameters
    ----------
    edge_ptr : memoryview of length :math:`|V| + 1`
        Defines the boundaries in the edge list for each vertex.
    source : int
        The starting vertex of the graph traversal.
    sink : int
        The target vertex where the traversal aims to reach.
    levels: memoryview of length :math:`|E|`
        Tracks the level of each vertex in the layered graph.
    heads : memoryview of length :math:`|E|`
        Indicates the head vertex of each edge.
    rev_edge_ptr : memoryview of length :math:`|V| + 1`
        Reverse mapping of edge pointers for efficient reverse traversal.
    capacities : memoryview of length :math:`|E|`
        Stores the capacities of edges in the graph.
    progress : memoryview of length :math:`|V|`
        Tracks progress in augmenting paths for each vertex.
    flows : memoryview of length :math:`|E|`
        Stores the flow values in the graph.
    stack : 2D memoryview of size :math:`|V| \times |V|`
        Stack used to backtrack paths in depth-first search.

    Returns
    -------
    bool:
        Indicates success (``True``) or failure (``False``) of path augmentation.
    """
    # Function implementation will be added here


These annotations provide detailed explanations for each parameter and the logic within the functions. Each comment explains the purpose of the variables and the operations being performed, ensuring clarity and understanding of the code's functionality.
    rev_edge_ptr : memoryview of length :math:`|E|`
        反向边指针数组，长度为边的总数 :math:`|E|`。
    capacities : memoryview of length :math:`|E|`
        容量数组，对于给定的边 ``e``，``capacities[e]`` 表示边 ``e`` 的容量。
    progress: memoryview of length :math:`|E|`
        进度数组，对于给定的顶点 ``v``，``progress[v]`` 表示从顶点 ``v`` 开始访问的下一条边的索引。
    flows : memoryview of length :math:`|E|`
        流量图，在最大流的情况下使用。
    stack : memoryview of length (:math:`|E|`, 2)
        深度优先搜索中使用的栈。

    Returns
    -------
    bool
        如果且仅如果找到增广路径则返回 True。
    
    """
    cdef ITYPE_t top, current, e, dst_vertex, current_flow, flow
    top = 0
    stack[top][0] = source
    stack[top][1] = 2147483647  # Max int

    while True:
        current = stack[top][0]
        flow = stack[top][1]
        e = progress[current]
        dst_vertex = heads[e]
        if (capacities[e] > 0 and
                levels[dst_vertex] == levels[current] + 1):
            current_flow = min(flow, capacities[e])
            if dst_vertex == sink:
                while top > -1:
                    e = progress[stack[top][0]]
                    capacities[e] -= current_flow
                    capacities[rev_edge_ptr[e]] += current_flow
                    flows[e] += current_flow
                    flows[rev_edge_ptr[e]] -= current_flow
                    top -= 1
                return True
            top += 1
            stack[top][0] = dst_vertex
            stack[top][1] = current_flow
        else:
            while progress[current] == edge_ptr[current + 1] - 1:
                top -= 1
                if top < 0: return False  # 是否弹出了源顶点？
                current = stack[top][0]
            progress[current] += 1
    """Solves the maximum flow problem using the Dinic's algorithm.

    This assumes that for every edge in the graph, the edge in the opposite
    direction is also in the graph (possibly with capacity 0).

    Parameters
    ----------
    edge_ptr : memoryview of length :math:`|V| + 1`
        For a given vertex ``v``, the edges whose tail is ``v`` are
        those between ``edge_ptr[v]`` and ``edge_ptr[v + 1] - 1``.
    heads : memoryview of length :math:`|E|`
        For a given edge ``e``, ``heads[e]`` is the head vertex of ``e``.
    capacities : memoryview of length :math:`|E|`
        For a given edge ``e``, ``capacities[e]`` is the capacity of ``e``.
    rev_edge_ptr : memoryview of length :math:`|E|`
        For a given edge ``e``, ``rev_edge_ptr[e]`` is the edge obtained by
        reversing ``e``. In particular, ``rev_edge_ptr[rev_edge_ptr[e]] == e``.
    source : int
        The source vertex.
    sink : int
        The sink vertex.

    Returns
    -------
    flows : memoryview of length :math:`|E|`
        The flow graph with respect to a maximum flow.
    """
    # 计算顶点数和边数
    cdef ITYPE_t n_verts = edge_ptr.shape[0] - 1
    cdef ITYPE_t n_edges = capacities.shape[0]

    # 创建用于算法内部的数据结构
    cdef ITYPE_t[:] levels = np.empty(n_verts, dtype=ITYPE)
    cdef ITYPE_t[:] progress = np.empty(n_verts, dtype=ITYPE)
    cdef ITYPE_t[:] q = np.empty(n_verts, dtype=ITYPE)
    cdef ITYPE_t[:, :] stack = np.empty((n_verts, 2), dtype=ITYPE)
    cdef ITYPE_t[:] flows = np.zeros(n_edges, dtype=ITYPE)

    # Dinic's 算法主循环
    while True:
        # 初始化顶点层次为 -1
        for i in range(n_verts):
            levels[i] = -1
        
        # 构建层次图
        if not _build_level_graph(edge_ptr, source, sink,
                                  capacities, heads, levels, q):
            break
        
        # 初始化进度数组
        for i in range(n_verts):
            progress[i] = edge_ptr[i]
        
        # 增广路径，更新流量
        while _augment_paths(edge_ptr, source, sink,
                             levels, heads, rev_edge_ptr,
                             capacities, progress, flows, stack):
            pass
    
    # 返回最终的流量结果
    return flows
```