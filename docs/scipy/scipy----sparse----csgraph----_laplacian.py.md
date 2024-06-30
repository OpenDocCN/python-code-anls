# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\_laplacian.py`

```
# 导入所需的库
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy, is_pydata_spmatrix

###############################################################################
# Graph laplacian
def laplacian(
    csgraph,
    normed=False,
    return_diag=False,
    use_out_degree=False,
    *,
    copy=True,
    form="array",
    dtype=None,
    symmetrized=False,
):
    """
    Return the Laplacian of a directed graph.

    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        compressed-sparse graph, with shape (N, N).
    normed : bool, optional
        If True, then compute symmetrically normalized Laplacian.
        Default: False.
    return_diag : bool, optional
        If True, then also return an array related to vertex degrees.
        Default: False.
    use_out_degree : bool, optional
        If True, then use out-degree instead of in-degree.
        This distinction matters only if the graph is asymmetric.
        Default: False.
    copy: bool, optional
        If False, then change `csgraph` in place if possible,
        avoiding doubling the memory use.
        Default: True, for backward compatibility.
    form: 'array', or 'function', or 'lo'
        Determines the format of the output Laplacian:

        * 'array' is a numpy array;
        * 'function' is a pointer to evaluating the Laplacian-vector
          or Laplacian-matrix product;
        * 'lo' results in the format of the `LinearOperator`.

        Choosing 'function' or 'lo' always avoids doubling
        the memory use, ignoring `copy` value.
        Default: 'array', for backward compatibility.
    dtype: None or one of numeric numpy dtypes, optional
        The dtype of the output. If ``dtype=None``, the dtype of the
        output matches the dtype of the input csgraph, except for
        the case ``normed=True`` and integer-like csgraph, where
        the output dtype is 'float' allowing accurate normalization,
        but dramatically increasing the memory use.
        Default: None, for backward compatibility.
    symmetrized: bool, optional
        If True, then the output Laplacian is symmetric/Hermitian.
        The symmetrization is done by ``csgraph + csgraph.T.conj``
        without dividing by 2 to preserve integer dtypes if possible
        prior to the construction of the Laplacian.
        The symmetrization will increase the memory footprint of
        sparse matrices unless the sparsity pattern is symmetric or
        `form` is 'function' or 'lo'.
        Default: False, for backward compatibility.

    Returns
    -------
    """
    # lap 是表示 Laplacian 矩阵的变量，可以是 ndarray、稀疏矩阵或 LinearOperator 类型。
    #     如果输入是稠密的，则它将是一个 NumPy 数组（稠密），否则将是一个稀疏矩阵，或者
    #     如果 form 参数是 'function' 或 'lo'，则 lap 可能是函数或 LinearOperator 对象。
    lap : ndarray, or sparse matrix, or `LinearOperator`
        The N x N Laplacian of csgraph. It will be a NumPy array (dense)
        if the input was dense, or a sparse matrix otherwise, or
        the format of a function or `LinearOperator` if
        `form` equals 'function' or 'lo', respectively.
    
    # diag 是 Laplacian 矩阵的主对角线数组，长度为 N。
    #     对于标准化的 Laplacian，这是顶点度数的平方根数组，如果度数为零则为 1。
    diag : ndarray, optional
        The length-N main diagonal of the Laplacian matrix.
        For the normalized Laplacian, this is the array of square roots
        of vertex degrees or 1 if the degree is zero.
    
    # Laplacian 矩阵是一个图的拉普拉斯矩阵，有时也称为 "Kirchhoff 矩阵" 或简称为 "拉普拉斯矩阵"，
    # 在谱图理论的许多领域中非常有用。
    Notes
    -----
    The Laplacian matrix of a graph is sometimes referred to as the
    "Kirchhoff matrix" or just the "Laplacian", and is useful in many
    parts of spectral graph theory.
    
    # 特别地，拉普拉斯矩阵的特征分解可以提供关于图的许多性质的洞察，
    #     例如，在谱数据嵌入和聚类中广泛使用。
    In particular, the eigen-decomposition of the Laplacian can give
    insight into many properties of the graph, e.g.,
    is commonly used for spectral data embedding and clustering.
    
    # 如果 copy=True 和 form="array"（默认），构造的拉普拉斯矩阵会使用两倍内存。
    #     选择 copy=False 除非 form="array" 或者矩阵在 coo 格式下是稀疏的，或者
    #     使用 normed=True 且输入是整数，强制浮点输出，否则不起作用。
    The constructed Laplacian doubles the memory use if ``copy=True`` and
    ``form="array"`` which is the default.
    Choosing ``copy=False`` has no effect unless ``form="array"``
    or the matrix is sparse in the ``coo`` format, or dense array, except
    for the integer input with ``normed=True`` that forces the float output.
    
    # 如果 form="array"，稀疏输入将重新格式化为 coo 格式，这是默认的行为。
    Sparse input is reformatted into ``coo`` if ``form="array"``,
    which is the default.
    
    # 如果输入的邻接矩阵不对称，那么拉普拉斯矩阵也是非对称的，除非使用 symmetrized=True。
    If the input adjacency matrix is not symmetric, the Laplacian is
    also non-symmetric unless ``symmetrized=True`` is used.
    
    # 用于归一化的输入邻接矩阵的对角元素被忽略，并且用零替换，用于归一化时的 normed=True。
    #     归一化使用输入邻接矩阵的行和的倒数平方根，因此如果行和包含负数或带有非零虚部的复数值，
    #     归一化可能会失败。
    Diagonal entries of the input adjacency matrix are ignored and
    replaced with zeros for the purpose of normalization where ``normed=True``.
    The normalization uses the inverse square roots of row-sums of the input
    adjacency matrix, and thus may fail if the row-sums contain
    negative or complex with a non-zero imaginary part values.
    
    # 如果输入的 csgraph 是对称的，则归一化也是对称的，使得标准化的拉普拉斯矩阵也是对称的。
    The normalization is symmetric, making the normalized Laplacian also
    symmetric if the input csgraph was symmetric.
    
    References
    ----------
    .. [1] Laplacian matrix. https://en.wikipedia.org/wiki/Laplacian_matrix
    
    # 示例部分展示了如何使用 csgraph.laplacian 函数生成不同类型图的拉普拉斯矩阵。
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csgraph
    
    Our first illustration is the symmetric graph
    
    >>> G = np.arange(4) * np.arange(4)[:, np.newaxis]
    >>> G
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]])
    
    and its symmetric Laplacian matrix
    
    >>> csgraph.laplacian(G)
    array([[ 0,  0,  0,  0],
           [ 0,  5, -2, -3],
           [ 0, -2,  8, -6],
           [ 0, -3, -6,  9]])
    
    The non-symmetric graph
    
    >>> G = np.arange(9).reshape(3, 3)
    >>> G
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    
    has different row- and column sums, resulting in two varieties
    of the Laplacian matrix, using an in-degree, which is the default
    
    >>> L_in_degree = csgraph.laplacian(G)
    >>> L_in_degree
    array([[ 9, -1, -2],
           [-3,  8, -5],
           [-6, -7,  7]])
    or alternatively an out-degree

    >>> L_out_degree = csgraph.laplacian(G, use_out_degree=True)
    计算使用出度的拉普拉斯矩阵

    >>> L_out_degree
    array([[ 3, -1, -2],
           [-3,  8, -5],
           [-6, -7, 13]])

    Constructing a symmetric Laplacian matrix, one can add the two as

    >>> L_in_degree + L_out_degree.T
    构造对称拉普拉斯矩阵，通过将两个矩阵相加实现

    array([[ 12,  -4,  -8],
            [ -4,  16, -12],
            [ -8, -12,  20]])

    or use the ``symmetrized=True`` option

    >>> csgraph.laplacian(G, symmetrized=True)
    使用 ``symmetrized=True`` 选项生成对称拉普拉斯矩阵

    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    that is equivalent to symmetrizing the original graph

    >>> csgraph.laplacian(G + G.T)
    等同于对原始图进行对称化处理后生成拉普拉斯矩阵

    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    The goal of normalization is to make the non-zero diagonal entries
    of the Laplacian matrix to be all unit, also scaling off-diagonal
    entries correspondingly. The normalization can be done manually, e.g.,

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    创建示例图 G

    >>> L, d = csgraph.laplacian(G, return_diag=True)
    计算图 G 的拉普拉斯矩阵 L 和度数向量 d

    >>> L
    array([[ 2, -1, -1],
           [-1,  2, -1],
           [-1, -1,  2]])
    >>> d
    array([2, 2, 2])
    >>> scaling = np.sqrt(d)
    计算缩放系数

    >>> scaling
    array([1.41421356, 1.41421356, 1.41421356])
    >>> (1/scaling)*L*(1/scaling)
    手动进行归一化处理

    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    Or using ``normed=True`` option

    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    使用 ``normed=True`` 选项进行归一化处理

    >>> L
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    which now instead of the diagonal returns the scaling coefficients

    >>> d
    返回缩放系数而非对角元素

    array([1.41421356, 1.41421356, 1.41421356])

    Zero scaling coefficients are substituted with 1s, where scaling
    has thus no effect, e.g.,

    >>> G = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    创建示例图 G，其中存在零缩放系数

    >>> G
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    计算图 G 的归一化拉普拉斯矩阵 L 和度数向量 d

    >>> L
    array([[ 0., -0., -0.],
           [-0.,  1., -1.],
           [-0., -1.,  1.]])
    >>> d
    array([1., 1., 1.])

    Only the symmetric normalization is implemented, resulting
    in a symmetric Laplacian matrix if and only if its graph is symmetric
    and has all non-negative degrees, like in the examples above.

    对称归一化只有一种实现方式，只有当图是对称且所有度数非负时，才会得到对称拉普拉斯矩阵，就像上述示例一样。

    The output Laplacian matrix is by default a dense array or a sparse matrix
    inferring its shape, format, and dtype from the input graph matrix:

    默认情况下，输出的拉普拉斯矩阵是一个密集数组或者稀疏矩阵，其形状、格式和数据类型均从输入图矩阵推断得出：

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype(np.float32)
    创建一个浮点型的示例图 G

    >>> G
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]], dtype=float32)
    >>> csgraph.laplacian(G)
    计算图 G 的拉普拉斯矩阵

    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]], dtype=float32)

    but can alternatively be generated matrix-free as a LinearOperator:

    也可以以矩阵自由形式作为线性操作符生成：

    >>> L = csgraph.laplacian(G, form="lo")
    >>> L
    # 1. 为稀疏图构造 Laplacian 矩阵以进行谱聚类和嵌入
    >>> from scipy.sparse import diags, random
    使用 scipy.sparse 中的 diags 和 random 函数导入必要的库

    # 2. 创建一个包含 35 个顶点的有向线性图，使用稀疏邻接矩阵 G
    >>> N = 35
    设定图的顶点数量 N 为 35
    >>> G = diags(np.ones(N-1), 1, format="csr")
    使用 diags 函数创建一个带有格式 "csr" 的稀疏邻接矩阵 G

    # 3. 设定一个随机种子 rng，并向图 G 添加随机稀疏噪声
    >>> rng = np.random.default_rng()
    设定一个随机种子 rng
    >>> G += 1e-2 * random(N, N, density=0.1, random_state=rng)
    将随机稀疏噪声添加到图 G 中，密度为 0.1，基于随机种子 rng

    # 4. 设定初始近似特征向量 X
    >>> X = rng.random((N, 2))
    使用 rng 生成一个大小为 (N, 2) 的随机数组作为初始近似特征向量 X

    # 5. 设定一个全 1 向量 Y 作为非归一化 Laplacian 的平凡特征向量
    >>> Y = np.ones((N, 1))
    创建一个大小为 (N, 1) 的全 1 向量 Y 作为非归一化 Laplacian 的平凡特征向量

    # 6. 在一个循环中交替地处理图的权重符号，以确定谱最大和最小割的标签
    >>> for cut in ["max", "min"]:
    对于每种割类型（"max" 或 "min"）

    # 7. 反转图的权重
    ...     G = -G  # 1.
    反转图 G 的权重（对图进行符号变换）

    # 8. 构造 Laplacian 矩阵 L，使用对称化选项和无矩阵形式
    ...     L = csgraph.laplacian(G, symmetrized=True, form="lo")  # 2.
    构造 Laplacian 矩阵 L，采用对称化选项并使用无矩阵形式 "lo"

    # 9. 调用 lobpcg 特征值求解器计算 Fiedler 向量和对应的标签
    ...     _, eves = lobpcg(L, X, Y=Y, largest=False, tol=1e-2)  # 3.
    使用 lobpcg 函数计算 Fiedler 向量，并确定对应的标签，其中 largest=False

    # 10. 根据 Fiedler 向量的符号修正所有向量
    ...     eves *= np.sign(eves[0, 0])  # 4.
    根据 Fiedler 向量的第一个分量的符号修正所有特征向量

    # 11. 打印出基于 Fiedler 向量的谱割标签
    ...     print(cut + "-cut labels:\\n", 1 * (eves[:, 0]>0))  # 5.
    打印输出基于 Fiedler 向量的谱割标签，1 * (eves[:, 0]>0) 表示大于零的部分作为标签

    # 12. 展示了一个（稍有噪声的）线性图的结果，最大割将图的所有边剥离，并将奇数顶点和偶数顶点着色到不同颜色
    As anticipated for a (slightly noisy) linear graph,
    the max-cut strips all the edges of the graph coloring all
    odd vertices into one color and all even vertices into another one,
    作为稍有噪声的线性图的预期结果，最大割将图的所有边分离，并将奇数顶点和偶数顶点分别着色

    # 13. 而平衡的最小割通过删除单条边将图分成两半
    while the balanced min-cut partitions the graph
    in the middle by deleting a single edge.
    而平衡的最小割通过删除单条边将图分割成两半

    # 14. 两种确定的分割均为最优解
    Both determined partitions are optimal.
    两种确定的分割均为最优解
    # 检查 csgraph 是否是 pydata 稀疏矩阵
    is_pydata_sparse = is_pydata_spmatrix(csgraph)
    
    # 如果 csgraph 是 pydata 稀疏矩阵，则记录其类别
    if is_pydata_sparse:
        pydata_sparse_cls = csgraph.__class__
        # 将 pydata 稀疏矩阵转换为 scipy 稀疏矩阵
        csgraph = convert_pydata_sparse_to_scipy(csgraph)
    
    # 检查 csgraph 的维度是否为 2，并且行数与列数相同（必须是方阵）
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        # 抛出值错误异常，要求 csgraph 必须是方阵
        raise ValueError('csgraph must be a square matrix or array')
    
    # 如果需要进行归一化，并且 csgraph 的数据类型是有符号整数或无符号整数，则转换数据类型为 np.float64
    if normed and (
        np.issubdtype(csgraph.dtype, np.signedinteger)
        or np.issubdtype(csgraph.dtype, np.uint)
    ):
        csgraph = csgraph.astype(np.float64)
    
    # 根据 form 参数选择创建拉普拉斯矩阵的函数
    if form == "array":
        # 如果 csgraph 是稀疏矩阵，则选择稀疏矩阵版本的拉普拉斯矩阵创建函数，否则选择密集矩阵版本的
        create_lap = (
            _laplacian_sparse if issparse(csgraph) else _laplacian_dense
        )
    else:
        # 如果 csgraph 是稀疏矩阵，则选择稀疏矩阵版本的浮点数拉普拉斯矩阵创建函数，否则选择密集矩阵版本的浮点数版本
        create_lap = (
            _laplacian_sparse_flo
            if issparse(csgraph)
            else _laplacian_dense_flo
        )
    
    # 根据 use_out_degree 参数确定拉普拉斯矩阵计算中的轴
    degree_axis = 1 if use_out_degree else 0
    
    # 调用 create_lap 函数创建拉普拉斯矩阵 lap 和度向量 d
    lap, d = create_lap(
        csgraph,
        normed=normed,
        axis=degree_axis,
        copy=copy,
        form=form,
        dtype=dtype,
        symmetrized=symmetrized,
    )
    
    # 如果 csgraph 最初是 pydata 稀疏矩阵，则将 lap 转换回 pydata 稀疏矩阵的格式
    if is_pydata_sparse:
        lap = pydata_sparse_cls.from_scipy_sparse(lap)
    
    # 如果 return_diag 设置为 True，则返回 lap 和 d；否则只返回 lap
    if return_diag:
        return lap, d
    return lap
# 设置步长为对角线长度加一
def _setdiag_dense(m, d):
    step = len(d) + 1
    # 将 m 的对角线元素设置为数组 d 的值
    m.flat[::step] = d

# 创建并返回一个函数，该函数对向量 v 执行 Laplace 操作
def _laplace(m, d):
    return lambda v: v * d[:, np.newaxis] - m @ v

# 创建并返回一个函数，该函数对向量 v 执行归一化的 Laplace 操作
def _laplace_normed(m, d, nd):
    laplace = _laplace(m, d)
    return lambda v: nd[:, np.newaxis] * laplace(v * nd[:, np.newaxis])

# 创建并返回一个函数，该函数对向量 v 执行对称的 Laplace 操作
def _laplace_sym(m, d):
    return (
        lambda v: v * d[:, np.newaxis]
        - m @ v
        - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m))
    )

# 创建并返回一个函数，该函数对向量 v 执行对称归一化的 Laplace 操作
def _laplace_normed_sym(m, d, nd):
    laplace_sym = _laplace_sym(m, d)
    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])

# 创建并返回一个 LinearOperator 对象，用于表示稀疏矩阵的线性操作
def _linearoperator(mv, shape, dtype):
    return LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)

# 计算图形的 Laplacian 矩阵，根据不同参数进行处理
def _laplacian_sparse_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    # 关键字参数 `copy` 在这里未被使用，没有任何效果
    del copy

    # 如果未指定数据类型，则使用图形的数据类型
    if dtype is None:
        dtype = graph.dtype

    # 计算图的行或列的和，返回一个扁平化的数组
    graph_sum = np.asarray(graph.sum(axis=axis)).ravel()
    # 获取图的对角线元素
    graph_diagonal = graph.diagonal()
    # 计算图的度数减去对角线元素，得到 Laplacian 的非对角线部分
    diag = graph_sum - graph_diagonal

    # 如果需要对称化处理
    if symmetrized:
        # 将图的转置加到自身，实现对称化
        graph_sum += np.asarray(graph.sum(axis=1 - axis)).ravel()
        diag = graph_sum - graph_diagonal - graph_diagonal

    # 如果进行归一化处理
    if normed:
        # 找出孤立节点的掩码
        isolated_node_mask = diag == 0
        # 计算权重 w，如果进行对称化则使用对称归一化的 Laplacian 函数
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(graph, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(graph, graph_sum, 1.0 / w)
        # 根据所需的返回格式返回结果
        if form == "function":
            return md, w.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, w.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")
    else:
        # 如果不进行归一化处理，则根据对称化选项返回相应的 Laplacian 函数
        if symmetrized:
            md = _laplace_sym(graph, graph_sum)
        else:
            md = _laplace(graph, graph_sum)
        # 根据所需的返回格式返回结果
        if form == "function":
            return md, diag.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, diag.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")

# 计算图的 Laplacian 矩阵，根据不同参数进行处理
def _laplacian_sparse(graph, normed, axis, copy, form, dtype, symmetrized):
    # 关键字参数 `form` 在这里未被使用，没有任何效果
    del form

    # 如果未指定数据类型，则使用图形的数据类型
    if dtype is None:
        dtype = graph.dtype

    # 检查是否需要复制图形
    needs_copy = False
    if graph.format in ('lil', 'dok'):
        m = graph.tocoo()
    else:
        m = graph
        if copy:
            needs_copy = True

    # 如果进行对称化处理，则将图形与其转置的共轭相加
    if symmetrized:
        m += m.T.conj()

    # 计算图的行或列的和，返回一个扁平化的数组，并减去对角线元素得到 w
    w = np.asarray(m.sum(axis=axis)).ravel() - m.diagonal()

    # 如果进行归一化处理
    if normed:
        # 将图转换为 COO 格式，如果需要复制则复制原始图
        m = m.tocoo(copy=needs_copy)
        # 找出孤立节点的掩码
        isolated_node_mask = (w == 0)
        # 计算权重 w
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        # 对 m 进行归一化处理
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        # 设置对角线元素为 1 减去孤立节点的掩码
        m.setdiag(1 - isolated_node_mask)
    # 如果条件不满足，则执行以下代码块
    else:
        # 如果矩阵 m 的格式为 'dia'，则创建其副本 m
        if m.format == 'dia':
            m = m.copy()
        # 如果矩阵 m 的格式不为 'dia'，则将其转换为 COO 格式（如果需要，进行复制操作）
        else:
            m = m.tocoo(copy=needs_copy)
        # 将矩阵 m 中的数据元素乘以 -1
        m.data *= -1
        # 将矩阵 m 的对角线设置为 w
        m.setdiag(w)

    # 返回类型转换后的矩阵 m 和类型转换后的 w，确保不进行复制操作
    return m.astype(dtype, copy=False), w.astype(dtype)
# 计算稠密图的拉普拉斯矩阵，返回结果和权重向量
def _laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized):
    # 如果形式不是数组，则引发值错误异常
    if form != "array":
        raise ValueError(f'{form!r} must be "array"')

    # 如果未指定数据类型，则使用图的数据类型
    if dtype is None:
        dtype = graph.dtype

    # 如果需要复制图形数据，创建图的副本
    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    # 如果未指定数据类型，则使用副本的数据类型
    if dtype is None:
        dtype = m.dtype

    # 如果对称化标志为真，将图形与其共轭转置相加
    if symmetrized:
        m += m.T.conj()

    # 将对角线元素设为零
    np.fill_diagonal(m, 0)

    # 计算图在指定轴上的和
    w = m.sum(axis=axis)

    # 如果进行归一化处理
    if normed:
        # 找出孤立节点的位置
        isolated_node_mask = (w == 0)
        # 对权重进行处理，将孤立节点的权重设为1，其它节点设为sqrt(w)
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        # 归一化矩阵
        m /= w
        m /= w[:, np.newaxis]
        # 将矩阵乘以-1，并设置对角线元素
        m *= -1
        _setdiag_dense(m, 1 - isolated_node_mask)
    else:
        # 将矩阵乘以-1，并设置对角线元素
        m *= -1
        _setdiag_dense(m, w)

    # 返回结果矩阵和权重向量，转换为指定的数据类型
    return m.astype(dtype, copy=False), w.astype(dtype, copy=False)
```