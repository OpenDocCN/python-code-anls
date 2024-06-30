# `D:\src\scipysrc\scikit-learn\sklearn\externals\_scipy\sparse\csgraph\_laplacian.py`

```
"""
This file is a copy of the scipy.sparse.csgraph._laplacian module from SciPy 1.12

scipy.sparse.csgraph.laplacian supports sparse arrays only starting from Scipy 1.12,
see https://github.com/scipy/scipy/pull/19156. This vendored file can be removed as
soon as Scipy 1.12 becomes the minimum supported version.

Laplacian of a compressed-sparse graph
"""

# SPDX-License-Identifier: BSD-3-Clause

import numpy as np  # 导入NumPy库，用于数值计算
from scipy.sparse import issparse  # 导入issparse函数，判断对象是否为稀疏矩阵
from scipy.sparse.linalg import LinearOperator  # 导入LinearOperator类，用于定义线性操作符


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
        Compressed-sparse graph, with shape (N, N).
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
    copy : bool, optional
        If False, then change `csgraph` in place if possible,
        avoiding doubling the memory use.
        Default: True, for backward compatibility.
    form : 'array', or 'function', or 'lo'
        Determines the format of the output Laplacian:

        * 'array' is a numpy array;
        * 'function' is a pointer to evaluating the Laplacian-vector
          or Laplacian-matrix product;
        * 'lo' results in the format of the `LinearOperator`.

        Choosing 'function' or 'lo' always avoids doubling
        the memory use, ignoring `copy` value.
        Default: 'array', for backward compatibility.
    dtype : None or one of numeric numpy dtypes, optional
        The dtype of the output. If ``dtype=None``, the dtype of the
        output matches the dtype of the input csgraph, except for
        the case ``normed=True`` and integer-like csgraph, where
        the output dtype is 'float' allowing accurate normalization,
        but dramatically increasing the memory use.
        Default: None, for backward compatibility.
    symmetrized : bool, optional
        If True, then the output Laplacian is symmetric/Hermitian.
        The symmetrization is done by ``csgraph + csgraph.T.conj``
        without dividing by 2 to preserve integer dtypes if possible
        prior to the construction of the Laplacian.
        The symmetrization will increase the memory footprint of
        sparse matrices unless the sparsity pattern is symmetric or
        `form` is 'function' or 'lo'.
        Default: False, for backward compatibility.
    """
    # 在给定的图(csgraph)上计算拉普拉斯矩阵

    # 检查csgraph是否为稀疏矩阵，如果不是则转换为稀疏矩阵
    if not issparse(csgraph):
        csgraph = np.asarray(csgraph)

    # 从scipy.sparse.linalg导入LinearOperator类后，返回相关的Laplacian操作
    if form == "lo":
        return LinearOperator(dtype=dtype, shape=csgraph.shape, matvec=lambda x: csgraph @ x)

    # 如果需要对输入的csgraph进行复制，则复制一份csgraph的副本
    if copy:
        csgraph = csgraph.copy()

    # 根据指定的use_out_degree参数选择使用出度或入度来计算拉普拉斯矩阵
    if use_out_degree:
        csgraph = csgraph.T

    # 根据symmetrized参数决定是否对csgraph进行对称化处理
    if symmetrized:
        csgraph = csgraph + csgraph.T.conj()

    # 根据normed参数计算对称归一化的拉普拉斯矩阵
    if normed:
        degrees = csgraph.sum(axis=1)
        degrees = np.where(degrees != 0, 1 / np.sqrt(degrees), 0)
        csgraph = csgraph * degrees
        csgraph = csgraph.T * degrees

    # 如果return_diag为True，则同时返回与顶点度数相关的数组
    if return_diag:
        if normed:
            return csgraph, degrees
        else:
            return csgraph, csgraph.sum(axis=1)

    # 返回计算得到的拉普拉斯矩阵
    return csgraph - np.diag(csgraph.sum(axis=1))
    # 返回值 lap 表示输入图 csgraph 的 Laplacian 矩阵，可能是 ndarray（密集矩阵）、稀疏矩阵或 LinearOperator（线性操作器）
    # 如果输入是密集的，则返回 NumPy 数组；如果是稀疏的，则返回稀疏矩阵；如果 form 参数为 'function' 或 'lo'，则返回相应的函数或线性操作器
    lap : ndarray, or sparse matrix, or `LinearOperator`
        The N x N Laplacian of csgraph. It will be a NumPy array (dense)
        if the input was dense, or a sparse matrix otherwise, or
        the format of a function or `LinearOperator` if
        `form` equals 'function' or 'lo', respectively.

    # 如果 normed=True，对于归一化的 Laplacian，这里的 diag 是 Laplacian 矩阵的主对角线数组
    # 对于归一化 Laplacian，这是顶点度数的平方根数组，如果度数为零，则为 1
    diag : ndarray, optional
        The length-N main diagonal of the Laplacian matrix.
        For the normalized Laplacian, this is the array of square roots
        of vertex degrees or 1 if the degree is zero.

    # Laplacian 矩阵是图的拉普拉斯矩阵，有时被称为“基尔霍夫矩阵”或简称“拉普拉斯矩阵”，在谱图理论的许多部分都很有用
    Notes
    -----
    The Laplacian matrix of a graph is sometimes referred to as the
    "Kirchhoff matrix" or just the "Laplacian", and is useful in many
    parts of spectral graph theory.

    # Laplacian 的特征分解可以揭示图的许多性质，例如谱数据嵌入和聚类
    In particular, the eigen-decomposition of the Laplacian can give
    insight into many properties of the graph, e.g.,
    is commonly used for spectral data embedding and clustering.

    # 如果 copy=True 和 form="array"（默认），则构建的 Laplacian 会使内存使用量翻倍
    # 选择 copy=False 对 form="array" 或 COO 格式的稀疏矩阵没有影响，除非输入是整数且 normed=True 强制输出为浮点数
    The constructed Laplacian doubles the memory use if ``copy=True`` and
    ``form="array"`` which is the default.
    Choosing ``copy=False`` has no effect unless ``form="array"``
    or the matrix is sparse in the ``coo`` format, or dense array, except
    for the integer input with ``normed=True`` that forces the float output.

    # 如果 form="array"（默认），则稀疏输入会被重格式化为 COO 格式
    Sparse input is reformatted into ``coo`` if ``form="array"``,
    which is the default.

    # 如果输入邻接矩阵不对称，则构建的 Laplacian 也不对称，除非使用 symmetrized=True
    If the input adjacency matrix is not symmetric, the Laplacian is
    also non-symmetric unless ``symmetrized=True`` is used.

    # 用于归一化的输入邻接矩阵的对角线条目会被忽略，并用零替换，这是在 normed=True 时的目的
    Diagonal entries of the input adjacency matrix are ignored and
    replaced with zeros for the purpose of normalization where ``normed=True``.

    # 归一化使用输入邻接矩阵的行和的倒数平方根，如果行和包含负数或具有非零虚部的复数值，则可能失败
    The normalization uses the inverse square roots of row-sums of the input
    adjacency matrix, and thus may fail if the row-sums contain
    negative or complex with a non-zero imaginary part values.

    # 如果输入的 csgraph 对称，那么归一化也是对称的，使得归一化 Laplacian 也是对称的
    The normalization is symmetric, making the normalized Laplacian also
    symmetric if the input csgraph was symmetric.

    # 参考文献
    References
    ----------
    .. [1] Laplacian matrix. https://en.wikipedia.org/wiki/Laplacian_matrix

    # 示例
    Examples
    --------
    # 导入必要的库
    >>> import numpy as np
    >>> from scipy.sparse import csgraph

    # 我们首先展示一个对称图例子
    Our first illustration is the symmetric graph

    # 创建一个对称图 G
    >>> G = np.arange(4) * np.arange(4)[:, np.newaxis]
    >>> G
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]])

    # 计算其对称 Laplacian 矩阵
    >>> csgraph.laplacian(G)
    array([[ 0,  0,  0,  0],
           [ 0,  5, -2, -3],
           [ 0, -2,  8, -6],
           [ 0, -3, -6,  9]])

    # 创建一个非对称图例子
    The non-symmetric graph

    # 创建一个非对称图 G
    >>> G = np.arange(9).reshape(3, 3)
    >>> G
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    # 由于行和列的和不同，使用入度创建两种 Laplacian 矩阵的变体，这是默认行为
    # 计算入度 Laplacian 矩阵
    >>> L_in_degree = csgraph.laplacian(G)
    >>> L_in_degree
    array([[ 9, -1, -2],
           [-3,  8, -5],
           [-6, -7,  7]])
    or alternatively an out-degree

    >>> L_out_degree = csgraph.laplacian(G, use_out_degree=True)
    # 计算出度为节点度的拉普拉斯矩阵，将其存储在变量 L_out_degree 中
    >>> L_out_degree
    array([[ 3, -1, -2],
           [-3,  8, -5],
           [-6, -7, 13]])

    Constructing a symmetric Laplacian matrix, one can add the two as

    >>> L_in_degree + L_out_degree.T
    # 将入度拉普拉斯矩阵和转置后的出度拉普拉斯矩阵相加，构建对称拉普拉斯矩阵
    array([[ 12,  -4,  -8],
            [ -4,  16, -12],
            [ -8, -12,  20]])

    or use the ``symmetrized=True`` option

    >>> csgraph.laplacian(G, symmetrized=True)
    # 使用 symmetrized=True 选项构建对称拉普拉斯矩阵
    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    that is equivalent to symmetrizing the original graph

    >>> csgraph.laplacian(G + G.T)
    # 将原图及其转置相加，然后计算其拉普拉斯矩阵，得到与 symmetrized=True 相等的结果
    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    The goal of normalization is to make the non-zero diagonal entries
    of the Laplacian matrix to be all unit, also scaling off-diagonal
    entries correspondingly. The normalization can be done manually, e.g.,

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True)
    # 计算图 G 的拉普拉斯矩阵和对角线元素，存储在变量 L 和 d 中
    >>> L
    array([[ 2, -1, -1],
           [-1,  2, -1],
           [-1, -1,  2]])
    >>> d
    array([2, 2, 2])
    >>> scaling = np.sqrt(d)
    # 计算对角元素的平方根作为缩放系数
    >>> scaling
    array([1.41421356, 1.41421356, 1.41421356])
    >>> (1/scaling)*L*(1/scaling)
    # 根据缩放系数对拉普拉斯矩阵进行归一化处理
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    Or using ``normed=True`` option

    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    # 使用 normed=True 选项进行归一化，返回缩放系数而非对角元素
    >>> L
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    which now instead of the diagonal returns the scaling coefficients

    >>> d
    array([1.41421356, 1.41421356, 1.41421356])

    Zero scaling coefficients are substituted with 1s, where scaling
    has thus no effect, e.g.,

    >>> G = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    >>> G
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    # 处理缩放系数为零的情况，用1替换，不对拉普拉斯矩阵进行缩放
    >>> L
    array([[ 0., -0., -0.],
           [-0.,  1., -1.],
           [-0., -1.,  1.]])
    >>> d
    array([1., 1., 1.])

    Only the symmetric normalization is implemented, resulting
    in a symmetric Laplacian matrix if and only if its graph is symmetric
    and has all non-negative degrees, like in the examples above.

    The output Laplacian matrix is by default a dense array or a sparse matrix
    inferring its shape, format, and dtype from the input graph matrix:

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype(np.float32)
    >>> G
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]], dtype=float32)
    >>> csgraph.laplacian(G)
    # 默认情况下，生成稠密数组或稀疏矩阵作为输出拉普拉斯矩阵，其形状、格式和数据类型由输入图形矩阵推断得出
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]], dtype=float32)

    but can alternatively be generated matrix-free as a LinearOperator:

    >>> L = csgraph.laplacian(G, form="lo")
    # 也可以选择以 LinearOperator 形式生成拉普拉斯矩阵，而非具体的矩阵对象
    >>> L
    <3x3 _CustomLinearOperator with dtype=float32>
    # 定义一个形状为 3x3 的自定义线性操作符，数据类型为 float32

    >>> L(np.eye(3))
    # 使用定义的 L 函数计算单位矩阵 np.eye(3) 的 Laplacian 矩阵并返回结果
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]])

    or as a lambda-function:

    >>> L = csgraph.laplacian(G, form="function")
    # 使用 csgraph.laplacian 函数创建一个以函数形式表示的 Laplacian 矩阵 L
    >>> L
    # 打印输出 L 的信息
    <function _laplace.<locals>.<lambda> at 0x0000012AE6F5A598>
    
    >>> L(np.eye(3))
    # 使用函数形式的 Laplacian 矩阵 L 计算单位矩阵 np.eye(3) 的 Laplacian 矩阵并返回结果
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]])

    The Laplacian matrix is used for
    spectral data clustering and embedding
    as well as for spectral graph partitioning.
    Our final example illustrates the latter
    for a noisy directed linear graph.

    >>> from scipy.sparse import diags, random
    # 导入 scipy.sparse 中的 diags 和 random 模块
    >>> from scipy.sparse.linalg import lobpcg
    # 导入 scipy.sparse.linalg 中的 lobpcg 函数

    Create a directed linear graph with ``N=35`` vertices
    using a sparse adjacency matrix ``G``:

    >>> N = 35
    # 定义图中顶点的数量 N 为 35
    >>> G = diags(np.ones(N-1), 1, format="csr")
    # 使用 scipy.sparse.diags 创建一个格式为 csr 的稀疏对角矩阵 G，其中对角线上的元素为 1

    Fix a random seed ``rng`` and add a random sparse noise to the graph ``G``:

    >>> rng = np.random.default_rng()
    # 使用 np.random.default_rng() 初始化一个随机数生成器 rng
    >>> G += 1e-2 * random(N, N, density=0.1, random_state=rng)
    # 将随机稀疏噪声加到图 G 中，密度为 0.1，随机状态为 rng

    Set initial approximations for eigenvectors:

    >>> X = rng.random((N, 2))
    # 使用 rng 生成一个 N x 2 大小的随机数组 X 作为特征向量的初始近似值

    The constant vector of ones is always a trivial eigenvector
    of the non-normalized Laplacian to be filtered out:

    >>> Y = np.ones((N, 1))
    # 创建一个全为 1 的 N x 1 大小的向量 Y 作为 Laplacian 矩阵中的常数向量

    Alternating (1) the sign of the graph weights allows determining
    labels for spectral max- and min- cuts in a single loop.
    Since the graph is undirected, the option ``symmetrized=True``
    must be used in the construction of the Laplacian.
    The option ``normed=True`` cannot be used in (2) for the negative weights
    here as the symmetric normalization evaluates square roots.
    The option ``form="lo"`` in (2) is matrix-free, i.e., guarantees
    a fixed memory footprint and read-only access to the graph.
    Calling the eigenvalue solver ``lobpcg`` (3) computes the Fiedler vector
    that determines the labels as the signs of its components in (5).
    Since the sign in an eigenvector is not deterministic and can flip,
    we fix the sign of the first component to be always +1 in (4).

    >>> for cut in ["max", "min"]:
    ...     G = -G  # 1. 切换图的权重符号
    ...     L = csgraph.laplacian(G, symmetrized=True, form="lo")  # 2. 构建 Laplacian 矩阵
    ...     _, eves = lobpcg(L, X, Y=Y, largest=False, tol=1e-3)  # 3. 调用 lobpcg 求解特征向量
    ...     eves *= np.sign(eves[0, 0])  # 4. 确定特征向量的符号
    ...     print(cut + "-cut labels:\n", 1 * (eves[:, 0]>0))  # 5. 打印切割标签
    max-cut labels:
    [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]
    min-cut labels:
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    As anticipated for a (slightly noisy) linear graph,
    the max-cut strips all the edges of the graph coloring all
    odd vertices into one color and all even vertices into another one,
    while the balanced min-cut partitions the graph
    in the middle by deleting a single edge.
    Both determined partitions are optimal.
    ```
    # 如果 csgraph 的维度不是二维或者行列数不相等，则抛出数值错误异常
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError("csgraph must be a square matrix or array")
    
    # 如果 normed 为真，并且 csgraph 的数据类型是有符号整数或无符号整数，则将 csgraph 转换为 np.float64 类型
    if normed and (
        np.issubdtype(csgraph.dtype, np.signedinteger)
        or np.issubdtype(csgraph.dtype, np.uint)
    ):
        csgraph = csgraph.astype(np.float64)
    
    # 根据参数 form 的值选择使用稀疏还是密集形式来创建拉普拉斯矩阵的函数
    if form == "array":
        create_lap = _laplacian_sparse if issparse(csgraph) else _laplacian_dense
    else:
        create_lap = (
            _laplacian_sparse_flo if issparse(csgraph) else _laplacian_dense_flo
        )
    
    # 根据 use_out_degree 决定度的计算轴，如果为真，则在 axis=1，否则在 axis=0
    degree_axis = 1 if use_out_degree else 0
    
    # 调用相应的创建拉普拉斯矩阵的函数，返回拉普拉斯矩阵 lap 和度向量 d
    lap, d = create_lap(
        csgraph,
        normed=normed,
        axis=degree_axis,
        copy=copy,
        form=form,
        dtype=dtype,
        symmetrized=symmetrized,
    )
    
    # 如果 return_diag 为真，则返回拉普拉斯矩阵 lap 和度向量 d；否则只返回拉普拉斯矩阵 lap
    if return_diag:
        return lap, d
    return lap
def _setdiag_dense(m, d):
    # 计算步长，用于设置对角线元素
    step = len(d) + 1
    # 将数组 d 中的元素设置为 m 的对角线元素
    m.flat[::step] = d


def _laplace(m, d):
    # 返回一个函数，计算 Laplace 矩阵与向量 v 的乘积
    return lambda v: v * d[:, np.newaxis] - m @ v


def _laplace_normed(m, d, nd):
    # 计算归一化 Laplace 矩阵与向量 v 的乘积
    laplace = _laplace(m, d)
    return lambda v: nd[:, np.newaxis] * laplace(v * nd[:, np.newaxis])


def _laplace_sym(m, d):
    # 返回一个函数，计算对称 Laplace 矩阵与向量 v 的乘积
    return (
        lambda v: v * d[:, np.newaxis]
        - m @ v
        - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m))
    )


def _laplace_normed_sym(m, d, nd):
    # 计算归一化对称 Laplace 矩阵与向量 v 的乘积
    laplace_sym = _laplace_sym(m, d)
    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])


def _linearoperator(mv, shape, dtype):
    # 返回一个线性操作器对象，用给定的 matvec 函数和参数构造
    return LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)


def _laplacian_sparse_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    # 关键字参数 `copy` 在这里未被使用，没有任何影响
    del copy

    if dtype is None:
        dtype = graph.dtype

    # 计算图的每行或每列的和，返回一个展平的数组
    graph_sum = np.asarray(graph.sum(axis=axis)).ravel()
    # 提取图的对角线元素
    graph_diagonal = graph.diagonal()
    # 计算图的度数减去对角线元素，得到度数
    diag = graph_sum - graph_diagonal
    if symmetrized:
        # 如果需要对称化，计算图每行和每列的和，并重新计算度数
        graph_sum += np.asarray(graph.sum(axis=1 - axis)).ravel()
        diag = graph_sum - graph_diagonal - graph_diagonal

    if normed:
        # 对于归一化操作
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            # 计算归一化对称 Laplace 矩阵
            md = _laplace_normed_sym(graph, graph_sum, 1.0 / w)
        else:
            # 计算归一化 Laplace 矩阵
            md = _laplace_normed(graph, graph_sum, 1.0 / w)
        if form == "function":
            # 返回函数和权重数组
            return md, w.astype(dtype, copy=False)
        elif form == "lo":
            # 返回线性操作器和权重数组
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, w.astype(dtype, copy=False)
        else:
            # 抛出异常，无效的形式参数
            raise ValueError(f"Invalid form: {form!r}")
    else:
        # 对于非归一化操作
        if symmetrized:
            # 计算对称 Laplace 矩阵
            md = _laplace_sym(graph, graph_sum)
        else:
            # 计算 Laplace 矩阵
            md = _laplace(graph, graph_sum)
        if form == "function":
            # 返回函数和度数数组
            return md, diag.astype(dtype, copy=False)
        elif form == "lo":
            # 返回线性操作器和度数数组
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, diag.astype(dtype, copy=False)
        else:
            # 抛出异常，无效的形式参数
            raise ValueError(f"Invalid form: {form!r}")


def _laplacian_sparse(graph, normed, axis, copy, form, dtype, symmetrized):
    # 关键字参数 `form` 在这里未被使用，没有任何影响
    del form

    if dtype is None:
        dtype = graph.dtype

    needs_copy = False
    if graph.format in ("lil", "dok"):
        # 如果图的格式是 lil 或 dok，则转换为 coo 格式
        m = graph.tocoo()
    else:
        m = graph
        if copy:
            # 如果需要复制图，则设置需要复制的标志
            needs_copy = True

    if symmetrized:
        # 如果需要对称化，则将图与其共轭转置相加
        m += m.T.conj()

    # 计算图的每行或每列的和，返回一个展平的数组，并计算度数
    w = np.asarray(m.sum(axis=axis)).ravel() - m.diagonal()
    if normed:
        # 对于归一化操作
        m = m.tocoo(copy=needs_copy)
        isolated_node_mask = w == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        m.setdiag(1 - isolated_node_mask)
    else:
        # 如果格式不是 "dia"，则将稀疏矩阵 m 转换为 COO 格式
        if m.format == "dia":
            m = m.copy()  # 复制稀疏矩阵 m
        else:
            m = m.tocoo(copy=needs_copy)  # 转换为 COO 格式，根据需要复制
        # 将稀疏矩阵 m 的数据元素乘以 -1
        m.data *= -1
        # 设置 m 的对角线元素为 w
        m.setdiag(w)

    # 将稀疏矩阵 m 和数组 w 转换为指定的数据类型 dtype，返回不复制任何数据
    return m.astype(dtype, copy=False), w.astype(dtype)
# 根据输入的图形对象和参数计算密集图的拉普拉斯矩阵
def _laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized):
    # 如果形式不是数组，则引发值错误异常
    if form != "array":
        raise ValueError(f'{form!r} must be "array"')

    # 如果未指定数据类型，则使用图形对象的数据类型
    if dtype is None:
        dtype = graph.dtype

    # 根据复制标志，创建图形的副本或视图
    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    # 如果未指定数据类型，则使用m的数据类型
    if dtype is None:
        dtype = m.dtype

    # 如果需要对称化图形，则添加其转置的共轭
    if symmetrized:
        m += m.T.conj()

    # 将主对角线元素置零
    np.fill_diagonal(m, 0)

    # 计算图形在给定轴上的加权和
    w = m.sum(axis=axis)

    # 如果进行归一化处理
    if normed:
        # 找出孤立节点的标记
        isolated_node_mask = w == 0
        # 将孤立节点权重设置为1，其余为sqrt(w)
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        # 对称归一化处理
        m /= w
        m /= w[:, np.newaxis]
        m *= -1
        # 设置稀疏对角线元素
        _setdiag_dense(m, 1 - isolated_node_mask)
    else:
        # 非归一化处理，直接乘以-1，并设置稀疏对角线元素
        m *= -1
        _setdiag_dense(m, w)

    # 返回处理后的密集拉普拉斯矩阵和权重数组，转换为指定数据类型
    return m.astype(dtype, copy=False), w.astype(dtype, copy=False)
```