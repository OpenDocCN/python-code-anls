# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_spectral_embedding.py`

```
"""Spectral Embedding."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import warnings
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg

from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import (
    check_array,
    check_random_state,
    check_symmetric,
)
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import _deterministic_vector_sign_flip
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.fixes import parse_version, sp_version


def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node.

    Parameters
    ----------
    graph : array-like of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    node_id : int
        The index of the query node of the graph.

    Returns
    -------
    connected_components_matrix : array-like of shape (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node.
    """
    # 获取节点数量
    n_node = graph.shape[0]
    
    # 如果图是稀疏矩阵，转换为 CSR 格式以加速行访问
    if sparse.issparse(graph):
        graph = graph.tocsr()
    
    # 初始化连接节点的布尔数组
    connected_nodes = np.zeros(n_node, dtype=bool)
    
    # 初始化待探索的节点布尔数组，并设置初始节点为查询节点
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    
    # 迭代探索节点，直到无法扩展为止
    for _ in range(n_node):
        # 记录上一轮迭代连接的节点数量
        last_num_component = connected_nodes.sum()
        
        # 将待探索节点和已连接节点进行逻辑或操作，更新已连接节点
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        
        # 如果本轮迭代没有新节点连接，则停止迭代
        if last_num_component >= connected_nodes.sum():
            break
        
        # 获取当前待探索节点的索引
        indices = np.where(nodes_to_explore)[0]
        
        # 清空待探索节点数组
        nodes_to_explore.fill(False)
        
        # 遍历当前待探索节点，获取其邻居节点并更新待探索节点数组
        for i in indices:
            if sparse.issparse(graph):
                # 对稀疏图，获取邻居节点信息，目前使用全行切片代替一维稀疏切片
                neighbors = graph[[i], :].toarray().ravel()
            else:
                neighbors = graph[i]
            
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    
    # 返回连接的节点布尔数组，表示最大连通子图的节点集合
    return connected_nodes


def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    # 如果输入的图是稀疏矩阵
    if sparse.issparse(graph):
        # 在 Scipy 1.11.3 之前，`connected_components` 只支持 32 位索引。
        # PR: https://github.com/scipy/scipy/pull/18913
        # 第一次集成于 1.11.3: https://github.com/scipy/scipy/pull/19279
        # TODO(jjerphan): 一旦 SciPy 1.11.3 成为最低支持版本，使用 `accept_large_sparse=True`。
        
        # 检查当前 SciPy 版本是否大于等于 1.11.3，以确定是否接受大稀疏矩阵
        accept_large_sparse = sp_version >= parse_version("1.11.3")
        # 检查并转换图形为数组格式，接受稀疏矩阵，根据版本接受大稀疏矩阵
        graph = check_array(
            graph, accept_sparse=True, accept_large_sparse=accept_large_sparse
        )
        # 对稀疏图进行连通性分析，获取连通分量的数量
        n_connected_components, _ = connected_components(graph)
        # 返回是否只有一个连通分量
        return n_connected_components == 1
    else:
        # 对于密集图，从节点 0 开始找到所有连通分量
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]
# 将拉普拉斯矩阵的对角线设置为指定值，并将其转换为适合特征值分解的稀疏格式。

def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.

    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.

    value : float
        The value of the diagonal.

    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]

    # We need all entries in the diagonal to values
    # 如果 laplacian 不是稀疏矩阵
    if not sparse.issparse(laplacian):
        # 如果需要规范化 laplacian 的对角线
        if norm_laplacian:
            # 将 laplacian 对角线上的每个元素设置为 value
            laplacian.flat[:: n_nodes + 1] = value
    else:
        # 如果 laplacian 是稀疏矩阵，转换为 COO 格式
        laplacian = laplacian.tocoo()
        # 如果需要规范化 laplacian 的对角线
        if norm_laplacian:
            # 找到对角线上的索引并设置对应的值为 value
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        
        # 计算 laplacian 的非零元素行与列之差的唯一值数量
        n_diags = np.unique(laplacian.row - laplacian.col).size
        # 如果非零元素的行列差值唯一值数量小于等于 7
        if n_diags <= 7:
            # 选择最适合矩阵向量乘积的 dia 格式
            laplacian = laplacian.todia()
        else:
            # csr 格式对于 arpack 算法有最快的矩阵向量乘积速度
            laplacian = laplacian.tocsr()
    
    # 返回处理后的 laplacian 矩阵
    return laplacian
    # so that the eigenvector decomposition works as expected.
    # 为了确保特征向量分解按预期工作。
    
    # Note : Laplacian Eigenmaps is the actual algorithm implemented here.
    # Laplacian Eigenmaps 是此处实现的实际算法。
    
    # Read more in the :ref:`User Guide <spectral_embedding>`.
    # 可在用户指南的 :ref:`User Guide <spectral_embedding>` 中详细了解更多信息。
    
    # Parameters
    # 参数说明开始
    
    # adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)
    # 图的邻接矩阵，可以是数组形式或稀疏图形式，形状为 (n_samples, n_samples)
    
    # n_components : int, default=8
    # 投影子空间的维度，默认为 8
    
    # eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
    # 使用的特征值分解策略。AMG 需要安装 pyamg 库。在处理非常大、稀疏问题时可能更快，但也可能导致不稳定性。
    # 如果为 None，则使用 'arpack'。
    
    # random_state : int, RandomState instance or None, default=None
    # 伪随机数生成器，用于 lobpcg 特征向量分解的初始化（当 eigen_solver == 'amg' 时），以及 K-Means 初始化。
    # 使用整数可以使结果在不同调用间确定性。详见术语表中的 "Glossary"。
    
    # .. note::
    # 当使用 eigen_solver == 'amg' 时，需要同时固定全局 numpy 种子以获得确定性结果。
    # 请参阅 https://github.com/pyamg/pyamg/issues/139 获取更多信息。
    
    # eigen_tol : float, default="auto"
    # Laplacian 矩阵特征值分解的停止准则。
    # 如果 eigen_tol="auto"，则传递的容差将取决于 eigen_solver：
    # - 如果 eigen_solver="arpack"，则 eigen_tol=0.0；
    # - 如果 eigen_solver="lobpcg" 或 eigen_solver="amg"，则 eigen_tol=None，
    #   这会根据其启发式自动配置 lobpcg 求解器的值。详见 scipy.sparse.linalg.lobpcg。
    
    # Note that when using eigen_solver="amg" values of tol<1e-5 may lead to convergence issues and should be avoided.
    # 注意，当使用 eigen_solver="amg" 时，tol<1e-5 的值可能导致收敛问题，应避免使用。
    
    # .. versionadded:: 1.2
    #   Added 'auto' option.
    #   版本添加：增加了 'auto' 选项。
    
    # norm_laplacian : bool, default=True
    # 如果为 True，则计算对称归一化的拉普拉斯矩阵。
    
    # drop_first : bool, default=True
    # 是否丢弃第一个特征向量。对于谱嵌入来说，应该设为 True，因为第一个特征向量应该是连接图的常量向量；
    # 对于谱聚类来说，应该设为 False，以保留第一个特征向量。
    
    # Returns
    # 返回结果
    
    # embedding : ndarray of shape (n_samples, n_components)
    # 降维后的样本集合的数组，形状为 (n_samples, n_components)
    
    # Notes
    # 注意事项
    
    # Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    # has one connected component. If there graph has many components, the first
    # few eigenvectors will simply uncover the connected components of the graph.
    # 谱嵌入（拉普拉斯特征映射）在图形有一个连接组件时最为有用。如果图形有多个组件，
    # 前几个特征向量将简单地揭示图形的连接组件。
    
    # References
    # 参考文献
    random_state = check_random_state(random_state)

随机状态初始化函数，确保随机数生成器在每次调用时都使用相同的种子，以便结果可复现。


    return _spectral_embedding(
        adjacency,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        norm_laplacian=norm_laplacian,
        drop_first=drop_first,
    )

调用 `_spectral_embedding` 函数进行谱嵌入计算：
- `adjacency`: 表示输入的邻接矩阵或者相似度矩阵。
- `n_components`: 指定要嵌入的特征向量数量。
- `eigen_solver`: 指定要使用的特征值求解方法。
- `random_state`: 用于控制随机数生成的种子。
- `eigen_tol`: 特征值计算的精度容限。
- `norm_laplacian`: 是否归一化拉普拉斯矩阵。
- `drop_first`: 是否去掉第一个特征向量。

该函数返回谱嵌入的结果，即特征向量表示的低维空间投影。
# 检查邻接矩阵是否对称，并确保其对称性
adjacency = check_symmetric(adjacency)

# 如果 eigen_solver 设置为 "amg"，尝试导入 smoothed_aggregation_solver 方法
if eigen_solver == "amg":
    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError as e:
        # 如果导入失败，抛出 ValueError 异常
        raise ValueError(
            "The eigen_solver was set to 'amg', but pyamg is not available."
        ) from e

# 如果 eigen_solver 未设置，则默认使用 "arpack"
if eigen_solver is None:
    eigen_solver = "arpack"

# 获取邻接矩阵的节点数量
n_nodes = adjacency.shape[0]

# 如果 drop_first 为 True，则将 n_components 值增加 1
if drop_first:
    n_components = n_components + 1

# 如果图不是完全连通，发出警告，因为谱嵌入可能不会按预期工作
if not _graph_is_connected(adjacency):
    warnings.warn(
        "Graph is not fully connected, spectral embedding may not work as expected."
    )

# 计算拉普拉斯矩阵和对角元素
laplacian, dd = csgraph_laplacian(
    adjacency, normed=norm_laplacian, return_diag=True
)

# 如果 eigen_solver 设置为 "arpack"，或者不是 "lobpcg" 并且拉普拉斯矩阵不是稀疏的或节点数量小于 5 * n_components
if (
    eigen_solver == "arpack"
    or eigen_solver != "lobpcg"
    and (not sparse.issparse(laplacian) or n_nodes < 5 * n_components)
):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        
        # Here we'll use shift-invert mode for fast eigenvalues
        # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            tol = 0 if eigen_tol == "auto" else eigen_tol
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            laplacian = check_array(
                laplacian, accept_sparse="csr", accept_large_sparse=False
            )
            _, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=tol, v0=v0
            )
            # Extract the embedding by reversing the order of the components
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                # If the Laplacian was normalized, recover the original embedding
                # by reversing the normalization process
                embedding = embedding / dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1
    # 如果使用 AMG 方法求解特征值问题：
    # 使用 AMG 获取预处理器以加速特征值问题。
    if eigen_solver == "amg":
        # 检查 Laplacian 是否为稀疏矩阵，如果不是则发出警告
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        
        # 将 laplacian 转换为指定数据类型的数组，并接受稀疏矩阵作为输入
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        
        # 设置 Laplacian 对角线为全一，并进行归一化（如果需要）
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        
        # Laplacian 矩阵始终是奇异的，至少有一个零特征值，对应于常数特征向量。
        # 使用奇异矩阵作为预处理器可能导致 LOBPCG 中的随机失败，并且不被现有理论支持：
        # 参考文献：https://doi.org/10.1007/s10208-015-9297-1
        
        # 通过在 Laplacian 上加上一个对角线偏移来避免所有对角线元素都为一的情况。
        # 这种偏移确实会改变特征对，因此我们将偏移后的矩阵输入求解器，并在之后将其设置回原始状态。
        diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        
        # 如果 laplacian 是 csr_array，并且 sparse 模块具有 csr_array 属性，
        # 则需要将其转换为 csr_matrix 对象，因为 pyamg 不适用于 csr_array。
        if hasattr(sparse, "csr_array") and isinstance(laplacian, sparse.csr_array):
            laplacian = sparse.csr_matrix(laplacian)
        
        # 使用平滑聚合求解器处理 laplacian 的 csr 形式
        ml = smoothed_aggregation_solver(check_array(laplacian, accept_sparse="csr"))
        
        # 将 laplacian 还原为原始状态，减去之前添加的对角线偏移
        laplacian -= diag_shift
        
        # 将 ml 转换为预处理器 M
        M = ml.aspreconditioner()
        
        # 创建初始近似值 X 作为特征向量
        X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
        X[:, 0] = dd.ravel()
        X = X.astype(laplacian.dtype)
        
        # 如果 eigen_tol 为 "auto"，则 tol 设置为 None，否则使用指定的 eigen_tol
        tol = None if eigen_tol == "auto" else eigen_tol
        
        # 使用 LOBPCG 方法求解 laplacian 的特征值问题
        _, diffusion_map = lobpcg(laplacian, X, M=M, tol=tol, largest=False)
        
        # 获取扩散映射结果，embedding 为其转置
        embedding = diffusion_map.T
        
        # 如果需要归一化 Laplacian，则从特征向量输出中恢复 u = D^-1/2 x
        if norm_laplacian:
            embedding = embedding / dd
        
        # 如果 embedding 的第一维度为 1，则引发 ValueError
        if embedding.shape[0] == 1:
            raise ValueError
    # 如果 eigen_solver 是 "lobpcg"，则执行以下代码块
    if eigen_solver == "lobpcg":
        # 检查 laplacian 是否为 np.float64 或 np.float32 类型的数组，并允许稀疏矩阵
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        # 如果节点数 n_nodes 小于 5 * n_components + 1，则执行以下代码块
        if n_nodes < 5 * n_components + 1:
            # 如果 laplacian 是稀疏矩阵，则转换为稠密数组
            if sparse.issparse(laplacian):
                laplacian = laplacian.toarray()
            # 计算 laplacian 的特征值和特征向量
            _, diffusion_map = eigh(laplacian, check_finite=False)
            # 取得扩散映射的前 n_components 个特征向量
            embedding = diffusion_map.T[:n_components]
            # 如果 norm_laplacian 为 True，则按照公式恢复 u = D^-1/2 x 的值
            if norm_laplacian:
                embedding = embedding / dd
        else:
            # 将 laplacian 对角线设置为 1，用于归一化拉普拉斯矩阵
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # 增加请求的特征向量数量，因为 lobpcg 在低维度下表现不佳，并创建初始近似 X
            X = random_state.standard_normal(
                size=(laplacian.shape[0], n_components + 1)
            )
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            # 如果 eigen_tol 是 "auto"，则 tol 设为 None，否则设为给定的 eigen_tol
            tol = None if eigen_tol == "auto" else eigen_tol
            # 使用 lobpcg 求解特征值问题
            _, diffusion_map = lobpcg(
                laplacian, X, tol=tol, largest=False, maxiter=2000
            )
            # 取得扩散映射的前 n_components 个特征向量
            embedding = diffusion_map.T[:n_components]
            # 如果 norm_laplacian 为 True，则按照公式恢复 u = D^-1/2 x 的值
            if norm_laplacian:
                embedding = embedding / dd
            # 如果 embedding 的第一维度长度为 1，则引发 ValueError 异常
            if embedding.shape[0] == 1:
                raise ValueError

    # 对 embedding 进行确定性的向量符号翻转处理
    embedding = _deterministic_vector_sign_flip(embedding)
    # 如果 drop_first 为 True，则返回 embedding 的第 1 到 n_components 列的转置
    if drop_first:
        return embedding[1:n_components].T
    else:
        # 否则，返回 embedding 的前 n_components 列的转置
        return embedding[:n_components].T
class SpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.

    affinity : {'nearest_neighbors', 'rbf', 'precomputed', \
                'precomputed_nearest_neighbors'} or callable, \
                default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf' : construct the affinity matrix by computing a radial basis
           function (RBF) kernel.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
         - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
           of precomputed nearest neighbors, and constructs the affinity matrix
           by selecting the ``n_neighbors`` nearest neighbors.
         - callable : use passed in function as affinity
           the function takes in data matrix (n_samples, n_features)
           and return affinity matrix (n_samples, n_samples).

    gamma : float, default=None
        Kernel coefficient for rbf kernel. If None, gamma will be set to
        1/n_features.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems.
        If None, then ``'arpack'`` is used.
    """
    eigen_tol : float, default="auto"
        Laplacian矩阵特征分解的停止准则。
        如果 `eigen_tol="auto"`，则传入的容差依赖于 `eigen_solver` 的设置：

        - 如果 `eigen_solver="arpack"`，则 `eigen_tol=0.0`；
        - 如果 `eigen_solver="lobpcg"` 或 `eigen_solver="amg"`，则
          `eigen_tol=None`，这会根据其启发式策略自动配置 `lobpcg` 解算器的值。详见
          :func:`scipy.sparse.linalg.lobpcg`。

        注意，当使用 `eigen_solver="lobpcg"` 或 `eigen_solver="amg"` 时，`tol<1e-5` 可能导致收敛问题，应避免使用这些值。

        .. versionadded:: 1.2

    n_neighbors : int, default=None
        构建最近邻图时的最近邻数目。
        如果为 None，则 n_neighbors 将被设置为 max(n_samples/10, 1)。

    n_jobs : int, default=None
        并行作业的数量。
        ``None`` 表示使用1个核心，除非在 :obj:`joblib.parallel_backend` 上下文中。
        ``-1`` 表示使用所有处理器。有关详细信息，请参见 :term:`术语表 <n_jobs>`。

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        训练矩阵的谱嵌入。

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        从样本或预计算数据构建的关联矩阵。

    n_features_in_ : int
        在 :term:`fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 :term:`fit` 过程中观察到的特征名称。仅在 `X` 全为字符串类型特征名时定义。

        .. versionadded:: 1.0

    n_neighbors_ : int
        实际使用的最近邻数目。

    See Also
    --------
    Isomap : 通过等距映射进行非线性降维。

    References
    ----------

    - :doi:`A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      <10.1007/s11222-007-9033-z>`

    - `On Spectral Clustering: Analysis and an algorithm, 2001
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      <https://citeseerx.ist.psu.edu/doc_view/pid/796c5d6336fc52aa84db575fb821c78918b65f58>`_

    - :doi:`Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      <10.1109/34.868688>`

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import SpectralEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = SpectralEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    # 定义参数约束字典，用于指定每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "affinity": [
            StrOptions(
                {
                    "nearest_neighbors",
                    "rbf",
                    "precomputed",
                    "precomputed_nearest_neighbors",
                },
            ),
            callable,  # affinity 参数可以是一个可调用对象
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],  # gamma 参数可以是实数大于0或者为None
        "random_state": ["random_state"],  # random_state 参数必须为字符串 "random_state"
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],  # eigen_solver 参数必须是 "arpack", "lobpcg" 或 "amg" 中的一个，或者为None
        "eigen_tol": [Interval(Real, 0, None, closed="left"), StrOptions({"auto"})],  # eigen_tol 参数可以是实数大于0或者为字符串 "auto"
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],  # n_neighbors 参数可以是整数大于等于1或者为None
        "n_jobs": [None, Integral],  # n_jobs 参数可以为None或者整数
    }

    # 初始化方法，设置类的属性值
    def __init__(
        self,
        n_components=2,
        *,
        affinity="nearest_neighbors",
        gamma=None,
        random_state=None,
        eigen_solver=None,
        eigen_tol="auto",
        n_neighbors=None,
        n_jobs=None,
    ):
        self.n_components = n_components  # 设置 n_components 属性值
        self.affinity = affinity  # 设置 affinity 属性值
        self.gamma = gamma  # 设置 gamma 属性值
        self.random_state = random_state  # 设置 random_state 属性值
        self.eigen_solver = eigen_solver  # 设置 eigen_solver 属性值
        self.eigen_tol = eigen_tol  # 设置 eigen_tol 属性值
        self.n_neighbors = n_neighbors  # 设置 n_neighbors 属性值
        self.n_jobs = n_jobs  # 设置 n_jobs 属性值

    # 返回额外的标签信息，根据 affinity 参数值决定是否返回 "pairwise" 标签
    def _more_tags(self):
        return {
            "pairwise": self.affinity
            in [
                "precomputed",
                "precomputed_nearest_neighbors",
            ]
        }
    # 计算数据的亲和矩阵
    def _get_affinity_matrix(self, X, Y=None):
        """Calculate the affinity matrix from data
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : array-like of shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Y: Ignored

        Returns
        -------
        affinity_matrix of shape (n_samples, n_samples)
        """
        # 如果亲和性是 "precomputed"，则直接将 X 作为亲和矩阵
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
            return self.affinity_matrix_
        # 如果亲和性是 "precomputed_nearest_neighbors"，使用最近邻图进行计算
        if self.affinity == "precomputed_nearest_neighbors":
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric="precomputed"
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            return self.affinity_matrix_
        # 如果亲和性是 "nearest_neighbors"，根据最近邻参数计算亲和矩阵
        if self.affinity == "nearest_neighbors":
            if sparse.issparse(X):
                # 警告：最近邻亲和度暂不支持稀疏输入，自动切换至 RBF 亲和度
                warnings.warn(
                    "Nearest neighbors affinity currently does "
                    "not support sparse input, falling back to "
                    "rbf affinity"
                )
                self.affinity = "rbf"
            else:
                self.n_neighbors_ = (
                    self.n_neighbors
                    if self.n_neighbors is not None
                    else max(int(X.shape[0] / 10), 1)
                )
                self.affinity_matrix_ = kneighbors_graph(
                    X, self.n_neighbors_, include_self=True, n_jobs=self.n_jobs
                )
                # 当前只支持对称的亲和矩阵
                self.affinity_matrix_ = 0.5 * (
                    self.affinity_matrix_ + self.affinity_matrix_.T
                )
                return self.affinity_matrix_
        # 如果亲和性是 "rbf"，使用 RBF 核函数计算亲和矩阵
        if self.affinity == "rbf":
            self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        # 使用自定义的亲和函数计算亲和矩阵
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : {array-like, sparse matrix}, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input data X, ensuring it's sparse and has at least 2 samples
        X = self._validate_data(X, accept_sparse="csr", ensure_min_samples=2)

        # Set up random state for reproducibility
        random_state = check_random_state(self.random_state)

        # Compute affinity matrix based on input data X
        affinity_matrix = self._get_affinity_matrix(X)

        # Perform spectral embedding using computed affinity matrix
        self.embedding_ = _spectral_embedding(
            affinity_matrix,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            eigen_tol=self.eigen_tol,
            random_state=random_state,
        )

        # Return the instance of the fitted model
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : {array-like, sparse matrix} of shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Spectral embedding of the training matrix.
        """
        # Fit the model to data X and obtain the spectral embedding
        self.fit(X)

        # Return the computed embedding
        return self.embedding_
```