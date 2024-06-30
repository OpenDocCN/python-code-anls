# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_spectral.py`

```
# Algorithms for spectral clustering

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix

from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..manifold._spectral_embedding import _spectral_embedding
from ..metrics.pairwise import KERNEL_PARAMS, pairwise_kernels
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import as_float_array, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ._kmeans import k_means


def cluster_qr(vectors):
    """Find the discrete partition closest to the eigenvector embedding.

    This implementation was proposed in [1]_.

    .. versionadded:: 1.1

    Parameters
    ----------
    vectors : array-like, shape: (n_samples, n_clusters)
        The embedding space of the samples.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The cluster labels of vectors.

    References
    ----------
    .. [1] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
        Anil Damle, Victor Minden, Lexing Ying
        <10.1093/imaiai/iay008>`

    """
    k = vectors.shape[1]  # 获取矩阵中的聚类数目
    _, _, piv = qr(vectors.T, pivoting=True)  # 使用 QR 分解找到排列顺序
    ut, _, v = svd(vectors[piv[:k], :].T)  # 使用 SVD 分解特征向量
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))  # 计算特征向量的最大值
    return vectors.argmax(axis=1)  # 返回每行的最大值索引作为标签


def discretize(
    vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None
):
    """Search for a partition matrix which is closest to the eigenvector embedding.

    This implementation was proposed in [1]_.

    Parameters
    ----------
    vectors : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.

    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.

    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails

    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached

    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    .. [1] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_

    Notes
    -----

    The eigenvector embedding is used to iteratively search for the
    """
    # 根据输入的随机种子值获取随机状态对象
    random_state = check_random_state(random_state)

    # 将输入的向量数据转换为浮点数数组，并复制一份副本
    vectors = as_float_array(vectors, copy=copy)

    # 计算浮点数表示的机器精度
    eps = np.finfo(float).eps
    # 获取向量数据的维度信息
    n_samples, n_components = vectors.shape

    # 将每个特征向量归一化到长度为 sqrt(n_samples) 的向量空间
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        # 如果特征向量的第一个元素不为零，则调整向量方向为负方向
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # 将特征向量的行归一化，确保样本点位于以原点为中心的单位超球面上
    # 这一步将样本从嵌入空间转换到分区矩阵的空间
    vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, np.newaxis]

    # SVD 重新启动次数初始化为 0
    svd_restarts = 0
    # 设定迭代是否已经收敛的标志为 False
    has_converged = False

    # 如果发生异常，尝试随机化并重新运行 SVD，最多尝试 max_svd_restarts 次
    while (svd_restarts < max_svd_restarts) and not has_converged:
        # 当还未达到最大重启次数且尚未收敛时，执行以下循环

        # 初始化旋转矩阵的第一列，使用随机选择的特征向量的一行
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # 初始化剩余部分的旋转矩阵，选择与之前选取的特征向量行尽可能正交的行
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # 累积 c，确保当前行与之前选取的行以及当前行尽可能正交
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            # 在未收敛时执行以下循环
            n_iter += 1

            # 计算离散化后的投影 t_discrete
            t_discrete = np.dot(vectors, rotation)

            # 根据每行最大值确定离散化后的标签
            labels = t_discrete.argmax(axis=1)

            # 创建稀疏矩阵 vectors_discrete，用于后续计算
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            # 计算 t_svd，用于奇异值分解
            t_svd = vectors_discrete.T * vectors

            try:
                # 进行奇异值分解
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                # 如果奇异值分解未收敛，增加重启次数并重新随机初始化
                svd_restarts += 1
                print("SVD did not converge, randomizing and trying again")
                break

            # 计算当前的 ncut_value
            ncut_value = 2.0 * (n_samples - S.sum())

            # 检查是否满足收敛条件或达到最大迭代次数
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # 否则更新目标函数值并继续迭代
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        # 如果最终未收敛，抛出奇异值分解未收敛的异常
        raise LinAlgError("SVD did not converge")
    return labels
# 使用 @validate_params 装饰器对 spectral_clustering 函数进行参数验证
@validate_params(
    {"affinity": ["array-like", "sparse matrix"]},
    prefer_skip_nested_validation=False,
)
# 定义 spectral_clustering 函数，实现将标准化拉普拉斯的投影应用于聚类
def spectral_clustering(
    affinity,
    *,
    n_clusters=8,  # 聚类的数量，默认为 8
    n_components=None,  # 用于谱嵌入的特征向量数量，默认为 n_clusters
    eigen_solver=None,  # 特征值求解方法，默认为 'arpack'，可选 'lobpcg' 或 'amg'
    random_state=None,  # 用于 lobpcg 特征向量分解和 K-Means 初始化的随机数生成器，默认为 None
    n_init=10,  # K-Means 算法的运行次数，选择最佳结果，默认为 10
    eigen_tol="auto",  # 特征值求解的容差，默认为 "auto"
    assign_labels="kmeans",  # 分配标签的方法，默认为 'kmeans'，也可选 'discretize'
    verbose=False,  # 是否输出详细信息，默认为 False
):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts [1]_, [2]_.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    affinity : {array-like, sparse matrix} of shape (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : int, default=None
        Number of clusters to extract.

    n_components : int, default=n_clusters
        Number of eigenvectors to use for the spectral embedding.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition method. If None then ``'arpack'`` is used.
        See [4]_ for more details regarding ``'lobpcg'``.
        Eigensolver ``'amg'`` runs ``'lobpcg'`` with optional
        Algebraic MultiGrid preconditioning and requires pyamg to be installed.
        It can be faster on very large sparse problems [6]_ and [7]_.

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.
    eigen_tol : float, default="auto"
        奇异值分解拉普拉斯矩阵的停止标准。
        如果 `eigen_tol="auto"`，则传入的容差取决于 `eigen_solver` 的设置：

        - 如果 `eigen_solver="arpack"`，则 `eigen_tol=0.0`；
        - 如果 `eigen_solver="lobpcg"` 或 `eigen_solver="amg"`，则
          `eigen_tol=None`，这会根据他们的启发式自动配置 `lobpcg` 解算器的值。详见
          :func:`scipy.sparse.linalg.lobpcg` 了解详情。

        注意，当使用 `eigen_solver="lobpcg"` 或 `eigen_solver="amg"` 时，`tol<1e-5`
        可能导致收敛问题，应避免使用这些值。

        .. versionadded:: 1.2
           新增 'auto' 选项。

    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        在嵌入空间中分配标签的策略。
        在拉普拉斯嵌入后，有三种分配标签的方式。k-means 是一种常见选择，
        但对初始化敏感。离散化是另一种方法，对随机初始化不太敏感 [3]_。
        cluster_qr 方法 [5]_ 直接从特征向量中提取聚类。与 k-means 和离散化不同，
        cluster_qr 没有调节参数，并非迭代方法，但在质量和速度上可能优于
        k-means 和离散化。

        .. versionchanged:: 1.1
           新增了新的标签分配方法 'cluster_qr'。

    verbose : bool, default=False
        冗余模式开关。

        .. versionadded:: 0.24

    Returns
    -------
    labels : array of integers, shape: n_samples
        聚类的标签数组。

    Notes
    -----
    图应该只包含一个连通分量，否则结果几乎无意义。

    该算法解决了 `k=2` 的归一化割问题：它是一种归一化谱聚类。

    References
    ----------

    .. [1] :doi:`Normalized cuts and image segmentation, 2000
           Jianbo Shi, Jitendra Malik
           <10.1109/34.868688>`

    .. [2] :doi:`A Tutorial on Spectral Clustering, 2007
           Ulrike von Luxburg
           <10.1007/s11222-007-9033-z>`

    .. [3] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_

    .. [4] :doi:`Toward the Optimal Preconditioned Eigensolver:
           Locally Optimal Block Preconditioned Conjugate Gradient Method, 2001
           A. V. Knyazev
           SIAM Journal on Scientific Computing 23, no. 2, pp. 517-541.
           <10.1137/S1064827500366124>`

    .. [5] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
           Anil Damle, Victor Minden, Lexing Ying
           <10.1093/imaiai/iay008>`
    # 使用 SpectralClustering 进行谱聚类分析
    clusterer = SpectralClustering(
        n_clusters=n_clusters,          # 设置聚类数目
        n_components=n_components,      # 设置要保留的特征向量的个数
        eigen_solver=eigen_solver,      # 设置特征值分解的方法
        random_state=random_state,      # 设置随机数生成器的种子以确保可重复性
        n_init=n_init,                  # 设置算法的初始化次数
        affinity="precomputed",         # 使用预先计算好的相似度矩阵进行聚类
        eigen_tol=eigen_tol,            # 设置特征值分解的容忍度
        assign_labels=assign_labels,    # 设置标签分配策略
        verbose=verbose,                # 控制算法运行时的详细程度
    ).fit(affinity)                     # 对预先计算好的相似度矩阵进行聚类并拟合数据
    
    return clusterer.labels_          # 返回聚类结果中每个样本点的标签
class SpectralClustering(ClusterMixin, BaseEstimator):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex, or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster, such as when clusters are
    nested circles on the 2D plane.

    If the affinity matrix is the adjacency matrix of a graph, this method
    can be used to find normalized graph cuts [1]_, [2]_.

    When calling ``fit``, an affinity matrix is constructed using either
    a kernel function such the Gaussian (aka RBF) kernel with Euclidean
    distance ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    or a k-nearest neighbors connectivity matrix.

    Alternatively, a user-provided affinity matrix can be specified by
    setting ``affinity='precomputed'``.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    n_clusters : int, default=8
        The dimension of the projection subspace.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used. See [4]_ for more details regarding `'lobpcg'`.

    n_components : int, default=None
        Number of eigenvectors to use for the spectral embedding. If None,
        defaults to `n_clusters`.

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``, ``affinity='precomputed'``
        or ``affinity='precomputed_nearest_neighbors'``.
    """
    affinity : str or callable, default='rbf'
        # 用于构建相似度矩阵的方法。
        # - 'nearest_neighbors': 使用最近邻方法构建相似度矩阵。
        # - 'rbf': 使用径向基函数（RBF）核构建相似度矩阵。
        # - 'precomputed': 将 `X` 解释为预先计算好的相似度矩阵，其中较大的值表示实例之间更大的相似性。
        # - 'precomputed_nearest_neighbors': 将 `X` 解释为预先计算的距离稀疏图，并从每个实例的 `n_neighbors` 个最近邻中构建二进制相似度矩阵。
        # - :func:`~sklearn.metrics.pairwise.pairwise_kernels` 支持的核函数之一。
        
        # 仅应使用产生相似度分数（随相似性增加而增加的非负值）的核函数。此属性不被聚类算法检查。

    n_neighbors : int, default=10
        # 使用最近邻方法构建相似度矩阵时要使用的邻居数量。对于 `affinity='rbf'`，此参数被忽略。

    eigen_tol : float, default="auto"
        # Laplacian 矩阵的特征分解停止准则。
        # 如果 `eigen_tol="auto"`，则传递的容差将取决于 `eigen_solver`：
        # - 如果 `eigen_solver="arpack"`，则 `eigen_tol=0.0`；
        # - 如果 `eigen_solver="lobpcg"` 或 `eigen_solver="amg"`，则 `eigen_tol=None`，这会根据它们的启发式自动配置 `lobpcg` 求解器的值。详见 :func:`scipy.sparse.linalg.lobpcg`。

        # 注意，当使用 `eigen_solver="lobpcg"` 或 `eigen_solver="amg"` 时，`tol<1e-5` 的值可能导致收敛问题，应避免使用。

        # .. versionadded:: 1.2
        #    添加了 'auto' 选项。

    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        # 在嵌入空间中分配标签的策略。
        # 在 Laplacian 嵌入后，有两种分配标签的方式。k-means 是一种常见选择，但对初始化敏感。
        # 离散化是另一种方法，对随机初始化不太敏感 [3]_。
        # cluster_qr 方法 [5]_ 直接从谱聚类中的特征向量中提取聚类。与 k-means 和离散化不同，cluster_qr 方法没有调整参数，不进行迭代，但在质量和速度方面可能优于 k-means 和离散化。

        # .. versionchanged:: 1.1
        #    添加了新的标签分配方法 'cluster_qr'。

    degree : float, default=3
        # 多项式核的次数。其他核函数忽略此参数。

    coef0 : float, default=1
        # 多项式核和 sigmoid 核的零系数。其他核函数忽略此参数。
    # kernel_params : dict of str to any, default=None
    # 参数（关键字参数）和值，用于传递给可调用对象的核函数。其他核函数会忽略此参数。

    # n_jobs : int, default=None
    # 当 `affinity='nearest_neighbors'` 或 `affinity='precomputed_nearest_neighbors'` 时，运行的并行作业数。
    # 邻居搜索将并行进行。
    # ``None`` 表示 1，除非在 :obj:`joblib.parallel_backend` 上下文中。
    # ``-1`` 表示使用所有处理器。详见 :term:`Glossary <n_jobs>`。

    # verbose : bool, default=False
    # 冗余模式。

    # Attributes
    # ----------
    # affinity_matrix_ : array-like of shape (n_samples, n_samples)
    # 用于聚类的相似度矩阵。仅在调用 ``fit`` 后可用。

    # labels_ : ndarray of shape (n_samples,)
    # 每个点的标签

    # n_features_in_ : int
    # 在 :term:`fit` 期间看到的特征数。

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    # 在 `X` 具有全部为字符串的特征名称时定义。仅在 `fit` 期间可用。

    # See Also
    # --------
    # sklearn.cluster.KMeans : K-Means 聚类。
    # sklearn.cluster.DBSCAN : 带有噪声的基于密度的空间聚类应用程序。

    # Notes
    # -----
    # 距离矩阵，其中 0 表示相同元素，较高值表示非常不相似元素，可以通过应用高斯（也称为 RBF，热）核转换为适合算法的亲和/相似度矩阵::
    #
    #     np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
    #
    # 这里 ``delta`` 是表示高斯核宽度的自由参数。
    #
    # 另一种方法是采用点的 k-最近邻连接矩阵的对称版本。

    # 如果安装了 pyamg 包，则会使用它：这大大加快了计算速度。

    # References
    # ----------
    # .. [1] :doi:`Normalized cuts and image segmentation, 2000
    #        Jianbo Shi, Jitendra Malik
    #        <10.1109/34.868688>`
    #
    # .. [2] :doi:`A Tutorial on Spectral Clustering, 2007
    #        Ulrike von Luxburg
    #        <10.1007/s11222-007-9033-z>`
    #
    # .. [3] `Multiclass spectral clustering, 2003
    #        Stella X. Yu, Jianbo Shi
    #        <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`
    #
    # .. [4] :doi:`Toward the Optimal Preconditioned Eigensolver:
    #        Locally Optimal Block Preconditioned Conjugate Gradient Method, 2001
    #        A. V. Knyazev
    #        SIAM Journal on Scientific Computing 23, no. 2, pp. 517-541.
    #        <10.1137/S1064827500366124>`
    #
    # .. [5] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
    #        Anil Damle, Victor Minden, Lexing Ying
    #        <10.1093/imaiai/iay008>`
    Examples
    --------
    >>> from sklearn.cluster import SpectralClustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralClustering(n_clusters=2,
    ...         assign_labels='discretize',
    ...         random_state=0).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering
    SpectralClustering(assign_labels='discretize', n_clusters=2,
        random_state=0)
    """
    
    # 定义参数约束字典，用于验证和限制参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],  # 簇的数量，必须是大于等于1的整数
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],  # 特征值求解器，可以是指定的字符串集合或None
        "n_components": [Interval(Integral, 1, None, closed="left"), None],  # 特征向量的数量，必须是大于等于1的整数或None
        "random_state": ["random_state"],  # 随机数种子或None
        "n_init": [Interval(Integral, 1, None, closed="left")],  # 聚类算法的初始化次数，必须是大于等于1的整数
        "gamma": [Interval(Real, 0, None, closed="left")],  # 核函数参数，必须是大于等于0的实数
        "affinity": [
            callable,  # 亲和矩阵计算函数，可以是可调用对象
            StrOptions(  # 或者指定的字符串集合
                set(KERNEL_PARAMS)
                | {"nearest_neighbors", "precomputed", "precomputed_nearest_neighbors"}
            ),
        ],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],  # 最近邻数，必须是大于等于1的整数
        "eigen_tol": [
            Interval(Real, 0.0, None, closed="left"),  # 特征值求解容差，必须是大于等于0.0的实数
            StrOptions({"auto"}),  # 或者是字符串"auto"
        ],
        "assign_labels": [StrOptions({"kmeans", "discretize", "cluster_qr"})],  # 聚类标签分配方法，必须是指定的字符串集合之一
        "degree": [Interval(Real, 0, None, closed="left")],  # 多项式核函数的阶数，必须是大于等于0的实数
        "coef0": [Interval(Real, None, None, closed="neither")],  # 核函数常数项，可以是任意实数
        "kernel_params": [dict, None],  # 核函数的额外参数，可以是字典或None
        "n_jobs": [Integral, None],  # 并行计算的作业数，必须是整数或None
        "verbose": ["verbose"],  # 是否输出详细信息，可以是字符串"verbose"
    }
    
    def __init__(
        self,
        n_clusters=8,
        *,
        eigen_solver=None,
        n_components=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol="auto",
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
    ):
        # 初始化函数，设置各个参数的初始值
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    @_fit_context(prefer_skip_nested_validation=True)
    # 执行谱聚类算法并返回聚类标签

    def fit_predict(self, X, y=None):
        """Perform spectral clustering on `X` and return cluster labels.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, similarities / affinities between
            instances if ``affinity='precomputed'``, or distances between
            instances if ``affinity='precomputed_nearest_neighbors``. If a
            sparse matrix is provided in a format other than ``csr_matrix``,
            ``csc_matrix``, or ``coo_matrix``, it will be converted into a
            sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        # 调用父类的 `fit_predict` 方法进行谱聚类，并返回聚类结果
        return super().fit_predict(X, y)

    # 返回一个字典，指示是否支持成对数据（pairwise）的操作
    def _more_tags(self):
        return {
            "pairwise": self.affinity
            in [
                "precomputed",
                "precomputed_nearest_neighbors",
            ]
        }
```