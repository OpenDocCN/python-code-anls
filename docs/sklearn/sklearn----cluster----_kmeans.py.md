# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_kmeans.py`

```
# K-means clustering.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入抽象基类模块
from abc import ABC, abstractmethod
# 导入数字类型检查模块
from numbers import Integral, Real

# 导入科学计算库numpy和稀疏矩阵处理模块scipy.sparse
import numpy as np
import scipy.sparse as sp

# 导入基本估计器类、特征前缀名称混合类、聚类混合类、转换器混合类、拟合上下文工具函数
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
# 导入迭代过程中的收敛警告异常
from ..exceptions import ConvergenceWarning
# 导入距离计算函数和欧几里得距离函数
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
# 导入数组检查函数、随机数种子检查函数
from ..utils import check_array, check_random_state
# 导入OpenMP线程数控制函数
from ..utils._openmp_helpers import _openmp_effective_n_threads
# 导入参数验证函数Interval、StrOptions和参数验证函数
from ..utils._param_validation import Interval, StrOptions, validate_params
# 导入行规范化函数和稳定累积和函数
from ..utils.extmath import row_norms, stable_cumsum
# 导入并行计算控制器、线程池控制器装饰器函数
from ..utils.parallel import (
    _get_threadpool_controller,
    _threadpool_controller_decorator,
)
# 导入稀疏矩阵函数均值和方差计算函数
from ..utils.sparsefuncs import mean_variance_axis
# 导入CSR格式稀疏矩阵行分配函数
from ..utils.sparsefuncs_fast import assign_rows_csr
# 导入验证函数集合：样本权重检查、数组类检查非标量、拟合检查函数
from ..utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_is_fitted,
)
# 导入K均值聚类通用函数：块大小、密集簇内惯性计算函数、稀疏簇内惯性计算函数、判定相同聚类函数
from ._k_means_common import (
    CHUNK_SIZE,
    _inertia_dense,
    _inertia_sparse,
    _is_same_clustering,
)
# 导入Elkan算法迭代处理密集簇函数、稀疏簇函数、初始化界限密集函数、初始化界限稀疏函数
from ._k_means_elkan import (
    elkan_iter_chunked_dense,
    elkan_iter_chunked_sparse,
    init_bounds_dense,
    init_bounds_sparse,
)
# 导入Lloyd算法迭代处理密集簇函数、稀疏簇函数
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
# 导入K均值小批量更新密集簇函数、稀疏簇函数
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse

###############################################################################
# Initialization heuristic

# 参数验证装饰器，验证输入参数格式
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "sample_weight": ["array-like", None],
        "x_squared_norms": ["array-like", None],
        "random_state": ["random_state"],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
# K均值聚类初始化函数，使用k-means++算法选择种子点
def kmeans_plusplus(
    X,
    n_clusters,
    *,
    sample_weight=None,
    x_squared_norms=None,
    random_state=None,
    n_local_trials=None,
):
    """Init n_clusters seeds according to k-means++.

    .. versionadded:: 0.24

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.

    n_clusters : int
        The number of centroids to initialize.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is ignored if `init`
        is a callable or a user provided array.

        .. versionadded:: 1.3

    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.
    random_state : int or RandomState instance, default=None
        确定用于质心初始化的随机数生成。传递一个整数以在多次函数调用间生成可重复的输出。
        参见 :term:`词汇表 <random_state>`。

    n_local_trials : int, default=None
        每个质心（除第一个外）的种子试验次数，选择最大程度减少惯性的那个。
        设置为 None 以使试验次数对种子数（2+log(k)）呈对数增长，这是推荐的设置。
        将其设置为 1 可禁用贪婪聚类选择，并恢复经验上显示效果不如其贪婪变体的标准 k-means++ 算法。

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        k-means 的初始质心。

    indices : ndarray of shape (n_clusters,)
        数据数组 X 中所选质心的索引位置。对于给定的索引和质心，X[index] = center。

    Notes
    -----
    以智能方式选择 k-means 聚类的初始聚类中心，加快收敛速度。参见：Arthur, D. 和 Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Examples
    --------

    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  2],
           [ 1,  0]])
    >>> indices
    array([3, 2])
    """
    # 检查数据
    check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    if X.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
        )

    # 检查参数
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}."
        )

    random_state = check_random_state(random_state)

    # 调用私有的 k-means++ 算法
    centers, indices = _kmeans_plusplus(
        X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials
    )

    return centers, indices
    # 根据 k-means++ 算法初始化聚类中心的计算组件
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    # 创建一个空的数组，用于存储聚类中心点
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # 如果未指定每个中心点的本地试验次数，则默认为 2 + log(k)
    if n_local_trials is None:
        # 这是Arthur/Vassilvitskii尝试过的方法，但除了在结论中提到有帮助外，未报告其他具体结果
        n_local_trials = 2 + int(np.log(n_clusters))

    # 随机选择第一个聚类中心，并记录其索引
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)

    # 根据数据稀疏性选择第一个聚类中心的数据点，并将其存储在centers数组中
    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # 初始化最近距离列表，并计算当前的潜在成本
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq @ sample_weight

    # 选择剩余的 n_clusters-1 个聚类中心点
    # 对于每一个聚类中心的索引，执行以下操作
    for c in range(1, n_clusters):
        # 根据距离最近的现有中心点的平方距离来以比例采样选择中心候选点
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        # 根据累积加权和确定采样后的候选点索引
        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        # XXX: 数值不精确可能导致候选点索引超出范围
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # 计算到候选中心点的距离
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # 更新每个候选点的最近距离平方和潜力
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # 决定哪个候选点是最佳的中心点
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # 将在局部尝试中找到的最佳中心候选点永久添加
        if sp.issparse(X):
            centers[c] = X[[best_candidate]].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


这段代码实现了 K-means++ 算法中的中心点选择过程。
###############################################################################
# K-means batch estimation by EM (expectation maximization)

# 定义一个函数 _tolerance，用于计算与数据集相关的容差值
def _tolerance(X, tol):
    """Return a tolerance which is dependent on the dataset."""
    # 如果容差值为 0，则直接返回 0
    if tol == 0:
        return 0
    # 如果输入的 X 是稀疏矩阵，则计算每列的方差
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        # 否则，计算每列的方差
        variances = np.var(X, axis=0)
    # 返回方差的平均值乘以容差值作为最终的容差
    return np.mean(variances) * tol


# 使用装饰器 validate_params 对 k_means 函数的参数进行验证
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
        "return_n_iter": [bool],
    },
    prefer_skip_nested_validation=False,
)
# 定义 K-means 聚类算法的主函数 k_means
def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
):
    """Perform K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is not used during
        initialization if `init` is a callable or a user provided array.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        - `'k-means++'` : selects initial cluster centers for k-mean
          clustering in a smart way to speed up convergence. See section
          Notes in k_init for more details.
        - `'random'`: choose `n_clusters` observations (rows) at random from data
          for the initial centroids.
        - If an array is passed, it should be of shape `(n_clusters, n_features)`
          and gives the initial centers.
        - If a callable is passed, it should take arguments `X`, `n_clusters` and a
          random state and return an initialization.
    n_init : 'auto' or int, default="auto"
        # K-means算法的初始中心点种子数量。算法将使用不同的种子运行n_init次，最终选择inertia（惯性）最小的一次作为输出结果。
        # 当 n_init='auto' 时，运行次数取决于init的值：
        # - 如果使用 'init='random'' 或者 init 是可调用对象，则运行10次。
        # - 如果使用 'init='k-means++'' 或者 init 是类似数组，则运行1次。
        # 在版本1.2中添加了 'auto' 选项。
        # 在版本1.4中，默认值改为 'auto'。

    max_iter : int, default=300
        # K-means算法的最大迭代次数。

    verbose : bool, default=False
        # 冗余模式开关。如果为True，将输出详细的运行信息。

    tol : float, default=1e-4
        # 相对容差值，用于判断两次迭代之间的聚类中心差异是否足够小，以宣布收敛。

    random_state : int, RandomState instance or None, default=None
        # 用于初始化中心点的随机数生成器。使用整数以使随机性确定性。

    copy_x : bool, default=True
        # 在预先计算距离时，先对数据进行居中操作，可以提高数值精度。
        # 如果 copy_x=True（默认），则不修改原始数据。
        # 如果为False，则修改原始数据，并在函数返回前重新放置，但通过减去和添加数据的均值可能会引入小的数值差异。
        # 注意，如果原始数据不是C连续的，即使 copy_x=False，也会进行复制。如果原始数据是稀疏的但不是CSR格式，即使 copy_x=False，也会进行复制。

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        # 使用的K-means算法。
        # "lloyd"：经典的EM风格算法。
        # "elkan"：通过使用三角不等式，在某些具有明确定义聚类的数据集上可能更有效率。但由于额外分配了一个形状为 (n_samples, n_clusters) 的数组，因此更占用内存。
        # 在版本0.18中添加了Elkan算法。
        # 在版本1.1中，将"full"重命名为"lloyd"，弃用了"auto"和"full"。改变了"auto"以使用"lloyd"而不是"elkan"。

    return_n_iter : bool, default=False
        # 是否返回迭代次数。

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        # K-means算法在最后一次迭代找到的聚类中心点。

    label : ndarray of shape (n_samples,)
        # label[i] 是距离第i个观察值最近的聚类中心的代码或索引。
    # inertia : float
    #     训练集中所有观测点到最近质心的距离平方和的最终值，表示聚类结果的质量。

    # best_n_iter : int
    #     最佳聚类结果对应的迭代次数。仅在 `return_n_iter` 设置为 True 时返回。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import k_means
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centroid, label, inertia = k_means(
    ...     X, n_clusters=2, n_init="auto", random_state=0
    ... )
    >>> centroid
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> label
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> inertia
    16.0
    """
    # 使用 KMeans 对象进行聚类
    est = KMeans(
        n_clusters=n_clusters,       # 聚类的簇数
        init=init,                   # 初始化方法
        n_init=n_init,               # 运行 k-means 算法的次数
        max_iter=max_iter,           # 单次运行的最大迭代次数
        verbose=verbose,             # 控制详细程度的整数值
        tol=tol,                     # 容忍的迭代停止条件
        random_state=random_state,   # 随机数生成器的种子
        copy_x=copy_x,               # 是否复制数据，默认为 True
        algorithm=algorithm,         # 使用的 k-means 算法
    ).fit(X, sample_weight=sample_weight)  # 对输入数据 X 进行聚类，可选的样本权重

    # 如果设置了 return_n_iter，则返回额外的迭代次数信息
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
    """Initialize the k-means clustering process using the Elkan algorithm.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main Cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_samples = X.shape[0]  # Number of samples
    n_clusters = centers_init.shape[0]  # Number of clusters

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init  # Initialize centers
    centers_new = np.zeros_like(centers)  # New centers buffer
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)  # Weight in clusters
    labels = np.full(n_samples, -1, dtype=np.int32)  # Labels for each sample
    labels_old = labels.copy()  # Copy of labels for convergence check
    center_half_distances = euclidean_distances(centers) / 2  # Half distances between centers
    distance_next_center = np.partition(
        np.asarray(center_half_distances), kth=1, axis=0
    )[1]  # Distance to next closest center
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)  # Upper bounds
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)  # Lower bounds
    center_shift = np.zeros(n_clusters, dtype=X.dtype)  # Center shift

    if sp.issparse(X):
        init_bounds = init_bounds_sparse  # Use sparse initialization function
        elkan_iter = elkan_iter_chunked_sparse  # Use sparse iteration function
        _inertia = _inertia_sparse  # Use sparse inertia calculation
    else:
        init_bounds = init_bounds_dense  # Use dense initialization function
        elkan_iter = elkan_iter_chunked_dense  # Use dense iteration function
        _inertia = _inertia_dense  # Use dense inertia calculation

    # Initialize bounds based on data density
    init_bounds(
        X,
        centers,
        center_half_distances,
        labels,
        upper_bounds,
        lower_bounds,
        n_threads=n_threads,
    )

    strict_convergence = False  # Flag indicating strict convergence check
    # 循环执行 K-means 算法的最大迭代次数
    for i in range(max_iter):
        # 执行 Elkan 算法的一个迭代步骤，更新中心点和标签
        elkan_iter(
            X,  # 数据集
            sample_weight,  # 样本权重
            centers,  # 当前中心点位置
            centers_new,  # 新的中心点位置（将在本次迭代中更新）
            weight_in_clusters,  # 聚类中心的权重
            center_half_distances,  # 中心点到最近其他中心的距离的一半
            distance_next_center,  # 每个中心点到下一个迭代中最近中心点的距离
            upper_bounds,  # 上界
            lower_bounds,  # 下界
            labels,  # 当前的聚类标签
            center_shift,  # 中心点的移动距离
            n_threads,  # 并行线程数
        )

        # 计算新的中心点之间的半欧几里得距离
        center_half_distances = euclidean_distances(centers_new) / 2
        # 找出每个中心点到其最近的其他中心点的距离
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0
        )[1]

        # 如果设置了详细输出
        if verbose:
            # 计算当前聚类效果的惯性
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}")

        # 交换中心点的位置，准备下一轮迭代
        centers, centers_new = centers_new, centers

        # 如果当前的标签和上一次迭代的标签完全一致，则认为严格收敛
        if np.array_equal(labels, labels_old):
            # 如果设置了详细输出，显示严格收敛信息
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # 如果没有严格收敛，检查是否基于公差(tol)收敛
            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot <= tol:
                # 如果设置了详细输出，显示基于公差收敛的信息
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        # 更新上一次的标签，用于下一轮迭代的比较
        labels_old[:] = labels

    # 如果没有严格收敛，则重新运行 E 步骤以确保预测标签与聚类中心匹配
    if not strict_convergence:
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    # 计算最终的惯性
    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    # 返回最终的聚类标签、惯性、中心点和迭代次数
    return labels, inertia, centers, i + 1
# 使用 `_threadpool_controller_decorator` 装饰器控制线程池上下文，限制第二级并行（例如 BLAS）中的线程数量，以避免过度订阅。
@_threadpool_controller_decorator(limits=1, user_api="blas")
# 定义 `_kmeans_single_lloyd` 函数，执行 k-means Lloyd 算法的单次运行，假设已经完成预处理。
def _kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    # 获取聚类中心的数量
    n_clusters = centers_init.shape[0]

    # 为避免在每次迭代中重新分配内存，创建缓冲区
    centers = centers_init  # 初始化聚类中心
    centers_new = np.zeros_like(centers)  # 创建一个与 centers 结构相同的全零数组
    labels = np.full(X.shape[0], -1, dtype=np.int32)  # 创建一个与样本数量相同的全为 -1 的整数数组，用于存储样本的类别标签
    labels_old = labels.copy()  # 复制 labels，用于迭代过程中跟踪旧的标签状态
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)  # 创建一个与聚类中心数量相同的全零数组，用于存储每个聚类中心的加权和
    center_shift = np.zeros(n_clusters, dtype=X.dtype)  # 创建一个与聚类中心数量相同的全零数组，用于存储每个聚类中心的移动量

    # 根据输入矩阵的稀疏性选择适当的迭代函数和惯性计算函数
    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse  # 如果 X 是稀疏矩阵，则使用稀疏数据的迭代器
        _inertia = _inertia_sparse  # 如果 X 是稀疏矩阵，则使用稀疏数据的惯性计算函数
    else:
        lloyd_iter = lloyd_iter_chunked_dense  # 如果 X 是稠密矩阵，则使用稠密数据的迭代器
        _inertia = _inertia_dense  # 如果 X 是稠密矩阵，则使用稠密数据的惯性计算函数

    strict_convergence = False  # 初始化严格收敛标志为 False，表示不进行严格收敛判定
    # 对于给定的最大迭代次数执行迭代聚类算法
    for i in range(max_iter):
        # 执行 Lloyd 算法的一次迭代，更新聚类中心和标签
        lloyd_iter(
            X,                  # 数据集
            sample_weight,      # 样本权重
            centers,            # 当前聚类中心
            centers_new,        # 新计算的聚类中心
            weight_in_clusters, # 聚类中心的样本权重
            labels,             # 当前的数据点标签
            center_shift,       # 聚类中心的偏移量
            n_threads,          # 使用的线程数
        )

        # 如果 verbose 为真，则计算当前惯性并输出迭代信息
        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}.")

        # 交换当前聚类中心和新计算的聚类中心的引用
        centers, centers_new = centers_new, centers

        # 检查当前标签是否与上一次迭代的标签相同，若相同则判断为严格收敛
        if np.array_equal(labels, labels_old):
            # 若 verbose 为真，则输出严格收敛的信息
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # 若未严格收敛，则检查是否满足指定的容差条件
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                # 若 verbose 为真，则输出容差收敛的信息
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        # 更新 labels_old 数组，记录当前的标签状态以便下一次迭代比较
        labels_old[:] = labels

    # 如果没有严格收敛，重新运行 E 步骤以匹配预测标签与聚类中心
    if not strict_convergence:
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    # 计算最终的惯性并返回聚类结果、惯性值、最终聚类中心和迭代次数
    inertia = _inertia(X, sample_weight, centers, labels, n_threads)
    return labels, inertia, centers, i + 1
# 定义一个函数 `_labels_inertia`，用于执行 K-means EM 算法的 E 步骤
def _labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    # 获取样本数和聚类中心数
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    # 初始化标签数组，全为 -1，表示未分配
    labels = np.full(n_samples, -1, dtype=np.int32)
    
    # 初始化聚类中心的偏移量数组
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)

    # 根据输入数据的稀疏性选择相应的函数来执行迭代计算标签
    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse  # 使用稀疏数据版本的标签迭代函数
        _inertia = _inertia_sparse  # 使用稀疏数据版本的惯性计算函数
    else:
        _labels = lloyd_iter_chunked_dense  # 使用密集数据版本的标签迭代函数
        _inertia = _inertia_dense  # 使用密集数据版本的惯性计算函数

    # 调用标签迭代函数计算标签
    _labels(
        X,
        sample_weight,
        centers,
        centers_new=None,
        weight_in_clusters=None,
        labels=labels,
        center_shift=center_shift,
        n_threads=n_threads,
        update_centers=False,
    )

    # 如果需要计算惯性，调用惯性计算函数
    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return labels, inertia

    # 返回计算得到的标签
    return labels


# 定义 `_labels_inertia` 函数的线程池限制版本装饰器
_labels_inertia_threadpool_limit = _threadpool_controller_decorator(
    limits=1, user_api="blas"
)(_labels_inertia)


# 定义 `_BaseKMeans` 类，作为 KMeans 和 MiniBatchKMeans 的基类
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):
    """Base class for KMeans and MiniBatchKMeans"""

    # 参数约束字典，限定了 KMeans 类的参数类型和取值范围
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }
    def __init__(
        self,
        n_clusters,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
    ):
        # 初始化方法，设置聚类的参数
        self.n_clusters = n_clusters  # 设置聚类的簇数
        self.init = init  # 设置初始化中心的方法（如"k-means++"或"random"）
        self.max_iter = max_iter  # 设置最大迭代次数
        self.tol = tol  # 设置迭代停止的容忍度
        self.n_init = n_init  # 设置聚类的初始尝试次数
        self.verbose = verbose  # 设置详细程度
        self.random_state = random_state  # 设置随机种子

    def _check_params_vs_input(self, X, default_n_init=None):
        # 检查参数与输入数据的关系
        # 检查样本数量是否大于等于簇数
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # 计算容忍度的实际值
        self._tol = _tolerance(X, self.tol)

        # 处理初始尝试次数的设定
        if self.n_init == "auto":
            # 根据初始化方法设定初始尝试次数
            if isinstance(self.init, str) and self.init == "k-means++":
                self._n_init = 1
            elif isinstance(self.init, str) and self.init == "random":
                self._n_init = default_n_init
            elif callable(self.init):
                self._n_init = default_n_init
            else:  # array-like
                self._n_init = 1
        else:
            self._n_init = self.n_init

        # 如果初始中心位置为数组且初始尝试次数不为1，则发出警告
        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only"
                    f" one init in {self.__class__.__name__} instead of "
                    f"n_init={self._n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1

    @abstractmethod
    def _warn_mkl_vcomp(self, n_active_threads):
        """Issue an estimator specific warning when vcomp and mkl are both present

        This method is called by `_check_mkl_vcomp`.
        """

    def _check_mkl_vcomp(self, X, n_samples):
        """Check when vcomp and mkl are both present"""
        # 检查是否同时存在 vcomp 和 mkl
        # 在稠密数据的 lloyd_iter_chunked_dense 中的 prange 内的 BLAS 调用会导致小内存泄漏
        # 仅当可用线程数小于划分块数时才会发生，这种情况下 OpenMP 库为 vcomp，BLAS 库为 MKL
        if sp.issparse(X):
            return

        # 计算活跃线程数
        n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        
        # 如果活跃线程数小于已设定的线程数，则检查线程池的信息
        if n_active_threads < self._n_threads:
            modules = _get_threadpool_controller().info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules
            ]
            # 如果同时存在 vcomp 和 mkl，则发出警告
            if has_vcomp and has_mkl:
                self._warn_mkl_vcomp(n_active_threads)
    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        # 检查初始中心点 centers 是否与 X 和 n_clusters 兼容
        if centers.shape[0] != self.n_clusters:
            # 如果 centers 的行数不等于 n_clusters，抛出数值错误异常
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        # 检查 centers 的列数是否与 X 的特征数相匹配
        if centers.shape[1] != X.shape[1]:
            # 如果 centers 的列数不等于 X 的特征数，抛出数值错误异常
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        """Validate and preprocess test data X."""
        # 使用 _validate_data 方法验证和预处理测试数据 X
        X = self._validate_data(
            X,
            accept_sparse="csr",        # 接受稀疏矩阵格式为 csr
            reset=False,                # 不重置输入数据 X
            dtype=[np.float64, np.float32],  # 数据类型限定为 np.float64 或 np.float32
            order="C",                  # 数组在内存中的存储顺序为 C 风格
            accept_large_sparse=False,  # 不接受大型稀疏矩阵
        )
        return X

    def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        sample_weight,
        init_size=None,
        n_centroids=None,
        """
        Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X. `sample_weight` is not used
            during initialization if `init` is a callable or a user provided
            array.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
        # 获取样本数量
        n_samples = X.shape[0]
        # 获取聚类数量，如果未指定，则使用默认值 self.n_clusters
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        # 如果指定了 init_size 并且小于样本数量，则随机抽样一部分样本用于初始化
        if init_size is not None and init_size < n_samples:
            # 从样本中随机选择 init_size 个索引
            init_indices = random_state.randint(0, n_samples, init_size)
            # 根据选定的索引重新设置 X，x_squared_norms 和 sample_weight
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
            sample_weight = sample_weight[init_indices]

        # 根据不同的初始化方法进行初始化
        if isinstance(init, str) and init == "k-means++":
            # 使用 k-means++ 方法初始化中心点
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
            )
        elif isinstance(init, str) and init == "random":
            # 使用随机方法初始化中心点
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
                p=sample_weight / sample_weight.sum(),
            )
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            # 如果 init 是数组类型，则直接使用其作为中心点
            centers = init
        elif callable(init):
            # 如果 init 是可调用对象，则调用它来获取中心点
            centers = init(X, n_clusters, random_state=random_state)
            # 确保中心点数组的数据类型和顺序与输入的 X 一致
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            # 验证中心点的形状是否正确
            self._validate_center_shape(X, centers)

        # 如果中心点是稀疏矩阵，则转换为稠密数组
        if sp.issparse(centers):
            centers = centers.toarray()

        # 返回初始化得到的中心点
        return centers
    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Convenience method for fitting the model on X and predicting cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # Ensure the model is fitted before predicting
        check_is_fitted(self)

        # Validate and preprocess the input data X
        X = self._check_test_data(X)

        # Prepare sample weights; not directly used in prediction
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

        # Compute labels using optimized helper function
        labels = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            self.cluster_centers_,
            n_threads=self._n_threads,
            return_inertia=False,
        )

        return labels

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Computes clustering on X and transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        # Fit the model and transform X in one step for efficiency
        return self.fit(X, sample_weight=sample_weight)._transform(X)
    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        # Ensure the estimator is fitted before transformation
        check_is_fitted(self)

        # Ensure X is compatible with the trained model
        X = self._check_test_data(X)

        # Perform the transformation using the internal method
        return self._transform(X)

    def _transform(self, X):
        """
        Guts of transform method; no input validation.
        """
        # Compute Euclidean distances from X to cluster centers
        return euclidean_distances(X, self.cluster_centers_)

    def score(self, X, y=None, sample_weight=None):
        """
        Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        # Ensure the estimator is fitted before scoring
        check_is_fitted(self)

        # Ensure X is compatible with the trained model
        X = self._check_test_data(X)

        # Validate and adjust sample weights if provided
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Compute scores using a helper function
        _, scores = _labels_inertia_threadpool_limit(
            X, sample_weight, self.cluster_centers_, self._n_threads
        )

        # Return the negative score as per K-means objective
        return -scores

    def _more_tags(self):
        """
        Additional tags method.

        Returns
        -------
        dict
            Dictionary containing additional tags.
        """
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
        }
class KMeans(_BaseKMeans):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling \
            based on an empirical probability distribution of the points' \
            contribution to the overall inertia. This technique speeds up \
            convergence. The algorithm implemented is "greedy k-means++". It \
            differs from the vanilla k-means++ by making several trials at \
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from \
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)\
        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters and a\
        random state and return an initialization.

        For an example of how to use the different `init` strategy, see the example
        entitled :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    """
    copy_x : bool, default=True
        当预先计算距离时，先对数据进行居中处理可以提高数值精度。如果 copy_x 为 True（默认值），则不修改原始数据。
        如果为 False，则修改原始数据，并在函数返回前恢复原始状态，但通过减去然后再加上数据均值可能会引入小的数值差异。
        注意，如果原始数据不是 C 连续的，即使 copy_x 为 False，也会进行复制。如果原始数据是稀疏的但不是 CSR 格式，
        即使 copy_x 为 False，也会进行复制。

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        使用的 K-means 算法。经典的EM风格算法是 "lloyd"。"elkan" 变体对一些具有明确定义聚类的数据集可能更有效，
        因为利用了三角不等式。但由于要分配一个额外的数组，形状为 `(n_samples, n_clusters)`，因此内存消耗更大。

        .. versionchanged:: 0.18
            添加了 Elkan 算法

        .. versionchanged:: 1.1
            将 "full" 更名为 "lloyd"，并弃用了 "auto" 和 "full"。
            将 "auto" 改为使用 "lloyd" 而不是 "elkan"。

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        聚类中心的坐标。如果算法在完全收敛之前停止（参见 ``tol`` 和 ``max_iter``），这些中心点可能与 ``labels_`` 不一致。

    labels_ : ndarray of shape (n_samples,)
        每个样本点的标签。

    inertia_ : float
        样本点到它们最近的聚类中心的加权平方距离之和，如果提供了样本权重则加权。

    n_iter_ : int
        运行的迭代次数。

    n_features_in_ : int
        在拟合期间观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合期间观察到的特征名称。仅当 `X` 的特征名称都是字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : 另一种在线实现方法，使用小批量更新聚类中心位置。
        对于大规模学习（例如 n_samples > 10k），MiniBatchKMeans 可能比默认的批处理实现更快。

    Notes
    -----
    K-means 问题可以使用 Lloyd 或 Elkan 算法解决。

    平均复杂度由 O(k n T) 给出，其中 n 是样本数，T 是迭代次数。

    最坏情况复杂度由 O(n^(k+2/p)) 给出，其中 n = n_samples，p = n_features。
    更多细节参见 :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>`。

    在实践中，K-means 算法非常快速（是最快的之一）。
    """
    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "copy_x": ["boolean"],
        "algorithm": [StrOptions({"lloyd", "elkan"})],
    }
    """
    # 定义参数约束字典，继承自_BaseKMeans类的参数约束，并增加了copy_x和algorithm两个参数的类型约束

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        """
        # 初始化函数，设置K均值聚类算法的参数

        Parameters
        ----------
        n_clusters : int, default=8
            聚类的数量。
        init : {'k-means++', 'random', ndarray, callable}, default='k-means++'
            初始化聚类中心的方法。
        n_init : int or None, default='auto'
            进行初始化聚类中心时的重复次数。
        max_iter : int, default=300
            每次聚类迭代的最大次数。
        tol : float, default=1e-4
            算法收敛的容忍度。
        verbose : int, default=0
            控制输出详细程度。
        random_state : int, RandomState instance or None, default=None
            控制随机数生成。
        copy_x : bool, default=True
            是否复制输入数据。
        algorithm : {'lloyd', 'elkan'}, default='lloyd'
            使用的算法类型。

        Attributes
        ----------
        copy_x : bool
            是否复制输入数据。
        algorithm : {'lloyd', 'elkan'}
            使用的算法类型。
        """
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

        self.copy_x = copy_x  # 初始化self.copy_x参数
        self.algorithm = algorithm  # 初始化self.algorithm参数

    def _check_params_vs_input(self, X):
        """
        # 检查参数与输入数据的一致性，继承自父类_BaseKMeans的方法

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入数据。

        Raises
        ------
        RuntimeWarning
            如果使用'elkan'算法且聚类数为1，则会发出警告，并将算法切换为'lloyd'。
        """
        super()._check_params_vs_input(X, default_n_init=10)

        self._algorithm = self.algorithm  # 将self.algorithm赋值给self._algorithm
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                (
                    "algorithm='elkan' doesn't make sense for a single "
                    "cluster. Using 'lloyd' instead."
                ),
                RuntimeWarning,
            )
            self._algorithm = "lloyd"  # 如果使用'elkan'算法且聚类数为1，则切换为'lloyd'
    # 定义一个方法 _warn_mkl_vcomp，用于在特定条件下发出警告消息
    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        # 发出警告消息，提醒用户在特定情况下可能存在内存泄漏问题
        warnings.warn(
            "KMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )

    # 应用装饰器 @_fit_context，指定 prefer_skip_nested_validation=True 参数
    @_fit_context(prefer_skip_nested_validation=True)
# 定义一个函数用于执行 Minibatch K-Means 算法的增量更新步骤。
def _mini_batch_step(
    X,
    sample_weight,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. If sparse, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    # 执行标签分配，将每个样本分配到最近的中心点，并计算惯性(inertia)
    labels, inertia = _labels_inertia(X, sample_weight, centers, n_threads=n_threads)

    # 根据标签更新中心点
    if sp.issparse(X):
        # 如果输入数据是稀疏矩阵，调用稀疏矩阵更新方法
        _minibatch_update_sparse(
            X, sample_weight, centers, centers_new, weight_sums, labels, n_threads
        )
    else:
        # 如果输入数据是密集矩阵，调用密集矩阵更新方法
        _minibatch_update_dense(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_sums,
            labels,
            n_threads,
        )

    # 重新分配权重极低的簇
    # 如果需要随机重新分配并且重新分配比例大于0
    to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

    # 如果需要重新分配的样本数大于总样本数的一半
    if to_reassign.sum() > 0.5 * X.shape[0]:
        # 找出不需要重新分配的索引
        indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]):]
        # 将这些索引对应的重新分配标志设为 False
        to_reassign[indices_dont_reassign] = False
    # 计算需要重新分配的数量
    n_reassigns = to_reassign.sum()

    # 如果有需要重新分配的
    if n_reassigns:
        # 从样本中均匀概率地选择新的聚类中心
        new_centers = random_state.choice(
            X.shape[0], replace=False, size=n_reassigns
        )
        # 如果设置了详细模式，则打印重新分配聚类中心的信息
        if verbose:
            print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

        # 如果数据是稀疏矩阵
        if sp.issparse(X):
            # 使用 CSR 格式的行赋值函数来重新分配聚类中心
            assign_rows_csr(
                X,
                new_centers.astype(np.intp, copy=False),
                np.where(to_reassign)[0].astype(np.intp, copy=False),
                centers_new,
            )
        else:
            # 直接将新的聚类中心赋值给对应的中心点
            centers_new[to_reassign] = X[new_centers]

    # 重置被重新分配中心的权重和，但是不要将它们重置得太小以避免立即重新分配。
    # 这是一种比较粗糙的解决方案，因为它也修改了学习率。
    weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

# 返回惯性（inertia）作为结果
return inertia
class MiniBatchKMeans(_BaseKMeans):
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).
    """
    # 控制基于连续未改善的小批量数量的早期停止，用于平滑惯性。
    # 如果希望禁用基于惯性的收敛检测，将 max_no_improvement 设置为 None。
    max_no_improvement : int, default=10

    # 初始化大小，用于加速初始化过程中的随机抽样数量。
    # 仅算法会在数据的随机子集上运行批量 KMeans 初始化。
    # 这个值需要大于 n_clusters。
    # 如果为 `None`，启发式方法是 `init_size = 3 * batch_size`，
    # 如果 `3 * batch_size < n_clusters`，否则 `init_size = 3 * n_clusters`。
    init_size : int, default=None

    # 尝试的随机初始化次数。
    # 不同于 KMeans，该算法只运行一次，使用惯性最佳的 `n_init` 个初始化之一。
    # 对于稀疏高维问题，推荐多次运行（参见 `kmeans_sparse_high_dim`）。
    # 当 `n_init='auto'` 时，运行次数取决于 init 的值：
    # 如果使用 `init='random'` 或 `init` 是可调用的，则为 3；
    # 如果使用 `init='k-means++'` 或 `init` 是类似数组，则为 1。
    n_init : 'auto' or int, default="auto"

    # 控制中心重新分配的最大计数分数的分数。
    # 较高的值意味着低计数中心更容易重新分配，这导致模型收敛时间更长，
    # 但应该能得到更好的聚类结果。然而，过高的值可能导致收敛问题，特别是在小批量大小的情况下。
    reassignment_ratio : float, default=0.01

    # 聚类中心的坐标数组，形状为 (n_clusters, n_features)。
    cluster_centers_ : ndarray of shape (n_clusters, n_features)

    # 每个点的标签数组，形状为 (n_samples,)，如果 compute_labels 设置为 True。
    labels_ : ndarray of shape (n_samples,)

    # 如果 compute_labels 设置为 True，则与选择的分区相关联的惯性标准的值。
    # 如果 compute_labels 设置为 False，则它是基于批次惯性的指数加权平均的惯性近似值。
    # 惯性定义为样本到其聚类中心的平方距离之和，如果提供了样本权重，则加权。
    inertia_ : float

    # 完整数据集的迭代次数。
    n_iter_ : int

    # 处理的小批量数。
    n_steps_ : int

    # 在 `fit` 过程中看到的特征数量。
    n_features_in_ : int
    # `_parameter_constraints`是一个字典，用于定义`MiniBatchKMeans`的参数约束条件，继承自`_BaseKMeans._parameter_constraints`
    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,  # 继承基类 `_BaseKMeans` 的参数约束条件
        "batch_size": [Interval(Integral, 1, None, closed="left")],  # 批处理大小必须是大于等于1的整数
        "compute_labels": ["boolean"],  # compute_labels参数必须是布尔值
        "max_no_improvement": [Interval(Integral, 0, None, closed="left"), None],  # 最大不改进次数必须是大于等于0的整数或者无限制
        "init_size": [Interval(Integral, 1, None, closed="left"), None],  # 初始大小必须是大于等于1的整数或者无限制
        "reassignment_ratio": [Interval(Real, 0, None, closed="left")],  # 重新分配比率必须是大于等于0的实数
    }
    
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        max_iter=100,
        batch_size=1024,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init="auto",
        reassignment_ratio=0.01,
    ):
        # 调用父类的初始化方法，设置聚类数、初始化方法、最大迭代次数、详细程度、随机种子、容忍度和初始聚类次数
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        # 设置最大不改善次数、批处理大小、是否计算标签、初始化大小和重新分配比率
        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    def _check_params_vs_input(self, X):
        # 调用父类的参数检查方法，验证输入数据和默认初始化次数为3
        super()._check_params_vs_input(X, default_n_init=3)

        # 设置批处理大小，取批处理大小和样本数据行数的较小值
        self._batch_size = min(self.batch_size, X.shape[0])

        # 初始化大小
        self._init_size = self.init_size
        if self._init_size is None:
            # 如果初始化大小为None，则设置为3倍的批处理大小
            self._init_size = 3 * self._batch_size
            # 如果初始化大小小于聚类数，则设置为3倍的聚类数
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            # 如果初始化大小小于聚类数，则发出警告
            warnings.warn(
                (
                    f"init_size={self._init_size} should be larger than "
                    f"n_clusters={self.n_clusters}. Setting it to "
                    "min(3*n_clusters, n_samples)"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._init_size = 3 * self.n_clusters
        # 将初始化大小限制在样本数据行数范围内
        self._init_size = min(self._init_size, X.shape[0])

        # 重新分配比率
        if self.reassignment_ratio < 0:
            # 如果重新分配比率小于0，则引发值错误
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        # 发出警告，提示在Windows下使用MKL时可能存在内存泄漏问题
        warnings.warn(
            "MiniBatchKMeans is known to have a memory leak on "
            "Windows with MKL, when there are less chunks than "
            "available threads. You can prevent it by setting "
            f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
            "setting the environment variable "
            f"OMP_NUM_THREADS={n_active_threads}"
        )

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        """
        Helper function to encapsulate the early stopping logic
        """
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self._batch_size

        # count steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because it's inertia from initialization.
        if step == 1:
            if self.verbose:
                print(
                    f"Minibatch step {step}/{n_steps}: mean batch "
                    f"inertia: {batch_inertia}"
                )
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        # Early stopping based on absolute tolerance on squared change of
        # centers position
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in inertia) at step "
                    f"{step}/{n_steps}"
                )
            return True

        return False

    def _random_reassign(self):
        """
        Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用装饰器 @_fit_context，并设置 prefer_skip_nested_validation 参数为 True
    @_fit_context(prefer_skip_nested_validation=True)
```