# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_dbscan.py`

```
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块，用于处理警告信息
import warnings
# 导入数字类型检查模块
from numbers import Integral, Real

# 导入科学计算库 numpy
import numpy as np
# 导入稀疏矩阵处理模块
from scipy import sparse

# 导入基础估计器、聚类混合模块和拟合上下文
from ..base import BaseEstimator, ClusterMixin, _fit_context
# 导入度量相关模块
from ..metrics.pairwise import _VALID_METRICS
# 导入最近邻模块
from ..neighbors import NearestNeighbors
# 导入参数验证工具函数
from ..utils._param_validation import Interval, StrOptions, validate_params
# 导入验证工具函数
from ..utils.validation import _check_sample_weight
# 导入内部 DBSCAN 实现
from ._dbscan_inner import dbscan_inner

# 参数验证装饰器，用于验证输入参数类型和值范围
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=False,
)
# DBSCAN 聚类主函数定义
def dbscan(
    X,
    eps=0.5,
    *,
    min_samples=5,
    metric="minkowski",
    metric_params=None,
    algorithm="auto",
    leaf_size=30,
    p=2,
    sample_weight=None,
    n_jobs=None,
):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    X : {array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : str or callable, default='minkowski'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.
        X may be a :term:`sparse graph <sparse graph>`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
"""
    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

leaf_size参数，传递给BallTree或cKDTree的叶子大小。这会影响构建和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质。


    p : float, default=2
        The power of the Minkowski metric to be used to calculate distance
        between points.

p参数，Minkowski度量的幂，用于计算点之间的距离。


    sample_weight : array-like of shape (n_samples,), default=None
        Weight of each sample, such that a sample with a weight of at least
        ``min_samples`` is by itself a core sample; a sample with negative
        weight may inhibit its eps-neighbor from being core.
        Note that weights are absolute, and default to 1.

sample_weight参数，每个样本的权重。具有至少``min_samples``权重的样本本身是核心样本；负权重的样本可能会阻止其eps-邻居成为核心。注意权重是绝对值，默认为1。


    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. ``None`` means
        1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors. See :term:`Glossary <n_jobs>` for more details.
        If precomputed distance are used, parallel execution is not available
        and thus n_jobs will have no effect.

n_jobs参数，用于邻居搜索的并行作业数。``None``表示1，除非在``joblib.parallel_backend``上下文中。``-1``表示使用所有处理器。详细信息请参见 :term:`Glossary <n_jobs>`。


    Returns
    -------
    core_samples : ndarray of shape (n_core_samples,)
        Indices of core samples.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.  Noisy samples are given the label -1.

返回值：
- core_samples：形状为(n_core_samples,)的ndarray，核心样本的索引。
- labels：形状为(n_samples,)的ndarray，每个点的聚类标签。噪声样本的标签为-1。


    See Also
    --------
    DBSCAN : An estimator interface for this clustering algorithm.
    OPTICS : A similar estimator interface clustering at multiple values of
        eps. Our implementation is optimized for memory usage.

参见：
- DBSCAN：此聚类算法的估计器接口。
- OPTICS：多个eps值的类似估计器接口聚类。我们的实现经过了内存使用的优化。


    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

注意事项：
- 示例请参见 :ref:`examples/cluster/plot_dbscan.py <sphx_glr_auto_examples_cluster_plot_dbscan.py>`。
- 此实现批量计算所有邻域查询，将内存复杂度增加到O(n.d)，其中d是平均邻居数，而原始的DBSCAN的内存复杂度为O(n)。根据``algorithm``的不同，查询这些最近邻居可能会吸引更高的内存复杂度。


    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

避免查询复杂性的一种方法是使用 :func:`NearestNeighbors.radius_neighbors_graph <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` 预先计算稀疏的邻域块，使用``mode='distance'``，然后在这里使用``metric='precomputed'``。


    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

另一种减少内存和计算时间的方法是移除（接近的）重复点，并使用``sample_weight``代替。


    :class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower
    memory usage.

:class:`~sklearn.cluster.OPTICS` 提供了一个类似的聚类方法，具有较低的内存使用。


    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

参考文献：
- Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"基于密度的算法，用于在带有噪声的大型空间数据库中发现聚类" <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_。在：Knowledge Discovery and Data Mining第2届国际会议论文集，Portland, OR, AAAI Press, pp. 226-231. 1996
    # 导入 DBSCAN 算法模块
    from sklearn.cluster import dbscan
    
    # 定义一个函数，接受数据集 X 和一系列参数，并返回核心样本索引和聚类标签
    def dbscan_clustering(X, eps, min_samples, metric='euclidean', metric_params=None,
                          algorithm='auto', leaf_size=30, p=None, n_jobs=None):
        """
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            待聚类的样本数据集
    
        eps : float, optional, default=0.5
            DBSCAN 算法的邻域半径参数
    
        min_samples : int, optional, default=5
            DBSCAN 算法的最小样本数参数
    
        metric : string, or callable, optional
            距离度量参数，默认为 'euclidean'
    
        metric_params : dict, optional
            距离度量的参数
    
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            算法参数，用于计算最近邻
    
        leaf_size : int, optional (default=30)
            BallTree 或 KDTree 的叶子大小
    
        p : float, optional
            Minkowski 距离的幂参数
    
        n_jobs : int or None, optional (default=None)
            用于计算的并行作业数，None 表示 1 作业
    
        Returns
        -------
        core_sample_indices_ : array, shape = [n_core_samples]
            核心样本的索引数组
    
        labels_ : array, shape = [n_samples]
            每个样本的聚类标签，-1 表示离群点
    
        Notes
        -----
        返回的结果是 DBSCAN 算法对给定数据集 X 进行聚类后得到的核心样本索引和聚类标签数组。
        """
        # 创建 DBSCAN 的估计器对象
        est = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )
        # 使用数据集 X 进行拟合，支持样本权重参数
        est.fit(X, sample_weight=sample_weight)
        # 返回聚类后的核心样本索引和聚类标签数组
        return est.core_sample_indices_, est.labels_
class DBSCAN(ClusterMixin, BaseEstimator):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    This implementation has a worst case memory complexity of :math:`O({n}^2)`,
    which can occur when the `eps` param is large and `min_samples` is low,
    while the original DBSCAN only uses linear memory.
    For further details, see the Notes below.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself. If
        `min_samples` is set to a higher value, DBSCAN will find denser clusters,
        whereas if it is set to a lower value, the found clusters will be more
        sparse.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None):
        # Initialize DBSCAN clustering algorithm with specified parameters
        super().__init__()
        # Assigning parameters to instance variables
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


# 定义并行运行的作业数目，默认为 None
# 如果未指定，则为 1，除非在 :obj:`joblib.parallel_backend` 上下文中
# `-1` 表示使用所有处理器。详见 :term:`Glossary <n_jobs>`。



    Attributes
    ----------
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.


# 核心样本的索引数组，形状为 (n_core_samples,)
# 包含核心样本的索引。



    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.


# 训练过程中找到的每个核心样本的副本数组，形状为 (n_core_samples, n_features)
# 包含训练中找到的每个核心样本的副本。



    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.


# 数据集中每个点的聚类标签数组，形状为 (n_samples)
# 给定给 fit() 方法的数据集中每个点的聚类标签。
# 噪声样本的标签为 -1。



    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24


# 在 :term:`fit` 过程中看到的特征数量。
# 
# .. versionadded:: 0.24
#    版本添加说明: 0.24



    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0


# 在 :term:`fit` 过程中看到的特征名称数组，形状为 (`n_features_in_`,)
# 仅在 `X` 具有所有字符串特征名称时定义。
# 
# .. versionadded:: 1.0
#    版本添加说明: 1.0



    See Also
    --------
    OPTICS : A similar clustering at multiple values of eps. Our implementation
        is optimized for memory usage.


# 参见
# --------
# OPTICS : 多个 eps 值下的类似聚类算法。我们的实现优化了内存使用。



    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.


# 注意
# -----
# 例如，请参见 :ref:`examples/cluster/plot_dbscan.py
# <sphx_glr_auto_examples_cluster_plot_dbscan.py>`。
# 
# 此实现批量计算所有邻域查询，将内存复杂度增加到 O(n.d)，其中 d 是平均邻居数，
# 而原始的 DBSCAN 的内存复杂度为 O(n)。在查询这些最近邻域时，它可能会吸引更高的
# 内存复杂度，这取决于 ``algorithm``。



    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.


# 避免查询复杂性的一种方法是使用 :func:`NearestNeighbors.radius_neighbors_graph
# <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` 预先计算稀疏邻域，
# 使用 ``mode='distance'``，然后在此处使用 ``metric='precomputed'``。



    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.


# 另一种减少内存和计算时间的方法是删除（接近的）重复点，而使用 ``sample_weight``。



    :class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower memory
    usage.


# :class:`~sklearn.cluster.OPTICS` 提供了类似的聚类，内存使用更低。



    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    :doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
    <10.1145/3068335>`
    ACM Transactions on Database Systems (TODS), 42(3), 19.


# 参考文献
# ----------
# Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
# Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
# <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
# 在: 第二届国际知识发现与数据挖掘会议文集, Portland, OR, AAAI Press, pp. 226-231. 1996
# 
# Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
# :doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
# <10.1145/3068335>`
# ACM Transactions on Database Systems (TODS), 42(3), 19.
    _parameter_constraints: dict = {
        "eps": [Interval(Real, 0.0, None, closed="neither")],
        "min_samples": [Interval(Integral, 1, None, closed="left")],
        "metric": [
            StrOptions(set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        "metric_params": [dict, None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0.0, None, closed="left"), None],
        "n_jobs": [Integral, None],
    }


# 定义参数的约束条件字典，限定了每个参数的类型和取值范围
_parameter_constraints: dict = {
    "eps": [Interval(Real, 0.0, None, closed="neither")],  # eps参数为实数，大于0
    "min_samples": [Interval(Integral, 1, None, closed="left")],  # min_samples参数为整数，不小于1
    "metric": [  # metric参数为字符串，要么是预设的一组有效度量标准，要么是可调用对象
        StrOptions(set(_VALID_METRICS) | {"precomputed"}),
        callable,
    ],
    "metric_params": [dict, None],  # metric_params参数为字典或None
    "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],  # algorithm参数为预设的算法名称集合中的一种
    "leaf_size": [Interval(Integral, 1, None, closed="left")],  # leaf_size参数为整数，不小于1
    "p": [Interval(Real, 0.0, None, closed="left"), None],  # p参数为实数，大于0或None
    "n_jobs": [Integral, None],  # n_jobs参数为整数或None
}

def __init__(
    self,
    eps=0.5,
    *,
    min_samples=5,
    metric="euclidean",
    metric_params=None,
    algorithm="auto",
    leaf_size=30,
    p=None,
    n_jobs=None,
):
    # 初始化DBSCAN对象，设置参数
    self.eps = eps  # 设置eps参数
    self.min_samples = min_samples  # 设置min_samples参数
    self.metric = metric  # 设置metric参数
    self.metric_params = metric_params  # 设置metric_params参数
    self.algorithm = algorithm  # 设置algorithm参数
    self.leaf_size = leaf_size  # 设置leaf_size参数
    self.p = p  # 设置p参数
    self.n_jobs = n_jobs  # 设置n_jobs参数

@_fit_context(
    # DBSCAN.metric is not validated yet
    prefer_skip_nested_validation=False
)
def fit_predict(self, X, y=None, sample_weight=None):
    """Compute clusters from a data or distance matrix and predict labels.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
        (n_samples, n_samples)
        Training instances to cluster, or distances between instances if
        ``metric='precomputed'``. If a sparse matrix is provided, it will
        be converted into a sparse ``csr_matrix``.

    y : Ignored
        Not used, present here for API consistency by convention.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight of each sample, such that a sample with a weight of at least
        ``min_samples`` is by itself a core sample; a sample with a
        negative weight may inhibit its eps-neighbor from being core.
        Note that weights are absolute, and default to 1.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels. Noisy samples are given the label -1.
    """
    self.fit(X, sample_weight=sample_weight)  # 调用fit方法拟合数据
    return self.labels_  # 返回聚类标签数组

def _more_tags(self):
    return {"pairwise": self.metric == "precomputed"}
```