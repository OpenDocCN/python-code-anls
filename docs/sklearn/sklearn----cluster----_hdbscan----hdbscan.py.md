# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\hdbscan.py`

```
"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering
         of Applications with Noise
"""

# Authors: Leland McInnes <leland.mcinnes@gmail.com>
#          Steve Astels <sastels@gmail.com>
#          John Healy <jchealy@gmail.com>
#          Meekail Zain <zainmeekail@gmail.com>
# Copyright (c) 2015, Leland McInnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from numbers import Integral, Real
from warnings import warn

import numpy as np
from scipy.sparse import csgraph, issparse

from ...base import BaseEstimator, ClusterMixin, _fit_context
from ...metrics import pairwise_distances
from ...metrics._dist_metrics import DistanceMetric
from ...metrics.pairwise import _VALID_METRICS
from ...neighbors import BallTree, KDTree, NearestNeighbors
from ...utils._param_validation import Interval, StrOptions
from ...utils.validation import _allclose_dense_sparse, _assert_all_finite
from ._linkage import (
    MST_edge_dtype,
    make_single_linkage,
    mst_from_data_matrix,
    mst_from_mutual_reachability,
)
from ._reachability import mutual_reachability_graph
from ._tree import HIERARCHY_dtype, labelling_at_cut, tree_to_labels

# Set of valid metrics for fast nearest neighbor search using KDTree or BallTree
FAST_METRICS = set(KDTree.valid_metrics + BallTree.valid_metrics)

# Encodings are arbitrary but must be strictly negative.
# The current encodings are chosen as extensions to the -1 noise label.
# Avoided enums so that the end user only deals with simple labels.
_OUTLIER_ENCODING: dict = {
    "infinite": {
        "label": -2,
        # 对于无限点，概率也可以是1，因为无限点肯定是异常值，但是HDBSCAN库的实现约定使用0。
        "prob": 0,
    },
    "missing": {
        "label": -3,
        # 选择NaN概率来强调相应的数据在聚类问题中未被考虑。
        "prob": np.nan,
    },
}

def _brute_mst(mutual_reachability, min_samples):
    """
    Builds a minimum spanning tree (MST) from the provided mutual-reachability
    values. This function dispatches to a custom Cython implementation for
    dense arrays, and `scipy.sparse.csgraph.minimum_spanning_tree` for sparse
    arrays/matrices.

    Parameters
    ----------
    mututal_reachability_graph: {ndarray, sparse matrix} of shape \
            (n_samples, n_samples)
        Weighted adjacency matrix of the mutual reachability graph.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Returns
    -------
    mst : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.
    """
    # If the mutual reachability graph is not sparse, use a custom function
    # to compute MST from mutual reachability values.
    if not issparse(mutual_reachability):
        return mst_from_mutual_reachability(mutual_reachability)

    # Check if any row in the mutual reachability matrix has fewer than
    # `min_samples` non-zero elements.
    indptr = mutual_reachability.indptr
    num_points = mutual_reachability.shape[0]
    if any((indptr[i + 1] - indptr[i]) < min_samples for i in range(num_points)):
        # Raise an error if any point has fewer than `min_samples` neighbors.
        raise ValueError(
            f"There exists points with fewer than {min_samples} neighbors. Ensure"
            " your distance matrix has non-zero values for at least"
            f" `min_sample`={min_samples} neighbors for each points (i.e. K-nn"
            " graph), or specify a `max_distance` in `metric_params` to use when"
            " distances are missing."
        )

    # Check for connected components in the mutual reachability matrix.
    # If more than one connected component is found, raise an error indicating
    # that the graph is disconnected.
    n_components = csgraph.connected_components(
        mutual_reachability, directed=False, return_labels=False
    )
    if n_components > 1:
        raise ValueError(
            f"Sparse mutual reachability matrix has {n_components} connected"
            " components. HDBSCAN cannot be perfomed on a disconnected graph. Ensure"
            " that the sparse distance matrix has only one connected component."
        )

    # Compute the minimum spanning tree for the sparse mutual reachability graph.
    sparse_min_spanning_tree = csgraph.minimum_spanning_tree(mutual_reachability)
    rows, cols = sparse_min_spanning_tree.nonzero()
    # Create a structured array representing MST edges with rows, columns, and data.
    mst = np.rec.fromarrays(
        [rows, cols, sparse_min_spanning_tree.data],
        dtype=MST_edge_dtype,
    )
    return mst


def _process_mst(min_spanning_tree):
    """
    Builds a single-linkage tree (SLT) from the provided minimum spanning tree
    (MST). The MST is first sorted then processed by a custom Cython routine.

    Parameters
    ----------
    min_spanning_tree : ndarray
        Minimum spanning tree represented as an array of edges.

    Returns
    -------
    slt : ndarray
        The single-linkage tree constructed from the minimum spanning tree.
    """
    # 对最小生成树的边按权重进行排序
    row_order = np.argsort(min_spanning_tree["distance"])
    # 根据排序结果重新排列最小生成树的边
    min_spanning_tree = min_spanning_tree[row_order]
    # 将边列表转换为标准的层次聚类格式
    return make_single_linkage(min_spanning_tree)
# 定义一个函数 `_hdbscan_brute`，用于构建单链接树（SLT）从输入数据 `X`。如果 `metric="precomputed"`，则 `X` 必须是一个对称的距离数组。
def _hdbscan_brute(
    X,
    min_samples=5,
    alpha=None,
    metric="euclidean",
    n_jobs=None,
    copy=False,
    **metric_params,
):
    """
    Builds a single-linkage tree (SLT) from the input data `X`. If
    `metric="precomputed"` then `X` must be a symmetric array of distances.
    Otherwise, the pairwise distances are calculated directly and passed to
    `mutual_reachability_graph`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        Either the raw data from which to compute the pairwise distances,
        or the precomputed distances.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.

        - If metric is a string or callable, it must be one of
          the options allowed by :func:`~sklearn.metrics.pairwise_distances`
          for its metric parameter.

        - If metric is "precomputed", X is assumed to be a distance matrix and
          must be square.

    n_jobs : int, default=None
        The number of jobs to use for computing the pairwise distances. This
        works by breaking down the pairwise matrix into n_jobs even slices and
        computing them in parallel. This parameter is passed directly to
        :func:`~sklearn.metrics.pairwise_distances`.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite `X`, a copy will first be made, guaranteeing that
        the original data will be unchanged. Currently, it only applies when
        `metric="precomputed"`, when passing a dense array or a CSR sparse
        array/matrix.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    # 如果距离度量方式为预计算距离矩阵
    if metric == "precomputed":
        # 检查距离矩阵是否为对称矩阵，如果不是则引发值错误异常
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                "The precomputed distance matrix is expected to be symmetric, however"
                f" it has shape {X.shape}. Please verify that the"
                " distance matrix was constructed correctly."
            )
        # 检查距离矩阵是否对称，如果不是则引发值错误异常
        if not _allclose_dense_sparse(X, X.T):
            raise ValueError(
                "The precomputed distance matrix is expected to be symmetric, however"
                " its values appear to be asymmetric. Please verify that the distance"
                " matrix was constructed correctly."
            )

        # 如果需要拷贝，则复制距离矩阵，否则直接引用
        distance_matrix = X.copy() if copy else X
    else:
        # 使用 pairwise_distances 函数计算特征矩阵 X 的距离矩阵
        distance_matrix = pairwise_distances(
            X, metric=metric, n_jobs=n_jobs, **metric_params
        )
    # 将距离矩阵按 alpha 值缩放
    distance_matrix /= alpha

    # 获取距离矩阵的最大距离参数
    max_distance = metric_params.get("max_distance", 0.0)
    # 如果距离矩阵是稀疏矩阵且格式不为 CSR，则转换为 CSR 格式
    if issparse(distance_matrix) and distance_matrix.format != "csr":
        # 在调用 `csgraph.connected_components` 时需要 CSR 格式以避免转换开销
        distance_matrix = distance_matrix.tocsr()

    # 注意 `distance_matrix` 在此处是原地修改的，但过了这个点后不再需要，因此是安全的操作
    # 使用 mutual_reachability_graph 函数计算互达图
    mutual_reachability_ = mutual_reachability_graph(
        distance_matrix, min_samples=min_samples, max_distance=max_distance
    )
    # 使用 _brute_mst 函数计算最小生成树
    min_spanning_tree = _brute_mst(mutual_reachability_, min_samples=min_samples)
    # 如果最小生成树中有任何边权重为无穷大，发出警告
    if np.isinf(min_spanning_tree["distance"]).any():
        warn(
            (
                "The minimum spanning tree contains edge weights with value "
                "infinity. Potentially, you are missing too many distances "
                "in the initial distance matrix for the given neighborhood "
                "size."
            ),
            UserWarning,
        )
    # 返回处理后的最小生成树结果
    return _process_mst(min_spanning_tree)
def _hdbscan_prims(
    X,
    algo,
    min_samples=5,
    alpha=1.0,
    metric="euclidean",
    leaf_size=40,
    n_jobs=None,
    **metric_params,
):
    """
    Builds a single-linkage tree (SLT) from the input data `X`. If
    `metric="precomputed"` then `X` must be a symmetric array of distances.
    Otherwise, the pairwise distances are calculated directly and passed to
    `mutual_reachability_graph`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The raw data.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. `metric` must be one of the options allowed by
        :func:`~sklearn.metrics.pairwise_distances` for its metric
        parameter.

    leaf_size : int, default=40
        Leaf size passed to `NearestNeighbors` for efficient querying.

    n_jobs : int, default=None
        The number of jobs to use for computing the pairwise distances. This
        works by breaking down the pairwise matrix into n_jobs even slices and
        computing them in parallel. This parameter is passed directly to
        :func:`~sklearn.metrics.pairwise_distances`.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    # The Cython routines used require contiguous arrays
    X = np.asarray(X, order="C")  # 将输入数据转换为 C-order 的 NumPy 数组

    # Get distance to kth nearest neighbour
    nbrs = NearestNeighbors(
        n_neighbors=min_samples,
        algorithm=algo,
        leaf_size=leaf_size,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
        p=None,
    ).fit(X)  # 使用输入数据训练 NearestNeighbors 模型

    neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True)  # 计算每个样本到其第 k 个最近邻的距离
    core_distances = np.ascontiguousarray(neighbors_distances[:, -1])  # 将计算得到的核心距离转换为连续存储的 NumPy 数组
    dist_metric = DistanceMetric.get_metric(metric, **metric_params)  # 获取距离度量对象

    # Mutual reachability distance is implicit in mst_from_data_matrix
    min_spanning_tree = mst_from_data_matrix(X, core_distances, dist_metric, alpha)  # 构建最小生成树
    return _process_mst(min_spanning_tree)  # 处理最小生成树并返回结果
def remap_single_linkage_tree(tree, internal_to_raw, non_finite):
    """
    Takes an internal single_linkage_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be merged into the final node at np.inf distance and
    considered noise points.

    Parameters
    ----------
    tree : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index
    non_finite : ndarray
        Boolean array of which entries in the raw data are non-finite
    """
    # 计算有限点的数量
    finite_count = len(internal_to_raw)

    # 计算非有限点（异常点）的数量
    outlier_count = len(non_finite)
    
    # 遍历树的每个节点
    for i, _ in enumerate(tree):
        left = tree[i]["left_node"]
        right = tree[i]["right_node"]

        # 如果左子节点是有限点，则映射为原始索引，否则加上异常点数量
        if left < finite_count:
            tree[i]["left_node"] = internal_to_raw[left]
        else:
            tree[i]["left_node"] = left + outlier_count
        
        # 如果右子节点是有限点，则映射为原始索引，否则加上异常点数量
        if right < finite_count:
            tree[i]["right_node"] = internal_to_raw[right]
        else:
            tree[i]["right_node"] = right + outlier_count

    # 创建一个存储异常点的树
    outlier_tree = np.zeros(len(non_finite), dtype=HIERARCHY_dtype)
    last_cluster_id = max(
        tree[tree.shape[0] - 1]["left_node"], tree[tree.shape[0] - 1]["right_node"]
    )
    last_cluster_size = tree[tree.shape[0] - 1]["cluster_size"]
    
    # 遍历每个异常点，分配新的聚类标识和最大距离
    for i, outlier in enumerate(non_finite):
        outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)
        last_cluster_id += 1
        last_cluster_size += 1
    
    # 将异常点树连接到原始树上
    tree = np.concatenate([tree, outlier_tree])
    return tree


def _get_finite_row_indices(matrix):
    """
    Returns the indices of the purely finite rows of a
    sparse matrix or dense ndarray
    """
    # 如果输入是稀疏矩阵，则找出所有完全有限的行的索引
    if issparse(matrix):
        row_indices = np.array(
            [i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]
        )
    else:
        # 如果输入是密集矩阵，则找出所有行和为有限值的行的索引
        (row_indices,) = np.isfinite(matrix.sum(axis=1)).nonzero()
    return row_indices


class HDBSCAN(ClusterMixin, BaseEstimator):
    """Cluster data using hierarchical density-based clustering.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs :class:`~sklearn.cluster.DBSCAN` over varying epsilon
    values and integrates the result to find a clustering that gives the best
    stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike
    :class:`~sklearn.cluster.DBSCAN`), and be more robust to parameter selection.
    Read more in the :ref:`User Guide <hdbscan>`.

    For an example of how to use HDBSCAN, as well as a comparison to
    :class:`~sklearn.cluster.DBSCAN`, please see the :ref:`plotting demo
    <sphx_glr_auto_examples_cluster_plot_hdbscan.py>`.

    .. versionadded:: 1.3

    Parameters
    ----------
    """
   `
# 定义最小簇大小，作为被视为簇的最少样本数；小于此大小的组将被视为噪声。
min_cluster_size : int, default=5

# 定义核心点所需的邻域样本数，包括点本身在内。
# 当为 `None` 时，默认为 `min_cluster_size`。
min_samples : int, default=None

# 聚类选择的 epsilon 距离阈值。低于此值的聚类将被合并。
# 更多信息请参见 [5]_。
cluster_selection_epsilon : float, default=0.0

# `"eom"` 聚类选择算法返回的簇的大小限制。当 `max_cluster_size=None` 时没有限制。
# 如果 `cluster_selection_method="leaf"`，则此参数无效。
max_cluster_size : int, default=None

# 计算特征数组中实例之间距离时使用的度量标准。
# - 如果度量标准是字符串或可调用对象，必须是 `sklearn.metrics.pairwise_distances` 允许的选项之一。
# - 如果度量标准是 "precomputed"，则假定 X 是距离矩阵且必须是方阵。
metric : str or callable, default='euclidean'

# 传递给距离度量的参数。
metric_params : dict, default=None

# 用于计算核心距离的算法。
# - `"auto"`：默认选项，尝试使用 `sklearn.neighbors.KDTree`，否则使用 `sklearn.neighbors.BallTree`。
# - 如果 `X` 在 `fit` 过程中是稀疏的或度量无效，则解析为使用 `"brute"` 算法。
# - `'kdtree'` 选项在 1.4 版本中已弃用，并将在 1.6 版本中更名为 `'kd_tree'`。
# - `'balltree'` 选项在 1.4 版本中已弃用，并将在 1.6 版本中更名为 `'ball_tree'`。
algorithm : {"auto", "brute", "kd_tree", "ball_tree"}, default="auto"

# 当作为核心距离算法的 KDTree 或 BallTree 使用时，用于快速最近邻查询的叶子大小。
# 对于 `algorithm="brute"`，此参数被忽略。
leaf_size : int, default=40
    n_jobs : int, default=None
        Number of jobs to run in parallel for distance calculation.
        `None` defaults to 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` uses all available processors. See :term:`Glossary <n_jobs>`
        for more details.

    cluster_selection_method : {"eom", "leaf"}, default="eom"
        Method used to select clusters from the condensed tree. For HDBSCAN*,
        `"eom"` uses an Excess of Mass algorithm to find persistent clusters.
        Alternatively, `"leaf"` selects clusters at the leaves for finer and
        more homogeneous clusters.

    allow_single_cluster : bool, default=False
        By default, HDBSCAN* does not produce a single cluster. Setting this
        to True overrides this behavior, allowing single-cluster results if
        deemed valid for the dataset.

    store_centers : str, default=None
        Specifies which cluster centers to compute and store:

        - `None` does not compute or store any centers.
        - `"centroid"` computes centers as the weighted average of positions.
          Results are not guaranteed to be observed data points.
        - `"medoid"` computes centers as points minimizing distance to all
          other points in the cluster. Slower than `"centroid"`, but guarantees
          output as an observed data point. Well-defined for arbitrary metrics.
        - `"both"` computes and stores both `"centroid"` and `"medoid"` centers.

    copy : bool, default=False
        If `copy=True`, ensures data passed to :term:`fit` remains unchanged
        by making copies before any in-place modifications. Applies when
        `metric="precomputed"`, using dense array/CSR sparse matrix, and
        `algorithm="brute"`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset passed to :term:`fit`.
        Outliers are labeled as follows:

        - `-1` for noisy samples.
        - `-2` for samples with infinite elements (+/- np.inf).
        - `-3` for samples with missing data, even if they also have infinite elements.
    
    """
    # 定义参数约束字典，包含各参数的取值范围或可选项
    _parameter_constraints = {
        "min_cluster_size": [Interval(Integral, left=2, right=None, closed="left")],
        # 最小簇大小的约束为大于等于2的整数
        "min_samples": [Interval(Integral, left=1, right=None, closed="left"), None],
        # 最小样本数的约束为大于等于1的整数或无约束
        "cluster_selection_epsilon": [
            Interval(Real, left=0, right=None, closed="left")
        ],
        # 簇选择的 epsilon 的约束为大于等于0的实数
        "max_cluster_size": [
            None,
            Interval(Integral, left=1, right=None, closed="left"),
        ],
        # 最大簇大小的约束为大于等于1的整数或无约束
        "metric": [
            StrOptions(FAST_METRICS | set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        # 距离度量的约束为预定义的快速度量、有效度量集合或可调用对象
        "metric_params": [dict, None],
        # 距离度量参数的约束为字典或无约束
        "alpha": [Interval(Real, left=0, right=None, closed="neither")],
        # alpha 参数的约束为大于0的实数
        # TODO(1.6): Remove "kdtree" and "balltree"  option
        "algorithm": [
            StrOptions(
                {"auto", "brute", "kd_tree", "ball_tree", "kdtree", "balltree"},
                deprecated={"kdtree", "balltree"},
            ),
        ],
        # 算法选择的约束为自动选择、暴力搜索、kd_tree、ball_tree等选项
        "leaf_size": [Interval(Integral, left=1, right=None, closed="left")],
        # 叶子大小的约束为大于等于1的整数
        "n_jobs": [Integral, None],
        # 并行作业数的约束为整数或无约束
        "cluster_selection_method": [StrOptions({"eom", "leaf"})],
        # 簇选择方法的约束为"eom"或"leaf"
        "allow_single_cluster": ["boolean"],
        # 允许单个簇的约束为布尔值
        "store_centers": [None, StrOptions({"centroid", "medoid", "both"})],
        # 存储中心点的约束为无约束或"centroid"、"medoid"、"both"
        "copy": ["boolean"],
        # 复制的约束为布尔值
    }
    """
    
    """
    # 定义 HDBSCAN 类的初始化方法，接受多个参数用于设置聚类算法的参数
    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=None,
        metric="euclidean",
        metric_params=None,
        alpha=1.0,
        algorithm="auto",
        leaf_size=40,
        n_jobs=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        store_centers=None,
        copy=False,
    """
    ):
        # 设置最小聚类大小
        self.min_cluster_size = min_cluster_size
        # 设置最小样本数
        self.min_samples = min_samples
        # 设置 alpha 参数
        self.alpha = alpha
        # 设置最大聚类大小
        self.max_cluster_size = max_cluster_size
        # 设置聚类选择 epsilon
        self.cluster_selection_epsilon = cluster_selection_epsilon
        # 设置距离度量
        self.metric = metric
        # 设置距离度量的参数
        self.metric_params = metric_params
        # 设置聚类算法
        self.algorithm = algorithm
        # 设置叶子大小
        self.leaf_size = leaf_size
        # 设置并行工作的任务数
        self.n_jobs = n_jobs
        # 设置聚类选择方法
        self.cluster_selection_method = cluster_selection_method
        # 设置是否允许单一聚类
        self.allow_single_cluster = allow_single_cluster
        # 设置是否存储聚类中心点
        self.store_centers = store_centers
        # 设置是否复制输入数据
        self.copy = copy

    @_fit_context(
        # 禁用 HDBSCAN.metric 的嵌套验证
        prefer_skip_nested_validation=False
    )
    def fit_predict(self, X, y=None):
        """Cluster X and return the associated cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                ndarray of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            `metric='precomputed'`.

        y : None
            Ignored.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Cluster labels.
        """
        # 使用给定数据 X 进行聚类拟合
        self.fit(X)
        # 返回聚类标签
        return self.labels_
    # 计算并存储每个簇的质心/中心点。
    # 
    # 这要求 `X` 是原始特征数组，而不是预先计算的距离。这个辅助方法不直接返回输出，
    # 而是将它们存储在 `self.{centroids, medoids}_` 属性中。
    # 根据 `self.store_centers` 的值来决定计算和存储哪些属性。
    #
    # Parameters
    # ----------
    # X : ndarray of shape (n_samples, n_features)
    #     估算器拟合时使用的特征数组。
    #
    def _weighted_cluster_center(self, X):
        # 非噪声簇的数量
        n_clusters = len(set(self.labels_) - {-1, -2})
        # 创建一个与数据长度相同的掩码数组
        mask = np.empty((X.shape[0],), dtype=np.bool_)
        # 是否创建质心
        make_centroids = self.store_centers in ("centroid", "both")
        # 是否创建中心点
        make_medoids = self.store_centers in ("medoid", "both")

        if make_centroids:
            # 如果需要创建质心，则初始化空的质心数组
            self.centroids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        if make_medoids:
            # 如果需要创建中心点，则初始化空的中心点数组
            self.medoids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)

        # 需要逐个处理每个簇，因为每个簇的样本数可能不同，无法创建均匀的三维数组。
        for idx in range(n_clusters):
            # 创建当前簇的掩码
            mask = self.labels_ == idx
            # 获取当前簇的数据
            data = X[mask]
            # 获取当前簇的强度或概率
            strength = self.probabilities_[mask]

            if make_centroids:
                # 如果需要创建质心，则计算加权平均质心
                self.centroids_[idx] = np.average(data, weights=strength, axis=0)

            if make_medoids:
                # 如果需要创建中心点，则实现加权最小距离中心点的计算
                # TODO: 实现加权最小距离中心点的后端
                dist_mat = pairwise_distances(
                    data, metric=self.metric, **self._metric_params
                )
                dist_mat = dist_mat * strength
                # 找到加权距离矩阵中和最小的索引作为中心点索引
                medoid_index = np.argmin(dist_mat.sum(axis=1))
                self.medoids_[idx] = data[medoid_index]

        # 返回函数执行结果
        return
    # 返回使用 DBSCAN 算法聚类结果的标签数组，不包含边界点

    def dbscan_clustering(self, cut_distance, min_cluster_size=5):
        """Return clustering given by DBSCAN without border points.

        Return clustering that would be equivalent to running DBSCAN* for a
        particular cut_distance (or epsilon) DBSCAN* can be thought of as
        DBSCAN without the border points.  As such these results may differ
        slightly from `cluster.DBSCAN` due to the difference in implementation
        over the non-core points.

        This can also be thought of as a flat clustering derived from constant
        height cut through the single linkage tree.

        This represents the result of selecting a cut value for robust single linkage
        clustering. The `min_cluster_size` allows the flat clustering to declare noise
        points (and cluster smaller than `min_cluster_size`).

        Parameters
        ----------
        cut_distance : float
            The mutual reachability distance cut value to use to generate a
            flat clustering.

        min_cluster_size : int, default=5
            Clusters smaller than this value with be called 'noise' and remain
            unclustered in the resulting flat clustering.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            An array of cluster labels, one per datapoint.
            Outliers are labeled as follows:

            - Noisy samples are given the label -1.
            - Samples with infinite elements (+/- np.inf) are given the label -2.
            - Samples with missing data are given the label -3, even if they
              also have infinite elements.
        """
        # 根据指定的切割距离和最小簇大小进行标签生成
        labels = labelling_at_cut(
            self._single_linkage_tree_, cut_distance, min_cluster_size
        )
        
        # 从在 `fit` 过程中生成的标签推断出索引
        infinite_index = self.labels_ == _OUTLIER_ENCODING["infinite"]["label"]
        missing_index = self.labels_ == _OUTLIER_ENCODING["missing"]["label"]

        # 覆盖无限/缺失的异常样本标签（否则简单视为噪声）
        labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]
        labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]
        
        # 返回处理过的标签数组
        return labels

    def _more_tags(self):
        # 返回一个字典，指示是否允许出现 NaN 值，取决于度量标准是否为 "precomputed"
        return {"allow_nan": self.metric != "precomputed"}
```