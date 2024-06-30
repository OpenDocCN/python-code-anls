# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_base.py`

```
"""Base and mixin classes for nearest neighbors."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools  # 导入 itertools 模块，用于高效的迭代工具
import numbers  # 导入 numbers 模块，用于数字类型的操作
import warnings  # 导入 warnings 模块，用于警告处理
from abc import ABCMeta, abstractmethod  # 从 abc 模块导入 ABCMeta 和 abstractmethod，用于定义抽象基类
from functools import partial  # 从 functools 模块导入 partial，用于创建偏函数

from numbers import Integral, Real  # 从 numbers 模块导入 Integral 和 Real 类型

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from joblib import effective_n_jobs  # 从 joblib 库导入 effective_n_jobs 函数
from scipy.sparse import csr_matrix, issparse  # 从 scipy.sparse 导入 csr_matrix 和 issparse 函数

from ..base import BaseEstimator, MultiOutputMixin, is_classifier  # 导入自定义模块中的基类、混合类和分类器判断函数
from ..exceptions import DataConversionWarning, EfficiencyWarning  # 从自定义模块导入数据转换警告和效率警告异常类
from ..metrics import DistanceMetric, pairwise_distances_chunked  # 从自定义模块导入距离度量和分块计算距离函数
from ..metrics._pairwise_distances_reduction import (  # 从自定义模块导入距离函数的相关类和方法
    ArgKmin,
    RadiusNeighbors,
)
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS  # 从自定义模块导入成对距离函数集合
from ..utils import (  # 从自定义模块导入各种实用工具函数和类
    check_array,
    gen_even_slices,
)
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入参数验证相关的类和函数
from ..utils.fixes import parse_version, sp_base_version  # 导入版本解析函数和基础版本函数
from ..utils.multiclass import check_classification_targets  # 导入多类别分类目标检查函数
from ..utils.parallel import Parallel, delayed  # 导入并行处理相关的类和函数
from ..utils.validation import (  # 导入验证函数和类
    _to_object_array,
    check_is_fitted,
    check_non_negative,
)
from ._ball_tree import BallTree  # 从自定义模块导入 BallTree 类
from ._kd_tree import KDTree  # 从自定义模块导入 KDTree 类

SCIPY_METRICS = [  # 定义 SciPy 支持的距离度量列表
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "cosine",
    "dice",
    "hamming",
    "jaccard",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

if sp_base_version < parse_version("1.11"):
    # 在 SciPy 1.11 之前的版本中，以下度量已弃用并在 SciPy 1.11 中移除
    SCIPY_METRICS += ["kulsinski"]

if sp_base_version < parse_version("1.9"):
    # 在 SciPy 1.9 之前的版本中，以下度量已弃用并在 SciPy 1.9 中移除
    SCIPY_METRICS += ["matching"]

VALID_METRICS = dict(
    ball_tree=BallTree.valid_metrics,  # BallTree 算法支持的度量方法
    kd_tree=KDTree.valid_metrics,  # KDTree 算法支持的度量方法
    brute=sorted(set(PAIRWISE_DISTANCE_FUNCTIONS).union(SCIPY_METRICS)),  # brute force 方法支持的度量方法
)

VALID_METRICS_SPARSE = dict(
    ball_tree=[],  # 稀疏矩阵下 BallTree 算法支持的度量方法为空列表
    kd_tree=[],  # 稀疏矩阵下 KDTree 算法支持的度量方法为空列表
    brute=(PAIRWISE_DISTANCE_FUNCTIONS.keys() - {"haversine", "nan_euclidean"}),  # 稀疏矩阵下 brute force 方法支持的度量方法
)


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``.

    Assume weights have already been validated.

    Parameters
    ----------
    dist : ndarray
        The input distances.

    weights : {'uniform', 'distance'}, callable or None
        The kind of weighting used.

    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        If ``weights == 'uniform'``, then returns None.
    """
    if weights in (None, "uniform"):
        return None
    # 如果权重选择为 "distance"，则进行以下处理
    if weights == "distance":
        # 如果用户试图对一个与一个或多个训练点距离为零的点进行分类，
        # 那么这些训练点的权重设置为1.0，其他点设置为0.0
        if dist.dtype is np.dtype(object):
            # 对于每个点的距离信息，检查其是否可迭代
            # （例如：RadiusNeighborClassifier.predict 可能将距离为异常值的点设为 1e-6）
            for point_dist_i, point_dist in enumerate(dist):
                if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                    # 如果距离为0.0，将其设置为True，否则设置为其倒数
                    dist[point_dist_i] = point_dist == 0.0
                else:
                    dist[point_dist_i] = 1.0 / point_dist
        else:
            # 使用忽略除法错误的上下文，将距离的倒数赋给 dist
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
            # 找出其中包含无穷大值的行，并将这些行的 dist 设为对应的 inf_mask
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        # 返回处理后的距离数据
        return dist

    # 如果权重选择为一个可调用的函数，则直接使用该函数计算权重并返回结果
    if callable(weights):
        return weights(dist)
def _is_sorted_by_data(graph):
    """Return whether the graph's non-zero entries are sorted by data.

    The non-zero entries are stored in graph.data and graph.indices.
    For each row (or sample), the non-zero entries can be either:
        - sorted by indices, as after graph.sort_indices();
        - sorted by data, as after _check_precomputed(graph);
        - not sorted.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    Returns
    -------
    res : bool
        Whether input graph is sorted by data.
    """
    assert graph.format == "csr"  # 断言确保图的格式为 CSR 格式
    out_of_order = graph.data[:-1] > graph.data[1:]  # 检查非零元素数据是否按照数据排序
    line_change = np.unique(graph.indptr[1:-1] - 1)  # 找出行索引（样本）发生变化的位置
    line_change = line_change[line_change < out_of_order.shape[0]]  # 过滤掉超出范围的行索引变化
    return out_of_order.sum() == out_of_order[line_change].sum()


def _check_precomputed(X):
    """Check precomputed distance matrix.

    If the precomputed distance matrix is sparse, it checks that the non-zero
    entries are sorted by distances. If not, the matrix is copied and sorted.

    Parameters
    ----------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.

    Returns
    -------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.
    """
    if not issparse(X):  # 如果 X 不是稀疏矩阵，则转换为普通数组
        X = check_array(X)
        check_non_negative(X, whom="precomputed distance matrix.")  # 检查非负性
        return X
    else:
        graph = X  # 如果 X 是稀疏矩阵，则将其赋给 graph 变量

    if graph.format not in ("csr", "csc", "coo", "lil"):  # 检查稀疏矩阵的格式是否支持
        raise TypeError(
            "Sparse matrix in {!r} format is not supported due to "
            "its handling of explicit zeros".format(graph.format)
        )
    copied = graph.format != "csr"  # 标记是否需要复制图
    graph = check_array(graph, accept_sparse="csr")  # 确保图是 CSR 格式
    check_non_negative(graph, whom="precomputed distance matrix.")  # 再次检查非负性
    graph = sort_graph_by_row_values(graph, copy=not copied, warn_when_not_sorted=True)  # 根据行值排序图

    return graph


@validate_params(
    {
        "graph": ["sparse matrix"],
        "copy": ["boolean"],
        "warn_when_not_sorted": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=True):
    """Sort a sparse graph such that each row is stored with increasing values.

    .. versionadded:: 1.2

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Distance matrix to other samples, where only non-zero elements are
        considered neighbors. Matrix is converted to CSR format if not already.

    copy : bool, default=False
        Whether to create a copy of the graph, if necessary.

    warn_when_not_sorted : bool, default=True
        Whether to warn when the graph is not sorted by row values.

    """
    copy : bool, default=False
        如果为True，则在排序之前复制图形。如果为False，则原地排序。
        如果图形不是CSR格式，则必须将copy设置为True，以允许转换为CSR格式，否则会引发错误。

    warn_when_not_sorted : bool, default=True
        如果为True，则在输入图形未按行值排序时会引发EfficiencyWarning警告。

    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        距离矩阵到其他样本的距离，只考虑非零元素作为邻居。矩阵采用CSR格式。

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from sklearn.neighbors import sort_graph_by_row_values
    >>> X = csr_matrix(
    ...     [[0., 3., 1.],
    ...      [3., 0., 2.],
    ...      [1., 2., 0.]])
    >>> X.data
    array([3., 1., 3., 2., 1., 2.])
    >>> X_ = sort_graph_by_row_values(X)
    >>> X_.data
    array([1., 3., 2., 3., 1., 2.])
    """
    # 检查图形格式是否为CSR并且是否已按数据排序
    if graph.format == "csr" and _is_sorted_by_data(graph):
        return graph

    # 如果输入图形未按行值排序且warn_when_not_sorted为True，则发出警告
    if warn_when_not_sorted:
        warnings.warn(
            (
                "Precomputed sparse input was not sorted by row values. Use the"
                " function sklearn.neighbors.sort_graph_by_row_values to sort the input"
                " by row values, with warn_when_not_sorted=False to remove this"
                " warning."
            ),
            EfficiencyWarning,
        )

    # 检查图形格式是否为支持的格式（"csr", "csc", "coo", "lil"）
    if graph.format not in ("csr", "csc", "coo", "lil"):
        raise TypeError(
            f"Sparse matrix in {graph.format!r} format is not supported due to "
            "its handling of explicit zeros"
        )
    # 如果图形格式不是CSR并且copy为False，则引发错误
    elif graph.format != "csr":
        if not copy:
            raise ValueError(
                "The input graph is not in CSR format. Use copy=True to allow "
                "the conversion to CSR format."
            )
        # 将图形转换为CSR格式
        graph = graph.asformat("csr")
    # 如果图形格式为CSR并且copy为True，则复制图形
    elif copy:  # csr format with copy=True
        graph = graph.copy()

    # 计算每行非零元素数量
    row_nnz = np.diff(graph.indptr)
    # 如果每个样本提供的邻居数相同
    if row_nnz.max() == row_nnz.min():
        n_samples = graph.shape[0]
        distances = graph.data.reshape(n_samples, -1)

        # 对距离数据进行排序
        order = np.argsort(distances, kind="mergesort")
        order += np.arange(n_samples)[:, None] * row_nnz[0]
        order = order.ravel()
        graph.data = graph.data[order]
        graph.indices = graph.indices[order]

    else:
        # 分别对每行的数据进行排序
        for start, stop in zip(graph.indptr, graph.indptr[1:]):
            order = np.argsort(graph.data[start:stop], kind="mergesort")
            graph.data[start:stop] = graph.data[start:stop][order]
            graph.indices[start:stop] = graph.indices[start:stop][order]

    return graph
def _kneighbors_from_graph(graph, n_neighbors, return_distance):
    """Decompose a nearest neighbors sparse graph into distances and indices.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    n_neighbors : int
        Number of neighbors required for each sample.

    return_distance : bool
        Whether or not to return the distances.

    Returns
    -------
    neigh_dist : ndarray of shape (n_samples, n_neighbors)
        Distances to nearest neighbors. Only present if `return_distance=True`.

    neigh_ind : ndarray of shape (n_samples, n_neighbors)
        Indices of nearest neighbors.
    """
    # 获取样本数
    n_samples = graph.shape[0]
    # 确保 graph 的格式为 CSR
    assert graph.format == "csr"

    # 计算每个样本所拥有的邻居数量
    row_nnz = np.diff(graph.indptr)
    row_nnz_min = row_nnz.min()
    # 如果指定了 n_neighbors，并且有样本的邻居数小于指定的 n_neighbors，抛出异常
    if n_neighbors is not None and row_nnz_min < n_neighbors:
        raise ValueError(
            "%d neighbors per samples are required, but some samples have only"
            " %d neighbors in precomputed graph matrix. Decrease number of "
            "neighbors used or recompute the graph with more neighbors."
            % (n_neighbors, row_nnz_min)
        )

    def extract(a):
        # 如果每个样本具有相同数量的邻居
        if row_nnz.max() == row_nnz_min:
            return a.reshape(n_samples, -1)[:, :n_neighbors]
        else:
            # 使用广播机制创建索引
            idx = np.tile(np.arange(n_neighbors), (n_samples, 1))
            idx += graph.indptr[:-1, None]
            return a.take(idx, mode="clip").reshape(n_samples, n_neighbors)

    # 如果需要返回距离，则返回距离和索引
    if return_distance:
        return extract(graph.data), extract(graph.indices)
    else:
        return extract(graph.indices)


def _radius_neighbors_from_graph(graph, radius, return_distance):
    """Decompose a nearest neighbors sparse graph into distances and indices.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    radius : float
        Radius of neighborhoods which should be strictly positive.

    return_distance : bool
        Whether or not to return the distances.

    Returns
    -------
    neigh_dist : ndarray of shape (n_samples,) of arrays
        Distances to nearest neighbors. Only present if `return_distance=True`.

    neigh_ind : ndarray of shape (n_samples,) of arrays
        Indices of nearest neighbors.
    """
    # 确保 graph 的格式为 CSR
    assert graph.format == "csr"

    # 检查是否不需要过滤
    no_filter_needed = bool(graph.data.max() <= radius)

    # 如果不需要过滤，则直接使用 graph 的数据、索引和指针
    if no_filter_needed:
        data, indices, indptr = graph.data, graph.indices, graph.indptr
    # 如果没有特定要求，执行以下操作
    else:
        # 创建一个布尔掩码，标记图数据小于等于给定半径的部分
        mask = graph.data <= radius
        # 如果需要返回距离信息
        if return_distance:
            # 从符合掩码条件的数据中提取出有效数据
            data = np.compress(mask, graph.data)
        # 从符合掩码条件的索引中提取出有效索引
        indices = np.compress(mask, graph.indices)
        # 重新计算行指针，以便符合新的掩码条件
        indptr = np.concatenate(([0], np.cumsum(mask)))[graph.indptr]

    # 将索引数组转换为指定类型（np.intp），如果无需复制，则不进行复制
    indices = indices.astype(np.intp, copy=no_filter_needed)

    # 如果需要返回距离信息
    if return_distance:
        # 将分割后的数据数组转换为对象数组，并返回邻居距离
        neigh_dist = _to_object_array(np.split(data, indptr[1:-1]))
    # 将分割后的索引数组转换为对象数组，并返回邻居索引
    neigh_ind = _to_object_array(np.split(indices, indptr[1:-1]))

    # 如果需要返回距离信息，则返回邻居距离和邻居索引
    if return_distance:
        return neigh_dist, neigh_ind
    # 否则，只返回邻居索引
    else:
        return neigh_ind
# 定义一个抽象基类，用于最近邻估计器的基础功能，同时支持多输出
class NeighborsBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for nearest neighbors estimators."""

    # 定义参数的约束条件字典，限定了各参数的取值范围和类型
    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "radius": [Interval(Real, 0, None, closed="both"), None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric": [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable],
        "metric_params": [dict, None],
        "n_jobs": [Integral, None],
    }

    # 抽象方法，初始化最近邻估计器的基本参数
    @abstractmethod
    def __init__(
        self,
        n_neighbors=None,
        radius=None,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors  # 最近邻数量
        self.radius = radius  # 半径参数
        self.algorithm = algorithm  # 算法选择，默认为自动选择
        self.leaf_size = leaf_size  # 叶子节点大小
        self.metric = metric  # 距离度量方法，默认为闵可夫斯基距离
        self.metric_params = metric_params  # 距离度量方法的额外参数
        self.p = p  # 距离度量方法的参数
        self.n_jobs = n_jobs  # 并行任务数

    # 检查算法和度量方法是否兼容
    def _check_algorithm_metric(self):
        if self.algorithm == "auto":
            if self.metric == "precomputed":
                alg_check = "brute"  # 如果度量方法是预先计算的，则强制使用brute算法
            elif (
                callable(self.metric)
                or self.metric in VALID_METRICS["ball_tree"]
                or isinstance(self.metric, DistanceMetric)
            ):
                alg_check = "ball_tree"  # 如果度量方法是可调用的，或者是ball_tree支持的度量方法，则使用ball_tree算法
            else:
                alg_check = "brute"  # 否则使用brute算法
        else:
            alg_check = self.algorithm  # 否则使用指定的算法

        if callable(self.metric):
            if self.algorithm == "kd_tree":
                # 对于kd_tree算法，可调用度量方法只能用于brute force和ball_tree
                raise ValueError(
                    "kd_tree does not support callable metric '%s'"
                    "Function call overhead will result"
                    "in very poor performance." % self.metric
                )
        elif self.metric not in VALID_METRICS[alg_check] and not isinstance(
            self.metric, DistanceMetric
        ):
            # 如果度量方法不在算法支持的度量方法列表中，并且不是DistanceMetric的实例，则报错
            raise ValueError(
                "Metric '%s' not valid. Use "
                "sorted(sklearn.neighbors.VALID_METRICS['%s']) "
                "to get valid options. "
                "Metric can also be a callable function." % (self.metric, alg_check)
            )

        if self.metric_params is not None and "p" in self.metric_params:
            if self.p is not None:
                # 如果metric_params中指定了参数p，则警告忽略__init__中的对应参数
                warnings.warn(
                    (
                        "Parameter p is found in metric_params. "
                        "The corresponding parameter from __init__ "
                        "is ignored."
                    ),
                    SyntaxWarning,
                    stacklevel=3,
                )
    def _more_tags(self):
        # 返回一个字典，用于指示交叉验证过程中如何正确分割数据
        return {"pairwise": self.metric == "precomputed"}
class KNeighborsMixin:
    """Mixin for k-neighbors searches."""

    def _kneighbors_reduce_func(self, dist, start, n_neighbors, return_distance):
        """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        n_neighbors : int
            Number of neighbors required for each sample.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : array of shape (n_samples_chunk, n_neighbors)
            Returned only if `return_distance=True`.

        neigh : array of shape (n_samples_chunk, n_neighbors)
            The neighbors indices.
        """
        # Generate an array of indices corresponding to the number of samples in the chunk
        sample_range = np.arange(dist.shape[0])[:, None]
        # Find indices of the nearest neighbors within each chunk
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        # Keep only the nearest `n_neighbors` indices
        neigh_ind = neigh_ind[:, :n_neighbors]
        # Ensure the nearest neighbors are sorted correctly
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]
        
        # Depending on the metric and return_distance flag, prepare the result
        if return_distance:
            if self.effective_metric_ == "euclidean":
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = dist[sample_range, neigh_ind], neigh_ind
        else:
            result = neigh_ind
        return result


class RadiusNeighborsMixin:
    """Mixin for radius-based neighbors searches."""

    def _radius_neighbors_reduce_func(self, dist, start, radius, return_distance):
        """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        radius : float
            The radius considered when making the nearest neighbors search.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : list of ndarray of shape (n_samples_chunk,)
            Returned only if `return_distance=True`.

        neigh : list of ndarray of shape (n_samples_chunk,)
            The neighbors indices.
        """
        # Find indices of neighbors within radius for each sample in the chunk
        neigh_ind = [np.where(d <= radius)[0] for d in dist]

        # Depending on the metric and return_distance flag, prepare the result
        if return_distance:
            if self.effective_metric_ == "euclidean":
                dist = [np.sqrt(d[neigh_ind[i]]) for i, d in enumerate(dist)]
            else:
                dist = [d[neigh_ind[i]] for i, d in enumerate(dist)]
            results = dist, neigh_ind
        else:
            results = neigh_ind
        return results
    # 返回半径内邻居的索引列表或稀疏图
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        # 如果未提供输入数据 X，则使用训练集中的数据计算邻居
        # 如果未指定半径，则使用默认的半径值
        # 如果指定要返回距离信息，则返回邻居点和它们的距离
        # 如果指定对结果进行排序，则按距离对结果进行排序
    
    
    ```  
    # 返回半径内邻居的稀疏图或其他指定模式的表示
    def radius_neighbors_graph(
        self, X=None, radius=None, mode="connectivity", sort_results=False
    ):
        # 如果未提供输入数据 X，则使用训练集中的数据计算邻居
        # 如果未指定半径，则使用默认的半径值
        # mode 参数指定要返回的表示模式，默认为 "connectivity" 表示连接性图
        # 如果指定对结果进行排序，则按某种顺序对结果进行排序
```