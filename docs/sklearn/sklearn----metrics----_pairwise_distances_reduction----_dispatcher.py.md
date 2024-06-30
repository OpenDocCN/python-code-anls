# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_pairwise_distances_reduction\_dispatcher.py`

```
    from abc import abstractmethod  # 导入抽象基类模块中的abstractmethod装饰器
    from typing import List  # 导入用于类型提示的List类型

    import numpy as np  # 导入NumPy库，用于数值计算
    from scipy.sparse import issparse  # 从SciPy稀疏矩阵模块中导入issparse函数

    from ... import get_config  # 导入项目中的get_config函数
    from .._dist_metrics import (  # 导入距离度量相关模块
        BOOL_METRICS,  # 布尔型距离度量列表
        METRIC_MAPPING64,  # 64位度量映射字典
        DistanceMetric,  # 距离度量基类
    )
    from ._argkmin import (  # 导入最小K参数相关模块
        ArgKmin32,  # 32位参数K最小值
        ArgKmin64,  # 64位参数K最小值
    )
    from ._argkmin_classmode import (  # 导入类模式参数K最小值相关模块
        ArgKminClassMode32,  # 32位类模式参数K最小值
        ArgKminClassMode64,  # 64位类模式参数K最小值
    )
    from ._base import _sqeuclidean_row_norms32, _sqeuclidean_row_norms64  # 导入基础模块中的行欧氏距离平方计算函数
    from ._radius_neighbors import (  # 导入半径邻居相关模块
        RadiusNeighbors32,  # 32位半径邻居
        RadiusNeighbors64,  # 64位半径邻居
    )
    from ._radius_neighbors_classmode import (  # 导入类模式半径邻居相关模块
        RadiusNeighborsClassMode32,  # 32位类模式半径邻居
        RadiusNeighborsClassMode64,  # 64位类模式半径邻居
    )


def sqeuclidean_row_norms(X, num_threads):
    """Compute the squared euclidean norm of the rows of X in parallel.

    Parameters
    ----------
    X : ndarray or CSR matrix of shape (n_samples, n_features)
        Input data. Must be c-contiguous.

    num_threads : int
        The number of OpenMP threads to use.

    Returns
    -------
    sqeuclidean_row_norms : ndarray of shape (n_samples,)
        Arrays containing the squared euclidean norm of each row of X.
    """
    if X.dtype == np.float64:
        return np.asarray(_sqeuclidean_row_norms64(X, num_threads))  # 如果X的数据类型为float64，调用64位行欧氏距离平方计算函数并返回结果数组
    if X.dtype == np.float32:
        return np.asarray(_sqeuclidean_row_norms32(X, num_threads))  # 如果X的数据类型为float32，调用32位行欧氏距离平方计算函数并返回结果数组

    raise ValueError(
        "Only float64 or float32 datasets are supported at this time, "
        f"got: X.dtype={X.dtype}."
    )  # 如果X的数据类型不是float64或float32，则抛出值错误异常


class BaseDistancesReductionDispatcher:
    """Abstract base dispatcher for pairwise distance computation & reduction.

    Each dispatcher extending the base :class:`BaseDistancesReductionDispatcher`
    dispatcher must implement the :meth:`compute` classmethod.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        excluded = {
            # PyFunc cannot be supported because it necessitates interacting with
            # the CPython interpreter to call user defined functions.
            "pyfunc",  # 排除"pyfunc"，因为它需要与CPython解释器交互来调用用户定义的函数
            "mahalanobis",  # 数值不稳定，排除"mahalanobis"
            # In order to support discrete distance metrics, we need to have a
            # stable simultaneous sort which preserves the order of the indices
            # because there generally is a lot of occurrences for a given values
            # of distances in this case.
            # TODO: implement a stable simultaneous_sort.
            "hamming",  # 排除"hamming"，需要实现稳定的同时排序以支持离散距离度量
            *BOOL_METRICS,  # 排除布尔型距离度量列表中的所有度量
        }
        return sorted(({"sqeuclidean"} | set(METRIC_MAPPING64.keys())) - excluded)
        # 返回支持的有效距离度量列表，包括"sqeuclidean"和64位度量映射字典的键集，排除已排除的度量
    def is_usable_for(cls, X, Y, metric) -> bool:
        """Return True if the dispatcher can be used for the
        given parameters.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.

        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)
            Input data.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        Returns
        -------
        True if the dispatcher can be used, else False.
        """

        # FIXME: the current Cython implementation is too slow for a large number of
        # features. We temporarily disable it to fallback on SciPy's implementation.
        # See: https://github.com/scikit-learn/scikit-learn/issues/28191
        # 检查条件：如果输入 X 和 Y 都是稀疏矩阵，并且 metric 是字符串且包含 "euclidean"
        if (
            issparse(X)
            and issparse(Y)
            and isinstance(metric, str)
            and "euclidean" in metric
        ):
            return False

        # 内部函数：检查输入的 ndarray 是否是按 C 顺序存储的
        def is_numpy_c_ordered(X):
            return hasattr(X, "flags") and getattr(X.flags, "c_contiguous", False)

        # 内部函数：检查输入的稀疏矩阵是否是有效的 CSR 格式
        def is_valid_sparse_matrix(X):
            return (
                issparse(X)
                and X.format == "csr"
                and
                # TODO: 支持 CSR 矩阵中没有非零元素的情况
                X.nnz > 0
                and
                # TODO: 支持 CSR 矩阵使用 int64 索引和 indptr
                # 参考: https://github.com/scikit-learn/scikit-learn/issues/23653
                X.indices.dtype == X.indptr.dtype == np.int32
            )

        # 判断是否可以使用当前方法进行计算的总条件
        is_usable = (
            get_config().get("enable_cython_pairwise_dist", True)
            and (is_numpy_c_ordered(X) or is_valid_sparse_matrix(X))
            and (is_numpy_c_ordered(Y) or is_valid_sparse_matrix(Y))
            and X.dtype == Y.dtype
            and X.dtype in (np.float32, np.float64)
            and (metric in cls.valid_metrics() or isinstance(metric, DistanceMetric))
        )

        return is_usable

    @classmethod
    @abstractmethod
    def compute(
        cls,
        X,
        Y,
        **kwargs,
    ):
        """Compute the reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        **kwargs : additional parameters for the reduction

        Notes
        -----
        This method is an abstract class method: it has to be implemented
        for all subclasses.
        """
class ArgKmin(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances.

    ArgKmin is typically used to perform
    bruteforce k-nearest neighbors queries.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
    ):
        """Compute k-nearest neighbors indices for each vector in X.

        Args:
            X: Array-like, shape (n_queries, n_features)
                The query vectors for which nearest neighbors are to be found.
            Y: Array-like, shape (n_samples_Y, n_features)
                The target vectors among which nearest neighbors are searched.
            k: int
                Number of nearest neighbors to find.
            metric: str, default="euclidean"
                The distance metric to use for finding nearest neighbors.
            chunk_size: int, optional
                Size of chunks to use for distance computation.
            metric_kwargs: dict, optional
                Additional keyword arguments to be passed to the distance metric function.
            strategy: str, optional
                Strategy to use for nearest neighbor search.
            return_distance: bool, default=False
                Whether to return distances along with indices.

        Returns:
            indices: Array-like, shape (n_queries, k)
                Indices of the k-nearest neighbors in Y for each query vector in X.
            distances: Array-like, shape (n_queries, k)
                Distances to the k-nearest neighbors. Returned only if return_distance=True.
        """
        pass  # Placeholder for actual computation


class RadiusNeighbors(BaseDistancesReductionDispatcher):
    """Compute radius-based neighbors for two sets of vectors.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    The distance function `dist` depends on the values of the `metric`
    and `metric_kwargs` parameters.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
        sort_results=False,
    ):
        """Compute indices of neighbors within a specified radius for each query vector in X.

        Args:
            X: Array-like, shape (n_queries, n_features)
                The query vectors for which neighbors within radius are to be found.
            Y: Array-like, shape (n_samples_Y, n_features)
                The target vectors among which neighbors are searched.
            radius: float
                Radius within which neighbors are considered.
            metric: str, default="euclidean"
                The distance metric to use for finding neighbors.
            chunk_size: int, optional
                Size of chunks to use for distance computation.
            metric_kwargs: dict, optional
                Additional keyword arguments to be passed to the distance metric function.
            strategy: str, optional
                Strategy to use for neighbor search.
            return_distance: bool, default=False
                Whether to return distances along with indices.
            sort_results: bool, default=False
                Whether to sort the results by distance.

        Returns:
            indices: List of arrays
                Indices of neighbors within the specified radius for each query vector in X.
            distances: List of arrays, optional
                Distances to the neighbors. Returned only if return_distance=True.
        """
        pass  # Placeholder for actual computation


class ArgKminClassMode(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y with labels.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances. Computes weighted mode of labels.

    ArgKminClassMode is typically used to perform bruteforce k-nearest neighbors
    queries when the weighted mode of the labels for the k-nearest neighbors
    are required, such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        """Return valid distance metrics excluding metrics unsuitable for ArgKminClassMode."""
        excluded = {
            # Euclidean is technically usable for ArgKminClassMode
            # but its current implementation would not be competitive.
            # TODO: implement Euclidean specialization using GEMM.
            "euclidean",
            "sqeuclidean",
        }
        return list(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        weights,
        Y_labels,
        unique_Y_labels,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
    ):
        """Compute k-nearest neighbors indices and weighted mode of labels for each vector in X.

        Args:
            X: Array-like, shape (n_queries, n_features)
                The query vectors for which nearest neighbors are to be found.
            Y: Array-like, shape (n_samples_Y, n_features)
                The target vectors among which nearest neighbors are searched.
            k: int
                Number of nearest neighbors to find.
            weights: Array-like, shape (n_samples_Y,)
                Weights associated with each sample in Y.
            Y_labels: Array-like, shape (n_samples_Y,)
                Labels associated with each sample in Y.
            unique_Y_labels: Array-like
                Unique labels present in Y_labels, used for mode computation.
            metric: str, default="euclidean"
                The distance metric to use for finding nearest neighbors.
            chunk_size: int, optional
                Size of chunks to use for distance computation.
            metric_kwargs: dict, optional
                Additional keyword arguments to be passed to the distance metric function.
            strategy: str, optional
                Strategy to use for nearest neighbor search.

        Returns:
            indices: Array-like, shape (n_queries, k)
                Indices of the k-nearest neighbors in Y for each query vector in X.
            mode_labels: Array-like, shape (n_queries,)
                Weighted mode of labels for the k-nearest neighbors.
        """
        pass  # Placeholder for actual computation


class RadiusNeighborsClassMode(BaseDistancesReductionDispatcher):
    """Compute radius-based class modes of row vectors of X using the
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
        sort_results=False,
    ):
        """Compute class modes of neighbors within a specified radius for each vector in X.

        Args:
            X: Array-like, shape (n_queries, n_features)
                The query vectors for which neighbors within radius are to be found.
            Y: Array-like, shape (n_samples_Y, n_features)
                The target vectors among which neighbors are searched.
            radius: float
                Radius within which neighbors are considered.
            metric: str, default="euclidean"
                The distance metric to use for finding neighbors.
            chunk_size: int, optional
                Size of chunks to use for distance computation.
            metric_kwargs: dict, optional
                Additional keyword arguments to be passed to the distance metric function.
            strategy: str, optional
                Strategy to use for neighbor search.
            return_distance: bool, default=False
                Whether to return distances along with indices.
            sort_results: bool, default=False
                Whether to sort the results by distance.

        Returns:
            mode_labels: List of arrays
                Weighted mode of labels for neighbors within the specified radius for each query vector in X.
            distances: List of arrays, optional
                Distances to the neighbors. Returned only if return_distance=True.
        """
        pass  # Placeholder for actual computation
    @classmethod
    def valid_metrics(cls) -> List[str]:
        # 定义不包含的指标集合，这些指标在 RadiusNeighborsClassMode 中不适用
        excluded = {
            # 欧氏距离在理论上可用于 RadiusNeighborsClassMode
            # 但不会具有竞争力。
            # TODO: 使用 GEMM 实现欧氏距离的专用版本。
            "euclidean",
            "sqeuclidean",
        }
        # 返回有效的度量指标列表，排除了不适用的指标
        return sorted(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        weights,
        Y_labels,
        unique_Y_labels,
        outlier_label,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,


这段代码是一个 Python 类的方法定义。以下是各行代码的注释解释：

1. `@classmethod`
   - 声明这是一个类方法，而不是实例方法。

2. `def valid_metrics(cls) -> List[str]:`
   - 定义了一个类方法 `valid_metrics`，返回一个字符串列表，表示有效的度量指标。

3. `excluded = {`
   - 创建一个集合 `excluded`，用于存储不适用于 `RadiusNeighborsClassMode` 的度量指标。

4. `"euclidean",`
   - 注释：欧氏距离在理论上可用于 `RadiusNeighborsClassMode`，但不具备竞争力。

5. `"sqeuclidean",`
   - 注释：平方欧氏距离在 `RadiusNeighborsClassMode` 中也不适用。

6. `}`
   - 结束 `excluded` 集合的定义。

7. `return sorted(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)`
   - 返回一个经过排序后的有效度量指标列表，排除了 `excluded` 中的指标。

8. `@classmethod`
   - 声明这是另一个类方法。

9. `def compute(`
   - 定义了一个名为 `compute` 的类方法，接受多个参数，用于计算操作。

这些注释详细解释了代码中每行的作用和意图。
```