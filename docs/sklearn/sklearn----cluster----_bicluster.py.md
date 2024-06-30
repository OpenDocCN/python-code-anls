# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_bicluster.py`

```
"""Spectral biclustering algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np
from scipy.linalg import norm
from scipy.sparse import dia_matrix, issparse
from scipy.sparse.linalg import eigsh, svds

# 导入基础类和混合类
from ..base import BaseEstimator, BiclusterMixin, _fit_context
from ..utils import check_random_state, check_scalar
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import make_nonnegative, randomized_svd, safe_sparse_dot
from ..utils.validation import assert_all_finite
from ._kmeans import KMeans, MiniBatchKMeans

# 暴露给外部的类列表
__all__ = ["SpectralCoclustering", "SpectralBiclustering"]

# 函数：对矩阵 X 进行行列独立缩放归一化处理
def _scale_normalize(X):
    """Normalize ``X`` by scaling rows and columns independently.

    Returns the normalized matrix and the row and column scaling
    factors.
    """
    # 将矩阵 X 转换为非负矩阵
    X = make_nonnegative(X)
    # 计算行缩放因子
    row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1))).squeeze()
    # 计算列缩放因子
    col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0))).squeeze()
    # 处理可能存在的 NaN 值
    row_diag = np.where(np.isnan(row_diag), 0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0, col_diag)
    # 根据稀疏性进行不同的缩放处理
    if issparse(X):
        n_rows, n_cols = X.shape
        r = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
        c = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
        an = r * X * c
    else:
        an = row_diag[:, np.newaxis] * X * col_diag
    return an, row_diag, col_diag

# 函数：对矩阵 X 进行双随机归一化处理
def _bistochastic_normalize(X, max_iter=1000, tol=1e-5):
    """Normalize rows and columns of ``X`` simultaneously so that all
    rows sum to one constant and all columns sum to a different
    constant.
    """
    # 根据论文，可以通过偏差减少和平衡算法更有效地进行处理
    X = make_nonnegative(X)
    X_scaled = X
    for _ in range(max_iter):
        # 使用独立缩放归一化处理矩阵 X_scaled
        X_new, _, _ = _scale_normalize(X_scaled)
        # 计算当前矩阵与归一化后矩阵的差异
        if issparse(X):
            dist = norm(X_scaled.data - X.data)
        else:
            dist = norm(X_scaled - X_new)
        X_scaled = X_new
        # 如果差异小于给定的容差值，则结束迭代
        if dist is not None and dist < tol:
            break
    return X_scaled

# 函数：根据 Kluger 的对数交互方案对矩阵 X 进行归一化处理
def _log_normalize(X):
    """Normalize ``X`` according to Kluger's log-interactions scheme."""
    # 将矩阵 X 转换为非负矩阵，确保最小值为 1
    X = make_nonnegative(X, min_value=1)
    # 稀疏矩阵无法进行对数运算，抛出异常
    if issparse(X):
        raise ValueError(
            "Cannot compute log of a sparse matrix,"
            " because log(x) diverges to -infinity as x"
            " goes to 0."
        )
    # 计算对数交互方案的归一化矩阵 L
    L = np.log(X)
    row_avg = L.mean(axis=1)[:, np.newaxis]
    col_avg = L.mean(axis=0)
    avg = L.mean()
    return L - row_avg - col_avg + avg

# 抽象基类：用于谱双聚类的基础类
class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for spectral biclustering."""
    # 参数约束字典，指定了每个参数的有效取值范围或类型限制
    _parameter_constraints: dict = {
        "svd_method": [StrOptions({"randomized", "arpack"})],  # svd_method 只能是 "randomized" 或 "arpack"
        "n_svd_vecs": [Interval(Integral, 0, None, closed="left"), None],  # n_svd_vecs 是大于等于 0 的整数或者未指定
        "mini_batch": ["boolean"],  # mini_batch 是布尔类型
        "init": [StrOptions({"k-means++", "random"}), np.ndarray],  # init 可以是 "k-means++" 或 "random" 字符串，或者 numpy 数组
        "n_init": [Interval(Integral, 1, None, closed="left")],  # n_init 是大于等于 1 的整数
        "random_state": ["random_state"],  # random_state 是随机状态对象
    }
    
    # 初始化方法，设定聚类算法的初始参数
    @abstractmethod
    def __init__(
        self,
        n_clusters=3,  # 聚类数目，默认为 3
        svd_method="randomized",  # SVD 方法，默认为 "randomized"
        n_svd_vecs=None,  # SVD 向量数目，默认为 None
        mini_batch=False,  # 是否使用小批量模式，默认为 False
        init="k-means++",  # 初始化方法，默认为 "k-means++"
        n_init=10,  # 算法运行的初始化次数，默认为 10
        random_state=None,  # 随机数种子，默认为 None
    ):
        self.n_clusters = n_clusters  # 初始化聚类数目
        self.svd_method = svd_method  # 初始化 SVD 方法
        self.n_svd_vecs = n_svd_vecs  # 初始化 SVD 向量数目
        self.mini_batch = mini_batch  # 初始化是否使用小批量模式
        self.init = init  # 初始化聚类算法的初始化方法
        self.n_init = n_init  # 初始化算法运行的初始化次数
        self.random_state = random_state  # 初始化随机数种子
    
    # 参数检查方法，根据输入数据验证参数的有效性
    @abstractmethod
    def _check_parameters(self, n_samples):
        """Validate parameters depending on the input data."""
    
    # 拟合方法修饰器，创建 X 的双聚类
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Create a biclustering for X.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
    
        y : Ignored
            Not used, present for API consistency by convention.
    
        Returns
        -------
        self : object
            SpectralBiclustering instance.
        """
        X = self._validate_data(X, accept_sparse="csr", dtype=np.float64)  # 验证并转换输入数据 X
        self._check_parameters(X.shape[0])  # 检查参数的有效性，根据样本数目进行验证
        self._fit(X)  # 调用内部方法进行拟合
        return self  # 返回当前实例
    def _svd(self, array, n_components, n_discard):
        """Returns first `n_components` left and right singular
        vectors u and v, discarding the first `n_discard`.
        """
        # 如果使用随机化方法进行奇异值分解
        if self.svd_method == "randomized":
            kwargs = {}
            # 如果指定了奇异值向量的数量，设置参数 n_oversamples
            if self.n_svd_vecs is not None:
                kwargs["n_oversamples"] = self.n_svd_vecs
            # 调用 randomized_svd 函数进行奇异值分解
            u, _, vt = randomized_svd(
                array, n_components, random_state=self.random_state, **kwargs
            )

        # 如果使用 arpack 方法进行奇异值分解
        elif self.svd_method == "arpack":
            # 使用 svds 函数进行奇异值分解，设置 k 为分解的组件数量，ncv 为奇异值向量的数量
            u, _, vt = svds(array, k=n_components, ncv=self.n_svd_vecs)
            # 如果 vt 中存在 NaN 值
            if np.any(np.isnan(vt)):
                # 由于 A * A.T 的某些特征值为负数，导致 sqrt() 后为 np.nan，vt 中的某些向量会为 np.nan
                A = safe_sparse_dot(array.T, array)
                random_state = check_random_state(self.random_state)
                # 使用 [-1, 1] 的范围初始化 v0，类似 ARPACK 的方法
                v0 = random_state.uniform(-1, 1, A.shape[0])
                # 调用 eigsh 函数计算 A 的特征值和特征向量
                _, v = eigsh(A, ncv=self.n_svd_vecs, v0=v0)
                vt = v.T
            # 如果 u 中存在 NaN 值
            if np.any(np.isnan(u)):
                A = safe_sparse_dot(array, array.T)
                random_state = check_random_state(self.random_state)
                v0 = random_state.uniform(-1, 1, A.shape[0])
                _, u = eigsh(A, ncv=self.n_svd_vecs, v0=v0)

        # 确保 u 和 vt 中的所有值都是有限的
        assert_all_finite(u)
        assert_all_finite(vt)
        # 返回去掉前 n_discard 列之后的 u 和 vt 的转置
        u = u[:, n_discard:]
        vt = vt[n_discard:]
        return u, vt.T

    def _k_means(self, data, n_clusters):
        # 如果使用 MiniBatchKMeans 算法
        if self.mini_batch:
            # 初始化 MiniBatchKMeans 模型
            model = MiniBatchKMeans(
                n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
            )
        else:
            # 初始化 KMeans 模型
            model = KMeans(
                n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
            )
        # 使用数据拟合模型
        model.fit(data)
        # 获取聚类中心点
        centroid = model.cluster_centers_
        # 获取数据点的标签
        labels = model.labels_
        return centroid, labels

    def _more_tags(self):
        # 返回额外的标签，用于测试
        return {
            "_xfail_checks": {
                "check_estimators_dtypes": "raises nan error",
                "check_fit2d_1sample": "_scale_normalize fails",
                "check_fit2d_1feature": "raises apply_along_axis error",
                "check_estimator_sparse_matrix": "does not fail gracefully",
                "check_estimator_sparse_array": "does not fail gracefully",
                "check_methods_subset_invariance": "empty array passed inside",
                "check_dont_overwrite_parameters": "empty array passed inside",
                "check_fit2d_predict1d": "empty array passed inside",
            }
        }
# 定义一个基于谱共聚类的类，继承自BaseSpectral
class SpectralCoclustering(BaseSpectral):
    """Spectral Co-Clustering algorithm (Dhillon, 2001).

    Clusters rows and columns of an array `X` to solve the relaxed
    normalized cut of the bipartite graph created from `X` as follows:
    the edge between row vertex `i` and column vertex `j` has weight
    `X[i, j]`.

    The resulting bicluster structure is block-diagonal, since each
    row and each column belongs to exactly one bicluster.

    Supports sparse matrices, as long as they are nonnegative.

    Read more in the :ref:`User Guide <spectral_coclustering>`.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of biclusters to find.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', use
        :func:`sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', use
        :func:`scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized'.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'}, or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        The bicluster label of each row.

    column_labels_ : array-like of shape (n_cols,)
        The bicluster label of each column.

    biclusters_ : tuple of two ndarrays
        The tuple contains the `rows_` and `columns_` arrays.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """
    # 类的主体部分未提供，注释仅包括类的文档字符串和参数说明
    _parameter_constraints: dict = {
        **BaseSpectral._parameter_constraints,
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
    }



    # 定义参数约束字典，继承自BaseSpectral的参数约束并添加了针对'n_clusters'的限制条件
    _parameter_constraints: dict = {
        **BaseSpectral._parameter_constraints,
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
    }



    def __init__(
        self,
        n_clusters=3,
        *,
        svd_method="randomized",
        n_svd_vecs=None,
        mini_batch=False,
        init="k-means++",
        n_init=10,
        random_state=None,
    ):



    # 构造函数初始化方法，设置了多个参数，包括聚类数目（n_clusters）、SVD方法（svd_method）、
    # SVD向量数目（n_svd_vecs）、是否使用小批量（mini_batch）、初始化方法（init）、
    # 初始化次数（n_init）、随机状态（random_state）
    def __init__(
        self,
        n_clusters=3,
        *,
        svd_method="randomized",
        n_svd_vecs=None,
        mini_batch=False,
        init="k-means++",
        n_init=10,
        random_state=None,
    ):



        super().__init__(
            n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state
        )



        # 调用父类的初始化方法，传递聚类数目、SVD方法、SVD向量数目、是否使用小批量、
        # 初始化方法、初始化次数、随机状态等参数
        super().__init__(
            n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state
        )



    def _check_parameters(self, n_samples):
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters should be <= n_samples={n_samples}. Got"
                f" {self.n_clusters} instead."
            )



    # 检查参数的有效性，确保聚类数目不超过样本数目n_samples，否则抛出值错误异常
    def _check_parameters(self, n_samples):
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters should be <= n_samples={n_samples}. Got"
                f" {self.n_clusters} instead."
            )



    def _fit(self, X):
        normalized_data, row_diag, col_diag = _scale_normalize(X)
        n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        u, v = self._svd(normalized_data, n_sv, n_discard=1)
        z = np.vstack((row_diag[:, np.newaxis] * u, col_diag[:, np.newaxis] * v))

        _, labels = self._k_means(z, self.n_clusters)

        n_rows = X.shape[0]
        self.row_labels_ = labels[:n_rows]
        self.column_labels_ = labels[n_rows:]

        self.rows_ = np.vstack([self.row_labels_ == c for c in range(self.n_clusters)])
        self.columns_ = np.vstack(
            [self.column_labels_ == c for c in range(self.n_clusters)]
        )



    # 私有方法，执行聚类的拟合过程，包括数据归一化、SVD分解、K均值聚类，并设置行和列的标签和指示矩阵
    def _fit(self, X):
        normalized_data, row_diag, col_diag = _scale_normalize(X)
        n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        u, v = self._svd(normalized_data, n_sv, n_discard=1)
        z = np.vstack((row_diag[:, np.newaxis] * u, col_diag[:, np.newaxis] * v))

        _, labels = self._k_means(z, self.n_clusters)

        n_rows = X.shape[0]
        self.row_labels_ = labels[:n_rows]
        self.column_labels_ = labels[n_rows:]

        self.rows_ = np.vstack([self.row_labels_ == c for c in range(self.n_clusters)])
        self.columns_ = np.vstack(
            [self.column_labels_ == c for c in range(self.n_clusters)]
        )
# 定义了一个名为 SpectralBiclustering 的类，继承自 BaseSpectral 类
class SpectralBiclustering(BaseSpectral):
    """Spectral biclustering (Kluger, 2003).

    Partitions rows and columns under the assumption that the data has
    an underlying checkerboard structure. For instance, if there are
    two row partitions and three column partitions, each row will
    belong to three biclusters, and each column will belong to two
    biclusters. The outer product of the corresponding row and column
    label vectors gives this checkerboard structure.

    Read more in the :ref:`User Guide <spectral_biclustering>`.

    Parameters
    ----------
    n_clusters : int or tuple (n_row_clusters, n_column_clusters), default=3
        The number of row and column clusters in the checkerboard
        structure.

    method : {'bistochastic', 'scale', 'log'}, default='bistochastic'
        Method of normalizing and converting singular vectors into
        biclusters. May be one of 'scale', 'bistochastic', or 'log'.
        The authors recommend using 'log'. If the data is sparse,
        however, log normalization will not work, which is why the
        default is 'bistochastic'.

        .. warning::
           if `method='log'`, the data must not be sparse.

    n_components : int, default=6
        Number of singular vectors to check.

    n_best : int, default=3
        Number of best singular vectors to which to project the data
        for clustering.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', uses
        :func:`~sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', uses
        `scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized'.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features), \
            default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------

    """
    _parameter_constraints: dict = {
        **BaseSpectral._parameter_constraints,
        "n_clusters": [Interval(Integral, 1, None, closed="left"), tuple],
        "method": [StrOptions({"bistochastic", "scale", "log"})],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_best": [Interval(Integral, 1, None, closed="left")],
    }



# 定义参数约束字典，包含基类 BaseSpectral 的参数约束并扩展特定于 SpectralBiclustering 的约束
_parameter_constraints: dict = {
    **BaseSpectral._parameter_constraints,  # 继承基类的参数约束
    "n_clusters": [Interval(Integral, 1, None, closed="left"), tuple],  # n_clusters 参数的约束条件
    "method": [StrOptions({"bistochastic", "scale", "log"})],  # method 参数的取值约束
    "n_components": [Interval(Integral, 1, None, closed="left")],  # n_components 参数的约束条件
    "n_best": [Interval(Integral, 1, None, closed="left")],  # n_best 参数的约束条件
}



    def __init__(
        self,
        n_clusters=3,
        *,
        method="bistochastic",
        n_components=6,
        n_best=3,
        svd_method="randomized",
        n_svd_vecs=None,
        mini_batch=False,
        init="k-means++",
        n_init=10,
        random_state=None,
    ):
        super().__init__(
            n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state
        )
        self.method = method
        self.n_components = n_components
        self.n_best = n_best



# 构造函数初始化 SpectralBiclustering 类的实例
def __init__(
    self,
    n_clusters=3,                   # 聚类簇数，默认为 3
    *,
    method="bistochastic",          # 使用的方法，默认为 "bistochastic"
    n_components=6,                 # 分解的成分数，默认为 6
    n_best=3,                       # 最优解的数量，默认为 3
    svd_method="randomized",        # SVD 方法，默认为 "randomized"
    n_svd_vecs=None,                # SVD 向量数，默认为 None
    mini_batch=False,               # 是否使用小批量，默认为 False
    init="k-means++",               # 初始化方法，默认为 "k-means++"
    n_init=10,                      # 运行 k-means 初始化的次数，默认为 10
    random_state=None,              # 随机数种子，默认为 None
):
    super().__init__(               # 调用父类构造函数，传递参数
        n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state
    )
    self.method = method            # 设置当前对象的 method 属性
    self.n_components = n_components  # 设置当前对象的 n_components 属性
    self.n_best = n_best            # 设置当前对象的 n_best 属性



# 初始化函数说明：
# 初始化 SpectralBiclustering 类的实例，设置各种参数和属性，调用父类构造函数来完成基本的设置。
# method 控制使用的方法，n_components 控制分解的成分数，n_best 控制保留的最优解数。
    # 检查聚类算法参数的有效性，确保 n_clusters 不超过样本数 n_samples
    def _check_parameters(self, n_samples):
        if isinstance(self.n_clusters, Integral):
            # 如果 n_clusters 是整数，则检查其是否大于样本数 n_samples
            if self.n_clusters > n_samples:
                raise ValueError(
                    f"n_clusters should be <= n_samples={n_samples}. Got"
                    f" {self.n_clusters} instead."
                )
        else:  # 如果 n_clusters 是元组
            try:
                # 尝试解包元组获取 n_row_clusters 和 n_column_clusters
                n_row_clusters, n_column_clusters = self.n_clusters
                # 检查 n_row_clusters 是否为整数，并在范围 [1, n_samples] 内
                check_scalar(
                    n_row_clusters,
                    "n_row_clusters",
                    target_type=Integral,
                    min_val=1,
                    max_val=n_samples,
                )
                # 检查 n_column_clusters 是否为整数，并在范围 [1, n_samples] 内
                check_scalar(
                    n_column_clusters,
                    "n_column_clusters",
                    target_type=Integral,
                    min_val=1,
                    max_val=n_samples,
                )
            except (ValueError, TypeError) as e:
                # 捕获解包或类型错误，抛出指示 n_clusters 参数错误的 ValueError
                raise ValueError(
                    "Incorrect parameter n_clusters has value:"
                    f" {self.n_clusters}. It should either be a single integer"
                    " or an iterable with two integers:"
                    " (n_row_clusters, n_column_clusters)"
                    " And the values are should be in the"
                    " range: (1, n_samples)"
                ) from e

        # 检查 n_best 是否小于等于 n_components
        if self.n_best > self.n_components:
            raise ValueError(
                f"n_best={self.n_best} must be <= n_components={self.n_components}."
            )

    # 执行主要的拟合过程，根据选择的方法对输入数据 X 进行规范化，并进行奇异值分解
    def _fit(self, X):
        # 初始化需要保留的奇异值个数为 self.n_components
        n_sv = self.n_components
        # 根据选择的方法对数据进行规范化和处理
        if self.method == "bistochastic":
            # 如果方法是 "bistochastic"，则对 X 进行双随机规范化，并增加一个奇异值
            normalized_data = _bistochastic_normalize(X)
            n_sv += 1
        elif self.method == "scale":
            # 如果方法是 "scale"，则对 X 进行尺度规范化，并增加一个奇异值
            normalized_data, _, _ = _scale_normalize(X)
            n_sv += 1
        elif self.method == "log":
            # 如果方法是 "log"，则对 X 进行对数规范化
            normalized_data = _log_normalize(X)
        # 根据不同的方法确定要丢弃的奇异值个数
        n_discard = 0 if self.method == "log" else 1
        # 对规范化后的数据进行奇异值分解，得到左奇异向量 u 和右奇异向量 v
        u, v = self._svd(normalized_data, n_sv, n_discard)
        # 计算 u 和 v 的转置
        ut = u.T
        vt = v.T

        # 尝试获取 n_row_clusters 和 n_col_clusters，如果 n_clusters 是元组的话
        try:
            n_row_clusters, n_col_clusters = self.n_clusters
        except TypeError:
            # 如果 n_clusters 不是元组，则将其视为两个相同的值
            n_row_clusters = n_col_clusters = self.n_clusters

        # 在 ut 上执行最佳拟合分段聚类，返回最佳的 ut
        best_ut = self._fit_best_piecewise(ut, self.n_best, n_row_clusters)

        # 在 vt 上执行最佳拟合分段聚类，返回最佳的 vt
        best_vt = self._fit_best_piecewise(vt, self.n_best, n_col_clusters)

        # 根据最佳的 vt 对 X 进行投影和聚类，得到行标签
        self.row_labels_ = self._project_and_cluster(X, best_vt.T, n_row_clusters)

        # 根据最佳的 ut 对 X 的转置进行投影和聚类，得到列标签
        self.column_labels_ = self._project_and_cluster(X.T, best_ut.T, n_col_clusters)

        # 根据行标签和列标签创建行的二进制标记矩阵
        self.rows_ = np.vstack(
            [
                self.row_labels_ == label
                for label in range(n_row_clusters)
                for _ in range(n_col_clusters)
            ]
        )

        # 根据行标签和列标签创建列的二进制标记矩阵
        self.columns_ = np.vstack(
            [
                self.column_labels_ == label
                for _ in range(n_row_clusters)
                for label in range(n_col_clusters)
            ]
        )
    # 定义一个私有方法 `_fit_best_piecewise`，用于找到最能通过分段常数向量近似的前 `n_best` 个向量。
    # 分段向量是通过 k-means 算法找到的；根据欧几里得距离选择最佳向量。

    def _fit_best_piecewise(self, vectors, n_best, n_clusters):
        """Find the ``n_best`` vectors that are best approximated by piecewise
        constant vectors.

        The piecewise vectors are found by k-means; the best is chosen
        according to Euclidean distance.

        """
        
        # 定义一个局部函数 `make_piecewise`，将输入向量 `v` 转换为分段向量
        def make_piecewise(v):
            # 对向量 v 进行 k-means 聚类，返回聚类中心和标签
            centroid, labels = self._k_means(v.reshape(-1, 1), n_clusters)
            # 返回根据标签选择的分段向量，将其展平为一维数组
            return centroid[labels].ravel()
        
        # 对输入的 `vectors` 数组中的每个向量应用 `make_piecewise` 函数，得到分段向量数组
        piecewise_vectors = np.apply_along_axis(make_piecewise, axis=1, arr=vectors)
        
        # 计算每个原始向量与其对应的分段向量之间的欧几里得距离
        dists = np.apply_along_axis(norm, axis=1, arr=(vectors - piecewise_vectors))
        
        # 根据距离的排序，选择距离最小的前 `n_best` 个原始向量作为结果
        result = vectors[np.argsort(dists)[:n_best]]
        
        # 返回最终选择的向量数组作为结果
        return result

    # 定义一个私有方法 `_project_and_cluster`，用于将 `data` 投影到 `vectors` 上并对结果进行聚类。
    def _project_and_cluster(self, data, vectors, n_clusters):
        """Project ``data`` to ``vectors`` and cluster the result."""
        
        # 将 `data` 投影到 `vectors` 上，得到投影后的结果
        projected = safe_sparse_dot(data, vectors)
        
        # 对投影后的结果进行 k-means 聚类，返回聚类结果的标签
        _, labels = self._k_means(projected, n_clusters)
        
        # 返回聚类结果的标签作为最终输出
        return labels
```