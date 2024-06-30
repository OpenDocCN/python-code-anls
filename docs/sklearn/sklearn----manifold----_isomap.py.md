# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_isomap.py`

```
"""Isomap for manifold learning"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块，用于处理警告信息
import warnings
# 导入整数和实数类型判断模块
from numbers import Integral, Real

# 导入NumPy库，用于科学计算
import numpy as np
# 导入稀疏矩阵判断函数
from scipy.sparse import issparse
# 导入稀疏图算法相关函数
from scipy.sparse.csgraph import connected_components, shortest_path

# 导入基础估计器、特征输出前缀混合类、变换器混合类和_fit_context
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
# 导入核PCA（主成分分析）模块
from ..decomposition import KernelPCA
# 导入有效距离度量列表
from ..metrics.pairwise import _VALID_METRICS
# 导入最近邻模型、最近邻图生成函数
from ..neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
# 导入核中心化预处理模块
from ..preprocessing import KernelCenterer
# 导入参数验证模块：区间、字符串选项
from ..utils._param_validation import Interval, StrOptions
# 导入图相关工具函数：修复连接组件
from ..utils.graph import _fix_connected_components
# 导入验证模块：检查估计器是否已拟合
from ..utils.validation import check_is_fitted


class Isomap(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Isomap Embedding.

    Non-linear dimensionality reduction through Isometric Mapping

    Read more in the :ref:`User Guide <isomap>`.

    Parameters
    ----------
    n_neighbors : int or None, default=5
        Number of neighbors to consider for each point. If `n_neighbors` is an int,
        then `radius` must be `None`.

    radius : float or None, default=None
        Limiting distance of neighbors to return. If `radius` is a float,
        then `n_neighbors` must be set to `None`.

        .. versionadded:: 1.1

    n_components : int, default=2
        Number of coordinates for the manifold.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        'auto' : Attempt to choose the most efficient solver
        for the given problem.

        'arpack' : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.

        'dense' : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.

    tol : float, default=0
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.

    max_iter : int, default=None
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.

    path_method : {'auto', 'FW', 'D'}, default='auto'
        Method to use in finding shortest path.

        'auto' : attempt to choose the best algorithm automatically.

        'FW' : Floyd-Warshall algorithm.

        'D' : Dijkstra's algorithm.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                          default='auto'
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    metric : str, or callable, default="minkowski"
        距离度量标准，用于计算特征数组中实例之间的距离。如果metric是字符串或可调用对象，
        必须是`sklearn.metrics.pairwise_distances`函数中metric参数允许的选项之一。

        .. versionadded:: 0.22

    p : float, default=2
        Minkowski度量的参数，用于`sklearn.metrics.pairwise.pairwise_distances`。
        当p = 1时，等效于曼哈顿距离（l1），当p = 2时，等效于欧氏距离（l2）。对于任意p，
        使用Minkowski距离（l_p）。

        .. versionadded:: 0.22

    metric_params : dict, default=None
        距离度量函数的额外关键字参数。

        .. versionadded:: 0.22

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        存储嵌入向量。

    kernel_pca_ : object
        用于实现嵌入的:class:`~sklearn.decomposition.KernelPCA`对象。

    nbrs_ : sklearn.neighbors.NearestNeighbors 实例
        存储最近邻实例，包括BallTree或KDTree（如果适用）。

    dist_matrix_ : array-like, shape (n_samples, n_samples)
        存储训练数据的测地距离矩阵。

    n_features_in_ : int
        在`fit`期间观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在`fit`期间观察到的特征名称。仅当`X`具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.decomposition.PCA : 线性的主成分分析方法。
    sklearn.decomposition.KernelPCA : 使用核函数和PCA的非线性降维方法。
    MDS : 使用多维缩放进行流形学习。
    TSNE : 使用t分布随机近邻嵌入。
    LocallyLinearEmbedding : 使用局部线性嵌入进行流形学习。
    SpectralEmbedding : 用于非线性降维的谱嵌入方法。

    References
    ----------

    .. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
           framework for nonlinear dimensionality reduction. Science 290 (5500)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import Isomap
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = Isomap(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    # 参数约束字典，定义了每个参数的类型和约束条件
    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],  # n_neighbors参数为整数且至少为1
        "radius": [Interval(Real, 0, None, closed="both"), None],  # radius参数为实数且大于等于0
        "n_components": [Interval(Integral, 1, None, closed="left")],  # n_components参数为整数且至少为1
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],  # eigen_solver参数为字符串，取值为{"auto", "arpack", "dense"}中的一个
        "tol": [Interval(Real, 0, None, closed="left")],  # tol参数为实数且大于0
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],  # max_iter参数为整数且至少为1
        "path_method": [StrOptions({"auto", "FW", "D"})],  # path_method参数为字符串，取值为{"auto", "FW", "D"}中的一个
        "neighbors_algorithm": [StrOptions({"auto", "brute", "kd_tree", "ball_tree"})],  # neighbors_algorithm参数为字符串，取值为{"auto", "brute", "kd_tree", "ball_tree"}中的一个
        "n_jobs": [Integral, None],  # n_jobs参数为整数或者None
        "p": [Interval(Real, 1, None, closed="left")],  # p参数为实数且大于等于1
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],  # metric参数为字符串，取值为_VALID_METRICS集合并包含"precomputed"，或者为可调用对象
        "metric_params": [dict, None],  # metric_params参数为字典或者None
    }

    # 初始化函数，设置Isomap的各个参数
    def __init__(
        self,
        *,
        n_neighbors=5,  # 默认值为5
        radius=None,  # 默认值为None
        n_components=2,  # 默认值为2
        eigen_solver="auto",  # 默认值为"auto"
        tol=0,  # 默认值为0
        max_iter=None,  # 默认值为None
        path_method="auto",  # 默认值为"auto"
        neighbors_algorithm="auto",  # 默认值为"auto"
        n_jobs=None,  # 默认值为None
        metric="minkowski",  # 默认值为"minkowski"
        p=2,  # 默认值为2
        metric_params=None,  # 默认值为None
    ):
        # 设置各个参数到实例变量
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    # 计算重构误差的方法
    def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.

        Returns
        -------
        reconstruction_error : float
            Reconstruction error.

        Notes
        -----
        The cost function of an isomap embedding is

        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:

        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        # 计算 G 矩阵，表示距离矩阵的平方乘以-0.5
        G = -0.5 * self.dist_matrix_**2
        # 对 G 矩阵进行中心化处理
        G_center = KernelCenterer().fit_transform(G)
        # 获取核PCA的特征值
        evals = self.kernel_pca_.eigenvalues_
        # 返回重构误差，使用给定公式计算
        return np.sqrt(np.sum(G_center**2) - np.sum(evals**2)) / G.shape[0]

    # 标注修饰器，但是具体作用尚未验证
    @_fit_context(
        # Isomap.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    # 计算数据 X 的嵌入向量。
    def fit(self, X, y=None):
        """Compute the embedding vectors for data X.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse matrix, precomputed tree, or NearestNeighbors
            object.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        # 调用 _fit_transform 方法进行拟合
        self._fit_transform(X)
        # 返回拟合后的实例 self
        return self

    @_fit_context(
        # Isomap.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    # 进行拟合并返回转换后的数据 X
    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            X transformed in the new space.
        """
        # 调用 _fit_transform 方法进行拟合
        self._fit_transform(X)
        # 返回嵌入后的数据 X
        return self.embedding_
    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features)
            If neighbors_algorithm='precomputed', X is assumed to be a
            distance matrix or a sparse graph of shape
            (n_queries, n_samples_fit).

        Returns
        -------
        X_new : array-like, shape (n_queries, n_components)
            X transformed in the new space.
        """
        # Ensure the model is fitted before transforming
        check_is_fitted(self)
        
        # Determine whether to use k-neighbors or radius-neighbors
        if self.n_neighbors is not None:
            # Compute distances and indices of the k-nearest neighbors of X
            distances, indices = self.nbrs_.kneighbors(X, return_distance=True)
        else:
            # Compute distances and indices of neighbors within a specified radius of X
            distances, indices = self.nbrs_.radius_neighbors(X, return_distance=True)

        # Create the graph of shortest distances from X to training data via nearest neighbors
        # Initialize an array to hold geodesic distances
        n_samples_fit = self.nbrs_.n_samples_fit_
        n_queries = distances.shape[0]

        # Determine the dtype for the computation
        if hasattr(X, "dtype") and X.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64

        # Initialize the graph matrix G_X
        G_X = np.zeros((n_queries, n_samples_fit), dtype=dtype)
        
        # Populate G_X using a loop to avoid excessive memory usage
        for i in range(n_queries):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] + distances[i][:, None], 0)

        # Apply the kernel transformation to G_X
        G_X **= 2
        G_X *= -0.5

        # Transform G_X using kernel PCA and return the result
        return self.kernel_pca_.transform(G_X)

    def _more_tags(self):
        # Additional tags for the estimator
        return {"preserves_dtype": [np.float64, np.float32]}
```