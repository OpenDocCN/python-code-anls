# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_unsupervised.py`

```
# 导入必要的模块和类
"""Unsupervised nearest neighbors learner"""

# 导入相关的基础模块和类
from ..base import _fit_context
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin


# 定义 NearestNeighbors 类，继承了 KNeighborsMixin, RadiusNeighborsMixin 和 NeighborsBase
class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
    """Unsupervised learner for implementing neighbor searches.

    Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    p : float (positive), default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    # n_jobs : int, default=None
    # 定义并行作业的数量，用于邻居搜索。
    # ``None`` 表示默认为1，除非在 :obj:`joblib.parallel_backend` 上下文中。
    # ``-1`` 表示使用所有处理器。有关详细信息，请参见 :term:`Glossary <n_jobs>`。

    Attributes
    ----------
    # effective_metric_ : str
    # 用于计算到邻居的距离的度量标准。

    # effective_metric_params_ : dict
    # 用于计算到邻居的距离的度量标准的参数。

    # n_features_in_ : int
    # :term:`fit` 过程中看到的特征数量。
    # .. versionadded:: 0.24

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    # 在 `X` 具有所有字符串特征名时定义的特征名称列表。
    # .. versionadded:: 1.0

    # n_samples_fit_ : int
    # 拟合数据中的样本数量。

    See Also
    --------
    # KNeighborsClassifier : 实现 k-最近邻分类器。
    # RadiusNeighborsClassifier : 在给定半径内实现邻居投票的分类器。
    # KNeighborsRegressor : 基于 k-最近邻的回归。
    # RadiusNeighborsRegressor : 基于固定半径内邻居的回归。
    # BallTree : 多维空间中的空间划分数据结构，用于最近邻搜索。

    Notes
    -----
    # 有关 ``algorithm`` 和 ``leaf_size`` 选择的讨论，请参阅在线文档中的 :ref:`Nearest Neighbors <neighbors>`。

    # https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    Examples
    --------
    # 导入必要的库
    >>> import numpy as np
    >>> from sklearn.neighbors import NearestNeighbors
    # 创建样本数据
    >>> samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
    # 创建最近邻分类器对象
    >>> neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
    # 使用样本数据进行拟合
    >>> neigh.fit(samples)
    # NearestNeighbors(...)
    # 查找最近邻的示例
    >>> neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
    array([[2, 0]]...)
    # 使用指定半径查找邻居
    >>> nbrs = neigh.radius_neighbors(
    ...    [[0, 0, 1.3]], 0.4, return_distance=False
    ... )
    # 将结果转换为数组格式
    >>> np.asarray(nbrs[0][0])
    array(2)
    """
    
    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        # 调用父类的初始化方法，设置分类器的参数
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    @_fit_context(
        # NearestNeighbors.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """
        Fit the nearest neighbors estimator from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data. It can be either a 2D array of shape (n_samples, n_features)
            or a square matrix of shape (n_samples, n_samples) if 'precomputed' is specified
            for the metric parameter.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator. Returns the instance itself after fitting
            to allow method chaining and consistent API usage.
        """
        # Call the internal method _fit() with the provided training data X
        return self._fit(X)
```