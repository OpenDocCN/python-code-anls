# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_graph.py`

```
"""Nearest Neighbors graph functions"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import itertools

# 从父级目录导入指定模块和类
from ..base import ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
# 导入参数验证相关模块和函数
from ..utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)
# 导入验证模型是否拟合的函数
from ..utils.validation import check_is_fitted
# 导入K近邻相关的基础类和函数
from ._base import VALID_METRICS, KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
# 导入无监督最近邻算法的实现类
from ._unsupervised import NearestNeighbors


def _check_params(X, metric, p, metric_params):
    """Check the validity of the input parameters"""
    # 将参数名称和参数值打包成元组的迭代器
    params = zip(["metric", "p", "metric_params"], [metric, p, metric_params])
    # 获取X对象的参数字典
    est_params = X.get_params()
    # 遍历参数元组
    for param_name, func_param in params:
        # 如果函数参数与估计器参数不相等，则抛出值错误异常
        if func_param != est_params[param_name]:
            raise ValueError(
                "Got %s for %s, while the estimator has %s for the same parameter."
                % (func_param, param_name, est_params[param_name])
            )


def _query_include_self(X, include_self, mode):
    """Return the query based on include_self param"""
    # 如果include_self为"auto"，根据mode设置include_self的值
    if include_self == "auto":
        include_self = mode == "connectivity"

    # 如果不包括自身作为邻居
    if not include_self:
        X = None  # 将X设置为None

    return X  # 返回X


@validate_params(
    {
        "X": ["array-like", "sparse matrix", KNeighborsMixin],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "mode": [StrOptions({"connectivity", "distance"})],
        "metric": [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric_params": [dict, None],
        "include_self": ["boolean", StrOptions({"auto"})],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=False,  # metric is not validated yet
)
def kneighbors_graph(
    X,
    n_neighbors,
    *,
    mode="connectivity",
    metric="minkowski",
    p=2,
    metric_params=None,
    include_self=False,
    n_jobs=None,
):
    """Compute the (weighted) graph of k-Neighbors for points in X.

    Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample data.

    n_neighbors : int
        Number of neighbors for each sample.

    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    metric : {'euclidean', 'manhattan', ...} or callable, default='minkowski'
        The distance metric to use. If a callable is provided, it should
        return a distance matrix (o...
    # metric 参数，用于指定距离计算所使用的度量方式，默认为 'minkowski'
    # 当 p = 2 时，对应标准的欧氏距离
    # 更多信息请参考 `scipy.spatial.distance` 的文档
    # 以及 `sklearn.metrics.pairwise.distance_metrics` 中列出的有效度量值
    metric : str, default='minkowski'
    
    # p 参数，用于 Minkowski 距离的幂参数，默认为 2
    # 当 p = 1 时，等效于曼哈顿距离 (l1)，当 p = 2 时，等效于欧氏距离 (l2)
    # 对于任意的 p，使用 Minkowski 距离 (l_p)
    # 此参数应为正数
    p : float, default=2
    
    # metric_params 参数，用于度量函数的额外关键字参数，默认为 None
    metric_params : dict, default=None
    
    # include_self 参数，指定是否将每个样本标记为其自身的最近邻
    # 如果为 'auto'，则在 mode='connectivity' 时使用 True，在 mode='distance' 时使用 False
    include_self : bool or 'auto', default=False
    
    # n_jobs 参数，指定用于邻居搜索的并行作业数
    # None 表示使用一个作业，除非在 joblib.parallel_backend 上下文中
    # -1 表示使用所有处理器
    # 更多细节请参见术语表中的 'n_jobs'
    n_jobs : int, default=None
    
    # 返回值
    # 返回一个稀疏矩阵，形状为 (n_samples, n_samples)
    # 矩阵 A[i, j] 赋予连接 i 到 j 的边的权重
    # 矩阵采用 CSR 格式
    Returns
    -------
    A : sparse matrix of shape (n_samples, n_samples)
    
    # 参见
    # --------
    # radius_neighbors_graph: 计算 X 中点的 (加权) 邻居图
    See Also
    --------
    radius_neighbors_graph: Compute the (weighted) graph of Neighbors for points in X.
    
    # 示例
    # --------
    # >>> X = [[0], [3], [1]]
    # >>> from sklearn.neighbors import kneighbors_graph
    # >>> A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
    # >>> A.toarray()
    # array([[1., 0., 1.],
    #        [0., 1., 1.],
    #        [1., 0., 1.]])
    Examples
    --------
    """
    # 如果 X 不是 KNeighborsMixin 的实例，则创建 NearestNeighbors 对象并拟合 X
    if not isinstance(X, KNeighborsMixin):
        X = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        ).fit(X)
    else:
        # 否则，检查参数的有效性
        _check_params(X, metric, p, metric_params)
    
    # 根据 include_self 参数，获取查询数组 query
    query = _query_include_self(X._fit_X, include_self, mode)
    
    # 调用 X 的 kneighbors_graph 方法生成邻居图，并返回结果
    return X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)
@validate_params(
    {
        "X": ["array-like", "sparse matrix", RadiusNeighborsMixin],  # 参数 X 可以是数组、稀疏矩阵或 RadiusNeighborsMixin 对象
        "radius": [Interval(Real, 0, None, closed="both")],  # 参数 radius 是一个实数，范围是大于等于0的闭区间
        "mode": [StrOptions({"connectivity", "distance"})],  # 参数 mode 是一个字符串，可选值为 {'connectivity', 'distance'}
        "metric": [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable],  # 参数 metric 可以是预定义的字符串选项或者一个可调用对象
        "p": [Interval(Real, 0, None, closed="right"), None],  # 参数 p 是一个实数，范围是大于0的右闭区间，或者可以为 None
        "metric_params": [dict, None],  # 参数 metric_params 是一个字典或者可以为 None
        "include_self": ["boolean", StrOptions({"auto"})],  # 参数 include_self 是一个布尔值或者字符串选项 {'auto'}
        "n_jobs": [Integral, None],  # 参数 n_jobs 是一个整数或者可以为 None
    },
    prefer_skip_nested_validation=False,  # 是否跳过嵌套验证，默认为 False，即不跳过
)
def radius_neighbors_graph(
    X,
    radius,
    *,
    mode="connectivity",  # 默认参数 mode 是 'connectivity'
    metric="minkowski",  # 默认参数 metric 是 'minkowski'
    p=2,  # 默认参数 p 是 2
    metric_params=None,  # 默认参数 metric_params 是 None
    include_self=False,  # 默认参数 include_self 是 False
    n_jobs=None,  # 默认参数 n_jobs 是 None
):
    """Compute the (weighted) graph of Neighbors for points in X.

    Neighborhoods are restricted the points at a distance lower than
    radius.

    Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample data.

    radius : float
        Radius of neighborhoods.

    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    metric : str, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    include_self : bool or 'auto', default=False
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If 'auto', then True is used for mode='connectivity' and False
        for mode='distance'.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    A : sparse matrix of shape (n_samples, n_samples)
        Graph where A[i, j] is assigned the weight of edge that connects
        i to j. The matrix is of CSR format.

    See Also
    --------
    # 如果输入的数据 X 不是 RadiusNeighborsMixin 类型的实例
    if not isinstance(X, RadiusNeighborsMixin):
        # 使用 NearestNeighbors 创建一个实例 X，然后对其进行训练
        X = NearestNeighbors(
            radius=radius,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        ).fit(X)
    else:
        # 否则，检查输入的 X 的参数是否正确
        _check_params(X, metric, p, metric_params)

    # 根据 include_self 和 mode 参数，确定要查询的点
    query = _query_include_self(X._fit_X, include_self, mode)
    # 返回 X 的 radius neighbors graph
    return X.radius_neighbors_graph(query, radius, mode)
class KNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin, KNeighborsMixin, TransformerMixin, NeighborsBase
):
    """
    Transform X into a (weighted) graph of k nearest neighbors.

    The transformed data is a sparse graph as returned by kneighbors_graph.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    n_neighbors : int, default=5
        Number of neighbors for each sample in the transformed sparse graph.
        For compatibility reasons, as each sample is considered as its own
        neighbor, one extra neighbor will be computed when mode == 'distance'.
        In this case, the sparse graph contains (n_neighbors + 1) neighbors.

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

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        This parameter is expected to be positive.
    """
    # metric_params : dict, default=None
    #     用于度量函数的额外关键字参数。

    # n_jobs : int, default=None
    #     用于邻居搜索的并行作业数量。
    #     如果为 ``-1``，则作业数量将设置为 CPU 核心的数量。

    # Attributes
    # ----------
    # effective_metric_ : str or callable
    #     使用的距离度量标准。它将与 `metric` 参数相同，
    #     或者是它的同义词，例如，如果 `metric` 参数设置为 'minkowski'，
    #     并且 `p` 参数设置为 2，则为 'euclidean'。

    # effective_metric_params_ : dict
    #     用于度量函数的额外关键字参数。对于大多数度量标准，
    #     将与 `metric_params` 参数相同，但如果 `effective_metric_` 属性设置为
    #     'minkowski'，可能还包含 `p` 参数的值。

    # n_features_in_ : int
    #     在 :term:`fit` 过程中看到的特征数量。
    #     .. versionadded:: 0.24

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     在 :term:`fit` 过程中看到的特征名称。仅当 `X` 的特征名称都是字符串时定义。
    #     .. versionadded:: 1.0

    # n_samples_fit_ : int
    #     拟合数据中的样本数量。

    # See Also
    # --------
    # kneighbors_graph : 计算 X 中点的 k-邻居加权图。
    # RadiusNeighborsTransformer : 将 X 转换为半径内邻居的加权图。

    # Notes
    # -----
    # 关于如何将 :class:`~sklearn.neighbors.KNeighborsTransformer`
    # 与 :class:`~sklearn.manifold.TSNE` 结合使用的示例，请参见
    # :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`。

    # Examples
    # --------
    # >>> from sklearn.datasets import load_wine
    # >>> from sklearn.neighbors import KNeighborsTransformer
    # >>> X, _ = load_wine(return_X_y=True)
    # >>> X.shape
    # (178, 13)
    # >>> transformer = KNeighborsTransformer(n_neighbors=5, mode='distance')
    # >>> X_dist_graph = transformer.fit_transform(X)
    # >>> X_dist_graph.shape
    # (178, 178)
    ```

    # _parameter_constraints: dict = {
    #     **NeighborsBase._parameter_constraints,
    #     "mode": [StrOptions({"distance", "connectivity"})],
    # }
    # _parameter_constraints.pop("radius")

    # def __init__(
    #     self,
    #     *,
    #     mode="distance",
    #     n_neighbors=5,
    #     algorithm="auto",
    #     leaf_size=30,
    #     metric="minkowski",
    #     p=2,
    #     metric_params=None,
    #     n_jobs=None,
    # ):
    #     super(KNeighborsTransformer, self).__init__(
    #         n_neighbors=n_neighbors,
    #         radius=None,
    #         algorithm=algorithm,
    #         leaf_size=leaf_size,
    #         metric=metric,
    #         p=p,
    #         metric_params=metric_params,
    #         n_jobs=n_jobs,
    #     )
    #     self.mode = mode

    # @_fit_context(
    #     # KNeighborsTransformer.metric is not validated yet
    #     prefer_skip_nested_validation=False
    # )
    def fit(self, X, y=None):
        """
        Fit the k-nearest neighbors transformer from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : KNeighborsTransformer
            The fitted k-nearest neighbors transformer.
        """
        # 调用内部方法 _fit 进行实际的拟合操作
        self._fit(X)
        # 设置输出特征数为已拟合数据的样本数
        self._n_features_out = self.n_samples_fit_
        # 返回自身对象
        return self

    def transform(self, X):
        """
        Compute the (weighted) graph of Neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_samples_transform, n_features)
            Sample data.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples_transform, n_samples_fit)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        # 检查是否已经拟合，确保可以进行转换操作
        check_is_fitted(self)
        # 根据指定的 mode 和邻居数构建邻居图
        add_one = self.mode == "distance"
        return self.kneighbors_graph(
            X, mode=self.mode, n_neighbors=self.n_neighbors + add_one
        )

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        # 先拟合数据，再转换数据
        return self.fit(X).transform(X)

    def _more_tags(self):
        """
        Provide additional tags for internal testing.

        Returns
        -------
        dict
            Dictionary with additional tags.
        """
        return {
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "check is not applicable."
            }
        }
# 定义一个名为 RadiusNeighborsTransformer 的类，继承自多个Mixin类和基类 NeighborsBase。
class RadiusNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin,  # 混入类，提供类名前缀特征输出的功能
    RadiusNeighborsMixin,  # 混入类，提供基于半径的邻居查询功能
    TransformerMixin,  # 混入类，提供转换器的基本功能
    NeighborsBase,  # 基类，提供邻居基础功能
):
    """Transform X into a (weighted) graph of neighbors nearer than a radius.

    The transformed data is a sparse graph as returned by
    `radius_neighbors_graph`.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    radius : float, default=1.0
        Radius of neighborhood in the transformed sparse graph.

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

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        This parameter is expected to be positive.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    """
    # _parameter_constraints 是一个字典，包含参数的约束条件
    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "mode": [StrOptions({"distance", "connectivity"})],
    }
    # 从约束条件中移除 "n_neighbors" 键
    _parameter_constraints.pop("n_neighbors")

    # RadiusNeighborsTransformer 类的初始化方法
    def __init__(
        self,
        *,
        mode="distance",  # 模式参数，默认为 "distance"
        radius=1.0,  # 半径参数，默认为 1.0
        algorithm="auto",  # 算法参数，默认为 "auto"
        leaf_size=30,  # 叶子大小参数，默认为 30
        metric="minkowski",  # 距离度量参数，默认为 "minkowski"
        p=2,  # 距离度量参数的额外参数，默认为 2
        metric_params=None,  # 距离度量参数的附加关键字参数，默认为 None
        n_jobs=None,  # 并行作业数，默认为 None
    ):
        # 调用父类 NeighborsBase 的初始化方法，将 n_neighbors 参数设为 None
        super(RadiusNeighborsTransformer, self).__init__(
            n_neighbors=None,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        # 设置对象的 mode 属性为传入的 mode 参数值
        self.mode = mode

    # _fit_context 装饰器，用于装饰下一个函数或方法
    @_fit_context(
        # RadiusNeighborsTransformer.metric 尚未进行验证
        prefer_skip_nested_validation=False
    )
    # 从训练数据集中拟合半径邻居转换器

    def fit(self, X, y=None):
        """Fit the radius neighbors transformer from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : RadiusNeighborsTransformer
            The fitted radius neighbors transformer.
        """
        # 调用内部方法 _fit 进行拟合
        self._fit(X)
        # 设置输出特征数为拟合样本数
        self._n_features_out = self.n_samples_fit_
        # 返回拟合后的自身实例
        return self

    # 对 X 进行转换，计算其邻居的(加权)图形

    def transform(self, X):
        """Compute the (weighted) graph of Neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_samples_transform, n_features)
            Sample data.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples_transform, n_samples_fit)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        # 检查是否已经拟合过，确保可以进行转换
        check_is_fitted(self)
        # 调用 radius_neighbors_graph 方法生成邻居图，并根据指定的参数进行排序
        return self.radius_neighbors_graph(X, mode=self.mode, sort_results=True)

    # 拟合数据并进行转换

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        # 调用 fit 方法进行拟合，然后调用 transform 方法进行转换
        return self.fit(X).transform(X)

    # 提供额外的标签信息，目前用于指定某些检查不适用的情况

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "check is not applicable."
            }
        }
```