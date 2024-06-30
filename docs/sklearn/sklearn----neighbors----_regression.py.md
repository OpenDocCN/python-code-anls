# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_regression.py`

```
"""Nearest Neighbor Regression."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
#                           University of Copenhagen

# 引入警告模块
import warnings

# 引入 NumPy 库
import numpy as np

# 从父模块中导入 RegressorMixin 和 _fit_context
from ..base import RegressorMixin, _fit_context

# 从 metrics 模块中导入 DistanceMetric 类
from ..metrics import DistanceMetric

# 从 _param_validation 模块中导入 StrOptions 类
from ..utils._param_validation import StrOptions

# 从当前目录下的 _base 模块中导入 KNeighborsMixin, NeighborsBase 和 _get_weights 函数
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin, _get_weights


class KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase):
    """Regression based on k-nearest neighbors.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Read more in the :ref:`User Guide <regression>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.

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

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    """
    metric : str, DistanceMetric object or callable, default='minkowski'
        # 距离计算所使用的度量标准。默认为'minkowski'，当p=2时结果为标准的欧氏距离。
        # 可参考`scipy.spatial.distance`的文档和`sklearn.metrics.pairwise.distance_metrics`中列出的度量标准值。

        # 如果metric为"precomputed"，则假定X是一个距离矩阵，并且在fit时必须是方阵。
        # X可以是一个稀疏图，此时只考虑“非零”元素作为邻居。

        # 如果metric是一个可调用函数，则它接受两个表示1D向量的数组作为输入，并且必须返回一个值，表示这两个向量之间的距离。
        # 这适用于Scipy的度量标准，但比将度量标准名称作为字符串传递效率低。

        # 如果metric是一个DistanceMetric对象，则它将直接传递给底层的计算例程。
        
    metric_params : dict, default=None
        # 度量函数的附加关键字参数。

    n_jobs : int, default=None
        # 用于邻居搜索的并行作业数量。
        # ``None``表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为1。
        # ``-1``表示使用所有处理器。详见 :term:`Glossary <n_jobs>` 了解更多细节。
        # 不影响 :meth:`fit` 方法。

    Attributes
    ----------
    effective_metric_ : str or callable
        # 要使用的距离度量标准。它将与`metric`参数相同或者是其同义词，例如当`metric`参数设置为'minkowski'且`p`参数设置为2时，它可能是'euclidean'。

    effective_metric_params_ : dict
        # 度量函数的附加关键字参数。对于大多数度量标准，将与`metric_params`参数相同，但如果`effective_metric_`属性设置为'minkowski'，可能还包含`p`参数的值。

    n_features_in_ : int
        # 在 :term:`fit` 过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 :term:`fit` 过程中看到的特征名称。仅在`X`具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    n_samples_fit_ : int
        # 拟合数据中的样本数量。

    See Also
    --------
    NearestNeighbors : 用于实现邻居搜索的无监督学习器。
    RadiusNeighborsRegressor : 基于固定半径内的邻居进行回归。
    KNeighborsClassifier : 实现k最近邻投票的分类器。
    RadiusNeighborsClassifier : 在给定半径内实现邻居之间的投票的分类器。

    Notes
    -----
    # 参见在线文档中的 :ref:`最近邻 <neighbors>`
    """
    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "weights": [StrOptions({"uniform", "distance"}), callable, None],
    }
    _parameter_constraints["metric"].append(DistanceMetric)
    _parameter_constraints.pop("radius")
    """

    """
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights
    """

    """
    def _more_tags(self):
        # For cross-validation routines to split data correctly
        return {"pairwise": self.metric == "precomputed"}
    """

    """
    @_fit_context(
        # KNeighborsRegressor.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        """Fit the k-nearest neighbors regressor from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsRegressor
            The fitted k-nearest neighbors regressor.
        """
        return self._fit(X, y)
    """
    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        if self.weights == "uniform":
            # 如果权重为均匀分布，则不需要距离来进行加权，因此不计算距离。
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            # 否则，计算邻居的距离和索引。
            neigh_dist, neigh_ind = self.kneighbors(X)

        # 根据距离和权重计算加权值
        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            # 如果没有权重，则预测值为邻居的平均值
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            # 否则，根据加权平均数计算预测值
            y_pred = np.empty((neigh_dist.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
class RadiusNeighborsRegressor(RadiusNeighborsMixin, RegressorMixin, NeighborsBase):
    """Regression based on neighbors within a fixed radius.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Read more in the :ref:`User Guide <regression>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.

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

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    """

    # 继承自 RadiusNeighborsMixin, RegressorMixin, NeighborsBase 的半径邻居回归器
    """Regression based on neighbors within a fixed radius."""

    # 初始化方法，设置默认参数
    def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2):
        """
        Parameters
        ----------
        radius : float, default=1.0
            Range of parameter space to use by default for :meth:`radius_neighbors`
            queries.

        weights : {'uniform', 'distance'}, callable or None, default='uniform'
            Weight function used in prediction.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors.

        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.

        p : float, default=2
            Power parameter for the Minkowski metric.
        """

        # 调用父类的初始化方法
        super().__init__()

        # 将参数赋值给实例变量
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
    metric : str or callable, default='minkowski'
        # 距离计算所使用的度量标准，可以是字符串或可调用对象，默认为'minkowski'，当p = 2时即为标准的欧几里得距离。
        # 查看`scipy.spatial.distance`文档和`sklearn.metrics.pairwise.distance_metrics`中的度量标准列表。

        # 如果metric为"precomputed"，则假定X为距离矩阵，在拟合时必须是方阵。
        # X可以是稀疏图，此时只有"非零"元素可能被视为邻居。

        # 如果metric是可调用函数，则接受两个表示1D向量的数组作为输入，并返回一个值，表示这两个向量之间的距离。
        # 对于Scipy的度量标准而言，这种方法比将度量名称作为字符串传递要低效。

    metric_params : dict, default=None
        # 度量函数的额外关键字参数。

    n_jobs : int, default=None
        # 用于邻居搜索的并行作业数。
        # `None`表示除非在`joblib.parallel_backend`上下文中，否则默认为1。
        # `-1`表示使用所有处理器。详见“Glossary <n_jobs>”。

    Attributes
    ----------
    effective_metric_ : str or callable
        # 要使用的距离度量标准。它将与`metric`参数相同，或者是其同义词，例如，如果`metric`参数设置为'minkowski'，且`p`参数设置为2，则可能是'euclidean'。

    effective_metric_params_ : dict
        # 度量函数的额外关键字参数。对于大多数度量标准，与`metric_params`参数相同，但如果`effective_metric_`属性设置为'minkowski'，还可能包含`p`参数值。

    n_features_in_ : int
        # 在拟合过程中看到的特征数。

        # .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中看到的特征名称。仅当`X`具有全为字符串的特征名称时定义。

        # .. versionadded:: 1.0

    n_samples_fit_ : int
        # 拟合数据中的样本数。

    See Also
    --------
    NearestNeighbors : 用于实现邻居搜索的无监督学习器。
    KNeighborsRegressor : 基于k近邻的回归器。
    KNeighborsClassifier : 基于k近邻的分类器。
    RadiusNeighborsClassifier : 在给定半径内基于邻居的分类器。

    Notes
    -----
    # 参见在线文档中的“Nearest Neighbors <neighbors>”，讨论“algorithm”和“leaf_size”的选择。

    # https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    # 导入 RadiusNeighborsRegressor 类从 sklearn.neighbors 模块
    from sklearn.neighbors import RadiusNeighborsRegressor
    # 创建 RadiusNeighborsRegressor 实例，并设置半径为 1.0
    neigh = RadiusNeighborsRegressor(radius=1.0)
    # 使用训练数据 X 和目标数据 y 来拟合 RadiusNeighborsRegressor 模型
    neigh.fit(X, y)
    # 打印预测结果，预测输入为 [[1.5]] 时的输出
    print(neigh.predict([[1.5]]))
    [0.5]


    # _parameter_constraints 是一个字典，继承自 NeighborsBase._parameter_constraints
    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        # "weights" 参数可以是 {"uniform", "distance"} 的字符串、可调用对象或 None
        "weights": [StrOptions({"uniform", "distance"}), callable, None],
    }
    # 删除 _parameter_constraints 中的 "n_neighbors" 键值对
    _parameter_constraints.pop("n_neighbors")


    # 初始化 RadiusNeighborsRegressor 类的构造函数
    def __init__(
        self,
        radius=1.0,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        # 调用父类的构造函数，设置各种参数
        super().__init__(
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        # 设置权重参数
        self.weights = weights


    # 使用 _fit_context 装饰器修饰 fit 方法
    @_fit_context(
        # 暂时不跳过嵌套验证
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        """Fit the radius neighbors regressor from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : RadiusNeighborsRegressor
            The fitted radius neighbors regressor.
        """
        # 调用内部的 _fit 方法来进行拟合过程
        return self._fit(X, y)
    def predict(self, X):
        """Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), \
                dtype=double
            Target values.
        """
        # 调用 radius_neighbors 方法获取每个查询样本的邻居距离和索引
        neigh_dist, neigh_ind = self.radius_neighbors(X)

        # 根据邻居距离和权重计算样本的权重
        weights = _get_weights(neigh_dist, self.weights)

        # 将 self._y 备份到 _y 变量，如果是一维数组则转换为二维数组
        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        # 创建一个与 _y[0] 形状相同的数组，填充为 NaN
        empty_obs = np.full_like(_y[0], np.nan)

        # 如果权重为空，则计算平均值作为预测值
        if weights is None:
            y_pred = np.array(
                [
                    np.mean(_y[ind, :], axis=0) if len(ind) else empty_obs
                    for (i, ind) in enumerate(neigh_ind)
                ]
            )
        else:
            # 否则，根据权重计算加权平均值作为预测值
            y_pred = np.array(
                [
                    (
                        np.average(_y[ind, :], axis=0, weights=weights[i])
                        if len(ind)
                        else empty_obs
                    )
                    for (i, ind) in enumerate(neigh_ind)
                ]
            )

        # 如果预测值中存在 NaN 值，发出警告信息
        if np.any(np.isnan(y_pred)):
            empty_warning_msg = (
                "One or more samples have no neighbors "
                "within specified radius; predicting NaN."
            )
            warnings.warn(empty_warning_msg)

        # 如果 self._y 是一维数组，则将 y_pred 展平为一维数组
        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        # 返回预测结果 y_pred
        return y_pred
```