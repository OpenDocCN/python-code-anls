# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_lof.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入Real类，用于检查参数是否为实数
from numbers import Real

# 导入NumPy库，用于处理数值计算
import numpy as np

# 导入基础类和函数
from ..base import OutlierMixin, _fit_context
# 导入数组检查工具函数
from ..utils import check_array
# 导入参数验证工具函数，包括Interval和StrOptions
from ..utils._param_validation import Interval, StrOptions
# 导入available_if元估计器，用于条件导入
from ..utils.metaestimators import available_if
# 导入检查已拟合对象的验证函数
from ..utils.validation import check_is_fitted
# 导入基本邻居类
from ._base import KNeighborsMixin, NeighborsBase

# 定义本模块对外公开的类名列表
__all__ = ["LocalOutlierFactor"]

# LocalOutlierFactor类继承自三个基类：KNeighborsMixin, OutlierMixin, NeighborsBase
class LocalOutlierFactor(KNeighborsMixin, OutlierMixin, NeighborsBase):
    """使用局部异常因子（LOF）进行无监督异常检测。

    每个样本的异常分数称为局部异常因子。
    它衡量了给定样本相对于其邻居的局部密度偏差。
    这是一种局部方法，异常分数取决于对象相对于其周围邻域的孤立程度。
    具体来说，局部性由k个最近邻居决定，它们的距离用于估计局部密度。
    通过比较样本的局部密度与其邻居的局部密度，可以识别出具有显著较低密度的样本，这些被视为异常值。

    .. versionadded:: 0.19

    Parameters
    ----------
    n_neighbors : int, default=20
        默认用于查询:meth:`kneighbors`的邻居数。
        如果n_neighbors大于提供的样本数，则使用所有样本。

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        用于计算最近邻居的算法：

        - 'ball_tree' 使用:class:`BallTree`
        - 'kd_tree' 使用:class:`KDTree`
        - 'brute' 使用暴力搜索。
        - 'auto' 将尝试根据传递给:meth:`fit`方法的值自动决定最合适的算法。

        注意：在稀疏输入上拟合将覆盖此参数的设置，使用暴力搜索。

    leaf_size : int, default=30
        传递给:class:`BallTree`或:class:`KDTree`的叶子大小。这可以
        影响构建和查询的速度，以及存储树所需的内存。
        最佳值取决于问题的性质。
    metric : str or callable, default='minkowski'
        # 距离计算所使用的度量标准。默认为'minkowski'，当 p = 2 时，表示标准欧氏距离。
        # 可参考 `scipy.spatial.distance` 的文档以及 `sklearn.metrics.pairwise.distance_metrics` 中列出的有效度量标准值。
        # 如果 metric 是 "precomputed"，则假定 X 是一个距离矩阵，在拟合时必须是方阵。
        # X 可能是一个稀疏图，此时只有 "非零" 元素可能被视为邻居。
        # 如果 metric 是一个可调用函数，则它接受两个表示1D向量的数组作为输入，并返回一个值，表示这些向量之间的距离。
        # 使用可调用函数作为度量标准效率较低，不如直接使用字符串形式的度量标准名称。

    p : float, default=2
        # Minkowski 度量中的参数，当 p = 1 时，等效于曼哈顿距离（l1），当 p = 2 时，等效于欧氏距离（l2）。
        # 对于任意的 p，使用的是 minkowski 距离（l_p）。

    metric_params : dict, default=None
        # 度量函数的额外关键字参数。

    contamination : 'auto' or float, default='auto'
        # 数据集中异常值的比例。在拟合时，这个参数定义了样本得分的阈值。
        # - 如果是 'auto'，则阈值的确定方式与原始论文中描述的一致。
        # - 如果是一个浮点数，则 contamination 应在 (0, 0.5] 范围内。
        # 在版本 0.22 中，默认的 contamination 值从 0.1 更改为 'auto'。

    novelty : bool, default=False
        # LocalOutlierFactor 默认仅用于异常检测（novelty=False）。
        # 如果 novelty 设置为 True，则表示要将 LocalOutlierFactor 用于新奇性检测。
        # 在这种情况下，需要注意只能在新的未见数据上使用 predict、decision_function 和 score_samples 方法，
        # 而不能在训练集上使用；同时，结果可能与标准 LOF 结果有所不同。
        # 在版本 0.20 中添加。

    n_jobs : int, default=None
        # 用于邻居搜索的并行作业数。
        # None 表示除非在 joblib.parallel_backend 上下文中，否则为 1。
        # -1 表示使用所有处理器。详情参见“Glossary <n_jobs>”。

    Attributes
    ----------
    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "contamination": [
            StrOptions({"auto"}),  # 可选值为字符串"auto"
            Interval(Real, 0, 0.5, closed="right"),  # 可选值为闭区间[0, 0.5)
        ],
        "novelty": ["boolean"],  # 可选值为布尔类型，用于指示是否进行新奇性检测
    }
    _parameter_constraints.pop("radius")  # 移除参数约束中的 "radius"

    def __init__(
        self,
        n_neighbors=20,  # 默认的近邻数为20
        *,
        algorithm="auto",  # 算法选择，默认为自动选择
        leaf_size=30,  # 叶子大小，默认为30
        metric="minkowski",  # 度量方法，默认为闵可夫斯基距离
        p=2,  # 闵可夫斯基度量的参数p，默认为2
        metric_params=None,  # 度量方法的参数，默认为None
        contamination="auto",  # 污染参数，默认为自动选择
        novelty=False,  # 是否进行新奇性检测，默认为False
        n_jobs=None,  # 并行作业数，默认为None
    ):
        super().__init__(
            n_neighbors=n_neighbors,  # 调用父类的构造函数，设置邻居数参数
            algorithm=algorithm,  # 设置算法类型参数
            leaf_size=leaf_size,  # 设置叶子大小参数
            metric=metric,  # 设置距离度量参数
            p=p,  # 设置距离度量的参数p
            metric_params=metric_params,  # 设置距离度量的其他参数
            n_jobs=n_jobs,  # 设置并行作业数参数
        )
        self.contamination = contamination  # 设置异常值比例参数
        self.novelty = novelty  # 设置新颖性检测标志参数

    def _check_novelty_fit_predict(self):
        if self.novelty:
            msg = (
                "fit_predict is not available when novelty=True. Use "
                "novelty=False if you want to predict on the training set."
            )
            raise AttributeError(msg)
        return True

    @available_if(_check_novelty_fit_predict)
    def fit_predict(self, X, y=None):
        """Fit the model to the training set X and return the labels.

        **Not available for novelty detection (when novelty is set to True).**
        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """

        # As fit_predict would be different from fit.predict, fit_predict is
        # only available for outlier detection (novelty=False)

        return self.fit(X)._predict()

    @_fit_context(
        # LocalOutlierFactor.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    # 定义模型训练方法，用于拟合局部离群因子检测器到训练数据集
    def fit(self, X, y=None):
        """Fit the local outlier factor detector from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : LocalOutlierFactor
            The fitted local outlier factor detector.
        """
        # 调用内部方法 _fit 进行实际的拟合操作
        self._fit(X)

        # 获取拟合后样本数
        n_samples = self.n_samples_fit_
        
        # 检查邻居数是否大于样本数，发出警告并进行调整
        if self.n_neighbors > n_samples:
            warnings.warn(
                "n_neighbors (%s) is greater than the "
                "total number of samples (%s). n_neighbors "
                "will be set to (n_samples - 1) for estimation."
                % (self.n_neighbors, n_samples)
            )
        # 确定最终的邻居数
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        # 计算拟合数据集的邻居距离和邻居索引
        self._distances_fit_X_, _neighbors_indices_fit_X_ = self.kneighbors(
            n_neighbors=self.n_neighbors_
        )

        # 如果 _fit_X 的数据类型是 np.float32，则将距离数据类型转换为一致
        if self._fit_X.dtype == np.float32:
            self._distances_fit_X_ = self._distances_fit_X_.astype(
                self._fit_X.dtype,
                copy=False,
            )

        # 计算局部可达密度（local reachability density）
        self._lrd = self._local_reachability_density(
            self._distances_fit_X_, _neighbors_indices_fit_X_
        )

        # 计算 LOF（局部离群因子）分数，用于定义 offset_
        lrd_ratios_array = (
            self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]
        )
        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        # 如果 contamination 设置为 "auto"，则 offset_ 设为 -1.5
        # 否则，通过负 LOF 分数的百分位来定义 offset_
        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_, 100.0 * self.contamination
            )

        # 验证 negative_outlier_factor_ 的值是否在可接受范围内
        # 必须将 novelty 设置为 False 才能检测离群值
        if np.min(self.negative_outlier_factor_) < -1e7 and not self.novelty:
            warnings.warn(
                "Duplicate values are leading to incorrect results. "
                "Increase the number of neighbors for more accurate results."
            )

        # 返回已拟合的检测器实例
        return self

    # 检查 novelty 是否为 True，如果不是，则抛出 AttributeError 异常
    def _check_novelty_predict(self):
        if not self.novelty:
            msg = (
                "predict is not available when novelty=False, use "
                "fit_predict if you want to predict on training data. Use "
                "novelty=True if you want to use LOF for novelty detection "
                "and predict on new unseen data."
            )
            raise AttributeError(msg)
        return True

    # 在 novelty 为 True 时才可用的装饰器函数
    @available_if(_check_novelty_predict)
    def predict(self, X=None):
        """预测X的标签（1表示内部点，-1表示异常点），根据LOF算法。

        **仅适用于新颖性检测（novelty设置为True时）。**
        此方法允许将预测推广到*新观测*（不在训练集中）。请注意，使用 ``clf.fit(X)`` 然后
        ``clf.predict(X)``，novelty=True 的结果可能与使用 ``clf.fit_predict(X)``，novelty=False 的结果不同。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            要计算相对于训练样本的局部离群因子的查询样本或样本。

        Returns
        -------
        is_inlier : 形状为 (n_samples,) 的 ndarray
            返回-1表示异常/离群值，+1表示内部点。
        """
        return self._predict(X)

    def _predict(self, X=None):
        """根据LOF算法预测X的标签（1表示内部点，-1表示异常点）。

        如果X为None，则返回与fit_predict(X_train)相同的结果。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), 默认为None
            要计算相对于训练样本的局部离群因子的查询样本或样本。如果为None，则在不考虑它们为自身邻居的情况下对训练数据进行预测。

        Returns
        -------
        is_inlier : 形状为 (n_samples,) 的 ndarray
            返回-1表示异常/离群值，+1表示内部点。
        """
        check_is_fitted(self)

        if X is not None:
            shifted_opposite_lof_scores = self.decision_function(X)
            is_inlier = np.ones(shifted_opposite_lof_scores.shape[0], dtype=int)
            is_inlier[shifted_opposite_lof_scores < 0] = -1
        else:
            is_inlier = np.ones(self.n_samples_fit_, dtype=int)
            is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

        return is_inlier

    def _check_novelty_decision_function(self):
        """检查是否启用了新颖性决策函数。

        如果novelty为False，则抛出错误。使用novelty=True，如果要用LOF进行新颖性检测并计算新未见数据的decision_function。
        注意，通过考虑negative_outlier_factor_属性，训练样本的相反LOF始终可用。

        Raises
        ------
        AttributeError
            如果novelty为False，则引发错误。
        """
        if not self.novelty:
            msg = (
                "decision_function is not available when novelty=False. "
                "Use novelty=True if you want to use LOF for novelty "
                "detection and compute decision_function for new unseen "
                "data. Note that the opposite LOF of the training samples "
                "is always available by considering the "
                "negative_outlier_factor_ attribute."
            )
            raise AttributeError(msg)
        return True

    @available_if(_check_novelty_decision_function)
    # 定义一个方法用于计算偏移的反局部离群因子值
    def decision_function(self, X):
        """Shifted opposite of the Local Outlier Factor of X.

        Bigger is better, i.e. large values correspond to inliers.

        **Only available for novelty detection (when novelty is set to True).**
        The shift offset allows a zero threshold for being an outlier.
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        shifted_opposite_lof_scores : ndarray of shape (n_samples,)
            The shifted opposite of the Local Outlier Factor of each input
            samples. The lower, the more abnormal. Negative scores represent
            outliers, positive scores represent inliers.
        """
        # 返回调用 score_samples 方法计算的结果与偏移量 offset_ 的差值
        return self.score_samples(X) - self.offset_

    # 检查 novelty 是否为 True，否则抛出异常
    def _check_novelty_score_samples(self):
        if not self.novelty:
            msg = (
                "score_samples is not available when novelty=False. The "
                "scores of the training samples are always available "
                "through the negative_outlier_factor_ attribute. Use "
                "novelty=True if you want to use LOF for novelty detection "
                "and compute score_samples for new unseen data."
            )
            raise AttributeError(msg)
        # 返回 True 表示检查通过
        return True

    # 在 novelty 为 True 时生效的装饰器
    @available_if(_check_novelty_score_samples)
    def score_samples(self, X):
        """Opposite of the Local Outlier Factor of X.

        It is the opposite as bigger is better, i.e. large values correspond
        to inliers.

        **Only available for novelty detection (when novelty is set to True).**
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point. Because of this, the scores obtained via ``score_samples`` may
        differ from the standard LOF scores.
        The standard LOF scores for the training data is available via the
        ``negative_outlier_factor_`` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        opposite_lof_scores : ndarray of shape (n_samples,)
            The opposite of the Local Outlier Factor of each input samples.
            The lower, the more abnormal.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 将输入数据 X 转换成稀疏矩阵格式（如果适用）
        X = check_array(X, accept_sparse="csr")

        # 计算 X 中每个样本与训练集样本的最近邻距离和最近邻索引
        distances_X, neighbors_indices_X = self.kneighbors(
            X, n_neighbors=self.n_neighbors_
        )

        # 如果 X 的数据类型是 np.float32，则将 distances_X 转换为相同的数据类型
        if X.dtype == np.float32:
            distances_X = distances_X.astype(X.dtype, copy=False)

        # 计算每个样本的局部可达密度（LRD）
        X_lrd = self._local_reachability_density(
            distances_X,
            neighbors_indices_X,
        )

        # 计算局部可达密度比率
        lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

        # 返回结果，因为越大越好，所以取负平均值
        return -np.mean(lrd_ratios_array, axis=1)

    def _local_reachability_density(self, distances_X, neighbors_indices):
        """The local reachability density (LRD)

        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.

        Parameters
        ----------
        distances_X : ndarray of shape (n_queries, self.n_neighbors)
            Distances to the neighbors (in the training samples `self._fit_X`)
            of each query point to compute the LRD.

        neighbors_indices : ndarray of shape (n_queries, self.n_neighbors)
            Neighbors indices (of each query point) among training samples
            self._fit_X.

        Returns
        -------
        local_reachability_density : ndarray of shape (n_queries,)
            The local reachability density of each sample.
        """
        # 获取每个样本到其 k 个最近邻的距离的最大值
        dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]
        # 计算可达距离数组，取每个样本的 distances_X 与 dist_k 的元素最大值
        reach_dist_array = np.maximum(distances_X, dist_k)

        # 返回局部可达密度，避免出现 `nan`，使用 1e-10 作为修正值
        return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)

    def _more_tags(self):
        return {
            "preserves_dtype": [np.float64, np.float32],
        }
```