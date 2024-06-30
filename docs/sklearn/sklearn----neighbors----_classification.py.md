# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_classification.py`

```
"""Nearest Neighbor Classification"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块，用于处理警告信息
import warnings
# 导入检查是否为整数的函数
from numbers import Integral

# 导入 NumPy 库，并使用别名 np
import numpy as np

# 从 scikit-learn 库中导入必要的模块和函数
# _check_precomputed 函数用于检查是否使用预先计算的距离矩阵
from sklearn.neighbors._base import _check_precomputed

# 从 scikit-learn 库中导入基类和相关功能
from ..base import ClassifierMixin, _fit_context
# 导入用于降低成对距离计算的模块
from ..metrics._pairwise_distances_reduction import (
    ArgKminClassMode,
    RadiusNeighborsClassMode,
)
# 导入参数验证工具
from ..utils._param_validation import StrOptions
# 导入数组操作函数
from ..utils.arrayfuncs import _all_with_any_reduction_axis_1
# 导入数学扩展函数
from ..utils.extmath import weighted_mode
# 导入修复功能
from ..utils.fixes import _mode
# 导入验证函数
from ..utils.validation import _is_arraylike, _num_samples, check_is_fitted
# 从基类模块中导入相关函数和类
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin, _get_weights


def _adjusted_metric(metric, metric_kwargs, p=None):
    """调整距离度量函数的参数。

    Parameters
    ----------
    metric : str
        距离度量的名称。
    metric_kwargs : dict
        距离度量的关键字参数。
    p : int or None, default=None
        Minkowski 距离的参数。

    Returns
    -------
    str
        调整后的距离度量名称。
    dict
        调整后的距离度量关键字参数。
    """
    metric_kwargs = metric_kwargs or {}
    # 如果距离度量是 "minkowski"，则根据 p 值调整参数
    if metric == "minkowski":
        metric_kwargs["p"] = p
        # 如果 p 等于 2，则将距离度量修改为 "euclidean"
        if p == 2:
            metric = "euclidean"
    return metric, metric_kwargs


class KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase):
    """k-最近邻分类器。

    实现了基于 k 最近邻投票的分类器。

    参数
    ----------
    n_neighbors : int, default=5
        默认用于 :meth:`kneighbors` 查询的邻居数量。

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        用于预测的权重函数。可能的取值包括：

        - 'uniform' : 均匀权重。每个邻域内的点权重相等。
        - 'distance' : 根据距离的倒数加权。在这种情况下，距离更近的邻居
          对查询点的影响更大。
        - [callable] : 用户定义的函数，接受一个距离数组，并返回一个相同形状的数组，
          包含权重。

        参考示例：
        :ref:`sphx_glr_auto_examples_neighbors_plot_classification.py`
        中展示了 `weights` 参数对决策边界的影响。

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        用于计算最近邻的算法：

        - 'ball_tree' 使用 :class:`BallTree`
        - 'kd_tree' 使用 :class:`KDTree`
        - 'brute' 使用朴素的暴力搜索。
        - 'auto' 会尝试根据传递给 :meth:`fit` 方法的值决定最适合的算法。

        注意：在稀疏输入上进行拟合会覆盖此参数的设置，使用暴力搜索。

    leaf_size : int, default=30
        传递给 BallTree 或 KDTree 的叶子大小。这会影响构建和查询的速度，
        以及存储树所需的内存。最佳值取决于问题的性质。
    """
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30):
        super().__init__(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size)
        self.weights = weights
    # p : float, default=2
    #     Minkowski 度量的幂参数。当 p = 1 时，等同于曼哈顿距离（l1），当 p = 2 时，等同于欧几里得距离（l2）。
    #     对于任意的 p 值，使用 Minkowski 距离（l_p）来计算。该参数预期为正数。

    # metric : str or callable, default='minkowski'
    #     用于距离计算的度量标准。默认为 "minkowski"，当 p = 2 时，等同于标准的欧几里得距离。
    #     可参考 `scipy.spatial.distance` 的文档以及 :class:`~sklearn.metrics.pairwise.distance_metrics` 中列出的有效度量标准值。

    #     如果 metric 为 "precomputed"，则假设 X 是距离矩阵，在拟合期间必须是方阵。X 可能是稀疏图，此时只有 "非零" 元素可能被视为邻居。

    #     如果 metric 是一个可调用函数，它接受两个表示1D向量的数组作为输入，并且必须返回一个值，指示这些向量之间的距离。
    #     这适用于 Scipy 的度量标准，但比将度量标准名称作为字符串传递效率低。

    # metric_params : dict, default=None
    #     度量函数的额外关键字参数。

    # n_jobs : int, default=None
    #     用于邻居搜索的并行作业数。``None`` 表示使用1个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。
    #     ``-1`` 表示使用所有处理器。详见 :term:`术语表 <n_jobs>`。

    # Attributes
    # ----------
    # classes_ : array of shape (n_classes,)
    #     分类器已知的类标签。

    # effective_metric_ : str or callble
    #     使用的距离度量标准。它将与 `metric` 参数相同，或者是其同义词，例如如果 `metric` 参数设置为 'minkowski' 并且 `p` 参数设置为 2，则为 'euclidean'。

    # effective_metric_params_ : dict
    #     度量函数的额外关键字参数。对于大多数度量标准，与 `metric_params` 参数相同，但如果 `effective_metric_` 属性设置为 'minkowski'，则可能还包含 `p` 参数的值。

    # n_features_in_ : int
    #     在拟合期间看到的特征数量。

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     在拟合期间看到的特征名称。仅在 `X` 具有全部为字符串的特征名称时定义。

    # n_samples_fit_ : int
    #     拟合数据中的样本数量。

    # outputs_2d_ : bool
    #     在拟合期间，当 `y` 的形状为 (n_samples, ) 或 (n_samples, 1) 时为 False，否则为 True。

    # See Also
    # --------
    # RadiusNeighborsClassifier: 基于固定半径内邻居的分类器。
    # 创建一个字典 `_parameter_constraints`，从 `NeighborsBase._parameter_constraints` 中复制所有内容
    # 移除字典中的键 "radius"
    # 更新字典，将键 "weights" 的值更新为包含三种可能取值的列表：{"uniform", "distance"}，可调用对象，以及 None
    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints}
    _parameter_constraints.pop("radius")
    _parameter_constraints.update(
        {"weights": [StrOptions({"uniform", "distance"}), callable, None]}
    )
    
    # 初始化 KNeighborsClassifier 类
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
        # 调用父类的初始化方法，设置各个参数
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        # 设置当前类的 weights 属性
        self.weights = weights
    
    # 定义 fit 方法，并使用 @_fit_context 装饰器
    @_fit_context(
        # 设置装饰器参数 prefer_skip_nested_validation 为 False
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        """从训练数据集中拟合 k 最近邻分类器。
    
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            训练数据。
    
        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            目标值。
    
        Returns
        -------
        self : KNeighborsClassifier
            拟合后的 k 最近邻分类器。
        """
        # 调用类内部的 _fit 方法
        return self._fit(X, y)
    # 预测提供数据的类标签。

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        # 检查模型是否已拟合
        check_is_fitted(self, "_fit_method")
        
        # 如果权重为均匀的且使用 brute 方法，并且可用于 ArgKminClassMode 进行预测
        if self.weights == "uniform":
            if self._fit_method == "brute" and ArgKminClassMode.is_usable_for(
                X, self._fit_X, self.metric
            ):
                # 预测每个样本的概率
                probabilities = self.predict_proba(X)
                if self.outputs_2d_:
                    # 如果输出为二维，返回每个输出的类标签
                    return np.stack(
                        [
                            self.classes_[idx][np.argmax(probas, axis=1)]
                            for idx, probas in enumerate(probabilities)
                        ],
                        axis=1,
                    )
                # 否则，返回每个样本的类标签
                return self.classes_[np.argmax(probabilities, axis=1)]
            
            # 如果不需要距离信息来计算权重，则不计算距离
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            # 否则，计算样本到邻居的距离
            neigh_dist, neigh_ind = self.kneighbors(X)

        # 保存类别和标签信息
        classes_ = self.classes_
        _y = self._y
        
        # 如果输出不是二维，则调整形状以适应
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        # 计算输出的数量和查询样本的数量
        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        
        # 获取权重并检查是否有零权重的情况
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case if you are "
                "using a user-defined function."
            )

        # 创建用于存储预测结果的数组
        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        
        # 对每个输出进行循环，根据权重或无权重计算模式
        for k, classes_k in enumerate(classes_):
            if weights is None:
                # 如果没有权重，使用普通的模式来确定最常见的类别
                mode, _ = _mode(_y[neigh_ind, k], axis=1)
            else:
                # 否则，使用加权模式来确定最常见的类别
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            # 将结果转换为数组并存储在预测结果中
            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        # 如果输出不是二维，则将预测结果扁平化
        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        # 返回预测的类标签
        return y_pred

    # 返回额外的标签，指示模型支持多标签输出
    def _more_tags(self):
        return {"multilabel": True}
# 定义一个基于半径邻居的分类器类，实现在给定半径内邻居的投票机制。
class RadiusNeighborsClassifier(RadiusNeighborsMixin, ClassifierMixin, NeighborsBase):
    """Classifier implementing a vote among neighbors within a given radius.

    Read more in the :ref:`User Guide <classification>`.

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
        This parameter is expected to be positive.
    """
    # 距离计算所使用的度量方式，可以是字符串或可调用对象，默认为'minkowski'
    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        如果metric为"precomputed"，则X被假定为距离矩阵，并且在拟合期间必须是方阵。
        X可以是稀疏图，此时只有"非零"元素可以被视为邻居。

        如果metric是一个可调用函数，则它接受表示1D向量的两个数组作为输入，并且必须返回一个值，表示这两个向量之间的距离。
        对于Scipy的度量函数而言，这种方式有效，但效率低于直接传递度量名称作为字符串。

    # 异常样本（在给定半径内没有邻居的样本）的标签设置
    outlier_label : {manual label, 'most_frequent'}, default=None
        Label for outlier samples (samples with no neighbors in given radius).

        - manual label: str or int label (should be the same type as y)
          or list of manual labels if multi-output is used.
        - 'most_frequent' : assign the most frequent label of y to outliers.
        - None : when any outlier is detected, ValueError will be raised.

        异常样本的标签应从唯一的'y'标签中选择。如果选择了不同的值，则会引发警告，并且所有异常样本的类概率将被设为0。

    # 度量函数的额外关键字参数
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    # 用于邻居搜索的并行作业数
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 分类器已知的类标签
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    # 使用的距离度量方式，与'metric'参数相同或其同义词，例如，如果'metric'参数设置为'minkowski'，而'p'参数设置为2，则为'euclidean'。
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    # 度量函数的额外关键字参数，大多数情况下与'metric_params'参数相同，但如果'effective_metric_'属性设置为'minkowski'，可能还包含'p'参数的值。
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    # 在拟合期间观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在拟合期间观察到的特征名称。仅当`X`具有所有字符串类型的特征名称时定义。
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
    # _parameter_constraints 是一个字典，用于定义参数的约束条件
    _parameter_constraints: dict = {
        # 继承自 NeighborsBase 的参数约束条件
        **NeighborsBase._parameter_constraints,
        # "weights" 参数可以是 {"uniform", "distance"} 中的一个字符串、一个可调用对象或者为 None
        "weights": [StrOptions({"uniform", "distance"}), callable, None],
        # "outlier_label" 参数可以是整数、字符串、"array-like" 或者为 None
        "outlier_label": [Integral, str, "array-like", None],
    }
    # 移除 "n_neighbors" 参数的约束条件
    _parameter_constraints.pop("n_neighbors")

    # 初始化方法，设置了一系列参数以及它们的默认值
    def __init__(
        self,
        radius=1.0,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        outlier_label=None,
        metric_params=None,
        n_jobs=None,
    ):
        # 调用父类的初始化方法，设置 radius、algorithm、leaf_size、metric、p、metric_params 和 n_jobs 等参数
        super().__init__(
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        # 设置当前类的 weights 属性为传入的 weights 参数值
        self.weights = weights
        # 设置当前类的 outlier_label 属性为传入的 outlier_label 参数值
        self.outlier_label = outlier_label

    # 使用 _fit_context 装饰器对下面的函数进行装饰
    @_fit_context(
        # RadiusNeighborsClassifier.metric 目前还没有被验证
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        """Fit the radius neighbors classifier from the training dataset.

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
        self : RadiusNeighborsClassifier
            The fitted radius neighbors classifier.
        """
        # 调用内部方法 _fit 进行具体的拟合过程
        self._fit(X, y)

        # 将类别保存到局部变量 classes_
        classes_ = self.classes_
        # 将 _y 保存到局部变量 _y
        _y = self._y
        # 如果输出不是二维的，则将 _y 转换为二维数组，并重新设置 classes_
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        # 处理异常标签 outlier_label
        if self.outlier_label is None:
            outlier_label_ = None

        elif self.outlier_label == "most_frequent":
            outlier_label_ = []
            # 遍历多输出情况，为每个输出获取最频繁的标签
            for k, classes_k in enumerate(classes_):
                label_count = np.bincount(_y[:, k])
                outlier_label_.append(classes_k[label_count.argmax()])

        else:
            if _is_arraylike(self.outlier_label) and not isinstance(
                self.outlier_label, str
            ):
                # 如果异常标签是数组，并且不是字符串，则检查长度是否一致
                if len(self.outlier_label) != len(classes_):
                    raise ValueError(
                        "The length of outlier_label: {} is "
                        "inconsistent with the output "
                        "length: {}".format(self.outlier_label, len(classes_))
                    )
                outlier_label_ = self.outlier_label
            else:
                # 如果异常标签不是数组，则将其复制为每个类别的长度
                outlier_label_ = [self.outlier_label] * len(classes_)

            # 检查每个类别的异常标签是否为标量
            for classes, label in zip(classes_, outlier_label_):
                if _is_arraylike(label) and not isinstance(label, str):
                    raise TypeError(
                        "The outlier_label of classes {} is "
                        "supposed to be a scalar, got "
                        "{}.".format(classes, label)
                    )
                # 确保异常标签的数据类型与 y 的类别数据类型一致
                if np.append(classes, label).dtype != classes.dtype:
                    raise TypeError(
                        "The dtype of outlier_label {} is "
                        "inconsistent with classes {} in "
                        "y.".format(label, classes)
                    )

        # 将处理后的异常标签保存到对象的 outlier_label_ 属性中
        self.outlier_label_ = outlier_label_

        # 返回拟合后的对象本身
        return self
    # 预测给定数据的类标签。

    probs = self.predict_proba(X)
    # 使用训练好的模型预测输入数据的类别概率。

    classes_ = self.classes_
    # 获取模型已知的类别标签。

    if not self.outputs_2d_:
        probs = [probs]
        classes_ = [self.classes_]
    # 如果模型输出不是二维的，则将概率和类别标签转换为列表形式。

    n_outputs = len(classes_)
    # 确定模型输出的数量。

    n_queries = probs[0].shape[0]
    # 确定要进行预测的查询数据的数量。

    y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
    # 创建一个空的数组，用于存储预测的类标签。

    for k, prob in enumerate(probs):
        # 遍历每个输出，根据概率分配类标签。

        max_prob_index = prob.argmax(axis=1)
        # 找到每个样本中概率最大的索引，作为预测的类别。

        y_pred[:, k] = classes_[k].take(max_prob_index)
        # 将预测的类别赋给预测结果数组。

        outlier_zero_probs = (prob == 0).all(axis=1)
        # 找出所有概率为零的异常值。

        if outlier_zero_probs.any():
            zero_prob_index = np.flatnonzero(outlier_zero_probs)
            # 找出所有概率为零的样本的索引。

            y_pred[zero_prob_index, k] = self.outlier_label_[k]
            # 将异常值的类别标签设为模型指定的异常标签。

    if not self.outputs_2d:
        y_pred = y_pred.ravel()
        # 如果输出不是二维的，将预测结果展平为一维数组。

    return y_pred
    # 返回预测的类标签数组。

def _more_tags(self):
    return {"multilabel": True}
    # 返回一个描述额外特性的字典，表明模型支持多标签输出。
```