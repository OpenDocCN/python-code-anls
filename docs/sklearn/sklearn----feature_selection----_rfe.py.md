# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_rfe.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Recursive feature elimination for feature ranking"""

import warnings
from numbers import Integral

import numpy as np
from joblib import effective_n_jobs

# 导入基础类和函数
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
# 导入评估指标检查函数
from ..metrics import check_scoring
# 导入交叉验证检查函数
from ..model_selection import check_cv
# 导入评分函数
from ..model_selection._validation import _score
# 导入参数验证函数
from ..utils._param_validation import HasMethods, Interval, RealNotInt
# 导入元数据路由函数
from ..utils.metadata_routing import (
    _raise_for_unsupported_routing,
    _RoutingNotSupportedMixin,
)
# 导入元估计器相关函数
from ..utils.metaestimators import _safe_split, available_if
# 导入并行处理相关函数
from ..utils.parallel import Parallel, delayed
# 导入验证函数
from ..utils.validation import check_is_fitted
# 导入特征选择相关基类和函数
from ._base import SelectorMixin, _get_feature_importances


def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score and n_features per step for a fit across one fold.
    """
    # 分割训练集和测试集
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # 执行递归特征消除的拟合过程
    rfe._fit(
        X_train,
        y_train,
        lambda estimator, features: _score(
            # TODO(SLEP6): pass score_params here
            estimator,
            X_test[:, features],
            y_test,
            scorer,
            score_params=None,
        ),
    )

    # 返回每一步的评分和特征数
    return rfe.step_scores_, rfe.step_n_features_


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted `estimator_` if available, otherwise we check the
    unfitted `estimator`. We raise the original `AttributeError` if `attr` does
    not exist. This function is used together with `available_if`.
    """
    
    def check(self):
        if hasattr(self, "estimator_"):
            # 如果存在已拟合的 estimator_，则调用其方法
            getattr(self.estimator_, attr)
        else:
            # 否则调用未拟合的 estimator 的方法
            getattr(self.estimator, attr)

        return True

    return check


class RFE(_RoutingNotSupportedMixin, SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through
    any specific attribute or callable.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    ```
    estimator : ``Estimator`` instance
        # `estimator` 参数是一个 `Estimator` 实例，应当是一个具有 `fit` 方法的监督学习估计器，
        # 该方法提供有关特征重要性的信息（例如 `coef_`, `feature_importances_`）。

    n_features_to_select : int or float, default=None
        # 要选择的特征数量。如果为 `None`，则选择一半的特征。
        # 如果是整数，则表示要选择的特征的绝对数量。
        # 如果是介于 0 和 1 之间的浮点数，则表示要选择的特征的分数。

        .. versionchanged:: 0.24
           # 在版本 0.24 中新增了支持浮点数作为特征数量的比例。

    step : int or float, default=1
        # 如果大于或等于 1，则 `step` 表示每次迭代要移除的特征数（整数）。
        # 如果在 (0.0, 1.0) 范围内，则 `step` 表示每次迭代要移除的特征数的百分比（向下取整）。

    verbose : int, default=0
        # 控制输出的详细程度。

    importance_getter : str or callable, default='auto'
        # 如果为 'auto'，则使用估计器的 `coef_` 或 `feature_importances_` 属性来获取特征重要性。

        # 也接受一个字符串，指定从估计器中提取特征重要性的属性名称/路径（使用 `attrgetter` 实现）。
        # 例如，在 `sklearn.compose.TransformedTargetRegressor` 中可以使用 `regressor_.coef_`，
        # 或在 `sklearn.pipeline.Pipeline` 中使用 `named_steps.clf.feature_importances_`（假设最后一步命名为 `clf`）。

        # 如果是可调用对象，则会覆盖默认的特征重要性获取器。
        # 可调用对象将传递给已拟合的估计器，应返回每个特征的重要性。

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        # 类标签。仅在 `estimator` 是分类器时才可用。

    estimator_ : ``Estimator`` instance
        # 用于选择特征的拟合估计器。

    n_features_ : int
        # 选择的特征数量。

    n_features_in_ : int
        # 在拟合过程中观察到的特征数量。仅在底层估计器在拟合时公开这样的属性时才定义。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中观察到的特征名称。仅当 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    ranking_ : ndarray of shape (n_features,)
        # 特征排名，其中 `ranking_[i]` 对应于第 i 个特征的排名位置。被选中的（即估计最佳的）特征被分配排名 1。

    support_ : ndarray of shape (n_features,)
        # 选择的特征的掩码。

    See Also
    --------
    RFECV : 内置交叉验证的递归特征消除，用于选择最佳特征数量。
    SelectFromModel : Feature selection based on thresholds of importance
        weights.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, n_features_to_select=5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    """

    # 定义参数约束字典，指定了每个参数的类型和限制条件
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_features_to_select": [
            None,
            Interval(RealNotInt, 0, 1, closed="right"),
            Interval(Integral, 0, None, closed="neither"),
        ],
        "step": [
            Interval(Integral, 0, None, closed="neither"),
            Interval(RealNotInt, 0, 1, closed="neither"),
        ],
        "verbose": ["verbose"],
        "importance_getter": [str, callable],
    }

    # 初始化方法，接受多个参数来配置特征选择器对象
    def __init__(
        self,
        estimator,
        *,
        n_features_to_select=None,
        step=1,
        verbose=0,
        importance_getter="auto",
    ):
        self.estimator = estimator  # 设置内部属性，用于估算特征重要性
        self.n_features_to_select = n_features_to_select  # 设置要选择的特征数
        self.step = step  # 设置每次迭代中要移除的特征数
        self.importance_getter = importance_getter  # 设置重要性评估方法
        self.verbose = verbose  # 设置详细程度的标志

    # 获取估算器的类型信息作为只读属性
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    # 当估算器是分类器时，返回类标签的只读属性
    @property
    def classes_(self):
        """Classes labels available when `estimator` is a classifier.

        Returns
        -------
        ndarray of shape (n_classes,)
        """
        return self.estimator_.classes_

    # 使用装饰器标记的内部方法，用于估算器验证的上下文
    @_fit_context(
        # RFE.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        """Fit the RFE model and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        **fit_params : dict
            Additional parameters passed to the `fit` method of the underlying
            estimator.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 调用辅助函数检查是否支持当前路由，如果不支持会抛出异常
        _raise_for_unsupported_routing(self, "fit", **fit_params)
        # 调用本类的 _fit 方法进行实际的拟合过程
        return self._fit(X, y, **fit_params)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Reduce X to the selected features and predict using the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        # 检查当前对象是否已经拟合（训练），如果未拟合会抛出异常
        check_is_fitted(self)
        # 使用内部估计器对象进行预测，先转换输入数据 X 后再进行预测
        return self.estimator_.predict(self.transform(X))

    @available_if(_estimator_has("score"))
    def score(self, X, y, **fit_params):
        """Reduce X to the selected features and return the score of the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        **fit_params : dict
            Parameters to pass to the `score` method of the underlying
            estimator.

            .. versionadded:: 1.0

        Returns
        -------
        score : float
            Score of the underlying base estimator computed with the selected
            features returned by `rfe.transform(X)` and `y`.
        """
        # 检查当前对象是否已经拟合（训练），如果未拟合会抛出异常
        check_is_fitted(self)
        # 使用内部估计器对象计算分数，先转换输入数据 X 后再计算分数
        return self.estimator_.score(self.transform(X), y, **fit_params)

    def _get_support_mask(self):
        # 检查当前对象是否已经拟合（训练），如果未拟合会抛出异常
        check_is_fitted(self)
        # 返回支持的特征掩码，即选择的特征在原特征集中的布尔掩码
        return self.support_

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        # 检查当前对象是否已经拟合（训练），如果未拟合会抛出异常
        check_is_fitted(self)
        # 计算输入样本 X 的决策函数值，先转换输入数据 X 后再计算决策函数值
        return self.estimator_.decision_function(self.transform(X))
    # 如果模型支持 `predict_proba` 方法，则装饰该方法，使其可用
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # 确保模型已经拟合（训练）
        check_is_fitted(self)
        # 调用模型的 predict_proba 方法进行预测
        return self.estimator_.predict_proba(self.transform(X))

    # 如果模型支持 `predict_log_proba` 方法，则装饰该方法，使其可用
    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # 确保模型已经拟合（训练）
        check_is_fitted(self)
        # 调用模型的 predict_log_proba 方法进行预测
        return self.estimator_.predict_log_proba(self.transform(X))

    def _more_tags(self):
        # 定义附加的评估器标签，如 poor_score, requires_y, allow_nan
        tags = {
            "poor_score": True,
            "requires_y": True,
            "allow_nan": True,
        }

        # 如果评估器有 _get_tags 方法，根据其定义调整 allow_nan 属性
        if hasattr(self.estimator, "_get_tags"):
            tags["allow_nan"] = self.estimator._get_tags()["allow_nan"]

        return tags
class RFECV(RFE):
    """Recursive feature elimination with cross-validation to select features.

    The number of features selected is tuned automatically by fitting an :class:`RFE`
    selector on the different cross-validation splits (provided by the `cv` parameter).
    The performance of the :class:`RFE` selector are evaluated using `scorer` for
    different number of selected features and aggregated together. Finally, the scores
    are averaged across folds and the number of features selected is set to the number
    of features that maximize the cross-validation score.
    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.

    min_features_to_select : int, default=1
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.

        .. versionadded:: 0.20

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    scoring : str, callable or None, default=None
        A string (see :ref:`scoring_parameter`) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.
    """
    n_jobs : int or None, default=None
        # 控制并行运行的作业数。默认为None，除非在joblib.parallel_backend上下文中，否则默认为1。
        # -1表示使用所有处理器。详细信息请参见“术语表”中的“n_jobs”。

        .. versionadded:: 0.18

    importance_getter : str or callable, default='auto'
        # 控制获取特征重要性的方式。默认为'auto'，使用estimator的'coef_'或'feature_importances_'属性。

        # 也接受指定特征重要性提取属性名称/路径的字符串。
        # 例如，在:class:`~sklearn.compose.TransformedTargetRegressor`中使用'regressor_.coef_'，
        # 或在:class:`~sklearn.pipeline.Pipeline`的最后一步命名为'clf'时使用'named_steps.clf.feature_importances_'。

        # 如果是callable，则覆盖默认的特征重要性获取器。
        # 可调用对象接受已拟合的estimator，并应返回每个特征的重要性。

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        # 类标签。仅当estimator是分类器时可用。

    estimator_ : ``Estimator`` instance
        # 用于选择特征的已拟合estimator实例。

    cv_results_ : dict of ndarrays
        # 所有数组（字典的值）按特征使用数量升序排序。
        # 即，数组的第一个元素表示使用最少特征的模型，而最后一个元素表示使用所有可用特征的模型。
        # 此字典包含以下键：

        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            # 第(k)折的交叉验证分数。

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            # 在所有折叠中得分的平均值。

        std_test_score : ndarray of shape (n_subsets_of_features,)
            # 在所有折叠中得分的标准差。

        n_features : ndarray of shape (n_subsets_of_features,)
            # 每个步骤使用的特征数。

        .. versionadded:: 1.0

    n_features_ : int
        # 使用交叉验证选择的特征数。

    n_features_in_ : int
        # 在“fit”期间看到的特征数。仅当底层estimator在“fit”时公开这样的属性时才定义。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在“fit”期间看到的特征名称。仅当`X`的特征名称全部为字符串时定义。

        .. versionadded:: 1.0
    _parameter_constraints: dict = {
        **RFE._parameter_constraints,
        "min_features_to_select": [Interval(Integral, 0, None, closed="neither")],
        "cv": ["cv_object"],
        "scoring": [None, str, callable],
        "n_jobs": [None, Integral],
    }
    # 创建一个参数约束字典，继承自 RFE 类的参数约束，添加了新的约束条件
    _parameter_constraints.pop("n_features_to_select")
    # 移除不再需要的参数约束 "n_features_to_select"

    def __init__(
        self,
        estimator,
        *,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
    ):
        # 初始化函数，用于初始化 RFECV 对象
        self.estimator = estimator
        self.step = step
        self.importance_getter = importance_getter
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select
        # 设置对象的各种属性值

    @_fit_context(
        # RFECV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    # 装饰器，指定了一个特定的上下文装饰器，用于内部的拟合操作
```